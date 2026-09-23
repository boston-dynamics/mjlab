"""MuJoCo sensor context for the mujoco backend, using CUDA-GL interop."""

from __future__ import annotations

import ctypes
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from OpenGL import GL as gl

from mjlab.sensor import gl_cuda_interop

if TYPE_CHECKING:
  import mujoco_warp as mjwarp

  from mjlab.sensor.camera_sensor import CameraSensor, CameraSensorCfg
  from mjlab.sensor.raycast_sensor import RayCastSensor
  from mjlab.sim.mujoco_sim import MujocoModelBridge, MujocoSimData


@dataclass
class _CamState:
  """Per-camera renderer and RGB buffers.

  ``rgb_cuda_pbo`` is set for every camera on a CUDA device (interop is
  required there); cameras on a CPU device use
  ``rgb_raw_numpy``/``rgb_raw_host`` instead.
  """

  cam_idx: int
  width: int
  height: int
  renderer: mujoco.Renderer

  rgb: torch.Tensor | None = None

  # Host readback path: numpy view written by mjr_readPixels. rgb_raw_host
  # is the torch tensor backing that view's memory; kept alive here so it
  # isn't garbage-collected out from under rgb_raw_numpy.
  rgb_raw_numpy: np.ndarray | None = None
  rgb_raw_host: torch.Tensor | None = None
  # Device-resident staging buffer. None only on a CPU device.
  rgb_raw_device: torch.Tensor | None = None
  # CUDA-registered PBO (only when interop is active for this camera).
  rgb_cuda_pbo: gl_cuda_interop.GLCudaPixelBuffer | None = None


@dataclass
class _SharedOptions:
  """Render options shared by all cameras (validated to match)."""

  use_textures: bool
  use_shadows: bool
  enabled_geom_groups: tuple[int, ...]
  scene_option: mujoco.MjvOption = field(init=False)

  def __post_init__(self) -> None:
    opt = mujoco.MjvOption()
    enabled = set(self.enabled_geom_groups)
    for g in range(len(opt.geomgroup)):
      opt.geomgroup[g] = 1 if g in enabled else 0
    self.scene_option = opt


class RgbMujocoSensorContext:
  """RGB-only sensor context for the mujoco backend.

  Owns one ``mujoco.Renderer`` per camera (always constructed with
  ``offsamples=0``, since raw ``glReadPixels`` is incompatible with MSAA)
  and a scratch ``mujoco.MjData``, whose forward kinematics is
  recomputed for each env right before that env is rendered. On a CUDA
  device, RGB pixels are kept GPU-resident via CUDA-GL interop. On a CPU
  device, pixels come from a synchronous ``mjr_render``/``mjr_readPixels``
  host readback. Depth and segmentation are not supported.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    model: MujocoModelBridge | mjwarp.Model,
    data: MujocoSimData,
    camera_sensors: list[CameraSensor],
    raycast_sensors: Sequence[RayCastSensor],
    device: str,
  ) -> None:
    del model  # unused, but part of the interface
    if len(raycast_sensors) > 0:
      raise NotImplementedError(
        "RgbMujocoSensorContext does not support raycast sensors"
      )
    self._mj_model = mj_model
    self._data = data
    self._nworld = data.nworld
    self._device = torch.device(device)

    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    self._cam_idx_to_state: dict[int, _CamState] = {}
    self._on_gpu = self._device.type == "cuda"

    self._shared_options: _SharedOptions | None = None
    if self.camera_sensors:
      self._shared_options = self._validate_sensor_settings()
      self._update_offscreen_buffer_size()

    self._scratch_data = mujoco.MjData(mj_model)

    for camera_sensor in self.camera_sensors:
      self._init_camera(camera_sensor)

    # Mirror SensorContext: wire each sensor's back-reference.
    for camera_sensor in self.camera_sensors:
      camera_sensor.set_context(self)

  @property
  def has_cameras(self) -> bool:
    return bool(self._cam_idx_to_state)

  def render(self) -> None:
    """Run fwd kinematics + render/readback for every env, then postprocess."""
    if not self._cam_idx_to_state:
      return

    # MujocoSimData.qpos may live on GPU when device="cuda"; copy to host
    # once.
    qpos = self._data.qpos.detach().cpu().numpy()
    mocap_pos = self._data.mocap_pos.detach().cpu().numpy()
    mocap_quat = self._data.mocap_quat.detach().cpu().numpy()

    scene_option = (
      self._shared_options.scene_option if self._shared_options is not None else None
    )
    # Temporarily disable shadows on all lights if requested. Mutating
    # mj_model.light_castshadow is the offscreen-renderer trick for
    # matching the warp path's use_shadows semantics; restore on exit so
    # the shared model isn't permanently altered for the viewer or other
    # consumers.
    disable_shadows = (
      self._shared_options is not None and not self._shared_options.use_shadows
    )
    saved_castshadow: np.ndarray | None = None
    if disable_shadows:
      saved_castshadow = self._mj_model.light_castshadow.copy()
      self._mj_model.light_castshadow[:] = 0

    try:
      for env_i in range(self._nworld):
        self._scratch_data.qpos[:] = qpos[env_i]
        self._scratch_data.mocap_pos[:] = mocap_pos[env_i]
        self._scratch_data.mocap_quat[:] = mocap_quat[env_i]

        # TODO: If all physics backends can be made to provide the necessary kinematic
        # fields, then render() would not need to do kinematics itself.
        mujoco.mj_fwdKinematics(self._mj_model, self._scratch_data)

        for cam in self._cam_idx_to_state.values():
          cam.renderer.update_scene(
            data=self._scratch_data,
            camera=cam.cam_idx,
            scene_option=scene_option,
          )
          self._render_env(cam, env_i)
    finally:
      if saved_castshadow is not None:
        self._mj_model.light_castshadow[:] = saved_castshadow

    for cam in self._cam_idx_to_state.values():
      self._finalize_readback(cam)
    self._transfer_and_flip()

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    cam = self._get_cam(cam_idx)
    assert cam.rgb is not None
    return cam.rgb

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    raise NotImplementedError(
      "RgbMujocoSensorContext does not yet support depth rendering. Use the "
      "'mujoco-full' backend for depth, or mjwarp if applicable."
    )

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    raise NotImplementedError(
      "RgbMujocoSensorContext does not yet support segmentation rendering. Use the "
      "'mujoco-full' backend for segmentation, or mjwarp if applicable."
    )

  # Internal helpers.

  def _get_cam(self, cam_idx: int) -> _CamState:
    cam = self._cam_idx_to_state.get(cam_idx)
    if cam is None:
      available = list(self._cam_idx_to_state.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in RgbMujocoSensorContext. "
        f"Available camera IDs: {available}"
      )
    return cam

  def _render_env(self, cam: _CamState, env_i: int) -> None:
    """Render one env and read its pixels into this camera's raw storage.

    Reaches into ``cam.renderer``'s GL context, MjvScene, and MjrContext on
    purpose: the renderer still owns their lifecycle, this only bypasses
    its Python ``render()`` wrapper, which does its own (slower) readback.
    """
    r = cam.renderer
    r._gl_context.make_current()  # type: ignore[union-attr]
    if not self._on_gpu:
      assert cam.rgb_raw_numpy is not None
      mujoco.mjr_render(r._rect, r._scene, r._mjr_context)  # type: ignore[reportArgumentType]
      mujoco.mjr_readPixels(cam.rgb_raw_numpy[env_i], None, r._rect, r._mjr_context)  # type: ignore[reportArgumentType]
      return

    assert cam.rgb_cuda_pbo is not None
    ctx = r._mjr_context  # type: ignore[union-attr]
    if ctx.offSamples != 0:
      raise RuntimeError(
        "interop readback requires offsamples=0 (no MSAA), got "
        f"offsamples={ctx.offSamples}"
      )
    rect = r._rect  # type: ignore[union-attr]
    mujoco.mjr_render(rect, r._scene, ctx)  # type: ignore[reportArgumentType]
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, ctx.offFBO)
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, cam.rgb_cuda_pbo.gl_id)
    offset = env_i * cam.height * cam.width * 3
    gl.glReadPixels(
      rect.left,
      rect.bottom,
      rect.width,
      rect.height,
      gl.GL_RGB,
      gl.GL_UNSIGNED_BYTE,
      ctypes.c_void_p(offset),
    )
    gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

  def _finalize_readback(self, cam: _CamState) -> None:
    """No-op on the host path; device->device PBO copy on the interop path."""
    if not self._on_gpu:
      return
    assert cam.rgb_cuda_pbo is not None
    assert cam.rgb_raw_device is not None
    cam.rgb_cuda_pbo.copy_into(cam.rgb_raw_device)

  def _transfer_and_flip(self) -> None:
    """Apply the vertical flip, writing the result into ``cam.rgb``.

    - CPU device (``cam.rgb_raw_device is None``): flip via numpy slicing
      straight into ``cam.rgb``.
    - CUDA device: ``rgb_raw_device`` was already populated by the interop
      D2D copy in ``_finalize_readback``; flip via torch straight into
      ``cam.rgb``.
    """
    for cam in self._cam_idx_to_state.values():
      assert cam.rgb is not None
      if cam.rgb_raw_device is None:
        assert cam.rgb_raw_numpy is not None
        np.copyto(cam.rgb.numpy(), cam.rgb_raw_numpy[:, ::-1, :, :])
      else:
        cam.rgb.copy_(cam.rgb_raw_device.flip([1]))

  def _init_camera(self, sensor: CameraSensor) -> None:
    cfg: CameraSensorCfg = sensor.cfg
    cam_idx = sensor.camera_idx
    width, height = cfg.width, cfg.height

    unsupported = frozenset(cfg.data_types) & {"depth", "segmentation"}
    if unsupported:
      raise RuntimeError(
        "RgbMujocoSensorContext only supports 'rgb'. "
        f"Camera '{cfg.name}' requested: {unsupported}"
      )

    # offsamples=0 unconditionally: MSAA's multisample resolve blit is
    # incompatible with a raw glReadPixels into a PBO, and this applies
    # whether or not interop ultimately succeeds for this camera.
    original_offsamples = int(self._mj_model.vis.quality.offsamples)
    self._mj_model.vis.quality.offsamples = 0
    try:
      renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
    finally:
      self._mj_model.vis.quality.offsamples = original_offsamples

    cam = _CamState(cam_idx=cam_idx, width=width, height=height, renderer=renderer)
    self._setup_readback(cam)

    on_device = self._device.type != "cpu"
    cam.rgb = (
      torch.empty(
        (self._nworld, height, width, 3), dtype=torch.uint8, device=self._device
      )
      if on_device
      else torch.empty((self._nworld, height, width, 3), dtype=torch.uint8)
    )

    self._cam_idx_to_state[cam_idx] = cam

  def _setup_readback(self, cam: _CamState) -> None:
    """Set up CUDA-GL interop on a CUDA device, host readback on a CPU device."""
    if not self._on_gpu:
      cam.rgb_raw_host = torch.empty(
        (self._nworld, cam.height, cam.width, 3), dtype=torch.uint8
      )
      cam.rgb_raw_numpy = cam.rgb_raw_host.numpy()
      return

    gl_cuda_interop.prepare_gpu_interop(self._device)

    cam.rgb_raw_device = torch.empty(
      (self._nworld, cam.height, cam.width, 3),
      dtype=torch.uint8,
      device=self._device,
    )
    cam.renderer._gl_context.make_current()  # type: ignore[union-attr]
    cam.rgb_cuda_pbo = gl_cuda_interop.GLCudaPixelBuffer(
      self._nworld * cam.height * cam.width * 3
    )

  def _validate_sensor_settings(self) -> _SharedOptions:
    ref = self.camera_sensors[0].cfg
    for camera_sensor in self.camera_sensors[1:]:
      cfg = camera_sensor.cfg
      if cfg.use_textures != ref.use_textures:
        raise ValueError(
          "All camera sensors must share the same use_textures "
          f"setting. '{camera_sensor.cfg.name}' differs from '{ref.name}'."
        )
      if cfg.use_shadows != ref.use_shadows:
        raise ValueError(
          "All camera sensors must share the same use_shadows "
          f"setting. '{camera_sensor.cfg.name}' differs from '{ref.name}'."
        )
      if cfg.enabled_geom_groups != ref.enabled_geom_groups:
        raise ValueError(
          "All camera sensors must share the same "
          f"enabled_geom_groups. '{camera_sensor.cfg.name}' differs from "
          f"'{ref.name}'."
        )
    return _SharedOptions(
      use_textures=ref.use_textures,
      use_shadows=ref.use_shadows,
      enabled_geom_groups=tuple(ref.enabled_geom_groups),
    )

  def close(self) -> None:
    """Release all renderers and CUDA-interop resources.

    Isolates each camera's teardown so one failure (e.g. a CUDA context
    already torn down elsewhere) doesn't leak every later camera's GL
    context and renderer.
    """
    errors: list[Exception] = []
    for cam in self._cam_idx_to_state.values():
      try:
        if cam.rgb_cuda_pbo is not None:
          cam.renderer._gl_context.make_current()  # type: ignore[union-attr]
          cam.rgb_cuda_pbo.close()
          cam.rgb_cuda_pbo = None
        cam.renderer.close()
      except Exception as exc:  # noqa: BLE001 - collect and keep tearing down
        errors.append(exc)
    if errors:
      warnings.warn(
        f"{len(errors)} camera(s) failed to close cleanly: {errors}",
        stacklevel=2,
      )

  def _update_offscreen_buffer_size(self) -> None:
    max_w = max(cam.cfg.width for cam in self.camera_sensors)
    max_h = max(cam.cfg.height for cam in self.camera_sensors)
    if max_w > self._mj_model.vis.global_.offwidth:
      self._mj_model.vis.global_.offwidth = max_w
    if max_h > self._mj_model.vis.global_.offheight:
      self._mj_model.vis.global_.offheight = max_h
