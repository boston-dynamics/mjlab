"""MuJoCo sensor context for the mujoco backend.

Sequential per-env render loop using ``mujoco.Renderer``. This backend is slower than
the RgbMujocoSensorContext backend, which uses CUDA-GL interop. This backend supports
rgb, depth, and segmentation, whereas the RgbMujocoSensorContext backend only supports
rgb. Select this backend via ``SimulationCfg.sensor_context_backend="mujoco-full"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

if TYPE_CHECKING:
  import mujoco_warp as mjwarp

  from mjlab.sensor.camera_sensor import CameraSensor, CameraSensorCfg
  from mjlab.sensor.raycast_sensor import RayCastSensor
  from mjlab.sim.mujoco_sim import MujocoModelBridge, MujocoSimData


@dataclass
class _CamState:
  """Per-camera renderer and output buffers.

  Two optional ``mujoco.Renderer`` instances are held per camera:
  - ``renderer``: MSAA-enabled (the default); used for rgb and depth.
  - ``seg_renderer``: ``offsamples=0``; used for segmentation to avoid MSAA
    color-blending across geom ID boundaries.

  Each renderer, buffer, and tensor is ``None`` when its corresponding data
  type(s) are not requested for this camera.
  """

  cam_idx: int
  width: int
  height: int
  renderer: mujoco.Renderer | None
  seg_renderer: mujoco.Renderer | None
  data_types: frozenset[str]
  # CPU buffers
  rgb_buf: np.ndarray | None = None
  depth_buf: np.ndarray | None = None
  seg_buf: np.ndarray | None = None
  # Tensors. On CPU, these are zero-copy views into the CPU buffers.
  # On GPU, these are tensors on the GPU.
  rgb_tensor: torch.Tensor | None = None
  depth_tensor: torch.Tensor | None = None
  seg_tensor: torch.Tensor | None = None


@dataclass
class _SharedOptions:
  """Render options shared by all cameras (validated to match)."""

  use_textures: bool
  use_shadows: bool
  enabled_geom_groups: tuple[int, ...]
  scene_option: mujoco.MjvOption = field(init=False)

  def __post_init__(self) -> None:
    opt = mujoco.MjvOption()
    # Geom groups: enable only the ones requested.
    enabled = set(self.enabled_geom_groups)
    for g in range(len(opt.geomgroup)):
      opt.geomgroup[g] = 1 if g in enabled else 0
    self.scene_option = opt


class MujocoSensorContext:
  """RGB/depth/segmentation sensor context for the mujoco backend.

  Owns one ``mujoco.Renderer`` per camera and a scratch ``mujoco.MjData``
  used for sequential per-env ``mj_forward`` + render. Renders into CPU
  numpy buffers, then copies to device tensors after each ``render()``
  call unless running on CPU; getters return tensors on ``device``.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    model: MujocoModelBridge | mjwarp.Model,
    data: MujocoSimData | mjwarp.Data,
    camera_sensors: list[CameraSensor],
    raycast_sensors: Sequence[RayCastSensor],
    device: str,
  ) -> None:
    del model  # unused, but part of the interface
    if len(raycast_sensors) > 0:
      raise NotImplementedError("MujocoSensorContext does not support raycast sensors")
    self._mj_model = mj_model
    self._data = data
    self._nworld = data.nworld
    self._device = torch.device(device)

    self.camera_sensors = sorted(camera_sensors, key=lambda s: s.camera_idx)
    self._cam_idx_to_state: dict[int, _CamState] = {}

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

  # Public API consumed by CameraSensor._compute_data and MujocoSimulation.

  @property
  def has_cameras(self) -> bool:
    return bool(self._cam_idx_to_state)

  def render(self) -> None:
    """Run mj_forward + render once per env, populating all camera buffers in place."""
    if not self._cam_idx_to_state:
      return

    # MujocoSimData.qpos may live on GPU when device="cuda"; copy to host.
    qpos = self._data.qpos.detach().cpu().numpy()
    mocap_pos = self._data.mocap_pos.detach().cpu().numpy()
    mocap_quat = self._data.mocap_quat.detach().cpu().numpy()

    scene_option = (
      self._shared_options.scene_option if self._shared_options is not None else None
    )
    # Temporarily disable shadows on all lights if requested. Mutating
    # mj_model.light_castshadow is the offscreen-renderer trick for matching
    # the warp path's use_shadows semantics; restore on exit so the shared
    # model isn't permanently altered for the viewer or other consumers.
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
        # mj_fwdKinematics computes body/geom/site/camera/light poses from qpos
        # without running collision detection, constraint solving, or dynamics.

        # TODO: If all physics backends can be made to provide the necessary kinematic
        # fields, then render() would not need to do kinematics itself.
        mujoco.mj_fwdKinematics(self._mj_model, self._scratch_data)
        for cam in self._cam_idx_to_state.values():
          scene_update_kwargs = {
            "data": self._scratch_data,
            "camera": cam.cam_idx,
            "scene_option": scene_option,
          }
          if cam.renderer is not None:
            cam.renderer.update_scene(**scene_update_kwargs)
          if cam.seg_renderer is not None:
            cam.seg_renderer.update_scene(**scene_update_kwargs)
          self._render_cam_modes(cam, env_i)
    finally:
      if saved_castshadow is not None:
        self._mj_model.light_castshadow[:] = saved_castshadow

    # Transfer the rendered data to the device
    # When running on CPU, no action is needed here, because
    # cam.*_tensor is a zero-copy view into the CPU buffer.
    if self._device.type != "cpu":
      for cam in self._cam_idx_to_state.values():
        if cam.rgb_tensor is not None:
          assert cam.rgb_buf is not None
          cam.rgb_tensor.copy_(torch.from_numpy(cam.rgb_buf))
        if cam.depth_tensor is not None:
          assert cam.depth_buf is not None
          cam.depth_tensor.copy_(torch.from_numpy(cam.depth_buf).unsqueeze(-1))
        if cam.seg_tensor is not None:
          assert cam.seg_buf is not None
          cam.seg_tensor.copy_(torch.from_numpy(cam.seg_buf))

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    cam = self._get_cam(cam_idx)
    if cam.rgb_tensor is None:
      raise RuntimeError(f"Camera ID {cam_idx} does not have RGB rendering enabled.")
    return cam.rgb_tensor

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    cam = self._get_cam(cam_idx)
    if cam.depth_tensor is None:
      raise RuntimeError(f"Camera ID {cam_idx} does not have depth rendering enabled.")
    return cam.depth_tensor

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    cam = self._get_cam(cam_idx)
    if cam.seg_tensor is None:
      raise RuntimeError(
        f"Camera ID {cam_idx} does not have segmentation rendering enabled."
      )
    return cam.seg_tensor

  # Internal helpers.

  def _get_cam(self, cam_idx: int) -> _CamState:
    cam = self._cam_idx_to_state.get(cam_idx)
    if cam is None:
      available = list(self._cam_idx_to_state.keys())
      raise KeyError(
        f"Camera ID {cam_idx} not found in MujocoSensorContext. "
        f"Available camera IDs: {available}"
      )
    return cam

  def _render_cam_modes(self, cam: _CamState, env_i: int) -> None:
    """Render each requested data type for a single camera into env_i's slice.

    Mode switching is done in-place for rgb and depth, but segmentation uses
    a separate renderer.
    """
    if cam.rgb_buf is not None:
      assert cam.renderer is not None
      cam.renderer.render(out=cam.rgb_buf[env_i])
    if cam.depth_buf is not None:
      assert cam.renderer is not None
      cam.renderer.enable_depth_rendering()
      cam.renderer.render(out=cam.depth_buf[env_i])
      cam.renderer.disable_depth_rendering()
    if cam.seg_buf is not None:
      assert cam.seg_renderer is not None
      # mujoco.Renderer.render() does not honor `out=` for segmentation: it
      # constructs a new (H, W, 2) int32 result internally. Copy into our
      # buffer so downstream torch views stay valid.
      seg = cam.seg_renderer.render()
      cam.seg_buf[env_i] = seg

  def _init_camera(self, sensor: CameraSensor) -> None:
    cfg: CameraSensorCfg = sensor.cfg
    cam_idx = sensor.camera_idx
    width, height = cfg.width, cfg.height
    data_types = frozenset(cfg.data_types)

    renderer: mujoco.Renderer | None = None
    if "rgb" in data_types or "depth" in data_types:
      renderer = mujoco.Renderer(self._mj_model, height=height, width=width)

    seg_renderer: mujoco.Renderer | None = None
    if "segmentation" in data_types:
      # Temporarily zero offsamples so the framebuffer is created without MSAA
      # MSAA (Multi-Sample Anti-Aliasing) averages neighboring geom ID colors
      # at edges, producing wrong IDs for segmentation images.
      original_offsamples = int(self._mj_model.vis.quality.offsamples)
      self._mj_model.vis.quality.offsamples = 0
      seg_renderer = mujoco.Renderer(self._mj_model, height=height, width=width)
      self._mj_model.vis.quality.offsamples = original_offsamples
      seg_renderer.enable_segmentation_rendering()

    state = _CamState(
      cam_idx=cam_idx,
      width=width,
      height=height,
      renderer=renderer,
      seg_renderer=seg_renderer,
      data_types=data_types,
    )

    on_device = self._device.type != "cpu"
    if "rgb" in data_types:
      state.rgb_buf = np.empty((self._nworld, height, width, 3), dtype=np.uint8)
      cpu = torch.from_numpy(state.rgb_buf)
      state.rgb_tensor = (
        torch.empty_like(cpu, device=self._device) if on_device else cpu
      )
    if "depth" in data_types:
      state.depth_buf = np.empty((self._nworld, height, width), dtype=np.float32)
      # Match the warp context output shape [N, H, W, 1] via an unsqueeze (zero-copy).
      cpu = torch.from_numpy(state.depth_buf).unsqueeze(-1)
      state.depth_tensor = (
        torch.empty_like(cpu, device=self._device) if on_device else cpu
      )
    if "segmentation" in data_types:
      state.seg_buf = np.empty((self._nworld, height, width, 2), dtype=np.int32)
      cpu = torch.from_numpy(state.seg_buf)
      state.seg_tensor = (
        torch.empty_like(cpu, device=self._device) if on_device else cpu
      )

    self._cam_idx_to_state[cam_idx] = state

  def _validate_sensor_settings(self) -> _SharedOptions:
    """Mirror SensorContext._validate_sensor_settings for the mujoco backend."""
    ref = self.camera_sensors[0].cfg
    for camera_sensor in self.camera_sensors[1:]:
      cfg = camera_sensor.cfg
      if cfg.use_textures != ref.use_textures:
        raise ValueError(
          "All camera sensors must share the same use_textures "
          f"setting. '{camera_sensor.cfg.name}' differs from "
          f"'{ref.name}'."
        )
      if cfg.use_shadows != ref.use_shadows:
        raise ValueError(
          "All camera sensors must share the same use_shadows "
          f"setting. '{camera_sensor.cfg.name}' differs from "
          f"'{ref.name}'."
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
    """Release all renderers, freeing EGL/GL resources."""
    for cam in self._cam_idx_to_state.values():
      if cam.renderer is not None:
        cam.renderer.close()
        cam.renderer = None
      if cam.seg_renderer is not None:
        cam.seg_renderer.close()
        cam.seg_renderer = None

  def _update_offscreen_buffer_size(self) -> None:
    """Ensure the model's offscreen buffer is large enough for all cameras.

    Automatically grows ``model.vis.global_.offwidth`` and ``offheight`` to the
    maximum camera width and height across all sensors so callers don't have to
    set these manually in the scene spec.
    """
    max_w = max(cam.cfg.width for cam in self.camera_sensors)
    max_h = max(cam.cfg.height for cam in self.camera_sensors)
    if max_w > self._mj_model.vis.global_.offwidth:
      self._mj_model.vis.global_.offwidth = max_w
    if max_h > self._mj_model.vis.global_.offheight:
      self._mj_model.vis.global_.offheight = max_h
