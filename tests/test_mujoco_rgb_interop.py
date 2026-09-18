"""Parity and hard-failure tests for the CUDA-GL interop RGB readback path."""

from __future__ import annotations

import pytest
import torch
from conftest import make_mujoco_camera_scene_and_sim

from mjlab.sensor import CameraSensorCfg, gl_cuda_interop

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _render_rgb(device: str, sensor_context_backend: str) -> torch.Tensor:
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  scene, sim = make_mujoco_camera_scene_and_sim(
    device=device,
    sensors=(cam_cfg,),
    sensor_context_backend=sensor_context_backend,
  )
  sim.sense()
  rgb = scene["test_cam"].data.rgb
  assert rgb is not None
  return rgb.clone()


def test_interop_matches_cpu_host_readback():
  """CUDA interop and CPU host readback must produce bit-identical RGB."""
  try:
    gl_cuda_interop.prepare_gpu_interop(torch.device("cuda", 0))
  except ValueError as e:
    pytest.skip(f"interop unavailable on this machine: {e}")

  interop_rgb = _render_rgb(device="cuda", sensor_context_backend="mujoco")
  host_rgb = _render_rgb(device="cpu", sensor_context_backend="mujoco")

  assert torch.equal(interop_rgb.cpu(), host_rgb)


_MULTI_CAM_XML = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"
            rgba="0.5 0.5 0.5 1"/>
      <geom name="red_box" type="box" size="0.5 0.5 0.5" pos="0 0 0.5"
            rgba="1 0 0 1"/>
      <geom name="blue_box" type="box" size="0.5 0.5 0.5" pos="3 0 0.5"
            rgba="0 0 1 1"/>
      <camera name="overhead_cam" pos="0 0 3" quat="1 0 0 0"
              fovy="45" resolution="32 24"/>
      <camera name="side_cam" pos="3 -3 1" xyaxes="1 0 0 0 0 1"
              fovy="45" resolution="32 24"/>
    </worldbody>
  </mujoco>
"""


def test_multi_camera_interop_matches_cpu_host_readback():
  """Each camera's interop PBO must stay isolated to that camera's own image.

  Every camera owns its own GL context, renderer, and PBO
  (``RgbMujocoSensorContext._init_camera``); this guards against a
  regression that mixes up buffers or contexts across cameras, which
  single-camera tests can't catch.
  """
  try:
    gl_cuda_interop.prepare_gpu_interop(torch.device("cuda", 0))
  except ValueError as e:
    pytest.skip(f"interop unavailable on this machine: {e}")

  overhead_cfg = CameraSensorCfg(
    name="overhead_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  side_cfg = CameraSensorCfg(
    name="side_cam",
    camera_name="world/side_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  sensors = (overhead_cfg, side_cfg)

  interop_scene, interop_sim = make_mujoco_camera_scene_and_sim(
    device="cuda",
    sensors=sensors,
    xml=_MULTI_CAM_XML,
    sensor_context_backend="mujoco",
  )
  interop_sim.sense()
  interop_overhead = interop_scene["overhead_cam"].data.rgb
  interop_side = interop_scene["side_cam"].data.rgb
  assert interop_overhead is not None
  assert interop_side is not None
  interop_overhead = interop_overhead.clone()
  interop_side = interop_side.clone()

  host_scene, host_sim = make_mujoco_camera_scene_and_sim(
    device="cpu",
    sensors=sensors,
    xml=_MULTI_CAM_XML,
    sensor_context_backend="mujoco",
  )
  host_sim.sense()
  host_overhead = host_scene["overhead_cam"].data.rgb
  host_side = host_scene["side_cam"].data.rgb
  assert host_overhead is not None
  assert host_side is not None

  assert torch.equal(interop_overhead.cpu(), host_overhead)
  assert torch.equal(interop_side.cpu(), host_side)
  # Same resolution, different views of different-colored boxes: catches a
  # regression that copies one camera's buffer into another's.
  assert not torch.equal(interop_overhead, interop_side)


def test_raises_when_interop_unavailable(monkeypatch: pytest.MonkeyPatch):
  """A CUDA device without interop support must fail construction, not fall back."""
  monkeypatch.setattr(gl_cuda_interop, "cudart", None)
  with pytest.raises(ValueError, match="cuda-python"):
    _render_rgb(device="cuda", sensor_context_backend="mujoco")
