"""Tests for CameraSensor on the mujoco-full sensor context backend."""

from __future__ import annotations

import torch
from conftest import make_mujoco_camera_scene_and_sim

from mjlab.scene import Scene
from mjlab.sensor import CameraSensorCfg, CameraSensorData
from mjlab.sensor.mujoco_sensor_context import MujocoSensorContext
from mjlab.sensor.registry import SENSOR_CONTEXT_MUJOCO_FULL
from mjlab.sim.mujoco_sim import MujocoSimulation


def _make_scene_and_sim(
  sensors: tuple, num_envs: int = 2
) -> tuple[Scene, MujocoSimulation]:
  return make_mujoco_camera_scene_and_sim(
    sensors=sensors,
    num_envs=num_envs,
    sensor_context_backend=SENSOR_CONTEXT_MUJOCO_FULL,
  )


def _render(data_types: tuple, width: int = 32, height: int = 24) -> CameraSensorData:
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=width,
    height=height,
    data_types=data_types,
  )
  scene, sim = _make_scene_and_sim(sensors=(cam_cfg,))
  sim.sense()
  data = scene["test_cam"].data
  assert isinstance(data, CameraSensorData)
  return data


def test_context_type_is_legacy_mujoco():
  """Scene wires up MujocoSensorContext for the legacy backend name."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=16,
    height=12,
    data_types=("rgb",),
  )
  scene, _ = _make_scene_and_sim(sensors=(cam_cfg,))
  assert isinstance(scene.sensor_context, MujocoSensorContext)


def test_depth_shape():
  """Depth output is a float32 tensor with shape [N, H, W, 1]."""
  data = _render(data_types=("depth",))

  assert data.depth is not None
  assert data.depth.shape == (2, 24, 32, 1)
  assert data.depth.dtype == torch.float32
  assert data.rgb is None
  assert data.segmentation is None
  assert torch.unique(data.depth).numel() > 1, (
    "Depth is constant — the floor and box are at different distances from "
    "the overhead camera, so rendering may have failed"
  )


def test_segmentation_shape():
  """Segmentation output is an int32 tensor with shape [N, H, W, 2]."""
  data = _render(data_types=("segmentation",))

  assert data.segmentation is not None
  assert data.segmentation.shape == (2, 24, 32, 2)
  assert data.segmentation.dtype == torch.int32
  assert data.rgb is None
  assert data.depth is None
  object_ids = data.segmentation[..., 0]
  distinct_ids = torch.unique(object_ids)
  assert distinct_ids.numel() > 1, (
    "Segmentation has a single id — the floor, box, and background should "
    "produce at least two distinct geom ids, so rendering may have failed"
  )


def test_rgb_and_depth_together():
  """A camera can render rgb and depth from the same renderer simultaneously."""
  data = _render(data_types=("rgb", "depth"))

  assert data.rgb is not None
  assert data.rgb.shape == (2, 24, 32, 3)
  assert data.depth is not None
  assert data.depth.shape == (2, 24, 32, 1)
  assert data.segmentation is None
