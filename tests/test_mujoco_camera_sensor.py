"""Tests for CameraSensor on the MuJoCo backend (RgbMujocoSensorContext)."""

from __future__ import annotations

import pytest
import torch
from conftest import (
  get_test_device,
  make_mujoco_camera_scene_and_sim,
)

from mjlab.sensor import CameraSensorCfg, CameraSensorData
from mjlab.sensor.rgb_mujoco_sensor_context import RgbMujocoSensorContext


def test_rgb_shape_and_device():
  """RGB output is a uint8 tensor with shape [N, H, W, 3]."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  scene, sim = make_mujoco_camera_scene_and_sim(sensors=(cam_cfg,))

  sim.sense()
  data = scene["test_cam"].data

  assert isinstance(data, CameraSensorData)
  assert data.rgb is not None
  assert data.rgb.shape == (2, 24, 32, 3)
  assert data.rgb.dtype == torch.uint8
  assert data.rgb.device.type == torch.device(get_test_device()).type
  assert data.depth is None
  assert data.segmentation is None


def test_rgb_is_not_all_zeros():
  """The colored scene should produce some non-zero pixels."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=32,
    height=24,
    data_types=("rgb",),
  )
  scene, sim = make_mujoco_camera_scene_and_sim(sensors=(cam_cfg,))

  sim.sense()
  data = scene["test_cam"].data
  assert data.rgb is not None
  assert data.rgb.any(), "RGB is all zeros — rendering may have failed"


def test_context_type_is_mujoco():
  """Scene wires up RgbMujocoSensorContext when running on the mujoco backend."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=16,
    height=12,
    data_types=("rgb",),
  )
  scene, _ = make_mujoco_camera_scene_and_sim(sensors=(cam_cfg,))
  assert isinstance(scene.sensor_context, RgbMujocoSensorContext)


def test_sense_without_cameras_is_noop():
  """sim.sense() does not crash when no cameras are attached."""
  scene, sim = make_mujoco_camera_scene_and_sim(sensors=())
  assert scene.sensor_context is None
  sim.sense()  # should be a no-op


def test_resolution_larger_than_default_auto_sizes_buffer():
  """Camera resolution above the model's default offscreen buffer is auto-expanded."""
  # mjlab's scene.xml sets offwidth=1920, offheight=1080; pick larger.
  width = 2560
  height = 1440
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=width,
    height=height,
    data_types=("rgb",),
  )
  scene, _ = make_mujoco_camera_scene_and_sim(sensors=(cam_cfg,))
  assert isinstance(scene.sensor_context, RgbMujocoSensorContext)
  mj_model = scene.sensor_context._mj_model
  assert mj_model.vis.global_.offwidth >= width
  assert mj_model.vis.global_.offheight >= height


def test_use_shadows_false_restores_model_after_render():
  """use_shadows=False must not permanently mutate mj_model.light_castshadow."""
  cam_cfg = CameraSensorCfg(
    name="test_cam",
    camera_name="world/overhead_cam",
    width=16,
    height=12,
    data_types=("rgb",),
    use_shadows=False,
  )
  scene, sim = make_mujoco_camera_scene_and_sim(sensors=(cam_cfg,))
  assert isinstance(scene.sensor_context, RgbMujocoSensorContext)
  mj_model = scene.sensor_context._mj_model

  original_castshadow = mj_model.light_castshadow.copy()
  assert original_castshadow.any(), (
    "Test scene must have at least one shadow-casting light"
  )

  sim.sense()

  assert (mj_model.light_castshadow == original_castshadow).all()


def test_mismatched_settings_raise():
  """Cameras with different shared options raise ValueError."""
  xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"/>
        <camera name="c1" pos="0 0 3" quat="1 0 0 0" resolution="16 12"/>
        <camera name="c2" pos="0 0 3" quat="1 0 0 0" resolution="16 12"/>
      </worldbody>
    </mujoco>
  """
  cam1 = CameraSensorCfg(
    name="cam1",
    camera_name="world/c1",
    width=16,
    height=12,
    data_types=("rgb",),
    use_shadows=False,
  )
  cam2 = CameraSensorCfg(
    name="cam2",
    camera_name="world/c2",
    width=16,
    height=12,
    data_types=("rgb",),
    use_shadows=True,
  )
  with pytest.raises(ValueError, match="use_shadows"):
    make_mujoco_camera_scene_and_sim(sensors=(cam1, cam2), xml=xml)
