"""Sensor-based equivalents of functions from observations.py.

These implementations do not rely on kinematic data from mjdata,
making them compatible with both the mujoco and mjwarp physics backends.
"""

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def sensor_ee_to_cube_distance(
  env: "ManagerBasedRlEnv",
  grasp_sensor_name: str,
  cube_sensor_name: str,
  base_quat_sensor_name: str,
) -> torch.Tensor:
  """Distance from end effector to cube in base frame, via sensor data."""
  grasp_sensor: BuiltinSensor = env.scene[grasp_sensor_name]
  cube_sensor: BuiltinSensor = env.scene[cube_sensor_name]
  quat_sensor: BuiltinSensor = env.scene[base_quat_sensor_name]
  disp_w = cube_sensor.data - grasp_sensor.data
  return quat_apply(quat_inv(quat_sensor.data), disp_w)


def sensor_cube_to_goal_distance(
  env: "ManagerBasedRlEnv",
  command_name: str,
  cube_sensor_name: str,
  base_quat_sensor_name: str,
) -> torch.Tensor:
  """Distance from cube to goal in base frame, via sensor data."""
  cube_sensor: BuiltinSensor = env.scene[cube_sensor_name]
  quat_sensor: BuiltinSensor = env.scene[base_quat_sensor_name]
  target_pos_w = cast(torch.Tensor, env.command_manager.get_command(command_name))
  disp_w = target_pos_w - cube_sensor.data
  return quat_apply(quat_inv(quat_sensor.data), disp_w)


def sensor_target_position(
  env: "ManagerBasedRlEnv",
  command_name: str,
  grasp_sensor_name: str,
  base_quat_sensor_name: str,
) -> torch.Tensor:
  """Target position relative to the grasp site in the EE frame, via sensor data."""
  grasp_sensor: BuiltinSensor = env.scene[grasp_sensor_name]
  quat_sensor: BuiltinSensor = env.scene[base_quat_sensor_name]
  target_pos_w = cast(torch.Tensor, env.command_manager.get_command(command_name))
  disp_w = target_pos_w - grasp_sensor.data
  return quat_apply(quat_inv(quat_sensor.data), disp_w)


__all__ = [
  "sensor_ee_to_cube_distance",
  "sensor_cube_to_goal_distance",
  "sensor_target_position",
]
