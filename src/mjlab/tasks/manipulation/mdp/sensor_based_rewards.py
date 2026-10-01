"""Sensor-based equivalents of functions from rewards.py.

These implementations do not rely on kinematic data from mjdata,
making them compatible with both the mujoco and mjwarp physics backends.
"""

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import BuiltinSensor
from mjlab.tasks.manipulation.mdp.sensor_based_commands import (
  SensorMultiCubeLiftingCommand,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def sensor_staged_position_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  grasp_sensor_name: str,
  cube_sensor_name: str,
  reaching_std: float,
  bringing_std: float,
) -> torch.Tensor:
  """Sensor-based equivalent of staged_position_reward.

  Returns:
      Per-environment reward tensor of shape (N,).
  """
  grasp_sensor: BuiltinSensor = env.scene[grasp_sensor_name]
  cube_sensor: BuiltinSensor = env.scene[cube_sensor_name]
  target_pos_w = cast(torch.Tensor, env.command_manager.get_command(command_name))
  reach_error = (cube_sensor.data - grasp_sensor.data).pow(2).sum(dim=-1)
  lift_error = (target_pos_w - cube_sensor.data).pow(2).sum(dim=-1)
  reaching = (-reach_error / reaching_std**2).exp()
  bringing = (-lift_error / bringing_std**2).exp()
  return reaching * (1.0 + bringing)


def sensor_bring_object_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  cube_sensor_name: str,
  std: float,
) -> torch.Tensor:
  """Sensor-based equivalent of bring_object_reward.

  Returns:
      Per-environment reward tensor of shape (N,).
  """
  cube_sensor: BuiltinSensor = env.scene[cube_sensor_name]
  target_pos_w = cast(torch.Tensor, env.command_manager.get_command(command_name))
  error = (target_pos_w - cube_sensor.data).pow(2).sum(dim=-1)
  return (-error / std**2).exp()


def sensor_multi_cube_staged_position_reward(
  env: "ManagerBasedRlEnv",
  grasp_sensor_name: str,
  command_name: str,
  reaching_std: float,
  bringing_std: float,
) -> torch.Tensor:
  """Sensor-based equivalent of multi_cube_staged_position_reward.

  Returns:
      Per-environment reward tensor of shape (N,).
  """
  grasp: BuiltinSensor = env.scene[grasp_sensor_name]
  command = cast(
    SensorMultiCubeLiftingCommand, env.command_manager.get_term(command_name)
  )
  obj_pos_w = command.target_object_pos()
  reach_error = (grasp.data - obj_pos_w).pow(2).sum(dim=-1)
  position_error = (command.target_pos - obj_pos_w).pow(2).sum(dim=-1)
  return (-reach_error / reaching_std**2).exp() * (
    1.0 + (-position_error / bringing_std**2).exp()
  )


__all__ = [
  "sensor_staged_position_reward",
  "sensor_bring_object_reward",
  "sensor_multi_cube_staged_position_reward",
]
