"""Sensor-based equivalents of functions from commands.py.

These implementations do not rely on kinematic data from mjdata,
making them compatible with both the mujoco and mjwarp physics backends.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import BuiltinSensor
from mjlab.tasks.manipulation.mdp.commands import (
  LiftingCommand,
  LiftingCommandCfg,
  MultiCubeLiftingCommand,
  MultiCubeLiftingCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class SensorLiftingCommand(LiftingCommand):
  """LiftingCommand variant that reads cube position from a BuiltinSensor.

  Overrides _update_metrics to use sensordata instead of
  self.object.data.root_link_pos_w, which is unavailable on the MuJoCo backend.
  self.object is still initialised by the parent __init__ but is unused here.
  """

  def _update_metrics(self) -> None:
    cfg = cast("SensorLiftingCommandCfg", self.cfg)
    cube_sensor: BuiltinSensor = self._env.scene[cfg.cube_sensor_name]
    object_pos_w = cube_sensor.data
    object_height = object_pos_w[:, 2]
    position_error = (self.target_pos - object_pos_w).norm(dim=-1)
    at_goal = (position_error < cfg.success_threshold).float()
    # torch.maximum triggers reportPrivateImportUsage; add+clamp is equivalent
    # for 0/1 tensors and preserves the latch-to-1 semantics.
    self.episode_success = (self.episode_success + at_goal).clamp(max=1.0)
    self.metrics["object_height"] = object_height
    self.metrics["position_error"] = position_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success


@dataclass(kw_only=True)
class SensorLiftingCommandCfg(LiftingCommandCfg):
  """LiftingCommandCfg variant that reads cube position from a BuiltinSensor."""

  cube_sensor_name: str = ""

  def build(self, env: "ManagerBasedRlEnv") -> SensorLiftingCommand:
    """Instantiate a SensorLiftingCommand for this config.

    Returns:
        A SensorLiftingCommand bound to the given environment.
    """
    return SensorLiftingCommand(self, env)


class SensorMultiCubeLiftingCommand(MultiCubeLiftingCommand):
  """MultiCubeLiftingCommand that reads cube positions from BuiltinSensors.

  Overrides _target_object_pos_from_sim to use sensordata instead of
  c.data.root_link_pos_w, which is unavailable on the MuJoCo backend. This is
  also relied on by the parent's _update_command, which refreshes the cached
  position after a timer-expiry resample's forward().
  """

  def _target_object_pos_from_sim(self) -> torch.Tensor:
    cfg = cast("SensorMultiCubeLiftingCommandCfg", self.cfg)
    cube_sensors: list[BuiltinSensor] = [
      self._env.scene[n] for n in cfg.cube_sensor_names
    ]
    all_pos = torch.stack([s.data for s in cube_sensors])  # (num_cubes, B, 3)
    return all_pos[self.target_selection, self._env_arange]

  def _update_metrics(self) -> None:
    self._cached_target_obj_pos = self._target_object_pos_from_sim()
    obj_pos = self._cached_target_obj_pos
    error = (self.target_pos - obj_pos).norm(dim=-1)
    at_goal = (error < self.cfg.success_threshold).float()
    # torch.maximum triggers reportPrivateImportUsage; add+clamp is equivalent
    # for 0/1 tensors and preserves the latch-to-1 semantics.
    self.episode_success = (self.episode_success + at_goal).clamp(max=1.0)
    self.metrics["position_error"] = error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success


@dataclass(kw_only=True)
class SensorMultiCubeLiftingCommandCfg(MultiCubeLiftingCommandCfg):
  """MultiCubeLiftingCommandCfg that reads cube positions from BuiltinSensors."""

  cube_sensor_names: tuple[str, ...] = ()

  def build(self, env: "ManagerBasedRlEnv") -> SensorMultiCubeLiftingCommand:
    """Instantiate a SensorMultiCubeLiftingCommand for this config.

    Returns:
        A SensorMultiCubeLiftingCommand bound to the given environment.
    """
    return SensorMultiCubeLiftingCommand(self, env)


__all__ = [
  "SensorLiftingCommand",
  "SensorLiftingCommandCfg",
  "SensorMultiCubeLiftingCommand",
  "SensorMultiCubeLiftingCommandCfg",
]
