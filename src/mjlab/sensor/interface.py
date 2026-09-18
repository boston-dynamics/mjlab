"""Common public interface shared by the sensor context backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  import mujoco
  import mujoco_warp as mjwarp
  import torch

  from mjlab.sensor.camera_sensor import CameraSensor
  from mjlab.sensor.raycast_sensor import RayCastSensor
  from mjlab.sim.mujoco_sim import MujocoModelBridge, MujocoSimData


class RenderSensorContextProtocol(Protocol):
  """Public interface common to all non-mjwarp sensor context backends.

  This is the surface `mjlab.sensor.camera_sensor.CameraSensor` and
  `mjlab.sim.mujoco_sim.MujocoSimulation` rely on when rendering camera sensors without
  knowing which backend is active. All non-mjwarp backends must satisfy it.

  The mjwarp SensorContext does not satisfy this protocol, because instead of
  providing a ``render`` function, SensorContext provides a mjwarp sensor context
  that Simulation uses when launching mjwarp.render.
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
    """Construct a sensor context backend.

    Args:
        mj_model: Compiled MuJoCo model shared by all envs.
        model: Simulation model backing ``data``.
        data: Simulation data providing qpos/mocap state to render from.
        camera_sensors: Camera sensors to render for.
        raycast_sensors: Raycast sensors sharing this context, if any.
        device: Device the returned tensors should live on.

    The constructor must call ``camera_sensor.set_context(self)`` for every
    sensor in ``camera_sensors`` to wire up its back-reference.
    """
    ...

  # Properties.

  @property
  def has_cameras(self) -> bool:
    """Whether any camera sensors were configured."""
    ...

  # Methods.

  def render(self) -> None:
    """Render all configured cameras for every env, populating their buffers."""
    ...

  def get_rgb(self, cam_idx: int) -> torch.Tensor:
    """Return the most recently rendered RGB tensor for the given camera."""
    ...

  def get_depth(self, cam_idx: int) -> torch.Tensor:
    """Return the most recently rendered depth tensor for the given camera."""
    ...

  def get_segmentation(self, cam_idx: int) -> torch.Tensor:
    """Return the most recently rendered segmentation tensor for the camera."""
    ...

  def close(self) -> None:
    """Release backend resources (e.g. EGL/GL contexts)."""
    ...


if TYPE_CHECKING:
  # Static conformance check: each concrete backend must satisfy the
  # protocol. These assignments fail type checking if a backend's public
  # surface drifts from the interface above. Never executed at runtime.
  from mjlab.sensor.mujoco_sensor_context import MujocoSensorContext
  from mjlab.sensor.rgb_mujoco_sensor_context import RgbMujocoSensorContext

  def _assert_backends_conform(
    mujoco_ctx: RgbMujocoSensorContext, legacy_ctx: MujocoSensorContext
  ) -> None:
    _mujoco: RenderSensorContextProtocol = mujoco_ctx
    _legacy: RenderSensorContextProtocol = legacy_ctx
