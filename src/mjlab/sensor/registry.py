"""Registry for pluggable sensor context backends for mujoco."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.sensor.sensor_context import SensorContext

if TYPE_CHECKING:
  from mjlab.sensor.interface import RenderSensorContextProtocol


# Names of the built-in sensor context backends. These names cannot be used
# for externally-registered sensor context backends.
SENSOR_CONTEXT_MJWARP = "mjwarp"
BUILTIN_SENSOR_CONTEXTS = (SENSOR_CONTEXT_MJWARP,)


_SENSOR_CONTEXT_REGISTRY: dict[
  str, type[SensorContext | RenderSensorContextProtocol]
] = {
  SENSOR_CONTEXT_MJWARP: SensorContext,
}


def register_sensor_context_backend(
  name: str, sensor_context_cls: type[SensorContext | RenderSensorContextProtocol]
) -> None:
  """Register a sensor context backend class under ``name``.

  Args:
    name: The value that selects this sensor context backend.
    sensor_context_cls: The class implementing :class:`SensorContext` or
      :class:`RenderSensorContextProtocol` for this backend.

  Raises:
    ValueError: If ``name`` is the built-in backend or already registered
      with a different class.
  """
  existing = _SENSOR_CONTEXT_REGISTRY.get(name)
  if existing is not None:
    if existing != sensor_context_cls:
      msg = (
        f"Sensor context backend {name!r} is already registered with a different class."
      )
      raise ValueError(msg)
  else:
    _SENSOR_CONTEXT_REGISTRY[name] = sensor_context_cls


def get_sensor_context_backend(
  name: str,
) -> type[SensorContext | RenderSensorContextProtocol]:
  """Look up a registered sensor context backend class by name.

  Returns:
    The sensor context backend class.

  Raises:
    ValueError: If no sensor context backend is registered under ``name``,
      listing the registered names to aid debugging.
  """
  try:
    return _SENSOR_CONTEXT_REGISTRY[name]
  except KeyError:
    known = sorted(_SENSOR_CONTEXT_REGISTRY)
    msg = (
      f"Unknown sensor context backend {name!r}. Available sensor backends are"
      f"{known}. Did you forget to register the backend?"
    )
    raise ValueError(msg) from None
