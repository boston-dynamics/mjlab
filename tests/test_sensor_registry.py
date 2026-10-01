"""Tests for sensor/registry.py."""

import pytest

from mjlab.sensor import registry
from mjlab.sensor.registry import (
  get_sensor_context_backend,
  register_sensor_context_backend,
)


@pytest.fixture(autouse=True)
def clean_registry():
  """Isolate each test from the module-level sensor context registry."""
  saved = dict(registry._SENSOR_CONTEXT_REGISTRY)
  yield
  registry._SENSOR_CONTEXT_REGISTRY.clear()
  registry._SENSOR_CONTEXT_REGISTRY.update(saved)


class BackendOne:
  pass


class BackendTwo:
  pass


def test_register_then_get_returns_class():
  register_sensor_context_backend("one", BackendOne)
  assert get_sensor_context_backend("one") is BackendOne


def test_register_idempotent_same_class():
  register_sensor_context_backend("one", BackendOne)
  register_sensor_context_backend("one", BackendOne)
  assert get_sensor_context_backend("one") is BackendOne


@pytest.mark.parametrize("name", ["mujoco", "mujoco-full"])
def test_register_builtin_name_raises(name):
  with pytest.raises(ValueError):  # Fails due to conflict with bultin name
    register_sensor_context_backend(name, BackendOne)


def test_register_conflicting_class_raises():
  register_sensor_context_backend("one", BackendOne)
  with pytest.raises(ValueError):  # Fails due to duplicate name
    register_sensor_context_backend("one", BackendTwo)


def test_get_unknown_lists_raises_and_lists_registered_names():
  register_sensor_context_backend("one", BackendOne)
  with pytest.raises(ValueError, match="'one'"):
    get_sensor_context_backend("unknown")
