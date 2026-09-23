"""Tests for GLCudaPixelBuffer and the interop availability helpers."""

from __future__ import annotations

import mujoco
import pytest
import torch
from OpenGL import GL

from mjlab.sensor import gl_cuda_interop


def test_validate_egl_torch_device_match_default_matches_cuda0(
  monkeypatch: pytest.MonkeyPatch,
):
  """Default MUJOCO_EGL_DEVICE_ID=0 matches torch cuda:0 with no CUDA_VISIBLE_DEVICES set."""
  monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
  gl_cuda_interop.validate_egl_torch_device_match(torch.device("cuda", 0))


def test_validate_egl_torch_device_match_mismatch(monkeypatch: pytest.MonkeyPatch):
  """A torch device that doesn't match MUJOCO_EGL_DEVICE_ID raises ValueError."""
  monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
  with pytest.raises(ValueError, match="MUJOCO_EGL_DEVICE_ID=0") as exc_info:
    gl_cuda_interop.validate_egl_torch_device_match(torch.device("cuda", 1))
  assert "index 1" in str(exc_info.value)


def test_validate_egl_torch_device_match_respects_cuda_visible_devices(
  monkeypatch: pytest.MonkeyPatch,
):
  """Comparison must use the physical GPU ordinal, not torch's remapped logical index."""
  # CUDA_VISIBLE_DEVICES=2 makes torch's cuda:0 physically GPU 2."""
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
  monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "2")
  gl_cuda_interop.validate_egl_torch_device_match(torch.device("cuda", 0))

  monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
  with pytest.raises(ValueError, match="physical index 2"):
    gl_cuda_interop.validate_egl_torch_device_match(torch.device("cuda", 0))


def test_validate_gpu_interop_setup_without_cuda_bindings(
  monkeypatch: pytest.MonkeyPatch,
):
  """Without cuda.bindings installed, setup validation raises ValueError."""
  monkeypatch.setattr(gl_cuda_interop, "cudart", None)
  with pytest.raises(ValueError, match="cuda-python"):
    gl_cuda_interop.validate_gpu_interop_setup(torch.device("cuda", 0))


def test_validate_gpu_interop_setup_device_mismatch(monkeypatch: pytest.MonkeyPatch):
  """Setup validation surfaces the same EGL/torch device mismatch as a ValueError."""
  monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "5")
  with pytest.raises(ValueError, match="MUJOCO_EGL_DEVICE_ID=5"):
    gl_cuda_interop.validate_gpu_interop_setup(torch.device("cuda", 0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pbo_round_trip_recovers_written_bytes():
  """Bytes written into the PBO via GL round-trip unchanged through CUDA copy_into."""
  device = torch.device("cuda", 0)
  try:
    gl_cuda_interop.prepare_gpu_interop(device)
  except ValueError as e:
    pytest.skip(str(e))

  ctx = mujoco.GLContext(4, 4)  # type: ignore[attr-defined]
  ctx.make_current()
  try:
    nbytes = 16
    pbo = gl_cuda_interop.GLCudaPixelBuffer(nbytes)
    try:
      pattern = bytes(range(nbytes))
      GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo.gl_id)
      GL.glBufferSubData(GL.GL_PIXEL_PACK_BUFFER, 0, nbytes, pattern)
      GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

      dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
      pbo.copy_into(dst)
      torch.cuda.synchronize()
      assert dst.cpu().numpy().tobytes() == pattern
    finally:
      pbo.close()
  finally:
    ctx.free()
