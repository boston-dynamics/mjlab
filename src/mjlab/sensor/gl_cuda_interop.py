"""GPU-resident pixel buffer that OpenGL writes into and CUDA reads back from."""

import os

import torch
from OpenGL import GL

try:
  from cuda.bindings import runtime as cudart
except ImportError:
  cudart = None


def unwrap(status_and_results):
  """Raise RuntimeError if a cudart call returned a non-success status.

  Otherwise, return the remaining cudart return values as a tuple. Call as
  ``unwrap(cudart.someCall(...))`` to check and unpack in one step.
  """
  assert cudart is not None
  status, *results = status_and_results
  if status != cudart.cudaError_t.cudaSuccess:
    raise RuntimeError(f"CUDA runtime error: {status!r}")
  return tuple(results)


def validate_egl_torch_device_match(torch_device: torch.device) -> None:
  """Check that MuJoCo's EGL device and torch's CUDA device are the same GPU.

  CUDA-GL interop only works when MuJoCo's EGL context and torch's CUDA
  context live on the same physical GPU. A mismatch causes registration
  failure or garbage pixels. MuJoCo selects its EGL device via
  ``MUJOCO_EGL_DEVICE_ID`` (default 0). Raises ``ValueError`` on mismatch.
  """
  egl_id = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
  torch_idx = torch_device.index
  if torch_idx is None:
    torch_idx = torch.cuda.current_device()
  physical_idx = _physical_cuda_index(torch_idx)
  if egl_id != physical_idx:
    raise ValueError(
      f"MUJOCO_EGL_DEVICE_ID={egl_id} does not match the physical GPU "
      f"backing torch CUDA device {torch_idx} (physical index "
      f"{physical_idx}); set MUJOCO_EGL_DEVICE_ID={physical_idx} to enable "
      "interop"
    )


def _physical_cuda_index(logical_idx: int) -> int:
  """Map a logical ``torch.cuda`` index to its physical GPU ordinal.

  ``MUJOCO_EGL_DEVICE_ID`` is a physical device ordinal, but
  ``torch_device.index``/``torch.cuda.current_device()`` is a logical index
  that ``CUDA_VISIBLE_DEVICES`` remaps (e.g. ``CUDA_VISIBLE_DEVICES=2``
  makes torch's ``cuda:0`` physically GPU 2). Comparing the two directly is
  wrong whenever ``CUDA_VISIBLE_DEVICES`` restricts or reorders devices.
  """
  visible = os.environ.get("CUDA_VISIBLE_DEVICES")
  if not visible:
    return logical_idx
  ids = [entry.strip() for entry in visible.split(",") if entry.strip()]
  if logical_idx >= len(ids):
    return logical_idx
  try:
    return int(ids[logical_idx])
  except ValueError:
    # UUID-based CUDA_VISIBLE_DEVICES entries can't be mapped to an
    # ordinal; fall back to the logical index.
    return logical_idx


def validate_gpu_interop_setup(torch_device: torch.device) -> None:
  """Check that GL<->CUDA interop is ready to use on ``torch_device``.

  Raises ``ValueError`` if not.
  """
  if cudart is None:
    raise ValueError("cuda.bindings (cuda-python) is not installed")
  validate_egl_torch_device_match(torch_device)


def create_cuda_context(torch_device: torch.device) -> None:
  """Force creation of the torch CUDA context on ``torch_device``."""
  torch.zeros(1, device=torch_device)


def prepare_gpu_interop(torch_device: torch.device) -> None:
  """Ensure that GL<->CUDA interop is ready to use on ``torch_device``"""
  validate_gpu_interop_setup(torch_device)
  create_cuda_context(torch_device)


class GLCudaPixelBuffer:
  """A GL Pixel Buffer Object registered with CUDA for D2D readback.

  OpenGL writes into it via ``glReadPixels`` (bind ``gl_id`` as
  ``GL_PIXEL_PACK_BUFFER``); CUDA reads it back via ``copy_into``. All GL
  calls assume MuJoCo's EGL context is current.
  """

  def __init__(self, nbytes: int) -> None:
    if cudart is None:
      raise RuntimeError("cuda.bindings (cuda-python) is not available")
    self._nbytes = nbytes
    # Safe defaults so close() is safe even if a later GL/CUDA call
    # below raises mid-construction.
    self._resource = None
    self._gl_id = 0
    self._gl_id = int(GL.glGenBuffers(1))
    try:
      GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._gl_id)
      # GL_STREAM_READ: written once per frame by glReadPixels, read once by
      # CUDA.
      GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
      GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
      (self._resource,) = unwrap(
        cudart.cudaGraphicsGLRegisterBuffer(
          self._gl_id,
          cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly,
        )
      )
    except Exception:
      # The constructor never completes, so no object exists for the
      # caller to call close() on; delete the GL buffer here or it leaks.
      GL.glDeleteBuffers(1, [self._gl_id])
      self._gl_id = 0
      raise

  @property
  def gl_id(self) -> int:
    return self._gl_id

  def copy_into(self, dst: torch.Tensor) -> None:
    """Copy the PBO contents device->device into ``dst`` on the current stream.

    ``dst`` must be contiguous and exactly ``nbytes`` large. Mapping the
    resource provides the CUDA-runtime guarantee that prior GL writes
    complete before the copy; using torch's current stream orders the copy
    before any subsequent torch op.
    """
    assert cudart is not None
    expected = dst.numel() * dst.element_size()
    if expected != self._nbytes:
      raise ValueError(
        f"copy_into size mismatch: dst is {expected} bytes, "
        f"buffer is {self._nbytes} bytes"
      )
    if not dst.is_contiguous():
      raise ValueError("copy_into requires a contiguous destination tensor")
    stream = torch.cuda.current_stream().cuda_stream
    unwrap(cudart.cudaGraphicsMapResources(1, self._resource, stream))
    try:
      ptr, _size = unwrap(cudart.cudaGraphicsResourceGetMappedPointer(self._resource))
      unwrap(
        cudart.cudaMemcpyAsync(
          dst.data_ptr(),
          ptr,
          self._nbytes,
          cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
          stream,
        )
      )
    finally:
      unwrap(cudart.cudaGraphicsUnmapResources(1, self._resource, stream))

  def close(self) -> None:
    """Unregister the CUDA resource and delete the GL buffer."""
    if self._resource is not None:
      assert cudart is not None
      unwrap(cudart.cudaGraphicsUnregisterResource(self._resource))
      self._resource = None
    if self._gl_id:
      GL.glDeleteBuffers(1, [self._gl_id])
      self._gl_id = 0
