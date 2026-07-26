"""Direct TensorRT ReID client.

This client keeps the same public contract as TritonReIDClient:
infer(list[BGR crop]) -> float32 embeddings [N, embedding_dim].
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List

import numpy as np

from .reid_preprocessing import preprocess_reid_crops


class TensorRTReIDClient:
    """Run a TensorRT ReID engine directly in-process."""

    def __init__(self, config):
        self.config = config
        self.engine_path = Path(config["tensorrt"]["engine_path"])
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self.mean = np.array(config["preprocessing"]["mean"], dtype=np.float32)
        self.std = np.array(config["preprocessing"]["std"], dtype=np.float32)
        self.color_space = config["preprocessing"].get("color_space", "RGB")
        self.channel_order = config["preprocessing"].get("channel_order", "CHW")
        self.input_shape = config["model"]["input_shape"]  # [H, W]
        self.embedding_dim = int(config["model"]["embedding_dim"])

        inference_config = config.get("inference", {})
        trt_config = config.get("tensorrt", {})
        self.max_batch_size = int(inference_config.get("max_batch_size", trt_config.get("max_batch", 1)))
        self.max_retry = int(inference_config.get("max_retry", 3))
        self.timeout_ms = int(inference_config.get("timeout_ms", 5000))
        self.device_id = int(trt_config.get("device_id", 0))

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("Direct TensorRT ReID requires TensorRT Python bindings.") from exc

        self.trt = trt
        self.cuda_backend = self._load_cuda_backend()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.lock = threading.Lock()

        self.cuda_backend.set_device(self.device_id)
        with self.engine_path.open("rb") as engine_file:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = self.cuda_backend.create_stream()
        self.input_name, self.output_name = self._resolve_io_names()
        self.input_dtype = self._tensor_np_dtype(self.input_name)
        self.output_dtype = self._tensor_np_dtype(self.output_name)
        self._input_allocation = None
        self._input_nbytes = 0
        self._output_allocation = None
        self._output_nbytes = 0

        print(f"✓ Loaded TensorRT ReID engine: {self.engine_path}")
        print(f"✓ TensorRT input='{self.input_name}' output='{self.output_name}' max_batch={self.max_batch_size}")

    def _load_cuda_backend(self):
        try:
            return _CudaPythonBackend()
        except ImportError:
            try:
                return _PyCudaBackend(self.device_id)
            except ImportError as exc:
                raise RuntimeError(
                    "Direct TensorRT ReID requires a CUDA Python binding. "
                    "Install one of: cuda-python==12.5.0 or pycuda."
                ) from exc

    def _resolve_io_names(self) -> tuple[str, str]:
        inputs = []
        outputs = []

        if hasattr(self.engine, "num_io_tensors"):
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    inputs.append(name)
                elif mode == self.trt.TensorIOMode.OUTPUT:
                    outputs.append(name)
        else:
            for idx in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(idx)
                if self.engine.binding_is_input(idx):
                    inputs.append(name)
                else:
                    outputs.append(name)

        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected one input and one output, got inputs={inputs}, outputs={outputs}")
        return inputs[0], outputs[0]

    def _tensor_np_dtype(self, name: str) -> np.dtype:
        if hasattr(self.engine, "get_tensor_dtype"):
            return self.trt.nptype(self.engine.get_tensor_dtype(name))
        return self.trt.nptype(self.engine.get_binding_dtype(name))

    def preprocess(self, crops: List[np.ndarray]) -> np.ndarray:
        return preprocess_reid_crops(
            crops,
            self.input_shape,
            self.mean,
            self.std,
            self.color_space,
            self.channel_order,
        )

    def infer(self, crops: List[np.ndarray], retry=None, max_batch_size=None) -> np.ndarray:
        if len(crops) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        request_batch_size = int(self.max_batch_size if max_batch_size is None else max_batch_size)
        if request_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {request_batch_size}")
        if request_batch_size > self.max_batch_size:
            request_batch_size = self.max_batch_size

        chunks = [
            self._infer_batch(crops[start:start + request_batch_size], retry=retry)
            for start in range(0, len(crops), request_batch_size)
        ]
        return np.vstack(chunks)

    def _infer_batch(self, crops: List[np.ndarray], retry=None) -> np.ndarray:
        retry = self.max_retry if retry is None else int(retry)
        last_error = None

        for attempt in range(retry):
            try:
                start_time = time.time()
                result = self._infer_batch_once(crops)
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.timeout_ms:
                    print(f"WARNING: TensorRT ReID inference exceeded timeout: {elapsed_ms:.1f} ms")
                return result
            except Exception as exc:
                last_error = exc
                if attempt < retry - 1:
                    print(f"WARNING: TensorRT inference failed (attempt {attempt + 1}/{retry}): {exc}")
                    time.sleep(0.1)

        raise RuntimeError(f"TensorRT inference failed after {retry} attempts: {last_error}") from last_error

    def _infer_batch_once(self, crops: List[np.ndarray]) -> np.ndarray:
        batch = self.preprocess(crops).astype(self.input_dtype, copy=False)
        batch_size = len(crops)

        with self.lock:
            self.cuda_backend.set_device(self.device_id)
            try:
                self._set_input_shape(batch.shape)
                output_shape = self._get_output_shape()
                expected_shape = (batch_size, self.embedding_dim)
                if output_shape != expected_shape:
                    raise ValueError(f"Unexpected TensorRT output shape: {output_shape}, expected {expected_shape}")

                output = np.empty(output_shape, dtype=self.output_dtype)
                d_input = self._ensure_input_allocation(batch.nbytes)
                d_output = self._ensure_output_allocation(output.nbytes)

                self._set_tensor_address(self.input_name, self.cuda_backend.ptr(d_input))
                self._set_tensor_address(self.output_name, self.cuda_backend.ptr(d_output))

                self.cuda_backend.memcpy_htod_async(d_input, batch, self.stream)
                self._execute()
                self.cuda_backend.memcpy_dtoh_async(output, d_output, self.stream)
                self.cuda_backend.synchronize(self.stream)
            finally:
                pass

        return output.astype(np.float32, copy=False)

    def _ensure_input_allocation(self, nbytes: int):
        if self._input_allocation is None or nbytes > self._input_nbytes:
            self.cuda_backend.mem_free(self._input_allocation)
            self._input_allocation = self.cuda_backend.mem_alloc(nbytes)
            self._input_nbytes = nbytes
        return self._input_allocation

    def _ensure_output_allocation(self, nbytes: int):
        if self._output_allocation is None or nbytes > self._output_nbytes:
            self.cuda_backend.mem_free(self._output_allocation)
            self._output_allocation = self.cuda_backend.mem_alloc(nbytes)
            self._output_nbytes = nbytes
        return self._output_allocation

    def _set_input_shape(self, shape: tuple[int, ...]) -> None:
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, shape)
            return
        binding_idx = self.engine.get_binding_index(self.input_name)
        self.context.set_binding_shape(binding_idx, shape)

    def _get_output_shape(self) -> tuple[int, ...]:
        if hasattr(self.context, "get_tensor_shape"):
            return tuple(int(dim) for dim in self.context.get_tensor_shape(self.output_name))
        binding_idx = self.engine.get_binding_index(self.output_name)
        return tuple(int(dim) for dim in self.context.get_binding_shape(binding_idx))

    def _set_tensor_address(self, name: str, address: int) -> None:
        if hasattr(self.context, "set_tensor_address"):
            self.context.set_tensor_address(name, address)
            return
        # Older TensorRT Python uses binding arrays during execute_async_v2.
        pass

    def _execute(self) -> None:
        if hasattr(self.context, "execute_async_v3"):
            if not self.context.execute_async_v3(stream_handle=self.cuda_backend.stream_handle(self.stream)):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            return
        raise RuntimeError("This TensorRT Python runtime does not expose execute_async_v3")

    def get_model_metadata(self):
        return {
            "backend": "tensorrt_direct",
            "engine_path": str(self.engine_path),
            "input": self.input_name,
            "output": self.output_name,
            "embedding_dim": self.embedding_dim,
            "max_batch_size": self.max_batch_size,
        }

    def get_model_config(self):
        return self.config

    def close(self):
        with self.lock:
            self.cuda_backend.mem_free(self._input_allocation)
            self.cuda_backend.mem_free(self._output_allocation)
            self._input_allocation = None
            self._output_allocation = None
            if self.stream is not None:
                self.cuda_backend.destroy_stream(self.stream)
            self.stream = None
            self.context = None
            self.engine = None


class _CudaPythonBackend:
    """Small adapter over cuda-python 12.x runtime bindings."""

    def __init__(self):
        from cuda import cudart

        self.cudart = cudart

    def _check(self, result, action: str):
        err = result[0] if isinstance(result, tuple) else result
        if err != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA {action} failed: {err}")
        return result[1:] if isinstance(result, tuple) else ()

    def set_device(self, device_id: int) -> None:
        self._check(self.cudart.cudaSetDevice(device_id), "set_device")

    def create_stream(self):
        return self._check(self.cudart.cudaStreamCreate(), "stream_create")[0]

    def destroy_stream(self, stream) -> None:
        self._check(self.cudart.cudaStreamDestroy(stream), "stream_destroy")

    def stream_handle(self, stream) -> int:
        return int(stream)

    def mem_alloc(self, nbytes: int):
        return self._check(self.cudart.cudaMalloc(nbytes), "malloc")[0]

    def mem_free(self, ptr) -> None:
        if ptr is not None:
            self._check(self.cudart.cudaFree(ptr), "free")

    def ptr(self, allocation) -> int:
        return int(allocation)

    def memcpy_htod_async(self, dst, src: np.ndarray, stream) -> None:
        self._check(
            self.cudart.cudaMemcpyAsync(
                dst,
                src,
                src.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "memcpy_htod_async",
        )

    def memcpy_dtoh_async(self, dst: np.ndarray, src, stream) -> None:
        self._check(
            self.cudart.cudaMemcpyAsync(
                dst,
                src,
                dst.nbytes,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "memcpy_dtoh_async",
        )

    def synchronize(self, stream) -> None:
        self._check(self.cudart.cudaStreamSynchronize(stream), "stream_synchronize")


class _PyCudaBackend:
    """Optional adapter over PyCUDA, kept as a fallback for other hosts."""

    def __init__(self, device_id: int):
        import pycuda.driver as cuda

        self.cuda = cuda
        cuda.init()
        self.context = cuda.Device(device_id).make_context()
        self.context.pop()

    def set_device(self, device_id: int) -> None:
        self.context.push()

    def create_stream(self):
        self.context.push()
        try:
            return self.cuda.Stream()
        finally:
            self.context.pop()

    def destroy_stream(self, stream) -> None:
        pass

    def stream_handle(self, stream) -> int:
        return int(stream.handle)

    def mem_alloc(self, nbytes: int):
        return self.cuda.mem_alloc(nbytes)

    def mem_free(self, ptr) -> None:
        pass

    def ptr(self, allocation) -> int:
        return int(allocation)

    def memcpy_htod_async(self, dst, src: np.ndarray, stream) -> None:
        self.cuda.memcpy_htod_async(dst, src, stream)

    def memcpy_dtoh_async(self, dst: np.ndarray, src, stream) -> None:
        self.cuda.memcpy_dtoh_async(dst, src, stream)

    def synchronize(self, stream) -> None:
        stream.synchronize()
        self.context.pop()
