from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple
import mlx.core as mx
import numpy as np

def _safe_namespace(name: str) -> str:
    """Sanitize a model identifier into a filesystem-friendly directory name."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "default"
    return safe[:128]

def _read_safetensors_header(path: Path) -> Optional[Tuple[dict, dict, int]]:
    """Read a safetensors header without touching tensor payload bytes.

    Returns ``(tensor_entries, metadata, data_start)`` where ``data_start`` is
    the absolute file offset of the tensor data buffer.
    """
    try:
        with open(path, "rb") as f:
            head_bytes = f.read(8)
            if len(head_bytes) < 8:
                return None
            header_size = int.from_bytes(head_bytes, "little")
            # Sanity-bound the header so a corrupted file can't trigger a
            # huge allocation.
            if header_size <= 0 or header_size > 64 * 1024 * 1024:
                return None
            header_bytes = f.read(header_size)
            if len(header_bytes) < header_size:
                return None
        header = json.loads(header_bytes)
        metadata = dict(header.pop("__metadata__", {}) or {})
        return header, metadata, 8 + header_size
    except (OSError, ValueError):
        return None

def _read_safetensors_metadata(path: Path) -> Optional[dict]:
    """Read only the ``__metadata__`` dict from a safetensors file, without
    mmap'ing the tensor payload. Used to populate the disk index on init.

    Returns ``None`` on any read/parse error.
    """
    header = _read_safetensors_header(path)
    if header is None:
        return None
    return header[1]

def _numel(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out

def _safetensors_dtype_info(dtype: str):
    """Return ``(numpy_dtype, mlx_dtype, bitcast_to)`` for supported dtypes."""
    if dtype == "BF16":
        return np.dtype("<u2"), mx.uint16, mx.bfloat16
    mapping = {
        "F16": (np.dtype("<f2"), mx.float16, None),
        "F32": (np.dtype("<f4"), mx.float32, None),
    }
    return mapping.get(dtype)

def _safetensors_tensor_bounds(
    entry: dict,
) -> Optional[Tuple[int, int, Tuple[int, ...]]]:
    try:
        start, end = entry["data_offsets"]
        shape = tuple(int(x) for x in entry["shape"])
        dtype_info = _safetensors_dtype_info(str(entry["dtype"]))
        if dtype_info is None:
            return None
        np_dtype, _, _ = dtype_info
        if int(end) < int(start):
            return None
        if _numel(shape) * np_dtype.itemsize != int(end) - int(start):
            return None
        return int(start), int(end), shape
    except (KeyError, TypeError, ValueError):
        return None

def _mlx_array_from_safetensors_bytes(buf, entry: dict) -> Optional[mx.array]:
    bounds = _safetensors_tensor_bounds(entry)
    if bounds is None:
        return None
    _, _, shape = bounds
    dtype_info = _safetensors_dtype_info(str(entry["dtype"]))
    if dtype_info is None:
        return None
    np_dtype, mlx_dtype, bitcast_to = dtype_info
    arr = np.frombuffer(buf, dtype=np_dtype, count=_numel(shape)).reshape(shape)
    out = mx.array(arr, dtype=mlx_dtype)
    if bitcast_to is not None:
        out = out.view(bitcast_to)
    return out

def _read_safetensors_tensor(
    path: Path, data_start: int, entry: dict
) -> Optional[mx.array]:
    bounds = _safetensors_tensor_bounds(entry)
    if bounds is None:
        return None
    start, end, _ = bounds
    try:
        with open(path, "rb") as f:
            f.seek(data_start + start)
            raw = f.read(end - start)
            if len(raw) != end - start:
                return None
    except OSError:
        return None
    return _mlx_array_from_safetensors_bytes(memoryview(raw), entry)

def _read_safetensors_axis0_slice_bytes(
    path: Path,
    data_start: int,
    entry: dict,
    axis0_start: int,
    axis0_end: int,
) -> Optional[Tuple[bytes, dict]]:
    bounds = _safetensors_tensor_bounds(entry)
    if bounds is None:
        return None
    start, _, shape = bounds
    if not shape:
        return None
    axis0_start = int(axis0_start)
    axis0_end = int(axis0_end)
    if axis0_start < 0 or axis0_end < axis0_start or axis0_end > shape[0]:
        return None
    dtype_info = _safetensors_dtype_info(str(entry["dtype"]))
    if dtype_info is None:
        return None
    np_dtype, _, _ = dtype_info
    row_bytes = _numel(shape[1:]) * np_dtype.itemsize
    byte_start = start + axis0_start * row_bytes
    byte_end = start + axis0_end * row_bytes
    try:
        with open(path, "rb") as f:
            f.seek(data_start + byte_start)
            raw = f.read(byte_end - byte_start)
            if len(raw) != byte_end - byte_start:
                return None
    except OSError:
        return None

    sliced_entry = dict(entry)
    sliced_entry["shape"] = [axis0_end - axis0_start, *shape[1:]]
    sliced_entry["data_offsets"] = [0, byte_end - byte_start]
    return raw, sliced_entry

def _read_safetensors_axis0_slice(
    path: Path,
    data_start: int,
    entry: dict,
    axis0_start: int,
    axis0_end: int,
) -> Optional[mx.array]:
    sliced = _read_safetensors_axis0_slice_bytes(
        path, data_start, entry, axis0_start, axis0_end
    )
    if sliced is None:
        return None
    raw, sliced_entry = sliced
    return _mlx_array_from_safetensors_bytes(memoryview(raw), sliced_entry)

