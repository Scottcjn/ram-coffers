"""Binding the console-side kernels to the node worker.

The node calls one thing — an :class:`~gen9_cluster.node.ExpertRunner` — and this
module decides what is behind it. Selection is by what is actually installed on
*this* console, not by what the inventory file claims, because the inventory is
written by a human and the console is the authority on its own libraries.

Order of preference on a given console:

1. ``libgen9_cpu.so`` if it built here (AVX2, and on Zen 2 that is the ceiling).
2. numpy, which is correct, portable, and slow enough that it should only ever
   be a bring-up path.

The GPU backends are deliberately *not* auto-selected. A Vulkan or HIP runner
needs a device, a queue, a compiled pipeline, and a memory allocation strategy
that outlives a single call; wiring that up belongs to the node's startup, and
guessing at it here would produce a fleet where some consoles silently fell back
to the CPU and nobody noticed until the plan's estimates stopped matching
reality. :func:`describe_backends` reports what is available so a deployment can
fail loudly instead.
"""

from __future__ import annotations

import ctypes
import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from . import fp8
from .node import ExpertRunner, ExpertWeights

#: Where the Makefile leaves the CPU library.
_DEFAULT_LIB = Path(__file__).resolve().parent.parent / "kernels" / "libgen9_cpu.so"


class CpuKernelRunner(ExpertRunner):
    """Runs experts through the AVX2 kernel in ``libgen9_cpu.so``."""

    name = "cpu-avx2"

    def __init__(self, library: Optional[Path] = None):
        path = Path(library) if library else _DEFAULT_LIB
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not built; run `make` in kernels/ (or pass a path)")
        self._lib = ctypes.CDLL(str(path))
        f32 = ctypes.POINTER(ctypes.c_float)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        self._lib.gen9_expert_f32.argtypes = [f32, f32, f32, f32, f32, f32,
                                              ctypes.c_int, ctypes.c_int]
        self._lib.gen9_expert_f32.restype = None
        self._lib.gen9_expert_fp8.argtypes = [u8, f32, u8, f32, u8, f32, f32,
                                              f32, f32, ctypes.c_int,
                                              ctypes.c_int]
        self._lib.gen9_expert_fp8.restype = None
        self._scratch: Optional[np.ndarray] = None

    def rows(self, activation: np.ndarray,
             experts: Sequence[ExpertWeights],
             gates: Sequence[float]) -> np.ndarray:
        x = np.ascontiguousarray(activation, dtype=np.float32)
        hidden = x.size
        out = np.empty((len(experts), hidden), dtype=np.float32)
        if not experts:
            return out
        intermediate = experts[0].gate.shape[0]
        if self._scratch is None or self._scratch.size < 2 * intermediate:
            self._scratch = np.zeros(2 * intermediate, dtype=np.float32)
        scratch = self._scratch

        for index, (weights, gate) in enumerate(zip(experts, gates)):
            # A row of a C-contiguous array is itself contiguous, so the kernel
            # writes its expert's output straight into place.
            row = out[index]
            if weights.quantised:
                self._run_fp8(weights, x, row, scratch, hidden, intermediate)
            else:
                self._run_f32(weights, x, row, scratch, hidden, intermediate)
            row *= np.float32(gate)
        return out

    def _run_f32(self, weights: ExpertWeights, x: np.ndarray,
                 partial: np.ndarray, scratch: np.ndarray, hidden: int,
                 intermediate: int) -> None:
        # Bind the contiguous arrays to names first. Taking .ctypes on a
        # temporary produced by ascontiguousarray inside the call would let that
        # temporary be collected while the kernel is still reading through the
        # pointer into it.
        f32 = ctypes.POINTER(ctypes.c_float)
        gate_w = _contiguous(weights.gate)
        up_w = _contiguous(weights.up)
        down_w = _contiguous(weights.down)
        self._lib.gen9_expert_f32(
            gate_w.ctypes.data_as(f32), up_w.ctypes.data_as(f32),
            down_w.ctypes.data_as(f32), x.ctypes.data_as(f32),
            partial.ctypes.data_as(f32), scratch.ctypes.data_as(f32),
            ctypes.c_int(hidden), ctypes.c_int(intermediate))

    def _run_fp8(self, weights: ExpertWeights, x: np.ndarray,
                 partial: np.ndarray, scratch: np.ndarray, hidden: int,
                 intermediate: int) -> None:
        """The FP8 path: the kernel dequantises inline, so the weights are
        never materialised at fp32 and the read is a quarter the bytes.

        The C kernel indexes scales per row, which is the same layout as the
        flat blocks in :mod:`gen9_cluster.fp8` exactly when the row length is a
        whole number of 128-element blocks. Anything else is refused here
        rather than read at the wrong offsets.
        """
        if weights.scales is None:
            raise ValueError("FP8 expert arrived without block scales")
        if hidden % fp8.BLOCK or intermediate % fp8.BLOCK:
            raise ValueError(
                f"the FP8 kernel needs both dimensions to be a multiple of "
                f"{fp8.BLOCK}; got hidden={hidden}, "
                f"intermediate={intermediate}")
        f32 = ctypes.POINTER(ctypes.c_float)
        u8 = ctypes.POINTER(ctypes.c_uint8)
        gate_w = np.ascontiguousarray(weights.gate, dtype=np.uint8)
        up_w = np.ascontiguousarray(weights.up, dtype=np.uint8)
        down_w = np.ascontiguousarray(weights.down, dtype=np.uint8)
        flat = np.ascontiguousarray(weights.scales, dtype=np.float32).reshape(-1)
        per_matrix = fp8.n_blocks(hidden * intermediate)
        gate_s = np.ascontiguousarray(flat[:per_matrix])
        up_s = np.ascontiguousarray(flat[per_matrix:2 * per_matrix])
        down_s = np.ascontiguousarray(flat[2 * per_matrix:3 * per_matrix])
        self._lib.gen9_expert_fp8(
            gate_w.ctypes.data_as(u8), gate_s.ctypes.data_as(f32),
            up_w.ctypes.data_as(u8), up_s.ctypes.data_as(f32),
            down_w.ctypes.data_as(u8), down_s.ctypes.data_as(f32),
            x.ctypes.data_as(f32), partial.ctypes.data_as(f32),
            scratch.ctypes.data_as(f32), ctypes.c_int(hidden),
            ctypes.c_int(intermediate))


def _contiguous(array: np.ndarray) -> np.ndarray:
    """A C-contiguous fp32 view, copying only when the input is not one."""
    return np.ascontiguousarray(array, dtype=np.float32)


def select_runner(prefer: Optional[str] = None) -> ExpertRunner:
    """Pick the best runner this console can actually use.

    ``prefer`` names a backend from the inventory; if it is unavailable here the
    fallback is used and the caller can see the difference in
    :attr:`ExpertRunner.name`, which the node reports in its STATUS reply. That
    is how a fleet notices that one console has quietly become four times slower
    than its plan assumed.
    """
    if prefer in (None, "cpu-avx2"):
        try:
            return CpuKernelRunner()
        except (FileNotFoundError, OSError):
            return ExpertRunner()
    # Vulkan/HIP runners are constructed explicitly by the node's startup with
    # a device and pipeline; there is nothing safe to auto-select here.
    try:
        return CpuKernelRunner()
    except (FileNotFoundError, OSError):
        return ExpertRunner()


def describe_backends() -> List[str]:
    """What compute paths exist on this machine, for ``g9-probe``."""
    lines = []
    lines.append(f"cpu kernel     {'built' if _DEFAULT_LIB.exists() else 'not built'}"
                 f" ({_DEFAULT_LIB})")
    spv = _DEFAULT_LIB.parent / "expert.comp.spv"
    lines.append(f"vulkan shader  {'compiled' if spv.exists() else 'not compiled'}")
    vulkaninfo = shutil.which("vulkaninfo")
    lines.append(f"vulkaninfo     {vulkaninfo or 'absent'}")
    hip = _DEFAULT_LIB.parent / "libgen9_hip.so"
    lines.append(f"hip kernel     {'built' if hip.exists() else 'not built'}")
    lines.append(f"hipcc          {shutil.which('hipcc') or 'absent'}")
    lines.append(f"rocminfo       {shutil.which('rocminfo') or 'absent'}")
    lines.append(f"cpu threads    {os.cpu_count()}")
    return lines
