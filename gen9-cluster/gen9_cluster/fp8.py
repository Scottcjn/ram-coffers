"""FP8 E4M3FN with 128-element block scales, in numpy.

DeepSeek ships its weights in FP8 with a scale per 128-element block, and that
is what makes the model fit on hardware like this at all: 1 byte per parameter
instead of 2 halves the number of consoles needed, and decode is bandwidth-bound
so it roughly doubles the token rate as well.

This module is the *portable* implementation. It exists for three jobs: loading
FP8 shards on a console whose compiled kernel is not available, checking the
compiled kernel against something independent, and letting the reference runner
work in FP8 without pretending numpy has an E4M3 dtype. Where
``libgen9_cpu.so`` exists the C path in ``kernels/fp8.c`` is faster and the two
agree bit for bit on the table; this is not a second, divergent definition of
the format.

E4M3FN, as used by DeepSeek and by ``torch.float8_e4m3fn``: 1 sign bit,
4 exponent bits with bias 7, 3 mantissa bits, no infinities, and 0x7F/0xFF as
the only NaNs. Max normal is 448, min subnormal is 2**-9.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

#: Elements sharing one scale factor. Matches DeepSeek's checkpoints.
BLOCK = 128

#: Largest finite magnitude the format can hold.
MAX_NORMAL = 448.0


def _build_table() -> np.ndarray:
    table = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0xF
        mantissa = code & 0x7
        if exponent == 0:
            magnitude = mantissa * 0.125 * 2.0 ** -6
        elif exponent == 0xF and mantissa == 0x7:
            magnitude = float("nan")
        else:
            magnitude = (1.0 + mantissa * 0.125) * 2.0 ** (exponent - 7)
        table[code] = sign * magnitude
    return table


#: code -> value, built once. Decoding is a gather, not arithmetic.
TABLE = _build_table()


def decode(codes: np.ndarray) -> np.ndarray:
    """Decode raw E4M3FN bytes, without applying any scale."""
    return TABLE[np.asarray(codes, dtype=np.uint8)]


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode a blockwise-quantised tensor to fp32.

    ``codes`` is any shape; blocks run along the flattened array, which is the
    layout the checkpoints use. ``scales`` holds one float per block.
    """
    flat = np.asarray(codes, dtype=np.uint8).reshape(-1)
    blocks = (flat.size + BLOCK - 1) // BLOCK
    scale_array = np.asarray(scales, dtype=np.float32).reshape(-1)
    if scale_array.size != blocks:
        raise ValueError(f"{flat.size} values need {blocks} block scales, "
                         f"got {scale_array.size}")
    padded = blocks * BLOCK
    values = np.zeros(padded, dtype=np.float32)
    values[:flat.size] = TABLE[flat]
    scaled = values.reshape(blocks, BLOCK) * scale_array[:, None]
    return scaled.reshape(-1)[:flat.size].reshape(np.shape(codes))


def quantize(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantise to E4M3FN with per-block scales.

    The scale is chosen so the largest magnitude in the block lands on the
    format's maximum. This is what keeps the error relative rather than
    absolute, and it is why a block of tiny weights next to a large outlier
    does not collapse to zero.
    """
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    blocks = (flat.size + BLOCK - 1) // BLOCK
    padded = np.zeros(blocks * BLOCK, dtype=np.float32)
    padded[:flat.size] = flat
    grid = padded.reshape(blocks, BLOCK)

    peak = np.abs(grid).max(axis=1)
    scales = np.where(peak > 0, peak / MAX_NORMAL, 1.0).astype(np.float32)
    normalised = grid / scales[:, None]

    # Nearest code by value. 256 candidates, so a search beats writing out the
    # bit-twiddling a second time, and it cannot disagree with TABLE.
    finite = np.where(np.isnan(TABLE), np.inf, TABLE)
    order = np.argsort(finite)
    ordered = finite[order]
    idx = np.searchsorted(ordered, normalised.reshape(-1))
    idx = np.clip(idx, 1, len(ordered) - 1)
    lower, upper = ordered[idx - 1], ordered[idx]
    pick = np.where(np.abs(normalised.reshape(-1) - lower)
                    <= np.abs(upper - normalised.reshape(-1)), idx - 1, idx)
    codes = order[pick].astype(np.uint8)
    return (codes.reshape(blocks * BLOCK)[:flat.size].reshape(
        np.shape(values)), scales)


def n_blocks(count: int) -> int:
    """How many block scales ``count`` values need."""
    return (count + BLOCK - 1) // BLOCK
