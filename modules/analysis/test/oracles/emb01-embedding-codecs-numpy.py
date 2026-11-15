#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent EMB-01 fixtures. Run with /usr/bin/python3 (NumPy required).

Definitions: GeoTessera core.dequantize_embedding; IEEE binary16 via NumPy's dtype;
AlphaEarth S22.1 power=2, scale=127.5, nearest-even rounding and signed saturation.
Synthetic numbers only; no model weights or datasets. JSON null represents non-finite.
"""
import json
from pathlib import Path
import numpy as np


def serial(values):
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values).ravel()]


def pack(values, dtype):
    data = np.asarray(values, dtype=dtype).tobytes()
    return np.frombuffer(data + bytes((-len(data)) % 4), dtype='<u4').astype(np.uint64).tolist()


def decoded(values, mask):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(values).all(axis=1) & (np.asarray(mask) != 0)
    values[~valid] = np.nan
    normalized = values.astype(np.float64)
    lengths = np.linalg.norm(normalized, axis=1)
    nonzero = valid & (lengths > 0)
    normalized[nonzero] /= lengths[nonzero, None]
    return dict(vectors=serial(values), normalized=serial(normalized), validity=valid.astype(int).tolist())


def main():
    pixel_count = 67  # partial 64-thread workgroup; 201 components needs final-word padding
    mask = np.ones(pixel_count, dtype=np.uint32)
    mask[[4, 65]] = 0
    signed = ((np.arange(pixel_count * 3) * 37) % 256 - 128).astype(np.int8)
    signed[:9] = [-128, -1, 127, 0, 0, 0, 3, -4, 0]
    scales = np.resize(np.array([0.125, 0, -0.5, 2, np.nan, np.inf, -np.inf], dtype=np.float32), pixel_count)
    scales[63] = 1e30  # finite decoded values whose unscaled squared norm overflows float32
    scales[-2:] = [1e38, 0.0625]  # overflow in decoded product; last pixel is valid
    with np.errstate(over='ignore', invalid='ignore'):
        tessera_values = signed.reshape(-1, 3).astype(np.float32) * scales[:, None]
    tessera = dict(words=pack(signed, 'i1'), scales=serial(scales),
                   scaleKinds=['nan' if np.isnan(value) else 'positiveInfinity' if value == np.inf else
                               'negativeInfinity' if value == -np.inf else 'finite' for value in scales],
                   **decoded(tessera_values, mask))
    # Include both half-word positions, signed zeros, subnormals, finite extrema and all exponent-31 cases.
    half_bits = np.resize(np.array([0, 0x8000, 1, 0x8001, 0x3ff, 0x400, 0x3c00, 0xbc00,
                                   0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e00, 0xfe01,
                                   0x3555, 0x0401, 0x03fe, 0x8400, 0x3c01], dtype='<u2'), pixel_count * 3)
    half_bits[-3:] = [1, 0x8001, 0x3c00]
    half = dict(words=pack(half_bits, '<u2'), **decoded(half_bits.view('<f2').astype(np.float32), mask))
    vectors = np.resize(np.array([-1, 0, 1, -1 / 9, 1 / 9, 0.25, -0.25, 4, -4,
                                  np.nan, np.inf, -np.inf, 0.0001, -0.0001, 0.5], dtype=np.float32), pixel_count * 3)
    vectors[-3:] = [1, -1, 0.25]
    rows = vectors.reshape(-1, 3)
    valid = np.isfinite(rows).all(axis=1) & (mask != 0)
    with np.errstate(invalid='ignore'):
        codes = np.clip(np.rint(np.sign(rows) * np.sqrt(np.minimum(np.abs(rows), 1)) * np.float32(127.5)), -127, 127)
    codes[~valid] = -128
    encode = dict(vectors=serial(vectors),
                  valueKinds=['nan' if np.isnan(value) else 'positiveInfinity' if value == np.inf else
                              'negativeInfinity' if value == -np.inf else 'finite' for value in vectors],
                  words=pack(codes.astype(np.int8), 'i1'), validity=valid.astype(int).tolist())
    result = dict(pixelCount=pixel_count, dimensions=3, mask=mask.tolist(), tessera=tessera, half=half, encode=encode)
    path = Path(__file__).parent / 'fixtures/emb01-embedding-codecs-numpy.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
