// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';

/** Largest magnitude below which every integer is exactly representable as Float32. */
export const EXACT_FLOAT32_INTEGER_LIMIT = 2 ** 24;

/**
 * Bounded GPU min/max of a prepared Int32/Uint32 column: one two-value extent per source chunk and
 * an eight-byte readback each, never row data. Null rows are included conservatively, so a stored
 * payload outside the range rejects the column even if its row is invalid.
 * Throws when any value is outside ±2^24, where Float32 style conversion would round.
 */
export async function assertExactFloat32IntegerRange(
  device: Device,
  vector: GPUVector<'sint32'> | GPUVector<'uint32'>
): Promise<void> {
  const {extent, GPUVectorEvaluator} = await import('@luma.gl/graph/gpu-simple-graph');
  const evaluator = GPUVectorEvaluator.fromGPUVector(vector);
  const extents = evaluator.gpuDataEvaluators.map(chunk => extent(chunk));
  try {
    for (const result of extents) {
      const output = await result.evaluate(device);
      const bytes = await output.data[0].buffer.readAsync(0, 8);
      const pair =
        vector.format === 'sint32'
          ? new Int32Array(bytes.slice().buffer)
          : new Uint32Array(bytes.slice().buffer);
      // Integer style column exceeds exact Float32 range (±2^24); prepare Float64 or rescale.
      if (pair[0] < -EXACT_FLOAT32_INTEGER_LIMIT || pair[1] > EXACT_FLOAT32_INTEGER_LIMIT)
        throw new Error('Integer style values exceed the exact Float32 range of 2^24.');
    }
  } finally {
    for (const result of extents) result.destroy();
    evaluator.destroy();
  }
}
