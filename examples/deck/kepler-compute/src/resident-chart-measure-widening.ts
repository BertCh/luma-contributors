// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {GPUVectorEvaluator} from '@luma.gl/graph/gpu-simple-graph';
import type {KeplerComputeSession, KeplerMemoryReservation} from '@luma-contributors/kepler';
import {assertExactFloat32IntegerRange} from './integer-style-range';

/** Float32 copy of an Int32 or Uint32 chart measure; the owner destroys it with the explorer. */
export type ResidentChartWidenedMeasure = {
  vector: GPUVector<'float32'>;
  destroy(): void;
};

/**
 * Widens an Int32 or Uint32 column to Float32 on the GPU. Throws before any copy when a stored
 * value lies outside ±2^24, where the conversion would round. Float32 columns need no widening.
 */
export async function widenResidentChartMeasure(props: {
  session: KeplerComputeSession;
  vector: GPUVector<'sint32'> | GPUVector<'uint32'>;
}): Promise<ResidentChartWidenedMeasure> {
  const {session, vector} = props;
  const reservation: KeplerMemoryReservation = session.registry.reserveMemory(
    vector.length * 4,
    'gpu',
    'scratch'
  );
  let evaluator: GPUVectorEvaluator | undefined;
  try {
    // Lazy import keeps the gpgpu operation graph out of unrelated chart module loads.
    const {castData, GPUVectorEvaluator} = await import(
      '@luma.gl/graph/gpu-simple-graph'
    );
    await assertExactFloat32IntegerRange(session.device, vector);
    evaluator = GPUVectorEvaluator.fromGPUVector(vector).mapGPUData(chunk =>
      castData(chunk, {inputFormat: vector.format, outputFormat: 'float32'})
    );
    const widened = (await evaluator.evaluate(session.device)) as GPUVector<'float32'>;
    const owned = evaluator;
    return {
      vector: widened,
      destroy() {
        owned.destroy();
        reservation.release();
      }
    };
  } catch (error) {
    evaluator?.destroy();
    reservation.release();
    throw error;
  }
}
