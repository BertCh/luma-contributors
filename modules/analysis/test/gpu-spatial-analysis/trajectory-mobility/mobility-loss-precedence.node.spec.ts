// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeMobilityIndicatorsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
it('suppresses invalidated staypoint row contents before reading indices', () => {
  for (const start of [0, 0xffffffff]) {
    const result = computeMobilityIndicatorsOnCPU(
      {
        timestamps: Float32Array.of(0),
        staypoints: {
          tracks: Uint32Array.of(0),
          centroids: Float32Array.of(0, 0),
          startRows: Uint32Array.of(start),
          durations: Float32Array.of(1),
          locationIds: Uint32Array.of(0),
          count: Uint32Array.of(1),
          overflow: Uint32Array.of(1)
        }
      },
      {trackCount: 1}
    );
    for (const port of [
      'radiusOfGyration',
      'jumpLengths',
      'waitingTimes',
      'randomEntropy',
      'uncorrelatedEntropy',
      'realEntropy',
      'trackingQuality'
    ] as const)
      expect(result[port]).toEqual(Float32Array.of(NaN));
    expect(result.homeLocations[0]).toBe(0xffffffff);
    expect(result.workLocations[0]).toBe(0xffffffff);
    expect(result.candidateOverflow[0]).toBe(1);
  }
});

import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {CPUMobilityIndicatorsOperation} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
function vector(
  format: 'float32' | 'float32x2' | 'uint32',
  values: Float32Array | Uint32Array
): GPUCPUVector {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}
it('suppresses named outputs with and without an optional output loss view', async () => {
  for (const start of [0, 0xffffffff])
    for (const withStatus of [false, true]) {
      const operation = new CPUMobilityIndicatorsOperation({
        trackCount: 1,
        rowCount: 1,
        staypointCapacity: 1,
        timestamps: 'timestamps',
        staypoints: {
          tracks: 'tracks',
          centroids: 'centroids',
          startRows: 'starts',
          durations: 'durations',
          locationIds: 'locations',
          status: {count: 'count', overflow: 'overflow'}
        },
        output: {
          radiusOfGyration: 'radius',
          homeLocations: 'home',
          ...(withStatus ? {status: {candidateOverflow: 'loss'}} : {})
        }
      });
      const result = await operation.execute({
        inputs: {
          timestamps: vector('float32', Float32Array.of(0)),
          stayTracks: vector('uint32', Uint32Array.of(0)),
          centroids: vector('float32x2', Float32Array.of(0, 0)),
          stayStartRows: vector('uint32', Uint32Array.of(start)),
          durations: vector('float32', Float32Array.of(1)),
          locationIds: vector('uint32', Uint32Array.of(0)),
          stayCount: vector('uint32', Uint32Array.of(1)),
          stayOverflow: vector('uint32', Uint32Array.of(1))
        }
      });
      expect(result['radius']).toEqual(Float32Array.of(NaN));
      expect(result['home']).toEqual(Uint32Array.of(0xffffffff));
      if (withStatus) expect(result['loss']).toEqual(Uint32Array.of(1));
    }
});
