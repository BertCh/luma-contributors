// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GraphHarness} from '../trajectory-staypoints/movement-spec-utils';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it('direct zonal polygon uncertainty counts an invalid ring and clears after repair', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'direct-zonal-status-oracle');
  try {
    const positions = fixture.output('polygon-positions', 'float32x2', 8);
    const counts = fixture.output('counts', 'uint32', 2);
    const uncertainty = fixture.output('uncertainty', 'uint32', 1);
    fixture.graph.add(
      new GPUZonalStatistics({
        positions: fixture.input('points', Float32Array.from([1, 1, 3, 1]), 'float32x2', 2),
        features: {
          kind: 'polygons',
          positions: positions.view,
          featureOffsets: fixture.input('features', Uint32Array.from([0, 1, 2]), 'uint32', 3),
          polygonOffsets: fixture.input('polygons', Uint32Array.from([0, 1, 2]), 'uint32', 3),
          ringOffsets: fixture.input('rings', Uint32Array.from([0, 4, 8]), 'uint32', 3),
          candidateCapacity: 4,
          boundaryRule: 'closed'
        },
        output: {counts: counts.view, status: {uncertainCount: uncertainty.view}}
      })
    );
    // Each point is in exactly one disjoint finite vertex envelope. A nonfinite
    // vertex invalidates only the first ring; repair restores its square membership.
    // Ordinary points deliberately omit multipart offsets so the direct zonal path runs.
    for (const scenario of [
      {corner: NaN, counts: [0, 1], uncertain: [1]},
      {corner: 2, counts: [1, 1], uncertain: [0]},
      {corner: NaN, counts: [0, 1], uncertain: [1]}
    ]) {
      positions.buffer.write(
        Float32Array.from([0, 0, 2, 0, scenario.corner, 2, 0, 2, 2, 0, 4, 0, 4, 2, 2, 2])
      );
      counts.buffer.write(Uint32Array.from([37, 37]));
      uncertainty.buffer.write(Uint32Array.from([37]));
      fixture.run();
      expect(await fixture.readUint32(counts, 2)).toEqual(scenario.counts);
      expect(await fixture.readUint32(uncertainty, 1)).toEqual(scenario.uncertain);
    }
  } finally {
    fixture.destroy();
  }
});

it('nearest planar candidate count enumerates two point boxes and clears after invalid radius', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Query (0,0), point features (0,0),(1,0): both boxes meet the radius-2
  // query box, exactly two candidates. Negative radius accepts none. The
  // count is poisoned before EVERY encoding, so even the final zero is a write.
  for (const scenario of [
    {radius: 2, count: [2]},
    {radius: -1, count: [0]}
  ]) {
    const harness = new GraphHarness(device, 'nearest-candidate-status');
    const count = harness.output('candidate-count', 'uint32', 1);
    count.buffer.write(new Uint32Array([0xdeadbeef]));
    harness.graph.add(
      new GPUNearestFeatureJoin({
        queries: {
          kind: 'points',
          positions: harness.input('query', new Float32Array([0, 0]), 'float32x2', 1)
        },
        features: {
          kind: 'points',
          positions: harness.input('features', new Float32Array([0, 0, 1, 0]), 'float32x2', 2)
        },
        radius: harness.input('radius', new Float32Array([scenario.radius]), 'float32', 1),
        candidateCapacity: 4,
        output: {
          nearestFeatureIds: harness.output('nearest', 'uint32', 1).view,
          status: {
            overflow: harness.output('overflow', 'uint32', 1).view,
            candidateCount: count.view
          }
        }
      })
    );
    await harness.run(async () => {
      expect(await readUint32(count.buffer, 1)).toEqual(scenario.count);
    });
  }
});
