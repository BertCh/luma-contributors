// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues
} from '../../../src/raster-analysis/gpu-raster-fractional-coverage';
import {GPURasterEmbeddingZoneMean} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-zone-mean';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const candidateLoss of [false, true]) {
  it(`RT-126 real fractional producer suppresses and clears ${candidateLoss ? 'candidate-only' : 'tuple'} loss`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = createParityGraph(device);
    const grid = {
      width: 2,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const positions = rig.input('positions', [0, 0, 2, 0, 2, 1, 0, 1], 'float32x2');
    const capacity = candidateLoss ? 2 : 1;
    const zoneIds = rig.output('zones', capacity, 'uint32');
    const cellIds = rig.output('cells', capacity, 'uint32');
    const fractions = rig.output('fractions', capacity, 'float32');
    const zoneValidity = rig.output('zone-validity', 1, 'uint32');
    const count = rig.output('count', 1, 'uint32');
    const required = rig.output('required', 1, 'uint32');
    const overflow = rig.output('overflow', 1, 'uint32');
    const candidates = rig.output('candidates', 1, 'uint32');
    const coverage = {
      zoneIds: zoneIds.view,
      cellIds: cellIds.view,
      fractions: fractions.view,
      zoneValidity: zoneValidity.view,
      status: {
        count: count.view,
        requiredCount: required.view,
        overflow: overflow.view,
        candidateOverflow: candidates.view
      }
    };
    const means = rig.output('means', 2, 'float32');
    const weights = rig.output('weights', 1, 'float32');
    rig.graph.add(
      new GPURasterFractionalCoverage({
        grid,
        searchStructure: 'bounds',
        candidateCapacity: candidateLoss ? 1 : 2,
        zones: {
          positions: positions.view,
          featureOffsets: rig.input('features', [0, 1], 'uint32').view,
          polygonOffsets: rig.input('polygons', [0, 1], 'uint32').view,
          ringOffsets: rig.input('rings', [0, 4], 'uint32').view
        },
        parameters: rig.input(
          'parameters',
          Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
          'float32'
        ).view,
        output: coverage
      })
    );
    const vectors = rig.input('vectors', [1, 0, 0, 1], 'float32').view;
    const candidateOnlyMeans = candidateLoss
      ? rig.output('candidate-only-means', 2, 'float32')
      : undefined;
    rig.graph.add(
      new GPURasterEmbeddingZoneMean({
        pixelCount: 2,
        dimensions: 2,
        zoneCount: 1,
        vectors,
        tuples: coverage,
        output: {means: means.view, weights: weights.view}
      })
    );
    if (candidateOnlyMeans)
      rig.graph.add(
        new GPURasterEmbeddingZoneMean({
          id: 'candidate-only-controls',
          pixelCount: 2,
          dimensions: 2,
          zoneCount: 1,
          vectors,
          tuples: {
            zoneIds: zoneIds.view,
            cellIds: cellIds.view,
            fractions: fractions.view,
            status: {count: count.view, candidateOverflow: candidates.view}
          },
          output: {means: candidateOnlyMeans.view}
        })
      );
    const compiled = rig.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      if (candidateLoss) {
        expect(await readUint32(count.buffer, 1)).toEqual([1]);
        expect(await readUint32(candidates.buffer, 1)).toEqual([1]);
        // Only count and candidateOverflow control this second reducer: neither saturated
        // requiredCount nor cleared zoneValidity can accidentally certify suppression.
        expect(candidateOnlyMeans).toBeDefined();
        expect(await readFloat32(candidateOnlyMeans!.buffer, 2)).toEqual([NaN, NaN]);
        expect(await readUint32(required.buffer, 1)).toEqual([0xffffffff]);
        // Tuple capacity is adequate: only the pre-clipping candidate budget was exhausted.
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      } else {
        expect(await readUint32(count.buffer, 1)).toEqual([1]);
        expect(await readUint32(required.buffer, 1)).toEqual([2]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
        expect(await readUint32(candidates.buffer, 1)).toEqual([0]);
      }
      expect(await readFloat32(means.buffer, 2)).toEqual([NaN, NaN]);
      expect(await readFloat32(weights.buffer, 1)).toEqual([NaN]);
      if (candidateLoss) return;
      // Same compiled graph, now exactly cell 0. Its unit vector and area are [1,0] and 1.
      positions.buffer.write(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(count.buffer, 1)).toEqual([1]);
      expect(await readUint32(required.buffer, 1)).toEqual([1]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(candidates.buffer, 1)).toEqual([0]);
      expect(await readFloat32(means.buffer, 2)).toEqual([1, 0]);
      expect(await readFloat32(weights.buffer, 1)).toEqual([1]);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
  });
}
