// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

it('B1 final-round pair loss restores all 17 vertices with and without statistics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const statistics of [false, true]) {
    const fixture = createOutputOracleFixture(device);
    const {column, output, graph} = fixture;
    try {
      // Original partition of [0,6]x[0,4], zigzag shared boundary and a triangular hole.
      const positions = Float32Array.of(
        0,
        0,
        2,
        0,
        2,
        1,
        4,
        2,
        2,
        3,
        2,
        4,
        0,
        4,
        2.75,
        1.9,
        2.75,
        2.1,
        3.1,
        2,
        2,
        0,
        6,
        0,
        6,
        4,
        2,
        4,
        2,
        3,
        4,
        2,
        2,
        1
      );
      graph.add(
        new GPUCoverageSimplification({
          positions: column('input', 'float32x2', positions),
          ringOffsets: column('rings', 'uint32', Uint32Array.of(0, 7, 10, 17)),
          polygonOffsets: column('polygons', 'uint32', Uint32Array.of(0, 2, 3)),
          featureOffsets: column('features', 'uint32', Uint32Array.of(0, 1, 2)),
          parameters: column('parameters', 'float32', Float32Array.of(5, 0, 0, 0)),
          simplifyBoundary: true,
          topologyIterationCount: 1,
          topologyPairCapacity: 8,
          output: {
            positions: output('positions', 'float32x2', 17),
            ringOffsets: output('offsets', 'uint32', 4),
            topologyStatistics: statistics ? output('statistics', 'uint32', 4) : undefined,
            status: {
              requiredCount: output('required', 'uint32', 1),
              overflow: output('overflow', 'uint32', 1)
            }
          }
        })
      );
      const result = await fixture.runGPU();
      expect(result['required']).toEqual([17]);
      expect(result['offsets']).toEqual([0, 7, 10, 17]);
      expect(result['overflow']).toEqual([0]);
      result['positions'].forEach(value => expect(Number.isFinite(value)).toBe(true));
      expect(result['positions']).toEqual(Array.from(positions));
      if (statistics) expect(result['statistics'][3]).toBe(1);
    } finally {
      fixture.destroy();
    }
  }
});
