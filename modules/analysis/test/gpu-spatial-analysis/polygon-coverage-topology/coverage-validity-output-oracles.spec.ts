// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCoverageValidity} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/gpu-coverage-validity';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';

it('counts invalid segments on the owning feature with feature counts alone and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Feature 0 is empty, feature 1 owns the containing square, and feature 2 owns the triangle
  // plus a disjoint square. Only the THREE triangle edges overlap another feature's interior.
  // Counts [0,0,3] differ from per-polygon [0,3,0], catching polygon/feature-row substitution.
  for (const counts of [true, false]) {
    const fixture = createOutputOracleFixture(device);
    const {graph, column, output} = fixture;
    graph.add(
      new GPUCoverageValidity({
        positions: column(
          'input',
          'float32x2',
          new Float32Array([0, 0, 8, 0, 8, 8, 0, 8, 2, 2, 4, 2, 2, 4, 10, 0, 12, 0, 12, 2, 10, 2])
        ),
        ringOffsets: column('rings', 'uint32', new Uint32Array([0, 4, 7, 11])),
        polygonOffsets: column('polygons', 'uint32', new Uint32Array([0, 1, 2, 3])),
        featureOffsets: column('features', 'uint32', new Uint32Array([0, 0, 1, 3])),
        parameters: column('parameters', 'float32', new Float32Array(4)),
        output: {
          segmentFlags: output('flags', 'uint32', 11),
          ...(counts ? {featureInvalidCounts: output('featureInvalidCounts', 'uint32', 3)} : {})
        }
      })
    );
    try {
      const result = await fixture.runGPU();
      expect(result['flags']).toEqual([0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0]);
      if (counts) expect(result['featureInvalidCounts']).toEqual([0, 0, 3]);
    } finally {
      fixture.destroy();
    }
  }
});
