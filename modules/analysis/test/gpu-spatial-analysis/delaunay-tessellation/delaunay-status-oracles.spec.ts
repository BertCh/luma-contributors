// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Four copies of one point: one insertion, three duplicates, no triangle or
// illegal edge, hence complete. A square has 2n-2-h = 2 triangles; capacity 1
// truncates exactly one. A non-finite coordinate cancels before any insertion.
// All status words, including zero-only iterationLimitReached, are poisoned.
it('Delaunay status follows independent degenerate and square cardinalities on replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', Float32Array.of(2, 2, 2, 2, 2, 2, 2, 2));
  const status = {
    count: output('count', 'uint32', 1),
    requiredCount: output('required', 'uint32', 1),
    overflow: output('overflow', 'uint32', 1),
    invalidCount: output('invalid', 'uint32', 1),
    duplicateCount: output('duplicates', 'uint32', 1),
    converged: output('converged', 'uint32', 1),
    iterationCount: output('iterations', 'uint32', 1),
    iterationLimitReached: output('limit', 'uint32', 1)
  };
  graph.add(
    new GPUDelaunayTessellation({
      positions,
      output: {triangles: output('triangles', 'uint32x3', 1), status}
    })
  );
  const compiled = graph.compile();
  try {
    for (const frame of [
      {positions: [2, 2, 2, 2, 2, 2, 2, 2], status: [0, 0, 0, 0, 3, 1, 0], iterations: 1},
      {positions: [0, 0, 1, 0, 1, 1, 0, 1], status: [1, 2, 1, 0, 0, 1, 0], iterations: undefined},
      {positions: [NaN, 0, 1, 0, 2, 0, 3, 0], status: [0, 0, 0, 1, 0, 0, 0], iterations: 0},
      {positions: [2, 2, 2, 2, 2, 2, 2, 2], status: [0, 0, 0, 0, 3, 1, 0], iterations: 1}
    ]) {
      fixture.writeInput(positions, Float32Array.from(frame.positions));
      const result = await fixture.submitTwice(compiled, async () => ({
        status: await fixture.readRows([
          status.count,
          status.requiredCount,
          status.overflow,
          status.invalidCount,
          status.duplicateCount,
          status.converged,
          status.iterationLimitReached
        ]),
        iterations: await fixture.read(status.iterationCount)
      }));
      expect(result.status).toEqual(frame.status);
      // A single distinct site has exactly one active insertion; invalid input has none.
      if (frame.iterations !== undefined) expect(result.iterations).toEqual([frame.iterations]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
