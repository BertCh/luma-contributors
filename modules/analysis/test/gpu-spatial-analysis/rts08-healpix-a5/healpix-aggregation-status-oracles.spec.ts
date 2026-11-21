// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUHealpixCellAggregation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-cell-aggregation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';

// These are already-keyed order-1 cells, independent of point-to-cell arithmetic.
// Sorted multisets: 0 occurs twice, 5 twice, 7 once, 9 once. UNIQ adds 4^(1+1)=16.
const CASES = [
  {
    name: 'complete',
    capacity: 4,
    mask: [1, 1, 1, 1, 1, 1],
    values: [2, 3, 5, 7, 11, 13],
    keys: [0, 5, 7, 9],
    counts: [2, 2, 1, 1],
    status: [4, 4, 0, 0]
  },
  {
    name: 'two cells fit',
    capacity: 2,
    mask: [1, 1, 1, 1, 1, 1],
    values: [2, 3, 5, 7, 11, 13],
    keys: [0, 5],
    counts: [2, 2],
    status: [2, 4, 1, 0]
  },
  // Rows 2 (masked), 4 (NaN), 5 (infinity) are excluded exactly once each.
  {
    name: 'three excluded rows',
    capacity: 4,
    mask: [1, 1, 0, 1, 1, 1],
    values: [2, 3, 5, 7, NaN, Infinity],
    keys: [0, 5],
    counts: [2, 1, 0, 0],
    status: [2, 2, 0, 3]
  },
  {
    name: 'all rows masked',
    capacity: 4,
    mask: [0, 0, 0, 0, 0, 0],
    values: [2, 3, 5, 7, 11, 13],
    keys: [],
    counts: [0, 0, 0, 0],
    status: [0, 0, 0, 6]
  }
];

for (const scheme of ['nested', 'uniq'] as const) {
  it.each(
    CASES
  )(`HEALPix ${scheme} counts and all status scalars overwrite poison: $name`, async scene => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOracleFixture(device);
    const keyOffset = scheme === 'uniq' ? 16 : 0;
    const cells = fixture.output('uint32x2', scene.capacity);
    const counts = fixture.output('uint32', scene.capacity);
    const status = {
      count: fixture.output('uint32', 1),
      requiredCount: fixture.output('uint32', 1),
      overflow: fixture.output('uint32', 1),
      invalidCount: fixture.output('uint32', 1)
    };
    fixture.graph.add(
      new GPUHealpixCellAggregation({
        order: 1,
        scheme,
        cells: fixture.input(
          'uint32x2',
          [0, 0, 5, 5, 7, 9].flatMap(key => [key + keyOffset, 0])
        ),
        values: fixture.input('float32', scene.values),
        mask: fixture.input('uint32', scene.mask),
        cellCapacity: scene.capacity,
        output: {
          cells: cells.view,
          counts: counts.view,
          status: {
            count: status.count.view,
            requiredCount: status.requiredCount.view,
            overflow: status.overflow.view,
            invalidCount: status.invalidCount.view
          }
        }
      })
    );
    const compiled = fixture.graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        for (const output of [cells, counts, ...Object.values(status)]) output.poison();
        submitGraph(device, compiled, undefined);
        expect(await counts.read()).toEqual(scene.counts);
        expect(await cells.read()).toEqual([
          ...scene.keys.flatMap(key => [key + keyOffset, 0]),
          ...new Array((scene.capacity - scene.keys.length) * 2).fill(0)
        ]);
        const actual = await Promise.all(Object.values(status).map(output => output.read()));
        expect(actual).toEqual(scene.status.map(value => [value]));
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
