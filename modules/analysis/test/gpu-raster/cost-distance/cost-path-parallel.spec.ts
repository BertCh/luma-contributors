// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUCostDistancePath} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path';
import {
  GPU_COST_DISTANCE_NONE,
  GPU_COST_DISTANCE_UNRESOLVED
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createCensusRasterHarness, expectCensusValues} from '../census-raster-oracle-utils';

for (const {width, prefix} of [
  {width: 256, prefix: [255, 254, 253]},
  {width: 257, prefix: [256, 255, 254]}
]) {
  for (const capacity of [3, width]) {
    it(`cost path route boundary ${width} preserves ordered prefix at capacity ${capacity}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createCensusRasterHarness(device);
      // A westward row has the closed-form path target, target-1, ..., 0. No relaxation
      // algorithm or CPU twin is used as an oracle; code 16 points one column west.
      const links = new Array<number>(width).fill(16);
      links[0] = 0;
      const result = await harness.run(
        new GPUCostDistancePath({
          grid: makeGPURasterMetadata({width, height: 1}),
          backLinks: harness.input('uint32', links),
          target: harness.input('uint32', [width - 1]),
          output: {
            path: {
              ids: harness.output('ids', 'uint32', capacity),
              status: {
                count: harness.output('count', 'uint32', 1),
                overflow: harness.output('overflow', 'uint32', 1),
                requiredCount: harness.output('required', 'uint32', 1),
                candidateOverflow: harness.output('candidate', 'uint32', 1)
              }
            },
            status: {iterationLimitReached: harness.output('unresolved', 'uint32', 1)}
          }
        })
      );
      expectCensusValues(result['ids'].slice(0, 3), prefix);
      if (capacity === width) {
        expectCensusValues(
          result['ids'],
          Array.from({length: width}, (_, row) => width - 1 - row)
        );
      }
      expectCensusValues(result['count'], [capacity]);
      expectCensusValues(result['overflow'], [Number(capacity < width)]);
      expectCensusValues(result['required'], [width]);
      expectCensusValues(result['candidate'], [0]);
      expectCensusValues(result['unresolved'], [0]);
    });
  }
}

for (const failure of [
  'unresolved',
  'cycle',
  'invalid-code',
  'off-grid',
  'unreached',
  'target-outside'
] as const) {
  it(`parallel cost path suppresses ${failure} after a long prefix`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createCensusRasterHarness(device);
    const links = new Array<number>(257).fill(16);
    links[0] =
      failure === 'unresolved'
        ? GPU_COST_DISTANCE_UNRESOLVED
        : failure === 'cycle'
          ? 1
          : failure === 'invalid-code'
            ? 3
            : failure === 'unreached'
              ? GPU_COST_DISTANCE_NONE
              : 16;
    const result = await harness.run(
      new GPUCostDistancePath({
        grid: makeGPURasterMetadata({width: 257, height: 1}),
        backLinks: harness.input('uint32', links),
        target: harness.input('uint32', [failure === 'target-outside' ? 257 : 256]),
        output: {
          path: {
            ids: harness.output('ids', 'uint32', 3),
            status: {
              count: harness.output('count', 'uint32', 1),
              overflow: harness.output('overflow', 'uint32', 1),
              requiredCount: harness.output('required', 'uint32', 1)
            }
          },
          status: {iterationLimitReached: harness.output('unresolved', 'uint32', 1)}
        }
      })
    );
    // 256 -> ... -> 0 reaches no source. For a cycle, 0 -> 1 -> 0; code 16
    // from cell zero goes off-grid. Only the explicit unresolved sentinel sets that status.
    expectCensusValues(result['count'], [0]);
    expectCensusValues(result['overflow'], [0]);
    expectCensusValues(result['required'], [0]);
    expectCensusValues(result['unresolved'], [Number(failure === 'unresolved')]);
  });
}
