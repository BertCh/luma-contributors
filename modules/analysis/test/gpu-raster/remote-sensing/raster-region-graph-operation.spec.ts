// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {CPURasterRegionGraphOperation} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';
import {
  GPURasterRegionGraphOperation,
  type RasterRegionGraphOperationProps
} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const mergeMode of ['fixed', 'adaptive'] as const) {
  for (const missing of [false, true]) {
    it(`${mergeMode} region operations agree with missing=${missing}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: RasterRegionGraphOperationProps = {
        grid: makeGPURasterMetadata({width: 3, height: 2}),
        labels: 'labels',
        values: 'values',
        regionCount: 3,
        mergeMode,
        mergeThreshold: 2,
        maximumIterations: 4,
        output: {
          adjacency: 'adjacency',
          weights: 'weights',
          mapping: 'mapping',
          status: {iterationLimitReached: 'iterationLimitReached'}
        }
      };
      const outputs = ['adjacency', 'weights', 'mapping', 'iterationLimitReached'];
      const results = [];
      for (const Operation of [GPURasterRegionGraphOperation, CPURasterRegionGraphOperation]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              labels: {format: 'uint32', values: new Uint32Array([0, 1, 2, 0, 1, 2])},
              values: {
                format: 'float32',
                values: new Float32Array(
                  missing ? [NaN, NaN, NaN, NaN, NaN, NaN] : [0, 2, 4, 0, 2, 4]
                )
              }
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      }
      for (const output of outputs) {
        expect(results[0][output].length).toBe(results[1][output].length);
        for (let index = 0; index < results[0][output].length; index++) {
          const expected = results[1][output][index];
          // All integer outputs exact; float weights absolute tolerance 1e-5, NaNs exact.
          if (output !== 'weights' || Number.isNaN(expected))
            expect(results[0][output][index]).toBe(expected);
          else expect(Math.abs(results[0][output][index] - expected)).toBeLessThanOrEqual(1e-5);
        }
      }
    });
  }
}
