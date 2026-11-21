// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterWatershedSegmentationOperation} from '../../src/raster-analysis/raster-watershed/gpu-raster-watershed-segmentation-operation';
import {CPURasterWatershedSegmentationOperation} from '../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const maximumSteps of [1, 2]) {
  it(`derives watershed completion from the two-cell FIFO queue at budget ${maximumSteps}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [
      GPURasterWatershedSegmentationOperation,
      CPURasterWatershedSegmentationOperation
    ]) {
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: Float32Array.of(1, 1)},
          markers: {format: 'uint32', values: Uint32Array.of(7, 0)}
        },
        operations: [
          new Operation({
            grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
            input: {id: 'band', format: 'float32', storage: {kind: 'buffer', values: 'values'}},
            markers: 'markers',
            maximumSteps,
            output: {
              labels: 'labels',
              status: {converged: 'converged', iterationLimitReached: 'limit'}
            }
          })
        ],
        outputs: ['converged', 'limit']
      });
      // Initially [cell 0]; pop 0 enqueues cell 1, which remains pending at budget one.
      // Pop 1 empties the queue at budget two. Both complete scalar words are exact.
      expect(Array.from(result['converged'])).toEqual([maximumSteps === 1 ? 0 : 1]);
      expect(Array.from(result['limit'])).toEqual([maximumSteps === 1 ? 1 : 0]);
    }
  });
}
