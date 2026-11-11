// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {CPURasterWatershedSegmentationOperation} from '../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';
import {
  GPURasterWatershedSegmentationOperation,
  type RasterWatershedSegmentationOperationProps
} from '../../src/raster-analysis/raster-watershed/gpu-raster-watershed-segmentation-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
  for (const noMarkers of [false, true]) {
    it(`${method} watershed graph outputs agree with noMarkers=${noMarkers} (tolerance zero)`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: RasterWatershedSegmentationOperationProps = {
        grid: makeGPURasterMetadata({width: 7, height: 1}),
        input: {
          id: 'band',
          format: 'float32',
          storage: {kind: 'buffer', values: 'values'},
          validity: 'validity'
        },
        markers: 'markers',
        method,
        output: {
          labels: 'labels',
          status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
        }
      };
      const outputs = ['labels', 'converged', 'iterationLimitReached'];
      const results = [];
      for (const Operation of [
        GPURasterWatershedSegmentationOperation,
        CPURasterWatershedSegmentationOperation
      ]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              values: {format: 'float32', values: new Float32Array([5, 2, 2, 2, 2, 2, 5])},
              validity: {format: 'uint32', values: new Uint32Array(7).fill(1)},
              markers: {
                format: 'uint32',
                values: new Uint32Array(noMarkers ? [0, 0, 0, 0, 0, 0, 0] : [1, 0, 0, 0, 0, 0, 2])
              }
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      }
      for (const output of outputs)
        expect(Array.from(results[0][output])).toEqual(Array.from(results[1][output]));
    });
  }
}

for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
  it(`${method} treats the rounded float32 nodata marker as missing`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: RasterWatershedSegmentationOperationProps = {
      grid: makeGPURasterMetadata({width: 2, height: 1}),
      input: {
        id: 'band',
        format: 'float32',
        storage: {kind: 'buffer', values: 'values'},
        noDataValue: 0.1
      },
      markers: 'markers',
      method,
      output: {
        labels: 'labels',
        status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
      }
    };
    const results = [];
    for (const Operation of [
      GPURasterWatershedSegmentationOperation,
      CPURasterWatershedSegmentationOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            values: {format: 'float32', values: new Float32Array([0.1, 2])},
            markers: {format: 'uint32', values: new Uint32Array([7, 0])}
          },
          operations: [new Operation(props)],
          outputs: ['labels', 'converged', 'iterationLimitReached']
        })
      );
    }
    // Every output is integer, compared exactly and pinned to the hand-derived empty-seed result.
    for (const result of results) {
      expect(Array.from(result['labels'])).toEqual([0, 0]);
      expect(Array.from(result['converged'])).toEqual([1]);
      expect(Array.from(result['iterationLimitReached'])).toEqual([0]);
    }
    expect(results[0]).toEqual(results[1]);
  });

  for (const oversized of ['values', 'validity'] as const) {
    it(`${method} operations both reject oversized ${oversized}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const props: RasterWatershedSegmentationOperationProps = {
        grid: makeGPURasterMetadata({width: 1, height: 1}),
        input: {
          id: 'band',
          format: 'float32',
          storage: {kind: 'buffer', values: 'values'},
          validity: 'validity'
        },
        markers: 'markers',
        method,
        output: {
          labels: 'labels',
          status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
        }
      };
      for (const Operation of [
        GPURasterWatershedSegmentationOperation,
        CPURasterWatershedSegmentationOperation
      ]) {
        await expect(
          runOperationGraph(device, {
            inputs: {
              values: {
                format: 'float32',
                values: new Float32Array(oversized === 'values' ? [3, 100] : [3])
              },
              validity: {
                format: 'uint32',
                values: new Uint32Array(oversized === 'validity' ? [1, 1] : [1])
              },
              markers: {format: 'uint32', values: new Uint32Array([7])}
            },
            operations: [new Operation(props)],
            outputs: ['labels', 'converged', 'iterationLimitReached']
          })
        ).rejects.toThrow(
          oversized === 'values' ? /exactly one sample per pixel/ : /exactly one flag per pixel/
        );
      }
    });
  }
}
