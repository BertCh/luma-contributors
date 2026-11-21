// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPULabelPointOperation} from '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation';
import {
  GPULabelPointOperation,
  type LabelPointOperationProps
} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point-operation';
import {flattenPolygonFeatures} from '../geometry-measures/polygon-layout';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const smallFeatureVertexLimit of [0, 32]) {
  for (const edgeCase of [false, true]) {
    it(`label twins agree on every column (${edgeCase ? 'empty/collapsed' : 'square/triangle'}, vertex limit ${smallFeatureVertexLimit})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = flattenPolygonFeatures(
        edgeCase
          ? [
              [],
              [
                [
                  [2, 3],
                  [2, 7]
                ]
              ]
            ]
          : [
              [
                [
                  [0, 0],
                  [4, 0],
                  [4, 4],
                  [0, 4]
                ]
              ],
              [
                [
                  [0, 0],
                  [3, 0],
                  [0, 4]
                ]
              ]
            ]
      );
      const props: LabelPointOperationProps = {
        positions: 'positions',
        ringOffsets: 'ringOffsets',
        polygonOffsets: 'polygonOffsets',
        featureOffsets: 'featureOffsets',
        featureCount: 2,
        smallFeatureVertexLimit,
        output: {points: 'points', publishedCenterClearance: 'distances', degenerate: 'degenerate'}
      };
      const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
      for (const Operation of [GPULabelPointOperation, CPULabelPointOperation]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: input.positions},
              ringOffsets: {format: 'uint32', values: input.ringOffsets},
              polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
              featureOffsets: {format: 'uint32', values: input.featureOffsets}
            },
            operations: [new Operation(props)],
            outputs: ['points', 'distances', 'degenerate']
          })
        );
      }
      expect(Array.from(results[0]['degenerate'])).toEqual(Array.from(results[1]['degenerate']));
      // Absolute 0.002 position units covers the greedy GPU grid's final spacing and f32 error
      // on these unique-pole fixtures; NaN and degenerate rows must agree exactly.
      for (const column of ['points', 'distances']) {
        expect(results[0][column].length).toBe(results[1][column].length);
        results[1][column].forEach((expected, index) => {
          if (Number.isNaN(expected)) expect(Number.isNaN(results[0][column][index])).toBe(true);
          else
            expect(Math.abs(results[0][column][index] - expected)).toBeLessThanOrEqual(
              edgeCase ? 0 : 0.002
            );
        });
      }
    });
  }
}

it('label twins treat descending final ring offsets as an empty feature', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: LabelPointOperationProps = {
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    featureCount: 1,
    output: {points: 'points', publishedCenterClearance: 'distances', degenerate: 'degenerate'}
  };
  const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
  for (const Operation of [GPULabelPointOperation, CPULabelPointOperation]) {
    results.push(
      await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: new Float32Array([0, 0, 4, 0, 4, 4, 0, 4])},
          ringOffsets: {format: 'uint32', values: new Uint32Array([0, 4, 0])},
          polygonOffsets: {format: 'uint32', values: new Uint32Array([0, 2])},
          featureOffsets: {format: 'uint32', values: new Uint32Array([0, 1])}
        },
        operations: [new Operation(props)],
        outputs: ['points', 'distances', 'degenerate']
      })
    );
  }
  // Empty rows and flags have an exact comparison, including both point components.
  for (const result of results) {
    expect(Array.from(result['points'])).toEqual([NaN, NaN]);
    expect(Array.from(result['distances'])).toEqual([NaN]);
    expect(Array.from(result['degenerate'])).toEqual([1]);
  }
});
