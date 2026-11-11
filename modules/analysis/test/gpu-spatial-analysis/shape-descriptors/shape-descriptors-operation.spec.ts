// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUShapeDescriptorsOperation} from '../../../src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation';
import {
  GPUShapeDescriptorsOperation,
  type ShapeDescriptorsOperationProps
} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import {getGPUShapeDescriptorsParameterValues} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors';
import {flattenPolygonFeatures} from '../geometry-measures/polygon-layout';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const OUTPUT = {
  areas: 'areas',
  perimeters: 'perimeters',
  polsbyPopper: 'polsbyPopper',
  schwartzberg: 'schwartzberg',
  elongation: 'elongation',
  orientation: 'orientation',
  convexity: 'convexity',
  clockwise: 'clockwise',
  sliver: 'sliver'
};

for (const convexityMethod of ['gift-wrapping', 'monotone-chain'] as const) {
  for (const edgeCase of [false, true]) {
    it(`shape twins compare all nine columns (${convexityMethod}, ${edgeCase ? 'empty/degenerate' : 'ordinary/holed'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = flattenPolygonFeatures(
        edgeCase
          ? [
              [],
              [
                [
                  [0, 0],
                  [2, 0]
                ]
              ]
            ]
          : [
              [
                [
                  [0, 0],
                  [4, 0],
                  [4, 2],
                  [0, 2]
                ]
              ],
              [
                [
                  [0, 0],
                  [6, 0],
                  [6, 2],
                  [0, 2]
                ],
                [
                  [2, 0.5],
                  [4, 0.5],
                  [4, 1.5],
                  [2, 1.5]
                ]
              ]
            ]
      );
      const props: ShapeDescriptorsOperationProps = {
        positions: 'positions',
        ringOffsets: 'ringOffsets',
        polygonOffsets: 'polygonOffsets',
        featureOffsets: 'featureOffsets',
        parameters: 'parameters',
        featureCount: 2,
        convexityMethod,
        output: OUTPUT
      };
      const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
      for (const Operation of [GPUShapeDescriptorsOperation, CPUShapeDescriptorsOperation]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: input.positions},
              ringOffsets: {format: 'uint32', values: input.ringOffsets},
              polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
              featureOffsets: {format: 'uint32', values: input.featureOffsets},
              parameters: {
                format: 'float32',
                values: getGPUShapeDescriptorsParameterValues({sliverThreshold: 0.3})
              }
            },
            operations: [new Operation(props)],
            outputs: Object.values(OUTPUT)
          })
        );
      }
      for (const column of Object.values(OUTPUT)) {
        expect(results[0][column].length).toBe(2);
        results[1][column].forEach((expected, index) => {
          const actual = results[0][column][index];
          if (column === 'clockwise' || column === 'sliver') expect(actual).toBe(expected);
          else if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
          // 1e-4 relative (with an absolute 1e-4 floor), as documented by the contributor.
          else
            expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
              1e-4 * Math.max(1, Math.abs(expected))
            );
        });
      }
    });
  }
}

for (const convexityMethod of ['gift-wrapping', 'monotone-chain'] as const) {
  for (const regression of ['zero-area-hole', 'descending-offsets'] as const) {
    it(`shape twins compare every column for ${regression} (${convexityMethod})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const positions =
        regression === 'zero-area-hole'
          ? new Float32Array([0, 0, 8, 0, 8, 8, 0, 8, 1, 2, 7, 6, 1, 6, 7, 2])
          : new Float32Array([0, 0, 4, 0, 4, 4, 0, 4]);
      const props: ShapeDescriptorsOperationProps = {
        positions: 'positions',
        ringOffsets: 'ringOffsets',
        polygonOffsets: 'polygonOffsets',
        featureOffsets: 'featureOffsets',
        parameters: 'parameters',
        featureCount: 1,
        convexityMethod,
        output: OUTPUT
      };
      const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
      for (const Operation of [GPUShapeDescriptorsOperation, CPUShapeDescriptorsOperation]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: positions},
              ringOffsets: {
                format: 'uint32',
                values: new Uint32Array([0, 4, regression === 'zero-area-hole' ? 8 : 0])
              },
              polygonOffsets: {format: 'uint32', values: new Uint32Array([0, 2])},
              featureOffsets: {format: 'uint32', values: new Uint32Array([0, 1])},
              parameters: {format: 'float32', values: getGPUShapeDescriptorsParameterValues()}
            },
            operations: [new Operation(props)],
            outputs: Object.values(OUTPUT)
          })
        );
      }
      for (const column of Object.values(OUTPUT)) {
        expect(results[0][column].length).toBe(1);
        expect(results[1][column].length).toBe(1);
        const expected = results[1][column][0];
        const actual = results[0][column][0];
        if (column === 'clockwise' || column === 'sliver') expect(actual).toBe(expected);
        else if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
        // The contributor documents 1e-4 relative precision, with the same absolute floor.
        else
          expect(Math.abs(actual - expected)).toBeLessThanOrEqual(
            1e-4 * Math.max(1, Math.abs(expected))
          );
      }
      for (const result of results) {
        if (regression === 'zero-area-hole') {
          expect(result['orientation'][0]).toBeCloseTo(Math.PI / 2, 4);
          expect(result['elongation'][0]).toBeCloseTo(1 - Math.sqrt(256 / 259), 5);
        } else {
          expect(Number.isNaN(result['convexity'][0])).toBe(true);
        }
      }
    });
  }
}
