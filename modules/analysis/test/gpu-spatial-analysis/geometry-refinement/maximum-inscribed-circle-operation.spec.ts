// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMaximumInscribedCircleOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {
  GPUMaximumInscribedCircleOperation,
  type MaximumInscribedCircleOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle-operation';
import {flattenPolygonFeatures} from '../geometry-measures/polygon-layout';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const edgeCase of [false, true]) {
  it(`circle twins compare all circle, witness and status fields (${edgeCase ? 'empty/non-finite' : 'ordinary'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = flattenPolygonFeatures(
      edgeCase
        ? [[], [[[NaN, 0]]]]
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
                [10, 10],
                [14, 10],
                [14, 14],
                [10, 14]
              ]
            ],
            [
              [
                [0, 0],
                [4, 0],
                [0, 4]
              ]
            ]
          ]
    );
    const props: MaximumInscribedCircleOperationProps = {
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      featureCount: input.featureOffsets.length - 1,
      tolerance: 0.01,
      cellCapacity: 4096,
      maximumIterations: 65536,
      output: {circles: 'circles', witnesses: 'witnesses', status: 'status'}
    };
    const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
    for (const Operation of [
      GPUMaximumInscribedCircleOperation,
      CPUMaximumInscribedCircleOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: input.positions},
            ringOffsets: {format: 'uint32', values: input.ringOffsets},
            polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
            featureOffsets: {format: 'uint32', values: input.featureOffsets}
          },
          operations: [new Operation(props)],
          outputs: ['circles', 'witnesses', 'status']
        })
      );
    }
    expect(Array.from(results[0]['status'])).toEqual(edgeCase ? [2, 3] : [0, 0, 0]);
    expect(Array.from(results[0]['status'])).toEqual(Array.from(results[1]['status']));
    for (const result of results) {
      for (let feature = 0; feature < props.featureCount; feature++) {
        if (result['status'][feature] === 0) {
          const error = result['circles'][feature * 4 + 3];
          expect(error).toBeGreaterThanOrEqual(0);
          expect(error).toBeLessThanOrEqual(Math.fround(props.tolerance));
        }
      }
      if (!edgeCase) {
        // Square cells adjacent to the optimum have error (sqrt(2) - 1) * halfSize.
        // HalfSize 1/32 exceeds tolerance; 1/64 is the first successful level.
        expect(Math.abs(result['circles'][3] - (Math.SQRT2 - 1) / 64)).toBeLessThanOrEqual(2e-6);
        // The triangle's bounds-center seed (2, 2) is on its hypotenuse, with radius zero.
        // Its analytic inradius is 4 - 2*sqrt(2), so the search must improve the seed.
        const triangleRadius = result['circles'][10];
        const triangleError = result['circles'][11];
        expect(triangleRadius).toBeGreaterThan(1);
        expect(triangleRadius).toBeLessThanOrEqual(4 - 2 * Math.SQRT2 + 1e-6);
        expect(triangleRadius + triangleError + 1e-6).toBeGreaterThanOrEqual(4 - 2 * Math.SQRT2);
      }
    }
    // Centers, radii and witnesses use 0.02 absolute units for the 0.01 search tolerance.
    // Error fields use 2e-6 absolute units: a few float32 spacings at this coordinate scale.
    // NaNs and status words must agree exactly.
    for (const column of ['circles', 'witnesses']) {
      expect(results[0][column].length).toBe(results[1][column].length);
      results[1][column].forEach((expected, index) => {
        if (Number.isNaN(expected)) expect(Number.isNaN(results[0][column][index])).toBe(true);
        else
          expect(Math.abs(results[0][column][index] - expected)).toBeLessThanOrEqual(
            column === 'circles' && index % 4 === 3 ? 2e-6 : 0.02
          );
      });
    }
  });
}
