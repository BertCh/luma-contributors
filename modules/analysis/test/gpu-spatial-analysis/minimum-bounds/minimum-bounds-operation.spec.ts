// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMinimumBoundsOperation} from '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation';
import {
  GPUMinimumBoundsOperation,
  type MinimumBoundsOperationProps
} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Eight float32 rounding units of the relevant geometric scale cover cancellation in the
// normalized projections and fused coordinate reconstruction. This absolute allowance applies
// only to computed coordinate zeros and mathematically zero segment height/area. Positive
// measures retain the original 2e-5 relative tolerance, with no absolute floor.
const CANCELLATION_RELATIVE_TOLERANCE = 8 * 2 ** -23;

for (const empty of [false, true]) {
  it(`compares every minimum-bounds column (${empty ? 'empty groups' : 'multiple shapes and degeneracies'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions = new Float32Array([
      0, 0, 4, 0, 0, 3, 1, 1, 10, 0, 12, 1, 11, 3, 9, 2, 20, 1, 23, 5, -2, 7
    ]);
    const labels = empty
      ? new Uint32Array(11).fill(0xffffffff)
      : new Uint32Array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3]);
    const props: MinimumBoundsOperationProps = {
      positions: 'positions',
      labels: 'labels',
      rowCount: 11,
      groupCount: 5,
      output: {
        rectangleCorners: 'corners',
        rectangleSizes: 'sizes',
        circles: 'circles',
        longestLines: 'lines',
        diameters: 'diameters',
        bounds: 'bounds',
        hullSizes: 'hullSizes',
        status: {overflow: 'overflow'}
      }
    };
    const outputs = [
      'corners',
      'sizes',
      'circles',
      'lines',
      'diameters',
      'bounds',
      'hullSizes',
      'overflow'
    ];
    const results = [];
    for (const Operation of [GPUMinimumBoundsOperation, CPUMinimumBoundsOperation]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: positions},
            labels: {format: 'uint32', values: labels}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    }
    for (const port of outputs) {
      const expected = results[1][port];
      const actual = results[0][port];
      expect(actual.length).toBe(expected.length);
      for (let index = 0; index < expected.length; index++) {
        const diagnostic = `${port}[${index}]: GPU=${actual[index]}, CPU=${expected[index]}`;
        if (Number.isNaN(expected[index]))
          expect(Number.isNaN(actual[index]), diagnostic).toBe(true);
        // Integer fields and copied endpoints/envelopes remain exact, including numeric zeros.
        else if (
          port === 'hullSizes' ||
          port === 'overflow' ||
          port === 'lines' ||
          port === 'bounds'
        )
          expect(actual[index] === expected[index], diagnostic).toBe(true);
        else {
          const components =
            port === 'corners' ? 8 : port === 'sizes' ? 4 : port === 'circles' ? 3 : 1;
          const group = Math.floor(index / components);
          const component = index % components;
          const hullSize = results[1]['hullSizes'][group];
          const diameter = results[1]['diameters'][group];
          const bounds = results[1]['bounds'].subarray(group * 4, group * 4 + 4);
          const coordinateScale = Math.max(diameter, ...Array.from(bounds, Math.abs));
          let absoluteTolerance = 0;
          if (
            hullSize > 1 &&
            expected[index] === 0 &&
            (port === 'corners' || (port === 'circles' && component < 2))
          ) {
            absoluteTolerance = CANCELLATION_RELATIVE_TOLERANCE * coordinateScale;
          } else if (hullSize === 2 && port === 'sizes' && (component === 1 || component === 3)) {
            absoluteTolerance =
              CANCELLATION_RELATIVE_TOLERANCE * (component === 3 ? diameter * diameter : diameter);
          }
          expect(Math.abs(actual[index] - expected[index]), diagnostic).toBeLessThanOrEqual(
            Math.max(absoluteTolerance, Math.abs(expected[index]) * 2e-5)
          );
        }
      }
    }
  });
}
