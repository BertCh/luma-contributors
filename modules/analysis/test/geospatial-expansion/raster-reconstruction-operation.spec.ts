// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterReconstructionOperation} from '../../src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation';
import {
  GPURasterReconstructionOperation,
  type RasterReconstructionOperationProps
} from '../../src/raster-analysis/raster-morphology/gpu-raster-reconstruction-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import fixtures from './raster-morphology-operation-fixtures.json';

it('compares every reconstruction output exactly for both methods/connectivities and all-invalid input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.reconstruction) {
    for (const invalid of [false, true]) {
      const props: RasterReconstructionOperationProps = {
        grid: {
          width: fixture.width,
          height: fixture.height,
          affine: [1, 0, 0, 0, -1, fixture.height],
          pixelInterpretation: 'area'
        },
        marker: 'marker',
        mask: 'mask',
        method: fixture.method as 'dilation' | 'erosion',
        connectivity: fixture.connectivity as 4 | 8,
        maximumIterations: 64,
        output: {
          values: 'values',
          validity: 'validity',
          status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
        }
      };
      const marker = Float32Array.from(fixture.marker, value => (invalid ? NaN : value));
      const run = (
        operation: GPURasterReconstructionOperation | CPURasterReconstructionOperation
      ) =>
        runOperationGraph(device, {
          inputs: {
            marker: {format: 'float32', values: marker},
            mask: {format: 'float32', values: Float32Array.from(fixture.mask)}
          },
          operations: [operation],
          outputs: ['values', 'validity', 'converged', 'rounds', 'limit']
        });
      const cpu = await run(new CPURasterReconstructionOperation(props));
      const gpu = await run(new GPURasterReconstructionOperation(props));
      for (const port of ['validity', 'converged', 'rounds', 'limit'])
        expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
      // Min/max selects existing input values: numerical equality, with no sign-of-zero check.
      for (let cell = 0; cell < marker.length; cell++) {
        if (Number.isNaN(cpu['values'][cell])) expect(gpu['values'][cell]).toBeNaN();
        else expect(gpu['values'][cell] === cpu['values'][cell]).toBe(true);
      }
    }
  }
});

it('compares every singleton output while ignoring surplus marker and mask rows independently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RasterReconstructionOperationProps = {
    grid: {width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'},
    marker: 'marker',
    mask: 'mask',
    output: {
      values: 'values',
      validity: 'validity',
      status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
    }
  };
  for (const marker of [Float32Array.of(2), Float32Array.of(2, NaN)]) {
    for (const mask of [Float32Array.of(3), Float32Array.of(3, NaN)]) {
      // The NaN tails are outside the grid and are not read by the contributor.
      const run = (
        operation: GPURasterReconstructionOperation | CPURasterReconstructionOperation
      ) =>
        runOperationGraph(device, {
          inputs: {
            marker: {format: 'float32', values: marker},
            mask: {format: 'float32', values: mask}
          },
          operations: [operation],
          outputs: ['values', 'validity', 'converged', 'rounds', 'limit']
        });
      const cpu = await run(new CPURasterReconstructionOperation(props));
      const gpu = await run(new GPURasterReconstructionOperation(props));
      for (const [port, expected] of Object.entries({
        values: 2,
        validity: 1,
        converged: 1,
        rounds: 1,
        limit: 0
      })) {
        expect(Array.from(cpu[port])).toEqual([expected]);
        expect(Array.from(gpu[port])).toEqual([expected]);
      }
    }
  }
});
