// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURasterFillHolesOperation} from '../../src/raster-analysis/raster-morphology/cpu-raster-fill-holes-operation';
import {
  GPURasterFillHolesOperation,
  type RasterFillHolesOperationProps
} from '../../src/raster-analysis/raster-morphology/gpu-raster-fill-holes-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import fixtures from './raster-morphology-operation-fixtures.json';

it('compares every hole-fill output exactly for holes, passages, diagonal gaps and empty foreground', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.fillHoles) {
    const props: RasterFillHolesOperationProps = {
      grid: {
        width: fixture.width,
        height: fixture.height,
        affine: [1, 0, 0, 0, -1, fixture.height],
        pixelInterpretation: 'area'
      },
      input: 'input',
      connectivity: fixture.connectivity as 4 | 8,
      maximumIterations: 64,
      output: {
        values: 'filled',
        status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
      }
    };
    const run = (operation: GPURasterFillHolesOperation | CPURasterFillHolesOperation) =>
      runOperationGraph(device, {
        inputs: {input: {format: 'uint32', values: Uint32Array.from(fixture.input)}},
        operations: [operation],
        outputs: ['filled', 'converged', 'rounds', 'limit']
      });
    const cpu = await run(new CPURasterFillHolesOperation(props));
    const gpu = await run(new GPURasterFillHolesOperation(props));
    // Integer copies, binary results and synchronous round counts require no tolerance.
    for (const port of ['filled', 'converged', 'rounds', 'limit'])
      expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
    expect(Array.from(cpu['filled'])).toEqual(fixture.expected);
  }
});

it('compares every output for a unit-grid background with a surplus foreground row', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: RasterFillHolesOperationProps = {
    grid: {width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'},
    input: 'input',
    output: {
      values: 'filled',
      status: {converged: 'converged', iterationCount: 'rounds', iterationLimitReached: 'limit'}
    }
  };
  const run = (operation: GPURasterFillHolesOperation | CPURasterFillHolesOperation) =>
    runOperationGraph(device, {
      inputs: {input: {format: 'uint32', values: Uint32Array.of(0, 7)}},
      operations: [operation],
      outputs: ['filled', 'converged', 'rounds', 'limit']
    });
  const cpu = await run(new CPURasterFillHolesOperation(props));
  const gpu = await run(new GPURasterFillHolesOperation(props));
  for (const [port, expected] of Object.entries({filled: 0, converged: 1, rounds: 1, limit: 0})) {
    expect(Array.from(cpu[port])).toEqual([expected]);
    expect(Array.from(gpu[port])).toEqual([expected]);
  }
});
