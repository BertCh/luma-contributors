// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  computeRasterWalkingCostOnCPU,
  CPURasterWalkingCostOperation
} from '../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {GPURasterWalkingCostOperation} from '../../src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation';
import {getGPURasterWalkingCostParameterValues} from '../../src/raster-analysis/gpu-raster-walking-cost';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {
  createWalkingInput,
  createWalkingProps,
  createWalkingGraphInputs,
  createSingletonWalkingProps,
  createSingletonWalkingInput
} from './raster-walking-cost-operation-fixture';
import fixtures from './raster-walking-cost-operation-fixtures.json';

it('compares all directed walking-cost outputs for both models, no sources, barriers and parameter mismatch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kind of ['tobler', 'grass'] as const) {
    for (const scenario of ['ordinary', 'no-sources', 'barrier', 'mismatch'] as const) {
      const props = createWalkingProps(kind);
      const input = createWalkingInput(props);
      if (scenario === 'no-sources') input.sources.fill(0);
      if (scenario === 'barrier') {
        input.elevation.validity![7] = 0;
        input.friction.values[8] = -1;
      }
      if (scenario === 'mismatch')
        input.parameters = getGPURasterWalkingCostParameterValues(
          {
            model:
              kind === 'grass'
                ? {kind: 'tobler'}
                : {kind: 'grass', coefficients: [2, 3, -0.5, -2], downhillSlopeThreshold: -0.25}
          },
          props.grid
        );
      const run = (operation: GPURasterWalkingCostOperation | CPURasterWalkingCostOperation) =>
        runOperationGraph(device, {
          inputs: createWalkingGraphInputs(input),
          operations: [operation],
          outputs: ['costs', 'validity', 'converged', 'rounds', 'limit', 'invalid']
        });
      const cpu = await run(new CPURasterWalkingCostOperation(props));
      const gpu = await run(new GPURasterWalkingCostOperation(props));
      // These scenes have distinct shortest paths, so the verifying round count also agrees.
      for (const port of ['validity', 'converged', 'rounds', 'limit', 'invalid'])
        expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
      for (let cell = 0; cell < input.sources.length; cell++) {
        const expected = cpu['costs'][cell];
        if (Number.isNaN(expected)) expect(gpu['costs'][cell]).toBeNaN();
        else if (!Number.isFinite(expected) || expected === 0)
          expect(gpu['costs'][cell] === expected).toBe(true);
        else
          expect(Math.abs(gpu['costs'][cell] - expected)).toBeLessThanOrEqual(
            3e-5 * Math.abs(expected)
          );
      }
      expect(cpu['costs']).toEqual(computeRasterWalkingCostOnCPU(input, props).costs);
    }
  }
});

it('compares all outputs with geographic and Mercator ground scales near 60 degrees north', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.groundCases) {
    const props = createWalkingProps('grass');
    props.grid = {
      width: 3,
      height: 1,
      affine: fixture.affine as [number, number, number, number, number, number],
      pixelInterpretation: 'area'
    };
    props.cellSizeMode = fixture.mode as 'geographic' | 'web-mercator';
    const input = {
      elevation: {
        id: 'elevation',
        format: 'float32' as const,
        values: new Float32Array(3),
        validity: new Uint32Array(3).fill(1)
      },
      friction: {
        id: 'friction',
        format: 'float32' as const,
        values: new Float32Array(3),
        validity: new Uint32Array(3).fill(1)
      },
      sources: Uint32Array.of(1, 0, 0),
      parameters: getGPURasterWalkingCostParameterValues(
        {model: props.model, frictionWeight: 0},
        props.grid
      )
    };
    const run = (operation: GPURasterWalkingCostOperation | CPURasterWalkingCostOperation) =>
      runOperationGraph(device, {
        inputs: createWalkingGraphInputs(input),
        operations: [operation],
        outputs: ['costs', 'validity', 'converged', 'rounds', 'limit', 'invalid']
      });
    const cpu = await run(new CPURasterWalkingCostOperation(props));
    const gpu = await run(new GPURasterWalkingCostOperation(props));
    for (const port of ['validity', 'converged', 'rounds', 'limit', 'invalid'])
      expect(Array.from(gpu[port])).toEqual(Array.from(cpu[port]));
    expect(gpu['costs'][0] === 0).toBe(true);
    for (let cell = 1; cell < 3; cell++)
      expect(Math.abs(gpu['costs'][cell] - cpu['costs'][cell])).toBeLessThanOrEqual(
        3e-5 * cpu['costs'][cell]
      );
  }
});

it('compares every unit-grid output with active and inactive source words followed by surplus rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = createSingletonWalkingProps();
  for (const sources of [Uint32Array.of(1, 0), Uint32Array.of(0, 1)]) {
    const input = createSingletonWalkingInput(sources);
    const run = (operation: GPURasterWalkingCostOperation | CPURasterWalkingCostOperation) =>
      runOperationGraph(device, {
        inputs: createWalkingGraphInputs(input),
        operations: [operation],
        outputs: ['costs', 'validity', 'converged', 'rounds', 'limit', 'invalid']
      });
    const cpu = await run(new CPURasterWalkingCostOperation(props));
    const gpu = await run(new GPURasterWalkingCostOperation(props));
    for (const result of [cpu, gpu]) {
      expect(result['costs'].length).toBe(1);
      expect(result['costs'][0] === (sources[0] !== 0 ? 0 : Infinity)).toBe(true);
      for (const [port, expected] of Object.entries({
        validity: 1,
        converged: 1,
        rounds: 1,
        limit: 0,
        invalid: 0
      }))
        expect(Array.from(result[port])).toEqual([expected]);
    }
  }
});
