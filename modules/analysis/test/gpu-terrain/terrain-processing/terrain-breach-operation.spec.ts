// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainBreachOperation} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-breach-operation';
import {GPUTerrainBreachOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-breach-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeBreachInput, makeBreachProps} from './terrain-breach-operation-fixture';

it('compares all breach outputs on distinct pits, a tie, depth exhaustion, masks and invalid burning', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of [
    'ordinary',
    'infeasible',
    'masked',
    'burn',
    'two-dimensional',
    'padding',
    'one-step',
    'exhausted'
  ] as const) {
    const input = makeBreachInput(),
      props = makeBreachProps();
    if (scene === 'infeasible') props.maximumCutDepth = 1;
    if (scene === 'masked') {
      input.validity[2] = 0;
      input.pitCells[1] = 0xffffffff;
    }
    if (scene === 'burn') input.burnDepth[1] = -1;
    if (scene === 'two-dimensional') {
      props.grid = {...props.grid, width: 5, height: 3};
      input.elevation = Float32Array.of(0, 8, 9, 6, 0, 0, 4, 2, 5, 0, 0, 7, 9, 3, 0);
      input.pitCells = Uint32Array.of(7, 6);
      input.burnDepth = new Float32Array(15);
      input.validity = new Uint32Array(15).fill(1);
    }
    if (scene === 'one-step') props.maximumSteps = 1;
    if (scene === 'exhausted') input.validity.fill(0);
    if (scene === 'padding') {
      // 33 pits and 15 window cells leave both pit and entry dispatches partially occupied.
      props.pitCount = 33;
      props.grid = {...props.grid, width: 5, height: 3};
      input.elevation = Float32Array.of(0, 8, 9, 6, 0, 0, 4, 2, 5, 0, 0, 7, 9, 3, 0);
      input.pitCells = Uint32Array.from({length: 33}, (_, index) =>
        index % 3 === 0 ? 0xffffffff : 7
      );
      input.burnDepth = new Float32Array(15);
      input.validity = new Uint32Array(15).fill(1);
    }
    const results = [],
      outputs = Object.values(props.output);
    for (const Operation of [GPUTerrainBreachOperation, CPUTerrainBreachOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            elevation: {format: 'float32', values: input.elevation},
            pitCells: {format: 'uint32', values: input.pitCells},
            validity: {format: 'uint32', values: input.validity},
            burnDepth: {format: 'float32', values: input.burnDepth}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    for (const port of outputs) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++) {
        const actual = results[0][port][index],
          expected = results[1][port][index];
        if (Number.isNaN(expected)) expect(actual).toBeNaN();
        else if ((port !== 'edited' && port !== 'cutDepth') || expected === 0)
          expect(actual === expected).toBe(true);
        // Numeric results have relative 2e-5 tolerance; every integer, including paths, is exact.
        else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * 2e-5);
      }
    }
  }
});
