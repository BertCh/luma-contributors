// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {GPU_MAX_P_STATUS_LAYOUT} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createMaxPEnclaveBudgetPath} from './regionalization-final1-fixtures';
import {
  createMaxPInput,
  createMaxPGrowthInput,
  createMaxPProps,
  createMaxPCycleInput,
  createMaxPRefinementInput,
  getRegionalizationOperationInputs
} from './regionalization-operation-test-utils';

it('spatial-3:F7 limits both Max-p operations to one growth and one enclave step', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    ...createMaxPProps(),
    rowCount: 8,
    startCount: 1,
    maximumGrowthSteps: 1,
    maximumIterations: 0
  };
  for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(createMaxPGrowthInput()),
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    // No row is eligible in the first seed wave; every unreached row becomes an infeasible singleton.
    expect(Array.from(result['labels'])).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(Array.from(result['status'])).toEqual([8, 0, 8, 0, 0, 0]);
    expect(result['objective'][0]).toBe(0);
  }
});

it('spatial-3:F7 bounds enclave propagation separately from seed growth on both operations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    ...createMaxPProps(),
    rowCount: 4,
    columnCount: 1,
    startCount: 1,
    maximumGrowthSteps: 1,
    maximumIterations: 0
  };
  for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(createMaxPEnclaveBudgetPath()),
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    // Only row 0 has enough extensive mass; one synchronous enclave step reaches row 1, not row 2.
    expect(result['labels'][1]).toBe(result['labels'][0]);
    expect(result['labels'][2]).not.toBe(result['labels'][0]);
    expect(result['status'][GPU_MAX_P_STATUS_LAYOUT.feasible]).toBe(0);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
  }
});

for (const maximumIterations of [1, 2]) {
  it(`limits Max-p seed-four refinement to ${maximumIterations} rounds on both operations`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createMaxPRefinementInput(4);
    const props = {
      ...createMaxPProps(),
      rowCount: 9,
      columnCount: 1,
      startCount: 1,
      standardize: false,
      maximumIterations
    };
    for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: getRegionalizationOperationInputs(input),
        operations: [new Operation(props)],
        outputs: Object.values(props.output)
      });
      expect(result['status'][GPU_MAX_P_STATUS_LAYOUT.improvementIterationCount]).toBe(
        maximumIterations
      );
      expect(Array.from(result['labels'])).toEqual(
        maximumIterations === 1 ? [1, 0, 0, 1, 1, 2, 1, 2, 2] : [0, 0, 0, 1, 1, 2, 1, 2, 2]
      );
    }
  });
}

for (const scene of [
  'growth',
  'standardized',
  'improvement',
  'infeasible',
  'zero-threshold',
  'invalid'
] as const)
  it(`Max-P CPU/GPU compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = {
      ...createMaxPProps(),
      rowCount: 8,
      startCount: 8,
      standardize: scene === 'standardized',
      maximumIterations: scene === 'improvement' ? 8 : 0
    };
    const input =
      scene === 'improvement'
        ? createMaxPRefinementInput(2)
        : scene === 'growth' || scene === 'standardized'
          ? createMaxPGrowthInput()
          : createMaxPInput();
    props.rowCount = input.offsets.length - 1;
    if (scene === 'improvement') {
      props.columnCount = 1;
      props.startCount = 1;
    }
    if (scene === 'infeasible') input.threshold[0] = 5;
    if (scene === 'zero-threshold') input.threshold[0] = 0;
    if (scene === 'invalid') input.extensiveValues[0] = -1;
    const inputs = getRegionalizationOperationInputs(input);
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUMaxPRegionsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUMaxPRegionsOperation(props)],
      outputs
    });
    // Dense labels and every status word are exact; numerical === deliberately ignores zero sign.
    for (const output of ['labels', 'status']) {
      expect(gpu[output].length).toBe(cpu[output].length);
      for (let index = 0; index < cpu[output].length; index++)
        expect(gpu[output][index] === cpu[output][index], `${output}[${index}]`).toBe(true);
    }
    expect(gpu['objective'].length).toBe(1);
    expect(cpu['objective'].length).toBe(1);
    const expected = cpu['objective'][0];
    if (expected === 0) expect(gpu['objective'][0] === 0).toBe(true);
    else expect(Math.abs(gpu['objective'][0] - expected) / expected).toBeLessThan(1e-4);
    if (scene === 'growth') {
      expect(Array.from(cpu['labels'])).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
      expect(cpu['objective'][0]).toBe(150);
      expect(cpu['status'][0]).toBe(2);
    }
    if (scene === 'improvement') {
      for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
        // Fresh operations: runOperationGraph owns and destroys each instance.
        const original = await runOperationGraph(device, {
          inputs,
          operations: [new Operation({...props, maximumIterations: 0})],
          outputs
        });
        const refined = Operation === GPUMaxPRegionsOperation ? gpu : cpu;
        expect(Array.from(original['labels'])).toEqual([0, 1, 1, 0, 2, 2, 0, 2, 2]);
        expect(Array.from(original['status'])).toEqual([3, 1, 0, 0, 0, 0]);
        expect(original['objective'].length).toBe(1);
        expect(Math.abs(original['objective'][0] - 1141 / 6) / (1141 / 6)).toBeLessThan(1e-4);
        expect(Array.from(refined['labels'])).toEqual([0, 1, 1, 0, 1, 2, 0, 2, 2]);
        expect(Array.from(refined['status'])).toEqual([3, 1, 0, 0, 0, 1]);
        expect(refined['status'][5]).toBeGreaterThan(0);
        expect(refined['objective'][0]).toBeLessThan(original['objective'][0]);
        // Pairwise SSDs after row 4 moves: 266/3 + 74/3 + 182/3 = 174.
        expect(Math.abs(refined['objective'][0] - 174) / 174).toBeLessThan(1e-4);
      }
    }
  });

it('Max-P compares seeded equal-distance claim ties exactly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {...createMaxPProps(), columnCount: 1, startCount: 1};
  for (const seed of [0, 1]) {
    const input = {
      ...createMaxPInput(),
      offsets: Uint32Array.of(0, 0, 1, 3, 6),
      neighbors: Uint32Array.of(0, 0, 1, 0, 1, 2),
      weights: Float32Array.of(1, 2, 3, 4, 5, 6),
      values: new Float32Array(4),
      extensiveValues: new Float32Array(4).fill(1),
      threshold: Float32Array.of(2),
      parameters: Uint32Array.of(seed)
    };
    const inputs = getRegionalizationOperationInputs(input);
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUMaxPRegionsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUMaxPRegionsOperation(props)],
      outputs
    });
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      // Integer partitions/status and the zero objective are exact; === ignores zero sign.
      for (let index = 0; index < cpu[output].length; index++)
        expect(gpu[output][index] === cpu[output][index]).toBe(true);
    }
    expect(Array.from(cpu['status'])).toEqual([2, 1, 0, 0, 0, 0]);
  }
});

it('Max-P refinement rejects the same six-cycle move as standalone AZP', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createMaxPCycleInput();
  const props = {
    ...createMaxPProps(),
    rowCount: 7,
    columnCount: 1,
    startCount: 1,
    maximumIterations: 8
  };
  const inputs = getRegionalizationOperationInputs(input);
  const outputs = Object.values(props.output);
  for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs
    });
    // Seed 147 yields the cycle and singleton. Two-hop admissibility must prevent row 0 leaving.
    expect(Array.from(result['labels'])).toEqual([0, 0, 0, 0, 0, 0, 1]);
    expect(Array.from(result['status'])).toEqual([2, 1, 0, 0, 0, 0]);
    expect(result['objective'].length).toBe(1);
    expect(Math.abs(result['objective'][0] - 250 / 3) / (250 / 3)).toBeLessThan(1e-4);
  }
});
