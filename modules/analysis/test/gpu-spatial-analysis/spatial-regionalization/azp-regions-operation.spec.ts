// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {GPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRegionalizationBudgetPath} from './regionalization-final1-fixtures';
import {
  createAZPInput,
  createAZPCycleInput,
  createAZPExtensiveRoundingInput,
  createAZPProps,
  getRegionalizationOperationInputs
} from './regionalization-operation-test-utils';

it('spatial-3:F7 bounds both AZP operations to one boundary-move round', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    ...createAZPProps(),
    rowCount: 6,
    columnCount: 1,
    regionCapacity: 2,
    maximumIterations: 1,
    extensiveValues: undefined,
    extensiveThreshold: undefined
  };
  for (const Operation of [GPUAZPRegionsOperation, CPUAZPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(createRegionalizationBudgetPath()),
      operations: [new Operation(props)],
      outputs: Object.values(props.output)
    });
    expect(Array.from(result['labels'])).toEqual([0, 0, 0, 0, 1, 1]);
    expect(Array.from(result['status'])).toEqual([1, 0, 1, 0]);
    expect(result['objective'][0]).toBe(50);
  }
});

for (const scene of [
  'batched-moves',
  'standardized',
  'extensive-floor',
  'size-floor',
  'disconnected',
  'invalid'
] as const)
  it(`AZP CPU/GPU compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = {
      ...createAZPProps(),
      standardize: scene === 'standardized',
      minimumRegionSize: scene === 'size-floor' ? 2 : 1
    };
    const input = createAZPInput();
    if (scene === 'extensive-floor') input.extensiveThreshold![0] = 8;
    if (scene === 'disconnected') input.initialLabels = Uint32Array.of(0, 1, 0, 1, 2, 3, 2, 3);
    if (scene === 'invalid') input.extensiveValues![0] = -1;
    const inputs = getRegionalizationOperationInputs(input);
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUAZPRegionsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUAZPRegionsOperation(props)],
      outputs
    });
    // Selected rows and every status word are exact; === deliberately ignores the sign of zero.
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
    if (scene === 'batched-moves') {
      expect(Array.from(cpu['labels'])).toEqual([0, 1, 1, 1, 2, 3, 3, 3]);
      expect(Array.from(cpu['status'])).toEqual([1, 1, 0, 0]);
    }
  });

it('AZP compares exact conflicting gain ties under two seeds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    ...createAZPProps(),
    rowCount: 4,
    columnCount: 1,
    regionCapacity: 3,
    extensiveValues: undefined,
    extensiveThreshold: undefined
  };
  for (const seed of [0, 1]) {
    const input = {
      ...createAZPInput(),
      offsets: Uint32Array.of(0, 0, 1, 2, 3),
      neighbors: Uint32Array.of(0, 0, 1),
      weights: Float32Array.of(2, 3, 5),
      values: Float32Array.of(0, 10, 0, 10),
      initialLabels: Uint32Array.of(0, 0, 1, 2),
      extensiveValues: undefined,
      extensiveThreshold: undefined,
      parameters: Uint32Array.of(seed)
    };
    const inputs = getRegionalizationOperationInputs(input);
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUAZPRegionsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUAZPRegionsOperation(props)],
      outputs
    });
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      // All outputs here, including the zero SSD, are exact. === ignores sign of zero.
      for (let index = 0; index < cpu[output].length; index++)
        expect(gpu[output][index] === cpu[output][index]).toBe(true);
    }
    expect(Array.from(cpu['labels'])).toEqual(seed === 0 ? [0, 2, 1, 2] : [1, 0, 1, 2]);
  }
});

it('AZP rejects the six-cycle removal outside its induced two-hop neighborhood', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createAZPCycleInput();
  const props = {
    ...createAZPProps(),
    rowCount: 7,
    columnCount: 1,
    regionCapacity: 2,
    extensiveValues: undefined,
    extensiveThreshold: undefined
  };
  const inputs = getRegionalizationOperationInputs(input);
  const outputs = Object.values(props.output);
  for (const Operation of [GPUAZPRegionsOperation, CPUAZPRegionsOperation]) {
    const result = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs
    });
    // Device hypothesis from isRemovalSafe: row 3 is outside {1,2,4,5}, leaving two pieces.
    expect(Array.from(result['labels'])).toEqual([0, 0, 0, 0, 0, 0, 1]);
    expect(Array.from(result['status'])).toEqual([0, 1, 0, 0]);
    expect(result['objective'].length).toBe(1);
    // Direct pairwise SSD is 5 * 10^2 / 6. Allow only 1e-4 relative objective error.
    expect(Math.abs(result['objective'][0] - 250 / 3) / (250 / 3)).toBeLessThan(1e-4);
  }
});

it('AZP documents and checks the float64 extensive-remainder difference', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = createAZPExtensiveRoundingInput();
  const props = {...createAZPProps(), rowCount: 3, columnCount: 1, regionCapacity: 2};
  const inputs = getRegionalizationOperationInputs(input);
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUAZPRegionsOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUAZPRegionsOperation(props)],
    outputs
  });
  // Verify the GPU hypothesis separately: float32 stores 1 + 2^24 as 2^24, so the remainder
  // is zero. The CPU retains one and permits row 1 to leave. This is an intentional divergence.
  expect(Array.from(gpu['labels'])).toEqual([0, 0, 1]);
  expect(Array.from(gpu['status'])).toEqual([0, 1, 0, 0]);
  expect(gpu['objective'].length).toBe(1);
  expect(Math.abs(gpu['objective'][0] - 50) / 50).toBeLessThan(1e-4);
  expect(Array.from(cpu['labels'])).toEqual([0, 1, 1]);
  expect(Array.from(cpu['status'])).toEqual([1, 1, 0, 0]);
  expect(cpu['objective'].length).toBe(1);
  expect(cpu['objective'][0] === 0).toBe(true);
});
