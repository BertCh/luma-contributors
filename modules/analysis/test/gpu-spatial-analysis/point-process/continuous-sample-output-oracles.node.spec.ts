// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {GPUContinuousSpatialSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation';
import {GPUSpatialBalancedSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample-operation';
import {CPUSpatialBalancedSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-spatial-balanced-sample-operation';
import {GPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes-operation';
import {CPUSpaceTimeHawkesOperation} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {GPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {CPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {
  constructSamplingContributor,
  createSamplingCPUInputs
} from './validation-sampling-operation-test-utils';
import {BALANCED_PROPS, getBalancedInputs} from './balanced-sample-output-oracles-fixture';
import {HAWKES_INPUTS, getHawkesProps} from '../space-time-hawkes/hawkes-output-oracles-fixture';
import {getMantelInputs, getMantelProps} from '../space-time-tests/mantel-output-oracles-fixture';
import {CPUContinuousSpatialSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CONTINUOUS_INPUTS,
  CONTINUOUS_PROPS,
  expectContinuousOracle
} from './continuous-sample-output-oracles-fixture';

it('CPU continuous sampling publishes the independent Philox line oracle including unused tails', async () => {
  const device = createNullWebGPUDevice();
  try {
    expectContinuousOracle(
      await runOperationGraph(device, {
        inputs: CONTINUOUS_INPUTS,
        operations: [new CPUContinuousSpatialSampleOperation(CONTINUOUS_PROPS)],
        outputs: Object.values(CONTINUOUS_PROPS.output)
      })
    );
  } finally {
    device.destroy();
  }
});

const mismatchedCases = [
  {
    name: 'continuous frameRowCount',
    props: {...CONTINUOUS_PROPS, frameRowCount: 3},
    inputs: CONTINUOUS_INPUTS,
    GPUOperation: GPUContinuousSpatialSampleOperation,
    CPUOperation: CPUContinuousSpatialSampleOperation
  },
  {
    name: 'continuous legacySiteCount',
    props: {
      ...CONTINUOUS_PROPS,
      legacySiteCount: 0,
      frame: {type: 'line' as const, positions: 'positions', legacySites: 'legacy'}
    },
    inputs: {
      ...CONTINUOUS_INPUTS,
      legacy: {format: 'float32' as const, values: new Float32Array([1])}
    },
    GPUOperation: GPUContinuousSpatialSampleOperation,
    CPUOperation: CPUContinuousSpatialSampleOperation
  },
  {
    name: 'balanced rowCount',
    props: {...BALANCED_PROPS, rowCount: 3},
    inputs: getBalancedInputs(),
    GPUOperation: GPUSpatialBalancedSampleOperation,
    CPUOperation: CPUSpatialBalancedSampleOperation
  },
  {
    name: 'Hawkes rowCount',
    props: {...getHawkesProps({}), rowCount: 3},
    inputs: {
      ...HAWKES_INPUTS,
      positions: {format: 'float32x2' as const, values: new Float32Array([0, 0, 2, 0])},
      times: {format: 'float32' as const, values: new Float32Array([0, 1])}
    },
    GPUOperation: GPUSpaceTimeHawkesOperation,
    CPUOperation: CPUSpaceTimeHawkesOperation
  },
  {
    name: 'Mantel rowCount',
    props: getMantelProps(false),
    inputs: {
      ...getMantelInputs(false),
      offsets: {format: 'uint32' as const, values: new Uint32Array([0, 1, 1])},
      neighbors: {format: 'uint32' as const, values: new Uint32Array([1])},
      weights: {format: 'float32' as const, values: new Float32Array([1])},
      distances: {format: 'float32' as const, values: new Float32Array([1])},
      times: {format: 'float32' as const, values: new Float32Array([0, 1])}
    },
    GPUOperation: GPUMantelTestOperation,
    CPUOperation: CPUMantelTestOperation
  }
];

for (const item of mismatchedCases)
  it(`CPU and GPU factories refuse the same declared ${item.name} mismatch`, async () => {
    // Each union member's constructor is paired with its own props above.
    const gpu = new item.GPUOperation(item.props as never);
    const cpu = new item.CPUOperation(item.props as never);
    const namedInputs = item.inputs as Record<
      string,
      {format: 'float32' | 'float32x2' | 'uint32'; values: Float32Array | Uint32Array}
    >;
    const inputs = Object.fromEntries(
      Object.entries(gpu.inputs).map(([port, name]) => [port, namedInputs[name]])
    );
    await expect(
      Promise.resolve().then(() => cpu.execute({inputs: createSamplingCPUInputs(inputs)}))
    ).rejects.toThrow(/length|rowCount|legacySiteCount/);
    expect(() =>
      constructSamplingContributor(
        gpu,
        Object.fromEntries(
          Object.entries(inputs).map(([port, input]) => [
            port,
            {
              format: input.format,
              length: input.values.length / getGPUVectorFormatInfo(input.format).components
            }
          ])
        )
      )
    ).toThrow(/frameRowCount|legacySiteCount|rowCount/);
  });

for (const item of [
  {...mismatchedCases[0], props: {...CONTINUOUS_PROPS, frame: {...CONTINUOUS_PROPS.frame}}},
  {...mismatchedCases[2], props: {...BALANCED_PROPS}},
  {...mismatchedCases[3], props: {...getHawkesProps({}), rowCount: 2}}
])
  it(`same CPU/GPU instance keeps its captured ${item.name} after caller count reuse`, async () => {
    const gpu = new item.GPUOperation(item.props as never);
    const cpu = new item.CPUOperation(item.props as never);
    Object.assign(item.props, 'frameRowCount' in item.props ? {frameRowCount: 3} : {rowCount: 3});
    const values = item.inputs as Record<
      string,
      {format: 'float32' | 'float32x2' | 'uint32'; values: Float32Array | Uint32Array}
    >;
    const inputs = Object.fromEntries(
      Object.entries(gpu.inputs).map(([port, name]) => [port, values[name]])
    );
    await expect(
      Promise.resolve().then(() => cpu.execute({inputs: createSamplingCPUInputs(inputs)}))
    ).resolves.toBeDefined();
    expect(() =>
      constructSamplingContributor(
        gpu,
        Object.fromEntries(
          Object.entries(inputs).map(([port, input]) => [
            port,
            {
              format: input.format,
              length: input.values.length / getGPUVectorFormatInfo(input.format).components
            }
          ])
        )
      )
    ).not.toThrow();
  });

it('continuous instances retain legacy port presence when the caller deletes its name', async () => {
  const props = {
    ...CONTINUOUS_PROPS,
    legacySiteCount: 0,
    frame: {
      type: 'line' as const,
      positions: 'positions',
      legacySites: 'legacy' as string | undefined
    }
  };
  const gpu = new GPUContinuousSpatialSampleOperation(props);
  const cpu = new CPUContinuousSpatialSampleOperation(props);
  delete props.frame.legacySites;
  const inputs = createSamplingCPUInputs({
    positions: CONTINUOUS_INPUTS.positions,
    legacySites: {format: 'float32', values: new Float32Array([1])}
  });
  await expect(Promise.resolve().then(() => cpu.execute({inputs}))).rejects.toThrow(
    /legacySiteCount/
  );
  expect(() =>
    constructSamplingContributor(gpu, {
      positions: {format: 'float32x2', length: 2},
      legacySites: {format: 'float32', length: 1}
    })
  ).toThrow(/legacySiteCount/);
});
