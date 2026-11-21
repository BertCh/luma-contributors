// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUShapeDescriptorsOperation} from '../../../src/gpu-spatial-analysis/shape-descriptors/cpu-shape-descriptors-operation';
import {GPUShapeDescriptorsOperation} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {SHAPE_INPUTS, expectShapeOracle, getShapeProps} from './shape-output-oracles-fixture';
import {SHAPE_OUTPUT_CASES} from './shape-output-cases';

it('shape captures inherited getter columns with an explicit requested list', async () => {
  const output = Object.assign(
    Object.create({
      get areas() {
        return 'areas';
      }
    }),
    {perimeters: 'perimeters'}
  );
  const props = getShapeProps(output);
  expect(
    Object.values(new GPUShapeDescriptorsOperation(props).outputs)
      .map(value => value.value)
      .sort()
  ).toEqual(['areas', 'perimeters']);
  const device = createNullWebGPUDevice();
  try {
    expectShapeOracle(
      await runOperationGraph(device, {
        inputs: SHAPE_INPUTS,
        operations: [new CPUShapeDescriptorsOperation(props)],
        outputs: ['areas', 'perimeters']
      }),
      ['areas', 'perimeters']
    );
  } finally {
    device.destroy();
  }
});

it('CPU shape operation matches closed forms with each optional column alone and omitted', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const output of SHAPE_OUTPUT_CASES)
      expectShapeOracle(
        await runOperationGraph(device, {
          inputs: SHAPE_INPUTS,
          operations: [new CPUShapeDescriptorsOperation(getShapeProps(output))],
          outputs: Object.values(output)
        }),
        Object.values(output)
      );
  } finally {
    device.destroy();
  }
});

it('shape oracle refuses a missing requested optional column', () => {
  expect(() => expectShapeOracle({}, ['areas'])).toThrow();
  expect(() => expectShapeOracle({areas: new Float32Array([8, 4])}, [])).toThrow();
});

it('poison harness rejects missing and unexpected operation declarations before allocating GPU buffers', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const requested of [['areas', 'clockwise'], []]) {
      await expect(
        runPoisonedOperation(
          device,
          new GPUShapeDescriptorsOperation(getShapeProps({areas: 'areas'})),
          {},
          requested
        )
      ).rejects.toThrow(/operation output declarations/);
    }
  } finally {
    device.destroy();
  }
});
