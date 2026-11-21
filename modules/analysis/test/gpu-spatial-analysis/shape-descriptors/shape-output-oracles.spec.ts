// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUShapeDescriptorsOperation} from '../../../src/gpu-spatial-analysis/shape-descriptors/gpu-shape-descriptors-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runPoisonedOperation} from '../point-process/output-oracle-harness';
import {SHAPE_INPUTS, expectShapeOracle, getShapeProps} from './shape-output-oracles-fixture';
import {SHAPE_OUTPUT_CASES} from './shape-output-cases';

it('GPU shape writes inherited getter columns against an explicit request', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const output = Object.assign(
    Object.create({
      get areas() {
        return 'areas';
      }
    }),
    {perimeters: 'perimeters'}
  );
  expectShapeOracle(
    await runPoisonedOperation(
      device,
      new GPUShapeDescriptorsOperation(getShapeProps(output)),
      SHAPE_INPUTS,
      ['areas', 'perimeters']
    ),
    ['areas', 'perimeters']
  );
});

it('GPU shape operation matches closed forms with each poisoned optional column alone and omitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const output of SHAPE_OUTPUT_CASES)
    expectShapeOracle(
      await runPoisonedOperation(
        device,
        new GPUShapeDescriptorsOperation(getShapeProps(output)),
        SHAPE_INPUTS,
        Object.values(output)
      ),
      Object.values(output)
    );
});
