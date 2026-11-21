// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createCopyOperation} from './contributor-output-oracles-fixture';

it('generic lowering copies distinct literal values into every named output including the tail', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      leftInput: {format: 'uint32', values: new Uint32Array([7, 19, 31])},
      rightInput: {format: 'uint32', values: new Uint32Array([101, 203, 409])},
      poisonInput: {format: 'uint32', values: new Uint32Array([999, 999, 999])}
    },
    operations: [createCopyOperation()],
    outputs: ['firstResult', 'secondResult']
  });
  // Identity copies, worked by hand: port order and the last row must survive lowering.
  expect(Array.from(rows['firstResult'])).toEqual([7, 19, 31]);
  expect(Array.from(rows['secondResult'])).toEqual([101, 203, 409]);
});
