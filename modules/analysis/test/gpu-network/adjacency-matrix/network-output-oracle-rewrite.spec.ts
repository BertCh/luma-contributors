// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOracleContext} from './network-output-oracle-test-utils';

it('W5-03 replays fractional and negative coordinates alongside uint32 controls', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createOracleContext(device);
  const {graph, input, output, write, read} = context;
  const positions = input('positions', 'float32x2', [0, 0]);
  const identifiers = input('identifiers', 'uint32x2', [1, 2]);
  const copiedPositions = output('copied-positions', 'float32x2');
  const copiedIdentifiers = output('copied-identifiers', 'uint32x2');
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'copy-replay',
      operation: 'output-oracle-replay',
      invocationCount: 2,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'identifiers', view: identifiers, type: 'u32', access: 'read'},
        {name: 'copiedPositions', view: copiedPositions, type: 'f32', access: 'read_write'},
        {name: 'copiedIdentifiers', view: copiedIdentifiers, type: 'u32', access: 'read_write'}
      ],
      body: 'copiedPositions[copiedPositionsOffset + index] = positions[positionsOffset + index]; copiedIdentifiers[copiedIdentifiersOffset + index] = identifiers[identifiersOffset + index];'
    })
  );
  try {
    context.run();
    expect(await read(copiedPositions)).toEqual([0, 0]);
    expect(await read(copiedIdentifiers)).toEqual([1, 2]);
    write(positions, [0.25, -1.5]);
    write(identifiers, [7, 0xffffffff]);
    context.run();
    expect(await read(copiedPositions)).toEqual([0.25, -1.5]);
    expect(await read(copiedIdentifiers)).toEqual([7, 0xffffffff]);
  } finally {
    context.destroy();
  }
});
