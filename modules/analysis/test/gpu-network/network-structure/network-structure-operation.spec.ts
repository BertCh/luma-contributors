// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkStructureOperation} from '../../../src/gpu-network/network-structure/cpu-network-structure-operation';
import {
  GPUNetworkStructureOperation,
  type NetworkStructureOperationProps
} from '../../../src/gpu-network/network-structure/gpu-network-structure-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('SCC-only structure twins ignore an unused empty reverse table', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: NetworkStructureOperationProps = {
    nodeCount: 2,
    edgeCount: 1,
    topology: {offsets: 'offsets', neighbors: 'neighbors'},
    reverseEdgeSlots: 'reverse',
    output: {strongComponents: 'components'}
  };
  const inputs = {
    offsets: {format: 'uint32' as const, values: new Uint32Array([0, 1, 1])},
    neighbors: {format: 'uint32' as const, values: new Uint32Array([1])},
    reverse: {format: 'uint32' as const, values: new Uint32Array(0)}
  };
  const results = [];
  for (const Operation of [GPUNetworkStructureOperation, CPUNetworkStructureOperation]) {
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [new Operation(props)],
      outputs: ['components']
    });
    results.push(rows['components']);
    // One edge 0 -> 1 gives two singleton SCCs. Integer labels compare with zero tolerance.
    expect(rows['components']).toEqual(new Uint32Array([0, 1]));
  }
  expect(results[0]).toEqual(results[1]);
});

for (const fixture of ['blocks', 'invalid-slot', 'isolated'] as const) {
  it(`structure twins compare every integer output exactly: ${fixture}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const offsets = new Uint32Array(fixture === 'blocks' ? [0, 2, 4, 7, 10, 13, 13] : [0, 1, 1]);
    const neighbors = new Uint32Array(
      fixture === 'blocks' ? [1, 2, 0, 2, 1, 0, 3, 2, 4, 4, 3, 3, 4] : [99]
    );
    const reverse = new Uint32Array(
      fixture === 'blocks' ? [2, 5, 0, 4, 3, 1, 7, 6, 10, 11, 8, 9, 12] : [0xffffffff]
    );
    const props: NetworkStructureOperationProps = {
      nodeCount: offsets.length - 1,
      edgeCount: neighbors.length,
      topology: {offsets: 'offsets', neighbors: 'neighbors'},
      reverseEdgeSlots: 'reverse',
      output: {
        strongComponents: 'components',
        bridges: 'bridges',
        articulationPoints: 'articulation',
        biconnectedComponents: 'blocks'
      }
    };
    if (fixture === 'isolated') {
      props.edgeCount = 0;
      props.output = {strongComponents: 'components', articulationPoints: 'articulation'};
    }
    const inputs = {
      offsets: {
        format: 'uint32' as const,
        values: fixture === 'isolated' ? new Uint32Array([0, 0, 0]) : offsets
      },
      neighbors: {
        format: 'uint32' as const,
        values: fixture === 'isolated' ? new Uint32Array(0) : neighbors
      },
      reverse: {
        format: 'uint32' as const,
        values: fixture === 'isolated' ? new Uint32Array(0) : reverse
      }
    };
    const outputs = Object.values(props.output) as string[];
    const results = [];
    for (const Operation of [GPUNetworkStructureOperation, CPUNetworkStructureOperation])
      results.push(
        await runOperationGraph(device, {inputs, operations: [new Operation(props)], outputs})
      );
    // All outputs are integer identifiers or flags; tolerance is zero.
    for (const value of outputs) expect(results[0][value]).toEqual(results[1][value]);
  });
}
