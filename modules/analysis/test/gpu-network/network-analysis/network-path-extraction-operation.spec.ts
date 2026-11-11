// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNetworkPathExtractionOperation} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {
  GPUNetworkPathExtractionOperation,
  type NetworkPathExtractionOperationProps
} from '../../../src/gpu-network/network-analysis/gpu-network-path-extraction-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const fixture of [
  'ordinary',
  'overflow',
  'incomplete',
  'inactive',
  'broken',
  'empty'
] as const) {
  it(`path twins compare all compact, edge and per-target outputs: ${fixture}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const empty = fixture === 'empty';
    const props: NetworkPathExtractionOperationProps = {
      nodeCount: 5,
      targetCapacity: empty ? 0 : 5,
      predecessors: 'predecessors',
      costs: 'costs',
      targets: 'targets',
      nodeIds: 'nodeIds',
      targetCount: 'active',
      predecessorsComplete: 'complete',
      output: {
        capacity: fixture === 'overflow' ? 4 : 6,
        ids: 'ids',
        lineOffsets: 'lines',
        pathCosts: empty ? undefined : 'pathCosts',
        pathFound: empty ? undefined : 'found',
        status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
      },
      edges: {
        edgeCount: 4,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        edgeIds: 'edgeIds',
        output: {
          capacity: fixture === 'overflow' ? 2 : 3,
          ids: 'edgeOutput',
          lineOffsets: 'edgeLines',
          status: {count: 'edgeCount', requiredCount: 'edgeRequired', overflow: 'edgeOverflow'}
        }
      }
    };
    const inputs: Record<string, OperationGraphInput> = {
      predecessors: {
        format: 'uint32',
        values: new Uint32Array(
          fixture === 'broken' ? [99, 0, 1, 0, 0xffffffff] : [0xffffffff, 0, 1, 0, 0xffffffff]
        )
      },
      costs: {format: 'float32', values: new Float32Array([0, 2, 5, 7, Infinity])},
      targets: {format: 'uint32', values: new Uint32Array(empty ? [] : [2, 3, 0, 4, 99])},
      nodeIds: {format: 'uint32', values: new Uint32Array([10, 11, 12, 13, 14])},
      active: {format: 'uint32', values: new Uint32Array([fixture === 'inactive' ? 0 : 5])},
      complete: {format: 'uint32', values: new Uint32Array([fixture === 'incomplete' ? 0 : 1])},
      offsets: {format: 'uint32', values: new Uint32Array([0, 3, 4, 4, 4, 4])},
      neighbors: {format: 'uint32', values: new Uint32Array([1, 1, 3, 2])},
      weights: {format: 'float32', values: new Float32Array([9, 2, 7, 3])},
      edgeIds: {format: 'uint32', values: new Uint32Array([90, 20, 70, 30])}
    };
    const outputs = [
      'ids',
      'lines',
      'count',
      'required',
      'overflow',
      'edgeOutput',
      'edgeLines',
      'edgeCount',
      'edgeRequired',
      'edgeOverflow',
      ...(empty ? [] : ['pathCosts', 'found'])
    ];
    const results = [];
    for (const Operation of [GPUNetworkPathExtractionOperation, CPUNetworkPathExtractionOperation])
      results.push(
        await runOperationGraph(device, {inputs, operations: [new Operation(props)], outputs})
      );
    // IDs, flags, counts, offsets, input path costs and Infinity are exact: tolerance zero.
    // Fresh graph buffers have zero node padding. Production callers consume only count rows.
    for (const value of outputs) expect(results[0][value]).toEqual(results[1][value]);
  });
}
