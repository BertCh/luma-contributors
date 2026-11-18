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
import {GPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/gpu-network-reachability-operation';
import {CPUNetworkReachabilityOperation} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';

for (const fixture of [
  'ordinary',
  'overflow',
  'incomplete',
  'inactive',
  'broken',
  'bounded',
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
      maximumPathLength: fixture === 'bounded' ? 1 : undefined,
      output: {
        capacity: fixture === 'overflow' ? 4 : 6,
        ids: 'ids',
        nodeRows: 'nodeRows',
        lineOffsets: 'lines',
        pathCosts: empty ? undefined : 'pathCosts',
        pathFound: empty ? undefined : 'found',
        status: {
          count: 'count',
          requiredCount: 'required',
          overflow: 'overflow',
          candidateOverflow: 'candidate'
        }
      },
      edges: {
        edgeCount: 4,
        topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        edgeIds: 'edgeIds',
        output: {
          capacity: fixture === 'overflow' ? 2 : 3,
          ids: 'edgeOutput',
          edgeRows: 'edgeRows',
          lineOffsets: 'edgeLines',
          status: {
            count: 'edgeCount',
            requiredCount: 'edgeRequired',
            overflow: 'edgeOverflow',
            candidateOverflow: 'edgeCandidate'
          }
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
      'nodeRows',
      'edgeRows',
      'lines',
      'count',
      'required',
      'overflow',
      'candidate',
      'edgeOutput',
      'edgeLines',
      'edgeCount',
      'edgeRequired',
      'edgeOverflow',
      'edgeCandidate',
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
    if (fixture === 'ordinary') {
      expect(results[0]['nodeRows']).toEqual(Uint32Array.of(0, 1, 2, 0, 3, 0));
      expect(results[0]['ids']).toEqual(Uint32Array.of(10, 11, 12, 10, 13, 10));
      expect(results[0]['edgeRows']).toEqual(Uint32Array.of(1, 3, 2));
      expect(results[0]['edgeOutput']).toEqual(Uint32Array.of(20, 30, 70));
      const positions = [
        [10, 20],
        [30, 40],
        [50, 60],
        [70, 80],
        [90, 100]
      ];
      expect(Array.from(results[0]['nodeRows'], row => positions[row])).toEqual([
        [10, 20],
        [30, 40],
        [50, 60],
        [10, 20],
        [70, 80],
        [10, 20]
      ]);
    }
    if (fixture === 'bounded') {
      expect(Array.from(results[0]['candidate'])).toEqual([1]);
      expect(Array.from(results[0]['edgeCandidate'])).toEqual([1]);
      expect(Array.from(results[0]['overflow'])).toEqual([0]);
      expect(Array.from(results[0]['edgeOverflow'])).toEqual([0]);
    }
  });
}

it('network:F5 bounded reachability clears the certificate and suppresses downstream paths on both backends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const activeIterations of [0, 1]) {
    for (const cpu of [false, true]) {
      const Search = cpu ? CPUNetworkReachabilityOperation : GPUNetworkReachabilityOperation;
      const Extract = cpu ? CPUNetworkPathExtractionOperation : GPUNetworkPathExtractionOperation;
      const rows = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2, 2)},
          neighbors: {format: 'uint32', values: Uint32Array.of(1, 2)},
          weights: {format: 'float32', values: Float32Array.of(1, 1)},
          starts: {format: 'uint32', values: Uint32Array.of(0)},
          active: {format: 'uint32', values: Uint32Array.of(activeIterations)},
          targets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)}
        },
        operations: [
          new Search({
            nodeCount: 3,
            topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
            startNodes: 'starts',
            activeIterations: 'active',
            maximumIterations: 2,
            localIterations: 1,
            output: {
              costs: 'costs',
              predecessors: 'predecessors',
              status: {converged: 'complete', iterationLimitReached: 'limited'}
            }
          }),
          new Extract({
            nodeCount: 3,
            targetCapacity: 3,
            predecessors: 'predecessors',
            costs: 'costs',
            targets: 'targets',
            predecessorsComplete: 'complete',
            output: {
              capacity: 9,
              ids: 'ids',
              pathFound: 'found',
              status: {count: 'count', overflow: 'overflow', requiredCount: 'required'}
            }
          })
        ],
        outputs: ['costs', 'complete', 'limited', 'count', 'found']
      });
      expect(rows['costs']).toEqual(Float32Array.of(0, activeIterations ? 1 : Infinity, Infinity));
      expect(rows['complete']).toEqual(Uint32Array.of(0));
      expect(rows['limited']).toEqual(Uint32Array.of(1));
      expect(rows['count']).toEqual(Uint32Array.of(0));
      expect(rows['found']).toEqual(Uint32Array.of(0, 0, 0));
    }
  }
});
