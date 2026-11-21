// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeNetworkReachabilityOnCPU,
  CPUNetworkReachabilityOperation
} from '../../../src/gpu-network/network-reachability/cpu-network-reachability-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('reachability CPU twin counts the single directed edge once', () => {
  const result = computeNetworkReachabilityOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 1, 1),
        neighbors: Uint32Array.of(1),
        weights: Float32Array.of(3)
      },
      startNodes: Uint32Array.of(0)
    },
    {maximumIterations: 4, localIterations: 1}
  );
  expect(Array.from(result.costs)).toEqual([0, 3]);
  expect(Array.from(result.relaxationCount)).toEqual([1, 0]);
});

it('reachability CPU twin has the independently enumerated unit-chain bands and counts', () => {
  const result = computeNetworkReachabilityOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 2, 3, 3),
        neighbors: Uint32Array.of(1, 99, 2),
        weights: Float32Array.of(1, 1, 1)
      },
      startNodes: Uint32Array.of(0),
      costBreaks: Float32Array.of(1, 2)
    },
    {maximumIterations: 4, localIterations: 1}
  );
  expect(Array.from(result.costs)).toEqual([0, 1, 2]);
  expect(Array.from(result.bands)).toEqual([0, 0, 1]);
  expect(Array.from(result.bandCounts)).toEqual([2, 1]);
  // Scans 0->1, 0->99 and 1->2; the invalid target counts as work, not a reached edge.
  expect(Array.from(result.relaxationCount)).toEqual([3, 0]);
  expect(Array.from(result.reachedEdgeCount)).toEqual([2, 0]);
});

it('reachability CPU operation publishes each optional band and counter against independent literals', async () => {
  const device = createNullWebGPUDevice();
  try {
    // Directed unit chain 0->1->2 has distances [0,1,2], bands [0,0,1] for
    // inclusive breaks [1,2] and populations [2,1]. The extra 0->99 slot makes
    // three visits but only two reached edges. An out-of-range start gives no visits.
    for (const start of [0, 99])
      for (const selected of [
        undefined,
        'relaxationCount',
        'bands',
        'bandCounts',
        'reachedEdgeCount'
      ] as const) {
        const withBreaks = selected === 'bands' || selected === 'bandCounts';
        const operation = new CPUNetworkReachabilityOperation({
          nodeCount: 3,
          topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          startNodes: 'starts',
          maximumIterations: 4,
          localIterations: 1,
          costBreaks: withBreaks ? 'breaks' : undefined,
          costBreakCount: withBreaks ? 2 : undefined,
          output: {
            costs: 'costs',
            bands: selected === 'bands' ? 'bands' : undefined,
            bandCounts: selected === 'bandCounts' ? 'bandCounts' : undefined,
            status:
              selected === 'relaxationCount' || selected === 'reachedEdgeCount'
                ? {[selected]: selected}
                : undefined
          }
        });
        const values = await runOperationGraph(device, {
          inputs: {
            offsets: {format: 'uint32', values: Uint32Array.of(0, 2, 3, 3)},
            neighbors: {format: 'uint32', values: Uint32Array.of(1, 99, 2)},
            weights: {format: 'float32', values: Float32Array.of(1, 1, 1)},
            starts: {format: 'uint32', values: Uint32Array.of(start)},
            breaks: {format: 'float32', values: Float32Array.of(1, 2)}
          },
          operations: [operation],
          outputs: ['costs', ...(selected ? [selected] : [])]
        });
        expect(Array.from(values['costs'])).toEqual(
          start === 0 ? [0, 1, 2] : [Infinity, Infinity, Infinity]
        );
        if (selected)
          expect(Array.from(values[selected])).toEqual(
            selected === 'bands'
              ? start === 0
                ? [0, 0, 1]
                : [4294967295, 4294967295, 4294967295]
              : selected === 'bandCounts'
                ? start === 0
                  ? [2, 1]
                  : [0, 0]
                : start === 0
                  ? selected === 'relaxationCount'
                    ? [3, 0]
                    : [2, 0]
                  : [0, 0]
          );
      }
  } finally {
    device.destroy();
  }
});
