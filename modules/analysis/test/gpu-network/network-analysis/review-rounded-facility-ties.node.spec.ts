// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUNetworkServiceAreasOperation,
  computeNetworkServiceAreasOnCPU
} from '../../../src/gpu-network/network-analysis/cpu-network-service-areas-operation';

it.each([
  undefined,
  8
])('B1 keeps the facility with a tight path when rounded totals coincide (budget=%s)', async maximumIterations => {
  const topology = {
    offsets: Uint32Array.of(0, 1, 2, 3, 4, 4),
    neighbors: Uint32Array.of(2, 3, 3, 4),
    weights: Float32Array.of(0.3, 0.7, 0.4, 1)
  };
  const facilities = Uint32Array.of(0, 1);
  const costPrecision = 'float32-per-edge';
  const options = {maximumIterations, localIterations: 1};
  const result = computeNetworkServiceAreasOnCPU({topology, facilities, costPrecision}, options);
  const operation = new CPUNetworkServiceAreasOperation({
    nodeCount: 5,
    facilityCapacity: 2,
    topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    facilities: 'facilities',
    costPrecision,
    ...options,
    output: {
      assignments: 'assignments',
      costs: 'costs',
      predecessors: 'predecessors',
      facilityNodeCounts: 'facilityNodeCounts'
    }
  });
  const inputs = Object.fromEntries(
    Object.entries({...topology, facilities}).map(([port, values]) => [
      port,
      {
        format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
        length: values.length,
        chunks: [{values, length: values.length}],
        pack: () => values
      }
    ])
  );
  // Independent IEEE binary32 evaluation: f32(f32(.3)+f32(.4))=.7000000476837158,
  // f32(.7)=.699999988079071. Adding 1 rounds both to 1.7000000476837158.
  // Facility 1 alone has a tight path to node 3 and supplies the label on edge 3 -> 4.
  const expected = {
    costs: Float32Array.of(0, 0, 0.30000001192092896, 0.699999988079071, 1.7000000476837158),
    assignments: Uint32Array.of(0, 1, 0, 1, 1),
    facilityNodeCounts: Uint32Array.of(2, 3),
    predecessors: Uint32Array.of(0xffffffff, 0xffffffff, 0, 1, 3)
  };
  expect(result).toMatchObject(expected);
  expect(await operation.execute({inputs})).toEqual(expected);
});
