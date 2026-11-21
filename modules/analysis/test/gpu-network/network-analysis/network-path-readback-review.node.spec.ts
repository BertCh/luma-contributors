// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Buffer} from 'node:buffer';
import {expect, it} from 'vitest';
import {
  decodeGPUNetworkPathExtractionOutput,
  type NetworkPathReadbackInput
} from '../../../src/gpu-network/network-analysis/network-path-readback';
import {computeNetworkPathExtractionOnCPU} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
const READ: NetworkPathReadbackInput = {
  ids: Uint32Array.of(55, 66),
  pathFound: Uint32Array.of(1),
  lineOffsets: Uint32Array.of(0, 2),
  count: Uint32Array.of(2)
};

it('B2 reads each compact payload against its own count', () => {
  for (const [capacity, edgeCapacity] of [
    [0, 2],
    [3, 1]
  ]) {
    const result = computeNetworkPathExtractionOnCPU({
      predecessors: Uint32Array.of(0xffffffff, 0, 1),
      predecessorEdges: Uint32Array.of(0xffffffff, 0, 1),
      costs: Float32Array.of(0, 1, 2),
      targets: Uint32Array.of(2),
      capacity,
      edges: {
        offsets: Uint32Array.of(0, 1, 2, 2),
        neighbors: Uint32Array.of(1, 2),
        weights: Float32Array.of(1, 1),
        capacity: edgeCapacity
      }
    });
    const path = decodeGPUNetworkPathExtractionOutput(result)[0];
    expect(path.fullyStored).toBe(capacity === 3);
    expect(path.edges!.fullyStored).toBe(edgeCapacity === 2);
    expect(Array.from(path.nodes)).toEqual(capacity === 3 ? [0, 1, 2] : []);
    expect(Array.from(path.edges!.ids)).toEqual(edgeCapacity === 2 ? [0, 1] : []);
    expect(Array.from(path.edges!.resolutionValidity)).toEqual(edgeCapacity === 2 ? [1, 1] : []);
  }
});
it('B3 respects an unaligned Node Buffer window with surrounding bytes', () => {
  const bytes = Buffer.alloc(16, 0xaa);
  bytes.writeUInt32LE(55, 1);
  bytes.writeUInt32LE(66, 5);
  expect(
    Array.from(decodeGPUNetworkPathExtractionOutput({...READ, ids: bytes.subarray(1, 9)})[0].nodes)
  ).toEqual([55, 66]);
});
it('B4 rejects short target reads and inconsistent compact layouts', () => {
  for (const input of [
    {...READ, pathFound: new Uint32Array()},
    {...READ, lineOffsets: Uint32Array.of(0, 2, 2)},
    {...READ, lineOffsets: Uint32Array.of(0, 1)},
    {
      ...READ,
      edges: {
        ids: new Uint32Array(),
        edgeRows: new Uint32Array(),
        lineOffsets: Uint32Array.of(0, 0),
        count: Uint32Array.of(0)
      }
    },
    {
      ...READ,
      edges: {
        ids: Uint32Array.of(0, 1),
        edgeRows: Uint32Array.of(0, 1),
        lineOffsets: Uint32Array.of(0, 1),
        count: Uint32Array.of(2)
      }
    }
  ])
    expect(() => decodeGPUNetworkPathExtractionOutput(input)).toThrow();
});
