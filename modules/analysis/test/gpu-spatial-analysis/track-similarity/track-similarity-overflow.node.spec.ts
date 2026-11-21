// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTrackSimilarity,
  GPU_TRACK_SIMILARITY_STATUS
} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('reports invalid pair status before a densified count can wrap to one vertex', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', 1048577);
    const lineOffsets = createTransientView(graph, 'offsets', 'uint32', 2);
    const pairA = createTransientView(graph, 'pair-a', 'uint32', 1);
    const pairB = createTransientView(graph, 'pair-b', 'uint32', 1);
    const pairStatus = createTransientView(graph, 'status', 'uint32', 1);
    const nodes = new GPUTrackSimilarity({
      id: 'overflow',
      densify: 1 / 4096,
      linesA: {kind: 'lines', positions, lineOffsets},
      pairA,
      pairB,
      output: {hausdorff: createTransientView(graph, 'distance', 'float32', 1), pairStatus}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(lineOffsets, new Uint32Array([0, positions.length]));
    memory.write(pairA, new Uint32Array([0]));
    memory.write(pairB, new Uint32Array([0]));
    executeCommandNodesOnCPU([nodes.find(node => node.id === 'overflow-status')!], memory);
    expect(memory.readUint32(pairStatus)[0]).toBe(GPU_TRACK_SIMILARITY_STATUS.invalidPair);
    // Invalid ranges must also be rejected before any position access.
    memory.write(lineOffsets, new Uint32Array([9, 3]));
    executeCommandNodesOnCPU([nodes.find(node => node.id === 'overflow-status')!], memory);
    expect(memory.readUint32(pairStatus)[0]).toBe(GPU_TRACK_SIMILARITY_STATUS.invalidPair);
  } finally {
    device.destroy();
  }
});

it('preflights each word-sized decision column before declaring any scratch', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPUTrackSimilarity({
      id: 'decision',
      linesA: {
        kind: 'lines',
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2)
      },
      pairA: createTransientView(graph, 'a', 'uint32', 17),
      pairB: createTransientView(graph, 'b', 'uint32', 17),
      frechetEpsilon: createTransientView(graph, 'epsilon', 'float32', 1),
      output: {frechetWithin: createTransientView(graph, 'within', 'uint32', 17)}
    });
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 64}
    });
    const declarations = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = [declarations.buffers.size, declarations.nodes.length];
    expect(() => contributor.getCommandNodes(graph)).toThrow(/pairA chunk 0 needs 68 bytes/);
    expect([declarations.buffers.size, declarations.nodes.length]).toEqual(before);
  } finally {
    device.destroy();
  }
});

it('does not require unused expanded counts for original-vertex-only measures', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', 1048577);
    const lineOffsets = createTransientView(graph, 'offsets', 'uint32', 2);
    const singletonPositions = createTransientView(graph, 'singleton', 'float32x2', 2);
    const singletonOffsets = createTransientView(graph, 'singleton-offsets', 'uint32', 2);
    const pairA = createTransientView(graph, 'pair-a', 'uint32', 1);
    const pairB = createTransientView(graph, 'pair-b', 'uint32', 1);
    const memory = createCPUGraphMemory();
    memory.write(singletonOffsets, Uint32Array.of(0, 1));
    for (const measure of [
      'maximumDistance',
      'symmetricSegmentPathDistance',
      'hausdorff',
      'frechet'
    ] as const) {
      const pairStatus = createTransientView(graph, `${measure}-status`, 'uint32', 1);
      const nodes = new GPUTrackSimilarity({
        id: measure,
        densify: 1 / 4096,
        linesA: {kind: 'lines', positions, lineOffsets},
        linesB: {kind: 'lines', positions: singletonPositions, lineOffsets: singletonOffsets},
        pairA,
        pairB,
        output: {[measure]: createTransientView(graph, measure, 'float32', 1), pairStatus}
      }).getCommandNodes(graph);
      for (const count of [1048576, 1048577]) {
        memory.write(lineOffsets, Uint32Array.of(0, count));
        executeCommandNodesOnCPU([nodes.find(node => node.id === `${measure}-status`)!], memory);
        expect(memory.readUint32(pairStatus)[0], `${measure} count ${count}`).toBe(
          (measure === 'hausdorff' || measure === 'frechet') && count === 1048577
            ? GPU_TRACK_SIMILARITY_STATUS.invalidPair
            : 0
        );
      }
      // A shorter track beyond the DP cap must not mark a measure that has no DP table.
      memory.write(lineOffsets, Uint32Array.of(0, 2));
      memory.write(singletonOffsets, Uint32Array.of(0, 2));
      executeCommandNodesOnCPU([nodes.find(node => node.id === `${measure}-status`)!], memory);
      expect(memory.readUint32(pairStatus)[0]).toBe(
        measure === 'frechet' ? GPU_TRACK_SIMILARITY_STATUS.frechetCapExceeded : 0
      );
      memory.write(singletonOffsets, Uint32Array.of(0, 1));
    }
  } finally {
    device.destroy();
  }
});
