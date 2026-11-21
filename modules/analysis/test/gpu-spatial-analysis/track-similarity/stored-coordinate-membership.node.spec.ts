// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  computeTrackSimilarityOnCPU,
  CPUTrackSimilarityOperation,
  GPUTrackSimilaritySearch
} from '../../../src/gpu-spatial-analysis/track-similarity';
import {PLANAR_TRACK_MEMBERSHIP_WGSL} from '../../../src/gpu-spatial-analysis/track-similarity/planar-track-metric';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

// Exact distances: 1+2^-25 > 1; sqrt(1+2^-26) > 1; 1 = 1.
const CASES = [
  {coordinates: [-(2 ** -25), 0, 1, 0], within: 0},
  {coordinates: [0, 0, 1, 2 ** -13], within: 0},
  {coordinates: [0, 0, 1, 0], within: 1}
];

it('uses original stored endpoints in direct and named CPU decisions', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const {coordinates, within} of CASES) {
      const positions = Float32Array.from(coordinates);
      const offsets = Uint32Array.of(0, 1, 2);
      const direct = computeTrackSimilarityOnCPU(
        {
          linesA: {positions, lineOffsets: offsets},
          pairA: Uint32Array.of(0),
          pairB: Uint32Array.of(1),
          frechetEpsilon: Float32Array.of(1)
        },
        {
          measures: [
            'frechet',
            'frechetWithin',
            'editDistanceOnRealSequence',
            'longestCommonSubsequence'
          ]
        }
      );
      expect(Array.from(direct.frechet)).toEqual([1]);
      expect(Array.from(direct.frechetWithin)).toEqual([within]);
      expect(Array.from(direct.editDistanceOnRealSequence)).toEqual([1 - within]);
      expect(Array.from(direct.longestCommonSubsequence)).toEqual([within]);
      expect(Array.from(direct.pairStatus)).toEqual([0]);
      const named = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: positions},
          offsets: {format: 'uint32', values: offsets},
          first: {format: 'uint32', values: Uint32Array.of(0)},
          second: {format: 'uint32', values: Uint32Array.of(1)},
          epsilon: {format: 'float32', values: Float32Array.of(1)}
        },
        operations: [
          new CPUTrackSimilarityOperation({
            linesA: {positions: 'positions', lineOffsets: 'offsets'},
            pairA: 'first',
            pairB: 'second',
            pairCount: 1,
            frechetEpsilon: 'epsilon',
            output: {
              frechet: 'distance',
              frechetWithin: 'within',
              editDistanceOnRealSequence: 'edits',
              longestCommonSubsequence: 'similarity',
              pairStatus: 'status'
            }
          })
        ],
        outputs: ['distance', 'within', 'edits', 'similarity', 'status']
      });
      for (const [name, expected] of Object.entries({
        distance: 1,
        within,
        edits: 1 - within,
        similarity: within,
        status: 0
      })) {
        expect(Array.from(named[name])).toEqual([expected]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('executes the GPU predicate and LCSS search bound with original stored endpoints', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const {coordinates, within} of CASES) {
      const graph = new GPUCommandGraph(device);
      const positions = createTransientView(graph, 'positions', 'float32x2', 2);
      const predicate = createTransientView(graph, 'predicate', 'uint32', 1);
      const node = createWGSLKernelNode(graph, {
        id: 'stored-membership',
        operation: 'test',
        invocationCount: 1,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'predicate', view: predicate, type: 'u32', access: 'read_write'}
        ],
        declarations: PLANAR_TRACK_MEMBERSHIP_WGSL,
        body: 'predicate[predicateOffset + index] = select(0u, 1u, pointWithin(vec2f(positions[positionsOffset], positions[positionsOffset + 1u]), vec2f(positions[positionsOffset + 2u], positions[positionsOffset + 3u]), 1.0));'
      });
      const offsets = createTransientView(graph, 'offsets', 'uint32', 3);
      const first = createTransientView(graph, 'first', 'uint32', 1);
      const second = createTransientView(graph, 'second', 'uint32', 1);
      const threshold = createTransientView(graph, 'threshold', 'float32', 1);
      const bound = createTransientView(graph, 'bound', 'float32', 1);
      const nodes = new GPUTrackSimilaritySearch({
        metric: 'longest-common-subsequence',
        linesA: {kind: 'lines', positions, lineOffsets: offsets},
        pairA: first,
        pairB: second,
        threshold,
        output: {
          similarity: createTransientView(graph, 'similarity', 'float32', 1),
          upperBound: bound,
          matched: createTransientView(graph, 'matched', 'uint32', 1),
          status: createTransientView(graph, 'status', 'uint32', 1)
        }
      }).getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(positions, Float32Array.from(coordinates));
      memory.write(offsets, Uint32Array.of(0, 1, 2));
      memory.write(first, Uint32Array.of(0));
      memory.write(second, Uint32Array.of(1));
      memory.write(threshold, Float32Array.of(1));
      executeCommandNodesOnCPU(
        [node, ...nodes.filter(candidate => candidate.id.endsWith('-bound'))],
        memory
      );
      expect(Array.from(memory.readUint32(predicate))).toEqual([within]);
      expect(Array.from(memory.readFloat32(bound))).toEqual([within]);
    }
  } finally {
    device.destroy();
  }
});

it('S2 rejects an outside pair when subnormal subtraction results flush to zero', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const points = createTransientView(graph, 'tiny-points', 'float32', 5);
    const result = createTransientView(graph, 'tiny-result', 'uint32', 1);
    const original = 'let offset = abs(firstPoint-secondPoint);';
    expect(PLANAR_TRACK_MEMBERSHIP_WGSL.includes(original), 'FTZ injection point').toBe(true);
    const declarations = PLANAR_TRACK_MEMBERSHIP_WGSL.replace(
      original,
      `let rawOffset = abs(firstPoint-secondPoint);
  let offset = vec2f(
    select(rawOffset.x, 0.0, (bitcast<u32>(rawOffset.x) & 0x7fffffffu) < 0x00800000u),
    select(rawOffset.y, 0.0, (bitcast<u32>(rawOffset.y) & 0x7fffffffu) < 0x00800000u));`
    );
    const node = createWGSLKernelNode(graph, {
      id: 'ftz-membership',
      operation: 'test',
      invocationCount: 1,
      bindings: [
        {name: 'points', view: points, type: 'f32', access: 'read'},
        {name: 'result', view: result, type: 'u32', access: 'read_write'}
      ],
      declarations,
      body: 'result[resultOffset] = select(0u, 1u, pointWithin(vec2f(points[pointsOffset], points[pointsOffset+1u]), vec2f(points[pointsOffset+2u], points[pointsOffset+3u]), points[pointsOffset+4u]));'
    });
    const radius = 2 ** -126;
    const memory = createCPUGraphMemory();
    memory.write(points, Float32Array.of(radius, radius, 1.75 * radius, 1.75 * radius, radius));
    executeCommandNodesOnCPU([node], memory);
    // Both input coordinates are normal. Exact squared separation = 2*(3r/4)^2 = 9r²/8 > r².
    expect(Array.from(memory.readUint32(result))).toEqual([0]);
  } finally {
    device.destroy();
  }
});
