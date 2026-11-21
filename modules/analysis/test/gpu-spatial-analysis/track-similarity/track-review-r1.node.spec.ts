// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  computeTrackSimilarityOnCPU,
  GPUTrackSimilarity,
  GPUTrackSimilaritySearch
} from '../../../src/gpu-spatial-analysis/track-similarity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('decides stored planar membership independently of displayed norms', () => {
  // 3²+4²=5²; 1²+(2^-30)²>1², even though binary64 hypot rounds to 1.
  for (const [x, y, radius, expected] of [
    [3, 4, 5, 1],
    [1, 2 ** -30, 1, 0]
  ]) {
    const result = computeTrackSimilarityOnCPU(
      {
        linesA: {positions: Float32Array.of(0, 0, x, y), lineOffsets: Uint32Array.of(0, 1, 2)},
        pairA: Uint32Array.of(0),
        pairB: Uint32Array.of(1),
        frechetEpsilon: Float32Array.of(radius)
      },
      {
        editDistanceTolerance: radius,
        lcssTolerance: radius,
        measures: [
          'frechet',
          'frechetWithin',
          'editDistanceOnRealSequence',
          'longestCommonSubsequence'
        ]
      }
    );
    expect(Array.from(result.frechetWithin)).toEqual([expected]);
    expect(Array.from(result.editDistanceOnRealSequence)).toEqual([1 - expected]);
    expect(Array.from(result.longestCommonSubsequence)).toEqual([expected]);
    expect(Array.from(result.pairStatus)).toEqual([0]);
  }
});

function scene(pairCount = 1) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const positions = view('float32x2', 2);
  const offsets = view('uint32', 3);
  const first = view('uint32', pairCount),
    second = view('uint32', pairCount);
  const threshold = view('float32', 1);
  const value = view('float32', pairCount),
    bound = view('float32', pairCount);
  const matched = view('uint32', pairCount),
    status = view('uint32', pairCount);
  return {
    device,
    graph,
    view,
    positions,
    offsets,
    first,
    second,
    threshold,
    value,
    bound,
    matched,
    status
  };
}

it('validates metrics and structured layout errors at construction', () => {
  const fixture = scene();
  const {
    device,
    graph,
    view,
    positions,
    offsets,
    first,
    second,
    threshold,
    value,
    bound,
    matched,
    status
  } = fixture;
  try {
    const props = {
      linesA: {kind: 'lines' as const, positions, lineOffsets: offsets},
      pairA: first,
      pairB: second
    };
    expect(
      () =>
        new GPUTrackSimilaritySearch({
          ...props,
          metric: 'unknown' as 'frechet',
          survivorCapacity: 0,
          threshold,
          output: {distance: value, lowerBound: bound, matched, status}
        })
    ).toThrow(/metric.*frechet.*longest-common-subsequence/);
    for (const invalid of [
      {...props, pairB: view('uint32', 2)},
      {...props, linesA: {...props.linesA, lineOffsets: view('uint32', 0)}}
    ]) {
      try {
        new GPUTrackSimilarity({...invalid, output: {frechet: value}});
        expect.fail('accepted invalid layout');
      } catch (error) {
        expect((error as Error).name).toBe('AnalysisValidationError');
        expect((error as Error).message).toMatch(/received/);
      }
    }
    expect(graph).toBeDefined();
  } finally {
    device.destroy();
  }
});

it('keeps large decision capacities in four-byte bindings and allocates no inactive decision view', () => {
  for (const metric of ['frechet', 'hausdorff'] as const) {
    const {
      device,
      graph,
      positions,
      offsets,
      first,
      second,
      threshold,
      value,
      bound,
      matched,
      status
    } = scene(8388609);
    try {
      const before = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
      const nodes = new GPUTrackSimilaritySearch({
        metric,
        linesA: {kind: 'lines', positions, lineOffsets: offsets},
        pairA: first,
        pairB: second,
        threshold,
        output: {distance: value, lowerBound: bound, matched, status}
      }).getCommandNodes(graph);
      expect(nodes.length).toBeGreaterThan(0);
      const names = [...(graph as unknown as {buffers: Map<string, unknown>}).buffers.keys()];
      expect(names.slice(before).some(name => name.includes('compact-decision'))).toBe(
        metric === 'frechet'
      );
    } finally {
      device.destroy();
    }
  }
});

it('search bounds retain extreme endpoints and reject invalid geographic data before pruning', () => {
  for (const geographic of [false, true]) {
    const {
      device,
      graph,
      positions,
      offsets,
      first,
      second,
      threshold,
      value,
      bound,
      matched,
      status
    } = scene();
    try {
      const nodes = new GPUTrackSimilaritySearch({
        metric: 'frechet',
        spatialContext: geographic
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : undefined,
        linesA: {kind: 'lines', positions, lineOffsets: offsets},
        pairA: first,
        pairB: second,
        threshold,
        output: {distance: value, lowerBound: bound, matched, status}
      }).getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(
        positions,
        geographic ? Float32Array.of(0, 0, 0, 91) : Float32Array.of(0, 0, 2 ** 70, 0)
      );
      memory.write(offsets, Uint32Array.of(0, 1, 2));
      memory.write(first, Uint32Array.of(0));
      memory.write(second, Uint32Array.of(1));
      memory.write(threshold, [geographic ? 1 : 2 ** 70]);
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.endsWith('-bound') || node.id.endsWith('-prune')),
        memory
      );
      expect(Array.from(memory.readUint32(status))).toEqual([geographic ? 2 : 0]);
      if (!geographic) expect(Array.from(memory.readFloat32(bound))).toEqual([2 ** 70]);
    } finally {
      device.destroy();
    }
  }
});

it('uses identical cause precedence for invalid, absent and inactive pairs', () => {
  const {device, graph, view} = scene();
  try {
    const positions = view('float32x2', 2),
      offsets = view('uint32', 2),
      first = view('uint32', 3),
      second = view('uint32', 3),
      active = view('uint32', 1),
      epsilon = view('float32', 1),
      status = view('uint32', 3);
    const nodes = new GPUTrackSimilarity({
      linesA: {kind: 'lines', positions, lineOffsets: offsets},
      pairA: first,
      pairB: second,
      activePairCount: active,
      frechetEpsilon: epsilon,
      maximumFrechetVertexCount: 1,
      output: {frechetWithin: view('uint32', 3), pairStatus: status}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(positions, [0, 0, NaN, 0]);
    memory.write(offsets, Uint32Array.of(0, 2));
    memory.write(first, Uint32Array.of(0, 1, 0));
    memory.write(second, Uint32Array.of(0, 0, 0));
    memory.write(active, Uint32Array.of(2));
    memory.write(epsilon, [NaN]);
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id.endsWith('-status')),
      memory
    );
    expect(Array.from(memory.readUint32(status))).toEqual([24, 4, 4]);
  } finally {
    device.destroy();
  }
});

it('executes the planar GPU predicate for non-axial equality and sub-norm components', async () => {
  const {PLANAR_TRACK_MEMBERSHIP_WGSL} = await import(
    '../../../src/gpu-spatial-analysis/track-similarity/planar-track-metric'
  );
  const {createWGSLKernelNode} = await import(
    '../../../src/analysis-core/kernels/wgsl-kernel-nodes'
  );
  const {device, graph, view} = scene();
  try {
    const input = view('float32', 12),
      output = view('uint32', 4);
    const node = createWGSLKernelNode(graph, {
      id: 'membership',
      operation: 'test',
      invocationCount: 4,
      bindings: [
        {name: 'input', view: input, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      declarations: PLANAR_TRACK_MEMBERSHIP_WGSL,
      body: 'output[outputOffset + index] = select(0u, 1u, pointWithin(vec2f(0.0), vec2f(input[inputOffset + index * 3u], input[inputOffset + index * 3u + 1u]), input[inputOffset + index * 3u + 2u]));'
    });
    const memory = createCPUGraphMemory();
    // Both integer triples are Pythagorean; the final pair has exact square 1+2^-60.
    memory.write(
      input,
      Float32Array.of(
        3,
        4,
        5,
        20055 / 16384,
        12728 / 16384,
        23753 / 16384,
        1,
        2 ** -30,
        1,
        2 ** 70,
        0,
        2 ** 70
      )
    );
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readUint32(output))).toEqual([1, 1, 0, 1]);
  } finally {
    device.destroy();
  }
});

it('preserves multipart one-empty-track evidence', () => {
  const result = computeTrackSimilarityOnCPU(
    {
      linesA: {
        positions: new Float32Array(),
        lineOffsets: Uint32Array.of(0),
        featureOffsets: Uint32Array.of(0, 0)
      },
      pairA: Uint32Array.of(0),
      pairB: Uint32Array.of(0)
    },
    {measures: ['hausdorff']}
  );
  expect(Array.from(result.pairStatus)).toEqual([1]);
  expect(Array.from(result.hausdorff)).toEqual([NaN]);
});

it.each([
  false,
  true
])('preserves multipart zero-track evidence after the shared layout fix (B6), requested=%s', requested => {
  const pairs = requested ? Uint32Array.of(0) : new Uint32Array();
  const result = computeTrackSimilarityOnCPU(
    {
      linesA: {
        positions: new Float32Array(),
        lineOffsets: Uint32Array.of(0),
        featureOffsets: Uint32Array.of(0)
      },
      pairA: pairs,
      pairB: pairs
    },
    {measures: ['hausdorff']}
  );
  expect(Array.from(result.pairStatus)).toEqual(pairs.length ? [4] : []);
  expect(Array.from(result.hausdorff)).toEqual(pairs.length ? [NaN] : []);
});

it('retains the exact Pythagorean LCSS equality in the search bound', () => {
  const {
    device,
    graph,
    positions,
    offsets,
    first,
    second,
    threshold,
    value,
    bound,
    matched,
    status
  } = scene();
  try {
    const nodes = new GPUTrackSimilaritySearch({
      metric: 'longest-common-subsequence',
      lcssTolerance: 23753 / 16384,
      linesA: {kind: 'lines', positions, lineOffsets: offsets},
      pairA: first,
      pairB: second,
      threshold,
      output: {similarity: value, upperBound: bound, matched, status}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(positions, Float32Array.of(0, 0, 20055 / 16384, 12728 / 16384));
    memory.write(offsets, Uint32Array.of(0, 1, 2));
    memory.write(first, Uint32Array.of(0));
    memory.write(second, Uint32Array.of(1));
    memory.write(threshold, Float32Array.of(1));
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id.endsWith('-bound') || node.id.endsWith('-prune')),
      memory
    );
    expect(Array.from(memory.readFloat32(bound))).toEqual([1]);
    expect(Array.from(memory.readUint32(status))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

it('pins the CPU cause vector for invalid geometry, absent and inactive pairs', () => {
  const result = computeTrackSimilarityOnCPU(
    {
      linesA: {positions: Float32Array.of(0, 0, NaN, 0), lineOffsets: Uint32Array.of(0, 2)},
      pairA: Uint32Array.of(0, 1, 0),
      pairB: Uint32Array.of(0, 0, 0),
      activePairCount: Uint32Array.of(2),
      frechetEpsilon: Float32Array.of(NaN)
    },
    {maximumFrechetVertexCount: 1, measures: ['frechet', 'frechetWithin', 'hausdorff']}
  );
  expect(Array.from(result.pairStatus)).toEqual([24, 4, 4]);
  expect(Array.from(result.frechet)).toEqual([NaN, NaN, NaN]);
  expect(Array.from(result.hausdorff)).toEqual([NaN, NaN, NaN]);
  expect(Array.from(result.frechetWithin)).toEqual([0, 0, 0]);
});
