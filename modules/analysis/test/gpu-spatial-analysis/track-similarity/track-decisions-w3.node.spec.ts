// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeTrackSimilarityOnCPU,
  GPUTrackSimilarity
} from '../../../src/gpu-spatial-analysis/track-similarity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('adds invalid Fréchet threshold evidence only to addressable pairs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'positions', 'float32x2', 2);
    const lineOffsets = createTransientView(graph, 'offsets', 'uint32', 3);
    const pairA = createTransientView(graph, 'pair-a', 'uint32', 4);
    const pairB = createTransientView(graph, 'pair-b', 'uint32', 4);
    const activePairCount = createTransientView(graph, 'active', 'uint32', 1);
    const frechetEpsilon = createTransientView(graph, 'epsilon', 'float32', 1);
    const frechetWithin = createTransientView(graph, 'within', 'uint32', 4);
    const pairStatus = createTransientView(graph, 'pair-status', 'uint32', 4);
    const nodes = new GPUTrackSimilarity({
      id: 'threshold-evidence',
      linesA: {kind: 'lines', positions, lineOffsets},
      pairA,
      pairB,
      activePairCount,
      frechetEpsilon,
      output: {frechetWithin, pairStatus}
    }).getCommandNodes(graph);
    const statusNodes = nodes.filter(
      node =>
        node.id === 'threshold-evidence-status' || node.id === 'threshold-evidence-threshold-status'
    );
    expect(statusNodes).toHaveLength(2);
    const memory = createCPUGraphMemory();
    memory.write(positions, Float32Array.of(0, 0, 2, 0));
    memory.write(lineOffsets, Uint32Array.of(0, 0, 2));
    memory.write(pairA, Uint32Array.of(1, 0, 9, 1));
    memory.write(pairB, Uint32Array.of(1, 1, 1, 1));
    memory.write(activePairCount, Uint32Array.of(3));
    memory.write(pairStatus, Uint32Array.of(255, 255, 255, 255));

    // Rows are an identical pair, an empty-track pair, an invalid index and inactive capacity.
    memory.write(frechetEpsilon, Float32Array.of(-1));
    executeCommandNodesOnCPU(statusNodes, memory);
    expect(Array.from(memory.readUint32(pairStatus))).toEqual([16, 17, 4, 4]);

    // A valid epsilon clears the threshold bit on the next invocation.
    memory.write(frechetEpsilon, Float32Array.of(0));
    executeCommandNodesOnCPU(statusNodes, memory);
    expect(Array.from(memory.readUint32(pairStatus))).toEqual([0, 1, 4, 4]);
  } finally {
    device.destroy();
  }
});

it('reports invalid geometry and thresholds independently of the Frechet cap', () => {
  for (const maximumFrechetVertexCount of [1, 2]) {
    for (const epsilon of [1, NaN]) {
      const result = computeTrackSimilarityOnCPU(
        {
          linesA: {positions: Float32Array.of(0, 0, 1, 0), lineOffsets: Uint32Array.of(0, 2)},
          pairA: Uint32Array.of(0),
          pairB: Uint32Array.of(0),
          frechetEpsilon: Float32Array.of(epsilon)
        },
        {maximumFrechetVertexCount, measures: ['frechet', 'frechetWithin', 'hausdorff']}
      );
      expect(Array.from(result.pairStatus)).toEqual([
        (maximumFrechetVertexCount === 1 ? 2 : 0) | (Number.isNaN(epsilon) ? 16 : 0)
      ]);
      expect(Array.from(result.hausdorff)).toEqual([0]);
      if (maximumFrechetVertexCount === 2) expect(Array.from(result.frechet)).toEqual([0]);
    }
  }
  for (const [longitude, latitude, expected] of [
    [NaN, NaN, 8],
    [5, 91, 8],
    [5, 90, 0]
  ]) {
    const result = computeTrackSimilarityOnCPU(
      {
        linesA: {
          positions: Float32Array.of(longitude, latitude),
          lineOffsets: Uint32Array.of(0, 1)
        },
        pairA: Uint32Array.of(0),
        pairB: Uint32Array.of(0),
        frechetEpsilon: Float32Array.of(1)
      },
      {
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        measures: ['frechetWithin', 'hausdorff']
      }
    );
    expect(Array.from(result.pairStatus)).toEqual([expected]);
    expect(Array.from(result.hausdorff)).toEqual([expected ? NaN : 0]);
    expect(Array.from(result.frechetWithin)).toEqual([expected ? 0 : 1]);
  }
});

it('uses the stored coordinate bits for threshold membership', () => {
  // Exact squared distance is 1 + 2^-26; the displayed float32 norm rounds to 1.
  for (const height of [0, 2 ** -13]) {
    const result = computeTrackSimilarityOnCPU(
      {
        linesA: {positions: Float32Array.of(0, 0, 1, height), lineOffsets: Uint32Array.of(0, 1, 2)},
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
    expect(Array.from(result.frechet)).toEqual([1]);
    expect(Array.from(result.frechetWithin)).toEqual([height ? 0 : 1]);
    expect(Array.from(result.editDistanceOnRealSequence)).toEqual([height ? 1 : 0]);
    expect(Array.from(result.longestCommonSubsequence)).toEqual([height ? 0 : 1]);
    expect(Array.from(result.pairStatus)).toEqual([0]);
  }
});

it('distinguishes zero tracks from an empty track and absent pairs', () => {
  for (const offsets of [Uint32Array.of(0), Uint32Array.of(0, 0)]) {
    for (const pairs of [new Uint32Array(), Uint32Array.of(0)]) {
      const result = computeTrackSimilarityOnCPU(
        {linesA: {positions: new Float32Array(), lineOffsets: offsets}, pairA: pairs, pairB: pairs},
        {measures: ['frechet']}
      );
      expect(result.frechet.length).toBe(pairs.length);
      expect(Array.from(result.pairStatus)).toEqual(
        pairs.length ? [offsets.length === 1 ? 4 : 1] : []
      );
    }
  }
});

it('checks pair evidence byte layouts and keeps uncapped measures usable', async () => {
  const {decodeGPUTrackSimilarityPairStatus: decode} = await import(
    '../../../src/gpu-spatial-analysis/track-similarity'
  );
  const bytes = new Uint8Array(13);
  new DataView(bytes.buffer).setUint32(1, 2, true);
  const prefix = decode(bytes.subarray(1, 9), 1);
  expect(prefix).toHaveLength(1);
  expect(prefix.map(row => row.flags)).toEqual([2]);
  expect(decode(bytes.subarray(1, 9), 0)).toEqual([]);
  const status = prefix[0];
  expect(status.uncappedMeasuresValid).toBe(true);
  expect(status.dynamicProgramMeasuresValid).toBe(false);
  expect(status.decisionValid).toBe(false);
  expect(decode(Uint32Array.of(16))[0].uncappedMeasuresValid).toBe(true);
  expect(() => decode(new Uint8Array(3))).toThrow(/whole/);
  expect(() => decode(new Uint32Array(1), 2)).toThrow(/activePairCount/);
  expect(decode(new ArrayBuffer(0), 0)).toEqual([]);
});

it('rejects each inactive search destination and names its metric', async () => {
  const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {createNullWebGPUDevice} = await import('../../utils/gpu-contributor-test-utils');
  const {GPUTrackSimilaritySearch} = await import(
    '../../../src/gpu-spatial-analysis/track-similarity'
  );
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(format: Format, length = 1) =>
    createTransientView(graph, `decision-${serial++}`, format, length);
  try {
    for (const metric of ['frechet', 'longest-common-subsequence'] as const) {
      const output = {
        matched: view('uint32'),
        status: view('uint32'),
        ...(metric === 'frechet'
          ? {distance: view('float32'), lowerBound: view('float32')}
          : {similarity: view('float32'), upperBound: view('float32')})
      };
      const props = {
        metric,
        linesA: {
          kind: 'lines' as const,
          positions: view('float32x2'),
          lineOffsets: view('uint32', 2)
        },
        pairA: view('uint32'),
        pairB: view('uint32'),
        threshold: view('float32'),
        output
      };
      expect(() => new GPUTrackSimilaritySearch(props)).not.toThrow();
      for (const name of metric === 'frechet'
        ? ['similarity', 'upperBound']
        : ['distance', 'lowerBound']) {
        expect(
          () =>
            new GPUTrackSimilaritySearch({...props, output: {...output, [name]: view('float32')}})
        ).toThrow(new RegExp(`output.${name}.*${metric}`));
      }
    }
  } finally {
    device.destroy();
  }
});

it('supports empty inputs in the named CPU operation and GPU construction', async () => {
  const {CPUTrackSimilarityOperation, GPUTrackSimilarity} = await import(
    '../../../src/gpu-spatial-analysis/track-similarity'
  );
  const {runOperationGraph} = await import('../../utils/mixed-execution-test-utils');
  const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {createNullWebGPUDevice} = await import('../../utils/gpu-contributor-test-utils');
  const device = createNullWebGPUDevice();
  try {
    for (const offsets of [Uint32Array.of(0), Uint32Array.of(0, 0)]) {
      for (const pairs of [new Uint32Array(), Uint32Array.of(0)]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: new Float32Array()},
            offsets: {format: 'uint32', values: offsets},
            pairs: {format: 'uint32', values: pairs}
          },
          operations: [
            new CPUTrackSimilarityOperation({
              linesA: {positions: 'positions', lineOffsets: 'offsets'},
              pairA: 'pairs',
              pairB: 'pairs',
              pairCount: pairs.length,
              output: {frechet: 'distance', pairStatus: 'status'}
            })
          ],
          outputs: ['distance', 'status']
        });
        expect(result['distance'].length).toBe(pairs.length);
        expect(Array.from(result['status'])).toEqual(
          pairs.length ? [offsets.length === 1 ? 4 : 1] : []
        );
        const graph = new GPUCommandGraph(device);
        expect(() =>
          new GPUTrackSimilarity({
            linesA: {
              kind: 'lines',
              positions: createTransientView(graph, 'p', 'float32x2', 0),
              lineOffsets: createTransientView(graph, 'o', 'uint32', offsets.length)
            },
            pairA: createTransientView(graph, 'a', 'uint32', pairs.length),
            pairB: createTransientView(graph, 'b', 'uint32', pairs.length),
            output: {frechet: createTransientView(graph, 'd', 'float32', pairs.length)}
          }).getCommandNodes(graph)
        ).not.toThrow();
      }
    }
  } finally {
    device.destroy();
  }
});

it('keeps representable extreme norms and their half-radius rejection', () => {
  for (const distance of [2 ** -80, 2 ** 70]) {
    const result = computeTrackSimilarityOnCPU(
      {
        linesA: {
          positions: Float32Array.of(0, 0, distance, 0),
          lineOffsets: Uint32Array.of(0, 1, 2)
        },
        pairA: Uint32Array.of(0),
        pairB: Uint32Array.of(1),
        frechetEpsilon: Float32Array.of(distance / 2)
      },
      {measures: ['frechet', 'frechetWithin']}
    );
    expect(Array.from(result.frechet)).toEqual([distance]);
    expect(Array.from(result.frechetWithin)).toEqual([0]);
    expect(Array.from(result.pairStatus)).toEqual([0]);
  }
});
