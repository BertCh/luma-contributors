// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpatialLag,
  type GPUSpatialLagProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {getGPUSpatialWeightsLossViews} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights';
import {getWeightsLossSources} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-loss';
import {computeSpatialWeightsValidationOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function setup() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `lag-r19-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 2),
    neighbors: view('uint32', 1),
    weights: view('float32', 1)
  };
  return {device, graph, view, weights};
}

it('SSR-126 preserves interleaved loss-view ordering and deduplicates shared outer/chunk views', () => {
  const {device, view, weights} = setup();
  try {
    const shared = view('uint32', 1),
      outerCandidate = view('uint32', 1),
      chunkFinal = view('uint32', 1),
      chunkCandidate = view('uint32', 1);
    const input = {
      rowCount: 2,
      status: {overflow: shared, candidateOverflow: outerCandidate},
      chunks: [
        {
          rowStart: 0,
          rowCount: 1,
          weights: {...weights, status: {overflow: shared, candidateOverflow: chunkCandidate}}
        },
        {
          rowStart: 1,
          rowCount: 1,
          weights: {...weights, status: {overflow: chunkFinal, candidateOverflow: shared}}
        }
      ]
    };
    expect(getGPUSpatialWeightsLossViews(input)).toEqual([
      shared,
      outerCandidate,
      chunkCandidate,
      chunkFinal
    ]);
    const grouped = getWeightsLossSources([input]);
    expect([...new Set([...grouped.overflow, ...grouped.candidateOverflow])]).not.toEqual(
      getGPUSpatialWeightsLossViews(input)
    );
  } finally {
    device.destroy();
  }
});

it('review B8 cross weights use source-value bounds instead of square validator bounds', () => {
  const {device, graph, view, weights} = setup();
  try {
    const invalidForSquare = computeSpatialWeightsValidationOnCPU(
      Uint32Array.of(0, 1),
      Uint32Array.of(2),
      Float32Array.of(1)
    );
    expect(Array.from(invalidForSquare.flags)).toEqual([2]);
    expect(Array.from(invalidForSquare.invalidCount)).toEqual([1]);
    const lag = new GPUSpatialLag({
      weights,
      values: view('float32', 3),
      output: view('float32', 1)
    });
    expect(lag.getValueRowCount()).toBe(3);
    expect(() => lag.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('review B4 ignores extra fields on a structurally wider status output', () => {
  const {device, graph, view, weights} = setup();
  try {
    const status = {count: view('uint32', 1), overflow: view('uint32', 1)};
    const nodes = new GPUSpatialLag({
      weights,
      values: view('float32', 1),
      output: view('float32', 1),
      status
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(status.count, Uint32Array.of(42));
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id.includes('-status-')),
      memory
    );
    expect(Array.from(memory.readUint32(status.count))).toEqual([42]);
    expect(Array.from(memory.readUint32(status.overflow))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

it('review B5 permits adjacent packed cause words through loss and recovery', () => {
  const {device, graph, view, weights} = setup();
  try {
    const packed = view('uint32', 2);
    const status = {
      overflow: graph.createDataView(packed.buffer, {format: 'uint32', length: 1, byteOffset: 0}),
      candidateOverflow: graph.createDataView(packed.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 4
      })
    };
    const inputStatus = {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)};
    const nodes = new GPUSpatialLag({
      weights: {...weights, status: inputStatus},
      values: view('float32', 1),
      output: view('float32', 1),
      status
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    for (const flags of [
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      memory.write(inputStatus.overflow, Uint32Array.of(flags[0]));
      memory.write(inputStatus.candidateOverflow, Uint32Array.of(flags[1]));
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.includes('-status-')),
        memory
      );
      expect(Array.from(memory.readUint32(packed))).toEqual(flags);
    }
  } finally {
    device.destroy();
  }
});

it('review S2 rejects writes aliasing outer or chunk input validation metadata', () => {
  const {device, view, weights} = setup();
  try {
    const invalidCount = view('uint32', 1);
    const part = {...weights, status: {invalidCount}};
    for (const input of [
      part,
      {rowCount: 1, status: {invalidCount}, chunks: [{rowStart: 0, rowCount: 1, weights}]},
      {rowCount: 1, chunks: [{rowStart: 0, rowCount: 1, weights: part}]}
    ]) {
      expect(
        () =>
          new GPUSpatialLag({
            weights: input,
            values: view('float32', 1),
            output: view('float32', 1),
            status: {candidateOverflow: invalidCount}
          })
      ).toThrow(/share buffers/);
    }
  } finally {
    device.destroy();
  }
});

it('SSR-113 publishes both causes once per parent and clears them on resubmission', () => {
  const {device, graph, view, weights} = setup();
  try {
    const outerLoss = view('uint32', 1),
      chunkLoss = view('uint32', 1),
      outerCandidateLoss = view('uint32', 1),
      chunkFinalLoss = view('uint32', 1);
    const status = {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)};
    const contributor = new GPUSpatialLag({
      values: view('float32', 2),
      output: view('float32', 2),
      status,
      weights: {
        rowCount: 2,
        status: {overflow: outerLoss, candidateOverflow: outerCandidateLoss},
        chunks: [
          {rowStart: 0, rowCount: 1, weights: {...weights, status: {candidateOverflow: chunkLoss}}},
          {rowStart: 1, rowCount: 1, weights: {...weights, status: {overflow: chunkFinalLoss}}}
        ]
      }
    } as GPUSpatialLagProps);
    const nodes = contributor.getCommandNodes(graph);
    const publishers = nodes.filter(node => node.id.includes('-status-'));
    expect(publishers.map(node => node.id)).toEqual([
      'spatial-lag-status-overflow-0',
      'spatial-lag-status-candidateOverflow-0'
    ]);
    const memory = createCPUGraphMemory();
    for (const [
      outerFinal,
      chunkCandidate,
      chunkFinal,
      outerCandidate,
      overflow,
      candidateOverflow
    ] of [
      [1, 0, 0, 0, 1, 0],
      [0, 1, 0, 0, 0, 1],
      [0, 0, 7, 0, 1, 0],
      [0, 0, 0, 9, 0, 1],
      [0, 0, 0, 0, 0, 0]
    ]) {
      memory.write(outerLoss, Uint32Array.of(outerFinal));
      memory.write(chunkLoss, Uint32Array.of(chunkCandidate));
      memory.write(chunkFinalLoss, Uint32Array.of(chunkFinal));
      memory.write(outerCandidateLoss, Uint32Array.of(outerCandidate));
      executeCommandNodesOnCPU(publishers, memory);
      expect(Array.from(memory.readUint32(status.overflow))).toEqual([overflow]);
      expect(Array.from(memory.readUint32(status.candidateOverflow))).toEqual([candidateOverflow]);
    }
  } finally {
    device.destroy();
  }
});

it('SSR-113 validates optional status formats, scalar lengths, overlap and graph membership', () => {
  const {device, graph, view, weights} = setup();
  try {
    const base = {values: view('float32', 1), output: view('float32', 1), weights};
    for (const invalid of [view('float32', 1), view('uint32', 2)]) {
      expect(
        () =>
          new GPUSpatialLag({...base, status: {overflow: invalid}} as unknown as GPUSpatialLagProps)
      ).toThrow(/status.overflow/);
    }
    const shared = view('uint32', 1);
    expect(
      () =>
        new GPUSpatialLag({
          ...base,
          status: {overflow: shared, candidateOverflow: shared}
        } as GPUSpatialLagProps)
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUSpatialLag({
          ...base,
          status: {overflow: base.weights.neighbors}
        } as GPUSpatialLagProps)
    ).toThrow(/share buffers/);
    const otherGraph = new GPUCommandGraph(device);
    const foreign = createTransientView(otherGraph, 'foreign-status', 'uint32', 1);
    expect(() =>
      new GPUSpatialLag({
        ...base,
        status: {overflow: foreign}
      } as GPUSpatialLagProps).getCommandNodes(graph)
    ).toThrow();
    const largeBuffer = view('uint32', 64);
    const oversizedBinding = graph.createDataView(largeBuffer.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 252
    });
    (device.limits as {maxStorageBufferBindingSize: number}).maxStorageBufferBindingSize = 128;
    expect(() =>
      new GPUSpatialLag({...base, status: {overflow: oversizedBinding}}).getCommandNodes(graph)
    ).toThrow(/status.overflow|binding/i);
  } finally {
    device.destroy();
  }
});

it('SSR-113 writes zero for a requested cause with no supplied loss sources', () => {
  const {device, graph, view, weights} = setup();
  try {
    const status = {candidateOverflow: view('uint32', 1)};
    const nodes = new GPUSpatialLag({
      weights,
      values: view('float32', 1),
      output: view('float32', 1),
      status
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(status.candidateOverflow, Uint32Array.of(99));
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id.includes('-status-')),
      memory
    );
    expect(Array.from(memory.readUint32(status.candidateOverflow))).toEqual([0]);
  } finally {
    device.destroy();
  }
});
