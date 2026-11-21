// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation-operation';
import {
  CPUSpatialWeightsValidationOperation,
  computeSpatialWeightsValidationOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-validation-operation';
import {GPUSpatialWeightsValidation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeWeightsAuditKernel} from './weights-audit-executor';

it('validation preserves subtype neighborCounts by identity while capturing its CSR bindings', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const neighborCounts = createTransientView(graph, 'subtype-counts', 'uint32', 2);
    const weights = {
      offsets: createTransientView(graph, 'subtype-offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'subtype-neighbors', 'uint32', 2),
      weights: createTransientView(graph, 'subtype-values', 'float32', 2),
      neighborCounts
    };
    const invalidCount = createTransientView(graph, 'subtype-invalid', 'uint32', 1);
    const validation = new GPUSpatialWeightsValidation({weights, output: {status: {invalidCount}}});
    weights.neighborCounts = createTransientView(graph, 'replacement-counts', 'uint32', 2);
    expect(validation.outputWeights.neighborCounts === neighborCounts).toBe(true);
    expect((validation.outputWeights as GPUSpatialWeights).status?.invalidCount).toBe(invalidCount);
    expect(validation.props.weights.neighbors).toBe(weights.neighbors);
  } finally {
    device.destroy();
  }
});

it('validation reads an output status getter once and retains the first destination', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const first = createTransientView(graph, 'first-invalid', 'uint32', 1),
      second = createTransientView(graph, 'second-invalid', 'uint32', 1);
    let reads = 0;
    const weights = {
      offsets: createTransientView(graph, 'getter-offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'getter-neighbors', 'uint32', 2),
      weights: createTransientView(graph, 'getter-values', 'float32', 2)
    };
    const validation = new GPUSpatialWeightsValidation({
      weights,
      output: {
        get status() {
          reads++;
          return {invalidCount: reads === 1 ? first : second};
        }
      }
    });
    expect(reads).toBe(1);
    expect((validation.outputWeights as GPUSpatialWeights).status?.invalidCount === first).toBe(
      true
    );
  } finally {
    device.destroy();
  }
});

it('validation refuses a two-word invalidCount slice during construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const packed = createTransientView(graph, 'packed-invalid', 'uint32', 8);
    const invalidCount = graph.createDataView(packed.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 2
    });
    const weights = {
      offsets: createTransientView(graph, 'scalar-offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'scalar-neighbors', 'uint32', 2),
      weights: createTransientView(graph, 'scalar-values', 'float32', 2)
    };
    expect(
      () => new GPUSpatialWeightsValidation({weights, output: {status: {invalidCount}}})
    ).toThrow(/invalidCount.*one|invalidCount.*1/);
  } finally {
    device.destroy();
  }
});

it('GPU and CPU validation operations enforce rowCount even when flags are omitted', () => {
  for (const Operation of [
    GPUSpatialWeightsValidationOperation,
    CPUSpatialWeightsValidationOperation
  ]) {
    expect(
      () =>
        new Operation({
          rowCount: 0,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          output: {status: {invalidCount: 'invalid'}}
        })
    ).toThrow(/rowCount/);
  }
});

it('GPU validation operation rejects rowCount disagreement when optional flags are absent', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const operation = new GPUSpatialWeightsValidationOperation({
      rowCount: 3,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      output: {status: {invalidCount: 'invalid'}}
    });
    expect(() =>
      operation.createContributor(
        name =>
          createTransientView(
            graph,
            name,
            name === 'weights' ? 'float32' : 'uint32',
            name === 'offsets' ? 9 : 14
          ) as never
      )
    ).toThrow(/rowCount/);
  } finally {
    device.destroy();
  }
});

it('CPU validation and emitted row WGSL match independent violation literals on every row', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const offsets = Uint32Array.of(1, 3, 5, 7, 9, 9, 11, 10, 15);
    const neighbors = Uint32Array.of(6, 0, 1, 4, 4, 9, 0, 3, 1, 6, 7, 0, 8, 7);
    const weights = Float32Array.of(1, 1, NaN, 1, Infinity, -1, 1, 1, 1, 0, 2, 1, 1, NaN);
    // weights-validation-audit.oracle.py. Bits are hand-checkable per row:
    // [offset|self|weight, order|weight, range|order|weight, self|order,
    //  empty, valid zero slot, decreasing offset, all five violations].
    const expected = [25, 20, 22, 12, 0, 0, 1, 31];
    const cpu = computeSpatialWeightsValidationOnCPU(offsets, neighbors, weights);
    expect(Array.from(cpu.flags)).toEqual(expected);
    expect(Array.from(cpu.invalidCount)).toEqual([6]);
    const flags = createTransientView(graph, 'flags', 'uint32', 8);
    const invalidCount = createTransientView(graph, 'invalid', 'uint32', 1);
    const validator = new GPUSpatialWeightsValidation<GPUSpatialWeights>({
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 9),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 14),
        weights: createTransientView(graph, 'weights', 'float32', 14)
      },
      output: {flags, status: {invalidCount}}
    });
    expect(validator.outputWeights.status?.invalidCount).toBe(invalidCount);
    const [kernel] = getGPUCommandNodeKernelSources(validator.getCommandNodes(graph));
    const buffers = {
      offsets,
      neighbors,
      weights,
      flags: new Uint32Array(8),
      violated: new Uint32Array(8),
      graphChunkGroupParameters: new Uint32Array(64)
    };
    buffers.graphChunkGroupParameters.set([0, 8, 8, 0]);
    for (const scale of [1, 2]) {
      // Positive rescaling preserves finiteness, signs and all structural bits.
      buffers.weights = Float32Array.from(weights, value => value * scale);
      const scaledCpu = computeSpatialWeightsValidationOnCPU(offsets, neighbors, buffers.weights);
      expect(Array.from(scaledCpu.flags)).toEqual(expected);
      expect(Array.from(scaledCpu.invalidCount)).toEqual([6]);
      executeWeightsAuditKernel(kernel, buffers);
      expect(Array.from(buffers.flags)).toEqual(expected);
      expect(buffers.violated.reduce((total, value) => total + value, 0)).toBe(6);
    }
  } finally {
    device.destroy();
  }
});
