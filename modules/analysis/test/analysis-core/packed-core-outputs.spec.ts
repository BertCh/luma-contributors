// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUAdamStep} from '../../src/analysis-core/kernels/gpu-adam-step';
import {getGPUAdamStepParameterValues} from '../../src/analysis-core/kernels/adam-step-parameters';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUPolygonTriangulation} from '../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {GPUSegmentRingAssembly} from '../../src/analysis-core/ring-assembly/gpu-segment-ring-assembly';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../src/testing/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';

function createFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = device.createBuffer({data: values, usage: Buffer.STORAGE});
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format);
  };
  const output = (name: string, words: number) => {
    const buffer = createOutputBuffer(device, words);
    buffers.push(buffer);
    // Each physical buffer is imported once. Every packed field is an offset view of this handle.
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', words)};
  };
  return {graph, buffers, input, output};
}

it('preserves packed ring orientation and independently expects a counter-clockwise unit square', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {graph, buffers, input, output} = createFixture(device);
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const endpoints = input(
      'endpoints',
      new Float32Array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0]),
      'float32'
    );
    const positions = output('positions', 10);
    const offsets = output('offsets', 2);
    const packed = output('hole-and-status', 128);
    graph.add(
      new GPUSegmentRingAssembly({
        endpoints: graph.createDataView(endpoints.buffer, {format: 'float32x4', length: 4}),
        output: {
          positions: graph.createDataView(positions.view.buffer, {format: 'float32x2', length: 5}),
          ringOffsets: offsets.view,
          ringIsHole: graph.createDataView(packed.view.buffer, {format: 'uint32', length: 1}),
          status: {
            count: graph.createDataView(packed.view.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: 4
            }),
            overflow: graph.createDataView(packed.view.buffer, {
              format: 'uint32',
              length: 1,
              byteOffset: 256
            })
          }
        }
      })
    );
    compiled = graph.compile();
    for (let repetition = 0; repetition < 2; repetition++) {
      submitGraph(device, compiled, undefined);
      const words = await readUint32(packed.buffer);
      expect([words[0], words[1], words[64]]).toEqual([0, 1, 0]);
      expect(await readUint32(offsets.buffer)).toEqual([0, 5]);
    }
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('publishes packed triangle indices, offsets and result columns against a one-triangle oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {graph, buffers, input, output} = createFixture(device);
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const positions = input('positions', new Float32Array([0, 0, 2, 0, 0, 3]), 'float32');
    const features = input('features-and-polygons', Uint32Array.of(0, 1), 'uint32');
    const rings = input('rings', Uint32Array.of(0, 3), 'uint32');
    const packed = output('triangle-columns', 192);
    graph.add(
      new GPUPolygonTriangulation({
        positions: graph.createDataView(positions.buffer, {format: 'float32x2', length: 3}),
        featureOffsets: features,
        polygonOffsets: features,
        ringOffsets: rings,
        output: {
          indices: graph.createDataView(packed.view.buffer, {format: 'uint32', length: 3}),
          triangleOffsets: graph.createDataView(packed.view.buffer, {
            format: 'uint32',
            length: 2,
            byteOffset: 12
          }),
          valid: graph.createDataView(packed.view.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 256
          }),
          triangleCount: graph.createDataView(packed.view.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 512
          })
        }
      })
    );
    compiled = graph.compile();
    for (let repetition = 0; repetition < 2; repetition++) {
      submitGraph(device, compiled, undefined);
      const words = await readUint32(packed.buffer);
      expect(words.slice(0, 3).sort((first, second) => first - second)).toEqual([0, 1, 2]);
      expect(words.slice(3, 5)).toEqual([0, 1]);
      expect([words[64], words[128]]).toEqual([1, 1]);
    }
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('updates packed Adam weights and moments against the zero-moment first-step identity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {graph, buffers, input, output} = createFixture(device);
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    const gradients = input('gradients', Float32Array.of(1, -2), 'float32');
    const loss = input('loss', Float32Array.of(2), 'float32');
    const parameters = input(
      'parameters',
      getGPUAdamStepParameterValues({learningRate: 0.05}),
      'float32'
    );
    const packed = output('weights-and-moments', 192);
    const iterationCount = output('iterations', 1);
    const converged = output('converged', 1);
    const iterationLimitReached = output('limited', 1);
    const invalidCount = output('invalid', 1);
    graph.add(
      new GPUAdamStep({
        gradients,
        loss,
        parameters,
        maximumIterations: 1,
        output: {
          weights: graph.createDataView(packed.view.buffer, {format: 'float32', length: 2}),
          firstMoment: graph.createDataView(packed.view.buffer, {
            format: 'float32',
            length: 2,
            byteOffset: 256
          }),
          secondMoment: graph.createDataView(packed.view.buffer, {
            format: 'float32',
            length: 2,
            byteOffset: 512
          }),
          lossHistory: graph.createDataView(packed.view.buffer, {
            format: 'float32',
            length: 2,
            byteOffset: 8
          }),
          status: {
            iterationCount: iterationCount.view,
            converged: converged.view,
            iterationLimitReached: iterationLimitReached.view,
            invalidCount: invalidCount.view
          }
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(packed.buffer);
    for (const [index, gradient] of [1, -2].entries()) {
      expect(values[index]).toBeCloseTo((-0.05 * gradient) / (Math.abs(gradient) + 1e-8), 6);
      expect(values[64 + index]).toBeCloseTo(0.1 * gradient, 6);
      expect(values[128 + index]).toBeCloseTo(0.001 * gradient ** 2, 6);
    }
    expect(await readUint32(iterationCount.buffer)).toEqual([1]);
    expect(await readUint32(iterationLimitReached.buffer)).toEqual([1]);
    expect(await readUint32(invalidCount.buffer)).toEqual([0]);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
