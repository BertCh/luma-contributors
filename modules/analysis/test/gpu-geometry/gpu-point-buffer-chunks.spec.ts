// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUPointBuffer,
  getGPUPointBufferParameterValues
} from '../../src/geospatial-analysis/point-buffer/index';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {readPointBufferRing} from './gpu-point-buffer-chunks.test-utils';
import {requireTestDevice} from '../utils/require-test-device';

for (const coordinateSystem of ['planar', 'geographic'] as const) {
  it(`${coordinateSystem} point rings match packed rows and a float64 circle/destination oracle`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const source = new Float32Array([
      179.999, 0, 12, 20, -30, -30, 40, 40, -50, -50, 60, 60, -70, -70
    ]);
    const radius = 1000;
    const verticesPerRing = 4;
    const parameterBuffer = createInputBuffer(device, getGPUPointBufferParameterValues(radius));
    buffers.push(parameterBuffer);
    const parameters = importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 2);
    const routeOutputs: {buffer: Buffer; prefix: number; length: number}[][] = [];
    for (const [routeIndex, lengths] of [[7], [1, 4, 2]].entries()) {
      let rowBase = 0;
      const outputBuffers: {buffer: Buffer; prefix: number; length: number}[] = [];
      const inputChunks = [];
      const outputChunks = [];
      for (const [chunkIndex, length] of lengths.entries()) {
        const prefix = chunkIndex + 1;
        const padded = new Float32Array(prefix + length * 2);
        padded.set(source.subarray(rowBase * 2, (rowBase + length) * 2), prefix);
        const inputBuffer = createInputBuffer(device, padded);
        buffers.push(inputBuffer);
        const inputBacking = importGraphBuffer(
          graph,
          `input-${routeIndex}-${chunkIndex}`,
          inputBuffer,
          'float32',
          padded.length
        );
        inputChunks.push(
          graph.createDataView(inputBacking.buffer, {
            format: 'float32x2',
            length,
            byteOffset: prefix * 4
          })
        );
        const outputPrefix = chunkIndex + 2;
        const outputLength = length * verticesPerRing;
        const buffer = createOutputBuffer(device, outputPrefix + outputLength * 2);
        buffers.push(buffer);
        const outputBacking = importGraphBuffer(
          graph,
          `output-${routeIndex}-${chunkIndex}`,
          buffer,
          'float32',
          outputPrefix + outputLength * 2
        );
        outputChunks.push(
          graph.createDataView(outputBacking.buffer, {
            format: 'float32x2',
            length: outputLength,
            byteOffset: outputPrefix * 4
          })
        );
        outputBuffers.push({buffer, prefix: outputPrefix, length: outputLength * 2});
        rowBase += length;
      }
      graph.add(
        new GPUPointBuffer({
          id: `point-buffer-${routeIndex}`,
          spatialContext:
            coordinateSystem === 'geographic'
              ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
              : {coordinateSpace: 'planar', metric: 'native'},
          verticesPerRing,
          parameters,
          points:
            routeIndex === 0
              ? inputChunks[0]
              : createVectorView('points', 'float32x2', inputChunks),
          offsets:
            routeIndex === 0
              ? outputChunks[0]
              : createVectorView('offsets', 'float32x2', outputChunks)
        })
      );
      routeOutputs.push(outputBuffers);
    }
    const compiled = graph.compile();
    try {
      for (let encoding = 0; encoding < 2; encoding++) {
        submitGraph(device, compiled, undefined);
        const results: number[][] = [];
        for (const chunks of routeOutputs) {
          const values = [];
          for (const {buffer, prefix, length} of chunks)
            values.push(...(await readFloat32(buffer, prefix + length)).slice(prefix));
          results.push(values);
        }
        expect(results[1]).toEqual(results[0]);
        for (let row = 0; row < 7; row++) {
          for (let vertex = 0; vertex < verticesPerRing; vertex++) {
            const bearing = (2 * Math.PI * vertex) / verticesPerRing;
            let expected = [radius * Math.sin(bearing), radius * Math.cos(bearing)];
            if (coordinateSystem === 'geographic') {
              const latitude = (source[row * 2 + 1] * Math.PI) / 180;
              const angle = radius / 6371008.8;
              const destinationLatitude = Math.asin(
                Math.sin(latitude) * Math.cos(angle) +
                  Math.cos(latitude) * Math.sin(angle) * Math.cos(bearing)
              );
              const longitudeOffset = Math.atan2(
                Math.sin(bearing) * Math.sin(angle) * Math.cos(latitude),
                Math.cos(angle) - Math.sin(latitude) * Math.sin(destinationLatitude)
              );
              expected = [
                (longitudeOffset * 180) / Math.PI,
                ((destinationLatitude - latitude) * 180) / Math.PI
              ];
            }
            for (let component = 0; component < 2; component++) {
              const index = (row * verticesPerRing + vertex) * 2 + component;
              expect(Math.abs(results[0][index] - expected[component])).toBeLessThan(
                coordinateSystem === 'geographic' ? 2e-6 : 0.001
              );
            }
          }
        }
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

it('buffers more points than the old single-binding ring output can hold', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  // 4,194,305 four-vertex rings exceed the old 128 MiB output binding by 32 bytes.
  const lengths = [1_398_101, 1_398_102, 1_398_102];
  const inputChunks = [];
  const outputChunks = [];
  const outputBuffers: Buffer[] = [];
  for (const [chunkIndex, length] of lengths.entries()) {
    const inputBuffer = createInputBuffer(device, new Float32Array(length * 2));
    const outputBuffer = createOutputBuffer(device, length * 8);
    buffers.push(inputBuffer, outputBuffer);
    outputBuffers.push(outputBuffer);
    inputChunks.push(
      importGraphBuffer(graph, `large-input-${chunkIndex}`, inputBuffer, 'float32x2', length)
    );
    outputChunks.push(
      importGraphBuffer(graph, `large-output-${chunkIndex}`, outputBuffer, 'float32x2', length * 4)
    );
  }
  const parameterBuffer = createInputBuffer(device, getGPUPointBufferParameterValues(2));
  buffers.push(parameterBuffer);
  graph.add(
    new GPUPointBuffer({
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      verticesPerRing: 4,
      parameters: importGraphBuffer(graph, 'large-parameters', parameterBuffer, 'float32', 2),
      points: createVectorView('large-points', 'float32x2', inputChunks),
      offsets: createVectorView('large-offsets', 'float32x2', outputChunks)
    })
  );
  const compiled = graph.compile();
  try {
    for (let encoding = 0; encoding < 2; encoding++) {
      submitGraph(device, compiled, undefined);
      for (const [chunkIndex, buffer] of outputBuffers.entries()) {
        for (const ringIndex of [0, Math.floor(lengths[chunkIndex] / 2), lengths[chunkIndex] - 1]) {
          const ring = await readPointBufferRing(buffer, ringIndex, 4);
          expect(ring).toHaveLength(8);
          for (let vertex = 0; vertex < 4; vertex++) {
            const label = `encoding ${encoding} chunk ${chunkIndex} ring ${ringIndex} vertex ${vertex}`;
            expect(ring[vertex * 2], `${label} x`).toBeCloseTo(
              2 * Math.sin((vertex * Math.PI) / 2),
              5
            );
            expect(ring[vertex * 2 + 1], `${label} y`).toBeCloseTo(
              2 * Math.cos((vertex * Math.PI) / 2),
              5
            );
          }
        }
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('writes consecutive shared-buffer chunks with empty and workgroup-boundary chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const lengths = [0, 1, 255, 256, 257, 0];
  const pointCount = lengths.reduce((total, length) => total + length, 0);
  const verticesPerRing = 4;
  // One leading word exercises uint32-aligned subviews; trailing words back the empty end chunk.
  const pointBuffer = createInputBuffer(device, new Float32Array(1 + pointCount * 2 + 2));
  const outputValues = new Float32Array(1 + pointCount * verticesPerRing * 2 + 2).fill(-17);
  const outputBuffer = createOutputBuffer(device, outputValues.length);
  outputBuffer.write(outputValues);
  const parameterBuffer = createInputBuffer(device, getGPUPointBufferParameterValues(3));
  const pointBacking = importGraphBuffer(
    graph,
    'shared-points',
    pointBuffer,
    'float32',
    1 + pointCount * 2 + 2
  );
  const outputBacking = importGraphBuffer(
    graph,
    'shared-offsets',
    outputBuffer,
    'float32',
    outputValues.length
  );
  let rowBase = 0;
  const pointChunks = [];
  const outputChunks = [];
  for (const length of lengths) {
    pointChunks.push(
      graph.createDataView(pointBacking.buffer, {
        format: 'float32x2',
        length,
        byteOffset: 4 + rowBase * 8
      })
    );
    outputChunks.push(
      graph.createDataView(outputBacking.buffer, {
        format: 'float32x2',
        length: length * verticesPerRing,
        byteOffset: 4 + rowBase * verticesPerRing * 8
      })
    );
    rowBase += length;
  }
  graph.add(
    new GPUPointBuffer({
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      verticesPerRing,
      parameters: importGraphBuffer(graph, 'shared-parameters', parameterBuffer, 'float32', 2),
      points: createVectorView('points', 'float32x2', pointChunks),
      offsets: createVectorView('offsets', 'float32x2', outputChunks)
    })
  );
  const compiled = graph.compile();
  try {
    for (let encoding = 0; encoding < 2; encoding++) {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(outputBuffer);
      expect(actual[0]).toBe(-17);
      expect(actual.slice(-2)).toEqual([-17, -17]);
      for (let pointIndex = 0; pointIndex < pointCount; pointIndex++) {
        for (let vertexIndex = 0; vertexIndex < verticesPerRing; vertexIndex++) {
          const componentBase = 1 + (pointIndex * verticesPerRing + vertexIndex) * 2;
          expect(actual[componentBase]).toBeCloseTo(3 * Math.sin((vertexIndex * Math.PI) / 2), 5);
          expect(actual[componentBase + 1]).toBeCloseTo(
            3 * Math.cos((vertexIndex * Math.PI) / 2),
            5
          );
        }
      }
    }
  } finally {
    compiled.destroy();
    pointBuffer.destroy();
    outputBuffer.destroy();
    parameterBuffer.destroy();
  }
});
