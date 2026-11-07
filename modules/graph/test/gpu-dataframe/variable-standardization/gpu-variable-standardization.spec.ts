// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GraphVectorView,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUVariableStandardization,
  type GPUVariableStandardizationMethod
} from '../../../src/gpu-dataframe/variable-standardization';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {standardizeReference} from './standardization-oracle';

const methods: GPUVariableStandardizationMethod[] = [
  'deviation-from-mean',
  'z-score',
  'mad',
  'range-adjust',
  'range-standardize'
];

function createFixture(
  device: Device,
  values: number[],
  method: GPUVariableStandardizationMethod,
  chunked: boolean,
  variance: 'sample' | 'population'
) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const inputBuffers: {buffer: Buffer; start: number; length: number}[] = [];
  const maskBuffers: {buffer: Buffer; start: number; length: number}[] = [];
  function inputVector<Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    data: number[],
    boundaries: number[]
  ) {
    const chunks: GraphDataView<Format>[] = [];
    let start = 0;
    for (const [index, length] of boundaries.entries()) {
      const typed =
        format === 'float32'
          ? new Float32Array(data.slice(start, start + length))
          : new Uint32Array(data.slice(start, start + length));
      const buffer = createInputBuffer(device, typed);
      buffers.push(buffer);
      if (name === 'values') inputBuffers.push({buffer, start, length});
      else maskBuffers.push({buffer, start, length});
      chunks.push(importGraphBuffer(graph, `${name}-${index}`, buffer, format, length));
      start += length;
    }
    return new GraphVectorView({
      id: name,
      name,
      format,
      length: data.length,
      valueLength: data.length,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: chunks
    });
  }
  const rowCount = values.length;
  const mask = values.map((_, row) => (row % 7 === 3 ? 0 : 1));
  const input = inputVector('values', 'float32', values, chunked ? [2, rowCount - 2] : [rowCount]);
  const maskView = inputVector('mask', 'uint32', mask, chunked ? [1, 3, rowCount - 4] : [rowCount]);
  const outputBuffer = createOutputBuffer(device, rowCount);
  buffers.push(outputBuffer);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'float32', rowCount);
  const chunkedOutput = new GraphVectorView({
    id: 'output-chunks',
    name: 'output-chunks',
    format: 'float32',
    length: rowCount,
    valueLength: rowCount,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: [
      graph.createDataView(output.buffer, {format: 'float32', length: 3}),
      graph.createDataView(output.buffer, {format: 'float32', length: rowCount - 3, byteOffset: 12})
    ]
  });
  graph.add(
    new GPUVariableStandardization({
      values: input,
      mask: maskView,
      output: chunked ? chunkedOutput : output,
      method,
      variance
    })
  );
  const compiled = graph.compile();
  return {
    mask,
    async run(nextValues: number[], nextMask: number[] = mask) {
      for (const {buffer, start, length} of maskBuffers)
        buffer.write(new Uint32Array(nextMask.slice(start, start + length)));
      for (const {buffer, start, length} of inputBuffers)
        buffer.write(new Float32Array(nextValues.slice(start, start + length)));
      submitGraph(device, compiled, undefined);
      return readFloat32(outputBuffer, rowCount);
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

it.each(
  methods
)('GPUVariableStandardization %s matches independent definitions across chunks and repeated frames', async method => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenes = [
    [0, 0, 12, 1000, -6, -3, -1, NaN, Infinity, -Infinity, 4, 2],
    Array(12).fill(7),
    Array(12).fill(Math.fround(3e38)),
    Array.from({length: 1025}, (_, row) => Math.fround(1e12 + (row % 13) * 65536)),
    Array.from({length: 12}, (_, row) => Math.fround(row % 2 ? 3e38 : -3e38)),
    Array(12).fill(NaN),
    [NaN, NaN, 9, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    Array.from({length: 12}, (_, row) => -row - 1),
    Array.from({length: 12}, (_, row) => row * row - 23)
  ];
  for (const variance of ['sample', 'population'] as const) {
    for (const group of [
      scenes.filter(values => values.length === 12),
      scenes.filter(values => values.length !== 12)
    ]) {
      const whole = createFixture(device, group[0], method, false, variance);
      const chunked = createFixture(device, group[0], method, true, variance);
      try {
        for (const values of [...group, group[0]]) {
          const reference = standardizeReference(values, method, whole.mask, variance);
          for (const result of [await whole.run(values), await chunked.run(values)]) {
            for (let row = 0; row < values.length; row++) {
              if (!Number.isFinite(Math.fround(reference[row])))
                expect(Number.isNaN(result[row])).toBe(true);
              else
                expect(
                  Math.abs(result[row] - reference[row]),
                  `${method} ${variance} value=${values[row]} row=${row} scene0=${values[0]} actual=${result[row]} expected=${reference[row]}`
                ).toBeLessThan(2e-5 * Math.max(1, Math.abs(reference[row])));
            }
          }
        }
        const zeroMask = group[0].map(() => 0);
        expect((await chunked.run(group[0], zeroMask)).every(Number.isNaN)).toBe(true);
        expect((await chunked.run(group[0])).some(Number.isFinite)).toBe(true);
      } finally {
        whole.destroy();
        chunked.destroy();
      }
    }
  }
});
