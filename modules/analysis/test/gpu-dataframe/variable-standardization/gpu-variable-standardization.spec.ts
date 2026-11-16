// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUVariableStandardization,
  type GPUVariableStandardizationMethod
} from '../../../src/dataframe-analysis/variable-standardization/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
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
  const maskView = inputVector('mask', 'uint32', mask, chunked ? [2, rowCount - 2] : [rowCount]);
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
      graph.createDataView(output.buffer, {format: 'float32', length: 2}),
      graph.createDataView(output.buffer, {format: 'float32', length: rowCount - 2, byteOffset: 8})
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
)('GPUVariableStandardization %s matches independent definitions and repeated frames', async method => {
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
        expect((await whole.run(group[0], zeroMask)).every(Number.isNaN)).toBe(true);
        expect((await whole.run(group[0])).some(Number.isFinite)).toBe(true);
      } finally {
        whole.destroy();
        chunked.destroy();
      }
    }
  }
});

it('GPUVariableStandardization transforms a packed input above the old 2^24 row limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Fixed input and scratch stay below 512 MiB; the closed-form range is [5, 15].
  const rowCount = 2 ** 24 + 3;
  const values = new Float32Array(rowCount).fill(5);
  values[0] = NaN;
  values[rowCount - 1] = 15;
  const graph = new GPUCommandGraph(device);
  const inputBuffer = createInputBuffer(device, values);
  const outputBuffer = createOutputBuffer(device, rowCount);
  graph.add(
    new GPUVariableStandardization({
      values: importGraphBuffer(graph, 'wide-values', inputBuffer, 'float32', rowCount),
      output: importGraphBuffer(graph, 'wide-output', outputBuffer, 'float32', rowCount),
      method: 'range-standardize'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const read = async (row: number) => {
      const bytes = await outputBuffer.readAsync(row * 4, 4);
      return new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
    };
    expect(await read(0)).toBeNaN();
    expect(await read(1)).toBe(0);
    expect(await read(rowCount - 2)).toBe(0);
    expect(await read(rowCount - 1)).toBe(1);
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
});

it.each([
  'deviation-from-mean',
  'z-score',
  'mad',
  'range-adjust'
] as const)('wide %s uses the global odd finite population', async method => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const finiteCount = 2 ** 24 + 1;
  const lowerCount = 2 ** 23 + 1;
  const upperCount = 2 ** 23;
  const rowCount = finiteCount + 4;
  const values = new Float32Array(rowCount).fill(1);
  values.fill(3, lowerCount, finiteCount);
  values[finiteCount] = NaN;
  values[finiteCount + 1] = Infinity;
  values[finiteCount + 2] = 1;
  values[finiteCount + 3] = 3;
  const mask = new Uint32Array(rowCount).fill(1);
  mask.fill(0, finiteCount + 2);
  const mean = (lowerCount + 3 * upperCount) / finiteCount;
  const squareSum = lowerCount * (1 - mean) ** 2 + upperCount * (3 - mean) ** 2;
  const absoluteMean =
    (lowerCount * Math.abs(1 - mean) + upperCount * Math.abs(3 - mean)) / finiteCount;
  for (const variance of ['sample', 'population'] as const) {
    const graph = new GPUCommandGraph(device);
    const input = createInputBuffer(device, values);
    const output = createOutputBuffer(device, rowCount);
    const maskBuffer = createInputBuffer(device, mask);
    graph.add(
      new GPUVariableStandardization({
        values: importGraphBuffer(graph, 'values', input, 'float32', rowCount),
        output: importGraphBuffer(graph, 'output', output, 'float32', rowCount),
        mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rowCount),
        method,
        variance
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const read = async (row: number) => {
        const bytes = await output.readAsync(row * 4, 4);
        return new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0];
      };
      const expected = (value: number) =>
        method === 'deviation-from-mean'
          ? value - mean
          : method === 'range-adjust'
            ? value / 2
            : method === 'mad'
              ? (value - mean) / absoluteMean
              : (value - mean) /
                Math.sqrt(squareSum / (finiteCount - Number(variance === 'sample')));
      for (const row of [0, lowerCount - 1, lowerCount, finiteCount - 1])
        expect(Math.abs((await read(row)) - expected(values[row]))).toBeLessThan(2e-6);
      expect(await read(finiteCount)).toBeNaN();
      expect(await read(finiteCount + 1)).toBeNaN();
      expect(await read(finiteCount + 2)).toBeNaN();
      expect(await read(finiteCount + 3)).toBeNaN();
    } finally {
      compiled.destroy();
      input.destroy();
      output.destroy();
      maskBuffer.destroy();
    }
  }
});

it.each(methods)('extreme normal scales preserve independent %s definitions', async method => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1e38, 1e-37, 1.2e-38]) {
    if (method === 'deviation-from-mean' && scale < 1e-37) continue; // Its true result is subnormal.
    const values = Array.from(new Float32Array([scale, 2 * scale, scale, NaN]));
    const fixture = createFixture(device, values, method, true, 'population');
    try {
      const expected = standardizeReference(values, method, fixture.mask, 'population');
      const actual = await fixture.run(values);
      for (let row = 0; row < 3; row++) {
        const unit = method === 'deviation-from-mean' ? scale : 1;
        expect(Math.abs(actual[row] - expected[row])).toBeLessThan(
          3e-5 * Math.max(unit, Math.abs(expected[row]))
        );
      }
      expect(actual[3]).toBeNaN();
    } finally {
      fixture.destroy();
    }
  }
});
