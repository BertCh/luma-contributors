// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterFillNoData} from '../../src/raster-analysis/gpu-raster-fill-nodata';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import fixture from '../oracles/raster-fill-four-quadrant.json';
async function setup() {
  const candidateDevice = await getWebGPUTestDevice();
  requireTestDevice(candidateDevice);
  const device = candidateDevice;
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  function input<Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: number[],
    format: Format
  ): GraphDataView<Format> {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${identifier++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  }
  function output<Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2' | 'uint32x3'>(
    length: number,
    format: Format
  ) {
    const components = format.endsWith('x2') ? 2 : format.endsWith('x3') ? 3 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length)
    };
  }
  return {
    graph,
    device,
    buffers,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}
function grid(width: number, height: number): GPURasterMetadata {
  return {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'};
}

for (const sample of fixture.cases) {
  it(`four-quadrant fill matches direct Rasterio: ${sample.name}`, async () => {
    const {graph, input, output, device, buffers} = await setup();
    const count = sample.values.length;
    const source = input(sample.values, 'float32');
    const mask = input(sample.mask, 'uint32');
    const values = output(count, 'float32');
    const validity = output(count, 'uint32');
    const filled = output(count, 'uint32');
    const unresolvedCount = output(1, 'uint32');
    graph.add(
      new GPURasterFillNoData({
        grid: grid(sample.width, sample.height),
        input: {
          id: 'source',
          format: 'float32',
          storage: {kind: 'buffer', values: source},
          validity: mask,
          noDataValue: -9999
        },
        maximumDistance: sample.maximumDistance,
        method: 'gdal-four-quadrant',
        output: {
          values: values.view,
          validity: validity.view,
          filled: filled.view,
          unresolvedCount: unresolvedCount.view
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const addition of [0, 100]) {
        buffers[0].write(
          Float32Array.from(sample.values, value => (value === -9999 ? value : value + addition))
        );
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(values.buffer, count);
        const actualValidity = await readUint32(validity.buffer, count);
        const actualFilled = await readUint32(filled.buffer, count);
        let unresolved = 0;
        for (let index = 0; index < count; index++) {
          const valid = sample.expected[index] !== -9999;
          expect(actualValidity[index]).toBe(Number(valid));
          expect(actualFilled[index]).toBe(Number(valid && !sample.mask[index]));
          if (valid) expect(actual[index]).toBeCloseTo(sample.expected[index] + addition, 3);
          else {
            expect(actual[index]).toBeNaN();
            unresolved++;
          }
        }
        expect(Array.from(await readUint32(unresolvedCount.buffer, 1))).toEqual([unresolved]);
      }
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  });
}

it('four-quadrant fill excludes non-finite, masked and raw nodata donors before calibration', async () => {
  const {graph, input, output, run} = await setup();
  const values = output(5, 'float32');
  const validity = output(5, 'uint32');
  const filled = output(5, 'uint32');
  const unresolvedCount = output(1, 'uint32');
  graph.add(
    new GPURasterFillNoData({
      grid: grid(5, 1),
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: input([2, Infinity, -9999, 80, NaN], 'float32')},
        validity: input([1, 1, 1, 0, 1], 'uint32'),
        noDataValue: -9999,
        scale: 3,
        offset: 1
      },
      maximumDistance: 2,
      method: 'gdal-four-quadrant',
      output: {
        values: values.view,
        validity: validity.view,
        filled: filled.view,
        unresolvedCount: unresolvedCount.view
      }
    })
  );
  const destroy = run();
  try {
    const actual = await readFloat32(values.buffer, 5);
    expect(Array.from(actual.slice(0, 3))).toEqual([7, 7, 7]);
    expect(actual[3]).toBeNaN();
    expect(actual[4]).toBeNaN();
    expect(Array.from(await readUint32(validity.buffer, 5))).toEqual([1, 1, 1, 0, 0]);
    expect(Array.from(await readUint32(filled.buffer, 5))).toEqual([0, 1, 1, 0, 0]);
  } finally {
    destroy();
  }
});

for (const method of ['nearest', 'inverse-distance', 'gdal-four-quadrant'] as const) {
  it(`retains distinct donor selection for ${method}`, async () => {
    const {graph, input, output, run} = await setup();
    const values = output(5, 'float32');
    const validity = output(5, 'uint32');
    const filled = output(5, 'uint32');
    const unresolvedCount = output(1, 'uint32');
    graph.add(
      new GPURasterFillNoData({
        grid: grid(5, 1),
        input: {
          id: 'source',
          format: 'float32',
          storage: {kind: 'buffer', values: input([2, 8, NaN, NaN, 10], 'float32')}
        },
        maximumDistance: 2,
        method,
        output: {
          values: values.view,
          validity: validity.view,
          filled: filled.view,
          unresolvedCount: unresolvedCount.view
        }
      })
    );
    const destroy = run();
    try {
      const actual = await readFloat32(values.buffer, 5);
      expect(actual[2]).toBeCloseTo(
        method === 'gdal-four-quadrant' ? 26 / 3 : method === 'nearest' ? 8 : 7,
        4
      );
    } finally {
      destroy();
    }
  });
}
