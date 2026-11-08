// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {addKeplerGroupBy, type KeplerGroupBySettings} from '../src/aggregation/kepler-group-by';
import {importGraphBuffer} from '../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

test('native Group By preserves null groups, first occurrence, all-row counts and finite diagnostics across chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const gpuDevice: Device = device;
  for (const variant of ['finite', 'nonfinite-input', 'overflow-output', 'distinct'] as const) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    function input<Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      data: Float32Array | Uint32Array,
      lengths: readonly number[]
    ) {
      let offset = 0;
      const views = lengths.map((length, index) => {
        const buffer = createInputBuffer(gpuDevice, data.slice(offset, offset + length));
        buffers.push(buffer);
        offset += length;
        return importGraphBuffer(graph, `${name}-${index}`, buffer, format, length);
      });
      return new GraphVectorView({
        id: name,
        name,
        format,
        length: data.length,
        valueLength: data.length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: views
      });
    }
    function output<Format extends 'float32' | 'uint32'>(name: string, format: Format) {
      const buffer = createOutputBuffer(gpuDevice, 5);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, `${name}-output`, buffer, format, 5)};
    }
    const key = input('key', 'uint32', new Uint32Array([2, 0, 0, 2, 1, 0, 0]), [2, 0, 5]);
    const keyValidity = input(
      'key-validity',
      'uint32',
      new Uint32Array([1, 1, 0, 1, 1, 1, 0]),
      [2, 0, 5]
    );
    const sourceValues = new Float32Array([4, NaN, 8, 6, NaN, 2, 12]);
    if (variant === 'nonfinite-input') sourceValues[0] = Infinity;
    if (variant === 'overflow-output') sourceValues[0] = sourceValues[3] = 3.4028234663852886e38;
    const values = input('value', 'float32', sourceValues, [2, 0, 5]);
    const validity = input(
      'value-validity',
      'uint32',
      new Uint32Array([1, 0, 1, 1, 0, 1, 1]),
      [2, 0, 5]
    );
    const keys = output('keys', 'uint32');
    const keyValid = output('key-valid', 'uint32');
    const firstRows = output('first', 'uint32');
    const statusBuffer = createOutputBuffer(gpuDevice, 4);
    buffers.push(statusBuffer);
    const status = importGraphBuffer(graph, 'status-output', statusBuffer, 'uint32', 4);
    const count = output('count', 'uint32');
    const countValidity = output('count-validity', 'uint32');
    const numerical = ['sum', 'mean', 'min', 'max'].map(operation => ({
      operation: operation as 'sum' | 'mean' | 'min' | 'max',
      values: output(operation, 'float32'),
      validity: output(`${operation}-validity`, 'uint32')
    }));
    const settings: KeplerGroupBySettings = {
      keyColumn: 'category',
      population: 'all',
      numericalPolicy: 'float32',
      aggregations:
        variant === 'distinct'
          ? []
          : [
              {column: 'text', operation: 'count', name: 'count'},
              ...numerical.map(item => ({
                column: 'value',
                operation: item.operation,
                name: item.operation
              }))
            ]
    };
    const measures =
      variant === 'distinct'
        ? {}
        : {
            count: {values: count.view, validity: countValidity.view},
            ...Object.fromEntries(
              numerical.map(item => [
                item.operation,
                {values: item.values.view, validity: item.validity.view}
              ])
            )
          };
    addKeplerGroupBy(graph, {
      source: {datasetId: 'groups', revision: 1},
      settings,
      sourceFields: ['category', 'text', 'value'],
      key,
      keyValidity,
      dictionaryCount: 4,
      columns: {value: {values, validity}},
      outputs: {
        keys: keys.view,
        keyValidity: keyValid.view,
        firstSourceRows: firstRows.view,
        measures,
        status
      }
    });
    const compiled = graph.compile();
    // Scratch is reported by the published compiled graph, not by a pre-compile plan.
    expect(compiled.stats.physicalTransientResourceBytes).toBeGreaterThan(0);
    try {
      submitGraph(device, compiled);
      const diagnostics = Array.from(await readUint32(statusBuffer, 4));
      expect(diagnostics[0]).toBe(4);
      expect(diagnostics[1]).toBe(0);
      expect(diagnostics[2]).toBe(variant === 'nonfinite-input' ? 1 : 0);
      expect(diagnostics[3]).toBe(variant === 'overflow-output' ? 2 : 0);
      expect(Array.from(await readUint32(keys.buffer, 5)).slice(0, 4)).toEqual([2, 0, 4, 1]);
      expect(Array.from(await readUint32(keyValid.buffer, 5))).toEqual([1, 1, 0, 1, 0]);
      expect(Array.from(await readUint32(firstRows.buffer, 5))).toEqual([0, 1, 2, 4, 0xffffffff]);
      if (variant === 'finite') {
        expect(Array.from(await readUint32(count.buffer, 5))).toEqual([2, 2, 2, 1, 0]);
        expect(Array.from(await readUint32(countValidity.buffer, 5))).toEqual([1, 1, 1, 1, 0]);
        const expected = {
          sum: [10, 2, 20, NaN, NaN],
          mean: [5, 2, 10, NaN, NaN],
          min: [4, 2, 8, NaN, NaN],
          max: [6, 2, 12, NaN, NaN]
        };
        for (const item of numerical) {
          expect(Array.from(await readFloat32(item.values.buffer, 5))).toEqual(
            expected[item.operation]
          );
          expect(Array.from(await readUint32(item.validity.buffer, 5))).toEqual([1, 1, 1, 0, 0]);
        }
      }
      expect(key.data.map(chunk => chunk.length)).toEqual([2, 0, 5]);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
