// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUCrossfilter} from '@luma.gl/graph/gpu-crossfilter';
import {GPUTransferMask} from '@luma-contributors/analysis/computation-record';
import {expect, test, vi} from 'vitest';
import {GPUCrossfilterSelectionBackend} from '../../../src/computation-views/coordinator/gpu-crossfilter-selection-backend';

test('GPU range selection filters rows, transfers CSR counts and reads packed nonzero counts', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const graph = new GPUCommandGraph(device, {id: 'selection-backend'});
  const buffers: Buffer[] = [];
  function importValues<Format extends 'float32' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) {
    const buffer = device!.createBuffer({
      id,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return {buffer, view: graph.createDataView(handle, {format, length: values.length})};
  }
  // Row 0 maps to groups 0 and 1, row 1 to group 0 and row 2 to group 1; row values are 0, 5, 10.
  const values = importValues('values', new Float32Array([0, 5, 10]), 'float32');
  const selected = importValues('selected', new Uint32Array(3), 'uint32');
  const groups = importValues('groups', new Uint32Array(2), 'uint32');
  const offsets = importValues('offsets', new Uint32Array([0, 2, 3, 4]), 'uint32');
  const neighbors = importValues('neighbors', new Uint32Array([0, 1, 0, 1]), 'uint32');
  const crossfilter = new GPUCrossfilter(graph, {
    id: 'range-brush',
    dimensions: [{kind: 'range', id: 'brush', input: values.view}],
    outputMask: selected.view
  });
  crossfilter.addToGraph();
  const transfer = new GPUTransferMask({
    id: 'row-to-group',
    source: selected.view,
    target: groups.view,
    relation: {layout: 'csr', offsets: offsets.view, neighbors: neighbors.view},
    mode: 'count'
  });
  const backend = new GPUCrossfilterSelectionBackend({
    crossfilter,
    minimumKeyCount: 0,
    keySpaces: {
      row: {keyCount: 3, dimensions: {map: 'brush'}},
      group: {keyCount: 2, dimensions: {}}
    },
    relations: {groups: {from: 'row', to: 'group', transfer, output: groups.view}}
  });
  const read = async (buffer: Buffer) => {
    const bytes = await buffer.readAsync();
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4));
  };
  try {
    backend.setClause({
      source: 'map',
      keySpace: 'row',
      label: 'two rows',
      predicate: {kind: 'range', field: 'value', range: [4, 11]}
    });
    const source = backend.getMask('row');
    expect(source.buffer).toBe(selected.view);
    expect(await read(selected.buffer)).toEqual([0, 1, 1]);
    expect(backend.transferMask(source, 'groups').buffer).toBe(groups.view);
    expect(await read(groups.buffer)).toEqual([1, 1]);
    expect(await backend.readCounts()).toEqual({row: 2, group: 2});
    backend.setClause({
      source: 'map',
      keySpace: 'row',
      label: 'one row',
      predicate: {kind: 'range', field: 'value', range: [-1, 1]}
    });
    expect(await backend.readCounts()).toEqual({row: 1, group: 2});
    expect(await read(groups.buffer)).toEqual([1, 1]);
    backend.clearSource('map');
    expect(await backend.readCounts()).toEqual({row: 3, group: 2});
    // Explicit key sets are not supported by the published crossfilter.
    expect(() =>
      backend.setClause({
        source: 'map',
        keySpace: 'row',
        label: 'keys',
        predicate: {kind: 'keys', keys: [0]}
      })
    ).toThrow(/Unsupported keys/);
  } finally {
    backend.destroy();
    crossfilter.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

test('counts compile before a self-excluding histogram mask is requested, including zero selected rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) throw new Error('WebGPU is required for selection backend verification');
  const graph = new GPUCommandGraph(device, {id: 'selection-backend-exclude'});
  // Observe real imports so the adapter-owned mask can be read without accessing private state.
  const imports = vi.spyOn(graph, 'importBuffer');
  const compile = vi.spyOn(graph, 'compile');
  const buffers: Buffer[] = [];
  function importValues<Format extends 'float32' | 'uint32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) {
    const buffer = device!.createBuffer({
      id,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return {buffer, view: graph.createDataView(handle, {format, length: values.length})};
  }
  const rowValues = [0, 5, 10];
  const histogramEdges = [-1, 4, 9, 11];
  const values = importValues('values', new Float32Array(rowValues), 'float32');
  const selected = importValues('selected', new Uint32Array(rowValues.length), 'uint32');
  const histogram = importValues('histogram', new Uint32Array(histogramEdges.length - 1), 'uint32');
  const crossfilter = new GPUCrossfilter(graph, {
    id: 'self-excluding-range',
    dimensions: [{kind: 'range', id: 'brush', input: values.view}],
    views: [
      {
        kind: 'histogram',
        id: 'histogram',
        dimension: 'brush',
        includeOwnSelection: false,
        input: values.view,
        output: histogram.view,
        edges: histogramEdges
      }
    ],
    outputMask: selected.view
  });
  crossfilter.addToGraph();
  const backend = new GPUCrossfilterSelectionBackend({
    crossfilter,
    minimumKeyCount: 0,
    keySpaces: {row: {keyCount: rowValues.length, dimensions: {histogram: 'brush'}}}
  });
  const read = async (buffer: Buffer, byteOffset = 0, length = rowValues.length) => {
    const bytes = await buffer.readAsync(byteOffset, length * Uint32Array.BYTES_PER_ELEMENT);
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, length));
  };
  try {
    expect(crossfilter.getViewMask('histogram')).toBeUndefined();
    for (const range of [
      [4, 11],
      [20, 30]
    ] as const) {
      backend.setClause({
        source: 'histogram',
        keySpace: 'row',
        label: 'range',
        predicate: {kind: 'range', field: 'value', range}
      });
      const expectedMask = rowValues.map(value => Number(value >= range[0] && value <= range[1]));
      // This must be the first backend read: it compiles before getMask can request all rows.
      expect(await backend.readCounts()).toEqual({
        row: expectedMask.reduce((sum, value) => sum + value, 0)
      });
      expect(compile).toHaveBeenCalledTimes(1);
      expect(await read(selected.buffer)).toEqual(expectedMask);
      const importCount = imports.mock.calls.length;
      const excluded = backend.getMask('row', {excludeSource: 'histogram'});
      expect(imports.mock.calls.length).toBe(importCount);
      expect(excluded.keyCount).toBe(rowValues.length);
      if (!(excluded.buffer instanceof GraphDataView))
        throw new Error('Expected a packed exclusion mask');
      const view = excluded.buffer;
      const importIndex = imports.mock.results.findIndex(
        result => result.type === 'return' && result.value === view.buffer
      );
      const importedBuffer = imports.mock.calls[importIndex]?.[1];
      if (!(importedBuffer instanceof Buffer))
        throw new Error('Expected an imported exclusion buffer');
      expect(await read(importedBuffer, view.byteOffset, view.length)).toEqual(
        rowValues.map(() => 1)
      );
      const expectedHistogram = histogramEdges
        .slice(1)
        .map(
          (upper, bin) =>
            rowValues.filter(
              value =>
                value >= histogramEdges[bin] &&
                (value < upper || (bin === histogramEdges.length - 2 && value === upper))
            ).length
        );
      expect(await read(histogram.buffer)).toEqual(expectedHistogram);
    }
  } finally {
    backend.destroy();
    crossfilter.destroy();
    buffers.forEach(buffer => buffer.destroy());
    imports.mockRestore();
    compile.mockRestore();
  }
});
