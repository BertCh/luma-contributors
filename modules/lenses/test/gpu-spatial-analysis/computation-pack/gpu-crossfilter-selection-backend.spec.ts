// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCrossfilter} from '@luma.gl/graph/gpu-crossfilter';
import {GPUTransferMask} from '@luma-contributors/analysis/computation-record';
import {expect, test} from 'vitest';
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
