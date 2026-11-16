// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord,
  createWGSLKernelNode
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUTrajectorySplit} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectorySplitOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split-operation';
import {getGPUTrajectorySplitParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-split';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  expectPackedTrajectorySplitReadback,
  expectSaturatedTrajectorySplitReadback
} from './trajectory-split-scalability-assertions';

it('splits 262145 two-row tracks with a closed-form prefix across three scan levels', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const trackCount = 262145;
  const rowCount = trackCount * 2;
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array(rowCount * 2)},
      timestamps: {
        format: 'float32',
        values: Float32Array.from({length: rowCount}, (_, row) => row % 2)
      },
      trackOffsets: {
        format: 'uint32',
        values: Uint32Array.from({length: trackCount + 1}, (_, track) => track * 2)
      },
      parameters: {format: 'float32', values: getGPUTrajectorySplitParameterValues({maximumGap: 0})}
    },
    operations: [
      new GPUTrajectorySplitOperation({
        mode: 'gap',
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        parameters: 'parameters',
        rowCount,
        segmentCapacity: trackCount,
        rowCapacity: rowCount,
        output: {
          segmentOffsets: 'offsets',
          segmentTracks: 'tracks',
          rowIds: 'rows',
          status: {count: 'count', overflow: 'overflow', requiredCount: 'required'}
        }
      })
    ],
    outputs: ['offsets', 'tracks', 'rows', 'count', 'overflow', 'required']
  });
  expect(Array.from(result['count'])).toEqual([trackCount]);
  expect(Array.from(result['required'])).toEqual([trackCount]);
  expect(Array.from(result['overflow'])).toEqual([0]);
  expect(Array.from(result['offsets'])).toEqual(
    Array.from({length: trackCount + 1}, (_, track) => track * 2)
  );
  expect(Array.from(result['tracks'])).toEqual(
    Array.from({length: trackCount}, (_, track) => track)
  );
  expect(Array.from(result['rows'])).toEqual(Array.from({length: rowCount}, (_, row) => row));
}, 120000);

for (const trackCount of [257, 65537]) {
  it(`handles empty, singleton and overlapping value-change tracks at ${trackCount} tracks`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const offsets = [0];
    const values: number[] = [];
    const expectedRows: number[] = [];
    const expectedOffsets = [0];
    const expectedTracks: number[] = [];
    for (let track = 0; track < trackCount; track++) {
      const begin = values.length;
      const length = [0, 1, 4][track % 3];
      values.push(...[1, 1, 2, 2].slice(0, length));
      offsets.push(values.length);
      if (length === 4) {
        expectedRows.push(begin, begin + 1, begin + 2, begin + 2, begin + 3);
        expectedTracks.push(track, track);
        expectedOffsets.push(expectedRows.length - 2, expectedRows.length);
      }
    }
    const rowCount = values.length;
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array(rowCount * 2)},
        timestamps: {format: 'float32', values: new Float32Array(rowCount)},
        trackOffsets: {format: 'uint32', values: new Uint32Array(offsets)},
        values: {format: 'uint32', values: new Uint32Array(values)}
      },
      operations: [
        new GPUTrajectorySplitOperation({
          mode: 'value-change',
          positions: 'positions',
          timestamps: 'timestamps',
          trackOffsets: 'trackOffsets',
          values: 'values',
          rowCount,
          segmentCapacity: expectedTracks.length,
          rowCapacity: expectedRows.length,
          output: {
            segmentOffsets: 'offsets',
            segmentTracks: 'tracks',
            rowIds: 'rows',
            status: {count: 'count', overflow: 'overflow'}
          }
        })
      ],
      outputs: ['offsets', 'tracks', 'rows', 'count', 'overflow']
    });
    expect(Array.from(result['offsets'])).toEqual(expectedOffsets);
    expect(Array.from(result['tracks'])).toEqual(expectedTracks);
    expect(Array.from(result['rows'])).toEqual(expectedRows);
    expect(Array.from(result['count'])).toEqual([expectedTracks.length]);
    expect(Array.from(result['overflow'])).toEqual([0]);
  }, 120000);
}

it('runs adjacent segment-track and count fields in one physical output buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (
    name: string,
    values: Uint32Array | Float32Array,
    format: 'uint32' | 'float32'
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format);
  };
  const shared = createOutputBuffer(device, 2);
  buffers.push(shared);
  const packed = importGraphBuffer(graph, 'packed', shared, 'uint32', 2);
  const count = graph.createDataView(packed.buffer, {format: 'uint32', byteOffset: 4, length: 1});
  const segmentTracks = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const offsetsBuffer = createOutputBuffer(device, 2);
  buffers.push(offsetsBuffer);
  const rowsBuffer = createOutputBuffer(device, 2);
  buffers.push(rowsBuffer);
  const overflowBuffer = createOutputBuffer(device, 1);
  buffers.push(overflowBuffer);
  graph.add(
    new GPUTrajectorySplit({
      mode: 'value-change',
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      timestamps: input('times', new Float32Array([0, 1]), 'float32') as GraphDataView<'float32'>,
      trackOffsets: input('offsets', new Uint32Array([0, 2]), 'uint32') as GraphDataView<'uint32'>,
      values: input('values', new Uint32Array([1, 1]), 'uint32') as GraphDataView<'uint32'>,
      output: {
        segmentTracks,
        segmentOffsets: importGraphBuffer(graph, 'out-offsets', offsetsBuffer, 'uint32', 2),
        rowIds: importGraphBuffer(graph, 'out-rows', rowsBuffer, 'uint32', 2),
        status: {count, overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expectPackedTrajectorySplitReadback({
      packed: await readUint32(shared, 2),
      offsets: await readUint32(offsetsBuffer, 2),
      rows: await readUint32(rowsBuffer, 2),
      overflow: await readUint32(overflowBuffer, 1)
    });
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('executes seeded near-overflow split prefixes without wrapping output writes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const segmentOverflow of [false, true]) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const output = (name: string, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const rows = output('rows', 4);
    const tracks = output('tracks', 2);
    const offsets = output('out-offsets', 3);
    const count = output('count', 1);
    const overflow = output('overflow', 1);
    const required = output('required', 1);
    const prefixes = output('prefixes', 2);
    const values = createTransientView(graph, 'values', 'uint32', 4);
    const trackOffsets = createTransientView(graph, 'track-offsets', 'uint32', 3);
    recordWGSLKernelNodes(graph);
    const nodes = new GPUTrajectorySplit({
      id: 'seeded',
      mode: 'value-change',
      values,
      trackOffsets,
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      timestamps: createTransientView(graph, 'times', 'float32', 4),
      output: {
        rowIds: rows.view,
        segmentTracks: tracks.view,
        segmentOffsets: offsets.view,
        status: {count: count.view, overflow: overflow.view, requiredCount: required.view}
      }
    }).getCommandNodes(graph);
    const write = nodes.find(node => node.id === 'seeded-write')!;
    const status = nodes.find(node => node.id === 'seeded-status')!;
    const writeRecord = getWGSLKernelNodeRecord(graph, write.id)!;
    const statusRecord = getWGSLKernelNodeRecord(graph, status.id)!;
    const find = (name: string) =>
      [...writeRecord.bindings, ...statusRecord.bindings].find(binding => binding.name === name)!
        .view;
    const rowPrefixes = find('trackRowOffsets');
    const segmentPrefixes = find('trackSegmentOffsets');
    const seed = createWGSLKernelNode(graph, {
      id: 'seed',
      operation: 'SplitSeed',
      invocationCount: 1,
      bindings: [
        {name: 'values', view: values, type: 'u32', access: 'read_write'},
        {name: 'tracks', view: trackOffsets, type: 'u32', access: 'read_write'},
        {name: 'rowPrefixes', view: rowPrefixes, type: 'u32', access: 'read_write'},
        {name: 'segmentPrefixes', view: segmentPrefixes, type: 'u32', access: 'read_write'},
        {name: 'totalRows', view: find('totalRows'), type: 'u32', access: 'read_write'},
        {name: 'totalSegments', view: find('totalSegments'), type: 'u32', access: 'read_write'},
        {name: 'rows', view: rows.view, type: 'u32', access: 'read_write'},
        {name: 'outTracks', view: tracks.view, type: 'u32', access: 'read_write'}
      ],
      body: `values[0] = 1u; values[1] = ${segmentOverflow ? '2u' : '1u'}; values[2] = 3u; values[3] = 4u;
        tracks[0] = 0u; tracks[1] = ${segmentOverflow ? '4u' : '2u'}; tracks[2] = 4u;
        rowPrefixes[0] = 0u; rowPrefixes[1] = 0xffffffffu;
        segmentPrefixes[0] = ${segmentOverflow ? '0xffffffffu' : '0u'}; segmentPrefixes[1] = 0xffffffffu;
        totalRows[0] = 0xffffffffu; totalSegments[0] = 0xffffffffu;
        rows[0] = 99u; rows[1] = 99u; rows[2] = 99u; rows[3] = 99u;
        outTracks[0] = 99u; outTracks[1] = 99u;`
    });
    const capture = createWGSLKernelNode(graph, {
      id: 'capture',
      operation: 'SplitSeed',
      invocationCount: 2,
      bindings: [
        {name: 'prefixes', view: rowPrefixes, type: 'u32', access: 'read'},
        {name: 'captured', view: prefixes.view, type: 'u32', access: 'read_write'}
      ],
      body: 'captured[index] = prefixes[index];'
    });
    graph.add({id: 'seeded-test', getCommandNodes: () => [seed, write, status, capture]});
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expectSaturatedTrajectorySplitReadback(
        {
          written: segmentOverflow
            ? await readUint32(tracks.buffer, 2)
            : await readUint32(rows.buffer, 4),
          prefixes: await readUint32(prefixes.buffer, 2),
          count: await readUint32(count.buffer, 1),
          required: await readUint32(required.buffer, 1),
          overflow: await readUint32(overflow.buffer, 1)
        },
        segmentOverflow
      );
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  }
});
