// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTrajectoryThinning} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {expect, it} from 'vitest';
import {GPUTrajectoryThinningOperation} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning-operation';
import {getGPUTrajectoryThinningParameterValues} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/gpu-trajectory-thinning';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('writes adjacent count and candidate subviews without touching their neighbor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const status = createOutputBuffer(device, 3);
  status.write(Uint32Array.of(77, 88, 99));
  const rows = createOutputBuffer(device, 4);
  const offsets = createOutputBuffer(device, 2);
  const overflow = createOutputBuffer(device, 1);
  const buffers = [status, rows, offsets, overflow];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format);
  };
  const storage = importGraphBuffer(graph, 'status', status, 'uint32', 3);
  graph.add(
    new GPUTrajectoryThinning({
      mode: 'minimum-time',
      maximumSearchRows: 1,
      positions: input('positions', new Float32Array(8), 'float32x2'),
      timestamps: input('times', Float32Array.of(0, 1, 2, 3), 'float32'),
      trackOffsets: input('tracks', Uint32Array.of(0, 4), 'uint32'),
      parameters: input('parameters', Float32Array.of(10, 0, 0, 0), 'float32'),
      output: {
        rowIds: importGraphBuffer(graph, 'rows', rows, 'uint32', 4),
        trackOffsets: importGraphBuffer(graph, 'out-offsets', offsets, 'uint32', 2),
        status: {
          count: graph.createDataView(storage.buffer, {format: 'uint32', length: 1}),
          candidateOverflow: graph.createDataView(storage.buffer, {
            format: 'uint32',
            byteOffset: 4,
            length: 1
          }),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status, 3)).toEqual([3, 1, 99]);
    expect((await readUint32(rows, 4)).slice(0, 3)).toEqual([0, 2, 3]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('thins independent five-row tracks and publishes their closed-form offsets in parallel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const trackCount = 65537;
  const rowCount = trackCount * 5;
  const expectedRows = Array.from({length: trackCount}, (_, track) => [
    track * 5,
    track * 5 + 2,
    track * 5 + 4
  ]).flat();
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: new Float32Array(rowCount * 2)},
      timestamps: {
        format: 'float32',
        values: Float32Array.from({length: rowCount}, (_, row) => row % 5)
      },
      trackOffsets: {
        format: 'uint32',
        values: Uint32Array.from({length: trackCount + 1}, (_, track) => track * 5)
      },
      parameters: {
        format: 'float32',
        values: getGPUTrajectoryThinningParameterValues({tolerance: 2})
      }
    },
    operations: [
      new GPUTrajectoryThinningOperation({
        mode: 'minimum-time',
        positions: 'positions',
        timestamps: 'timestamps',
        trackOffsets: 'trackOffsets',
        parameters: 'parameters',
        rowCount,
        trackCount,
        capacity: expectedRows.length,
        output: {
          rowIds: 'rows',
          trackOffsets: 'offsets',
          status: {count: 'count', overflow: 'overflow'}
        }
      })
    ],
    outputs: ['rows', 'offsets', 'count', 'overflow']
  });
  expect(Array.from(result['rows'])).toEqual(expectedRows);
  expect(Array.from(result['offsets'])).toEqual(
    Array.from({length: trackCount + 1}, (_, track) => track * 3)
  );
  expect(Array.from(result['count'])).toEqual([expectedRows.length]);
  expect(Array.from(result['overflow'])).toEqual([0]);
}, 120000);
