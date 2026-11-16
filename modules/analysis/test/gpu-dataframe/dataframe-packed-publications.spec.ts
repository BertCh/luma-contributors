// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {GPUTimeWindowFilter} from '../../src/dataframe-analysis/time-window-filter/gpu-time-window-filter';
import {getGPUTimeWindowFilterParameterValues} from '../../src/dataframe-analysis/time-window-filter/time-window-parameters';
import {GPUValueCube} from '../../src/dataframe-analysis/value-cube/gpu-value-cube';
import {GPUKeyJoin} from '../../src/dataframe-analysis/key-join/gpu-key-join';
import {GPUColumnProfile} from '../../src/dataframe-analysis/column-profile/gpu-column-profile';
import {
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT
} from '../../src/dataframe-analysis/column-profile/column-profile-parameters';
import {GPUTemporalReduction} from '../../src/dataframe-analysis/temporal-reduction/gpu-temporal-reduction';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

it.each([
  'value cube',
  'inner key join',
  'column profile',
  'temporal reduction',
  'time window'
] as const)('%s publishes packed writable fields without mixed storage usage', async family => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: number[]
  ): GraphDataView<Format> => {
    const buffer = createInputBuffer(
      device,
      format === 'float32' ? new Float32Array(values) : new Uint32Array(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${serial++}`, buffer, format, values.length);
  };
  const storage = createOutputBuffer(device, 1024);
  buffers.push(storage);
  // Import this physical buffer once, then address each writable field through a subview.
  const packed = importGraphBuffer(graph, 'packed', storage, 'uint32', 1024);
  const field = <Format extends 'float32' | 'uint32'>(
    format: Format,
    byteOffset: number,
    length = 1
  ) => graph.createDataView(packed.buffer, {format, byteOffset, length});
  if (family === 'value cube') {
    graph.add(
      new GPUValueCube({
        cells: input('uint32', [0, 0]),
        periods: input('uint32', [0, 0]),
        values: input('float32', [1, 3]),
        cellCount: 1,
        periodCount: 1,
        output: {
          counts: field('uint32', 0),
          sums: field('float32', 256),
          means: field('float32', 512)
        }
      })
    );
  } else if (family === 'inner key join') {
    graph.add(
      new GPUKeyJoin({
        kind: 'inner',
        leftKeys: input('uint32', [1]),
        rightKeys: input('uint32', [1, 2]),
        gather: [{column: input('float32', [7, 9]), output: field('float32', 1280)}],
        output: {
          rightRows: field('uint32', 0),
          matched: field('uint32', 256),
          rows: {
            ids: field('uint32', 512),
            status: {count: field('uint32', 768), overflow: field('uint32', 1024)}
          }
        }
      })
    );
  } else if (family === 'column profile') {
    graph.add(
      new GPUColumnProfile({
        columns: [{values: input('float32', [1, 3])}],
        hyperLogLogPrecision: 4,
        histogramBinCount: 1,
        output: {
          statistics: field('float32', 0, GPU_COLUMN_PROFILE_STATISTIC_COUNT),
          hyperLogLogRegisters: field('uint32', 256, 16),
          counts: field('uint32', 512, 2)
        }
      })
    );
  } else if (family === 'time window') {
    graph.add(
      new GPUTimeWindowFilter({
        timestamps: input('float32', [0, 1]),
        window: input(
          'float32',
          Array.from(getGPUTimeWindowFilterParameterValues({start: 0, end: 1}))
        ),
        output: {
          outputMask: field('uint32', 0, 2),
          ids: field('uint32', 256),
          status: {
            count: field('uint32', 512),
            overflow: field('uint32', 768),
            requiredCount: field('uint32', 1024)
          }
        }
      })
    );
  } else {
    graph.add(
      new GPUTemporalReduction({
        cellIds: input('uint32', [0, 0]),
        timestamps: input('float32', [0, 1]),
        values: input('float32', [1, 3]),
        parameters: input('float32', [0, 10]),
        cellCount: 1,
        bucketCount: 1,
        output: {
          counts: field('uint32', 0),
          first: field('float32', 256),
          last: field('float32', 512),
          minimums: field('float32', 768),
          maximums: field('float32', 1024),
          occupiedSlots: {
            ids: field('uint32', 1280),
            status: {count: field('uint32', 1536), overflow: field('uint32', 1792)}
          }
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const floats = await readFloat32(storage, 1024);
    const words = await readUint32(storage, 1024);
    if (family === 'value cube') {
      expect(words[0]).toBe(2);
      expect(floats[64]).toBe(4);
      expect(floats[128]).toBe(2);
    } else if (family === 'inner key join') {
      expect([words[0], words[64], words[128], words[192], words[256]]).toEqual([0, 1, 0, 1, 0]);
      expect(floats[320]).toBe(7);
    } else if (family === 'column profile') {
      expect(floats[GPU_COLUMN_PROFILE_STATISTIC.mean]).toBe(2);
      expect(floats[GPU_COLUMN_PROFILE_STATISTIC.variance]).toBe(1);
      expect(words.slice(128, 130)).toEqual([2, 0]);
    } else if (family === 'time window') {
      expect([words[0], words[1], words[64], words[128], words[192], words[256]]).toEqual([
        1, 1, 0, 1, 1, 2
      ]);
    } else {
      expect(words[0]).toBe(2);
      expect([floats[64], floats[128], floats[192], floats[256]]).toEqual([1, 3, 1, 3]);
      expect([words[320], words[384], words[448]]).toEqual([0, 1, 0]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
