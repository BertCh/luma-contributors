// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getPairStatisticsSegmentTotalNodes} from '../../../src/dataframe-analysis/pair-statistics/pair-statistics-cell-neighbors';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('three-level segment totals preserve exact counts, saturation flags and changed frames', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 65537;
  const floats = Float32Array.from({length: 3 * rows}, (_, index) =>
    index < rows ? 1 : index < 2 * rows ? ((index - rows) % 2 ? 3 : 1) : 0
  );
  const counts = new Uint32Array(3 * rows);
  counts[0] = 0xffffffff;
  counts[1] = 1;
  counts[rows] = 0xffffffff;
  counts.fill(1, 2 * rows);
  const floatInput = createInputBuffer(device, floats);
  const countInput = createInputBuffer(device, counts);
  const floatOutput = createOutputBuffer(device, 3);
  const countOutput = createOutputBuffer(device, 6);
  const graph = new GPUCommandGraph(device);
  const nodes = [
    ...getPairStatisticsSegmentTotalNodes(graph, {
      id: 'float',
      operation: 'PairSegmentTotals',
      rowCount: rows,
      segmentCount: 3,
      input: importGraphBuffer(graph, 'float-input', floatInput, 'float32', floats.length),
      output: importGraphBuffer(graph, 'float-output', floatOutput, 'float32', 3)
    }),
    ...getPairStatisticsSegmentTotalNodes(graph, {
      id: 'count',
      operation: 'PairSegmentTotals',
      rowCount: rows,
      segmentCount: 3,
      input: importGraphBuffer(graph, 'count-input', countInput, 'uint32', counts.length),
      output: importGraphBuffer(graph, 'count-output', countOutput, 'uint32', 6),
      detectOverflow: true
    })
  ];
  graph.add({getCommandNodes: () => nodes});
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(floatOutput, 3)).toEqual([rows, 2 * rows - 1, 0]);
    expect(await readUint32(countOutput, 6)).toEqual([0xffffffff, 0xffffffff, rows, 1, 0, 0]);
    floatInput.write(new Float32Array(3 * rows).fill(2));
    countInput.write(new Uint32Array(3 * rows).fill(1));
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(floatOutput, 3)).toEqual([2 * rows, 2 * rows, 2 * rows]);
    expect(await readUint32(countOutput, 6)).toEqual([rows, rows, rows, 0, 0, 0]);
  } finally {
    compiled.destroy();
    for (const buffer of [floatInput, countInput, floatOutput, countOutput]) buffer.destroy();
  }
});
