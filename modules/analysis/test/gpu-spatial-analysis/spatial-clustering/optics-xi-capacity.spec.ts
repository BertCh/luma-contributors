// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createOPTICSXiNodes} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-xi';
import {
  GPU_OPTICS_XI_MAXIMUM_ROWS,
  type GPUOPTICSProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('xi extracts a plateau spanning the full 65536-row capacity and every dispatch batch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const rows = GPU_OPTICS_XI_MAXIMUM_ROWS;
  const input = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, values.length);
  };
  const output = <Format extends 'uint32' | 'uint32x2'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'uint32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const ordering = input(
    'ordering',
    'uint32',
    Uint32Array.from({length: rows}, (_, row) => row)
  );
  const reachabilityValues = new Float32Array(rows).fill(1);
  reachabilityValues[0] = Infinity;
  const reachability = input('reachability', 'float32', reachabilityValues);
  const predecessors = input(
    'predecessors',
    'uint32',
    Uint32Array.from({length: rows}, (_, row) => (row === 0 ? 0xffffffff : row - 1))
  );
  const status = input('status', 'uint32', Uint32Array.of(rows, 1, 0));
  const labels = output('labels', 'uint32', rows);
  const intervals = output('intervals', 'uint32x2', rows);
  const summary = output('summary', 'uint32', 2);
  // Only the extraction helper runs: it reads a completed ordering, not positions or cores.
  const props: GPUOPTICSProps = {
    positions: createTransientView(graph, 'unused-positions', 'float32x2', rows),
    minimumSamples: 2,
    maximumDistance: 10,
    extractionMethod: 'xi',
    output: {
      ordering,
      coreDistances: createTransientView(graph, 'unused-core', 'float32', rows),
      reachability,
      predecessors,
      labels: labels.view,
      clusterIntervals: intervals.view,
      clusterSummary: summary.view
    }
  };
  graph.add(createOPTICSXiNodes(graph, props, 'xi-capacity', status));
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Infinity -> unit plateau -> appended infinity gives exactly one outer interval.
    // Hand-derived result also checked with sklearn.cluster_optics_xi at 65,536 rows.
    expect(await readUint32(summary.buffer, 2)).toEqual([1, 0]);
    const actualIntervals = await readUint32(intervals.buffer, rows * 2);
    expect(actualIntervals.slice(0, 2)).toEqual([0, rows - 1]);
    expect(actualIntervals.slice(2).every(value => value === 0xffffffff)).toBe(true);
    expect((await readUint32(labels.buffer, rows)).every(value => value === 0)).toBe(true);
    // Re-encode to verify that hierarchy, state and the dispatch gate reset each frame.
    submitGraph(device, compiled, undefined);
    expect(await readUint32(summary.buffer, 2)).toEqual([1, 0]);
    expect((await readUint32(labels.buffer, rows)).every(value => value === 0)).toBe(true);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
