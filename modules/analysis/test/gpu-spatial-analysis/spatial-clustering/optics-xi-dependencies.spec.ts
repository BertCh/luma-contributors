// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createOPTICSXiNodes} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-xi';
import type {GPUOPTICSProps} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

/** Exercises extraction directly, so long predecessor suffixes need no quadratic OPTICS run. */
async function runXiExtraction(
  reachabilityValues: Float32Array,
  options: Pick<GPUOPTICSProps, 'xi' | 'predecessorCorrection' | 'minimumClusterSize'>,
  repeat = false
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rows = reachabilityValues.length;
  const buffers: Buffer[] = [];
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
  const labels = output('labels', 'uint32', rows);
  const intervals = output('intervals', 'uint32x2', rows);
  const summary = output('summary', 'uint32', 2);
  const props: GPUOPTICSProps = {
    ...options,
    positions: createTransientView(graph, 'unused-positions', 'float32x2', rows),
    minimumSamples: 2,
    minimumClusterSize: options.minimumClusterSize ?? 3,
    maximumDistance: 10,
    extractionMethod: 'xi',
    output: {
      ordering: input(
        'ordering',
        'uint32',
        Uint32Array.from({length: rows}, (_, rank) => rank)
      ),
      coreDistances: createTransientView(graph, 'unused-core', 'float32', rows),
      reachability: input('reachability', 'float32', reachabilityValues),
      // Every finite row has an earlier predecessor. In the suffix fixture it is
      // outside the candidate starting at 4, so it cannot end predecessor correction.
      predecessors: input(
        'predecessors',
        'uint32',
        Uint32Array.from(reachabilityValues, value => (Number.isFinite(value) ? 0 : 0xffffffff))
      ),
      status: input('status', 'uint32', Uint32Array.of(rows, 1, 0)),
      labels: labels.view,
      clusterIntervals: intervals.view,
      clusterSummary: summary.view
    }
  };
  graph.add(createOPTICSXiNodes(graph, props, 'xi-dependencies'));
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    if (repeat) submitGraph(device, compiled, undefined);
    const actualSummary = await readUint32(summary.buffer, 2);
    const actualIntervals = await readUint32(intervals.buffer, rows * 2);
    expect(actualIntervals.slice(actualSummary[0] * 2).every(value => value === 0xffffffff)).toBe(
      true
    );
    return {
      labels: await readUint32(labels.buffer, rows),
      intervals: actualIntervals.slice(0, actualSummary[0] * 2),
      summary: actualSummary
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

// Exact hand-derived intervals. Independently checked with sklearn._xi_cluster at xi=0,
// because the GPU complement of the legal positive xi below is exactly 1 in float32.
// The nine non-separator rows are also the OPTICS reachability of positions
// [0, 1, 1, 2, 2, 3, 3, 4, 4] on a line, minimumSamples=2, maximumDistance=1.
const CROSSING_INTERVALS = [0, 2, 0, 4, 3, 6, 0, 6, 5, 8, 3, 8, 0, 8];
const CROSSING_LABELS = [0, 0, 0, 1, 1, 1, 1, 0xffffffff, 0xffffffff, 0xffffffff];

for (const tileCount of [1, 6553]) {
  it(`xi preserves greedy labels across crossing intervals in ${tileCount * 10} rows`, async () => {
    const xi = 2 ** -30;
    expect(Math.fround(1 - xi)).toBe(1);
    const tile = Float32Array.of(Infinity, 1, 0, 1, 0, 1, 0, 1, 0, Infinity);
    const reachability = new Float32Array(tileCount * tile.length);
    for (let tileIndex = 0; tileIndex < tileCount; tileIndex++) {
      reachability.set(tile, tileIndex * tile.length);
    }
    const result = await runXiExtraction(reachability, {xi}, tileCount === 1);
    expect(result.summary).toEqual([(tileCount * CROSSING_INTERVALS.length) / 2, 0]);
    for (let tileIndex = 0; tileIndex < tileCount; tileIndex++) {
      const rowOffset = tileIndex * tile.length;
      const intervalOffset = tileIndex * CROSSING_INTERVALS.length;
      expect(
        result.intervals.slice(intervalOffset, intervalOffset + CROSSING_INTERVALS.length)
      ).toEqual(CROSSING_INTERVALS.map(rank => rank + rowOffset));
      expect(result.labels.slice(rowOffset, rowOffset + tile.length)).toEqual(
        CROSSING_LABELS.map(label => (label === 0xffffffff ? label : label + tileIndex * 2))
      );
    }
    // [0,4] is emitted before [3,6], but rejected because [0,2] already owns rows.
    // [5,8] is also rejected; its previously unlabelled rows 7 and 8 stay noise.
    // A pointwise earliest-interval or laminar-leaf reduction would change these labels.
  }, 600_000);
}

it('xi preserves the first capacity intervals and clears labels on quadratic hierarchy overflow', async () => {
  const rows = 256;
  const reachability = Float32Array.from({length: rows}, (_, rank) => (rank % 2 ? 0 : 1));
  reachability[0] = Infinity;
  const result = await runXiExtraction(reachability, {xi: 2 ** -30, minimumClusterSize: 2});
  // Upward rank 2*k+1 matches k+1 downward areas in reverse order. This plot has
  // 128*129/2 = 8,256 intervals, even though output capacity is only 256. An exact
  // replacement must preserve this ordered prefix without materializing every pair.
  const expectedIntervals: number[] = [];
  for (let finish = 1; finish < rows && expectedIntervals.length < rows * 2; finish += 2) {
    for (let start = finish - 1; start >= 0 && expectedIntervals.length < rows * 2; start -= 2) {
      expectedIntervals.push(start, finish);
    }
  }
  expect(result.summary).toEqual([rows, 1]);
  expect(result.intervals).toEqual(expectedIntervals);
  expect(result.labels.every(label => label === 0xffffffff)).toBe(true);
});

for (const predecessorCorrection of [false, true]) {
  it(`xi retains a row-sized predecessor suffix at 65536 rows (correction=${predecessorCorrection})`, async () => {
    const rows = 65536;
    const reachability = new Float32Array(rows).fill(1.5);
    reachability.set([Infinity, 4, 6, 1, 1.5, 0]);
    reachability[rows - 1] = 3;
    const result = await runXiExtraction(reachability, {xi: 2 ** -30, predecessorCorrection});
    // The float32 complement is exactly 1, so the area starting at 4 survives the
    // equal-height plateau. Correction retreats to rank 5 and rejects the final
    // candidate because finish < upwardStart. A replacement may reject early at
    // upwardStart, but cannot truncate the walk and publish the uncorrected interval.
    // These expectations were also checked with an independent last-eligible-rank
    // NumPy predicate reduction, retaining the GPU's no-direction rule for equal values.
    const expectedIntervals = predecessorCorrection
      ? [2, 5, 2, rows - 1, 0, rows - 1]
      : [2, 5, 4, rows - 1, 2, rows - 1, 0, rows - 1];
    expect(result.summary).toEqual([expectedIntervals.length / 2, 0]);
    expect(result.intervals).toEqual(expectedIntervals);
    expect(result.labels.slice(0, 2)).toEqual([0xffffffff, 0xffffffff]);
    expect(result.labels.slice(2, 6)).toEqual([0, 0, 0, 0]);
    expect(result.labels.slice(6).every(label => label === 0xffffffff)).toBe(true);
  }, 600_000);
}
