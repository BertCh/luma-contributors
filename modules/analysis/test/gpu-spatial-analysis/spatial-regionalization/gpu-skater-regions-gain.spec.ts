// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUSkaterRegionsParameterValues,
  GPU_SKATER_REGIONS_PARAMETER_LENGTH,
  GPUSkaterRegions
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

// Math review point-process-clustering-02: the SKATER cut gain inside a quiet region whose mean is
// far from zero. The tree is a path given directly (no minimum spanning tree node), so the only
// arithmetic under test is the gain. Expected cuts and gains are float64 within-region sums of
// squares about float64 means, `SSD(region) - SSD(side A) - SSD(side B)`, over the float32 inputs:
// the definition, not the kernel's between-side form.

/** Path `0 - 1 - ... - (rows - 1)` in CSR form; the slot of edge `(row, row + 1)` in row `row`. */
function createPath(rows: number) {
  const offsets = [0];
  const neighbors: number[] = [];
  const treeEdgeFlags: number[] = [];
  for (let row = 0; row < rows; row++) {
    if (row > 0) {
      neighbors.push(row - 1);
      treeEdgeFlags.push(0);
    }
    if (row + 1 < rows) {
      neighbors.push(row + 1);
      // The flag sits on the lower-row slot of each tree edge.
      treeEdgeFlags.push(1);
    }
    offsets.push(neighbors.length);
  }
  const getEdgeSlot = (row: number) => offsets[row] + (row > 0 ? 1 : 0);
  return {offsets, neighbors, treeEdgeFlags, getEdgeSlot};
}

/** Within-segment sum of squares of `values[start, end)` about its own mean, in float64. */
function getSegmentSumOfSquares(values: Float32Array, start: number, end: number): number {
  let mean = 0;
  for (let row = start; row < end; row++) mean += values[row];
  mean /= end - start;
  let sumOfSquares = 0;
  for (let row = start; row < end; row++) sumOfSquares += (values[row] - mean) ** 2;
  return sumOfSquares;
}

/** Greedy SKATER on a path in float64: the cut rows (first row of the later side) and gains. */
function getPathCuts(values: Float32Array, regionCount: number) {
  const boundaries = [0, values.length];
  const cuts: {row: number; gain: number}[] = [];
  while (boundaries.length - 1 < regionCount) {
    let best = {row: -1, gain: -1};
    for (let region = 0; region + 1 < boundaries.length; region++) {
      const start = boundaries[region];
      const end = boundaries[region + 1];
      const whole = getSegmentSumOfSquares(values, start, end);
      for (let row = start + 1; row < end; row++) {
        const gain =
          whole -
          getSegmentSumOfSquares(values, start, row) -
          getSegmentSumOfSquares(values, row, end);
        if (gain > best.gain) best = {row, gain};
      }
    }
    cuts.push(best);
    boundaries.push(best.row);
    boundaries.sort((first, second) => first - second);
  }
  return cuts;
}

/**
 * Complete binary tree in CSR form: the parent of `row` is `(row - 1) >> 1`, so the neighbours of
 * a row are its parent and then its children, ascending. `getEdgeSlot(child)` is the slot of the
 * edge to `child` in its parent's row.
 */
function createBinaryTree(rows: number) {
  const offsets = [0];
  const neighbors: number[] = [];
  const treeEdgeFlags: number[] = [];
  const childSlots = new Map<number, number>();
  for (let row = 0; row < rows; row++) {
    if (row > 0) {
      neighbors.push((row - 1) >> 1);
      treeEdgeFlags.push(0);
    }
    for (const child of [2 * row + 1, 2 * row + 2]) {
      if (child < rows) {
        childSlots.set(child, neighbors.length);
        neighbors.push(child);
        treeEdgeFlags.push(1);
      }
    }
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors, treeEdgeFlags, getEdgeSlot: (child: number) => childSlots.get(child)};
}

type TreeLayout = {
  offsets: number[];
  neighbors: number[];
  treeEdgeFlags: number[];
  getEdgeSlot: (row: number) => number | undefined;
};

async function runPathSkater(values: Float32Array, regionCount: number) {
  return runTreeSkater(createPath(values.length), values, regionCount);
}

async function runTreeSkater(path: TreeLayout, values: Float32Array, regionCount: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = values.length;
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(
    {
      offsets: path.offsets,
      neighbors: path.neighbors,
      weights: path.neighbors.map(() => 1),
      distances: []
    },
    0
  );
  const labels = rig.output('uint32', rows);
  const cutEdges = rig.output('uint32', regionCount - 1);
  const cutGains = rig.output('float32', regionCount - 1);
  rig.graph.add(
    new GPUSkaterRegions({
      weights,
      treeEdgeFlags: rig.input(
        new Uint32Array(path.treeEdgeFlags),
        'uint32',
        path.treeEdgeFlags.length
      ),
      componentLabels: rig.input(new Uint32Array(rows), 'uint32', rows),
      values: rig.input(values, 'float32', rows),
      maximumRegionCount: regionCount,
      parameters: rig.input(
        getGPUSkaterRegionsParameterValues({regionCount}),
        'uint32',
        GPU_SKATER_REGIONS_PARAMETER_LENGTH
      ),
      output: {labels: labels.view, cutEdges: cutEdges.view, cutGains: cutGains.view}
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    labels: Array.from(await readUint32(labels.buffer, rows)),
    cutEdges: Array.from(await readUint32(cutEdges.buffer, regionCount - 1)),
    cutGains: Array.from(await readFloat32(cutGains.buffer, regionCount - 1)),
    getEdgeSlot: path.getEdgeSlot
  };
  compiled.destroy();
  rig.destroy();
  return result;
}

it('cuts two plateaus 0.001 apart at a mean of 2 where they meet, with the true gain', async () => {
  // 500 rows at 2.000 and 500 at 2.001. A cut after k <= 500 rows gains 0.25 k / (1000 (1000 - k))
  // in units of (0.001)^2 ... that is 2.5e-4 at k = 500 and less everywhere else. The sum of
  // squares of the region is 4002, whose float32 last place (2.4e-4) is the whole gain.
  const values = Float32Array.from({length: 1000}, (_, row) => (row < 500 ? 2 : 2.001));
  const result = await runPathSkater(values, 2);
  const expected = getPathCuts(values, 2);
  expect(expected[0].row).toBe(500);
  expect(result.cutEdges[0]).toBe(result.getEdgeSlot(499));
  expect(Math.abs(result.cutGains[0] - expected[0].gain)).toBeLessThan(0.01 * expected[0].gain);
  expect(new Set(result.labels.slice(0, 500))).toEqual(new Set([0]));
  expect(new Set(result.labels.slice(500))).toEqual(new Set([500]));
});

it('finds the three level changes of a quiet series at a mean of 3', async () => {
  // Four levels 0.01 apart with noise of 0.001 around 3: every true gain is far above the noise,
  // and far below the float32 last place of the sum of squares (4096 * 9 = 36864, last place 4e-3).
  const noise = (row: number) => {
    const hashed = Math.sin((row + 1) * 12.9898) * 43758.5453;
    return 2 * (hashed - Math.floor(hashed)) - 1;
  };
  const values = Float32Array.from(
    {length: 4096},
    (_, row) => 3 + 0.01 * Math.floor(row / 1024) + 0.001 * noise(row)
  );
  const result = await runPathSkater(values, 4);
  const expected = getPathCuts(values, 4);
  expect(expected.map(cut => cut.row).sort((first, second) => first - second)).toEqual([
    1024, 2048, 3072
  ]);
  expected.forEach((cut, step) => {
    expect(result.cutEdges[step], `step ${step}`).toBe(result.getEdgeSlot(cut.row - 1));
    expect(Math.abs(result.cutGains[step] - cut.gain), `step ${step}`).toBeLessThan(
      0.01 * cut.gain
    );
  });
});

it('sums a subtree of 32,767 rows at 3.0001 to its last place', async () => {
  // Complete binary tree of 131,071 rows rooted at row 0. The subtree below row 5 (32,767 rows)
  // holds 3.0001, everything else 3. A plain float32 serial sum drops the 0.0001 of every row once
  // the running sum passes 32,768 (last place 1/256, and 3.0001 is 768 of them plus 0.0001), so
  // the subtree mean would read 3 + d / 3 and the gain a ninth of the truth. By hand, with
  // d = fl(3.0001) - 3, the candidate gains are
  //   edge to row 5:  32767 * 98304 / 131071 * d^2             = 24575.6 d^2  (the best)
  //   edge to row 2:  65535 * 65536 / 131071 * (d 32767/65535)^2 =  8191.6 d^2
  //   edge to row 11: 16383 * 114688 / 131071 * (d 98304/114688)^2 = 10532 d^2
  // The remaining error is the last place of the region sum (393,216, last place 1/32) spread
  // over the 98,304 rows of the other side: 2e-7 on a mean difference of 1e-4.
  const rows = 131071;
  const tree = createBinaryTree(rows);
  const values = new Float32Array(rows).fill(3);
  const pending = [5];
  for (let cursor = 0; cursor < pending.length; cursor++) {
    const row = pending[cursor];
    values[row] = 3.0001;
    for (const child of [2 * row + 1, 2 * row + 2]) {
      if (child < rows) pending.push(child);
    }
  }
  expect(pending.length).toBe(32767);
  const result = await runTreeSkater(tree, values, 2);
  const difference = Math.fround(3.0001) - 3;
  const expectedGain = ((32767 * 98304) / 131071) * difference * difference;
  expect(result.cutEdges[0]).toBe(tree.getEdgeSlot(5));
  expect(Math.abs(result.cutGains[0] - expectedGain)).toBeLessThan(0.02 * expectedGain);
});
