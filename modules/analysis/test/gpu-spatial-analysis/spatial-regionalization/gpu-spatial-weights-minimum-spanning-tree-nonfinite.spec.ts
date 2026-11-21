// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT,
  GPUSpatialWeightsMinimumSpanningTree
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {computeKruskalForest} from './spatial-regionalization-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

/** 3 x 3 rook lattice in CSR form. */
function createLattice() {
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < 9; row++) {
    const x = row % 3;
    for (const candidate of [row - 3, x > 0 ? row - 1 : -1, x < 2 ? row + 1 : -1, row + 3]) {
      if (candidate >= 0 && candidate < 9) neighbors.push(candidate);
    }
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

// Distinct edge costs, so the forest does not depend on tie rules.
const BASE_VALUES = [0.3, 2.9, 1.1, 4.7, 0.5, 3.35, 7.2, 5.5, 0.9];

it.each([
  {name: 'one NaN in the first row', nonFinite: {0: NaN}},
  {name: 'a NaN in the centre row and an Infinity', nonFinite: {4: NaN, 8: Infinity}}
])('skips non-finite entries when standardizing: $name', async ({nonFinite}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = [...BASE_VALUES];
  for (const [row, value] of Object.entries(nonFinite)) values[Number(row)] = value;
  const lattice = createLattice();
  const rig = new WeightsRig(device);
  const weights = rig.uploadWeights(
    {
      offsets: lattice.offsets,
      neighbors: lattice.neighbors,
      weights: lattice.neighbors.map(() => 1),
      distances: []
    },
    0
  );
  const flags = rig.output('uint32', lattice.neighbors.length);
  const labels = rig.output('uint32', 9);
  const standardized = rig.output('float32', 9);
  const status = rig.output('uint32', GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length);
  rig.graph.add(
    new GPUSpatialWeightsMinimumSpanningTree({
      weights,
      values: rig.input(new Float32Array(values), 'float32', 9),
      output: {
        treeEdgeFlags: flags.view,
        componentLabels: labels.view,
        standardizedValues: standardized.view,
        status: status.view
      }
    })
  );
  const compiled = rig.graph.compile();
  submitGraph(device, compiled, undefined);

  // CPU reference over the finite entries only (population standard deviation).
  const finite = values.filter(Number.isFinite);
  const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const deviation = Math.sqrt(
    finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length
  );
  const expected = values.map(value =>
    Number.isFinite(value) ? (value - mean) / deviation : value
  );
  const actual = await readFloat32(standardized.buffer, 9);
  expected.forEach((value, row) => {
    if (Number.isFinite(value)) expect(actual[row]).toBeCloseTo(value, 4);
    else expect(Number.isFinite(actual[row]), `row ${row} stays non-finite`).toBe(false);
  });

  // Links that touch a non-finite row cost Infinity and are chosen last; the forest still spans.
  const expectedSlots = computeKruskalForest(lattice.offsets, lattice.neighbors, (row, neighbor) =>
    Number.isFinite(expected[row]) && Number.isFinite(expected[neighbor])
      ? (expected[row] - expected[neighbor]) ** 2
      : Infinity
  );
  const treeFlags = await readUint32(flags.buffer, lattice.neighbors.length);
  expect(treeFlags.flatMap((flag, slot) => (flag ? [slot] : []))).toEqual(expectedSlots);
  expect(expectedSlots).toHaveLength(8);
  expect(Array.from(await readUint32(labels.buffer, 9))).toEqual(new Array(9).fill(0));
  expect(
    (await readUint32(status.buffer, 1))[GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.invalidCount]
  ).toBe(Object.keys(nonFinite).length);
  compiled.destroy();
  rig.destroy();
});
