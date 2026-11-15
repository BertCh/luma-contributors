// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addLandscapeMetricsRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/landscape-metrics';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';

// A square of one-cell patches has hand-derived metrics. Reconciliation makes it one square
// patch, with the same area and adjacency but a different mesh and perimeter. Reversed IDs put
// the minimum root at the far corner, and unused rows exercise the sort's empty segments.
it.each([
  false,
  true
])('landscape measures 16,641 patches with rectangular cells (reconcile=%s)', async reconcilePatches => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `landscape-many-patches-${reconcilePatches}`);
  const size = 129;
  const cellCount = size * size;
  const patchCount = cellCount + 5;
  const labels = Uint32Array.from({length: cellCount}, (_, index) => cellCount - 1 - index);
  const classes = new Uint32Array(patchCount);
  classes[cellCount + 1] = 0xffffffff;
  classes[cellCount + 3] = 1;
  try {
    const patches = fixture.output('patch-metrics', 'float32x4', patchCount);
    const categories = fixture.output('class-metrics', 'float32x4', 3);
    const adjacency = fixture.output('adjacency', 'uint32', 9);
    const aggregation = fixture.output('aggregation', 'float32', 3);
    const landscape = fixture.output('landscape', 'float32x4', 1);
    const roots = fixture.output('roots', 'uint32', patchCount);
    const reconciled = fixture.output('reconciled', 'uint32', cellCount);
    addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', labels, 'uint32', cellCount),
      patchClasses: fixture.input('classes', classes, 'uint32', patchCount),
      width: size,
      height: size,
      classCount: 3,
      cellSize: [2, 3],
      connectivity: 4,
      reconcilePatches,
      output: {
        patchMetrics: patches.view,
        classMetrics: categories.view,
        classAdjacency: adjacency.view,
        aggregation: aggregation.view,
        landscape: landscape.view,
        ...(reconcilePatches ? {patchParents: roots.view, reconciledPatches: reconciled.view} : {})
      }
    });
    fixture.run();
    const patchValues = await fixture.readFloat32(patches, patchCount * 4);
    const directedSides = 4 * size * (size - 1);
    if (reconcilePatches) {
      expect(patchValues.slice(0, 4)).toEqual([6 * cellCount, 10 * size, cellCount, directedSides]);
      expect(patchValues.slice(4)).toEqual(new Array((patchCount - 1) * 4).fill(0));
      expect(await fixture.readUint32(reconciled, cellCount)).toEqual(new Array(cellCount).fill(0));
      expect(await fixture.readUint32(roots, patchCount)).toEqual([
        ...new Array(cellCount).fill(0),
        ...Array.from({length: 5}, (_, index) => cellCount + index)
      ]);
    } else {
      for (let owner = 0; owner < cellCount; owner++) {
        const cell = cellCount - 1 - owner;
        const column = cell % size;
        const row = Math.floor(cell / size);
        const neighbors =
          Number(column > 0) + Number(column < size - 1) + Number(row > 0) + Number(row < size - 1);
        expect(patchValues.slice(owner * 4, owner * 4 + 4)).toEqual([6, 10, 1, neighbors]);
      }
      expect(patchValues.slice(cellCount * 4)).toEqual(new Array(20).fill(0));
    }
    expect(await fixture.readUint32(adjacency, 9)).toEqual([directedSides, 0, 0, 0, 0, 0, 0, 0, 0]);
    const classValues = await fixture.readFloat32(categories, 12);
    expect(classValues.slice(0, 3)).toEqual([
      6 * cellCount,
      reconcilePatches ? 10 * size : 10 * cellCount,
      reconcilePatches ? 6 * cellCount : 6
    ]);
    expect(classValues[3]).toBeCloseTo(reconcilePatches ? 100 : 0, 4);
    for (const offset of [4, 8]) {
      expect(classValues.slice(offset, offset + 3)).toEqual([0, 0, 0]);
      expect(classValues[offset + 3]).toBeNaN();
    }
    const aggregates = await fixture.readFloat32(aggregation, 3);
    expect(aggregates[0]).toBeCloseTo(100, 4);
    expect(aggregates[1]).toBeNaN();
    expect(aggregates[2]).toBeNaN();
    const summary = await fixture.readFloat32(landscape, 4);
    expect(summary[0]).toBeNaN();
    expect(summary.slice(1)).toEqual([reconcilePatches ? 6 * cellCount : 6, 6 * cellCount, 1]);
  } finally {
    fixture.destroy();
  }
}, 120_000);

// Boundary/background switches must remain independent, including for excluded patch rows.
// Rows: [patch 0, excluded patch 1, patch 2, background]; patch 3 has no cells.
it.each([
  true,
  false
])('landscape preserves background contacts with domain boundaries=%s', async countBoundary => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `landscape-background-${countBoundary}`);
  try {
    const patches = fixture.output('patch-metrics', 'float32x4', 4);
    addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', new Uint32Array([0, 1, 2, 0xffffffff]), 'uint32', 4),
      patchClasses: fixture.input('classes', new Uint32Array([0, 1, 0, 0]), 'uint32', 4),
      width: 4,
      height: 1,
      classCount: 2,
      cellSize: [2, 3],
      connectivity: 4,
      backgroundClass: 1,
      countBackground: false,
      countBoundary,
      output: {patchMetrics: patches.view}
    });
    fixture.run();
    const values = await fixture.readFloat32(patches, 16);
    // Vertical domain contacts total 4; patch 0 also has a left domain contact of 3.
    expect(values.slice(0, 4)).toEqual([6, countBoundary ? 7 : 0, 1, 0]);
    // Excluded patch has valid neighbors on both sides (6), plus domain contacts (4).
    expect(values.slice(4, 8)).toEqual([6, countBoundary ? 10 : 6, 1, 2]);
    expect(values.slice(8, 12)).toEqual([6, countBoundary ? 4 : 0, 1, 0]);
    expect(values.slice(12)).toEqual([0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  4, 8
] as const)('landscape seam roots respect %s-connectivity and retain unused IDs', async connectivity => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `landscape-diagonal-${connectivity}`);
  try {
    const background = 0xffffffff;
    const labels = [8, background, 4, background, 6, background, 2, background, 0];
    const parents = fixture.output('parents', 'uint32', 10);
    const reconciled = fixture.output('reconciled', 'uint32', 9);
    addLandscapeMetricsRecipe(fixture.graph, {
      patches: fixture.input('patches', new Uint32Array(labels), 'uint32', 9),
      patchClasses: fixture.input('classes', new Uint32Array(10), 'uint32', 10),
      width: 3,
      height: 3,
      classCount: 1,
      cellSize: [1, 1],
      connectivity,
      reconcilePatches: true,
      output: {patchParents: parents.view, reconciledPatches: reconciled.view}
    });
    fixture.run();
    // Only the center and four corners are occupied. Four-connectivity leaves them isolated;
    // eight-connectivity connects every occupied cell through the center to corner ID 0.
    expect(await fixture.readUint32(parents, 10)).toEqual(
      Array.from({length: 10}, (_, owner) => (connectivity === 8 && owner % 2 === 0 ? 0 : owner))
    );
    expect(await fixture.readUint32(reconciled, 9)).toEqual(
      labels.map(owner => (connectivity === 8 && owner !== background ? 0 : owner))
    );
  } finally {
    fixture.destroy();
  }
});
