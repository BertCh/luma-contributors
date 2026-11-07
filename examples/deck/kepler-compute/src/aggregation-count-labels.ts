// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUPointDensityHexagonCenter} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {GPUKeplerAggregationLayout} from '@luma-contributors/kepler';
import {worldToLngLat} from '@math.gl/web-mercator';

/** Native `renderCountLabels` datum: the cell's geographic center and its point count. */
export type AggregationCountLabel = {position: [number, number]; count: number};

/** Center of one cell in the rebased metric frame (same formulas as the resident footprints). */
export function getAggregateCellCenter(
  layout: GPUKeplerAggregationLayout,
  cellId: number
): [number, number] {
  const column = cellId % layout.gridSize[0];
  const row = Math.floor(cellId / layout.gridSize[0]);
  return layout.kind === 'grid'
    ? [
        layout.bounds[0] + (column + 0.5) * layout.cellSize![0],
        layout.bounds[1] + (row + 0.5) * layout.cellSize![1]
      ]
    : getGPUPointDensityHexagonCenter(
        column,
        row,
        layout.bounds[0],
        layout.bounds[1],
        layout.radius!
      );
}

/**
 * Native grid/hexagon count labels: one label per nonempty cell that survives the percentile and
 * elevation tests, anchored at the cell center (the base of an extruded cell, where native draws
 * them). `visible` is the style kernel's per-cell visibility, absent on the fast path.
 */
export function getAggregateCountLabels(
  layout: GPUKeplerAggregationLayout,
  counts: ArrayLike<number>,
  visible: ArrayLike<number> | undefined,
  commonOrigin: readonly [number, number, number],
  commonUnitsPerMeter: number
): AggregationCountLabel[] {
  const labels: AggregationCountLabel[] = [];
  for (let cellId = 0; cellId < layout.cellCount; cellId++) {
    if (!counts[cellId] || (visible && !visible[cellId])) continue;
    const center = getAggregateCellCenter(layout, cellId);
    const [longitude, latitude] = worldToLngLat([
      commonOrigin[0] + center[0] * commonUnitsPerMeter,
      commonOrigin[1] + center[1] * commonUnitsPerMeter
    ]);
    labels.push({position: [longitude, latitude], count: counts[cellId]});
  }
  return labels;
}
