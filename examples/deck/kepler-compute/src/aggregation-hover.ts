// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUPointDensityHexagonCenter} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUKeplerAggregationLayout} from '@luma-contributors/kepler';

/** Kepler HIGHLIGH_COLOR_3D: the translucent white blended over a hovered extruded cell. */
export const AGGREGATION_EXTRUDED_HIGHLIGHT_COLOR: number[] = [255, 255, 255, 60];

/**
 * Native AggregationLayer sets `autoHighlight: visConfig.enable3d` with HIGHLIGH_COLOR_3D; flat
 * cells are not auto-highlighted and instead get a separate hovered-outline layer.
 */
export function getAggregationHighlightProps(extruded: boolean): {
  autoHighlight: boolean;
  highlightColor?: number[];
} {
  return extruded
    ? {autoHighlight: true, highlightColor: [...AGGREGATION_EXTRUDED_HIGHLIGHT_COLOR]}
    : {autoHighlight: false};
}

/**
 * Closed outline of one full cell (coverage 1) in the same rebased metric space as the resident
 * cell footprints, counterclockwise like `makeAggregateFootprints`.
 */
export function getAggregateCellOutline(
  layout: GPUKeplerAggregationLayout,
  cellId: number
): [number, number][] {
  const vertexCount = layout.kind === 'grid' ? 4 : 6;
  const column = cellId % layout.gridSize[0];
  const row = Math.floor(cellId / layout.gridSize[0]);
  const center =
    layout.kind === 'grid'
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
  const outline: [number, number][] = [];
  for (let vertex = 0; vertex <= vertexCount; vertex++) {
    const index = vertex % vertexCount;
    const angle = (index * Math.PI) / 3 + Math.PI / 2;
    const offset =
      layout.kind === 'grid'
        ? [
            (index === 0 || index === 3 ? -0.5 : 0.5) * layout.cellSize![0],
            (index < 2 ? -0.5 : 0.5) * layout.cellSize![1]
          ]
        : [Math.cos(angle) * layout.radius!, Math.sin(angle) * layout.radius!];
    outline.push([center[0] + offset[0], center[1] + offset[1]]);
  }
  return outline;
}

/** Native hovered-outline width: GeoJsonLayer `lineWidthScale: 8 * zoomFactor` meters. */
export function getAggregationOutlineWidth(zoomFactor: number): number {
  return 8 * zoomFactor;
}
