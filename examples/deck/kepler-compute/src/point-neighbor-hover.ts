// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ScatterplotLayer} from '@deck.gl/layers';
import type {Color, Layer} from '@deck.gl/core';

export const MAXIMUM_NEIGHBOR_HOVER_ROWS = 4096;
/** Native dims the whole point layer to this opacity while an object is hovered with neighbor display. */
export const NEIGHBOR_HOVER_DIMMED_OPACITY = 0.2;

type ColumnReader = {get(rowIndex: number): unknown};
export type PointNeighborSource = {getChild(name: string): ColumnReader | null};
export type PointNeighborColumns = {
  lng: string;
  lat: string;
  altitude?: string;
  neighbors?: string | null;
};

function toNumberList(value: unknown): number[] {
  if (!value) return [];
  const list =
    typeof (value as {toArray?: unknown}).toArray === 'function'
      ? Array.from((value as {toArray(): ArrayLike<number | bigint>}).toArray())
      : Array.isArray(value)
        ? value
        : [];
  return list.map(Number).filter(Number.isInteger);
}

/**
 * Native getNeighbors: the neighbors column holds a list of source row indices for the hovered row.
 * Returns positions of the neighbors followed by the hovered row, skipping rows with invalid
 * coordinates; reading is bounded so one hover never decodes an unbounded list.
 */
export function readPointNeighborPositions(
  source: PointNeighborSource,
  hoveredRow: number,
  columns: PointNeighborColumns
): [number, number, number][] {
  const longitudes = source.getChild(columns.lng);
  const latitudes = source.getChild(columns.lat);
  const altitudes = columns.altitude ? source.getChild(columns.altitude) : null;
  if (!longitudes || !latitudes) return [];
  const neighborColumn = columns.neighbors ? source.getChild(columns.neighbors) : null;
  const rows = neighborColumn
    ? toNumberList(neighborColumn.get(hoveredRow)).slice(0, MAXIMUM_NEIGHBOR_HOVER_ROWS)
    : [];
  const positions: [number, number, number][] = [];
  for (const rowIndex of [...rows, hoveredRow]) {
    if (rowIndex < 0) continue;
    const longitude = Number(longitudes.get(rowIndex));
    const latitude = Number(latitudes.get(rowIndex));
    const altitude = altitudes ? Number(altitudes.get(rowIndex)) : 0;
    // Null and nonfinite coordinates are hidden, as in the resident point upload.
    if (!Number.isFinite(longitude + latitude)) continue;
    positions.push([longitude, latitude, Number.isFinite(altitude) ? altitude : 0]);
  }
  return positions;
}

/**
 * Native hover overlay (`<id>-hovered`): a scatterplot of the neighbors plus the hovered point
 * with the layer's stroke/fill settings. Extension: per-row color and radius channels are not
 * read back, so the overlay uses the highlight color (native default) or the layer color and a
 * constant radius; with a size channel the radius is the radius scale.
 */
export function makePointNeighborHoverLayer(options: {
  id: string;
  positions: readonly [number, number, number][];
  highlightColor: Color;
  showHighlightColor: boolean;
  layerColor: Color;
  strokeColor?: Color;
  filled: boolean;
  stroked: boolean;
  thickness: number;
  billboard: boolean;
  radiusScale: number;
  radiusMaxPixels: number;
  visible: boolean;
}): Layer {
  const fill = options.showHighlightColor ? options.highlightColor : options.layerColor;
  const line = options.showHighlightColor
    ? options.highlightColor
    : (options.strokeColor ?? options.layerColor);
  return new ScatterplotLayer<[number, number, number]>({
    id: `${options.id}-hovered`,
    data: options.positions as [number, number, number][],
    pickable: false,
    visible: options.visible,
    wrapLongitude: true,
    getPosition: position => position,
    getRadius: 1,
    getFillColor: fill,
    getLineColor: line,
    getLineWidth: 1,
    lineWidthUnits: 'pixels',
    lineWidthScale: options.thickness,
    radiusUnits: 'meters',
    radiusScale: options.radiusScale,
    radiusMaxPixels: options.radiusMaxPixels,
    billboard: options.billboard,
    filled: options.filled,
    stroked: options.stroked,
    parameters: {depthCompare: 'always', depthWriteEnabled: false}
  });
}
