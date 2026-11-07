// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM} from '@deck.gl/core';

/**
 * Overlay for Dorling layouts in an arbitrary declared projected CRS, as a local tangent plane
 * anchored at a declared geographic origin (Deck METER_OFFSETS). The caller declares the CRS unit
 * size and the anchor; there is no automatic reprojection, so the overlay is accurate only while
 * the layout extent stays small relative to the Earth radius. Larger extents are refused.
 */
export type DeclaredProjectedCartogramOptions = {
  /** Meters per CRS unit (1 for metres, 0.3048006096 for US survey feet, 0.3048 for feet). */
  metersPerUnit: number;
  /** Plane coordinates [x, y] in CRS units that coincide with the geographic anchor. */
  planeOrigin: readonly [number, number];
  /** Geographic anchor [longitude, latitude] of planeOrigin. */
  anchor: readonly [number, number];
};

export const EARTH_RADIUS_METERS = 6378137;
/** Largest layout half-extent admitted (meters): about 0.03 percent scale error at 200 km. */
export const MAXIMUM_DECLARED_PLANE_HALF_EXTENT_METERS = 200000;

/** Relative length error of a flat tangent plane at a distance (sphere secant approximation). */
export function estimateTangentPlaneRelativeError(distanceMeters: number): number {
  const angle = distanceMeters / EARTH_RADIUS_METERS;
  return (angle * angle) / 6;
}

export function planDeclaredProjectedCartogram(
  options: DeclaredProjectedCartogramOptions,
  extent: {minimumX: number; minimumY: number; maximumX: number; maximumY: number}
): {ok: boolean; reason?: string; relativeError: number} {
  const {metersPerUnit, planeOrigin, anchor} = options;
  if (!(metersPerUnit > 0) || !Number.isFinite(metersPerUnit))
    return {
      ok: false,
      reason: 'Declared CRS unit size must be positive and finite',
      relativeError: 0
    };
  if (Math.abs(anchor[1]) > 85 || Math.abs(anchor[0]) > 180)
    return {
      ok: false,
      reason: 'Declared anchor must be a geographic position within 85 degrees latitude',
      relativeError: 0
    };
  const halfExtent =
    Math.max(
      Math.abs(extent.minimumX - planeOrigin[0]),
      Math.abs(extent.maximumX - planeOrigin[0]),
      Math.abs(extent.minimumY - planeOrigin[1]),
      Math.abs(extent.maximumY - planeOrigin[1])
    ) * metersPerUnit;
  const relativeError = estimateTangentPlaneRelativeError(halfExtent);
  if (!(halfExtent <= MAXIMUM_DECLARED_PLANE_HALF_EXTENT_METERS))
    return {
      ok: false,
      reason: 'Layout extends beyond the local-plane overlay limit; it remains table-only',
      relativeError
    };
  return {ok: true, relativeError};
}

/**
 * Layer props: positions are offset by planeOrigin in the shader via modelMatrix translation
 * (in meters), then scaled by metersPerUnit; radii scale by metersPerUnit as meters.
 */
export function getDeclaredProjectedCartogramLayerProps(
  options: DeclaredProjectedCartogramOptions
) {
  const {metersPerUnit, planeOrigin, anchor} = options;
  const tx = -planeOrigin[0] * metersPerUnit;
  const ty = -planeOrigin[1] * metersPerUnit;
  return {
    coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
    coordinateOrigin: [anchor[0], anchor[1], 0] as [number, number, number],
    modelMatrix: [
      metersPerUnit,
      0,
      0,
      0,
      0,
      metersPerUnit,
      0,
      0,
      0,
      0,
      metersPerUnit,
      0,
      tx,
      ty,
      0,
      1
    ],
    radiusUnits: 'meters' as const,
    radiusScale: metersPerUnit,
    radiusMinPixels: 0,
    radiusMaxPixels: Number.MAX_SAFE_INTEGER,
    billboard: false
  };
}

/**
 * Uniform-grid neighbor structure (counting sort by cell) for layouts above the 1,024-row solver
 * cap. Cell edge is twice the largest radius, so any overlapping pair lies in adjacent cells.
 * This is the CPU reference and sizing plan for a GPU grid pass; the solver cap is not lifted by it.
 */
export type DorlingNeighborGrid = {
  cellEdge: number;
  columns: number;
  rows: number;
  cellStart: Uint32Array;
  cellEntries: Uint32Array;
  maximumCellOccupancy: number;
};

export function buildDorlingNeighborGrid(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  radii: ArrayLike<number>
): DorlingNeighborGrid {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  let maximumRadius = 0;
  for (let index = 0; index < xs.length; index++) {
    minimumX = Math.min(minimumX, xs[index]);
    maximumX = Math.max(maximumX, xs[index]);
    minimumY = Math.min(minimumY, ys[index]);
    maximumY = Math.max(maximumY, ys[index]);
    maximumRadius = Math.max(maximumRadius, radii[index]);
  }
  const cellEdge = Math.max(2 * maximumRadius, Number.EPSILON);
  const columns = Math.max(1, Math.floor((maximumX - minimumX) / cellEdge) + 1);
  const rows = Math.max(1, Math.floor((maximumY - minimumY) / cellEdge) + 1);
  const cellOf = (index: number) =>
    Math.floor((ys[index] - minimumY) / cellEdge) * columns +
    Math.floor((xs[index] - minimumX) / cellEdge);
  const cellStart = new Uint32Array(columns * rows + 1);
  for (let index = 0; index < xs.length; index++) cellStart[cellOf(index) + 1]++;
  let maximumCellOccupancy = 0;
  for (let cell = 0; cell < columns * rows; cell++) {
    maximumCellOccupancy = Math.max(maximumCellOccupancy, cellStart[cell + 1]);
    cellStart[cell + 1] += cellStart[cell];
  }
  const fill = cellStart.slice(0, -1);
  const cellEntries = new Uint32Array(xs.length);
  for (let index = 0; index < xs.length; index++) cellEntries[fill[cellOf(index)]++] = index;
  return {cellEdge, columns, rows, cellStart, cellEntries, maximumCellOccupancy};
}

/** Count overlapping circle pairs (distance < radiusA + radiusB) using the grid. */
export function countOverlappingPairs(
  grid: DorlingNeighborGrid,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  radii: ArrayLike<number>
): number {
  let minimumX = Infinity;
  let minimumY = Infinity;
  for (let index = 0; index < xs.length; index++) {
    minimumX = Math.min(minimumX, xs[index]);
    minimumY = Math.min(minimumY, ys[index]);
  }
  let count = 0;
  for (let cellY = 0; cellY < grid.rows; cellY++)
    for (let cellX = 0; cellX < grid.columns; cellX++) {
      const cell = cellY * grid.columns + cellX;
      for (let a = grid.cellStart[cell]; a < grid.cellStart[cell + 1]; a++) {
        const first = grid.cellEntries[a];
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cellX + dx;
            const ny = cellY + dy;
            if (nx < 0 || ny < 0 || nx >= grid.columns || ny >= grid.rows) continue;
            const neighbor = ny * grid.columns + nx;
            for (let b = grid.cellStart[neighbor]; b < grid.cellStart[neighbor + 1]; b++) {
              const second = grid.cellEntries[b];
              if (second <= first) continue;
              if (
                Math.hypot(xs[first] - xs[second], ys[first] - ys[second]) <
                radii[first] + radii[second]
              )
                count++;
            }
          }
      }
    }
  return count;
}
