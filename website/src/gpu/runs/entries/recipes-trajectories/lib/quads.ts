// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A tiling of irregular quadrilateral zones in the four-level polygon layout (one ring per
// feature), with a drawing mesh, outline and a smooth value per zone. Neighboring zones share
// exact corner coordinates, so contiguity operations find them.

import type {RunBuffer, RunBuilder, PolygonsRunLayer} from '../../../define-run';

export type QuadZones = {
  count: number;
  columns: number;
  rows: number;
  /** Four ring vertices per zone (x, y). Also the drawing mesh vertices. */
  positions: Float32Array;
  /** Feature, polygon and ring offsets (one ring per zone). */
  featureOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Two triangles per zone and the owner of each. */
  triangleIndices: Uint32Array;
  triangleOwners: Uint32Array;
  triangleCount: number;
  /** Zone boundary as (x0, y0, x1, y1). */
  outline: Float32Array;
  outlineCount: number;
  centers: Float32Array;
  /** Smooth spatial field in [0, 1] per zone. */
  values: Float32Array;
};

/** Zones tiling the world; `columns * rows` of them with jittered shared corners. */
export function createQuadZones(
  random: () => number,
  world: {width: number; height: number},
  columns = 16,
  rows = 9,
  inset = 0
): QuadZones {
  const stepX = world.width / columns;
  const stepY = world.height / rows;
  const cornerX = new Float32Array((columns + 1) * (rows + 1));
  const cornerY = new Float32Array((columns + 1) * (rows + 1));
  for (let row = 0; row <= rows; row++) {
    for (let column = 0; column <= columns; column++) {
      const index = row * (columns + 1) + column;
      const isBorder = row === 0 || row === rows || column === 0 || column === columns;
      const jitter = isBorder ? 0 : 0.28;
      cornerX[index] = (column + (random() - 0.5) * jitter) * stepX;
      cornerY[index] = (row + (random() - 0.5) * jitter) * stepY;
    }
  }
  const phaseX = random() * 6;
  const phaseY = random() * 6;
  const count = columns * rows;
  const positions = new Float32Array(count * 8);
  const triangleIndices = new Uint32Array(count * 6);
  const triangleOwners = new Uint32Array(count * 2);
  const outline: number[] = [];
  const centers = new Float32Array(count * 2);
  const values = new Float32Array(count);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const zone = row * columns + column;
      const corners = [
        row * (columns + 1) + column,
        row * (columns + 1) + column + 1,
        (row + 1) * (columns + 1) + column + 1,
        (row + 1) * (columns + 1) + column
      ];
      let sumX = 0;
      let sumY = 0;
      for (const [slot, corner] of corners.entries()) {
        positions[zone * 8 + slot * 2] = cornerX[corner];
        positions[zone * 8 + slot * 2 + 1] = cornerY[corner];
        sumX += cornerX[corner] / 4;
        sumY += cornerY[corner] / 4;
      }
      if (inset > 0) {
        for (let slot = 0; slot < 4; slot++) {
          const x = positions[zone * 8 + slot * 2];
          const y = positions[zone * 8 + slot * 2 + 1];
          positions[zone * 8 + slot * 2] = x + (sumX - x) * inset;
          positions[zone * 8 + slot * 2 + 1] = y + (sumY - y) * inset;
        }
      }
      for (let slot = 0; slot < 4; slot++) {
        const next = (slot + 1) % 4;
        outline.push(
          positions[zone * 8 + slot * 2],
          positions[zone * 8 + slot * 2 + 1],
          positions[zone * 8 + next * 2],
          positions[zone * 8 + next * 2 + 1]
        );
      }
      const base = zone * 4;
      triangleIndices.set([base, base + 1, base + 2, base, base + 2, base + 3], zone * 6);
      triangleOwners[zone * 2] = zone;
      triangleOwners[zone * 2 + 1] = zone;
      centers[zone * 2] = sumX;
      centers[zone * 2 + 1] = sumY;
      const u = sumX / world.height;
      const v = sumY / world.height;
      values[zone] = Math.min(
        1,
        Math.max(0, 0.5 + 0.3 * Math.sin(u * 3.1 + phaseX) + 0.25 * Math.cos(v * 4.3 + phaseY))
      );
    }
  }
  return {
    count,
    columns,
    rows,
    positions,
    featureOffsets: Uint32Array.from({length: count + 1}, (_, index) => index),
    ringOffsets: Uint32Array.from({length: count + 1}, (_, index) => index * 4),
    triangleIndices,
    triangleOwners,
    triangleCount: count * 2,
    outline: Float32Array.from(outline),
    outlineCount: count * 4,
    centers,
    values
  };
}

export type QuadMesh = {
  positions: RunBuffer<'float32x2'>;
  featureOffsets: RunBuffer<'uint32'>;
  ringOffsets: RunBuffer<'uint32'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  outline: RunBuffer<'float32x4'>;
  centers: RunBuffer<'float32x2'>;
};

/** Uploads the zones. `prefix` keeps buffer ids unique when a run has two zone systems. */
export function uploadQuadZones(builder: RunBuilder, zones: QuadZones, prefix = 'zones'): QuadMesh {
  return {
    positions: builder.upload(`${prefix}-positions`, 'float32x2', zones.positions),
    featureOffsets: builder.upload(`${prefix}-features`, 'uint32', zones.featureOffsets),
    ringOffsets: builder.upload(`${prefix}-rings`, 'uint32', zones.ringOffsets),
    indices: builder.upload(`${prefix}-indices`, 'uint32', zones.triangleIndices),
    owners: builder.upload(`${prefix}-owners`, 'uint32', zones.triangleOwners),
    outline: builder.upload(`${prefix}-outline`, 'float32x4', zones.outline),
    centers: builder.upload(`${prefix}-centers`, 'float32x2', zones.centers)
  };
}

/** The four-level polygon port the recipes take (one ring per zone). */
export function getPolygonPort(mesh: QuadMesh) {
  return {
    kind: 'polygons' as const,
    positions: mesh.positions.view,
    featureOffsets: mesh.featureOffsets.view,
    polygonOffsets: mesh.featureOffsets.view,
    ringOffsets: mesh.ringOffsets.view
  };
}

/** Polygon layer filling the zones by `values`. */
export function getQuadFill(
  mesh: QuadMesh,
  zones: QuadZones,
  values: RunBuffer,
  look: Partial<PolygonsRunLayer> = {}
): PolygonsRunLayer {
  return {
    layer: 'polygons',
    positions: mesh.positions,
    indices: mesh.indices,
    owners: mesh.owners,
    values,
    triangleCount: zones.triangleCount,
    ...look
  };
}
