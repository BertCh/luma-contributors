// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp, toDegrees, toRadians} from '@math.gl/core';

/** Longitude, latitude anchors for labels on non-point geometry. */
export type LabelAnchor = [longitude: number, latitude: number];
type Position = readonly number[];

/** Point at half the planar length of a polyline (line and path labels). */
export function getLineLabelAnchor(coordinates: readonly Position[]): LabelAnchor | null {
  if (!coordinates.length) return null;
  let total = 0;
  for (let index = 1; index < coordinates.length; index++)
    total += Math.hypot(
      coordinates[index][0] - coordinates[index - 1][0],
      coordinates[index][1] - coordinates[index - 1][1]
    );
  if (total === 0) return [coordinates[0][0], coordinates[0][1]];
  let remaining = total / 2;
  for (let index = 1; index < coordinates.length; index++) {
    const length = Math.hypot(
      coordinates[index][0] - coordinates[index - 1][0],
      coordinates[index][1] - coordinates[index - 1][1]
    );
    if (remaining <= length) {
      const fraction = length === 0 ? 0 : remaining / length;
      return [
        coordinates[index - 1][0] + (coordinates[index][0] - coordinates[index - 1][0]) * fraction,
        coordinates[index - 1][1] + (coordinates[index][1] - coordinates[index - 1][1]) * fraction
      ];
    }
    remaining -= length;
  }
  const last = coordinates[coordinates.length - 1];
  return [last[0], last[1]];
}

/** Arc labels sit at the great-circle midpoint of source and target. */
export function getArcLabelAnchor(source: Position, target: Position): LabelAnchor {
  const vector = (position: Position): [number, number, number] => {
    const longitude = toRadians(position[0]);
    const latitude = toRadians(position[1]);
    return [
      Math.cos(latitude) * Math.cos(longitude),
      Math.cos(latitude) * Math.sin(longitude),
      Math.sin(latitude)
    ];
  };
  const first = vector(source);
  const second = vector(target);
  const sum: [number, number, number] = [
    first[0] + second[0],
    first[1] + second[1],
    first[2] + second[2]
  ];
  const norm = Math.hypot(sum[0], sum[1], sum[2]);
  // Antipodal endpoints have no unique midpoint; use the planar average.
  if (norm < 1e-9) return [(source[0] + target[0]) / 2, (source[1] + target[1]) / 2];
  return [toDegrees(Math.atan2(sum[1], sum[0])), toDegrees(Math.asin(clamp(sum[2] / norm, -1, 1)))];
}

/** Area-weighted centroid of a polygon's outer ring; degenerate rings use the bounding box center. */
export function getPolygonLabelAnchor(rings: readonly (readonly Position[])[]): LabelAnchor | null {
  const outer = rings[0];
  if (!outer?.length) return null;
  let area = 0;
  let centroidX = 0;
  let centroidY = 0;
  for (let index = 0; index < outer.length - 1; index++) {
    const cross = outer[index][0] * outer[index + 1][1] - outer[index + 1][0] * outer[index][1];
    area += cross;
    centroidX += (outer[index][0] + outer[index + 1][0]) * cross;
    centroidY += (outer[index][1] + outer[index + 1][1]) * cross;
  }
  if (Math.abs(area) > 1e-12) return [centroidX / (3 * area), centroidY / (3 * area)];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const position of outer) {
    minX = Math.min(minX, position[0]);
    maxX = Math.max(maxX, position[0]);
    minY = Math.min(minY, position[1]);
    maxY = Math.max(maxY, position[1]);
  }
  return [(minX + maxX) / 2, (minY + maxY) / 2];
}

/** MultiPolygon labels use the polygon with the largest absolute outer-ring area. */
export function getMultiPolygonLabelAnchor(
  polygons: readonly (readonly (readonly Position[])[])[]
): LabelAnchor | null {
  let best: {area: number; anchor: LabelAnchor} | null = null;
  for (const rings of polygons) {
    const anchor = getPolygonLabelAnchor(rings);
    if (!anchor) continue;
    let area = 0;
    const outer = rings[0];
    for (let index = 0; index < outer.length - 1; index++)
      area += outer[index][0] * outer[index + 1][1] - outer[index + 1][0] * outer[index][1];
    if (!best || Math.abs(area) > best.area) best = {area: Math.abs(area), anchor};
  }
  return best?.anchor ?? null;
}
