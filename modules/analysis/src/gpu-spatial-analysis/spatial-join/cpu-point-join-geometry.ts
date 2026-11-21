// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {CPUSpatialJoinGeometry} from './cpu-feature-join';
import {validatePointJoinOffsets} from './point-join-operation-utils';
import type {GPUPointInPolygonBoundaryRule} from './gpu-point-in-polygon-join';
import {computeTriangulationOrientation} from '../../analysis-core/polygon-triangulation/cpu-triangulation-predicates';

/** CPU polygon feature with polygon-part rings and a conservative XY envelope. @internal */
export type CPUPointJoinPolygon = {parts: number[][][]; bounds: number[]; uncertain: boolean};
/** Validates complete polygon offset hierarchies; holes stay with their owning polygon part. @internal */
export function readCPUPointJoinPolygons(
  id: string,
  geometry: CPUSpatialJoinGeometry,
  allowNonfinite = false
): CPUPointJoinPolygon[] {
  const {positions, featureOffsets, polygonOffsets, ringOffsets} = geometry;
  validateRequirement(
    id,
    geometry.kind === 'polygons' &&
      positions.length % 2 === 0 &&
      Boolean(featureOffsets && polygonOffsets && ringOffsets),
    'polygons requires complete XY rows and featureOffsets/polygonOffsets/ringOffsets'
  );
  validatePointJoinOffsets(id, 'polygons.ringOffsets', ringOffsets!, positions.length / 2);
  validatePointJoinOffsets(id, 'polygons.polygonOffsets', polygonOffsets!, ringOffsets!.length - 1);
  validatePointJoinOffsets(
    id,
    'polygons.featureOffsets',
    featureOffsets!,
    polygonOffsets!.length - 1
  );
  validateRequirement(
    id,
    !geometry.sourceIds || geometry.sourceIds.length === featureOffsets!.length - 1,
    'polygons.sourceIds needs one row per polygon feature'
  );
  return Array.from({length: featureOffsets!.length - 1}, (_, feature) => {
    const parts: number[][][] = [],
      bounds = [Infinity, Infinity, -Infinity, -Infinity];
    let uncertain = false;
    for (
      let polygon = featureOffsets![feature];
      polygon < featureOffsets![feature + 1];
      polygon++
    ) {
      const rings: number[][] = [];
      for (let ring = polygonOffsets![polygon]; ring < polygonOffsets![polygon + 1]; ring++) {
        const coordinates = Array.from(
          positions.subarray(2 * ringOffsets![ring], 2 * ringOffsets![ring + 1])
        );
        for (let vertex = 0; vertex < coordinates.length; vertex += 2) {
          const x = coordinates[vertex],
            y = coordinates[vertex + 1];
          validateRequirement(
            id,
            allowNonfinite || (Number.isFinite(x) && Number.isFinite(y)),
            'polygons.positions must contain finite XY coordinates'
          );
          if (!Number.isFinite(x) || !Number.isFinite(y)) {
            uncertain = true;
            continue;
          }
          bounds[0] = Math.min(bounds[0], x);
          bounds[1] = Math.min(bounds[1], y);
          bounds[2] = Math.max(bounds[2], x);
          bounds[3] = Math.max(bounds[3], y);
        }
        rings.push(coordinates);
      }
      parts.push(rings);
    }
    return {parts, bounds, uncertain};
  });
}
/** Exact orientation sign for finite binary64 coordinates, independent of WGSL arithmetic. @internal */
export function getCPUPointJoinOrientation(
  firstX: number,
  firstY: number,
  secondX: number,
  secondY: number,
  pointX: number,
  pointY: number
): number {
  const orientation = computeTriangulationOrientation(
    [firstX, firstY],
    [secondX, secondY],
    [pointX, pointY]
  );
  // Core's 2 means nonfinite, never a positive orientation or a collinear contact.
  return orientation === 2 ? Number.NaN : orientation;
}
/** Exact point classification with symbolic (+epsilon^2,+epsilon) half-open ownership. @internal */
export function classifyCPUPointJoinPolygon(
  pointX: number,
  pointY: number,
  feature: CPUPointJoinPolygon,
  rule: GPUPointInPolygonBoundaryRule
): boolean {
  let insideFeature = false,
    ownedFeature = false,
    boundary = false;
  for (const rings of feature.parts) {
    let insidePart = false,
      ownedPart = false;
    for (const ring of rings) {
      if (ring.length < 6) continue;
      let previousX = ring[ring.length - 2],
        previousY = ring[ring.length - 1];
      for (let vertex = 0; vertex < ring.length; vertex += 2) {
        const currentX = ring[vertex],
          currentY = ring[vertex + 1];
        const sign = getCPUPointJoinOrientation(
          previousX,
          previousY,
          currentX,
          currentY,
          pointX,
          pointY
        );
        let side = sign;
        if (sign === 0) {
          boundary ||=
            pointX >= Math.min(previousX, currentX) &&
            pointX <= Math.max(previousX, currentX) &&
            pointY >= Math.min(previousY, currentY) &&
            pointY <= Math.max(previousY, currentY);
          side =
            currentX > previousX
              ? 1
              : currentX < previousX
                ? -1
                : currentY > previousY
                  ? -1
                  : currentY < previousY
                    ? 1
                    : 0;
        }
        if (previousY > pointY !== currentY > pointY) {
          if ((currentY > previousY && sign > 0) || (currentY < previousY && sign < 0))
            insidePart = !insidePart;
          if ((currentY > previousY && side > 0) || (currentY < previousY && side < 0))
            ownedPart = !ownedPart;
        }
        previousX = currentX;
        previousY = currentY;
      }
    }
    insideFeature ||= insidePart;
    ownedFeature ||= ownedPart;
  }
  return boundary ? rule === 'closed' || (rule === 'half-open' && ownedFeature) : insideFeature;
}
