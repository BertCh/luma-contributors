// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {dggs} from '@luma.gl/shaders';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import {QUADBIN_TILE_WGSL} from './cell-topology-wgsl';
import {S2_NEIGHBORS_WGSL} from './s2-neighbor-wgsl';

/** Reserve uint32's largest value for undefined; all level-30 cube-grid distances fit below it. */
export const GPU_CELL_GRID_PATH_S2_MAXIMUM_RADIUS = 0xfffffffe;

/**
 * Unfold the shortest cube-grid route. A route on one face is Manhattan; adjacent faces unfold
 * across their shared seam; opposite faces unfold across one of four intermediate faces.
 * Detours across further faces cannot improve these L1 distances. The tangent coordinate is
 * shared across a seam, so its difference contributes exactly once. Integer centers are odd
 * lattice coordinates in [-size, size]; even the longest candidate, 4*size-2, fits uint32 at
 * level 30. No search radius or per-query storage is required. @internal
 */
const S2_GRID_DISTANCE_WGSL = /* wgsl */ `
fn s2PathCenter(face: u32, coordinate: vec2u, size: u32) -> vec3i {
  return S2_FACE_NORMAL[face] * i32(size) + S2_FACE_COLUMN[face] * (i32(2u * coordinate.x + 1u) - i32(size)) + S2_FACE_ROW[face] * (i32(2u * coordinate.y + 1u) - i32(size));
}
fn s2PathBoundary(face: u32, coordinate: vec2u, direction: vec3i, size: u32) -> u32 {
  let columnSign = s2IntegerDot(direction, S2_FACE_COLUMN[face]);
  let axisSign = select(s2IntegerDot(direction, S2_FACE_ROW[face]), columnSign, columnSign != 0);
  let value = select(coordinate.y, coordinate.x, columnSign != 0);
  return select(value, size - 1u - value, axisSign > 0);
}
fn s2PathDistance(first: vec2u, second: vec2u) -> u32 {
  let firstFace = dggs_s2_get_face(first); let secondFace = dggs_s2_get_face(second);
  let firstCoordinate = dggs_s2_get_ij(first); let secondCoordinate = dggs_s2_get_ij(second);
  if (firstFace == secondFace) {
    let difference = abs(vec2i(firstCoordinate) - vec2i(secondCoordinate));
    return u32(difference.x) + u32(difference.y);
  }
  let size = 1u << dggs_s2_get_level(first);
  let firstNormal = S2_FACE_NORMAL[firstFace]; let secondNormal = S2_FACE_NORMAL[secondFace];
  let firstCenter = s2PathCenter(firstFace, firstCoordinate, size);
  let secondCenter = s2PathCenter(secondFace, secondCoordinate, size);
  if (s2IntegerDot(firstNormal, secondNormal) == 0) {
    let tangent = vec3i(1) - abs(firstNormal) - abs(secondNormal);
    let span = u32(abs(s2IntegerDot(firstCenter, tangent) - s2IntegerDot(secondCenter, tangent))) / 2u;
    return s2PathBoundary(firstFace, firstCoordinate, secondNormal, size) + s2PathBoundary(secondFace, secondCoordinate, firstNormal, size) + 1u + span;
  }
  var distance = 0xffffffffu;
  for (var side = 0u; side < 6u; side++) {
    let direction = S2_FACE_NORMAL[side];
    if (s2IntegerDot(firstNormal, direction) != 0) { continue; }
    let tangent = vec3i(1) - abs(firstNormal) - abs(direction);
    let span = u32(abs(s2IntegerDot(firstCenter, tangent) - s2IntegerDot(secondCenter, tangent))) / 2u;
    let candidateDistance = s2PathBoundary(firstFace, firstCoordinate, direction, size) + s2PathBoundary(secondFace, secondCoordinate, direction, size) + size + 1u + span;
    distance = min(distance, candidateDistance);
  }
  return distance;
}
`;

/** Extra cell-family bodies inside the existing chunk-invariant row kernel. @internal */
export function getCellGridPathGridKernel(
  family: 'quadbin' | 's2',
  options: {
    readCell: (name: string) => string;
    writeCell: (slot: string, cell: string) => string;
    maximumPathLength: number;
    maximumSearchRadius: number;
    hasMask: boolean;
    distances: boolean;
    cells: boolean;
    counts: boolean;
  }
): {declarations: string; body: string} {
  const {readCell, writeCell} = options;
  const initial = `${readCell('origin')}\n${readCell('destination')}
  var pathDistance = 0xffffffffu;
  var isValid = ${options.hasMask ? 'rowMask[rowMaskOffset + index] != 0u' : 'true'};`;
  const publishDistance = options.distances
    ? 'outDistances[outDistancesOffset + index] = pathDistance;'
    : '';
  const publishCount = options.counts
    ? 'outCounts[outCountsOffset + index] = select(pathDistance + 1u, 0u, pathDistance == 0xffffffffu);'
    : '';
  if (family === 'quadbin')
    return {
      declarations: `${dggs.source}\n${CELL_KEY_WGSL}\n${QUADBIN_TILE_WGSL}`,
      body: `${initial}
  isValid = isValid && cellIsValidQuadbin(origin) && cellIsValidQuadbin(destination) && ((origin.x >> 20u) & 31u) == ((destination.x >> 20u) & 31u);
  let level = (origin.x >> 20u) & 31u;
  let size = i32(1u << min(level, 26u));
  let first = cellTopologyQuadbinGetTile(origin);
  let last = cellTopologyQuadbinGetTile(destination);
  var difference = vec2i(last) - vec2i(first);
  if (abs(difference.x) > size / 2) { difference.x -= select(-size, size, difference.x > 0); }
  if (isValid) { pathDistance = u32(max(abs(difference.x), abs(difference.y))); }
  ${publishDistance}
  ${publishCount}
  ${
    options.cells
      ? `for (var slot = 0u; slot < ${options.maximumPathLength}u; slot++) {
    var result = vec2u(0u);
    if (pathDistance != 0xffffffffu && slot <= pathDistance) {
      let progress = min(vec2u(slot), vec2u(abs(difference)));
      let delta = vec2i(progress) * select(vec2i(-1), vec2i(1), difference >= vec2i(0));
      let tile = vec2i(first) + delta;
      result = cellTopologyQuadbinGetKey(u32((tile.x % size + size) % size), u32(tile.y), level);
    }
    ${writeCell(`index * ${options.maximumPathLength}u + slot`, 'result')}
  }`
      : ''
  }`
    };
  return {
    declarations: `${dggs.source}\n${S2_NEIGHBORS_WGSL}\n${S2_GRID_DISTANCE_WGSL}`,
    body: `${initial}
  isValid = isValid && dggs_s2_is_valid_cell_id(origin) && dggs_s2_is_valid_cell_id(destination) && dggs_s2_get_level(origin) == dggs_s2_get_level(destination);
  if (isValid) {
    pathDistance = s2PathDistance(origin, destination);
    if (pathDistance > ${options.maximumSearchRadius}u) { pathDistance = 0xffffffffu; }
  }
  ${publishDistance}
  ${publishCount}
  ${
    options.cells
      ? `var current = origin;
  for (var slot = 0u; slot < ${options.maximumPathLength}u; slot++) {
    var result = vec2u(0u);
    if (pathDistance != 0xffffffffu && slot <= pathDistance) {
      result = current;
      if (slot < pathDistance) {
        for (var direction = 0u; direction < 4u; direction++) {
          let neighbor = s2Neighbor(current, direction);
          if (s2PathDistance(neighbor, destination) == pathDistance - slot - 1u) { current = neighbor; break; }
        }
      }
    }
    ${writeCell(`index * ${options.maximumPathLength}u + slot`, 'result')}
  }`
      : ''
  }`
  };
}
