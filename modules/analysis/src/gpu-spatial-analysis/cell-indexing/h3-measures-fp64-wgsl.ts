// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// Reuses the H3 faceijk.c boundary walk, with published fp64 arithmetic for its geometry.

import {getH3BoundaryWGSL} from './h3-boundary-wgsl';
import {getFaceBasisVectors} from './h3-index-extended-wgsl';
import {getCellMeasuresFP64Literal} from './cell-measures-fp64-wgsl';

const FACE_BASES = getFaceBasisVectors().flat().map(getCellMeasuresFP64Literal).join(',\n');
// Boundary vertices always use an even (Class II) substrate resolution, at most 16.
const SUBSTRATE_SCALES = Array.from({length: 9}, (_, index) =>
  getCellMeasuresFP64Literal(0.381966011250105 / (3 * 7 ** index))
).join(',\n');

const H3_BOUNDARY_GEOMETRY_FP64_WGSL = /* wgsl */ `
const CELL_MEASURES_H3_FACE_BASES = array<vec2f, 180>(${FACE_BASES});
const CELL_MEASURES_H3_SUBSTRATE_SCALES = array<vec2f, 9>(${SUBSTRATE_SCALES});
const CELL_MEASURES_SQRT3_2 = ${getCellMeasuresFP64Literal(Math.sqrt(3) / 2)};

fn cellMeasuresH3CoordinateToPlane64(coordinates: vec3i) -> CellMeasuresPlane64 {
  let first = cellMeasuresInteger64(coordinates.x - coordinates.z);
  let second = cellMeasuresInteger64(coordinates.y - coordinates.z);
  return CellMeasuresPlane64(sub_fp64(first, mul_fp64(vec2f(0.5, 0.0), second)), mul_fp64(second, CELL_MEASURES_SQRT3_2));
}
fn cellIndexH3BoundaryIntersect(first: CellMeasuresPlane64, second: CellMeasuresPlane64, edgeFirst: CellMeasuresPlane64, edgeSecond: CellMeasuresPlane64) -> CellMeasuresPlane64 {
  let horizontal = sub_fp64(second.x, first.x);
  let vertical = sub_fp64(second.y, first.y);
  let edgeHorizontal = sub_fp64(edgeSecond.x, edgeFirst.x);
  let edgeVertical = sub_fp64(edgeSecond.y, edgeFirst.y);
  let numerator = sub_fp64(mul_fp64(edgeHorizontal, sub_fp64(first.y, edgeFirst.y)), mul_fp64(edgeVertical, sub_fp64(first.x, edgeFirst.x)));
  let denominator = sub_fp64(mul_fp64(horizontal, edgeVertical), mul_fp64(edgeHorizontal, vertical));
  let fraction = div_fp64(numerator, denominator);
  return CellMeasuresPlane64(sum_fp64(first.x, mul_fp64(fraction, horizontal)), sum_fp64(first.y, mul_fp64(fraction, vertical)));
}
fn cellIndexH3BoundaryIntersectFaceEdge(first: CellMeasuresPlane64, second: CellMeasuresPlane64, adjustedResolution: u32, quadrant: u32) -> CellMeasuresPlane64 {
  let dimension = cellMeasuresInteger64(dggs_h3_get_max_dim_by_cii_resolution(adjustedResolution));
  let horizontal = mul_fp64(vec2f(3.0, 0.0), dimension);
  let vertical = mul_fp64(horizontal, CELL_MEASURES_SQRT3_2);
  let firstCorner = CellMeasuresPlane64(horizontal, vec2f(0.0));
  let secondCorner = CellMeasuresPlane64(mul_fp64(vec2f(-0.5, 0.0), horizontal), vertical);
  let thirdCorner = CellMeasuresPlane64(secondCorner.x, -vertical);
  if (quadrant == 1u) { return cellIndexH3BoundaryIntersect(first, second, firstCorner, secondCorner); }
  if (quadrant == 3u) { return cellIndexH3BoundaryIntersect(first, second, secondCorner, thirdCorner); }
  return cellIndexH3BoundaryIntersect(first, second, thirdCorner, firstCorner);
}
fn cellMeasuresH3FaceBasis64(face: u32, basis: u32) -> CellMeasuresVector64 {
  let offset = face * 9u + basis * 3u;
  return CellMeasuresVector64(CELL_MEASURES_H3_FACE_BASES[offset], CELL_MEASURES_H3_FACE_BASES[offset + 1u], CELL_MEASURES_H3_FACE_BASES[offset + 2u]);
}
fn cellIndexH3BoundaryHex2dToUnitVector(point: CellMeasuresPlane64, face: u32, adjustedResolution: u32) -> CellMeasuresVector64 {
  let scale = CELL_MEASURES_H3_SUBSTRATE_SCALES[adjustedResolution / 2u];
  let horizontal = cellMeasuresVectorScale64(cellMeasuresH3FaceBasis64(face, 1u), mul_fp64(point.x, scale));
  let vertical = cellMeasuresVectorScale64(cellMeasuresH3FaceBasis64(face, 2u), mul_fp64(point.y, scale));
  return cellMeasuresVectorNormalize64(cellMeasuresVectorAdd64(cellMeasuresH3FaceBasis64(face, 0u), cellMeasuresVectorAdd64(horizontal, vertical)));
}
// Coordinates are optional products of the shared boundary walk; measures retain both words.
fn cellIndexH3BoundaryUnitVectorToLngLat(value: CellMeasuresVector64) -> vec2f {
  let unitVector = cellMeasuresVectorHigh64(value);
  return vec2f(guardedAtan2(unitVector.y, unitVector.x), atan2(unitVector.z, length(unitVector.xy))) * DGGS_RADIANS_TO_DEGREES;
}
`;

/** Full H3 boundary precision, including substrate projection and distortion intersections. @internal */
export const H3_MEASURES_FP64_WGSL = /* wgsl */ `
${getH3BoundaryWGSL({
  unitType: 'CellMeasuresVector64',
  coordinateToPlane: 'cellMeasuresH3CoordinateToPlane64',
  source: H3_BOUNDARY_GEOMETRY_FP64_WGSL
})}

struct CellMeasures {
  count: u32,
  area: f32,
  perimeter: f32,
  edges: array<f32, 10>,
};
fn cellMeasuresIsPentagon(cell: vec2u) -> bool {
  return dggs_h3_is_valid_cell_id(cell) && dggs_h3_is_base_cell_pentagon(dggs_h3_get_base_cell(cell)) && dggs_h3_get_leading_non_zero_digit(cell) == 0u;
}
fn cellMeasuresGetH3(cell: vec2u) -> CellMeasures {
  var measures: CellMeasures;
  let boundary = cellIndexH3GetBoundary(cell);
  measures.count = boundary.count;
  if (boundary.count < 3u) { return measures; }
  let anchor = boundary.units[0];
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    let first = boundary.units[vertex];
    let second = boundary.units[(vertex + 1u) % boundary.count];
    let difference = cellMeasuresVectorSubtract64(second, first);
    let crossProduct = cellMeasuresVectorCross64(first, difference);
    let edge = atan2(cellMeasuresVectorLength64(crossProduct).x, cellMeasuresVectorDot64(first, second).x);
    measures.edges[vertex] = edge;
    measures.perimeter += edge;
    // A fan anchored on an actual boundary vertex needs no separately projected center.
    // Face-distortion vertices can make the boundary concave: retain signed triangles so
    // pieces outside the polygon cancel rather than adding their area twice.
    if (vertex > 0u && vertex + 1u < boundary.count) {
      let firstOffset = cellMeasuresVectorSubtract64(first, anchor);
      let secondOffset = cellMeasuresVectorSubtract64(second, anchor);
      let determinant = cellMeasuresVectorDot64(anchor, cellMeasuresVectorCross64(firstOffset, secondOffset));
      let denominator = sum_fp64(vec2f(1.0, 0.0), sum_fp64(cellMeasuresVectorDot64(anchor, first), sum_fp64(cellMeasuresVectorDot64(first, second), cellMeasuresVectorDot64(second, anchor))));
      measures.area += 2.0 * atan2(determinant.x, denominator.x);
    }
  }
  measures.area = abs(measures.area);
  return measures;
}
`;
