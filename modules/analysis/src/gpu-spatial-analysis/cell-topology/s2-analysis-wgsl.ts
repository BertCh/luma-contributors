// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {S2_CELL_MEASURES_WGSL} from '../cell-indexing/s2-cell-measures-wgsl';
import {GUARDED_ATAN2_WGSL} from '../../analysis-core/kernels/wgsl-common';

/**
 * Spherical S2 boundary arithmetic. Integer identity helpers are supplied by dggs. Includes
 * `guardedAtan2` ({@link GUARDED_ATAN2_WGSL}), so a kernel that embeds this must not add it again.
 * @internal
 */
export const S2_ANALYSIS_WGSL = /* wgsl */ `
${GUARDED_ATAN2_WGSL}
fn s2AnalysisVertex(cell: vec2u, slot: u32) -> vec3f {
  let coordinate = dggs_s2_get_ij(cell);
  let offset = vec2f(f32(select(0u, 1u, slot == 1u || slot == 2u)), f32(select(0u, 1u, slot >= 2u)));
  let surface = dggs_s2_ij_to_st(coordinate, dggs_s2_get_level(cell), offset);
  return normalize(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(cell), dggs_s2_st_to_uv(surface)));
}
// atan2 with a zero (or negative zero) second argument returns minus half pi on some GPU
// compilers, which made a cell 90 degrees from a face-center query look like it touched it.
// On the headless test GPU atan2(y, -0.0) is -sign(y) * pi / 2 (atan2(y, +0.0) is correct), and a
// polygon winding denominator can be -0.0 (FX-17). An exactly zero second argument therefore takes
// the quadrant angle explicitly (zero when both arguments are zero); see guardedAtan2.
fn s2AnalysisAtan2(y: f32, x: f32) -> f32 {
  return guardedAtan2(y, x);
}
fn s2AnalysisAngle(first: vec3f, second: vec3f) -> f32 {
  let cosine = dot(first, second);
  return select(atan2(length(cross(first, second)), cosine), 1.5707963267948966, cosine == 0.0);
}
fn s2AnalysisTriangle(first: vec3f, second: vec3f, third: vec3f) -> f32 {
  return 2.0 * atan2(abs(dot(first, cross(second - first, third - first))), 1.0 + dot(first, second) + dot(second, third) + dot(third, first));
}
${S2_CELL_MEASURES_WGSL}
fn s2AnalysisStableMeasures(cell: vec2u) -> vec2f {
  let measured = cellMeasuresGetS2(cell);
  return vec2f(measured.area, measured.perimeter);
}
fn s2AnalysisDistance(cell: vec2u, point: vec3f) -> f32 {
  var minimum = 3.141592653589793; var inside = true;
  for (var edge = 0u; edge < 4u; edge++) {
    let first = s2AnalysisVertex(cell, edge); let second = s2AnalysisVertex(cell, (edge + 1u) % 4u);
    let normal = normalize(cross(first, second));
    inside = inside && dot(normal, point) >= 0.0;
    minimum = min(minimum, s2AnalysisAngle(point, first));
    // Double-cross projection avoids cancellation when point is parallel to an f32-normalized
    // edge normal. Subtractive projection can leave a tiny point-parallel residue and falsely
    // classify a cell 90 degrees away as touching the query.
    let projected = cross(normal, cross(point, normal));
    if (length(projected) > 1e-10) {
      let nearest = normalize(projected);
      if (dot(cross(first, nearest), normal) >= 0.0 && dot(cross(nearest, second), normal) >= 0.0) { minimum = min(minimum, s2AnalysisAngle(point, nearest)); }
    }
  }
  return select(minimum, 0.0, inside);
}
`;
