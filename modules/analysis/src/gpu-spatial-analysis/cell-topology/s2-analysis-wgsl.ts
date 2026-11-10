// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
/** Spherical S2 boundary arithmetic. Integer identity helpers are supplied by dggs. @internal */
export const S2_ANALYSIS_WGSL = /* wgsl */ `
fn s2AnalysisVertex(cell: vec2u, slot: u32) -> vec3f {
  let coordinate = dggs_s2_get_ij(cell);
  let offset = vec2f(f32(select(0u, 1u, slot == 1u || slot == 2u)), f32(select(0u, 1u, slot >= 2u)));
  let surface = dggs_s2_ij_to_st(coordinate, dggs_s2_get_level(cell), offset);
  return normalize(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(cell), dggs_s2_st_to_uv(surface)));
}
// atan2 with a zero (or negative zero) second argument returns minus half pi on some GPU
// compilers, which made a cell 90 degrees from a face-center query look like it touched it.
fn s2AnalysisAngle(first: vec3f, second: vec3f) -> f32 {
  let cosine = dot(first, second);
  return select(atan2(length(cross(first, second)), cosine), 1.5707963267948966, cosine == 0.0);
}
fn s2AnalysisTriangle(first: vec3f, second: vec3f, third: vec3f) -> f32 {
  return 2.0 * atan2(abs(dot(first, cross(second - first, third - first))), 1.0 + dot(first, second) + dot(second, third) + dot(third, first));
}
// Preserve the small quadratic-map increments instead of subtracting rounded unit vertices.
// The determinant of each unnormalized corner triangle is exactly du*dv; the solid-angle
// denominator is well conditioned even at level 30. Face rotations preserve both measures.
fn s2AnalysisStableMeasures(cell: vec2u) -> vec2f {
  let level = dggs_s2_get_level(cell);
  let step = 1.0 / f32(1u << level);
  let surface = vec2f(dggs_s2_get_ij(cell)) * step;
  let lower = dggs_s2_st_to_uv(surface);
  var delta = (4.0 / 3.0) * step * select(2.0 * (vec2f(1.0) - surface) - vec2f(step), 2.0 * surface + vec2f(step), surface >= vec2f(0.5));
  if (level == 0u) { delta = vec2f(2.0); }
  let upper = lower + delta;
  let first = vec3f(1.0, lower.x, lower.y);
  let second = vec3f(1.0, upper.x, lower.y);
  let third = vec3f(1.0, upper.x, upper.y);
  let fourth = vec3f(1.0, lower.x, upper.y);
  let firstLength = length(first); let secondLength = length(second);
  let thirdLength = length(third); let fourthLength = length(fourth);
  let determinant = delta.x * delta.y;
  let firstDenominator = firstLength * secondLength * thirdLength + dot(first, second) * thirdLength + dot(second, third) * firstLength + dot(third, first) * secondLength;
  let secondDenominator = firstLength * thirdLength * fourthLength + dot(first, third) * fourthLength + dot(third, fourth) * firstLength + dot(fourth, first) * thirdLength;
  let area = 2.0 * (atan2(determinant, firstDenominator) + atan2(determinant, secondDenominator));
  let perimeter = atan2(delta.x * sqrt(1.0 + lower.y * lower.y), dot(first, second)) + atan2(delta.y * sqrt(1.0 + upper.x * upper.x), dot(second, third)) + atan2(delta.x * sqrt(1.0 + upper.y * upper.y), dot(third, fourth)) + atan2(delta.y * sqrt(1.0 + lower.x * lower.x), dot(fourth, first));
  return vec2f(area, perimeter);
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
