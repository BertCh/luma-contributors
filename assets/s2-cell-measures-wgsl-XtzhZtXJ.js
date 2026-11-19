var e=`
struct S2CellMeasures {
  area: f32,
  perimeter: f32,
  edges: vec4f,
};

// Preserve quadratic-map increments rather than subtracting rounded unit vertices. The
// determinant is exactly du*dv and the denominators remain well conditioned through level 30.
fn cellMeasuresGetS2(cell: vec2u) -> S2CellMeasures {
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
  let edges = vec4f(
    atan2(delta.x * sqrt(1.0 + lower.y * lower.y), dot(first, second)),
    atan2(delta.y * sqrt(1.0 + upper.x * upper.x), dot(second, third)),
    atan2(delta.x * sqrt(1.0 + upper.y * upper.y), dot(third, fourth)),
    atan2(delta.y * sqrt(1.0 + lower.x * lower.x), dot(fourth, first))
  );
  // Geometry visits (00, 01, 11, 10), reversing these (00, 10, 11, 01) edges.
  // Keep the perimeter's original summation order; only the public edge slots change.
  return S2CellMeasures(area, edges.x + edges.y + edges.z + edges.w, edges.wzyx);
}
`;export{e as t};