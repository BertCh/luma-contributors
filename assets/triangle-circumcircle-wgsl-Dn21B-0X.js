var e=`
fn scaleTriangleComponent(value: f32, exponent: i32) -> f32 {
  // Split the power so even the inverse of an exponent-128 extent uses ordinary ldexp exponents.
  let firstExponent = exponent / 2i;
  return ldexp(ldexp(value, firstExponent), exponent - firstExponent);
}
fn scaleTrianglePosition(value: vec2f, exponent: i32) -> vec2f {
  return vec2f(scaleTriangleComponent(value.x, exponent), scaleTriangleComponent(value.y, exponent));
}
struct ScaledTriangleEdge { direction: vec2f, exponent: i32 }
fn getScaledTriangleEdge(first: vec2f, second: vec2f) -> ScaledTriangleEdge {
  var edge = second - first;
  var exponent = 0i;
  if (!(abs(edge.x) <= 3.402823466e38 && abs(edge.y) <= 3.402823466e38)) {
    exponent = frexp(max(max(abs(first.x), abs(first.y)), max(abs(second.x), abs(second.y)))).exp;
    edge = scaleTrianglePosition(second, -exponent) - scaleTrianglePosition(first, -exponent);
  }
  let extent = max(abs(edge.x), abs(edge.y));
  if (extent == 0.0) { return ScaledTriangleEdge(vec2f(0.0), 0i); }
  let relativeExponent = frexp(extent).exp;
  return ScaledTriangleEdge(scaleTrianglePosition(edge, -relativeExponent), exponent + relativeExponent);
}
`,t=`
${e}
struct TriangleCircumcircle {
  center: vec2f,
  radius: f32,
  orientation: i32,
}
fn getTriangleAlignedTerm(fraction: f32, exponent: i32) -> f32 {
  // Terms below the f32 significand range cannot contribute in this frame. Avoid invalid ldexp powers.
  if (exponent < -149i) { return 0.0; }
  return scaleTriangleComponent(fraction, exponent);
}
fn sumTriangleCoordinate(origin: f32, first: f32, firstExponent: i32, second: f32, secondExponent: i32) -> f32 {
  let originParts = frexp(origin);
  let firstParts = frexp(first);
  let secondParts = frexp(second);
  let originPower = select(-1000i, originParts.exp, origin != 0.0);
  let firstPower = select(-1000i, firstParts.exp + firstExponent, first != 0.0);
  let secondPower = select(-1000i, secondParts.exp + secondExponent, second != 0.0);
  let power = max(originPower, max(firstPower, secondPower));
  if (power == -1000i) { return 0.0; }
  let offset = getTriangleAlignedTerm(firstParts.fract, firstPower - power) + getTriangleAlignedTerm(secondParts.fract, secondPower - power);
  let centered = offset + getTriangleAlignedTerm(originParts.fract, originPower - power);
  // Add the origin before restoration: an unrepresentable offset can still give a finite center.
  return scaleTriangleComponent(centered, power);
}
fn getTriangleCircumcircle(first: vec2f, second: vec2f, third: vec2f) -> TriangleCircumcircle {
  let orientation = orientSign(first, second, third);
  if (orientation == 0 || orientation == 2) {
    return TriangleCircumcircle(vec2f(0.0), 0.0, 0);
  }
  let firstScaled = getScaledTriangleEdge(first, second);
  let secondScaled = getScaledTriangleEdge(first, third);
  let firstEdge = firstScaled.direction;
  let secondEdge = secondScaled.direction;
  let denominator = 2.0 * (firstEdge.x * secondEdge.y - firstEdge.y * secondEdge.x);
  if (denominator == 0.0) { return TriangleCircumcircle(vec2f(0.0), 0.0, 0); }
  let firstLength = dot(firstEdge, firstEdge);
  let secondLength = dot(secondEdge, secondEdge);
  let firstTerm = vec2f(secondEdge.y, -secondEdge.x) * (firstLength / denominator);
  let secondTerm = vec2f(-firstEdge.y, firstEdge.x) * (secondLength / denominator);
  let offset = vec2f(
    sumTriangleCoordinate(0.0, firstTerm.x, firstScaled.exponent, secondTerm.x, secondScaled.exponent),
    sumTriangleCoordinate(0.0, firstTerm.y, firstScaled.exponent, secondTerm.y, secondScaled.exponent));
  let center = vec2f(
    sumTriangleCoordinate(first.x, firstTerm.x, firstScaled.exponent, secondTerm.x, secondScaled.exponent),
    sumTriangleCoordinate(first.y, firstTerm.y, firstScaled.exponent, secondTerm.y, secondScaled.exponent));
  let offsetExtent = max(abs(offset.x), abs(offset.y));
  // Radius overflow does not invalidate a representable Voronoi center.
  var radius = offsetExtent;
  if (offsetExtent > 0.0 && orientIsFinite(offsetExtent)) { radius *= length(offset / offsetExtent); }
  return TriangleCircumcircle(center, radius, orientation);
}
`;export{e as n,t};