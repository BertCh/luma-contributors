function e(e,t,n){return`
const FIELD_WIDTH: u32 = ${t}u;
const FIELD_HEIGHT: u32 = ${n}u;

fn isNanFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}

fn readFieldVelocity(column: u32, row: u32) -> vec2<f32> {
  let base = ${e}Offset + 2u * (row * FIELD_WIDTH + column);
  return vec2<f32>(${e}[base], ${e}[base + 1u]);
}

fn isNanVelocity(value: vec2<f32>) -> bool {
  return isNanFloat(value.x) || isNanFloat(value.y);
}

fn sampleField(position: vec2<f32>, extent: vec4<f32>) -> vec3<f32> {
  let local = (position - extent.xy) / extent.zw;
  if (isNanVelocity(local)) {
    return vec3<f32>(0.0);
  }
  if (local.x < 0.0 || local.y < 0.0 || local.x >= f32(FIELD_WIDTH) || local.y >= f32(FIELD_HEIGHT)) {
    return vec3<f32>(0.0);
  }
  let grid = clamp(
    local - vec2<f32>(0.5),
    vec2<f32>(0.0),
    vec2<f32>(f32(FIELD_WIDTH - 1u), f32(FIELD_HEIGHT - 1u))
  );
  let column0 = min(u32(grid.x), FIELD_WIDTH - 1u);
  let row0 = min(u32(grid.y), FIELD_HEIGHT - 1u);
  let column1 = min(column0 + 1u, FIELD_WIDTH - 1u);
  let row1 = min(row0 + 1u, FIELD_HEIGHT - 1u);
  let fraction = grid - vec2<f32>(f32(column0), f32(row0));
  let v00 = readFieldVelocity(column0, row0);
  let v10 = readFieldVelocity(column1, row0);
  let v01 = readFieldVelocity(column0, row1);
  let v11 = readFieldVelocity(column1, row1);
  if (isNanVelocity(v00) || isNanVelocity(v10) || isNanVelocity(v01) || isNanVelocity(v11)) {
    return vec3<f32>(0.0);
  }
  let bottom = v00 * (1.0 - fraction.x) + v10 * fraction.x;
  let top = v01 * (1.0 - fraction.x) + v11 * fraction.x;
  return vec3<f32>(bottom * (1.0 - fraction.y) + top * fraction.y, 1.0);
}
`}export{e as t};