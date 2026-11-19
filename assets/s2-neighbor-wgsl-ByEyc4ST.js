import{a as e,i as t}from"./cell-index-wgsl-Cw7syfdA.js";var n=`
const S2_HIERARCHY_IJ_TO_POSITION = array<u32, 16>(${t.flatMap(e=>[0,1,2,3].map(t=>`${e.indexOf(t)}u`)).join(`,`)});
const S2_HIERARCHY_ORIENTATION = array<u32, 4>(${e.map(e=>`${e}u`).join(`,`)});
const S2_FACE_NORMAL = array<vec3i, 6>(vec3i(1,0,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1));
const S2_FACE_COLUMN = array<vec3i, 6>(vec3i(0,1,0),vec3i(-1,0,0),vec3i(-1,0,0),vec3i(0,0,-1),vec3i(0,0,-1),vec3i(0,1,0));
const S2_FACE_ROW = array<vec3i, 6>(vec3i(0,0,1),vec3i(0,0,1),vec3i(0,-1,0),vec3i(0,-1,0),vec3i(1,0,0),vec3i(1,0,0));
fn s2Encode(face: u32, coordinate: vec2u, level: u32) -> vec2u {
  var orientation = face & 1u;
  var key = dggs_u64_shift_left(vec2u(0u, face), 61u) | dggs_u64_shift_left(vec2u(0u, 1u), 60u - 2u * level);
  for (var step = 0u; step < level; step++) {
    let bit = level - step - 1u;
    let quadrant = (((coordinate.x >> bit) & 1u) << 1u) | ((coordinate.y >> bit) & 1u);
    let position = S2_HIERARCHY_IJ_TO_POSITION[orientation * 4u + quadrant];
    orientation ^= S2_HIERARCHY_ORIENTATION[position];
    key |= dggs_u64_shift_left(vec2u(0u, position), 59u - 2u * step);
  }
  return key;
}
fn s2IntegerDot(first: vec3i, second: vec3i) -> i32 { let product = first * second; return product.x + product.y + product.z; }
fn s2Neighbor(cell: vec2u, slot: u32) -> vec2u {
  let level = dggs_s2_get_level(cell); let size = 1u << level;
  let face = dggs_s2_get_face(cell); let coordinate = dggs_s2_get_ij(cell);
  var next = vec2i(coordinate);
  if (slot < 2u) { next.x += select(-1, 1, slot == 1u); } else { next.y += select(-1, 1, slot == 3u); }
  if (all(next >= vec2i(0)) && all(next < vec2i(i32(size)))) { return s2Encode(face, vec2u(next), level); }
  let direction = select(-1, 1, slot == 1u || slot == 3u);
  let neighborNormal = select(S2_FACE_COLUMN[face], S2_FACE_ROW[face], slot >= 2u) * direction;
  let tangent = select(S2_FACE_ROW[face], S2_FACE_COLUMN[face], slot >= 2u);
  let varyingCoordinate = select(coordinate.y, coordinate.x, slot >= 2u);
  var neighborFace = 0u;
  for (var candidate = 0u; candidate < 6u; candidate++) { if (all(S2_FACE_NORMAL[candidate] == neighborNormal)) { neighborFace = candidate; break; } }
  let fixedColumn = s2IntegerDot(S2_FACE_NORMAL[face], S2_FACE_COLUMN[neighborFace]);
  let fixedRow = s2IntegerDot(S2_FACE_NORMAL[face], S2_FACE_ROW[neighborFace]);
  let column = select(select(varyingCoordinate, size - 1u - varyingCoordinate, s2IntegerDot(tangent, S2_FACE_COLUMN[neighborFace]) < 0), select(0u, size - 1u, fixedColumn > 0), fixedColumn != 0);
  let row = select(select(varyingCoordinate, size - 1u - varyingCoordinate, s2IntegerDot(tangent, S2_FACE_ROW[neighborFace]) < 0), select(0u, size - 1u, fixedRow > 0), fixedRow != 0);
  return s2Encode(neighborFace, vec2u(column, row), level);
}
`;export{n as t};