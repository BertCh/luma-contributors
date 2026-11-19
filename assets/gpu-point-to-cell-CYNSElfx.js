import{s as e}from"./kernel-C3tfQOn4.js";import{n as t,o as n,r,s as i,t as a}from"./cell-index-wgsl-Cw7syfdA.js";import{t as o}from"./cell-row-kernel-PAKzUnKS.js";import{n as s,o as c,r as l}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as u,T as d,b as f,c as p,f as m,p as h,v as g,x as _,y as v}from"./gpu-contributor-utils-BIwLQON5.js";import{a as y}from"./cell-validation-D7P5BmPo.js";import{n as b,r as x,t as S}from"./cell-keys-Dl1tIQ_Z.js";import{c as C}from"./wgsl-common-DfyXiYB0.js";import{i as w,r as T,v as E}from"./spatial-context-DwZEd4EZ.js";var D=[16,18,24,33,30,800,305,816,-1,8,1285,1290,22,-1,-1,297,-1,-1,4,1280,-1,271,-1,-1,-1,-1,-1,2,6,14,10,11,785,280,791,-1,0,1281,1289,5,-1,-1,274,-1,-1,260,1283,-1,264,-1,-1,-1,-1,-1,7,21,38,9,19,802,270,788,-1,3,1293,1309,1,-1,-1,262,-1,-1,516,1292,-1,256,-1,-1,-1,-1,-1,26,42,58,29,43,830,294,815,-1,12,1308,1324,13,-1,-1,277,-1,-1,772,1295,-1,259,-1,-1,-1,-1,-1,31,41,49,44,53,829,314,833,-1,15,1302,1313,28,-1,-1,298,-1,-1,1028,1288,-1,268,-1,-1,-1,-1,-1,50,48,817,32,798,801,792,786,-1,70,67,834,820,-1,-1,805,-1,-1,83,855,-1,842,-1,-1,-1,-1,-1,25,23,792,17,779,778,782,774,-1,45,39,805,803,-1,-1,795,-1,-1,63,827,-1,824,-1,-1,-1,-1,-1,36,20,782,34,787,777,806,789,-1,55,40,795,822,-1,-1,819,-1,-1,72,828,-1,841,-1,-1,-1,-1,-1,64,47,806,62,811,797,826,810,-1,84,69,819,850,-1,-1,844,-1,-1,97,857,-1,866,-1,-1,-1,-1,-1,75,65,826,61,821,812,817,809,-1,94,86,844,849,-1,-1,834,-1,-1,107,872,-1,869,-1,-1,-1,-1,-1,57,59,831,74,846,847,851,860,-1,37,807,813,52,-1,-1,838,-1,-1,24,791,-1,800,-1,-1,-1,-1,-1,46,60,840,56,836,848,831,845,-1,27,808,823,35,-1,-1,813,-1,-1,14,788,-1,785,-1,-1,-1,-1,-1,71,89,865,73,859,871,840,856,-1,51,837,852,54,-1,-1,823,-1,-1,38,815,-1,802,-1,-1,-1,-1,-1,96,104,875,98,878,883,865,879,-1,76,854,862,82,-1,-1,852,-1,-1,58,833,-1,830,-1,-1,-1,-1,-1,85,87,851,101,870,868,875,880,-1,66,835,838,81,-1,-1,862,-1,-1,49,816,-1,829,-1,-1,-1,-1,-1,95,92,83,79,78,842,319,827,-1,109,108,1380,349,-1,-1,333,-1,-1,1141,1398,-1,362,-1,-1,-1,-1,-1,90,77,63,80,68,824,328,828,-1,106,93,1359,355,-1,-1,344,-1,-1,885,1389,-1,369,-1,-1,-1,-1,-1,105,88,72,103,91,841,353,857,-1,113,99,1360,372,-1,-1,367,-1,-1,629,1386,-1,377,-1,-1,-1,-1,-1,119,111,97,115,110,866,363,872,-1,121,116,1383,376,-1,-1,368,-1,-1,373,1393,-1,374,-1,-1,-1,-1,-1,114,112,107,100,102,869,339,855,-1,118,120,1395,364,-1,-1,348,-1,-1,117,1401,-1,365,-1,-1,-1,-1,-1],O=[[4,-1,-1],[14,2,6],[24,1,5],[38,3,7],[49,0,9],[58,4,8],[63,11,15],[72,12,16],[83,10,19],[97,13,17],[107,14,18],[117,-1,-1]],k=D.map(e=>e<0?`65535u`:`${e}u`).join(`, `),A=O.filter(([,e])=>e>=0).map(([e,t,n])=>`(baseCell == ${e}u && (face == ${t}u || face == ${n}u))`).join(` ||
    `),j=`\
const CELL_INDEX_H3_BASE_CELL_TABLE = array<u32, 540>(${k});
const CELL_INDEX_H3_SQRT7_POWERS = array<f32, 16>(${Array.from({length:16},(e,t)=>Math.fround(Math.sqrt(7)**t).toPrecision(9)).join(`, `)});
const CELL_INDEX_H3_ROTATE_CCW = array<u32, 8>(0u, 5u, 3u, 1u, 6u, 4u, 2u, 7u);
const CELL_INDEX_H3_NO_CELL: u32 = 65535u;
// cos and sin of the aperture-7 rotation: 5 / (2 sqrt 7) and sqrt 3 / (2 sqrt 7).
const CELL_INDEX_H3_AP7_COS: f32 = 0.944911182523068;
const CELL_INDEX_H3_AP7_SIN: f32 = 0.3273268353539886;

fn cellIndexH3IsCwOffset(baseCell: u32, face: u32) -> bool {
  return ${A};
}

// (sin, cos) of an angle in degrees. The angle is reduced to [-45, 45] degrees exactly in degrees,
// then a Taylor polynomial gives full f32 accuracy independent of the builtin sin/cos precision.
fn cellIndexH3SinCosDegrees(degrees: f32) -> vec2f {
  let quadrant = round(degrees * (1.0 / 90.0));
  let reduced = (degrees - quadrant * 90.0) * 0.017453292519943295;
  let x2 = reduced * reduced;
  let sine = reduced * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 +
    x2 * (1.0 / 362880.0 + x2 * (-1.0 / 39916800.0))))));
  let cosine = 1.0 + x2 * (-0.5 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 +
    x2 * (1.0 / 40320.0 + x2 * (-1.0 / 3628800.0 + x2 * (1.0 / 479001600.0))))));
  let turn = u32(i32(quadrant)) & 3u;
  if (turn == 1u) {
    return vec2f(cosine, -sine);
  }
  if (turn == 2u) {
    return vec2f(-sine, -cosine);
  }
  if (turn == 3u) {
    return vec2f(-cosine, sine);
  }
  return vec2f(sine, cosine);
}

fn cellIndexH3GetNearestFace(point: vec3f) -> u32 {
  var face = 0u;
  var bestAlignment = -2.0;
  for (var candidate = 0u; candidate < 20u; candidate++) {
    let alignment = dot(point, dggs_h3_get_face_unit_vector_basis(candidate, 0u));
    if (alignment > bestAlignment) {
      bestAlignment = alignment;
      face = candidate;
    }
  }
  return face;
}

// H3 _hex2dToCoordIJK: quantizes a hex2d point to the containing cell's normalized IJK.
fn cellIndexH3Hex2dToIjk(point: vec2f) -> vec3i {
  let a1 = abs(point.x);
  let a2 = abs(point.y);
  let x2 = a2 / DGGS_H3_SQRT3_2;
  let x1 = a1 + x2 * 0.5;
  let m1 = i32(x1);
  let m2 = i32(x2);
  let r1 = x1 - f32(m1);
  let r2 = x2 - f32(m2);
  var i = 0;
  var j = 0;
  if (r1 < 0.5) {
    if (r1 < 1.0 / 3.0) {
      i = m1;
      j = select(m2 + 1, m2, r2 < (1.0 + r1) * 0.5);
    } else {
      j = select(m2 + 1, m2, r2 < 1.0 - r1);
      i = select(m1, m1 + 1, (1.0 - r1) <= r2 && r2 < 2.0 * r1);
    }
  } else if (r1 < 2.0 / 3.0) {
    j = select(m2 + 1, m2, r2 < 1.0 - r1);
    i = select(m1 + 1, m1, (2.0 * r1 - 1.0) < r2 && r2 < 1.0 - r1);
  } else {
    i = m1 + 1;
    j = select(m2 + 1, m2, r2 < r1 * 0.5);
  }
  if (point.x < 0.0) {
    if ((j & 1) == 0) {
      i -= 2 * (i - j / 2);
    } else {
      i -= 2 * (i - (j + 1) / 2) + 1;
    }
  }
  if (point.y < 0.0) {
    i -= (2 * j + 1) / 2;
    j = -j;
  }
  return dggs_h3_ijk_normalize(vec3i(i, j, 0));
}

// H3 _upAp7: Class III child to Class II parent (counter-clockwise aperture 7). Upstream dggs
// only has dggs_h3_up_ap7r (clockwise, (2i+j)/7); this one is (3i-j)/7, so it is not a duplicate.
fn cellIndexH3UpAp7(coord: vec3i) -> vec3i {
  let i = coord.x - coord.z;
  let j = coord.y - coord.z;
  return dggs_h3_ijk_normalize(vec3i(
    dggs_h3_divide_round_nearest_seven(3 * i - j),
    dggs_h3_divide_round_nearest_seven(i + 2 * j),
    0
  ));
}

fn cellIndexH3RotateDigitsCcw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = dggs_h3_get_digit(index, digitResolution);
    result = dggs_u64_set_bits(
      result,
      dggs_h3_digit_bit_offset(digitResolution),
      3u,
      CELL_INDEX_H3_ROTATE_CCW[digit]
    );
  }
  return result;
}

fn cellIndexH3RotateDigitsCw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = dggs_h3_get_digit(index, digitResolution);
    result = dggs_u64_set_bits(
      result,
      dggs_h3_digit_bit_offset(digitResolution),
      3u,
      dggs_h3_rotate_digit_60_cw(digit)
    );
  }
  return result;
}

// H3 _h3RotatePent60ccw: rotates counter-clockwise, skipping the deleted K subsequence.
fn cellIndexH3RotatePentagonDigitsCcw(index: vec2u) -> vec2u {
  var result = index;
  let resolution = dggs_h3_get_resolution(index);
  var foundFirstNonZeroDigit = false;
  for (var digitResolution = 1u; digitResolution <= resolution; digitResolution++) {
    let digit = CELL_INDEX_H3_ROTATE_CCW[dggs_h3_get_digit(result, digitResolution)];
    result = dggs_u64_set_bits(result, dggs_h3_digit_bit_offset(digitResolution), 3u, digit);
    if (!foundFirstNonZeroDigit && digit != 0u) {
      foundFirstNonZeroDigit = true;
      if (dggs_h3_get_leading_non_zero_digit(result) == 1u) {
        result = cellIndexH3RotateDigitsCcw(result);
      }
    }
  }
  return result;
}

// H3 _faceIjkToH3 for an IJK at 'resolution' on 'face'. Returns vec2u(0u) when no base cell fits.
fn cellIndexH3FaceIjkToIndex(face: u32, coord: vec3i, resolution: u32) -> vec2u {
  // Header with base cell 0 and every digit unused (7); digits and base cell are filled in below.
  var index = vec2u((1u << 27u) | (resolution << 20u) | 0x1FFFu, 0xFFFFFFFFu);
  var current = coord;
  for (var parentResolution = resolution; parentResolution > 0u; parentResolution--) {
    let last = current;
    var lastCenter = vec3i(0);
    if (dggs_h3_is_resolution_class_iii(parentResolution)) {
      current = cellIndexH3UpAp7(current);
      lastCenter = dggs_h3_down_ap7(current);
    } else {
      current = dggs_h3_up_ap7r(current);
      lastCenter = dggs_h3_down_ap7r(current);
    }
    let difference = dggs_h3_ijk_normalize(last - lastCenter);
    index = dggs_u64_set_bits(
      index,
      dggs_h3_digit_bit_offset(parentResolution),
      3u,
      u32(4 * difference.x + 2 * difference.y + difference.z)
    );
  }
  if (any(current < vec3i(0)) || any(current > vec3i(2))) {
    return vec2u(0u);
  }
  let packed = CELL_INDEX_H3_BASE_CELL_TABLE[
    face * 27u + u32(current.x) * 9u + u32(current.y) * 3u + u32(current.z)
  ];
  if (packed == CELL_INDEX_H3_NO_CELL) {
    return vec2u(0u);
  }
  let baseCell = packed & 0xFFu;
  let rotations = packed >> 8u;
  index = dggs_u64_set_bits(index, 45u, 7u, baseCell);
  if (dggs_h3_is_base_cell_pentagon(baseCell)) {
    if (dggs_h3_get_leading_non_zero_digit(index) == 1u) {
      if (cellIndexH3IsCwOffset(baseCell, face)) {
        index = cellIndexH3RotateDigitsCw(index);
      } else {
        index = cellIndexH3RotateDigitsCcw(index);
      }
    }
    for (var rotation = 0u; rotation < rotations; rotation++) {
      index = cellIndexH3RotatePentagonDigitsCcw(index);
    }
  } else {
    for (var rotation = 0u; rotation < rotations; rotation++) {
      index = cellIndexH3RotateDigitsCcw(index);
    }
  }
  return index;
}

fn cellIndexH3FromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u {
  // Exponent-bit tests catch NaN and infinity without a NaN compare, which Metal may fold.
  let exponentBits = vec2u(bitcast<u32>(lngLatDegrees.x), bitcast<u32>(lngLatDegrees.y)) & vec2u(0x7f800000u);
  if (
    any(exponentBits == vec2u(0x7f800000u)) ||
    abs(lngLatDegrees.x) >= 3.0e38 ||
    abs(lngLatDegrees.y) >= 3.0e38 ||
    resolution > DGGS_H3_MAX_RESOLUTION
  ) {
    return vec2u(0u);
  }
  let longitudeSinCos = cellIndexH3SinCosDegrees(lngLatDegrees.x);
  let latitudeSinCos = cellIndexH3SinCosDegrees(lngLatDegrees.y);
  let point = vec3f(
    latitudeSinCos.y * longitudeSinCos.y,
    latitudeSinCos.y * longitudeSinCos.x,
    latitudeSinCos.x
  );
  let face = cellIndexH3GetNearestFace(point);
  let faceCenter = dggs_h3_get_face_unit_vector_basis(face, 0u);
  let inverseScale = 1.0 / (dot(point, faceCenter) * DGGS_H3_RES0_U_GNOMONIC);
  var hexPoint = vec2f(
    dot(point, dggs_h3_get_face_unit_vector_basis(face, 1u)),
    dot(point, dggs_h3_get_face_unit_vector_basis(face, 2u))
  ) * inverseScale;
  if (dggs_h3_is_resolution_class_iii(resolution)) {
    hexPoint = vec2f(
      CELL_INDEX_H3_AP7_COS * hexPoint.x + CELL_INDEX_H3_AP7_SIN * hexPoint.y,
      -CELL_INDEX_H3_AP7_SIN * hexPoint.x + CELL_INDEX_H3_AP7_COS * hexPoint.y
    );
  }
  hexPoint *= CELL_INDEX_H3_SQRT7_POWERS[resolution];
  return cellIndexH3FaceIjkToIndex(face, cellIndexH3Hex2dToIjk(hexPoint), resolution);
}
`,M=20;function N(e,t,n=i.source){let r=n.indexOf(`fn ${e}`),a=r<0?-1:n.indexOf(`return values`,r),o=a<0?[]:[...n.slice(r,a).matchAll(/vec[23]f\(([^)]*)\)/g)].map(e=>e[1].split(`,`).map(Number));if(!(o.length===M&&o.every(e=>e.length===t&&e.every(Number.isFinite))))throw Error(`h3-index-extended: dggs shader table ${e} has ${o.length} rows; expected ${M} rows of ${t} finite numbers. The layout of the @luma.gl/shaders dggs source changed.`);return o}function P(e){let t=Math.fround(e),n=Math.fround(e-t);return`vec2f(${t.toPrecision(9)}, ${n.toPrecision(9)})`}function F(){let e=N(`dggs_h3_get_face_center_geo`,2),t=N(`dggs_h3_get_face_axis_azimuths`,3),n=[];return e.forEach(([e,r],i)=>{let a=Math.sin(e),o=Math.cos(e),s=Math.sin(r),c=Math.cos(r),l=[-a*c,-a*s,o],u=[-s,c,0],d=e=>[0,1,2].map(t=>Math.cos(e)*l[t]+Math.sin(e)*u[t]);n.push([o*c,o*s,a],d(t[i][0]),d(t[i][0]-Math.PI/2))}),n}var I=F().flat().map(P).join(`,
  `),L=.381966011250105,R=5/(2*Math.sqrt(7)),z=Math.sqrt(3)/(2*Math.sqrt(7)),B=Array.from({length:16},(e,t)=>P(7**Math.floor(t/2)*(t%2?Math.sqrt(7):1)/L)).join(`,
  `),V=[1];for(let e=1;e<=20;e++)V.push(V[e-1]*e);var H=`\
const CELL_INDEX_H3_EXT_FACE_BASES = array<vec2f, 180>(
  ${I}
);
const CELL_INDEX_H3_EXT_RESOLUTION_SCALES = array<vec2f, 16>(
  ${B}
);
const CELL_INDEX_H3_EXT_SINE_COEFFICIENTS = array<vec2f, 10>(
  ${Array.from({length:10},(e,t)=>P((t%2?-1:1)*1/V[2*t+1])).join(`,
  `)}
);
const CELL_INDEX_H3_EXT_COSINE_COEFFICIENTS = array<vec2f, 10>(
  ${Array.from({length:10},(e,t)=>P((t%2?-1:1)*1/V[2*t])).join(`,
  `)}
);
const CELL_INDEX_H3_EXT_RADIANS_PER_DEGREE = ${P(Math.PI/180)};
const CELL_INDEX_H3_EXT_AP7_COS = ${P(R)};
const CELL_INDEX_H3_EXT_AP7_SIN = ${P(z)};
const CELL_INDEX_H3_EXT_INVERSE_SQRT3_2 = ${P(2/Math.sqrt(3))};

// Runtime-zero bits ORed into intermediate values that error-free transformations must keep exact.
// Metal (and other drivers with fast math) rewrite (a + b) - a to b, which erases every low word;
// routing the intermediate through an integer OR with a value the compiler cannot prove to be zero
// stops that reassociation. It is zero for every finite input (set in cellIndexH3ExtendedFromLngLat).
var<private> cellIndexH3ExtOpaqueBits: u32 = 0u;

fn cellIndexH3ExtKeep(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) | cellIndexH3ExtOpaqueBits);
}

fn cellIndexH3ExtTwoSum(a: f32, b: f32) -> vec2f {
  let sum = cellIndexH3ExtKeep(a + b);
  let virtualB = cellIndexH3ExtKeep(sum - a);
  let errorA = cellIndexH3ExtKeep(a - cellIndexH3ExtKeep(sum - virtualB));
  let errorB = cellIndexH3ExtKeep(b - virtualB);
  return vec2f(sum, errorA + errorB);
}

fn cellIndexH3ExtQuickTwoSum(a: f32, b: f32) -> vec2f {
  let sum = cellIndexH3ExtKeep(a + b);
  return vec2f(sum, b - cellIndexH3ExtKeep(sum - a));
}

// Veltkamp split of a float32 into two 12-bit halves.
fn cellIndexH3ExtSplit(a: f32) -> vec2f {
  let scaled = cellIndexH3ExtKeep(a * 4097.0);
  let high = cellIndexH3ExtKeep(scaled - cellIndexH3ExtKeep(scaled - a));
  return vec2f(high, a - high);
}

// Dekker product: exact a * b as (product, error) without relying on fused multiply-add.
fn cellIndexH3ExtTwoProduct(a: f32, b: f32) -> vec2f {
  let product = cellIndexH3ExtKeep(a * b);
  let splitA = cellIndexH3ExtSplit(a);
  let splitB = cellIndexH3ExtSplit(b);
  let highProduct = cellIndexH3ExtKeep(splitA.x * splitB.x);
  let cross = cellIndexH3ExtKeep(splitA.x * splitB.y + splitA.y * splitB.x);
  let lowProduct = cellIndexH3ExtKeep(splitA.y * splitB.y);
  let partial = cellIndexH3ExtKeep((highProduct - product) + cross);
  return vec2f(product, partial + lowProduct);
}

fn cellIndexH3ExtAdd(a: vec2f, b: vec2f) -> vec2f {
  let sum = cellIndexH3ExtTwoSum(a.x, b.x);
  return cellIndexH3ExtQuickTwoSum(sum.x, sum.y + cellIndexH3ExtKeep(a.y + b.y));
}

fn cellIndexH3ExtMultiply(a: vec2f, b: vec2f) -> vec2f {
  let product = cellIndexH3ExtTwoProduct(a.x, b.x);
  return cellIndexH3ExtQuickTwoSum(
    product.x,
    product.y + cellIndexH3ExtKeep(cellIndexH3ExtKeep(a.x * b.y) + cellIndexH3ExtKeep(a.y * b.x))
  );
}

fn cellIndexH3ExtMultiplyFloat(a: vec2f, b: f32) -> vec2f {
  let product = cellIndexH3ExtTwoProduct(a.x, b);
  return cellIndexH3ExtQuickTwoSum(product.x, product.y + cellIndexH3ExtKeep(a.y * b));
}

fn cellIndexH3ExtNegate(a: vec2f) -> vec2f {
  return -a;
}

fn cellIndexH3ExtDivide(a: vec2f, b: vec2f) -> vec2f {
  let first = a.x / b.x;
  var remainder = cellIndexH3ExtAdd(a, cellIndexH3ExtNegate(cellIndexH3ExtMultiplyFloat(b, first)));
  let second = remainder.x / b.x;
  remainder = cellIndexH3ExtAdd(remainder, cellIndexH3ExtNegate(cellIndexH3ExtMultiplyFloat(b, second)));
  let third = remainder.x / b.x;
  return cellIndexH3ExtAdd(cellIndexH3ExtQuickTwoSum(first, second), vec2f(third, 0.0));
}

fn cellIndexH3ExtIsGreater(a: vec2f, b: vec2f) -> bool {
  return a.x > b.x || (a.x == b.x && a.y > b.y);
}

// (sin, cos) of an angle in degrees as double-floats, packed (sin.xy, cos.xy). The angle is reduced
// to [-45, 45] degrees exactly, then converted to radians and summed with a double-float Taylor series.
fn cellIndexH3ExtSinCosDegrees(degrees: f32) -> vec4f {
  let quadrant = round(degrees * (1.0 / 90.0));
  let reducedDegrees = degrees - quadrant * 90.0;
  let reduced = cellIndexH3ExtMultiplyFloat(CELL_INDEX_H3_EXT_RADIANS_PER_DEGREE, reducedDegrees);
  let squared = cellIndexH3ExtMultiply(reduced, reduced);
  var sineSeries = CELL_INDEX_H3_EXT_SINE_COEFFICIENTS[9];
  var cosineSeries = CELL_INDEX_H3_EXT_COSINE_COEFFICIENTS[9];
  for (var term = 8; term >= 0; term--) {
    sineSeries = cellIndexH3ExtAdd(
      cellIndexH3ExtMultiply(sineSeries, squared),
      CELL_INDEX_H3_EXT_SINE_COEFFICIENTS[term]
    );
    cosineSeries = cellIndexH3ExtAdd(
      cellIndexH3ExtMultiply(cosineSeries, squared),
      CELL_INDEX_H3_EXT_COSINE_COEFFICIENTS[term]
    );
  }
  let sine = cellIndexH3ExtMultiply(sineSeries, reduced);
  let cosine = cosineSeries;
  let turn = u32(i32(quadrant)) & 3u;
  if (turn == 1u) {
    return vec4f(cosine, -sine);
  }
  if (turn == 2u) {
    return vec4f(-sine, -cosine);
  }
  if (turn == 3u) {
    return vec4f(-cosine, sine);
  }
  return vec4f(sine, cosine);
}

// Double-float dot product of the unit vector 'point' (x, y, z as vec2f) with face basis 'basis'.
fn cellIndexH3ExtDotBasis(point: array<vec2f, 3>, face: u32, basis: u32) -> vec2f {
  let offset = (face * 3u + basis) * 3u;
  var sum = cellIndexH3ExtMultiply(point[0], CELL_INDEX_H3_EXT_FACE_BASES[offset]);
  sum = cellIndexH3ExtAdd(sum, cellIndexH3ExtMultiply(point[1], CELL_INDEX_H3_EXT_FACE_BASES[offset + 1u]));
  return cellIndexH3ExtAdd(sum, cellIndexH3ExtMultiply(point[2], CELL_INDEX_H3_EXT_FACE_BASES[offset + 2u]));
}

// H3 _hex2dToCoordIJK on double-float input: the integer and fractional parts are split exactly.
fn cellIndexH3ExtHex2dToIjk(pointX: vec2f, pointY: vec2f) -> vec3i {
  let a1 = select(pointX, -pointX, pointX.x < 0.0);
  let a2 = select(pointY, -pointY, pointY.x < 0.0);
  let x2 = cellIndexH3ExtMultiply(a2, CELL_INDEX_H3_EXT_INVERSE_SQRT3_2);
  let x1 = cellIndexH3ExtAdd(a1, x2 * 0.5);
  var m1 = i32(floor(x1.x));
  var m2 = i32(floor(x2.x));
  var r1 = (x1.x - f32(m1)) + x1.y;
  var r2 = (x2.x - f32(m2)) + x2.y;
  if (r1 < 0.0) {
    m1 -= 1;
    r1 += 1.0;
  } else if (r1 >= 1.0) {
    m1 += 1;
    r1 -= 1.0;
  }
  if (r2 < 0.0) {
    m2 -= 1;
    r2 += 1.0;
  } else if (r2 >= 1.0) {
    m2 += 1;
    r2 -= 1.0;
  }
  var i = 0;
  var j = 0;
  if (r1 < 0.5) {
    if (r1 < 1.0 / 3.0) {
      i = m1;
      j = select(m2 + 1, m2, r2 < (1.0 + r1) * 0.5);
    } else {
      j = select(m2 + 1, m2, r2 < 1.0 - r1);
      i = select(m1, m1 + 1, (1.0 - r1) <= r2 && r2 < 2.0 * r1);
    }
  } else if (r1 < 2.0 / 3.0) {
    j = select(m2 + 1, m2, r2 < 1.0 - r1);
    i = select(m1 + 1, m1, (2.0 * r1 - 1.0) < r2 && r2 < 1.0 - r1);
  } else {
    i = m1 + 1;
    j = select(m2 + 1, m2, r2 < r1 * 0.5);
  }
  if (pointX.x < 0.0) {
    if ((j & 1) == 0) {
      i -= 2 * (i - j / 2);
    } else {
      i -= 2 * (i - (j + 1) / 2) + 1;
    }
  }
  if (pointY.x < 0.0) {
    i -= (2 * j + 1) / 2;
    j = -j;
  }
  return dggs_h3_ijk_normalize(vec3i(i, j, 0));
}

fn cellIndexH3ExtendedFromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u {
  let exponentBits = vec2u(bitcast<u32>(lngLatDegrees.x), bitcast<u32>(lngLatDegrees.y)) & vec2u(0x7f800000u);
  if (
    any(exponentBits == vec2u(0x7f800000u)) ||
    abs(lngLatDegrees.x) >= 3.0e38 ||
    abs(lngLatDegrees.y) >= 3.0e38 ||
    resolution > DGGS_H3_MAX_RESOLUTION
  ) {
    return vec2u(0u);
  }
  // 1 only for an infinite or NaN exponent, which returned above; the compiler cannot prove that.
  cellIndexH3ExtOpaqueBits = ((bitcast<u32>(lngLatDegrees.x) >> 23u) & 0xffu) / 255u;
  let longitudeSinCos = cellIndexH3ExtSinCosDegrees(lngLatDegrees.x);
  let latitudeSinCos = cellIndexH3ExtSinCosDegrees(lngLatDegrees.y);
  let point = array<vec2f, 3>(
    cellIndexH3ExtMultiply(latitudeSinCos.zw, longitudeSinCos.zw),
    cellIndexH3ExtMultiply(latitudeSinCos.zw, longitudeSinCos.xy),
    latitudeSinCos.xy
  );
  // Nearest face by maximum alignment: a float32 screen, then double-float for near ties.
  let approximate = vec3f(point[0].x, point[1].x, point[2].x);
  var bestApproximate = -2.0;
  for (var candidate = 0u; candidate < 20u; candidate++) {
    bestApproximate = max(bestApproximate, dot(approximate, dggs_h3_get_face_unit_vector_basis(candidate, 0u)));
  }
  var face = 0u;
  var bestAlignment = vec2f(-2.0, 0.0);
  for (var candidate = 0u; candidate < 20u; candidate++) {
    if (dot(approximate, dggs_h3_get_face_unit_vector_basis(candidate, 0u)) >= bestApproximate - 1.0e-5) {
      let alignment = cellIndexH3ExtDotBasis(point, candidate, 0u);
      if (cellIndexH3ExtIsGreater(alignment, bestAlignment)) {
        bestAlignment = alignment;
        face = candidate;
      }
    }
  }
  let inverseCosine = cellIndexH3ExtDivide(vec2f(1.0, 0.0), bestAlignment);
  var hexX = cellIndexH3ExtMultiply(cellIndexH3ExtDotBasis(point, face, 1u), inverseCosine);
  var hexY = cellIndexH3ExtMultiply(cellIndexH3ExtDotBasis(point, face, 2u), inverseCosine);
  if (dggs_h3_is_resolution_class_iii(resolution)) {
    let rotatedX = cellIndexH3ExtAdd(
      cellIndexH3ExtMultiply(CELL_INDEX_H3_EXT_AP7_COS, hexX),
      cellIndexH3ExtMultiply(CELL_INDEX_H3_EXT_AP7_SIN, hexY)
    );
    let rotatedY = cellIndexH3ExtAdd(
      cellIndexH3ExtNegate(cellIndexH3ExtMultiply(CELL_INDEX_H3_EXT_AP7_SIN, hexX)),
      cellIndexH3ExtMultiply(CELL_INDEX_H3_EXT_AP7_COS, hexY)
    );
    hexX = rotatedX;
    hexY = rotatedY;
  }
  hexX = cellIndexH3ExtMultiply(hexX, CELL_INDEX_H3_EXT_RESOLUTION_SCALES[resolution]);
  hexY = cellIndexH3ExtMultiply(hexY, CELL_INDEX_H3_EXT_RESOLUTION_SCALES[resolution]);
  return cellIndexH3FaceIjkToIndex(face, cellIndexH3ExtHex2dToIjk(hexX, hexY), resolution);
}
`,U=`GPUPointToCell`,W=4294967295,G=class{id;props;chunkPlan;estimatedCommandNodeCount;spatialContext;constructor(e){this.id=e.id??`point-to-cell`,this.props=e,this.spatialContext=E(this.id,e.spatialContext??T,w);let t=this.id;n(t,e.family,e.resolution),m(t,`positions rows`,e.positions.length,1,W);for(let[n,r,i]of[[`positions`,e.positions,[`float32x2`]],[`mask`,e.mask,[`uint32`]],[`output.cells`,e.output.cells,[`uint32x2`]],[`output.validity`,e.output.validity,[`uint32`]]])if(r){for(let e of p(r))u(e,i,`${t} ${n}`);d(t,r.length===e.positions.length,`${n} length must equal the row count`),f(t,n,e.positions,r)}y(t,[[`output.cells`,e.output.cells],[`output.validity`,e.output.validity]]),this.chunkPlan=l(t,{positions:e.positions,mask:e.mask,cells:e.output.cells,validity:e.output.validity}),this.estimatedCommandNodeCount=1+p(e.positions).filter(e=>e.length>0).length,h(t,this.estimatedCommandNodeCount),v(t,[e.output.cells,e.output.validity],[e.positions,e.mask])}getCommandNodes(t){let{id:n,props:r}=this,{family:i,resolution:a,output:l}=r,u=i===`h3`&&r.precision===`extended`;_(n,t,[r.positions,r.mask,l.cells,l.validity]);for(let i of[r.positions,r.mask,l.cells,l.validity])for(let r of i?p(i):[])g(t.device,n,`row chunk`,e(r).size,`reduce the number of rows per chunk`),m(n,`row chunk buffer byteLength`,r.buffer.byteLength,0,t.device.limits.maxBufferSize);c(t,n,this.chunkPlan);let d=s(t,{id:`${n}-chunks`,plan:this.chunkPlan}),f=[...d.nodes],h=p(r.positions),v=r.mask?p(r.mask):void 0,y=p(l.cells),b=l.validity?p(l.validity):void 0;for(let[e,r]of h.entries()){if(r.length===0)continue;let s=v?.[e],c=b?.[e],l=[{name:`positions`,view:r,components:2,type:`f32`,access:`read`},...s?[{name:`rowMask`,view:s,components:1,type:`u32`,access:`read`}]:[],{name:`cellsOut`,view:y[e],components:2,type:`u32`,access:`read_write`},...c?[{name:`validityOut`,view:c,components:1,type:`u32`,access:`read_write`}]:[]],p=q(i,a,u);f.push(o(t,{id:h.length===1?`${n}-keys`:`${n}-keys-chunk-${e}`,operation:U,variant:u?`h3-extended`:i,bindings:l,length:r.length,parameters:d.slots[e],declarations:`const RESOLUTION: u32 = ${a}u;
${K(i,u)}
${C}`,body:`let longitude = positions[positionsOffset + 2u * index];
  let latitude = positions[positionsOffset + 2u * index + 1u];
  var isValid = isFiniteBits(longitude) && isFiniteBits(latitude);
  ${s?`isValid = isValid && rowMask[rowMaskOffset + index] != 0u;`:``}
  var key = vec2u(0u);
  if (isValid) {
    key = ${p};
  }
  isValid = isValid && any(key != vec2u(0u));
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  ${c?`validityOut[validityOutOffset + index] = select(0u, 1u, isValid);`:``}`}))}return f}};function K(e,n){switch(e){case`quadbin`:return`${S}\n${b}`;case`quadkey`:return t;case`geohash`:return a;case`s2`:return r;case`h3`:return n?`${i.source}\n${j}\n${H}`:`${i.source}\n${j}`}}function q(e,t,n){switch(e){case`quadbin`:{let e=x(`quadbin`,t);return`cellGetKey(
      quadbinGetCompactKey(quadbinGetTileX(longitude, RESOLUTION), quadbinGetTileY(latitude, RESOLUTION)),
      ${e.headerHigh}u, RESOLUTION, ${e.lowBit}u)`}case`quadkey`:return`cellIndexQuadkeyFromLngLat(longitude, latitude, RESOLUTION)`;case`geohash`:return`cellIndexGeohashFromLngLat(longitude, latitude, RESOLUTION)`;case`s2`:return`cellIndexS2FromLngLat(longitude, latitude, RESOLUTION)`;case`h3`:return`${n?`cellIndexH3ExtendedFromLngLat`:`cellIndexH3FromLngLat`}(vec2f(longitude, latitude), RESOLUTION)`}}export{D as i,F as n,j as r,G as t};