// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

// The forward H3 algorithm (`geoToFaceIjk`, `hex2dToCoordIjk`) is derived from Uber's Apache-2.0 H3.

import {dggs} from '@luma.gl/shaders';
import {getExactDegreeReductionWGSL} from './cell-index-wgsl';

/** Rows of each H3 face table in the `dggs` shader source: one per icosahedron face. */
const H3_FACE_COUNT = 20;

/**
 * Parses a per-face `vec2f(...)` / `vec3f(...)` constant table out of the shared `dggs` shader
 * source.
 *
 * The text layout of the published shader is not a public contract (its WGSL functions are), so
 * the result is checked: the function must be found and hold exactly 20 rows of `columnCount`
 * finite numbers. Anything else throws at module load, naming the table, instead of building face
 * bases from an empty or shifted table. The check runs once per table, when this module is first
 * imported.
 *
 * @internal
 */
export function parseDggsFaceTable(
  functionName: string,
  columnCount: number,
  source: string = dggs.source
): number[][] {
  const start = source.indexOf(`fn ${functionName}`);
  const end = start < 0 ? -1 : source.indexOf('return values', start);
  const rows =
    end < 0
      ? []
      : [...source.slice(start, end).matchAll(/vec[23]f\(([^)]*)\)/g)].map(match =>
          match[1].split(',').map(Number)
        );
  const isWellFormed =
    rows.length === H3_FACE_COUNT &&
    rows.every(row => row.length === columnCount && row.every(Number.isFinite));
  if (!isWellFormed) {
    throw new Error(
      `h3-index-extended: dggs shader table ${functionName} has ${rows.length} rows; expected ${H3_FACE_COUNT} rows of ${columnCount} finite numbers. The layout of the @luma.gl/shaders dggs source changed.`
    );
  }
  return rows;
}

/** Formats a float64 as a double-float `vec2f(high, low)` literal. */
function formatDoubleFloat(value: number): string {
  const high = Math.fround(value);
  const low = Math.fround(value - high);
  return `vec2f(${high.toPrecision(9)}, ${low.toPrecision(9)})`;
}

/** Float64 face center, hex i axis and hex j axis unit vectors for each of the 20 icosahedron faces. */
/** Binary64 face bases shared by forward indexing and inverse cell measures. @internal */
function createFaceBasisVectors(): number[][] {
  const centers = parseDggsFaceTable('dggs_h3_get_face_center_geo', 2);
  const azimuths = parseDggsFaceTable('dggs_h3_get_face_axis_azimuths', 3);
  const values: number[][] = [];
  centers.forEach(([latitude, longitude], face) => {
    const sinLat = Math.sin(latitude);
    const cosLat = Math.cos(latitude);
    const sinLng = Math.sin(longitude);
    const cosLng = Math.cos(longitude);
    const north = [-sinLat * cosLng, -sinLat * sinLng, cosLat];
    const east = [-sinLng, cosLng, 0];
    const direction = (azimuth: number) =>
      [0, 1, 2].map(axis => Math.cos(azimuth) * north[axis] + Math.sin(azimuth) * east[axis]);
    values.push(
      [cosLat * cosLng, cosLat * sinLng, sinLat],
      direction(azimuths[face][0]),
      direction(azimuths[face][0] - Math.PI / 2)
    );
  });
  return values;
}

const FACE_BASES = Object.freeze(createFaceBasisVectors().map(row => Object.freeze(row)));

/** Binary64 H3 bases; copies preserve the historical mutable return type. @internal */
export function getFaceBasisVectors(): number[][] {
  return FACE_BASES.map(row => [...row]);
}

const FACE_BASES_WGSL = getFaceBasisVectors().flat().map(formatDoubleFloat).join(',\n  ');

const RES0_U_GNOMONIC = 0.381966011250105;
const AP7_COS = 5 / (2 * Math.sqrt(7));
const AP7_SIN = Math.sqrt(3) / (2 * Math.sqrt(7));

// sqrt(7)^resolution / RES0_U_GNOMONIC: the exact integer power for even resolutions, times sqrt(7) for odd.
const RESOLUTION_SCALES_WGSL = Array.from({length: 16}, (_, resolution) =>
  formatDoubleFloat(
    (7 ** Math.floor(resolution / 2) * (resolution % 2 ? Math.sqrt(7) : 1)) / RES0_U_GNOMONIC
  )
).join(',\n  ');

const FACTORIALS = [1];
for (let term = 1; term <= 20; term++) {
  FACTORIALS.push(FACTORIALS[term - 1] * term);
}
// Taylor coefficients 1/n! with alternating signs: sine uses odd n, cosine even n.
const SINE_COEFFICIENTS_WGSL = Array.from({length: 10}, (_, term) =>
  formatDoubleFloat(((term % 2 ? -1 : 1) * 1) / FACTORIALS[2 * term + 1])
).join(',\n  ');
const COSINE_COEFFICIENTS_WGSL = Array.from({length: 10}, (_, term) =>
  formatDoubleFloat(((term % 2 ? -1 : 1) * 1) / FACTORIALS[2 * term])
).join(',\n  ');

/**
 * Extended-precision (double-float) forward H3 `latLngToCell`, to be concatenated AFTER
 * `dggs.source` and {@link H3_INDEX_WGSL} (it reuses `cellIndexH3FaceIjkToIndex`).
 *
 * Defines `fn cellIndexH3ExtendedFromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u` with the
 * same contract as `cellIndexH3FromLngLat`. Every position quantity is a `vec2f(high, low)`
 * double-float (about 44 bits of significand) built from error-free two-sum and Dekker two-product
 * transformations, so the unit vector, face projection and hex2d scaling keep roughly 1e-14
 * relative accuracy instead of 6e-8. The face constants are generated in float64 from the `dggs`
 * face centers and axis azimuths. The final cell test uses the exact fractional part of the hex2d
 * coordinate.
 *
 * The input degrees are still f32, so the result equals h3-js `latLngToCell` evaluated on the same
 * f32-rounded degrees (the f32 input is a quantization of the point, not a computation error).
 */
export const H3_INDEX_EXTENDED_WGSL: string = /* wgsl */ `\
const CELL_INDEX_H3_EXT_FACE_BASES = array<vec2f, 180>(
  ${FACE_BASES_WGSL}
);
const CELL_INDEX_H3_EXT_RESOLUTION_SCALES = array<vec2f, 16>(
  ${RESOLUTION_SCALES_WGSL}
);
const CELL_INDEX_H3_EXT_SINE_COEFFICIENTS = array<vec2f, 10>(
  ${SINE_COEFFICIENTS_WGSL}
);
const CELL_INDEX_H3_EXT_COSINE_COEFFICIENTS = array<vec2f, 10>(
  ${COSINE_COEFFICIENTS_WGSL}
);
const CELL_INDEX_H3_EXT_RADIANS_PER_DEGREE = ${formatDoubleFloat(Math.PI / 180)};
const CELL_INDEX_H3_EXT_AP7_COS = ${formatDoubleFloat(AP7_COS)};
const CELL_INDEX_H3_EXT_AP7_SIN = ${formatDoubleFloat(AP7_SIN)};
const CELL_INDEX_H3_EXT_INVERSE_SQRT3_2 = ${formatDoubleFloat(2 / Math.sqrt(3))};

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
${getExactDegreeReductionWGSL('cellIndexH3ExtReduceDegrees')}
fn cellIndexH3ExtSinCosDegrees(degrees: f32) -> vec4f {
  let reducedInput = cellIndexH3ExtReduceDegrees(degrees);
  let quadrant = round(reducedInput * (1.0 / 90.0));
  let reducedDegrees = reducedInput - quadrant * 90.0;
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
`;
