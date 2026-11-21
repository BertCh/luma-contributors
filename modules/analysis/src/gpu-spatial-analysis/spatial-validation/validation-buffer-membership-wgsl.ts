// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact inclusive radius membership of stored float32 coordinates. The ordinary dot-product
 * comparison handles separated values. Near a boundary, base-65536 integer arithmetic retains
 * all coordinate bits, including subnormals, without relying on float residuals under fast math.
 * A finite float32 is an integer times 2^-149: 18 limbs hold a coordinate difference and 36 hold
 * its square. All loops have these fixed bounds; private arrays occupy less than 2 KiB.
 * @internal
 */
function getValidationSquaredRadius(configuredRadius: number): bigint {
  // Compare integer distance² to floor(radius² * 2^298). This preserves every bit of the
  // caller's binary64 radius too: rounding that radius to f32 would move the boundary.
  const storage = new DataView(new ArrayBuffer(8));
  storage.setFloat64(0, configuredRadius, true);
  const bits = storage.getBigUint64(0, true);
  const exponent = Number((bits >> 52n) & 2047n);
  const significand = (bits & ((1n << 52n) - 1n)) | (exponent === 0 ? 0n : 1n << 52n);
  const shift = 2 * (exponent === 0 ? -1074 : exponent - 1023 - 52) + 298;
  return shift < 0
    ? (significand * significand) >> BigInt(-shift)
    : (significand * significand) << BigInt(shift);
}

function getValidationCoordinateInteger(value: number): bigint {
  const storage = new DataView(new ArrayBuffer(4));
  storage.setFloat32(0, value, true);
  const bits = storage.getUint32(0, true),
    exponent = (bits >>> 23) & 255;
  const magnitude =
    BigInt((bits & 0x7fffff) | (exponent === 0 ? 0 : 0x800000)) <<
    BigInt(exponent === 0 ? 0 : exponent - 1);
  return bits >>> 31 ? -magnitude : magnitude;
}

function getValidationSquaredDistance(
  firstX: number,
  firstY: number,
  secondX: number,
  secondY: number
): bigint {
  const horizontal =
      getValidationCoordinateInteger(firstX) - getValidationCoordinateInteger(secondX),
    vertical = getValidationCoordinateInteger(firstY) - getValidationCoordinateInteger(secondY);
  return horizontal * horizontal + vertical * vertical;
}

/** Adaptive membership compares to the defining neighbor, never its rounded radius. @internal */
export function isWithinValidationNeighborOnCPU(
  firstX: number,
  firstY: number,
  centerX: number,
  centerY: number,
  neighborX: number,
  neighborY: number
): boolean {
  const distance = Math.hypot(firstX - centerX, firstY - centerY),
    boundary = Math.hypot(neighborX - centerX, neighborY - centerY);
  if (distance < boundary * (1 - 8 * Number.EPSILON)) return true;
  if (distance > boundary * (1 + 8 * Number.EPSILON)) return false;
  return (
    getValidationSquaredDistance(firstX, firstY, centerX, centerY) <=
    getValidationSquaredDistance(neighborX, neighborY, centerX, centerY)
  );
}

/** Inclusive Euclidean buffer membership on the CPU twin's stored float32 coordinates. @internal */
export function isWithinValidationRadiusOnCPU(
  firstX: number,
  firstY: number,
  secondX: number,
  secondY: number,
  radius: number
): boolean {
  const distance = Math.hypot(firstX - secondX, firstY - secondY);
  if (distance < radius * (1 - 8 * Number.EPSILON)) return true;
  if (distance > radius * (1 + 8 * Number.EPSILON)) return false;
  return (
    getValidationSquaredDistance(firstX, firstY, secondX, secondY) <=
    getValidationSquaredRadius(radius)
  );
}

/** Generates the GPU counterpart, retaining the caller's binary64 configured radius. @internal */
export function getValidationBufferMembershipWGSL(configuredRadius: number): string {
  const squared = getValidationSquaredRadius(configuredRadius);
  const limbs = Array.from(
    {length: 36},
    (_, index) => `${Number((squared >> BigInt(index * 16)) & 65535n)}u`
  );
  return /* wgsl */ `
const VALIDATION_CONFIGURED_RADIUS_SQUARED: array<u32, 36> = array<u32, 36>(${limbs.join(', ')});
fn validationSameCoordinate(first: f32, second: f32) -> bool {
  let firstBits = bitcast<u32>(first); let secondBits = bitcast<u32>(second);
  return firstBits == secondBits || ((firstBits | secondBits) & 0x7fffffffu) == 0u;
}
fn validationMagnitude(bits: u32) -> array<u32, 18> {
  var words: array<u32, 18>;
  let exponent = (bits >> 23u) & 255u;
  let mantissa = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
  let shift = select(0u, exponent - 1u, exponent != 0u);
  for (var bit = 0u; bit < 24u; bit++) {
    let destination = shift + bit;
    words[destination / 16u] |= ((mantissa >> bit) & 1u) << (destination % 16u);
  }
  return words;
}
fn validationDifference(first: f32, second: f32) -> array<u32, 18> {
  let firstBits = bitcast<u32>(first);
  let secondBits = bitcast<u32>(second);
  let firstMagnitude = firstBits & 0x7fffffffu;
  let secondMagnitude = secondBits & 0x7fffffffu;
  let larger = validationMagnitude(max(firstMagnitude, secondMagnitude));
  let smaller = validationMagnitude(min(firstMagnitude, secondMagnitude));
  let add = ((firstBits ^ secondBits) & 0x80000000u) != 0u;
  var difference: array<u32, 18>;
  var carry = 0u;
  for (var limb = 0u; limb < 18u; limb++) {
    if (add) {
      let total = larger[limb] + smaller[limb] + carry;
      difference[limb] = total & 65535u;
      carry = total >> 16u;
    } else {
      let subtrahend = smaller[limb] + carry;
      difference[limb] = (larger[limb] - subtrahend) & 65535u;
      carry = u32(larger[limb] < subtrahend);
    }
  }
  return difference;
}
fn validationSquare(value: array<u32, 18>) -> array<u32, 36> {
  var result: array<u32, 36>;
  for (var first = 0u; first < 18u; first++) {
    var carry = 0u;
    for (var second = 0u; second < 18u; second++) {
      let product = value[first] * value[second];
      let low = (product & 65535u) + result[first + second] + (carry & 65535u);
      result[first + second] = low & 65535u;
      carry = (product >> 16u) + (carry >> 16u) + (low >> 16u);
    }
    result[first + 18u] = carry;
  }
  return result;
}
fn isWithinValidationRadius(first: vec2<f32>, second: vec2<f32>, adaptiveRadius: f32) -> bool {
  ${
    configuredRadius === 0
      ? `if ((bitcast<u32>(adaptiveRadius) & 0x7fffffffu) == 0u) {
    return validationSameCoordinate(first.x, second.x) && validationSameCoordinate(first.y, second.y);
  }`
      : ''
  }
  let radius = max(adaptiveRadius, ${Math.fround(configuredRadius) === 0 ? '0.0' : `${Math.fround(configuredRadius)}f`});
  let difference = first - second;
  let squared = dot(difference, difference);
  let threshold = radius * radius;
  // Only use normal, finite arithmetic here. The integer fallback also covers zero, overflow
  // and subnormal products. The margin covers both subtractions, products and the addition.
  let squaredBits = bitcast<u32>(squared);
  let thresholdBits = bitcast<u32>(threshold);
  let horizontalSafe = validationSameCoordinate(first.x, second.x) || (bitcast<u32>(difference.x) & 0x7fffffffu) >= 0x20000000u;
  let verticalSafe = validationSameCoordinate(first.y, second.y) || (bitcast<u32>(difference.y) & 0x7fffffffu) >= 0x20000000u;
  if (horizontalSafe && verticalSafe && squaredBits >= 0x00800000u && squaredBits < 0x7f800000u &&
      thresholdBits >= 0x00800000u && thresholdBits < 0x7f800000u) {
    if (squared < threshold * 0.9999990463256836) { return true; }
    if (squared > threshold * 1.0000009536743164) { return false; }
  }
  let horizontal = validationSquare(validationDifference(first.x, second.x));
  let vertical = validationSquare(validationDifference(first.y, second.y));
  var limit = validationSquare(validationMagnitude(bitcast<u32>(adaptiveRadius) & 0x7fffffffu));
  for (var remaining = 36u; remaining > 0u; remaining--) {
    let limb = remaining - 1u;
    if (limit[limb] != VALIDATION_CONFIGURED_RADIUS_SQUARED[limb]) {
      if (limit[limb] < VALIDATION_CONFIGURED_RADIUS_SQUARED[limb]) {
        for (var copy = 0u; copy < 36u; copy++) { limit[copy] = VALIDATION_CONFIGURED_RADIUS_SQUARED[copy]; }
      }
      break;
    }
  }
  var total: array<u32, 36>;
  var carry = 0u;
  for (var limb = 0u; limb < 36u; limb++) {
    let sum = horizontal[limb] + vertical[limb] + carry;
    total[limb] = sum & 65535u;
    carry = sum >> 16u;
  }
  for (var remaining = 36u; remaining > 0u; remaining--) {
    let limb = remaining - 1u;
    if (total[limb] != limit[limb]) { return total[limb] < limit[limb]; }
  }
  return true;
}
fn validationDistanceSquare(first: vec2<f32>, second: vec2<f32>) -> array<u32, 36> {
  let horizontal = validationSquare(validationDifference(first.x, second.x));
  let vertical = validationSquare(validationDifference(first.y, second.y));
  var result: array<u32, 36>; var carry = 0u;
  for (var limb = 0u; limb < 36u; limb++) {
    let total = horizontal[limb] + vertical[limb] + carry;
    result[limb] = total & 65535u; carry = total >> 16u;
  }
  return result;
}
fn isWithinValidationNeighbor(point: vec2<f32>, center: vec2<f32>, neighbor: vec2<f32>) -> bool {
  if (validationSameCoordinate(point.x, neighbor.x) && validationSameCoordinate(point.y, neighbor.y)) { return true; }
  let squared = validationDistanceSquare(point, center);
  let limit = validationDistanceSquare(neighbor, center);
  for (var remaining = 36u; remaining > 0u; remaining--) {
    let limb = remaining - 1u;
    if (squared[limb] != limit[limb]) { return squared[limb] < limit[limb]; }
  }
  return true;
}
fn isValidationNeighborCloser(point: vec2<f32>, center: vec2<f32>, neighbor: vec2<f32>) -> bool {
  let first=distance(point,center);let second=distance(neighbor,center);
  if(first>1e-18 && second>1e-18 && first<1e18 && second<1e18){
    if(first<second*(1.0-0.000001)){return true;}
    if(first>second*(1.0+0.000001)){return false;}
  }
  let squared=validationDistanceSquare(point,center);
  let limit=validationDistanceSquare(neighbor,center);
  for(var remaining=36u;remaining>0u;remaining--){
    let limb=remaining-1u;
    if(squared[limb]!=limit[limb]){return squared[limb]<limit[limb];}
  }
  return false;
}
`;
}
