// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {UINT64_WGSL} from './uint64-wgsl';

/** Adapts the public `getCoordinate` to finite, very large rectangular-grid bounds. @internal */
export const RECTANGULAR_GRID_COORDINATE_WGSL = /* wgsl */ `
${UINT64_WGSL}

fn isAtOrAboveWideGridEdge(signedCoordinates: vec3u, factors: vec3u) -> bool {
  // With the endpoint signs flipped, sum size*value - (size-edge)*minimum - edge*maximum.
  // In units of 2^-149 each term needs at most 253 + 24 + 32 bits. Their signed sum fits 320 bits.
  var difference: array<u32, 10>;
  for (var term = 0u; term < 3u; term++) {
    let bits = signedCoordinates[term];
    let exponent = (bits >> 23u) & 0xffu;
    let significand = (bits & 0x7fffffu) | select(0u, 0x800000u, exponent != 0u);
    let factor = factors[term];
    if (significand == 0u || factor == 0u) { continue; }
    let product = multiplyUint32Wide(significand, factor);
    let shift = max(exponent, 1u) - 1u;
    let firstWord = shift / 32u;
    let offset = shift % 32u;
    let lowerWords = shiftLeftUint64(product, offset);
    let upperWords = shiftRightUint64(product, 64u - offset);
    let productWords = vec3u(lowerWords.y, lowerWords.x, upperWords.y);

    // Accumulate the signed term using two's complement, extending its sign through the top word.
    let negative = select(0u, 1u, (bits & 0x80000000u) != 0u);
    let signMask = 0u - negative;
    var carry = negative;
    for (var word = firstWord; word < 10u; word++) {
      let relative = word - firstWord;
      if (relative >= 3u && carry == negative) { break; }
      let addend = select(0u, productWords[min(relative, 2u)], relative < 3u) ^ signMask;
      let partial = difference[word] + addend;
      let total = partial + carry;
      carry = select(0u, 1u, partial < addend || total < partial);
      difference[word] = total;
    }
  }
  return (difference[9] & 0x80000000u) == 0u;
}

fn getWideGridCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (value <= minimum) { return 0u; }
  if (value >= maximum) { return size - 1u; }
  let signedCoordinates = vec3u(bitcast<u32>(value),
    bitcast<u32>(minimum) ^ 0x80000000u, bitcast<u32>(maximum) ^ 0x80000000u);

  // Find the rightmost edge at or below the point. At most 32 steps for a u32 cell count.
  var lower = 0u;
  var upper = size;
  for (var step = 0u; step < 32u && upper - lower > 1u; step++) {
    let edge = lower + (upper - lower) / 2u;
    if (isAtOrAboveWideGridEdge(signedCoordinates, vec3u(size, size - edge, edge))) {
      lower = edge;
    } else {
      upper = edge;
    }
  }
  return lower;
}

fn getRectangularGridCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  let boundMagnitude = max(bitcast<u32>(minimum) & 0x7fffffffu, bitcast<u32>(maximum) & 0x7fffffffu);
  // Below 2^125 the width is less than 2^126, so a large width cannot flush its reciprocal.
  if (boundMagnitude >= 0x7e000000u && boundMagnitude < 0x7f800000u) {
    return getWideGridCoordinate(value, minimum, maximum, size);
  }
  return getCoordinate(value, minimum, maximum, size);
}`;
