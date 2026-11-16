// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Nonnegative D8 runoff in exact units of 2^-149. Float32 occupies at most 277 bits; summing at
 * most uint32-count terms needs 309 bits. Ten limbs cover that entire range; word 10 counts
 * invalid dependencies. This avoids rounded-prefix cancellation even across the full f32 range.
 * Each tour edge is stored in three four-word planes so no binding exceeds 32 bytes per cell.
 * @internal
 */
export const TERRAIN_TILED_WIDE_SUM_WGSL = `
alias TerrainRunoffSum = array<u32, 11>;
fn terrainZeroSum() -> TerrainRunoffSum { return TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); }
fn terrainEncodeRunoff(value: f32, state: u32) -> TerrainRunoffSum {
 var result = TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
 if (state != 1u) { result[10] = 1u; return result; }
 let bits = bitcast<u32>(value); let exponent = (bits >> 23u) & 255u;
 var significand = bits & 0x7fffffu; var shift = 0u;
 if (exponent != 0u) { significand |= 0x800000u; shift = exponent - 1u; }
 let word = shift / 32u; let bit = shift % 32u;
 result[word] = significand << bit; result[word + 1u] = (significand >> 1u) >> (31u - bit);
 return result;
}
fn terrainAddRunoff(first: TerrainRunoffSum, second: TerrainRunoffSum) -> TerrainRunoffSum {
 var result = TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); var carry = 0u;
 var firstWords = first; var secondWords = second;
 for (var word = 0u; word < 10u; word++) {
  let partial = firstWords[word] + secondWords[word]; let total = partial + carry;
  carry = u32(partial < firstWords[word]) | u32(total < partial); result[word] = total;
 }
 result[10] = first[10] + second[10]; return result;
}
fn terrainSubtractRunoff(first: TerrainRunoffSum, second: TerrainRunoffSum) -> TerrainRunoffSum {
 var result = TerrainRunoffSum(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); var borrow = 0u;
 var firstWords = first; var secondWords = second;
 for (var word = 0u; word < 10u; word++) {
  let partial = firstWords[word] - secondWords[word]; let total = partial - borrow;
  borrow = u32(firstWords[word] < secondWords[word]) | u32(partial < borrow); result[word] = total;
 }
 result[10] = first[10] - second[10]; return result;
}
fn terrainRunoffBitsBelow(value: TerrainRunoffSum, count: u32) -> bool {
 var words = value;
 let whole = count / 32u; let remainder = count % 32u; var found = false;
 for (var word = 0u; word < 10u; word++) {
  if (word < whole) { found = found || words[word] != 0u; }
  else if (word == whole) { found = found || (words[word] & ((1u << remainder) - 1u)) != 0u; }
 }
 return found;
}
fn terrainDecodeRunoff(value: TerrainRunoffSum) -> f32 {
 var words = value;
 var leading = 0u; var found = false;
 for (var reverse = 10u; reverse > 0u; reverse--) {
  let word = reverse - 1u;
  if (!found && words[word] != 0u) { leading = word * 32u + firstLeadingBit(words[word]); found = true; }
 }
 if (!found) { return 0.0; }
 if (leading < 23u) { return bitcast<f32>(value[0]); }
 let shift = leading - 23u; let word = shift / 32u; let bit = shift % 32u;
 var significand = words[word] >> bit;
 if (word + 1u < 10u) { significand |= (words[word + 1u] << 1u) << (31u - bit); }
 if (shift > 0u) {
  let position = shift - 1u; let roundBit = (words[position / 32u] >> (position % 32u)) & 1u;
  if (roundBit != 0u && (terrainRunoffBitsBelow(value, position) || (significand & 1u) != 0u)) { significand++; }
 }
 var exponent = leading - 22u;
 if (significand == 0x1000000u) { significand >>= 1u; exponent++; }
 if (exponent >= 255u) { var infinityBits = 0x7f800000u; return bitcast<f32>(infinityBits); }
 return bitcast<f32>((exponent << 23u) | (significand & 0x7fffffu));
}
`;

/** Fixed names for the three read planes of one terrain kernel. @internal */
export const TERRAIN_TILED_LOAD_SUM_WGSL = `
fn terrainLoadRunoff(edge: u32) -> TerrainRunoffSum {
 let first = firstPlaneOffset + edge * 4u; let second = secondPlaneOffset + edge * 4u; let third = thirdPlaneOffset + edge * 4u;
 return TerrainRunoffSum(firstPlane[first], firstPlane[first + 1u], firstPlane[first + 2u], firstPlane[first + 3u], secondPlane[second], secondPlane[second + 1u], secondPlane[second + 2u], secondPlane[second + 3u], thirdPlane[third], thirdPlane[third + 1u], thirdPlane[third + 2u]);
}
`;
/** Fixed names for three separate writable planes, including initialized padding. @internal */
export const TERRAIN_TILED_STORE_SUM_WGSL = `
fn terrainStoreRunoff(edge: u32, value: TerrainRunoffSum) {
 let first = firstOutputOffset + edge * 4u; let second = secondOutputOffset + edge * 4u; let third = thirdOutputOffset + edge * 4u;
 firstOutput[first] = value[0]; firstOutput[first + 1u] = value[1]; firstOutput[first + 2u] = value[2]; firstOutput[first + 3u] = value[3];
 secondOutput[second] = value[4]; secondOutput[second + 1u] = value[5]; secondOutput[second + 2u] = value[6]; secondOutput[second + 3u] = value[7];
 thirdOutput[third] = value[8]; thirdOutput[third + 1u] = value[9]; thirdOutput[third + 2u] = value[10]; thirdOutput[third + 3u] = 0u;
}
`;
