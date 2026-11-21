// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Canonical float32 bits: signed zeros agree; subnormal payloads remain distinct. */
export function canonicalizeInteractionCovariateBits(bits: number): number {
  return (bits & 0x7fffffff) === 0 ? 0 : bits;
}

/** Exact finite float32 in integer units of 2^-149; no floating arithmetic or flushing. */
function getExactCovariate(bits: number): bigint | undefined {
  const magnitude = bits & 0x7fffffff;
  const exponent = magnitude >>> 23;
  if (exponent === 255) return undefined;
  const significand = BigInt((magnitude & 0x7fffff) | (exponent === 0 ? 0 : 0x800000));
  const value = significand << BigInt(Math.max(0, exponent - 1));
  return bits >>> 31 ? -value : value;
}

/** Exact incidence-graph certificate, independently rooted in every connected component. */
export function isSpatialInteractionCovariateAbsorbed(
  origins: Uint32Array,
  destinations: Uint32Array,
  covariate: Float32Array,
  active: readonly number[],
  originCount: number,
  destinationCount: number,
  constraint: 'unconstrained' | 'origin' | 'destination' | 'doubly'
): boolean {
  const bits = new Uint32Array(covariate.buffer, covariate.byteOffset, covariate.length);
  if (constraint !== 'doubly') {
    const first = new Map<number, number>();
    for (const pair of active) {
      const group =
        constraint === 'unconstrained'
          ? 0
          : constraint === 'origin'
            ? origins[pair]
            : destinations[pair];
      const value = canonicalizeInteractionCovariateBits(bits[pair]);
      if ((value & 0x7fffffff) >= 0x7f800000) return false;
      if (first.has(group) && first.get(group) !== value) return false;
      first.set(group, value);
    }
    return true;
  }
  const adjacency = Array.from({length: originCount + destinationCount}, () => [] as number[]);
  for (const pair of active) {
    adjacency[origins[pair]].push(pair);
    adjacency[originCount + destinations[pair]].push(pair);
  }
  const potentials = new Map<number, bigint>();
  for (let root = 0; root < adjacency.length; root++) {
    if (potentials.has(root) || adjacency[root].length === 0) continue;
    potentials.set(root, 0n);
    const queue = [root];
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const vertex = queue[cursor];
      for (const pair of adjacency[vertex]) {
        const value = getExactCovariate(bits[pair]);
        if (value === undefined) return false;
        const other = vertex < originCount ? originCount + destinations[pair] : origins[pair];
        const expected = value - potentials.get(vertex)!;
        if (potentials.has(other)) {
          if (potentials.get(other) !== expected) return false;
        } else {
          potentials.set(other, expected);
          queue.push(other);
        }
      }
    }
  }
  return true;
}

/** Ten limbs cover float32 dyadics (278 bits) plus any uint32-length support path (32 bits). */
export const INTERACTION_EXACT_LIMB_COUNT = 10;
export const INTERACTION_EXACT_WGSL = `
alias ExactCovariate = array<u32, 10>;
fn negateExact(value: ExactCovariate) -> ExactCovariate {
  var result: ExactCovariate; var carry = 1u;
  for (var word=0u;word<10u;word++) {
    let next = ~value[word] + carry;
    carry = select(0u,1u,carry != 0u && next == 0u); result[word] = next;
  }
  return result;
}
fn addExact(first: ExactCovariate, second: ExactCovariate) -> ExactCovariate {
  var result: ExactCovariate; var carry=0u;
  for (var word=0u;word<10u;word++) {
    let partial = first[word] + second[word];
    let next = partial + carry;
    carry = select(0u,1u,partial < first[word] || next < partial);
    result[word] = next;
  }
  return result;
}
fn exactFromBits(bits: u32) -> ExactCovariate {
  var result: ExactCovariate;
  let magnitude = bits & 0x7fffffffu; let exponent=magnitude >> 23u;
  if (exponent == 255u) { return result; }
  let significand = (magnitude & 0x7fffffu) | select(0x800000u,0u,exponent == 0u);
  let shift = max(exponent,1u)-1u; let word=shift/32u; let remainder=shift%32u;
  result[word] = significand << remainder;
  if (remainder != 0u) { result[word+1u] = significand >> (32u-remainder); }
  if ((bits & 0x80000000u) != 0u) { return negateExact(result); }
  return result;
}
fn equalExact(first: ExactCovariate, second: ExactCovariate) -> bool {
  for (var word=0u;word<10u;word++) { if (first[word] != second[word]) { return false; } }
  return true;
}
fn canonicalBits(bits:u32) -> u32 { return select(bits,0u,(bits & 0x7fffffffu) == 0u); }
`;
