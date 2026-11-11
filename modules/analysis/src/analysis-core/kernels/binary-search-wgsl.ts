// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Element type of the sorted storage array searched by {@link getBinarySearchWGSL}. @internal */
export type BinarySearchKeyType = 'u32' | 'f32' | 'u64';

/**
 * Returns WGSL for lower- and upper-bound binary search over a sorted storage binding declared
 * through `createWGSLKernelNode` (or `getWGSLKernelBinding`), so `${bindingName}` and
 * `${bindingName}Offset` exist. Contributor contract rule 11: use this instead of a local
 * `lowerBound`.
 *
 * Generated functions, with `Suffix` = `functionSuffix` (default: the binding name with its first
 * letter in upper case):
 * - `lowerBoundSuffix(first: u32, last: u32, key: K) -> u32`: the first element index `i` in
 *   `[first, last)` with `element(i) >= key`, or `last` when there is none.
 * - `upperBoundSuffix(first: u32, last: u32, key: K) -> u32`: the first `i` in `[first, last)` with
 *   `element(i) > key`, or `last`.
 *
 * `[first, last)` are element indices; the elements must be sorted ascending in the order below.
 * A reversed range (`first > last`) is empty and both functions return `first` without reading an
 * element, so the result is always in `[first, max(first, last)]`.
 * The count of keys equal to `key` is `upper - lower`, and the last element at most `key` (the
 * `lastAtMost` pattern) is `upper - 1` when `upper > first`. Each call runs at most 33 iterations.
 *
 * Key types:
 * - `'u32'`: `K = u32`, element `i` is `binding[offset + i]`.
 * - `'f32'`: `K = f32`, compared through the order-preserving key of `getOrderedKey` (negative
 *   NaNs, `-inf`, ..., `-0`, `+0`, ..., `+inf`, positive NaNs). That is the order of a `GPUSort`
 *   on ordered keys and agrees with `<` for non-NaN values except that `-0 < +0`. The generated
 *   module carries a private copy of the key function named `getBinarySearchOrderedKeySuffix`, so it
 *   does not depend on `GET_ORDERED_KEY_WGSL`.
 * - `'u64'`: `K = vec2<u32>` as `(high, low)` words, the in-shader order of `uint64-wgsl.ts`, of
 *   `CELL_KEY_WGSL` and of published luma.gl (`dggs_u64_*`). Storage keeps the low word first, as
 *   Arrow `Int64` and the time words do: element `i` is `binding[offset + 2i]` (low) and
 *   `binding[offset + 2i + 1]` (high), compared as unsigned 64-bit integers.
 *
 * @param bindingName WGSL name of the sorted `u32` (`'u32'`, `'u64'`) or `f32` (`'f32'`) binding.
 * @param keyType Element type. Defaults to `'u32'`.
 * @param functionSuffix Suffix of the generated function names; needed when one module searches
 *   the same binding with two key types.
 * @internal
 */
export function getBinarySearchWGSL(
  bindingName: string,
  keyType: BinarySearchKeyType = 'u32',
  functionSuffix: string = bindingName.charAt(0).toUpperCase() + bindingName.slice(1)
): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(bindingName) || !/^[A-Za-z0-9_]+$/.test(functionSuffix)) {
    throw new Error(
      `getBinarySearchWGSL bindingName and functionSuffix must be WGSL identifiers, got ${bindingName}, ${functionSuffix}`
    );
  }
  const suffix = functionSuffix;
  const element = (index: string): string =>
    keyType === 'u64'
      ? `vec2<u32>(${bindingName}[${bindingName}Offset + 2u * ${index} + 1u], ${bindingName}[${bindingName}Offset + 2u * ${index}])`
      : `${bindingName}[${bindingName}Offset + ${index}]`;
  const keyWGSL = keyType === 'u64' ? 'vec2<u32>' : keyType;
  // `isBelow(a, b)` is a < b and `isAtMost(a, b)` is a <= b in the element order.
  let declarations = '';
  let isBelow: (left: string, right: string) => string;
  let isAtMost: (left: string, right: string) => string;
  if (keyType === 'f32') {
    declarations = `fn getBinarySearchOrderedKey${suffix}(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;
    isBelow = (left, right) =>
      `getBinarySearchOrderedKey${suffix}(${left}) < getBinarySearchOrderedKey${suffix}(${right})`;
    isAtMost = (left, right) =>
      `getBinarySearchOrderedKey${suffix}(${left}) <= getBinarySearchOrderedKey${suffix}(${right})`;
  } else if (keyType === 'u64') {
    isBelow = (left, right) =>
      `(${left}.x < ${right}.x || (${left}.x == ${right}.x && ${left}.y < ${right}.y))`;
    isAtMost = (left, right) =>
      `(${left}.x < ${right}.x || (${left}.x == ${right}.x && ${left}.y <= ${right}.y))`;
  } else {
    isBelow = (left, right) => `${left} < ${right}`;
    isAtMost = (left, right) => `${left} <= ${right}`;
  }
  const search = (name: string, isLeft: (middle: string) => string): string => `
fn ${name}${suffix}(first: u32, last: u32, key: ${keyWGSL}) -> u32 {
  var low = first;
  var high = max(first, last);
  while (low < high) {
    let middle = low + (high - low) / 2u;
    let value = ${element('middle')};
    if (${isLeft('value')}) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}
`;
  return /* wgsl */ `
${declarations}${search('lowerBound', value => isBelow(value, 'key'))}${search('upperBound', value => isAtMost(value, 'key'))}`;
}
