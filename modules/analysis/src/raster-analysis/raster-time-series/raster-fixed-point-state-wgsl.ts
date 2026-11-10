// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL `load<Name>(entry)` and `store<Name>(entry, words)` for a fixed-point state binding with
 * three `uint32` words per entry. Requires `FIXED_POINT_WGSL`.
 *
 * @internal
 */
export function getRasterFixedPointStateWGSL(bindingName: string, writable = true): string {
  const name = `${bindingName[0].toUpperCase()}${bindingName.slice(1)}`;
  return `
fn load${name}(entry: u32) -> vec3<u32> {
  return vec3<u32>(${bindingName}[${bindingName}Offset + 3u * entry], ${bindingName}[${bindingName}Offset + 3u * entry + 1u], ${bindingName}[${bindingName}Offset + 3u * entry + 2u]);
}
${
  writable
    ? `fn store${name}(entry: u32, words: vec3<u32>) {
  ${bindingName}[${bindingName}Offset + 3u * entry] = words.x;
  ${bindingName}[${bindingName}Offset + 3u * entry + 1u] = words.y;
  ${bindingName}[${bindingName}Offset + 3u * entry + 2u] = words.z;
}`
    : ''
}`;
}
