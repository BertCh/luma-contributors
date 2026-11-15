// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** S2 ancestors and Hilbert-ordered descendants on the published two-word integer helpers. @internal */
export const S2_TOPOLOGY_WGSL = /* wgsl */ `
fn cellTopologyS2GetParent(cell: vec2u, targetResolution: u32) -> vec2u {
  let shift = 2u * (30u - targetResolution);
  return dggs_u64_shift_left(dggs_u64_shift_right(cell, shift), shift) |
    dggs_u64_shift_left(vec2u(0u, 1u), shift);
}
fn cellTopologyS2GetChild(cell: vec2u, targetResolution: u32, depth: u32, slot: u32) -> vec2u {
  let parentMarker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - targetResolution + depth));
  let childMarker = dggs_u64_shift_right(parentMarker, 2u * depth);
  return dggs_u64_add(
    dggs_u64_subtract(cell, parentMarker),
    dggs_u64_add(childMarker, dggs_u64_shift_left(vec2u(0u, slot), 2u * (30u - targetResolution) + 1u))
  );
}
`;
