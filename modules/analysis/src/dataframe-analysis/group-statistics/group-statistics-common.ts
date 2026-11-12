// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Marker for "no group" in per-row and per-position group indices. @internal */
export const GROUP_NONE = 0xffffffff;

/** Read-only storage binding. @internal */
export function readBinding(
  name: string,
  view: GraphDataView,
  type: 'u32' | 'f32' = 'u32'
): WGSLKernelBinding {
  return {name, view, type, access: 'read'};
}

/** Read-write storage binding. @internal */
export function writeBinding(
  name: string,
  view: GraphDataView,
  type: 'u32' | 'f32' = 'u32'
): WGSLKernelBinding {
  return {name, view, type, access: 'read_write'};
}

/** Read-write `atomic<u32>` storage binding. @internal */
export function atomicBinding(name: string, view: GraphDataView): WGSLKernelBinding {
  return {name, view, type: 'atomic<u32>', access: 'read_write'};
}

/**
 * Value keys for order statistics and extremes. Works on the raw f32 bits so NaN handling never
 * depends on float comparisons: every non-finite value (NaN, +-Infinity) maps to `0xffffffff`
 * (sorts last), `-0` is canonicalised to `+0`, and finite values get the order-preserving key.
 *
 * @internal
 */
export const GROUP_VALUE_KEY_WGSL = /* wgsl */ `
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}

fn getValueKey(rawBits: u32) -> u32 {
  if (!isFiniteBits(rawBits)) {
    return 0xffffffffu;
  }
  var bits = rawBits;
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
`;
