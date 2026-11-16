// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Preserve ceil-sized Philox stream partitions without adding to, or multiplying past, the total.
 * Every intermediate and endpoint fits uint32, including empty blocks after the final draw.
 */
export const SCAN_DRAW_INTERVAL_WGSL = `
struct ScanDrawInterval { first: u32, stop: u32 }
fn getScanDrawInterval(total: u32, blocks: u32, blockIndex: u32) -> ScanDrawInterval {
  let chunk = total / blocks + u32(total % blocks != 0u);
  var first = total;
  if (chunk > 0u && blockIndex <= total / chunk) { first = blockIndex * chunk; }
  return ScanDrawInterval(first, first + min(chunk, total - first));
}
`;
