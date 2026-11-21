// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Compare positive finite represented bandwidth against exactly three represented cell widths.
 * Integer significands avoid reciprocal division and rounded multiplication at the boundary.
 * Subnormals share exponent 1 with the hidden significand bit omitted; the largest shifted
 * significand is below 2^26, so all integer arithmetic fits uint32.
 */
export const BANDWIDTH_RESOLUTION_WGSL = /* wgsl */ `
fn isBandwidthBelowThreeCells(bandwidth: f32, cell: f32) -> bool {
  let bandwidthBits = bitcast<u32>(bandwidth);
  let cellBits = bitcast<u32>(cell);
  let bandwidthExponent = (bandwidthBits >> 23u) & 255u;
  let cellExponent = (cellBits >> 23u) & 255u;
  let bandwidthPower = max(1u, bandwidthExponent);
  let cellPower = max(1u, cellExponent);
  if (bandwidthPower < cellPower) { return true; }
  let shift = bandwidthPower - cellPower;
  if (shift > 2u) { return false; }
  let bandwidthMantissa = (bandwidthBits & 0x7fffffu) | select(0u, 0x800000u, bandwidthExponent > 0u);
  let cellMantissa = (cellBits & 0x7fffffu) | select(0u, 0x800000u, cellExponent > 0u);
  return (bandwidthMantissa << shift) < 3u * cellMantissa;
}
`;
