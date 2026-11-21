// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Feature-to-ring range for polygon layouts, with logical view offsets and clamped sentinels.
 * Requires featureOffsets, polygonOffsets, POLYGON_COUNT and RING_COUNT. @internal
 */
export const POLYGON_FEATURE_RING_RANGES_WGSL = /* wgsl */ `fn getRingStart(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}

fn getRingEnd(feature: u32) -> u32 {
  let polygon = min(featureOffsets[featureOffsetsOffset + feature + 1u], POLYGON_COUNT);
  return min(polygonOffsets[polygonOffsetsOffset + polygon], RING_COUNT);
}`;

/** Positional shell lookup, including empty polygons and the legacy out-of-range fallback.
 * Requires polygonOffsets and POLYGON_COUNT. @internal
 */
export const POLYGON_SHELL_RING_WGSL = /* wgsl */ `fn isShellRing(ring: u32, isFirstRing: bool) -> bool {
  if (ring < polygonOffsets[polygonOffsetsOffset] || ring >= polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
    return isFirstRing;
  }
  var low = 0u;
  var high = POLYGON_COUNT;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] < ring) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return polygonOffsets[polygonOffsetsOffset + low] == ring;
}`;
