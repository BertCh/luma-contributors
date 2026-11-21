// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Auto crossover compares separately rounded products rather than reciprocal division.
 * Explicit fma(..., ..., 0) in WGSL gives the same single-rounded products under Metal.
 */
export function getDensityFrameRoute(
  bandwidth: number,
  cellSize: readonly [number, number],
  margin: number
): {valid: boolean; useBinned: boolean} {
  const encodedBandwidth = Math.fround(bandwidth);
  const horizontal = Math.abs(Math.fround(cellSize[0]));
  const vertical = Math.abs(Math.fround(cellSize[1]));
  const area = Math.fround(horizontal * vertical);
  const valid =
    [encodedBandwidth, horizontal, vertical, area].every(
      value => Number.isFinite(value) && value > 0
    ) &&
    Number.isFinite(Math.fround(encodedBandwidth / horizontal)) &&
    Number.isFinite(Math.fround(encodedBandwidth / vertical));
  const minimumCell = Math.min(horizontal, vertical);
  const support = Math.fround(5 * encodedBandwidth);
  return {
    valid,
    useBinned:
      valid &&
      support > Math.fround(32 * minimumCell) &&
      support <= Math.fround(Math.fround(2 * margin) * minimumCell)
  };
}

export const DENSITY_AUTO_ROUTE_WGSL = /* wgsl */ `
fn isDensityFrameValid(bandwidth: f32, cellSize: vec2f) -> bool {
  let area = abs(cellSize.x * cellSize.y);
  let sigma = vec2f(bandwidth) / abs(cellSize);
  return isFiniteValue(bandwidth) && bandwidth > 0.0 && all(abs(cellSize) > vec2f(0.0)) &&
    isFiniteValue(cellSize.x) && isFiniteValue(cellSize.y) && isFiniteValue(area) && area > 0.0 &&
    isFiniteValue(sigma.x) && isFiniteValue(sigma.y);
}
fn useDensityBinnedRoute(bandwidth: f32, cellSize: vec2f, margin: u32) -> bool {
  let minimumCell = min(abs(cellSize.x), abs(cellSize.y));
  let support = fma(5.0, bandwidth, 0.0);
  return isDensityFrameValid(bandwidth, cellSize) && support > fma(32.0, minimumCell, 0.0) &&
    support <= fma(2.0 * f32(margin), minimumCell, 0.0);
}
`;
