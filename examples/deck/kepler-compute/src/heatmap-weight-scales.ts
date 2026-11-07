// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Kepler 3.3 offers exactly these scales for the heatmap weight (CHANNEL_SCALES.size) on real and
 * integer fields: linearFieldScaleFunctions[size] = [linear, sqrt, log]. Quantile, quantize and
 * ordinal scales exist only for color channels; ordinal fields only offer 'point'. A resident
 * heatmap therefore has no further native weight scale to add.
 */
export const NATIVE_HEATMAP_WEIGHT_SCALES = ['linear', 'sqrt', 'log'] as const;
export type NativeHeatmapWeightScale = (typeof NATIVE_HEATMAP_WEIGHT_SCALES)[number];

export function isNativeHeatmapWeightScale(
  scale: string | number | symbol | undefined
): scale is NativeHeatmapWeightScale {
  return (NATIVE_HEATMAP_WEIGHT_SCALES as readonly unknown[]).includes(scale);
}
