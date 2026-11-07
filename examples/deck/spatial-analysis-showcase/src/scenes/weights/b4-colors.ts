// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialAnalysisColor} from '../../engine/layers';
export {MORAN_COLORS} from '../../../../spatial-analysis-shared/layer-constants';
export {GI_COLORS} from '../../../../spatial-analysis-shared/layer-constants';

/** Color of places that are not significant. */
export const NEUTRAL: SpatialAnalysisColor = [150, 160, 175, 150];

/** Color of a place with no data. */
export const NO_DATA: SpatialAnalysisColor = [140, 148, 160, 60];

/** Focus-row classes of the weights scene: 0 other, 1 neighbour, 2 focus. */
export const FOCUS_COLORS: readonly SpatialAnalysisColor[] = [
  [175, 184, 196, 70],
  [245, 140, 30, 235],
  [190, 25, 70, 255]
];

/** Colour of an island (a place without neighbours). */
export const ISLAND_COLOR: SpatialAnalysisColor = [225, 20, 60, 255];

/** Colors of the global-statistics views. */
export const BLACK_JOIN_COLOR: SpatialAnalysisColor = [196, 70, 40, 235];
export const WHITE_JOIN_COLOR: SpatialAnalysisColor = [120, 150, 190, 150];

/** Link color per theme. */
export function getLinkColor(theme: 'light' | 'dark', dense = false): SpatialAnalysisColor {
  if (dense) return theme === 'dark' ? [235, 240, 255, 55] : [30, 45, 80, 42];
  return theme === 'dark' ? [235, 240, 255, 95] : [30, 45, 80, 80];
}

/** Outline color per theme. */
export function getOutlineColor(theme: 'light' | 'dark'): SpatialAnalysisColor {
  return theme === 'dark' ? [235, 240, 255, 60] : [20, 30, 50, 70];
}
