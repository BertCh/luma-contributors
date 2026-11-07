// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type LegendLayoutMode = 'normal' | 'compact' | 'panel';

/** Height needed for the legend toggle with border, spacing and a 44px touch target. */
export const LEGEND_MINIMUM_MAP_HEIGHT = 58;
/** Below this space, duplicate map cartouches and inset charts move out of the way. */
export const LEGEND_COMFORTABLE_MAP_HEIGHT = 164;

/**
 * Selects where the map legend can live. Compact mode assumes cartouches and inset charts have
 * been hidden; panel mode keeps the legend reachable from the story panel footer.
 */
export function getLegendLayoutMode(standardSpace: number, compactSpace: number): LegendLayoutMode {
  if (standardSpace >= LEGEND_COMFORTABLE_MAP_HEIGHT) return 'normal';
  if (compactSpace >= LEGEND_MINIMUM_MAP_HEIGHT) return 'compact';
  return 'panel';
}
