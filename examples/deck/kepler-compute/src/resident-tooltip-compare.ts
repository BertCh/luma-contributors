// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {formatTooltipValue, type ResidentTooltipValue} from './resident-tooltip-fields';

export type ResidentTooltipCompareType = 'absolute' | 'relative';
export type ResidentTooltipComparison = {
  name: string;
  current: ResidentTooltipValue;
  pinned: ResidentTooltipValue;
  /** Null when either side is non-numeric or missing, or relative with a zero pinned value.
   * Native formulas (getTooltipDisplayDeltaValue): absolute value - pinned, relative value / pinned - 1. */
  difference: number | null;
};

/** Native compare mode: hovered value versus the click-pinned row value, per shared field. */
export function compareTooltipValues(
  pinned: Record<string, ResidentTooltipValue>,
  current: Record<string, ResidentTooltipValue>,
  compareType: ResidentTooltipCompareType = 'absolute'
): ResidentTooltipComparison[] {
  return Object.keys(current).map(name => {
    const currentValue = current[name];
    const pinnedValue = name in pinned ? pinned[name] : null;
    let difference: number | null = null;
    if (typeof currentValue === 'number' && typeof pinnedValue === 'number') {
      if (compareType === 'relative')
        difference = pinnedValue === 0 ? null : currentValue / pinnedValue - 1;
      else difference = currentValue - pinnedValue;
      if (difference !== null && !Number.isFinite(difference)) difference = null;
    }
    return {name, current: currentValue, pinned: pinnedValue, difference};
  });
}

/** Native minus sign shown for a numeric field whose delta cannot be computed. */
export const TOOLTIP_MINUS_SIGN = '\u2212';

/**
 * Native delta text: relative deltas use the full two-digit percent format, absolute deltas the
 * field display format or three trimmed decimals, and a non-negative delta gets a leading plus.
 */
export function formatTooltipDifference(
  difference: number | null,
  compareType: ResidentTooltipCompareType,
  fieldFormat?: string | null
): string {
  if (difference === null) return TOOLTIP_MINUS_SIGN;
  const text = formatTooltipValue(
    difference,
    compareType === 'relative' ? '.2%' : fieldFormat || '.3~f'
  );
  return /^[+\-\u2212\u2013]/.test(text) ? text : `+${text}`;
}
