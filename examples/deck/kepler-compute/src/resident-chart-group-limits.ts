// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';

/** Native MAX_CHART_POINTS: charts never plot more than this many groups. */
export const MAXIMUM_RESIDENT_GROUPS = 1000;

export type ResidentGroupBudget = {
  /** Groups the GPU summary uses (2..1,000). */
  groups: number;
  /** The caller's requested group count when it was reduced. */
  requested?: number;
  /** Stated reason, shown with the chart, when the budget was reduced. */
  note?: string;
};

/**
 * Numeric charts bin into at most 1,000 equal-width intervals, the native chart point cap. A larger
 * request (native numGroups) falls back to exactly 1,000 equal-width bins with a visible note.
 */
export function clampNumericGroupBudget(requested: number): ResidentGroupBudget {
  const groups = clamp(requested, 2, MAXIMUM_RESIDENT_GROUPS);
  return requested > MAXIMUM_RESIDENT_GROUPS
    ? {
        groups,
        requested,
        note: `Numeric groups: ${requested.toLocaleString()} requested, using ${MAXIMUM_RESIDENT_GROUPS.toLocaleString()} equal-width bins (native chart point cap).`
      }
    : {groups};
}

/**
 * Integer "auto" charts show unique values up to the group limit, otherwise equal-width intervals.
 * The limit is held to 1,000 unique values; a larger limit selects intervals beyond that.
 */
export function clampIntegerGroupLimit(requested: number): ResidentGroupBudget {
  const groups = clamp(requested, 0, MAXIMUM_RESIDENT_GROUPS);
  return requested > MAXIMUM_RESIDENT_GROUPS
    ? {
        groups,
        requested,
        note: `Integer groups: unique values are listed up to ${MAXIMUM_RESIDENT_GROUPS.toLocaleString()} (limit ${requested.toLocaleString()} requested); more distinct values use equal-width intervals.`
      }
    : {groups};
}

/** Why a dictionary larger than the group cap is refused (instead of silently truncating). */
export function describeCategoryOverflow(categoryCount: number): string {
  return `This category field has ${categoryCount.toLocaleString()} dictionary entries, above the ${MAXIMUM_RESIDENT_GROUPS.toLocaleString()}-group GPU summary cap. Native Kepler would rank groups by value and keep the top ${'`numGroups`'} (optionally with Others); the GPU top-N plus Other fallback counts dictionaries of at most 262,144 entries. Choose a field with at most ${MAXIMUM_RESIDENT_GROUPS.toLocaleString()} values or a numeric/time field, which falls back to equal-width bins.`;
}

/** Stated fallback for a dictionary above the cap: top groups by the chart's measure plus Other. */
export function describeCategoryTopGroups(categoryCount: number, limit: number): string {
  return `This category field has ${categoryCount.toLocaleString()} values, above the ${MAXIMUM_RESIDENT_GROUPS.toLocaleString()}-group cap: showing the top ${limit.toLocaleString()} groups (ranked by the chart measure when the chart was prepared) plus one Other bucket aggregated on the GPU.`;
}
