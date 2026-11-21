// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Encoding budget for the submitted-prefix regression. */
export function getSummaryReductionPrefixCount(scheduledOrder: readonly string[]): number {
  const mergeIndex = scheduledOrder.indexOf('study-window-summary-reduce-4');
  if (mergeIndex < 0) throw new Error('summary-reduce-4 is missing from the compiled schedule');
  return mergeIndex + 1;
}
