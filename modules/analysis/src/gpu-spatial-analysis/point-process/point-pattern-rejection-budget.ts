// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** @internal Largest attainable rejected fraction at or below the supplied host level. */
export function getPointPatternRejectionBudget(curveCount: number, level: number): number {
  let budget = Math.floor(level * curveCount);
  // Multiplication can round an attainable boundary down. Comparing the fraction itself
  // also distinguishes the immediately preceding binary64 level without a tolerance.
  while (budget < curveCount && (budget + 1) / curveCount <= level) budget++;
  while (budget > 0 && budget / curveCount > level) budget--;
  return budget;
}
