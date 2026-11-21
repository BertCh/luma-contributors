// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent tail decision for the recipe oracle, before displayed p-value rounding. */
export function isGeodaTailSignificant(
  count: number,
  permutationCount: number,
  significanceLevel: number
): boolean {
  // The packed cutoff has 24 significant bits; multiplying by the bounded draw count is exact in binary64.
  return (
    count !== 0xffffffff && count + 1 <= Math.fround(significanceLevel) * (permutationCount + 1)
  );
}
