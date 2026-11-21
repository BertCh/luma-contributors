// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Decode a measured boolean; a suppressed or invalid summary flag is unavailable. @internal */
export function readRegressionSummaryFlag(value: number): boolean | null {
  return value === 0 ? false : value === 1 ? true : null;
}
