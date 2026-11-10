// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Rows one workgroup of the published `GPUScan` scans; each further level divides by it. */
const SCAN_ROWS_PER_BLOCK = 256;

/**
 * Command nodes the published `GPUScan` (10.0.0-alpha.10) adds for `rowCount` rows: one block-scan
 * node per level, and one offset node per level but the last. A test builds the real scan and
 * compares, so a changed published node count fails it. @internal
 */
export function getScanCommandNodeCount(rowCount: number): number {
  let levelCount = 1;
  for (let length = rowCount; Math.ceil(length / SCAN_ROWS_PER_BLOCK) > 1; levelCount++) {
    length = Math.ceil(length / SCAN_ROWS_PER_BLOCK);
  }
  return 2 * levelCount - 1;
}
