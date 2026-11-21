// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Independent exact geometry: 64*16 + 128*2^-20; no production prefix code is used. */
export function createPublishedPrefixReviewFixture() {
  const positions = new Float32Array(193 * 2);
  const measures = new Float32Array(193);
  for (let row = 0; row <= 192; row++) {
    positions[2 * row] = Math.min(row, 64) * 16;
    positions[2 * row + 1] = Math.max(row - 64, 0) * 2 ** -20;
    measures[row] = Math.min(row, 64) * 16 + Math.max(row - 64, 0) * 2 ** -20;
  }
  return {positions, measures, total: 1024.0001220703125, endpoint: [1024, 0.0001220703125]};
}
