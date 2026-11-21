// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const PROFILE_VALUES = new Float32Array([2, 5, 1, 4, 9, 7, 8, 3]);
export const PROFILE_POSITIONS = new Float32Array([0.5, 0.5, 3.5, 0.5, 3.5, 1.5, 0.5, 1.5]);
// Unit-spaced samples at cell centers: line 0 reads [2,5,1,4], line 1 reads [3,8,7,9].
// Add positive rises / magnitudes of drops separately; reset each line. Four samples per line.
export const PROFILE_EXPECTED = {
  samplePositions: [0.5, 0.5, 1.5, 0.5, 2.5, 0.5, 3.5, 0.5, 3.5, 1.5, 2.5, 1.5, 1.5, 1.5, 0.5, 1.5],
  sampleCumulativeGain: [0, 3, 3, 6, 0, 5, 5, 7],
  sampleCumulativeLoss: [0, 0, 4, 4, 0, 0, 1, 1],
  lineMinimums: [1, 3],
  lineMaximums: [5, 9]
};
