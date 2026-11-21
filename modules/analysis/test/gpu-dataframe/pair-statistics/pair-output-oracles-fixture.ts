// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';

export function createDiggleOutputPoison(name: string, length: number): Float32Array | Uint32Array {
  return name === 'status' ? new Uint32Array(length).fill(123) : new Float32Array(length).fill(123);
}

export const DIGGLE_INPUT = {
  positions: new Float32Array([1, 1, 2, 1, 5, 1, 6, 1]),
  times: new Float32Array([0, 1, 3, 7]),
  parameters: getGPUSpaceTimeKFunctionParameterValues({
    bounds: [0, 0, 8, 3],
    timeWindow: [0, 8],
    maximumDistance: 4,
    maximumTimeLag: 8
  }),
  randomParameters: new Uint32Array([21, 256])
};
export const DIGGLE_OPTIONS = {
  distanceClassCount: 2,
  timeClassCount: 2,
  permutationCount: 5,
  edgeCorrection: 'none' as const
};
// SciPy/NumPy + independent integer/permutation-matrix oracle: diggle-output-oracles.oracle.py.
export const DIGGLE_EXPECTED = {
  d: [21.333333333333336, 0, 32, 0],
  k: [64, 64, 96, 96],
  spatialK: [8, 12],
  temporalK: [5.333333333333333, 8],
  statistics: [
    53.333333333333336, -10.666666666666664, 53.333333333333336, -10.666666666666664,
    -10.666666666666664, -10.666666666666664
  ],
  permutationMean: [-4.266666666666664, 0, 6.4, 0],
  permutationStandardDeviation: [14.310835055998657, 0, 14.310835055998657, 0],
  summary: [
    53.333333333333336, 2.1333333333333355, 819.2000000000003, 1, 5, 0.3333333333333333, 1,
    0.6666666666666666, 1.7888543819998317, 4, 4
  ],
  status: [0]
};
