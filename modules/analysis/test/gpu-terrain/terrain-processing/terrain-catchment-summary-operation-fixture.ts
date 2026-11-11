// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TerrainCatchmentSummaryOperationProps} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation';

export function makeCatchmentInput() {
  return {
    elevation: Float32Array.of(10, 8, 6, 4, 20, 16, 12, 8),
    basinIds: Uint32Array.of(0, 0, 1, 1, 0, 0, 1, 1),
    streams: Uint32Array.of(1, 2, 3, 1, 0, 1, 1, 0),
    flowDirections: Uint32Array.of(1, 1, 1, 0, 0, 1, 1, 0),
    validity: new Uint32Array(8).fill(1)
  };
}

export function makeCatchmentProps(): TerrainCatchmentSummaryOperationProps {
  return {
    id: 'catchment-test',
    grid: {width: 4, height: 2, affine: [2, 0, 0, 0, -3, 0], pixelInterpretation: 'area'},
    basinCount: 3,
    histogramBins: 4,
    elevation: 'elevation',
    basinIds: 'basinIds',
    streams: 'streams',
    flowDirections: 'flowDirections',
    validity: 'validity',
    output: {
      statistics: 'statistics',
      hypsometry: 'hypsometry',
      profiles: 'profiles',
      profileStatus: 'profileStatus'
    }
  };
}
