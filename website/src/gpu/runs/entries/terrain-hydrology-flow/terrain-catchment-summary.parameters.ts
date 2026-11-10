// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GpuSceneParameter} from '../../../types';

// histogramBins is the operation's compile-time prop; pourPoints sets how many basins GPUTerrainWatersheds
// delineates for it to summarize.
export const parameters: GpuSceneParameter[] = [
  {id: 'histogramBins', label: 'histogramBins', min: 4, max: 24, step: 2, default: 12},
  {
    id: 'pourPointCount',
    label: 'GPUTerrainWatersheds: pourPoints',
    min: 2,
    max: 8,
    step: 1,
    default: 6
  }
];
