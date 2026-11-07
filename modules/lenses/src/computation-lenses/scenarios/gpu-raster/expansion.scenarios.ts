// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenarios} from '../../define-lens-scenarios';
import type {LensScenarioDescriptor} from '../../types';

/** Lazily runs one bounded raster expansion example. */
function createExpansionLoader(id: string, element: string): LensScenarioDescriptor['load'] {
  return async () => ({
    run: device =>
      import('./expansion-runner').then(module =>
        module.runExpansion(device, element, id, 'gpu-raster')
      )
  });
}

registerLensScenarios(
  {
    entry: 'gpu-raster',
    rungs: ['result', 'graph', 'table', 'kernel'],
    dataset: {
      id: 'bounded-raster-expansion',
      licence: 'MIT',
      source: 'Synthetic explicit 3 by 3 reference grid'
    }
  },
  [
    {
      id: 'gpu-raster.raster-fill-holes',
      element: 'GPURasterFillHoles',
      title: 'GPURasterFillHoles',
      load: createExpansionLoader('gpu-raster.raster-fill-holes', 'GPURasterFillHoles')
    },
    {
      id: 'gpu-raster.raster-fill-no-data',
      element: 'GPURasterFillNoData',
      title: 'GPURasterFillNoData',
      load: createExpansionLoader('gpu-raster.raster-fill-no-data', 'GPURasterFillNoData')
    },
    {
      id: 'gpu-raster.raster-fractional-coverage',
      element: 'GPURasterFractionalCoverage',
      title: 'GPURasterFractionalCoverage',
      load: createExpansionLoader(
        'gpu-raster.raster-fractional-coverage',
        'GPURasterFractionalCoverage'
      )
    },
    {
      id: 'gpu-raster.raster-fractional-zonal-statistics',
      element: 'GPURasterFractionalZonalStatistics',
      title: 'GPURasterFractionalZonalStatistics',
      load: createExpansionLoader(
        'gpu-raster.raster-fractional-zonal-statistics',
        'GPURasterFractionalZonalStatistics'
      )
    },
    {
      id: 'gpu-raster.raster-medial-axis',
      element: 'GPURasterMedialAxis',
      title: 'GPURasterMedialAxis',
      load: createExpansionLoader('gpu-raster.raster-medial-axis', 'GPURasterMedialAxis')
    },
    {
      id: 'gpu-raster.raster-rank-filter',
      element: 'GPURasterRankFilter',
      title: 'GPURasterRankFilter',
      load: createExpansionLoader('gpu-raster.raster-rank-filter', 'GPURasterRankFilter')
    },
    {
      id: 'gpu-raster.raster-reconstruction',
      element: 'GPURasterReconstruction',
      title: 'GPURasterReconstruction',
      load: createExpansionLoader('gpu-raster.raster-reconstruction', 'GPURasterReconstruction')
    },
    {
      id: 'gpu-raster.raster-reprojection',
      element: 'GPURasterReprojection',
      title: 'GPURasterReprojection',
      load: createExpansionLoader('gpu-raster.raster-reprojection', 'GPURasterReprojection')
    },
    {
      id: 'gpu-raster.raster-skeleton',
      element: 'GPURasterSkeleton',
      title: 'GPURasterSkeleton',
      load: createExpansionLoader('gpu-raster.raster-skeleton', 'GPURasterSkeleton')
    },
    {
      id: 'gpu-raster.raster-tile-reprojection',
      element: 'GPURasterTileReprojection',
      title: 'GPURasterTileReprojection',
      load: createExpansionLoader(
        'gpu-raster.raster-tile-reprojection',
        'GPURasterTileReprojection'
      )
    },
    {
      id: 'gpu-raster.raster-walking-cost',
      element: 'GPURasterWalkingCost',
      title: 'GPURasterWalkingCost',
      load: createExpansionLoader('gpu-raster.raster-walking-cost', 'GPURasterWalkingCost')
    },
    {
      id: 'gpu-raster.raster-watershed-segmentation',
      element: 'GPURasterWatershedSegmentation',
      title: 'GPURasterWatershedSegmentation',
      load: createExpansionLoader(
        'gpu-raster.raster-watershed-segmentation',
        'GPURasterWatershedSegmentation'
      )
    }
  ]
);
