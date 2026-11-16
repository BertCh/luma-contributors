// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {makeCatalogRasterInput} from '../../analysis/catalog-raster-input';

/** Repository-authored grids without third-party observations. */
export const LANDSCAPE_METRICS_SAMPLE_LICENSE = {
  id: 'landscape-metrics',
  license: 'MIT',
  attribution: 'vis.gl contributors',
  reference: 'modules/kepler/src/curated/samples/landscape-metrics.ts'
} as const;

/** The first grid matches the analytical 4x2 recipe fixture with recoded classes. The independent
 * second grid has a different shape, anisotropic cells, a missing column and negative class codes. */
export function createLandscapeMetricsSample(variant: 'sample' | 'second' = 'sample') {
  const second = variant === 'second';
  const width = second ? 7 : 4;
  const height = second ? 3 : 2;
  const values = second
    ? new Int32Array([
        -7, -7, -999, -7, -7, 90, 90, -7, -7, -999, -7, -7, 90, 90, -7, -7, -999, -7, -7, 90, 90
      ])
    : new Int32Array([10, 10, 90, 90, 10, 10, 90, 90]);
  return {
    landcover: makeCatalogRasterInput(
      {
        id: second ? 'independent-cover-grid' : 'landscape-recipe-grid',
        grid: makeGPURasterMetadata({
          width,
          height,
          cellSize: second ? [2, 3] : [1, 1],
          origin: second ? [100, 200] : [0, 2],
          rowDirection: second ? 'north' : 'south'
        }),
        band: {
          index: second ? 3 : 1,
          name: second ? 'habitat-code' : 'land-cover',
          values,
          validity: Uint8Array.from(values, value => Number(value !== -999)),
          noDataValue: -999
        },
        crs: 'planar',
        cellSizeMode: 'uniform'
      },
      4096
    )
  };
}
