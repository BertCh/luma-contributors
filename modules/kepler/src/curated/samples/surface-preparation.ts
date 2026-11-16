// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeCatalogRasterInput, type CatalogRasterInput} from '../../analysis/catalog-raster-input';

/** Repository-authored synthetic values: MIT, no third-party dataset or decoder. */
export const SURFACE_PREPARATION_SAMPLE_LICENSE = {
  id: 'surface-preparation-gaps',
  license: 'MIT',
  attribution: 'vis.gl contributors; repository-authored synthetic raster',
  reference: 'modules/kepler/src/curated/samples/surface-preparation.ts'
} as const;

/** Same 4-cell original-donor fixture as expansion-recipes.spec.ts, located for map publication. */
export function makeSurfacePreparationSample(): CatalogRasterInput {
  return makeCatalogRasterInput(
    {
      id: SURFACE_PREPARATION_SAMPLE_LICENSE.id,
      grid: {
        width: 4,
        height: 1,
        affine: [0.01, 0, -74, 0, -0.01, 41],
        pixelInterpretation: 'area'
      },
      band: {
        index: 1,
        name: 'Measured surface',
        values: new Float32Array([2, NaN, NaN, NaN]),
        validity: new Uint8Array([1, 0, 0, 0]),
        noDataValue: null
      },
      crs: 'EPSG:4326',
      cellSizeMode: 'geographic'
    },
    4
  );
}

/** Independently named band, dimensions, origin and mask; fixture from backlog recipe composition. */
export function makeSurfacePreparationSecondSample(): CatalogRasterInput {
  return makeCatalogRasterInput(
    {
      id: 'independent-survey',
      grid: {width: 6, height: 1, affine: [0.02, 0, 12, 0, -0.01, 52], pixelInterpretation: 'area'},
      band: {
        index: 3,
        name: 'Survey temperature',
        values: new Float64Array([2, -9999, -9999, -9999, -9999, 10]),
        validity: new Uint8Array([1, 0, 0, 0, 0, 1]),
        noDataValue: -9999
      },
      crs: 'EPSG:4326',
      cellSizeMode: 'geographic'
    },
    6
  );
}
