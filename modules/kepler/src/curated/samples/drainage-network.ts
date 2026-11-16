// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeCatalogRasterInput} from '../../analysis/catalog-raster-input';
import type {KeplerCuratedAnalysis} from '../curated-analysis-types';

/** Integration-ready record; the shared curated registry is owned by stream X. */
export const CURATED_DRAINAGE_NETWORK_ANALYSIS: KeplerCuratedAnalysis = {
  id: 'drainage-network',
  question: 'shape-of-the-surface',
  title: 'Where does water go?',
  name: 'Drainage network and catchments',
  wave: 1,
  method:
    'Each cell of a complete planar elevation grid drains to its steepest downhill D8 neighbour. Depressions are filled and flats are routed by default. The recipe accumulates contributing cells, assigns stream order and traces connected stream reaches. Catchment cells and summaries are labelled by the terminal outlet or pit cell index; they are not separate catchments for each reach. Distances use the declared planar linear units.',
  slots: [
    {
      name: 'elevation',
      label: 'Elevation raster (selected numeric band)',
      kind: 'surface',
      fields: []
    }
  ],
  execution: {
    status: 'available',
    bindingId: 'curated-drainage-network',
    inputs: {elevation: 'elevation'},
    fieldParameters: {},
    parameters: {}
  },
  exposedParameters: ['streamThreshold', 'fillDepressions', 'resolveFlats'],
  sampleId: 'drainage-network-slope'
};

/** Repository-authored fixture; the primary band is the existing expansion-recipes.spec.ts DEM. */
export const DRAINAGE_NETWORK_SAMPLE = {
  id: 'drainage-network-slope',
  license: 'MIT',
  attribution: 'vis.gl contributors',
  reference: 'modules/kepler/src/curated/samples/drainage-network.ts'
} as const;

/** Hand-computed receivers, subtree sizes and path lengths, independent of the GPU recipe. */
export const DRAINAGE_NETWORK_EXPECTATIONS = {
  flowDirections: [4, 4, 4, 4, 4, 4, 1, 1, 0],
  accumulation: [1, 1, 1, 2, 2, 2, 3, 6, 9],
  catchments: [8, 8, 8, 8, 8, 8, 8, 8, 8],
  distance: [4, 3, 2, 3, 2, 1, 2, 1, 0],
  streamOrder: [1, 1, 1, 1, 1, 1, 1, 2, 2],
  reachCount: 4
} as const;

/** Diagonal lengths are sqrt(2^2 + 3^2); ties prefer SE over SW, and E over W. */
export const DRAINAGE_NETWORK_SECOND_EXPECTATIONS = {
  flowDirections: [2, 4, 2, 4, 1, 0, 1, 0],
  accumulation: [1, 1, 1, 1, 1, 4, 1, 4],
  catchments: [5, 5, 7, 7, 5, 5, 7, 7],
  distance: [Math.sqrt(13), 3, Math.sqrt(13), 3, 2, 0, 2, 0],
  streamOrder: [1, 1, 1, 1, 1, 2, 1, 2],
  reachCount: 6
} as const;

/** A complete 3 by 3 planar slope, matching the analysis recipe fixture exactly. */
export function createDrainageNetworkSample() {
  return {
    elevation: makeCatalogRasterInput(
      {
        id: DRAINAGE_NETWORK_SAMPLE.id,
        crs: 'planar',
        cellSizeMode: 'uniform',
        grid: {width: 3, height: 3, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        band: {
          index: 1,
          name: 'elevation',
          values: new Float32Array([9, 8, 7, 6, 5, 4, 3, 2, 1]),
          validity: new Uint8Array(9).fill(1),
          noDataValue: null
        }
      },
      9
    )
  };
}

/** Independently mapped, differently named rectangular grid with two outlets and anisotropic cells. */
export function createDrainageNetworkSecondDataset() {
  return {
    survey: makeCatalogRasterInput(
      {
        id: 'survey-two-outlets',
        crs: 'planar',
        cellSizeMode: 'uniform',
        grid: {
          width: 4,
          height: 2,
          affine: [2, 0, 1000000000, 0, -3, 4500000000],
          pixelInterpretation: 'area'
        },
        band: {
          index: 2,
          name: 'height_survey',
          values: new Float64Array([8, 7, 8, 7, 4, 1, 4, 1]),
          validity: new Uint8Array(8).fill(1),
          noDataValue: null
        }
      },
      8
    )
  };
}
