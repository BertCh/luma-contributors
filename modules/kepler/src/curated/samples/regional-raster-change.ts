// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {makeCatalogRasterInput} from '../../analysis/catalog-raster-input';
import type {CatalogFeatureCollection} from '../../analysis/catalog-geojson';
import type {KeplerCuratedAnalysis} from '../curated-analysis-types';

/** Original synthetic numbers/geometries; no third-party observations or source extracts. */
export const REGIONAL_RASTER_CHANGE_SAMPLE_LICENSE = {
  license: 'MIT',
  attribution: 'vis.gl contributors; repository-authored synthetic raster comparisons',
  reference: 'modules/kepler/src/curated/samples/regional-raster-change.ts'
} as const;

/** Integration payload for stream X: decoded single-band slots need no column-name parameter. */
export const REGIONAL_RASTER_CHANGE_ANALYSIS_UPDATE: Pick<
  KeplerCuratedAnalysis,
  'method' | 'slots' | 'execution' | 'exposedParameters' | 'sampleId'
> = {
  method:
    'Subtract two aligned calibrated planar raster bands on common observations. Assign whole cells to polygon zones by centre, then report arithmetic mean change, thresholded changed area and common-observation coverage of assigned grid cells. Missing observations remain unknown; differences are descriptive, not significance tests.',
  slots: [
    {name: 'before', label: 'Earlier decoded band', kind: 'surface', fields: []},
    {name: 'after', label: 'Later decoded band', kind: 'surface', fields: []},
    {name: 'zones', label: 'Zones in raster coordinates', kind: 'polygons', fields: []}
  ],
  execution: {
    status: 'available',
    bindingId: 'curated-regional-raster-change',
    inputs: {before: 'before', after: 'after', zones: 'zones'},
    fieldParameters: {},
    parameters: {}
  },
  exposedParameters: ['threshold', 'confirmComparable'],
  sampleId: 'regional-raster-change'
};

/**
 * First sample uses the exact paired observations and zone ownership of expansion-recipes.spec.ts,
 * with uniform cell area 2 (the recipe fixture's zone-specific areas are normalized to this grid).
 * The independent second dataset has different dimensions, north-up rows, band names, missingness,
 * identities and a zone with no grid support. Large origins exercise Float32 local translation.
 */
export function createRegionalRasterChangeSample(secondDataset = false) {
  const width = secondDataset ? 3 : 2;
  const height = 2;
  const stepX = secondDataset ? 2 : 1;
  const stepY = secondDataset ? -3 : 2;
  const originX = secondDataset ? 100000000 : 0;
  const originY = secondDataset ? 200000000 : 0;
  const grid = makeGPURasterMetadata({
    width,
    height,
    origin: [originX, originY],
    cellSize: [stepX, Math.abs(stepY)],
    rowDirection: stepY < 0 ? 'south' : 'north'
  });
  const createBand = (
    id: string,
    values: number[],
    validity: number[],
    index: number,
    name: string
  ) =>
    makeCatalogRasterInput(
      {
        id,
        grid,
        crs: 'planar',
        cellSizeMode: 'uniform',
        band: {
          index,
          name,
          values: Float32Array.from(values),
          validity: Uint8Array.from(validity),
          noDataValue: -9999
        }
      },
      65536
    );
  const createRectangle = (left: number, bottom: number, right: number, top: number) => ({
    type: 'Polygon',
    coordinates: [
      [
        [left + originX, bottom + originY],
        [right + originX, bottom + originY],
        [right + originX, top + originY],
        [left + originX, top + originY],
        [left + originX, bottom + originY]
      ]
    ]
  });
  const zones: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: secondDataset
      ? [
          {id: 'western-zone', geometry: createRectangle(0, -6, 4, 0)},
          {id: 'eastern-zone', geometry: createRectangle(4, -6, 6, 0)},
          {id: 'outside-zone', geometry: createRectangle(10, -6, 12, 0)}
        ]
      : [
          {id: 'first-zone', geometry: createRectangle(0, 0, 2, 2)},
          {id: 'second-zone', geometry: createRectangle(0, 2, 2, 4)}
        ]
  };
  return {
    before: secondDataset
      ? createBand('survey-old', [10, 20, 30, 40, 50, 60], [1, 1, 1, 1, 0, 1], 3, 'old_temperature')
      : createBand('baseline', [1, NaN, 5, 2], [1, 1, 1, 1], 1, 'before'),
    after: secondDataset
      ? createBand(
          'survey-new',
          [12, 18, -9999, 41, 55, 60],
          [1, 1, 0, 1, 1, 1],
          7,
          'new_temperature'
        )
      : createBand('followup', [3, 4, 4, 2], [1, 1, 1, 1], 1, 'after'),
    zones
  };
}
