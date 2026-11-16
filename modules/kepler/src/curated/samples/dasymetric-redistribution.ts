// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogDataInput} from '../../analysis/catalog-data-input';
import type {CatalogFeature, CatalogFeatureCollection} from '../../analysis/catalog-geojson';
import type {CatalogParameterValue} from '../../analysis/catalog-binding-types';
import {makeCatalogRasterInput} from '../../analysis/catalog-raster-input';
import {CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS} from '../bindings/curated-dasymetric-redistribution-binding';
import {createCuratedWorkflowAnalysisTemplate} from '../../workflows/workflow-analysis-template';

export {CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS} from '../bindings/curated-dasymetric-redistribution-binding';

export const DASYMETRIC_SAMPLE_LICENSE = {
  id: 'dasymetric-fractional-support',
  license: 'MIT',
  attribution: 'vis.gl contributors',
  reference: 'modules/kepler/src/curated/samples/dasymetric-redistribution.ts'
} as const;

/** The helper uses the same registered metadata as shared template discovery. */
export function createDasymetricRedistributionTemplate() {
  return createCuratedWorkflowAnalysisTemplate(CURATED_DASYMETRIC_REDISTRIBUTION_ANALYSIS);
}

function rectangle(
  id: string,
  minimumX: number,
  maximumX: number,
  properties: Record<string, unknown> = {},
  minimumY = 0,
  maximumY = 1
): CatalogFeature {
  return {
    id,
    properties,
    geometry: {
      type: 'Polygon',
      coordinates: [
        [
          [minimumX, minimumY],
          [maximumX, minimumY],
          [maximumX, maximumY],
          [minimumX, maximumY],
          [minimumX, minimumY]
        ]
      ]
    }
  };
}

/**
 * Repository-authored rectangles reproduce the committed recipe oracle's fractional geometry.
 * The second dataset independently changes IDs, columns, affine units, density and totals.
 */
export function createDasymetricRedistributionSample(secondDataset = false): {
  inputs: Record<string, CatalogDataInput>;
  parameters: Record<string, CatalogParameterValue>;
  expected: {
    zone: {targets: number[]; allocated: number[]; residual: number[]; support: number[]};
    overlap: {targets: number[]; allocated: number[]; residual: number[]};
    sourceTotals: number[];
  };
} {
  const totalColumn = secondDataset ? 'inhabitants_2026' : 'residents';
  const coordinateScale = secondDataset ? 2 : 1;
  const originX = secondDataset ? 10 : 0;
  const originY = secondDataset ? 20 : 0;
  const sourceTotals = [120, 80, 50].map(value => value * (secondDataset ? 0.5 : 1));
  const sources: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      [0.5, 2.5],
      [3.5, 5],
      [5, 6]
    ].map(([minimum, maximum], index) =>
      rectangle(
        `${secondDataset ? 'census' : 'source'}-${[9, 2, 7][index]}`,
        originX + minimum * coordinateScale,
        originX + maximum * coordinateScale,
        {[totalColumn]: sourceTotals[index]},
        originY,
        originY + coordinateScale
      )
    )
  };
  const targets: CatalogFeatureCollection = {
    type: 'FeatureCollection',
    features: [
      [0, 2],
      [2, 4],
      [5, 6],
      [8, 9]
    ].map(([minimum, maximum], index) =>
      rectangle(
        `${secondDataset ? 'district' : 'target'}-${[42, 8, 31, 0][index]}`,
        originX + minimum * coordinateScale,
        originX + maximum * coordinateScale,
        {},
        originY,
        originY + coordinateScale
      )
    )
  };
  return {
    inputs: {
      sources,
      targets,
      ancillary: makeCatalogRasterInput(
        {
          id: secondDataset ? 'habitable_density' : 'support-density',
          grid: {
            width: 6,
            height: 1,
            affine: [coordinateScale, 0, originX, 0, coordinateScale, originY],
            pixelInterpretation: 'area'
          },
          band: {
            index: secondDataset ? 2 : 1,
            name: 'Ancillary density',
            values: new Float32Array(secondDataset ? [1, 4, 2, 2, 3, 1] : [2, 0, 1, 3, 1, 0]),
            noDataValue: null,
            validity: new Uint8Array(6).fill(1)
          },
          crs: 'planar',
          cellSizeMode: 'uniform'
        },
        6
      )
    },
    parameters: {
      totalProperty: totalColumn,
      targetCellOwners: [0, 0, 1, 1, -1, 2],
      coordinatesMatchRaster: true,
      targetOwnershipConfirmed: true,
      denominator: 'zone'
    },
    expected: {
      zone: {
        targets: secondDataset ? [540 / 11, 120 / 11 + 10, 25, 0] : [80, 88, 0, 0],
        allocated: secondDataset ? [60, 10, 25] : [120, 48, 0],
        residual: secondDataset ? [0, 30, 0] : [0, 32, 50],
        support: secondDataset ? [22, 16, 4] : [1.5, 2.5, 0]
      },
      overlap: {
        targets: secondDataset ? [540 / 11, 120 / 11 + 40, 25, 0] : [80, 120, 0, 0],
        allocated: secondDataset ? [60, 40, 25] : [120, 80, 0],
        residual: secondDataset ? [0, 0, 0] : [0, 0, 50]
      },
      sourceTotals
    }
  };
}
