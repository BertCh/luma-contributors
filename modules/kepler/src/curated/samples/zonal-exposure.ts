// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogParameterValue} from '../../analysis/catalog-binding-types';
import type {CatalogDataInput} from '../../analysis/catalog-data-input';
import {makeCatalogTableInput} from '../../analysis/catalog-data-input';
import {makeCatalogRasterInput} from '../../analysis/catalog-raster-input';

export type ZonalExposureSample = {
  id: string;
  license: string;
  attribution: string;
  reference: string;
  inputs: Record<string, CatalogDataInput>;
  parameters: Record<string, CatalogParameterValue>;
};

/**
 * Synthetic licence-clean fixture from expansion-recipes.spec.ts, exposure case:
 * zone 0 total=60, observed=40, exposed=10; zone 1 has zero population; zone 2 is empty.
 * No third-party extract is bundled. Native nodata encodes the missing hazard.
 */
export function createZonalExposureSample(): ZonalExposureSample {
  return {
    id: 'zonal-exposure-counts',
    license: 'MIT',
    attribution: 'vis.gl contributors (synthetic fixture)',
    reference: 'modules/analysis/test/gpu-spatial-analysis/recipes/expansion-recipes.spec.ts',
    inputs: {
      zones: {
        type: 'FeatureCollection',
        features: [
          {
            id: 'count-zone-west-and-north',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [0, 0],
                  [10, 0],
                  [10, 10],
                  [20, 10],
                  [20, 20],
                  [0, 20],
                  [0, 0]
                ]
              ]
            }
          },
          {
            id: 'count-zone-south-east',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [10, 0],
                  [20, 0],
                  [20, 10],
                  [10, 10],
                  [10, 0]
                ]
              ]
            }
          },
          {
            id: 'count-zone-empty',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [30, 0],
                  [40, 0],
                  [40, 10],
                  [30, 10],
                  [30, 0]
                ]
              ]
            }
          }
        ]
      },
      surface: makeCatalogRasterInput(
        {
          id: 'synthetic-hazard',
          crs: 'planar',
          cellSizeMode: 'uniform',
          grid: {width: 2, height: 2, affine: [10, 0, 0, 0, -10, 20], pixelInterpretation: 'area'},
          band: {
            index: 1,
            name: 'hazard',
            values: new Float32Array([5, -999, 1, 8]),
            validity: new Uint8Array([1, 1, 1, 1]),
            noDataValue: -999
          }
        },
        4
      ),
      cells: makeCatalogTableInput(
        {
          id: 'cell-ownership',
          label: 'Whole-cell zones and population counts',
          rowCount: 4,
          columnTypes: {cell: 'int32', zone: 'int32', people: 'float64'},
          columns: {cell: [0, 1, 2, 3], zone: [0, 0, 0, 1], people: [10, 20, 30, 0]}
        },
        {maximumRows: 4, identityColumn: 'cell'}
      )
    },
    parameters: {
      cellIndexProperty: 'cell',
      zoneProperty: 'zone',
      populationProperty: 'people',
      weightMeaning: 'count',
      threshold: 4
    }
  };
}

/**
 * Independent second mapping: different shape, affine, names and row order. Values match the
 * backlog-recipe-composition.spec.ts density oracle: totals [12,12,0], observed [6,12,0],
 * exposed [6,10,0]. Invalid negative density and zero area are deliberately retained as inputs.
 */
export function createMappedZonalExposureSample(): ZonalExposureSample {
  return {
    id: 'zonal-exposure-densities',
    license: 'MIT',
    attribution: 'vis.gl contributors (synthetic fixture)',
    reference:
      'modules/analysis/test/gpu-spatial-analysis/recipes/backlog-recipe-composition.spec.ts',
    inputs: {
      zones: {
        type: 'FeatureCollection',
        features: [
          {
            id: 'density-zone-left',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [100, 200],
                  [104, 200],
                  [104, 203],
                  [100, 203],
                  [100, 200]
                ]
              ]
            }
          },
          {
            id: 'density-zone-right',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [104, 200],
                  [112, 200],
                  [112, 203],
                  [104, 203],
                  [104, 200]
                ]
              ]
            }
          },
          {
            id: 'density-zone-empty',
            geometry: {
              type: 'Polygon',
              coordinates: [
                [
                  [130, 200],
                  [132, 200],
                  [132, 203],
                  [130, 203],
                  [130, 200]
                ]
              ]
            }
          }
        ]
      },
      surface: makeCatalogRasterInput(
        {
          id: 'independent-hazard',
          crs: 'planar',
          cellSizeMode: 'uniform',
          grid: {width: 6, height: 1, affine: [2, 0, 100, 0, 3, 200], pixelInterpretation: 'area'},
          band: {
            index: 1,
            name: 'measured',
            values: new Float32Array([5, NaN, 1, 9, 8, 7]),
            validity: new Uint8Array([1, 1, 1, 1, 1, 1]),
            noDataValue: null
          }
        },
        6
      ),
      cells: makeCatalogTableInput(
        {
          id: 'mapped-cell-ownership',
          label: 'Renamed and shuffled cell rows',
          rowCount: 6,
          columnTypes: {
            rasterOffset: 'int32',
            district: 'int32',
            density: 'float64',
            squareKilometres: 'float64'
          },
          columns: {
            rasterOffset: [5, 2, 0, 4, 1, 3],
            district: [1, 1, 0, 1, 0, 1],
            density: [5, 4, 2, 10, 3, -1],
            squareKilometres: [2, 0.5, 3, 0, 2, 4]
          }
        },
        {maximumRows: 6, identityColumn: 'rasterOffset'}
      )
    },
    parameters: {
      cellIndexProperty: 'rasterOffset',
      zoneProperty: 'district',
      populationProperty: 'density',
      cellAreaProperty: 'squareKilometres',
      weightMeaning: 'density',
      threshold: 5,
      densityAreaUnit: 'km2',
      cellAreaUnit: 'km2'
    }
  };
}
