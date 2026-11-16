// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {CURATED_ZONAL_EXPOSURE_BINDING} from '../../src/curated/bindings/curated-zonal-exposure-binding';
import {
  createMappedZonalExposureSample,
  createZonalExposureSample,
  type ZonalExposureSample
} from '../../src/curated/samples/zonal-exposure';
import {makeArrowTableFromCatalogTable} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-table';
import type {CatalogRunResult} from '../../src/analysis/catalog-binding-types';

function checkColumn(
  result: CatalogRunResult,
  tableId: string,
  column: string,
  expected: readonly (number | string | null)[]
) {
  const table = result.tables.find(candidate => candidate.id === tableId);
  if (!table) throw new Error(`Zonal exposure missing ${tableId}: ${JSON.stringify(result)}`);
  const actual = table.columns[column];
  const mismatch =
    actual?.length !== expected.length ||
    expected.some((value, index) => {
      const observed = actual[index];
      return typeof value === 'number'
        ? typeof observed !== 'number' ||
            !Number.isFinite(observed) ||
            Math.abs(observed - value) > 1e-5 * Math.max(1, Math.abs(value))
        : observed !== value;
    });
  if (mismatch)
    throw new Error(
      `Zonal exposure ${tableId}.${column} mismatch: ${JSON.stringify({engine: result.engine, contributors: result.dispatchedContributors, expected, actual, summary: result.summary, diagnostics: result.diagnostics, stages: result.tables.find(candidate => candidate.id === 'recipe-status')})}`
    );
}

function checkGPU(result: CatalogRunResult, exposureEnabled: boolean) {
  const contributors = result.dispatchedContributors;
  if (
    result.engine !== 'gpu' ||
    result.fallbackReason ||
    !contributors.includes('GPUGroupStatistics') ||
    !contributors.includes('GPURasterNeighborhood') ||
    contributors.includes('GPURasterZonalStatistics') !== exposureEnabled ||
    result.summary['overflow'] !== 0 ||
    result.summary['invalidThreshold'] !== 0
  ) {
    throw new Error(
      `Zonal exposure must dispatch the real recipe contributors without overflow or fallback: ${JSON.stringify(result)}`
    );
  }
  const statuses = result.tables.find(table => table.id === 'recipe-status')!;
  if (!statuses || statuses.rowCount === 0)
    throw new Error(`Zonal exposure lost recipe statuses: ${JSON.stringify(result)}`);
  const summary = result.tables.find(table => table.id === 'zone-summary')!;
  const published = makeArrowTableFromCatalogTable(summary);
  if (
    JSON.stringify(Array.from(published.getChild('zone_id')!)) !==
      JSON.stringify(summary.columns['zone_id']) ||
    JSON.stringify(Array.from(published.getChild('exposed_share')!)) !==
      JSON.stringify(summary.columns['exposed_share']) ||
    summary.geometryColumn !== 'geometry' ||
    JSON.stringify(Array.from(published.getChild('geometry')!)) !==
      JSON.stringify(summary.columns['geometry'])
  ) {
    throw new Error(
      `Zonal exposure Arrow publication changed zone identities or exposure nulls: ${JSON.stringify(summary)}`
    );
  }
}

function checkZoneGeometry(result: CatalogRunResult, sample: ZonalExposureSample) {
  const zones = sample.inputs['zones'];
  if (zones.type !== 'FeatureCollection') throw new Error('Zonal sample must carry zone polygons');
  checkColumn(
    result,
    'zone-summary',
    'source_id',
    zones.features.map(feature => String(feature.id))
  );
  checkColumn(
    result,
    'zone-summary',
    'geometry',
    zones.features.map(feature => JSON.stringify(feature.geometry))
  );
}

// These literals are from the existing expansion-recipes and backlog-recipe-composition GPU
// fixtures, plus hand-calculated unweighted linear quantiles. No CPU recipe substitute runs.
it('dispatches zonal distributions and count/density exposure on sample and independently mapped data', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const primary = createZonalExposureSample();
  const primaryResult = await runCatalogOperationAsync(
    CURATED_ZONAL_EXPOSURE_BINDING,
    primary.parameters,
    primary.inputs,
    {device}
  );
  checkGPU(primaryResult, true);
  checkZoneGeometry(primaryResult, primary);
  checkColumn(primaryResult, 'zone-summary', 'zone_id', [0, 1, 2]);
  checkColumn(primaryResult, 'zone-summary', 'assigned_cell_count', [3, 1, 0]);
  checkColumn(primaryResult, 'zone-summary', 'valid_cell_count', [2, 1, 0]);
  checkColumn(primaryResult, 'zone-summary', 'median', [3, 8, null]);
  checkColumn(primaryResult, 'zone-summary', 'mode', [1, 8, null]);
  checkColumn(primaryResult, 'zone-summary', 'unique_count', [2, 1, 0]);
  checkColumn(primaryResult, 'zone-quantiles', 'value', [1, 3, 5, 8, 8, 8, null, null, null]);
  checkColumn(primaryResult, 'zone-summary', 'total_population', [60, 0, 0]);
  checkColumn(primaryResult, 'zone-summary', 'observed_population', [40, 0, 0]);
  checkColumn(primaryResult, 'zone-summary', 'exposed_population', [10, 0, 0]);
  checkColumn(primaryResult, 'zone-summary', 'coverage', [2 / 3, null, null]);
  checkColumn(primaryResult, 'zone-summary', 'exposed_share', [0.25, null, null]);

  const second = createMappedZonalExposureSample();
  const secondResult = await runCatalogOperationAsync(
    CURATED_ZONAL_EXPOSURE_BINDING,
    second.parameters,
    second.inputs,
    {device}
  );
  checkGPU(secondResult, true);
  checkZoneGeometry(secondResult, second);
  checkColumn(secondResult, 'zone-summary', 'zone_id', [0, 1, 2]);
  checkColumn(secondResult, 'zone-summary', 'assigned_cell_count', [2, 4, 0]);
  checkColumn(secondResult, 'zone-summary', 'valid_cell_count', [1, 4, 0]);
  checkColumn(secondResult, 'zone-summary', 'median', [5, 7.5, null]);
  checkColumn(secondResult, 'zone-summary', 'mode', [5, 1, null]);
  checkColumn(secondResult, 'zone-summary', 'unique_count', [1, 4, 0]);
  checkColumn(secondResult, 'zone-quantiles', 'value', [5, 5, 5, 1, 7.5, 9, null, null, null]);
  checkColumn(secondResult, 'zone-summary', 'total_population', [12, 12, 0]);
  checkColumn(secondResult, 'zone-summary', 'observed_population', [6, 12, 0]);
  checkColumn(secondResult, 'zone-summary', 'exposed_population', [6, 10, 0]);
  checkColumn(secondResult, 'zone-summary', 'coverage', [0.5, 1, null]);
  checkColumn(secondResult, 'zone-summary', 'exposed_share', [1, 5 / 6, null]);
});

it('omits the optional exposure branch and respects authoritative validity and unassigned ownership', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sample = createZonalExposureSample();
  const surface = sample.inputs['surface'];
  const cells = sample.inputs['cells'];
  if (surface.type !== 'raster' || cells.type !== 'table')
    throw new Error('Zonal sample input schema changed');
  surface.band.validity[3] = 0;
  sample.inputs['cells'] = {
    ...cells,
    table: {...cells.table, columns: {...cells.table.columns, zone: [null, 0, 0, 1]}}
  };
  const result = await runCatalogOperationAsync(
    CURATED_ZONAL_EXPOSURE_BINDING,
    {
      ...sample.parameters,
      weightMeaning: 'none',
      populationProperty: '',
      probabilities: [0.25, 0.75]
    },
    sample.inputs,
    {device}
  );
  checkGPU(result, false);
  checkColumn(result, 'zone-summary', 'assigned_cell_count', [2, 1, 0]);
  checkColumn(result, 'zone-summary', 'valid_cell_count', [1, 0, 0]);
  checkColumn(result, 'zone-summary', 'median', [1, null, null]);
  checkColumn(result, 'zone-summary', 'total_population', [null, null, null]);
  checkColumn(result, 'zone-summary', 'exposed_share', [null, null, null]);
  checkColumn(result, 'zone-quantiles', 'value', [1, 1, null, null, null, null]);
  if (result.summary['unassignedCellCount'] !== 1 || result.summary['exposureEnabled'] !== 0)
    throw new Error(`Zonal branch/ownership summary mismatch: ${JSON.stringify(result.summary)}`);
});
