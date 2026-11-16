// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {expect, test} from 'vitest';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import type {CatalogRasterInput} from '../../src/analysis/catalog-raster-input';
import {CURATED_SURFACE_PREPARATION_BINDING} from '../../src/curated/bindings/curated-surface-preparation-binding';
import {
  makeSurfacePreparationSample,
  makeSurfacePreparationSecondSample
} from '../../src/curated/samples/surface-preparation';
import {getWorkflowCatalogInput, makeWorkflowRasterData} from '../../src/workflows/workflow-data';

async function runMapped(
  surface: CatalogRasterInput,
  parameters: Record<string, CatalogParameterValue>
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const mapped = getWorkflowCatalogInput(
    makeWorkflowRasterData(surface, surface.grid.width * surface.grid.height),
    0
  );
  const result = await runCatalogOperationAsync(
    CURATED_SURFACE_PREPARATION_BINDING,
    parameters,
    {surface: mapped},
    {device}
  );
  const report = JSON.stringify({
    source: surface.id,
    parameters,
    engine: result.engine,
    dispatched: result.dispatchedContributors,
    summary: result.summary,
    columns: result.tables[0]?.columns
  });
  expect(result.engine, report).toBe('gpu');
  expect(result.dispatchedContributors, report).toEqual(
    parameters['filter'] ? ['GPURasterFillNoData', 'GPURasterRankFilter'] : ['GPURasterFillNoData']
  );
  const table = result.tables[0];
  const raster = table.raster;
  if (!raster) throw new Error(`surface preparation lost its computational raster: ${report}`);
  expect(raster.grid, report).toEqual(surface.grid);
  expect(raster.crs, report).toBe(surface.crs);
  expect(raster.cellSizeMode, report).toBe(surface.cellSizeMode);
  expect(raster.band.index, report).toBe(surface.band.index);
  expect(raster.band.noDataValue, report).toBeNull();
  expect(Array.from(raster.band.validity, Boolean), report).toEqual(table.columns['valid']);
  expect(
    Array.from(raster.band.values, (value, index) => (raster.band.validity[index] ? value : null)),
    report
  ).toEqual(table.columns['value']);
  const mappedOutput = getWorkflowCatalogInput(makeWorkflowRasterData(raster, table.rowCount), 0);
  expect(mappedOutput, report).toBe(raster);
  return {result, table, report};
}

test('mapped sample matches expansion-recipes original-donor nearest fixture and publishes gap footprints', async () => {
  const {result, table, report} = await runMapped(makeSurfacePreparationSample(), {
    maximumDistance: 1,
    method: 'nearest'
  });
  expect(table.columns['value'], report).toEqual([2, 2, null, null]);
  expect(table.columns['original_valid'], report).toEqual([true, false, false, false]);
  expect(table.columns['filled'], report).toEqual([false, true, false, false]);
  expect(table.columns['filtered'], report).toEqual([false, false, false, false]);
  expect(result.summary['unresolvedCount'], report).toBe(2);
  expect(table.geometryColumn, report).toBe('geometry');
  expect(
    table.columns['geometry'].every(geometry => JSON.parse(geometry as string).type === 'Polygon'),
    report
  ).toBe(true);
});

test('independently mapped second band matches backlog-recipe-composition fill/filter fixture', async () => {
  const surface = makeSurfacePreparationSecondSample();
  const {result, table, report} = await runMapped(surface, {
    maximumDistance: 1,
    method: 'nearest',
    filter: true,
    radius: 1,
    quantile: 0.5,
    borderMode: 'nodata',
    noDataPolicy: 'ignore'
  });
  expect(table.columns['value'], report).toEqual([2, 2, null, null, 10, 10]);
  expect(table.columns['original_valid'], report).toEqual([true, false, false, false, false, true]);
  expect(table.columns['filled'], report).toEqual([false, true, false, false, true, false]);
  expect(table.columns['filtered'], report).toEqual(Array(6).fill(false));
  expect(table.columns['source_id'], report).toEqual(Array(6).fill('independent-survey'));
  expect(table.columns['band_index'], report).toEqual(Array(6).fill(3));
  expect(table.columns['cell_index'], report).toEqual([0, 1, 2, 3, 4, 5]);
  expect(result.summary['unresolvedCount'], report).toBe(2);
});

test('inverse-distance averages original donors and never cascades filled values', async () => {
  const surface = makeSurfacePreparationSecondSample();
  const input = {
    ...surface,
    grid: {...surface.grid, width: 5},
    band: {
      ...surface.band,
      values: new Float32Array([0, NaN, 12, NaN, NaN]),
      validity: new Uint8Array([1, 0, 1, 0, 0]),
      noDataValue: null
    }
  };
  const {table, report} = await runMapped(input, {maximumDistance: 1, method: 'inverse-distance'});
  // The center gap has two unit-distance donors; the last gap is beyond both original donors.
  expect(table.columns['value'], report).toEqual([0, 6, 12, 12, null]);
});

test('rank filter changes observed and filled cells independently without creating observations', async () => {
  const source = makeSurfacePreparationSample();
  const surface = {
    ...source,
    band: {
      ...source.band,
      values: new Float32Array([0, NaN, 10, 0]),
      validity: new Uint8Array([1, 0, 1, 1])
    }
  };
  const {table, report} = await runMapped(surface, {
    maximumDistance: 1,
    method: 'inverse-distance',
    filter: true,
    radius: 1,
    borderMode: 'nodata',
    noDataPolicy: 'ignore',
    quantile: 0
  });
  expect(table.columns['value'], report).toEqual([0, 0, 0, 0]);
  expect(table.columns['original_valid'], report).toEqual([true, false, true, true]);
  expect(table.columns['filled'], report).toEqual([false, true, false, false]);
  expect(table.columns['filtered'], report).toEqual([false, true, true, false]);
});

test('propagating filter can invalidate original and filled centers and reports post-filter unresolved count', async () => {
  const {table, result, report} = await runMapped(makeSurfacePreparationSample(), {
    maximumDistance: 1,
    method: 'nearest',
    filter: true,
    radius: 1,
    borderMode: 'nodata',
    noDataPolicy: 'propagate'
  });
  expect(table.columns['value'], report).toEqual([null, null, null, null]);
  expect(table.columns['status'], report).toEqual([
    'filter-invalidated',
    'filter-invalidated',
    'unresolved',
    'unresolved'
  ]);
  expect(result.summary['filterInvalidatedCount'], report).toBe(2);
  expect(result.summary['unresolvedCount'], report).toBe(4);
});

test('nodata sentinel and nonfinite samples are excluded even when source mask claims validity', async () => {
  const source = makeSurfacePreparationSample();
  const surface = {
    ...source,
    band: {
      ...source.band,
      values: new Float64Array([3, -9999, Infinity, NaN]),
      validity: new Uint8Array([1, 1, 1, 1]),
      noDataValue: -9999
    }
  };
  const {table, report} = await runMapped(surface, {maximumDistance: 1, method: 'nearest'});
  expect(table.columns['value'], report).toEqual([3, 3, null, null]);
  expect(table.columns['original_valid'], report).toEqual([true, false, false, false]);
  expect(table.columns['filled'], report).toEqual([false, true, false, false]);
});

test('two-dimensional disk excludes diagonal donors outside the pixel radius', async () => {
  const source = makeSurfacePreparationSample();
  const surface = {
    ...source,
    grid: {...source.grid, width: 3, height: 3},
    band: {
      ...source.band,
      values: new Float32Array([2, NaN, 10, NaN, NaN, NaN, NaN, NaN, NaN]),
      validity: new Uint8Array([1, 0, 1, 0, 0, 0, 0, 0, 0])
    }
  };
  const {table, result, report} = await runMapped(surface, {
    maximumDistance: 1,
    method: 'inverse-distance'
  });
  expect(table.columns['value'], report).toEqual([2, 6, 10, 2, null, 10, null, null, null]);
  expect(table.columns['row'], report).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2]);
  expect(table.columns['column'], report).toEqual([0, 1, 2, 0, 1, 2, 0, 1, 2]);
  expect(result.summary['unresolvedCount'], report).toBe(4);
});

test('a completely missing surface remains unresolved rather than publishing zero estimates', async () => {
  const source = makeSurfacePreparationSample();
  const surface = {
    ...source,
    band: {...source.band, values: new Float32Array(4), validity: new Uint8Array(4)}
  };
  const {table, result, report} = await runMapped(surface, {maximumDistance: 1, method: 'nearest'});
  expect(table.columns['value'], report).toEqual([null, null, null, null]);
  expect(table.columns['filled'], report).toEqual(Array(4).fill(false));
  expect(result.summary['unresolvedCount'], report).toBe(4);
});

test.each([
  {values: new Int32Array([16777216, 16777217]), label: 'signed integer'},
  {values: new Uint32Array([16777216, 16777217]), label: 'unsigned integer'},
  {values: new Float64Array([16777216, 16777217]), label: 'Float64'}
])('source nodata survives float32 collisions in a $label band', async ({values}) => {
  const source = makeSurfacePreparationSample();
  for (const validity of [new Uint8Array([1, 0]), new Uint8Array([1, 1])]) {
    const surface = {
      ...source,
      grid: {...source.grid, width: 2},
      band: {...source.band, values, validity, noDataValue: 16777217}
    };
    const {table, result, report} = await runMapped(surface, {
      maximumDistance: 1,
      method: 'nearest'
    });
    expect(table.columns['value'], report).toEqual([16777216, 16777216]);
    expect(table.columns['original_valid'], report).toEqual([true, false]);
    expect(table.columns['filled'], report).toEqual([false, true]);
    expect(result.summary['unresolvedCount'], report).toBe(0);
  }
});

test('finite Float64 nodata outside float32 range does not become a rejected GPU sentinel', async () => {
  const source = makeSurfacePreparationSample();
  const surface = {
    ...source,
    grid: {...source.grid, width: 2},
    band: {
      ...source.band,
      values: new Float64Array([7, 1e100]),
      validity: new Uint8Array([1, 1]),
      noDataValue: 1e100
    }
  };
  const {table, result, report} = await runMapped(surface, {maximumDistance: 1, method: 'nearest'});
  expect(table.columns['value'], report).toEqual([7, 7]);
  expect(table.columns['original_valid'], report).toEqual([true, false]);
  expect(table.columns['filled'], report).toEqual([false, true]);
  expect(result.summary['unresolvedCount'], report).toBe(0);
});
