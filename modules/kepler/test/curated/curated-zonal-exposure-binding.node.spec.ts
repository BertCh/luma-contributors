// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {describe, expect, it, vi} from 'vitest';
import {
  normalizeCatalogParameters,
  runCatalogOperationAsync,
  validateCatalogInputs
} from '../../src/analysis/catalog-bindings';
import {
  CURATED_ZONAL_EXPOSURE_BINDING,
  MAXIMUM_ZONAL_EXPOSURE_CELLS,
  MAXIMUM_ZONAL_EXPOSURE_ZONES,
  prepareZonalExposure,
  publishZonalExposure,
  type ZonalExposureReadback
} from '../../src/curated/bindings/curated-zonal-exposure-binding';
import {
  createMappedZonalExposureSample,
  createZonalExposureSample,
  type ZonalExposureSample
} from '../../src/curated/samples/zonal-exposure';
import {CURATED_ZONAL_EXPOSURE_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-zonal-exposure';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {makeArrowTableFromCatalogTable} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-table';

function prepare(sample = createZonalExposureSample(), overrides = {}) {
  return prepareZonalExposure(
    sample.inputs,
    normalizeCatalogParameters(CURATED_ZONAL_EXPOSURE_BINDING, {...sample.parameters, ...overrides})
  );
}

function replaceColumn(sample: ZonalExposureSample, name: string, values: (number | null)[]) {
  const input = sample.inputs['cells'];
  if (input.type !== 'table') throw new Error('Test fixture cells must be a table');
  sample.inputs['cells'] = {
    ...input,
    table: {...input.table, columns: {...input.table.columns, [name]: values}}
  };
}

// Hand-calculated distribution and the existing expansion recipe's population oracle.
// This is a publication-adapter fixture, not evidence that the GPU produced these values.
function makeReadback(): ZonalExposureReadback {
  return {
    rowCounts: new Uint32Array([2, 1, 0]),
    validCounts: new Uint32Array([2, 1, 0]),
    medians: new Float32Array([3, 8, NaN]),
    modes: new Float32Array([1, 8, NaN]),
    uniqueCounts: new Uint32Array([2, 1, 0]),
    quantiles: new Float32Array([1, 3, 5, 8, 8, 8, NaN, NaN, NaN]),
    exposure: {
      totalPopulation: new Float32Array([60, 0, 0]),
      observedPopulation: new Float32Array([40, 0, 0]),
      exposedPopulation: new Float32Array([10, 0, 0]),
      coverage: new Float32Array([2 / 3, NaN, NaN]),
      exposedShare: new Float32Array([0.25, NaN, NaN])
    },
    stages: [
      {stage: 'zone-distributions', fields: {count: 3, requiredCount: 3, overflow: 0}},
      {stage: 'total-population', fields: {overflow: 0}},
      {stage: 'observed-population', fields: {overflow: 0}},
      {stage: 'exposed-population', fields: {overflow: 0}},
      {stage: 'exposure-threshold', fields: {invalidCount: 0}}
    ]
  };
}

describe('curated zonal exposure preparation', () => {
  it('accepts both typed fixtures, preserves caller arrays and reconstructs shuffled cell identity', () => {
    for (const sample of [createZonalExposureSample(), createMappedZonalExposureSample()]) {
      expect(() =>
        validateCatalogInputs(CURATED_ZONAL_EXPOSURE_BINDING, sample.inputs)
      ).not.toThrow();
      expect(sample.license).toBe('MIT');
      expect(sample.attribution).toContain('synthetic');
      expect(sample.reference).toMatch(/recipes\/.*spec.ts$/);
    }
    const sample = createZonalExposureSample();
    const prepared = prepare(sample);
    expect(Array.from(prepared.zones)).toEqual([0, 0, 0, 1]);
    expect(Array.from(prepared.assignedCounts)).toEqual([3, 1, 0]);
    expect(Array.from(prepared.hazard)).toEqual([5, NaN, 1, 8]);
    expect(Array.from(prepared.hazardValidity)).toEqual([1, 0, 1, 1]);
    if (sample.inputs['surface'].type !== 'raster') throw new Error('Expected raster');
    expect(Array.from(sample.inputs['surface'].band.values)).toEqual([5, -999, 1, 8]);
    const mapped = prepare(createMappedZonalExposureSample());
    expect(Array.from(mapped.zones)).toEqual([0, 0, 1, 1, 1, 1]);
    expect(Array.from(mapped.population!)).toEqual([2, 3, 4, -1, 10, 5]);
    expect(Array.from(mapped.cellAreas!)).toEqual([3, 2, 0.5, 4, 0, 2]);
    expect(mapped.areaUnit).toBe('km2');
  });

  it('supports distribution alone without population, areas, units or a threshold', () => {
    const prepared = prepare(createZonalExposureSample(), {
      weightMeaning: 'none',
      populationProperty: '',
      cellAreaProperty: ''
    });
    expect(prepared.population).toBeUndefined();
    expect(prepared.cellAreas).toBeUndefined();
    expect(prepared.threshold).toBeUndefined();
    expect(Array.from(prepared.probabilities)).toEqual([0, 0.5, 1]);
  });

  it('keeps null zone ownership unassigned without inventing zone zero membership', () => {
    const sample = createZonalExposureSample();
    replaceColumn(sample, 'zone', [null, 0, 0, 1]);
    const prepared = prepare(sample);
    expect(Array.from(prepared.zones)).toEqual([0xffffffff, 0, 0, 1]);
    expect(Array.from(prepared.assignedCounts)).toEqual([2, 1, 0]);
  });

  it.each([
    ['duplicate indices', 'cell', [0, 0, 2, 3], /cellIndexProperty.*permutation/],
    ['out of range index', 'cell', [0, 1, 2, 4], /cellIndexProperty.*permutation/],
    ['fractional ownership', 'zone', [0, 0.5, 0, 1], /zoneProperty.*integer/],
    ['out of range zone', 'zone', [0, 0, 0, 3], /zoneProperty.*zoneCount/],
    ['float32 overflow', 'people', [1e100, 1, 1, 1], /populationProperty.*float32/]
  ] as const)('rejects %s before allocation', (_, column, values, message) => {
    const sample = createZonalExposureSample();
    replaceColumn(sample, column, [...values]);
    expect(() => prepare(sample)).toThrow(message);
  });

  it('requires explicit matching density area units, area columns and active thresholds', () => {
    expect(() => prepare(createMappedZonalExposureSample(), {cellAreaUnit: 'm2'})).toThrow(
      /cellAreaUnit.*densityAreaUnit/
    );
    expect(() =>
      prepare(createMappedZonalExposureSample(), {cellAreaUnit: '', densityAreaUnit: ''})
    ).toThrow(/nonempty densityAreaUnit/);
    expect(() => prepare(createMappedZonalExposureSample(), {cellAreaProperty: ''})).toThrow(
      /cellAreaProperty.*existing/
    );
    expect(() => prepare(createMappedZonalExposureSample(), {populationProperty: ''})).toThrow(
      /populationProperty.*existing/
    );
    const sample = createZonalExposureSample();
    delete sample.parameters['threshold'];
    expect(() => prepare(sample)).toThrow(/curated-zonal-exposure threshold.*finite/);
    expect(() => prepare(sample, {weightMeaning: 'none'})).not.toThrow();
    const zones = sample.inputs['zones'];
    if (zones.type !== 'FeatureCollection') throw new Error('Expected zone features');
    sample.inputs['zones'] = {
      ...zones,
      features: Array(MAXIMUM_ZONAL_EXPOSURE_ZONES + 1).fill(zones.features[0])
    };
    expect(() => prepare(sample)).toThrow(/zones.features.length/);
    expect(() => prepare(sample, {probabilities: [0, 2]})).toThrow();
  });

  it('enforces native validity before float32 conversion and rejects incompatible grid metadata/capacity', () => {
    const sample = createZonalExposureSample();
    const surface = sample.inputs['surface'];
    if (surface.type !== 'raster') throw new Error('Expected raster');
    surface.band.validity[0] = 0;
    expect(prepare(sample).hazard[0]).toBeNaN();
    sample.inputs['surface'] = {
      ...surface,
      grid: {...surface.grid, width: MAXIMUM_ZONAL_EXPOSURE_CELLS + 1}
    };
    expect(() => prepare(sample)).toThrow(/grid cell count/);
    sample.inputs['surface'] = {...surface, crs: 'EPSG:4326'};
    expect(() => prepare(sample)).toThrow(/crs and cellSizeMode/);
  });

  it('validates before any GPU resource allocation and never substitutes a CPU implementation', async () => {
    const sample = createMappedZonalExposureSample();
    const createBuffer = vi.fn();
    await expect(
      runCatalogOperationAsync(
        CURATED_ZONAL_EXPOSURE_BINDING,
        {...sample.parameters, cellAreaUnit: 'm2'},
        sample.inputs,
        {device: {createBuffer} as unknown as Device}
      )
    ).rejects.toThrow(/cellAreaUnit/);
    expect(createBuffer).not.toHaveBeenCalled();
    await expect(
      runCatalogOperationAsync(CURATED_ZONAL_EXPOSURE_BINDING, sample.parameters, sample.inputs)
    ).rejects.toThrow(/GPU device required/);
  });
});

describe('curated zonal exposure publication and descriptor', () => {
  it('publishes dense zone identity, unweighted quantiles, observed-support shares and plain Arrow tables', () => {
    const result = publishZonalExposure(prepare(), makeReadback());
    const summary = result.tables[0];
    expect(summary.columns['zone_id']).toEqual([0, 1, 2]);
    expect(summary.columns['assigned_cell_count']).toEqual([3, 1, 0]);
    expect(summary.columns['valid_cell_count']).toEqual([2, 1, 0]);
    expect(summary.columns['median']).toEqual([3, 8, null]);
    expect(summary.columns['total_population']).toEqual([60, 0, 0]);
    expect(summary.columns['observed_population']).toEqual([40, 0, 0]);
    expect(summary.columns['exposed_share']).toEqual([0.25, null, null]);
    expect(summary.columns['coverage'][0]).toBeCloseTo(2 / 3, 6);
    expect(result.tables[1].columns['value']).toEqual([1, 3, 5, 8, 8, 8, null, null, null]);
    expect(result.tables[2].rowCount).toBe(7);
    expect(summary.geometryColumn).toBe('geometry');
    expect(summary.columns['source_id']).toEqual([
      'count-zone-west-and-north',
      'count-zone-south-east',
      'count-zone-empty'
    ]);
    const sourceZones = createZonalExposureSample().inputs['zones'];
    if (sourceZones.type !== 'FeatureCollection') throw new Error('Expected zone features');
    expect(summary.columns['geometry'].map(value => JSON.parse(String(value)))).toEqual(
      sourceZones.features.map(feature => feature.geometry)
    );
    expect(Array.from(makeArrowTableFromCatalogTable(summary).getChild('median')!)).toEqual([
      3,
      8,
      null
    ]);
    expect(result.summary['exposureEnabled']).toBe(1);
    for (const output of CURATED_ZONAL_EXPOSURE_BINDING.outputs) {
      const table = result.tables.find(candidate => candidate.id === output.tableId)!;
      expect(table.rowCount).toBeLessThanOrEqual(output.maximumRows);
      for (const column of output.identityColumns) expect(table.columns[column]).toBeDefined();
    }
  });

  it('withholds incomplete statistics and carries the responsible recipe status', () => {
    const readback = makeReadback();
    readback.stages[0].fields['overflow'] = 1;
    const result = publishZonalExposure(prepare(), readback);
    expect(result.summary['overflow']).toBe(1);
    expect(result.tables[0].columns['median']).toEqual([null, null, null]);
    expect(result.tables[0].columns['exposed_share']).toEqual([null, null, null]);
    expect(result.tables[0].columns['zone_id']).toEqual([0, 1, 2]);
    expect(result.tables[0].columns['status']).toEqual(['incomplete', 'incomplete', 'incomplete']);
    expect(result.diagnostics.some(diagnostic => diagnostic.code === 'recipe-overflow')).toBe(true);
  });

  it('preserves population coverage but withholds invalid-threshold classification', () => {
    const readback = makeReadback();
    readback.stages[4].fields['invalidCount'] = 1;
    const result = publishZonalExposure(prepare(), readback);
    expect(result.tables[0].columns['total_population']).toEqual([60, 0, 0]);
    expect(result.tables[0].columns['exposed_population']).toEqual([null, null, null]);
    expect(result.tables[0].columns['exposed_share']).toEqual([null, null, null]);
    expect(result.summary['invalidThreshold']).toBe(1);
  });

  it('uses shared descriptor contracts with conditional numeric roles and retained zone geometry and report outputs', () => {
    expect(
      validateKeplerOperationDescriptor(
        CURATED_ZONAL_EXPOSURE_DESCRIPTOR,
        CURATED_ZONAL_EXPOSURE_BINDING
      )
    ).toEqual([]);
    expect(
      CURATED_ZONAL_EXPOSURE_DESCRIPTOR.inputs[2].fields.find(field => field.role === 'cell-area')
        ?.required
    ).toEqual({parameter: 'weightMeaning', equals: ['density']});
    expect(CURATED_ZONAL_EXPOSURE_DESCRIPTOR.outputs[0].presentation).toEqual({
      column: 'median',
      scale: 'sequential'
    });
    expect(CURATED_ZONAL_EXPOSURE_DESCRIPTOR.outputs[0].kindFrom).toBe('zones');
    expect(CURATED_ZONAL_EXPOSURE_DESCRIPTOR.notes!.join(' ')).toContain('observed population');
  });
});
