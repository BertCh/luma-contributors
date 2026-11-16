// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareAnalysisResultPublication} from '../../../../examples/deck/kepler-compute/src/analysis-result-publication';
import type {Device} from '@luma.gl/core';
import * as analysisRecipes from '@luma-contributors/analysis/gpu-spatial-analysis';
import {afterEach, describe, expect, test, vi} from 'vitest';
import * as gpuSessions from '../../src/analysis/catalog-gpu-session';
import type {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import type {CatalogRasterInput} from '../../src/analysis/catalog-raster-input';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import {
  CURATED_SURFACE_PREPARATION_BINDING,
  prepareSurfacePreparation,
  publishSurfacePreparation
} from '../../src/curated/bindings/curated-surface-preparation-binding';
import {
  makeSurfacePreparationSample,
  makeSurfacePreparationSecondSample,
  SURFACE_PREPARATION_SAMPLE_LICENSE
} from '../../src/curated/samples/surface-preparation';
import {CURATED_SURFACE_PREPARATION_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-surface-preparation';
import {prepareCuratedDrainageNetwork} from '../../src/curated/bindings/curated-drainage-network-binding';
import {getWorkflowCatalogInput, makeWorkflowRasterData} from '../../src/workflows/workflow-data';
import {makeWorkflowOperationFromCatalogBinding} from '../../src/workflows/workflow-operations';

const PARAMETERS = {maximumDistance: 1, method: 'nearest'};

function fixtureReadback() {
  return {
    values: new Float32Array([2, 2, NaN, NaN]),
    validity: new Uint32Array([1, 1, 0, 0]),
    originalValidity: new Uint32Array([1, 0, 0, 0]),
    filled: new Uint32Array([0, 1, 0, 0]),
    filtered: new Uint32Array(4),
    unresolvedCount: 2
  };
}

describe('curated surface preparation boundary and publication', () => {
  test('maps two independent decoded bands without fake geometry or copied grids', () => {
    for (const surface of [makeSurfacePreparationSample(), makeSurfacePreparationSecondSample()]) {
      const mapped = getWorkflowCatalogInput(
        makeWorkflowRasterData(surface, surface.grid.width * surface.grid.height),
        0
      );
      const prepared = prepareSurfacePreparation({surface: mapped}, PARAMETERS);
      expect(prepared.surface).toBe(surface);
      expect(prepared.surface.band.values).toBe(surface.band.values);
      expect(prepared.settings).toMatchObject({...PARAMETERS, filter: false});
    }
    const operation = makeWorkflowOperationFromCatalogBinding(CURATED_SURFACE_PREPARATION_BINDING);
    expect(operation.inputs[0]).toMatchObject({name: 'surface', accepts: ['raster']});
    expect(SURFACE_PREPARATION_SAMPLE_LICENSE).toMatchObject({license: 'MIT'});
    expect(makeSurfacePreparationSample()).toEqual(makeSurfacePreparationSample());
  });

  test('rejects invalid shapes, CRS declarations, parameters and search capacity before allocation', () => {
    const surface = makeSurfacePreparationSample();
    expect(() => prepareSurfacePreparation({}, PARAMETERS)).toThrow(
      /curated-surface-preparation.*surface/
    );
    expect(() =>
      prepareSurfacePreparation({surface: {type: 'FeatureCollection', features: []}}, PARAMETERS)
    ).toThrow(/surface/);
    expect(() =>
      prepareSurfacePreparation(
        {surface: {...surface, band: {...surface.band, validity: new Uint8Array(1)}}},
        PARAMETERS
      )
    ).toThrow(/validity length/);
    expect(() =>
      prepareSurfacePreparation({surface: {...surface, cellSizeMode: 'uniform'}}, PARAMETERS)
    ).toThrow(/crs and cellSizeMode/);
    expect(() =>
      prepareSurfacePreparation(
        {surface: {...surface, grid: {...surface.grid, width: 65537}}},
        PARAMETERS
      )
    ).toThrow(/grid cell count/);
    const large = {
      ...surface,
      grid: {...surface.grid, width: 65536},
      band: {...surface.band, values: new Float32Array(65536), validity: new Uint8Array(65536)}
    };
    expect(() => prepareSurfacePreparation({surface: large}, {maximumDistance: 16})).toThrow(
      /search samples/
    );
    const invalidParameters: Record<string, CatalogParameterValue>[] = [
      {maximumDistance: 0},
      {maximumDistance: 1.5},
      {maximumDistance: 17},
      {method: 'bogus'},
      {filter: 'true'},
      {radius: 5},
      {quantile: NaN},
      {quantile: 2},
      {borderMode: 'bogus'},
      {noDataPolicy: 'bogus'}
    ];
    for (const parameters of invalidParameters) {
      expect(() => prepareSurfacePreparation({surface}, parameters)).toThrow(
        /curated-surface-preparation/
      );
    }
  });

  test('publishes the existing four-cell recipe fixture with unresolved values null and stable identities', () => {
    const surface = makeSurfacePreparationSample();
    const {settings} = prepareSurfacePreparation({surface}, PARAMETERS);
    const result = publishSurfacePreparation(surface, settings, fixtureReadback());
    const table = result.tables[0];
    expect(table.rowCount).toBe(4);
    expect(table.columns['value']).toEqual([2, 2, null, null]);
    expect(table.columns['cell_index']).toEqual([0, 1, 2, 3]);
    expect(table.columns['source_id']).toEqual(Array(4).fill(surface.id));
    expect(table.columns['original_valid']).toEqual([true, false, false, false]);
    expect(table.columns['filled']).toEqual([false, true, false, false]);
    expect(table.columns['filtered']).toEqual([false, false, false, false]);
    expect(table.columns['status']).toEqual(['observed', 'filled', 'unresolved', 'unresolved']);
    expect(JSON.parse(table.columns['geometry'][0] as string)).toEqual({
      type: 'Polygon',
      coordinates: [
        [
          [-74, 41],
          [-73.99, 41],
          [-73.99, 40.99],
          [-74, 40.99],
          [-74, 41]
        ]
      ]
    });
    expect(result.summary).toMatchObject({
      filledCount: 1,
      originalCount: 1,
      unresolvedCount: 2,
      distanceUnit: 'pixels'
    });
    expect(result.diagnostics[0]).toMatchObject({code: 'unresolved-cells', count: 2});
  });

  test('retains fill and filter overlap and separates filter invalidation from unresolved original gaps', () => {
    const surface = makeSurfacePreparationSample();
    const {settings} = prepareSurfacePreparation({surface}, {...PARAMETERS, filter: true});
    const result = publishSurfacePreparation(surface, settings, {
      ...fixtureReadback(),
      validity: new Uint32Array([0, 1, 0, 0]),
      filtered: new Uint32Array([0, 1, 0, 0]),
      unresolvedCount: 3
    });
    expect(result.tables[0].columns['filled'][1]).toBe(true);
    expect(result.tables[0].columns['filtered'][1]).toBe(true);
    expect(result.tables[0].columns['status']).toEqual([
      'filter-invalidated',
      'filtered',
      'unresolved',
      'unresolved'
    ]);
    expect(result.summary).toMatchObject({filterInvalidatedCount: 1, unresolvedCount: 3});
  });

  test('honors point-pixel footprint offsets and leaves projected coordinates out of GeoJSON', () => {
    const surface = makeSurfacePreparationSample();
    const {settings} = prepareSurfacePreparation({surface}, PARAMETERS);
    const pointResult = publishSurfacePreparation(
      {...surface, grid: {...surface.grid, pixelInterpretation: 'point'}},
      settings,
      fixtureReadback()
    );
    const ring = JSON.parse(pointResult.tables[0].columns['geometry'][0] as string).coordinates[0];
    expect(ring[0]).toEqual([-74.005, 41.005]);
    for (const crs of ['planar', 'EPSG:3857'] as const) {
      const result = publishSurfacePreparation({...surface, crs}, settings, fixtureReadback());
      expect(result.tables[0].geometryColumn).toBeUndefined();
      expect(result.tables[0].columns['geometry']).toBeUndefined();
      expect(prepareAnalysisResultPublication(result.tables[0]).analysisResult.destination).toBe(
        'report'
      );
      expect(
        result.diagnostics.some(diagnostic => diagnostic.code === 'projected-table-only')
      ).toBe(true);
    }
  });

  test('has complete descriptor defaults and output schema, and refuses implicit CPU execution', async () => {
    expect(
      CURATED_SURFACE_PREPARATION_DESCRIPTOR.parameters.map(parameter => parameter.name)
    ).toEqual(CURATED_SURFACE_PREPARATION_BINDING.parameters.map(parameter => parameter.name));
    expect(
      CURATED_SURFACE_PREPARATION_DESCRIPTOR.parameters.every(
        parameter => parameter.apply === 'rebuild' && parameter.defaultPolicy.from === 'constant'
      )
    ).toBe(true);
    expect(CURATED_SURFACE_PREPARATION_DESCRIPTOR.outputs[0].tableId).toBe(
      CURATED_SURFACE_PREPARATION_BINDING.outputs[0].tableId
    );
    const runDataGpu = vi.fn(CURATED_SURFACE_PREPARATION_BINDING.runDataGpu!);
    await expect(
      runCatalogOperationAsync({...CURATED_SURFACE_PREPARATION_BINDING, runDataGpu}, PARAMETERS, {
        surface: makeSurfacePreparationSample()
      })
    ).rejects.toThrow();
    expect(runDataGpu).not.toHaveBeenCalled();
  });
});

// Capture the actual binding upload and recipe boundary without claiming GPU numerical evidence.
async function inspectSurfacePreparationUpload(surface: CatalogRasterInput) {
  const count = surface.grid.width * surface.grid.height;
  const input = vi.fn(
    (name: string, values: Float32Array | Int32Array | Uint32Array, format: string) => ({
      name,
      values,
      format
    })
  );
  const session = {
    graph: {},
    input,
    output: (name: string, format: string, length: number) => ({name, view: {}, length, format}),
    run: vi.fn(),
    readFloat32: async () => new Float32Array(count),
    readUint32: async () => new Uint32Array(count),
    readScalar: async () => count
  } as unknown as CatalogGpuSession;
  vi.spyOn(gpuSessions, 'withCatalogGpuSession').mockImplementation(async (_device, _id, action) =>
    action(session)
  );
  const recipe = vi
    .spyOn(analysisRecipes, 'addSurfacePreparationRecipe')
    .mockReturnValue({} as ReturnType<typeof analysisRecipes.addSurfacePreparationRecipe>);
  await CURATED_SURFACE_PREPARATION_BINDING.runDataGpu!(
    {device: {} as Device},
    {surface},
    PARAMETERS
  );
  return {input, band: recipe.mock.lastCall![1].input};
}

afterEach(() => vi.restoreAllMocks());

describe('surface preparation nodata upload regressions', () => {
  test.each([
    {values: new Int32Array([16777216, 16777217]), format: 'sint32'},
    {values: new Uint32Array([16777216, 16777217]), format: 'uint32'},
    {values: new Float64Array([16777216, 16777217]), format: 'float32'}
  ])('keeps a valid donor distinct from a float32-colliding sentinel for $format / $values', async ({
    values,
    format
  }) => {
    const source = makeSurfacePreparationSample();
    // The first mask reproduces the reported case. The second proves sentinel folding even
    // when ingestion did not already mark it invalid; nonzero mask words are normalized.
    for (const validity of [new Uint8Array([1, 0]), new Uint32Array([7, 3])]) {
      const surface = {
        ...source,
        grid: {...source.grid, width: 2},
        band: {...source.band, values, validity, noDataValue: 16777217}
      };
      const {input, band} = await inspectSurfacePreparationUpload(surface);
      const uploadedValues = input.mock.calls.find(call => call[0] === 'source-values')!;
      const uploadedValidity = input.mock.calls.find(call => call[0] === 'source-validity')!;
      expect(band.format).toBe(format);
      expect(band).not.toHaveProperty('noDataValue');
      expect(uploadedValidity[1]).toEqual(new Uint32Array([1, 0]));
      expect(uploadedValues[2]).toBe(format);
      expect(Array.from(uploadedValues[1])).toEqual(
        format === 'float32' ? [16777216, 16777216] : [16777216, 16777217]
      );
      expect(surface.band.values).toBe(values);
      expect(Array.from(validity)).toEqual(validity instanceof Uint8Array ? [1, 0] : [7, 3]);
    }
  });

  test.each([
    {values: new Int8Array([-7, -8]), format: 'sint32', sentinel: -8},
    {values: new Int16Array([-32767, -32768]), format: 'sint32', sentinel: -32768},
    {values: new Uint8Array([254, 255]), format: 'uint32', sentinel: 255},
    {values: new Uint16Array([65534, 65535]), format: 'uint32', sentinel: 65535}
  ])('widens $values exactly to $format while preserving source nodata', async ({
    values,
    format,
    sentinel
  }) => {
    const source = makeSurfacePreparationSample();
    const surface = {
      ...source,
      grid: {...source.grid, width: 2},
      band: {...source.band, values, validity: new Uint8Array([1, 1]), noDataValue: sentinel}
    };
    const {input, band} = await inspectSurfacePreparationUpload(surface);
    const upload = input.mock.calls.find(call => call[0] === 'source-values')!;
    expect(band.format).toBe(format);
    expect(upload[1]).toBeInstanceOf(format === 'sint32' ? Int32Array : Uint32Array);
    expect(Array.from(upload[1])).toEqual(Array.from(values));
    expect(input.mock.calls.find(call => call[0] === 'source-validity')![1]).toEqual(
      new Uint32Array([1, 0])
    );
    expect(band).not.toHaveProperty('noDataValue');
  });

  test('classifies a finite Float64 sentinel outside float32 range before upload', async () => {
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
    const {input, band} = await inspectSurfacePreparationUpload(surface);
    expect(band).not.toHaveProperty('noDataValue');
    expect(input.mock.calls.find(call => call[0] === 'source-validity')![1]).toEqual(
      new Uint32Array([1, 0])
    );
    expect(input.mock.calls.find(call => call[0] === 'source-values')![2]).toBe('float32');
    expect(surface.band.values).toEqual(new Float64Array([7, 1e100]));
  });
});

test('retains computed samples and validity in a typed raster independently of the polygon display', () => {
  for (const source of [makeSurfacePreparationSample(), makeSurfacePreparationSecondSample()]) {
    for (const crs of ['EPSG:4326', 'EPSG:3857', 'planar'] as const) {
      const surface = {
        ...source,
        crs,
        cellSizeMode:
          crs === 'EPSG:4326'
            ? ('geographic' as const)
            : crs === 'EPSG:3857'
              ? ('web-mercator' as const)
              : ('uniform' as const)
      };
      const count = surface.grid.width * surface.grid.height;
      const readback = {
        values: Float32Array.from({length: count}, (_, index) => index + 10),
        validity: Uint32Array.from({length: count}, (_, index) => Number(index !== 1)),
        originalValidity: new Uint32Array(count),
        filled: new Uint32Array(count),
        filtered: new Uint32Array(count),
        unresolvedCount: 1
      };
      const {settings} = prepareSurfacePreparation({surface}, PARAMETERS);
      const output = publishSurfacePreparation(surface, settings, readback);
      const raster = output.tables[0].raster;
      if (!raster) throw new Error('prepared surface lost its computational raster');
      expect(raster.grid).toBe(surface.grid);
      expect(raster.crs).toBe(crs);
      expect(raster.cellSizeMode).toBe(surface.cellSizeMode);
      expect(raster.band.values).toBe(readback.values);
      expect(raster.band.validity).toBe(readback.validity);
      expect(raster.band.index).toBe(surface.band.index);
      expect(raster.band.noDataValue).toBeNull();
      const mapped = getWorkflowCatalogInput(makeWorkflowRasterData(raster, count), 0);
      expect(mapped).toBe(raster);
      if (crs === 'planar') {
        const drainage = prepareCuratedDrainageNetwork({elevation: mapped}, {});
        expect(drainage.raster).toBe(raster);
        expect(Array.from(drainage.elevationBand.validity)).toEqual(Array.from(readback.validity));
      }
      expect(output.tables[0].geometryColumn).toBe(crs === 'EPSG:4326' ? 'geometry' : undefined);
    }
  }
  expect(CURATED_SURFACE_PREPARATION_BINDING.outputs[0].dataKind).toBe('raster');
});
