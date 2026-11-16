// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getGPUEmergingHotSpotsParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeCatalogTable, appendCatalogRow} from '../../src/analysis/catalog-geojson';
import {
  computeSpaceTimeGiStar,
  computeBinPValues,
  computeAdjustedBinPValues,
  computeEmergingHotSpotCells
} from '../../../analysis/test/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spots-oracle';
import {checkEmergingHotSpotsProbabilitiesAndTrends} from './curated-emerging-hot-spots-oracle';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {
  CURATED_EMERGING_HOT_SPOTS_BINDING,
  prepareEmergingHotSpots
} from '../../src/curated/bindings/index';
import {
  createEmergingHotSpotsSample,
  EMERGING_HOT_SPOTS_SAMPLE_LICENSE
} from '../../src/curated/samples/index';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest,
  validateKeplerCuratedAnalysis
} from '../../src/curated/curated-analysis-mapping';
import {CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-emerging-hot-spots';
import {
  resolveKeplerOperationParameters,
  KEPLER_GRID_CELL_CAPS
} from '../../src/descriptors/default-parameters';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {createCuratedWorkflowAnalysisTemplate} from '../../src/workflows/workflow-analysis-template';
import {parseWorkflowAnalysisTemplate} from '../../src/workflows/workflow-analysis-template-parser';

const binding = CURATED_EMERGING_HOT_SPOTS_BINDING;
const analysis = getKeplerCuratedAnalysis('emerging-hot-spots')!;

describe('curated emerging hot spots', () => {
  it('registers the GPU recipe with a complete descriptor and portable template', () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    expect(
      validateKeplerOperationDescriptor(CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR, binding)
    ).toEqual([]);
    expect(validateKeplerCuratedAnalysis(analysis)).toEqual([]);
    expect(analysis.execution.status).toBe('available');
    const template = createCuratedWorkflowAnalysisTemplate(analysis);
    expect(parseWorkflowAnalysisTemplate(JSON.stringify(template))).toEqual(template);
    expect(template.exposed).toHaveLength(5);
    expect(template.slots[0].fields.map(field => field.role)).toEqual(['time']);
    expect(EMERGING_HOT_SPOTS_SAMPLE_LICENSE.license).toBe('MIT');
    expect(normalizeCatalogParameters(binding, getCatalogDefaultParameters(binding))).toEqual(
      getCatalogDefaultParameters(binding)
    );
  });

  it('prepares the recipe lattice and calendar fixture without losing integer epoch words', () => {
    const sample = createEmergingHotSpotsSample();
    expect(sample).toEqual(createEmergingHotSpotsSample());
    expect(sample).not.toEqual(createEmergingHotSpotsSample(101));
    const prepared = prepareEmergingHotSpots(sample, {coordinateSystem: 'planar', cellSize: 1});
    expect([prepared.width, prepared.height, prepared.cellCount]).toEqual([6, 5, 30]);
    expect(prepared.slices).toEqual({
      step: 'month',
      origin: {year: 2023, month: 1, day: 1, hour: 0},
      count: 12,
      interval: 1
    });
    expect(sample.features).toHaveLength(5472);
    const histogram = new Uint32Array(30 * 12);
    for (let index = 0; index < sample.features.length; index++) {
      const horizontal = prepared.positions[index * 2];
      const vertical = prepared.positions[index * 2 + 1];
      const cell = Math.floor(vertical) * 6 + Math.floor(horizontal);
      const month = new Date(Number(sample.features[index].properties!['observed'])).getUTCMonth();
      histogram[cell * 12 + month]++;
    }
    // Committed recipe-fixture counts, independent of the GPU recipe's binning.
    expect([...histogram.slice(0, 12)]).toEqual([11, 11, 9, 9, 10, 8, 9, 11, 12, 8, 10, 11]);
    expect([...histogram.slice(84, 96)]).toEqual([12, 15, 20, 26, 32, 40, 48, 53, 58, 65, 68, 78]);
    for (const index of [0, sample.features.length - 1]) {
      const timestamp =
        prepared.timestamps[index * 2] + prepared.timestamps[index * 2 + 1] * 2 ** 32;
      expect(timestamp).toBe(sample.features[index].properties!['observed']);
    }
  });

  it('maps an independent realization with renamed ISO time and runs preparation through the request', () => {
    const sample = createEmergingHotSpotsSample(101);
    for (const feature of sample.features)
      feature.properties = {
        recorded_at: new Date(Number(feature.properties!['observed'])).toISOString()
      };
    const profile = profileKeplerFeatureCollection(sample, {
      datasetId: 'independent',
      label: 'Independent events',
      coordinateSystem: 'planar',
      declarations: {rowSemantics: 'events'}
    });
    let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    mapping = updateKeplerCuratedAnalysisMapping(
      mapping,
      {slot: 'points', role: 'time', columns: ['recorded_at']},
      analysis,
      [profile]
    );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
      parameterOverrides: {cellSize: 1, coordinateSystem: 'planar'}
    });
    expect(request.inputDatasetIds).toEqual({points: 'independent'});
    expect(request.parameters['timeProperty']).toBe('recorded_at');
    const prepared = prepareEmergingHotSpots(sample, request.parameters);
    expect(prepared.slices.count).toBe(12);
    expect(prepared.timestamps.length).toBe(sample.features.length * 2);
  });

  it('normalizes seconds and timezone ISO epochs identically and preserves signed pre-epoch words', () => {
    const collection = (observed: unknown) => ({
      type: 'FeatureCollection' as const,
      features: [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {observed}}]
    });
    const parameters = {coordinateSystem: 'planar', cellSize: 1};
    const seconds = prepareEmergingHotSpots(collection(-1), {...parameters, timeUnit: 'seconds'});
    const iso = prepareEmergingHotSpots(collection('1970-01-01T00:59:59+01:00'), parameters);
    expect(seconds.timestamps).toEqual(iso.timestamps);
    expect([...seconds.timestamps]).toEqual([4294966296, 4294967295]);
    expect(
      prepareEmergingHotSpots(collection('2023-01-01T00:00:00'), parameters).timestamps
    ).toEqual(prepareEmergingHotSpots(collection('2023-01-01T00:00:00Z'), parameters).timestamps);
    expect(() => prepareEmergingHotSpots(collection(NaN), parameters)).toThrow(/observed/);
  });

  it('shares strict ISO normalization with profiling, including compact offsets and leap dates', () => {
    const collection = (observed: unknown) => ({
      type: 'FeatureCollection' as const,
      features: [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {observed}}]
    });
    for (const timestamp of [
      '2023-01-01T00:00:00+0000',
      '2024-02-29T01:30:00+0130',
      '2024-02-29T00:00:00Z',
      '2023-01-01T00:00:00',
      '2023-01-01',
      new Date('2023-01-01T00:00:00Z')
    ]) {
      const sample = collection(timestamp);
      const profile = profileKeplerFeatureCollection(sample, {
        datasetId: 'points',
        label: 'Events',
        coordinateSystem: 'planar',
        declarations: {rowSemantics: 'events'}
      });
      const mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
      expect(mapping.complete, String(timestamp)).toBe(true);
      const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
        parameterOverrides: {cellSize: 1, coordinateSystem: 'planar'}
      });
      const prepared = prepareEmergingHotSpots(sample, request.parameters);
      const expected =
        typeof timestamp === 'string'
          ? Date.parse(
              timestamp.length === 10 || !/(Z|[+-]\d{2}:?\d{2})$/.test(timestamp)
                ? timestamp + (timestamp.length === 10 ? 'T00:00:00Z' : 'Z')
                : timestamp
            )
          : timestamp.getTime();
      expect(prepared.timestamps[0] + prepared.timestamps[1] * 2 ** 32).toBe(expected);
    }
    for (const timestamp of [
      '2023-02-29T00:00:00Z',
      '2024-02-30T00:00:00+0000',
      '2023-04-31T00:00:00Z',
      '2023-01-01T24:00:00Z',
      '2023-01-01T00:00:00+2400'
    ]) {
      const sample = collection(timestamp);
      const profile = profileKeplerFeatureCollection(sample, {
        datasetId: 'points',
        label: 'Events',
        coordinateSystem: 'planar'
      });
      expect(profile.columns.find(column => column.name === 'observed')?.type).toBe('text');
      expect(createKeplerCuratedAnalysisMapping(analysis, [profile]).complete).toBe(false);
      expect(() =>
        prepareEmergingHotSpots(sample, {coordinateSystem: 'planar', cellSize: 1})
      ).toThrow(/curated-emerging-hot-spots.*valid ISO timestamp/);
    }
  });

  it('fits derived cells and cube bins on elongated and offset extents, while confirming unsafe overrides', () => {
    expect(KEPLER_GRID_CELL_CAPS[binding.id].maximumCellCount).toBe(4096);
    for (const [minimumX, minimumY, width, height] of [
      [0, 0, 10000, 10],
      [-10003, 1, 10000, 10],
      [3, 3, 1000, 1000],
      // At the uncapped derived size (200), this fits the cell limit but exceeds 65536 bins at 256 slices.
      [3, 3, 10000, 1000]
    ]) {
      const sample = {
        type: 'FeatureCollection' as const,
        features: [
          [minimumX, minimumY],
          [minimumX + width, minimumY + height]
        ].map(coordinates => ({
          geometry: {type: 'Point', coordinates},
          properties: {observed: Date.UTC(2023, 0, 1)}
        }))
      };
      const profile = profileKeplerFeatureCollection(sample, {
        datasetId: 'points',
        label: 'Events',
        coordinateSystem: 'planar',
        declarations: {rowSemantics: 'events'}
      });
      for (const sliceCount of [0, 12, 256]) {
        const resolved = resolveKeplerOperationParameters(
          binding,
          CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR,
          {points: profile},
          {fieldBindings: {timeProperty: 'observed'}, overrides: {sliceCount}}
        );
        const prepared = prepareEmergingHotSpots(sample, resolved.parameters);
        expect(prepared.cellCount).toBeLessThanOrEqual(4096);
        expect(prepared.cellCount * prepared.slices.count).toBeLessThanOrEqual(65536);
        expect(resolved.confirmations).toEqual([]);
        if (width === 10000) expect(Number(resolved.parameters['cellSize'])).toBeGreaterThan(2);
      }
      const explicit = resolveKeplerOperationParameters(
        binding,
        CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR,
        {points: profile},
        {overrides: {cellSize: 2, sliceCount: 256}}
      );
      expect(explicit.parameters['cellSize']).toBe(2);
      expect(explicit.confirmations.join(' ')).toContain('cell and cube-bin limits');
    }
    // The row cap and caller options may tighten, but cannot loosen, the binding's own limits.
    const sample = createEmergingHotSpotsSample();
    const profile = profileKeplerFeatureCollection(sample, {
      datasetId: 'points',
      label: 'Events',
      coordinateSystem: 'planar'
    });
    const resolved = resolveKeplerOperationParameters(
      binding,
      CURATED_EMERGING_HOT_SPOTS_DESCRIPTOR,
      {points: profile},
      {
        maximumCellCount: 1000000,
        fieldBindings: {timeProperty: 'observed'},
        overrides: {sliceCount: 256}
      }
    );
    const prepared = prepareEmergingHotSpots(sample, resolved.parameters);
    expect(prepared.cellCount * prepared.slices.count).toBeLessThanOrEqual(65536);
  });

  it('uses metres in the geographic local frame and restores original publication coordinates', () => {
    const sample = {
      type: 'FeatureCollection' as const,
      features: [
        [-87.65, 41.85],
        [-87.64, 41.86]
      ].map(coordinates => ({
        geometry: {type: 'Point', coordinates},
        properties: {observed: Date.UTC(2023, 0, 1)}
      }))
    };
    const prepared = prepareEmergingHotSpots(sample, {
      coordinateSystem: 'EPSG:4326',
      cellSize: 100
    });
    expect(prepared.width).toBeGreaterThan(5);
    expect(prepared.height).toBeGreaterThan(5);
    for (const feature of sample.features) {
      const [horizontal, vertical] = prepared.frame.forward(
        feature.geometry.coordinates as [number, number]
      );
      const [longitude, latitude] = prepared.frame.inverse(horizontal, vertical);
      expect(longitude).toBeCloseTo(feature.geometry.coordinates[0], 8);
      expect(latitude).toBeCloseTo(feature.geometry.coordinates[1], 8);
    }
  });

  it('rejects invalid geometry, missing mappings and bounded grid, time and event capacities before GPU allocation', () => {
    const sample = createEmergingHotSpotsSample();
    const parameters = {coordinateSystem: 'planar', cellSize: 1};
    expect(() => prepareEmergingHotSpots(sample, {...parameters, timeProperty: 'missing'})).toThrow(
      /curated-emerging-hot-spots.*missing/
    );
    expect(() => prepareEmergingHotSpots(sample, {...parameters, cellSize: 0.01})).toThrow(
      /cellCount/
    );
    expect(() => prepareEmergingHotSpots(sample, {...parameters, timeStep: 'hour'})).toThrow(
      /sliceCount/
    );
    expect(() => prepareEmergingHotSpots(sample, {...parameters, sliceCount: 257})).toThrow(
      /sliceCount/
    );
    expect(() => prepareEmergingHotSpots(sample, {...parameters, radius: 5})).toThrow(/radius/);
    expect(() =>
      prepareEmergingHotSpots(
        {...sample, features: Array(100001).fill(sample.features[0])},
        parameters
      )
    ).toThrow(/points.length/);
    expect(() =>
      prepareEmergingHotSpots({type: 'FeatureCollection', features: []}, parameters)
    ).toThrow(/points.length/);
    expect(() =>
      prepareEmergingHotSpots(
        {
          type: 'FeatureCollection',
          features: [
            {
              geometry: {
                type: 'LineString',
                coordinates: [
                  [0, 0],
                  [1, 1]
                ]
              },
              properties: {observed: 0}
            }
          ]
        },
        parameters
      )
    ).toThrow(/finite Point/);
    const corners = {
      type: 'FeatureCollection' as const,
      features: [
        [0, 0],
        [63, 63]
      ].map(coordinates => ({geometry: {type: 'Point', coordinates}, properties: {observed: 0}}))
    };
    expect(() => prepareEmergingHotSpots(corners, {...parameters, sliceCount: 17})).toThrow(
      /cellCount \* sliceCount/
    );
  });
});

for (const seedOffset of [0, 101]) {
  it(`probability and trend checks reject corrupted published columns for seed ${seedOffset}`, () => {
    const sample = createEmergingHotSpotsSample(seedOffset);
    const values = new Uint32Array(360);
    for (const feature of sample.features) {
      const [horizontal, vertical] = feature.geometry!.coordinates as [number, number];
      const cell = Math.floor(vertical) * 6 + Math.floor(horizontal);
      const month = new Date(Number(feature.properties!['observed'])).getUTCMonth();
      values[cell * 12 + month]++;
    }
    const shape = {gridWidth: 6, gridHeight: 5, sliceCount: 12, values};
    const packed = getGPUEmergingHotSpotsParameterValues({radius: 1.5, temporalWindow: 1});
    const scores = computeSpaceTimeGiStar(shape, packed, 4);
    const raw = computeBinPValues(scores);
    const adjusted = computeAdjustedBinPValues(scores, 'benjamini-hochberg');
    const oracle = computeEmergingHotSpotCells(scores, shape, packed, {adjustedPValues: adjusted});
    const uncorrected = computeEmergingHotSpotCells(scores, shape, packed, {adjustedPValues: raw});
    expect(
      uncorrected.category.filter((category, cell) => category !== oracle.category[cell]).length
    ).toBeGreaterThan(0);
    const bins = makeCatalogTable('cube', 'Cube', {
      gi_z: 'float64',
      p_value: 'float64',
      adjusted_p_value: 'float64'
    });
    for (let bin = 0; bin < scores.length; bin++)
      appendCatalogRow(bins, {
        gi_z: scores[bin],
        p_value: raw[bin],
        adjusted_p_value: adjusted[bin]
      });
    const cells = makeCatalogTable('cells', 'Cells', {
      trend_z: 'float64',
      trend_p: 'float64',
      last_valid_slice: 'int32'
    });
    for (let cell = 0; cell < 30; cell++)
      appendCatalogRow(cells, {
        trend_z: oracle.trendZ[cell],
        trend_p: oracle.trendP[cell],
        last_valid_slice: oracle.lastValidSlice[cell]
      });
    const check = () =>
      checkEmergingHotSpotsProbabilitiesAndTrends(bins, cells, shape, packed).mismatches;
    expect(check()).toEqual([]);
    // Replacing adjusted probabilities with raw ones must fail even if classification also
    // consumed the incorrect values. This is the reviewer's demonstrated false-positive path.
    bins.columns['adjusted_p_value'] = [...raw];
    expect(check().some(message => message.startsWith('adjusted_p_value'))).toBe(true);
    bins.columns['adjusted_p_value'] = [...adjusted];
    for (const [table, column] of [
      [bins, 'p_value'],
      [bins, 'adjusted_p_value'],
      [cells, 'trend_z'],
      [cells, 'trend_p'],
      [cells, 'last_valid_slice']
    ] as const) {
      const original = table.columns[column];
      table.columns[column] = original.map((value, row) =>
        row === 0 ? Number(value) + 0.25 : value
      );
      expect(
        check().some(message => message.startsWith(column)),
        column
      ).toBe(true);
      table.columns[column] = original;
    }
    const trendScores = cells.columns['trend_z'];
    cells.columns['trend_z'] = cells.columns['trend_p'];
    expect(check().some(message => message.startsWith('trend_z'))).toBe(true);
    cells.columns['trend_z'] = trendScores;
    expect(check()).toEqual([]);
  });
}
