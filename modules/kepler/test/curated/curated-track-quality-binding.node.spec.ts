// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareAnalysisResultPublication} from '../../../../examples/deck/kepler-compute/src/analysis-result-publication';
import {describe, expect, it} from 'vitest';
import {makeArrowTableFromCatalogTable} from '../../../../examples/deck/kepler-compute/src/analysis-catalog-table';
import {previewArrowSource} from '../../../../examples/deck/kepler-compute/src/source-preview';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  KEPLER_CATALOG_BINDINGS,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {
  CURATED_TRACK_QUALITY_BINDING,
  prepareTrackQuality,
  publishTrackQuality,
  type TrackQualityReadback
} from '../../src/curated/bindings/index';
import {
  createTrackQualitySample,
  TRACK_QUALITY_SAMPLE_LICENSE
} from '../../src/curated/samples/index';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {
  createKeplerCuratedAnalysisSample,
  getKeplerCuratedAnalysisSampleParameters,
  getKeplerCuratedAnalysisSampleMetadata
} from '../../src/curated/curated-analysis-samples';
import {
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping,
  buildKeplerCuratedAnalysisRequest,
  validateKeplerCuratedAnalysis
} from '../../src/curated/curated-analysis-mapping';
import {CURATED_TRACK_QUALITY_DESCRIPTOR} from '../../src/descriptors/descriptors-curated-track-quality';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
import {validateKeplerOperationDescriptor} from '../../src/descriptors/validate-operation-descriptor';
import {resolveKeplerOperationParameters} from '../../src/descriptors/default-parameters';
import {suggestKeplerAnalyses} from '../../src/descriptors/applicability';
import {profileGallerySample} from '../descriptors/gallery-sample-profiles';
import {profileKeplerNativeDataset} from '../../src/profile/native-dataset-profile';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {createCuratedWorkflowAnalysisTemplate} from '../../src/workflows/workflow-analysis-template';
import {
  parseWorkflowAnalysisTemplate,
  validateWorkflowAnalysisTemplate
} from '../../src/workflows/workflow-analysis-template-parser';

const binding = CURATED_TRACK_QUALITY_BINDING;
const analysis = getKeplerCuratedAnalysis('track-quality')!;
const parameters = {coordinateSystem: 'planar'};
// Hand-computed recipe-review [0,1,502,3,4], a 120-second gap, and a singleton.
const readback: TrackQualityReadback = {
  stepSpeeds: new Float32Array([0, 0.001, 0.501, 0.499, 0.001, 0, 1 / 120000, 0]),
  trackLengths: new Float32Array([1002, 1, 0]),
  trackDurations: new Float32Array([4000, 120000, 0]),
  maximumSpeeds: new Float32Array([0.501, 1 / 120000, 0]),
  flags: new Uint32Array([0, 0, 4, 4, 0, 0, 8, 0]),
  flaggedCounts: new Uint32Array([2, 1, 0])
};

function profileTracks(collection: CatalogFeatureCollection, renamed: boolean) {
  const trackColumn = renamed ? 'vehicle' : 'track_id';
  const timeColumn = renamed ? 'recorded_at' : 'timestamp';
  return profileKeplerNativeDataset(
    {
      rowCount: collection.features.length,
      geometryColumn: 'geometry',
      columns: [
        {
          name: 'geometry',
          type: 'string',
          values: collection.features.map(feature => JSON.stringify(feature.geometry))
        },
        ...[trackColumn, timeColumn].map(name => ({
          name,
          values: collection.features.map(feature => feature.properties![name])
        }))
      ]
    },
    {
      datasetId: renamed ? 'independent' : 'tracks',
      label: 'Declared tracks',
      coordinateSystem: 'planar'
    },
    {
      kind: 'trajectories',
      trackColumn,
      timeColumn,
      timestampEncoding: renamed ? 'iso8601' : 'unix-millisecond'
    }
  );
}

describe('curated track quality', () => {
  it('keeps points-time gallery recommendations unchanged when track quality is removed', () => {
    const profile = profileGallerySample('points-time');
    const withTrackQuality = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [profile]);
    const withoutTrackQuality = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS.filter(candidate => candidate.id !== binding.id),
      [profile]
    );
    expect(withTrackQuality.suggestions).toEqual(withoutTrackQuality.suggestions);
    expect(
      withTrackQuality.suggestions.map(suggestion => suggestion.curatedAnalysisId)
    ).not.toContain(analysis.id);
  });

  it('registers a GPU recipe, descriptor, licensed feature sample and executable reusable template', () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    expect(getKeplerOperationDescriptor(binding.id)).toBe(CURATED_TRACK_QUALITY_DESCRIPTOR);
    expect(validateKeplerOperationDescriptor(CURATED_TRACK_QUALITY_DESCRIPTOR, binding)).toEqual(
      []
    );
    expect(validateKeplerCuratedAnalysis(analysis)).toEqual([]);
    expect(analysis.method).toContain('Turning angles and acceleration are not computed');
    const template = createCuratedWorkflowAnalysisTemplate(analysis);
    expect(parseWorkflowAnalysisTemplate(JSON.stringify(template))).toEqual(template);
    expect(validateWorkflowAnalysisTemplate(template)).toMatchObject({
      executable: true,
      issues: []
    });
    expect(template.exposed.map(exposed => exposed.target.parameter)).toEqual([
      'maximumSpeed',
      'maximumGap',
      'speedOutlierMode',
      'timestampEncoding'
    ]);
    const encoding = template.exposed.find(
      exposed => exposed.target.parameter === 'timestampEncoding'
    )!;
    expect(encoding.defaultPolicy).toEqual({from: 'time-encoding', input: 'tracks', role: 'time'});
    const malformed = structuredClone(template);
    const malformedEncoding = malformed.exposed.find(
      exposed => exposed.target.parameter === 'timestampEncoding'
    )!;
    malformedEncoding.defaultPolicy = {from: 'time-encoding', input: 'tracks', role: ''};
    expect(() => parseWorkflowAnalysisTemplate(JSON.stringify(malformed))).toThrow(/role/);
    malformedEncoding.defaultPolicy = {from: 'time-encoding', input: 'missing', role: 'time'};
    expect(validateWorkflowAnalysisTemplate(malformed).executable).toBe(false);
    const wrongTarget = structuredClone(template);
    wrongTarget.exposed[0].defaultPolicy = encoding.defaultPolicy;
    expect(
      validateWorkflowAnalysisTemplate(wrongTarget).issues.some(
        issue => issue.code === 'invalid-default-policy'
      )
    ).toBe(true);
    expect(template.reads.cautions.join(' ')).toContain('plausibility checks');
    expect(template.sample).toEqual(TRACK_QUALITY_SAMPLE_LICENSE);
    expect(getKeplerCuratedAnalysisSampleMetadata('track-quality')).toEqual(
      TRACK_QUALITY_SAMPLE_LICENSE
    );
    expect(createKeplerCuratedAnalysisSample('track-quality')).toEqual({
      tracks: createTrackQualitySample()
    });
    expect(getKeplerCuratedAnalysisSampleParameters('track-quality')).toEqual(parameters);
    expect(createTrackQualitySample()).toEqual(createTrackQualitySample());
    expect(createTrackQualitySample(101)).not.toEqual(createTrackQualitySample());
  });

  for (const renamed of [false, true]) {
    it(`maps trajectory roles and resolves encoding and units (renamed=${renamed})`, () => {
      const collection = createTrackQualitySample(renamed ? 101 : 0);
      if (renamed)
        for (const feature of collection.features)
          feature.properties = {
            vehicle: feature.properties!['track_id'],
            recorded_at: new Date(Number(feature.properties!['timestamp'])).toISOString()
          };
      const profile = profileTracks(collection, renamed);
      let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
      for (const [role, column] of [
        ['track', renamed ? 'vehicle' : 'track_id'],
        ['time', renamed ? 'recorded_at' : 'timestamp']
      ])
        mapping = updateKeplerCuratedAnalysisMapping(
          mapping,
          {slot: 'tracks', role, columns: [column]},
          analysis,
          [profile]
        );
      const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
        parameterOverrides: parameters
      });
      expect(request.inputDatasetIds).toEqual({tracks: profile.datasetId});
      const resolved = resolveKeplerOperationParameters(
        binding,
        CURATED_TRACK_QUALITY_DESCRIPTOR,
        {tracks: profile},
        {overrides: request.parameters}
      );
      expect(resolved.parameters['timestampEncoding']).toBe(
        renamed ? 'iso8601' : 'unix-millisecond'
      );
      const prepared = prepareTrackQuality(collection, resolved.parameters);
      expect(prepared.trajectory.trackIds).toEqual(['outlier', 'gap', 'singleton']);
      expect([...prepared.trajectory.trackOffsets]).toEqual([0, 5, 7, 8]);
      expect(prepared.trajectory.fixes.map(fix => fix.sourceIndex)).toEqual([
        2, 7, 4, 0, 6, 5, 1, 3
      ]);
      expect([...prepared.thresholds]).toEqual([...new Float32Array([0.05, 60000])]);
      for (let row = 0; row < 8; row++)
        expect(
          prepared.trajectory.timestamps[row * 2] +
            prepared.trajectory.timestamps[row * 2 + 1] * 2 ** 32
        ).toBe(prepared.trajectory.fixes[row].timestamp);
    });
  }

  it('publishes fixture metrics, independent flag bits, counts and source identities with Arrow readiness', () => {
    const {trajectory, settings} = prepareTrackQuality(createTrackQualitySample(), parameters);
    const result = publishTrackQuality(trajectory, settings, readback);
    const [fixes, summary] = result.tables;
    expect(fixes.coordinateSystem).toBe('planar');
    expect(prepareAnalysisResultPublication(fixes).analysisResult.destination).toBe('report');
    const geographic = publishTrackQuality(
      trajectory,
      {...settings, coordinateSystem: 'EPSG:4326'},
      readback
    ).tables[0];
    expect(geographic.coordinateSystem).toBe('EPSG:4326');
    expect(prepareAnalysisResultPublication(geographic).analysisResult.destination).toBe('map');

    expect(result.tables.map(table => table.id)).toEqual(
      binding.outputs.map(output => output.tableId)
    );
    expect(fixes.rowCount).toBe(8);
    expect(fixes.columns['source_index']).toEqual([2, 7, 4, 0, 6, 5, 1, 3]);
    expect(fixes.columns['vertex_index']).toEqual(Array(8).fill(0));
    expect(fixes.columns['quality_flags']).toEqual([0, 0, 4, 4, 0, 0, 8, 0]);
    expect(fixes.columns['excessive_speed']).toEqual([
      false,
      false,
      true,
      true,
      false,
      false,
      false,
      false
    ]);
    expect(fixes.columns['time_gap']).toEqual([
      false,
      false,
      false,
      false,
      false,
      false,
      true,
      false
    ]);
    expect(fixes.columns['time_gap_seconds']).toEqual([null, 1, 1, 1, 1, null, 120, null]);
    expect(fixes.columns['step_speed'][2]).toBeCloseTo(501, 4);
    expect(summary.columns['track_length']).toEqual([1002, 1, 0]);
    expect(summary.columns['duration_seconds']).toEqual([4, 120, 0]);
    expect(summary.columns['flagged_count']).toEqual([2, 1, 0]);
    expect(result.summary).toMatchObject({
      fixCount: 8,
      trackCount: 3,
      flaggedCount: 3,
      unknownCount: 0,
      speedUnit: 'coordinate units/second',
      durationUnit: 'seconds'
    });
    const arrowTable = makeArrowTableFromCatalogTable(fixes);
    expect(arrowTable.getChild('timestamp')!.get(0)).toBe(trajectory.fixes[0].timestamp);
    expect(arrowTable.getChild('quality_flags')!.get(2)).toBe(4);
    expect(
      previewArrowSource(
        arrowTable,
        {
          columns: [
            'source_index',
            'vertex_index',
            'fix_index',
            'track_id',
            'quality_flags',
            'flagged'
          ]
        },
        ['geometry']
      ).canUpload
    ).toBe(true);
    for (let row = 0; row < 8; row++)
      expect(JSON.parse(String(fixes.columns['geometry'][row])).coordinates).toEqual([
        ...trajectory.fixes[row].position,
        trajectory.fixes[row].altitude
      ]);
    for (const output of binding.outputs)
      for (const column of output.identityColumns)
        expect(
          result.tables.find(table => table.id === output.tableId)!.columns[column]
        ).toBeDefined();
  });

  it('keeps mixed-zone ISO profiling and preparation aligned at a one-second gap', () => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: ['2026-01-01T00:00:00', '2026-01-01T00:00:01Z'].map((timestamp, row) => ({
        geometry: {type: 'Point', coordinates: [row, 0]},
        properties: {track_id: 'one', timestamp}
      }))
    };
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'mixed-zone',
      label: 'Mixed-zone fixes',
      coordinateSystem: 'planar'
    });
    expect(profile.columns.find(column => column.name === 'timestamp')).toMatchObject({
      timestampEncoding: 'iso8601',
      minimum: Date.UTC(2026, 0, 1),
      maximum: Date.UTC(2026, 0, 1) + 1000
    });
    const resolved = resolveKeplerOperationParameters(
      binding,
      CURATED_TRACK_QUALITY_DESCRIPTOR,
      {tracks: profile},
      {fieldBindings: {trackProperty: 'track_id', timeProperty: 'timestamp'}, overrides: parameters}
    );
    expect(resolved.parameters['timestampEncoding']).toBe('iso8601');
    const {trajectory, settings} = prepareTrackQuality(collection, resolved.parameters);
    expect(trajectory.fixes.map(fix => fix.sourceIndex)).toEqual([0, 1]);
    expect(trajectory.fixes.map(fix => fix.timestamp)).toEqual([
      Date.UTC(2026, 0, 1),
      Date.UTC(2026, 0, 1) + 1000
    ]);
    const output = publishTrackQuality(trajectory, settings, {
      stepSpeeds: new Float32Array([0, 0.001]),
      trackLengths: new Float32Array([1]),
      trackDurations: new Float32Array([1000]),
      maximumSpeeds: new Float32Array([0.001]),
      flags: new Uint32Array([0, 0]),
      flaggedCounts: new Uint32Array([0])
    });
    expect(output.tables[0].columns['time_gap_seconds']).toEqual([null, 1]);
  });

  it('rejects impossible ISO dates and fractional milliseconds during curated preparation', () => {
    for (const timestamp of ['2026-02-30T00:00:00Z', '2026-01-01T00:00:00.0001Z']) {
      const collection: CatalogFeatureCollection = {
        type: 'FeatureCollection',
        features: [
          {geometry: {type: 'Point', coordinates: [0, 0]}, properties: {track_id: 'one', timestamp}}
        ]
      };
      expect(() =>
        prepareTrackQuality(collection, {...parameters, timestampEncoding: 'iso8601'})
      ).toThrow(/timestamps/);
    }
  });

  it('preserves unknown flags and nonfinite metrics as null with an error diagnostic', () => {
    const {trajectory, settings} = prepareTrackQuality(createTrackQualitySample(), parameters);
    const flags = readback.flags.slice();
    flags[2] = 0xffffffff;
    const speeds = readback.stepSpeeds.slice();
    speeds[2] = NaN;
    const result = publishTrackQuality(trajectory, settings, {
      ...readback,
      flags,
      stepSpeeds: speeds
    });
    expect(result.tables[0].columns['quality_flags'][2]).toBeNull();
    expect(result.tables[0].columns['flagged'][2]).toBeNull();
    expect(result.tables[0].columns['step_speed'][2]).toBeNull();
    expect(result.summary['unknownCount']).toBe(1);
    expect(result.diagnostics).toContainEqual(
      expect.objectContaining({code: 'track-quality-unknown', severity: 'error', count: 1})
    );
  });

  it('reuses timed-line preparation, retains pre-epoch words and skips null geometry without renumbering', () => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [
        {geometry: null, properties: {}},
        {
          id: 'line',
          geometry: {
            type: 'LineString',
            coordinates: [
              [0, 0, 4, -1],
              [3, 4, 9, 1]
            ]
          },
          properties: {}
        }
      ]
    };
    const {trajectory} = prepareTrackQuality(collection, parameters);
    expect(
      trajectory.fixes.map(fix => [fix.sourceIndex, fix.vertexIndex, fix.timestamp, fix.altitude])
    ).toEqual([
      [1, 0, -1, 4],
      [1, 1, 1, 9]
    ]);
    expect([...trajectory.timestamps]).toEqual([0xffffffff, 0xffffffff, 1, 0]);
    expect(trajectory.skippedEmptyFeatureCount).toBe(1);
  });

  it('uses a geographic metric frame while retaining source coordinates', () => {
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: [-87.65, -87.649].map((longitude, index) => ({
        geometry: {type: 'Point', coordinates: [longitude, 41.85]},
        properties: {track_id: 'one', timestamp: index * 1000}
      }))
    };
    const {trajectory, settings} = prepareTrackQuality(collection);
    expect(trajectory.frame.gpuFrame).not.toBeNull();
    const distance = Math.hypot(
      trajectory.positions[2] - trajectory.positions[0],
      trajectory.positions[3] - trajectory.positions[1]
    );
    expect(distance).toBeGreaterThan(80);
    expect(distance).toBeLessThan(90);
    expect(settings.coordinateSystem).toBe('EPSG:4326');
    expect(trajectory.fixes[0].position).toEqual([-87.65, 41.85]);
  });

  it('rejects invalid thresholds, encoding, duplicate times and fix/track capacities before allocation', async () => {
    const sample = createTrackQualitySample();
    for (const [name, value] of [
      ['maximumSpeed', -1],
      ['maximumSpeed', NaN],
      ['maximumGap', -1],
      ['maximumGap', Infinity],
      ['speedOutlierMode', 'other'],
      ['timestampEncoding', 'other'],
      ['coordinateSystem', 'other']
    ] as const)
      expect(() => prepareTrackQuality(sample, {...parameters, [name]: value})).toThrow(
        new RegExp(`curated-track-quality.*${name}`)
      );
    expect([
      ...prepareTrackQuality(sample, {...parameters, maximumSpeed: 0, maximumGap: 0}).thresholds
    ]).toEqual([0, 0]);
    expect(() =>
      prepareTrackQuality(
        {type: 'FeatureCollection', features: [sample.features[0], sample.features[0]]},
        parameters
      )
    ).toThrow(/strictly increasing/);
    expect(() =>
      prepareTrackQuality({type: 'FeatureCollection', features: []}, parameters)
    ).toThrow(/fixCount/);
    expect(() =>
      prepareTrackQuality(
        {type: 'FeatureCollection', features: Array(2049).fill(sample.features[0])},
        parameters
      )
    ).toThrow(/fixCount/);
    expect(() =>
      prepareTrackQuality(
        {
          type: 'FeatureCollection',
          features: Array.from({length: 129}, (_, index) => ({
            ...sample.features[0],
            properties: {track_id: String(index), timestamp: 0}
          }))
        },
        parameters
      )
    ).toThrow(/trackCount/);
    const invalid = structuredClone(sample);
    invalid.features[0].properties!['timestamp'] = 0.25;
    expect(() => prepareTrackQuality(invalid, parameters)).toThrow(/sub-millisecond/);
    await expect(
      runCatalogOperationAsync(
        binding,
        {...getCatalogDefaultParameters(binding), ...parameters},
        {tracks: sample}
      )
    ).rejects.toThrow(/GPU|gpu|device/);
  });
});
