// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {describe, expect, it} from 'vitest';
import {
  createSandboxSample,
  SANDBOX_SAMPLES,
  type SandboxSampleId
} from '../../../../examples/deck/kepler-compute/src/sandbox-samples';
import {getCatalogBinding, KEPLER_CATALOG_BINDINGS} from '../../src/analysis/catalog-bindings';
import {
  evaluateKeplerOperationApplicability,
  suggestKeplerAnalyses
} from '../../src/descriptors/applicability';
import {
  resolveKeplerOperationDescriptor,
  getKeplerOperationDescriptorCoverage
} from '../../src/descriptors/operation-descriptors';
import {profileKeplerColumnTable} from '../../src/profile/dataset-profile';
import {
  profileKeplerNativeDataset,
  type KeplerNativeDatasetShape
} from '../../src/profile/native-dataset-profile';
import {createKeplerDatasetProfileCache} from '../../src/profile/dataset-profile-cache';
import {
  readGallerySample,
  profileGallerySample,
  readGalleryFeatures
} from './gallery-sample-profiles';

const gallery = SANDBOX_SAMPLES.map(sample => ({
  ...sample,
  profile: profileGallerySample(sample.id)
}));
// Questions follow the sample's source meaning. The map is exhaustive: no sample can skip its gate.
const EXPECTED_TOP_THREE: Record<SandboxSampleId, string> = {
  'points-time': 'idw-surface',
  districts: 'excess-risk',
  connections: 'network-structure',
  'directed-network': 'network-centrality',
  'h3-cells': 'geometry-polygon-buffer',
  trips: 'trajectory-thinning',
  'trip-observations': 'trajectory-thinning',
  'numeric-grid': 'idw-surface'
};
function getKey(suggestion: {bindingId: string; curatedAnalysisId?: string}) {
  return suggestion.curatedAnalysisId ?? suggestion.bindingId;
}
function suggest(profiles: (typeof gallery)[number]['profile'][], maximumCount = 6) {
  return suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, profiles, {maximumCount});
}

describe('actual gallery qualification on the integrated descriptor registry', () => {
  it('uses all eight actual samples without losing rows, missingness, H3 identity or UTC types', () => {
    expect(gallery.map(sample => sample.id).sort()).toEqual(Object.keys(EXPECTED_TOP_THREE).sort());
    expect(gallery).toHaveLength(8);
    for (const sample of gallery) {
      const source = createSandboxSample(sample.id);
      expect(sample.profile.rowCount).toBe(source.table.numRows);
      for (const column of sample.profile.columns) {
        const original = source.table.getChild(column.name);
        if (original)
          expect(column.nullCount, `${sample.id}.${column.name}`).toBe(original.nullCount);
      }
      for (const field of source.table.schema.fields) {
        if (arrow.DataType.isTimestamp(field.type))
          expect(sample.profile.columns.find(column => column.name === field.name)?.type).toBe(
            'time'
          );
      }
    }
    expect(
      profileGallerySample('h3-cells').columns.find(column => column.name === 'h3_cell')
    ).toMatchObject({type: 'text', nullCount: 2});
    expect(profileGallerySample('trip-observations').geometryColumns).not.toContain('vehicle_id');
  });

  for (const sample of gallery) {
    it(`${sample.id}: expected analysis is in the top three`, () => {
      const result = suggest([sample.profile]);
      expect(result.suggestions.slice(0, 3).map(getKey)).toContain(EXPECTED_TOP_THREE[sample.id]);
      if (sample.id === 'points-time') {
        expect(getKey(result.suggestions[0])).toBe('idw-surface');
        expect(result.suggestions[0].status).toBe('ready');
        expect(result.suggestions[0].applicability.fieldBindings['valueProperty']).toBe('reading');
        expect(result.suggestions[0].applicability.warnings.join(' ')).toMatch(/reading/);
      }
      if (sample.id === 'districts')
        expect(
          result.suggestions.find(entry => getKey(entry) === 'excess-risk')!.applicability
            .fieldBindings
        ).toMatchObject({casesProperty: 'events', populationProperty: 'population'});
    });
    it(`${sample.id}: strict diversity at every supported budget and complete Browse all`, () => {
      let allCandidateIds: string[] | undefined;
      for (const maximumCount of [5, 6, 7, 8]) {
        const result = suggest([sample.profile], maximumCount);
        expect(result.suggestions.length).toBeLessThanOrEqual(maximumCount);
        const counts = new Map<string, number>();
        for (const suggestion of result.suggestions) {
          counts.set(suggestion.question, (counts.get(suggestion.question) ?? 0) + 1);
          expect(suggestion.status).not.toBe('blocked');
          expect(suggestion.reasons.length).toBeGreaterThan(0);
        }
        for (const [question, count] of counts)
          expect(count, question).toBeLessThanOrEqual(Math.floor(maximumCount / 3));
        const all = [...result.suggestions, ...Object.values(result.more).flat()];
        expect(new Set(all.map(entry => entry.id)).size).toBe(all.length);
        const candidateIds = all.map(entry => entry.id).sort();
        expect(candidateIds).toEqual(allCandidateIds ?? candidateIds);
        allCandidateIds = candidateIds;
        for (const [question, entries] of Object.entries(result.more)) {
          expect(entries.every(entry => entry.question === question)).toBe(true);
          expect(entries.map(entry => entry.score)).toEqual(
            entries.map(entry => entry.score).sort((left, right) => right - left)
          );
        }
      }
    });
    it(`${sample.id}: deterministic without cache, immutable inputs and actionable blocked remedies`, () => {
      const before = JSON.stringify(sample.profile);
      const first = suggest([sample.profile]);
      expect(
        suggestKeplerAnalyses([...KEPLER_CATALOG_BINDINGS], [profileGallerySample(sample.id)])
      ).toEqual(first);
      expect(JSON.stringify(sample.profile)).toBe(before);
      for (const blocked of first.blocked) {
        expect(['blocked', 'needs-mapping']).toContain(blocked.status);
        expect(blocked.applicability.missing).toHaveLength(1);
        expect(blocked.applicability.missing[0].what.length).toBeGreaterThan(0);
        expect(blocked.applicability.missing[0].remedy?.length).toBeGreaterThan(0);
      }
    });
  }

  it('returns fewer entries rather than exceeding one family and retains overflow after the row fills', () => {
    const bindings = ['curated-rate-cluster-map', 'curated-excess-risk'].map(
      bindingId => getCatalogBinding(bindingId)!
    );
    for (const maximumCount of [1, 5]) {
      const result = suggestKeplerAnalyses(bindings, [profileGallerySample('districts')], {
        maximumCount
      });
      expect(result.suggestions).toHaveLength(1);
      expect(Object.values(result.more).flat()).toHaveLength(1);
    }
  });

  it('uses exact H3 boundaries, scalar-grid meaning and validated native fix counts', () => {
    expect(profileGallerySample('h3-cells')).toMatchObject({
      kind: 'cells',
      catalogGeometry: 'polygon'
    });
    expect(
      readGalleryFeatures('h3-cells').features.filter(feature => feature.geometry)
    ).toHaveLength(4);
    expect(profileGallerySample('numeric-grid')).toMatchObject({
      kind: 'cells',
      scalarGrid: {width: 16, height: 16, valueColumn: 'value'},
      rowSemantics: {value: 'surface-samples', source: 'declared'}
    });
    for (const sampleId of ['trips', 'trip-observations'] as const) {
      const source = readGallerySample(sampleId);
      const unvalidated = profileKeplerColumnTable(source.table, source.options);
      expect(suggest([unvalidated]).suggestions.map(getKey)).not.toContain('trajectory-thinning');
      const profile = profileGallerySample(sampleId);
      expect(profile).toMatchObject({
        kind: 'trajectories',
        vertexCount: 48,
        trajectory: {fixCount: 48, trackCount: 8}
      });
      const suggestion = suggest([profile]).suggestions.find(
        entry => getKey(entry) === 'trajectory-thinning'
      )!;
      expect(suggestion.status, JSON.stringify(suggestion.applicability.missing)).toBe('ready');
      expect(suggestion.applicability.parameters['timestampEncoding']).toBe(
        sampleId === 'trips' ? 'unix-second' : 'unix-millisecond'
      );
      const binding = getCatalogBinding('trajectory-thinning')!;
      const output = binding.run!(
        {tracks: readGalleryFeatures(sampleId)},
        suggestion.applicability.parameters
      );
      expect(output.tables.find(table => table.id === 'trajectory-tracks')!.rowCount).toBe(8);
      expect(output.tables.find(table => table.id === 'trajectory-fixes')!.rowCount).toBe(48);
      // D5's complete family remains intact; only validated native suggestions are promoted.
      expect(resolveKeplerOperationDescriptor(binding).descriptor.prominence).toBe(3);
    }
    expect(getKeplerOperationDescriptorCoverage(KEPLER_CATALOG_BINDINGS).derivedBindingIds).toEqual(
      []
    );
  });

  it('retains native facts and honors coordinate corrections through the revision cache', () => {
    for (const sampleId of ['numeric-grid', 'h3-cells', 'trips', 'trip-observations'] as const) {
      const profile = profileGallerySample(sampleId);
      const cache = createKeplerDatasetProfileCache({maximumEntries: 8});
      cache.set(profile);
      for (const declarations of [{coordinateModel: 'planar' as const}, undefined]) {
        const updated = cache.setDeclarations(sampleId, declarations)!;
        expect(updated.kind).toBe(profile.kind);
        expect(updated.scalarGrid).toEqual(profile.scalarGrid);
        expect(updated.trajectory).toEqual(profile.trajectory);
        if (updated.trajectory) {
          const binding = getCatalogBinding('trajectory-thinning')!;
          const assessment = evaluateKeplerOperationApplicability(
            binding,
            resolveKeplerOperationDescriptor(binding).descriptor,
            [updated]
          );
          expect(assessment.parameters['coordinateSystem']).toBe(
            declarations ? 'planar' : 'EPSG:4326'
          );
        }
      }
    }
  });

  it('rejects malformed native declarations, duplicate times and excessive expansion', () => {
    const grid = readGallerySample('numeric-grid');
    expect(() =>
      profileKeplerNativeDataset(grid.table, grid.options, {
        kind: 'scalar-grid',
        width: 15,
        height: 16,
        valueColumn: 'value'
      })
    ).toThrow(/rowCount.*width/);
    expect(() => profileKeplerNativeDataset(grid.table, grid.options, {kind: 'cells'})).toThrow(
      /polygon boundaries/
    );
    const tracks = readGallerySample('trip-observations');
    expect(() =>
      profileKeplerNativeDataset(tracks.table, tracks.options, {
        kind: 'trajectories',
        timestampEncoding: 'unix-millisecond'
      })
    ).toThrow(/trackColumn and timeColumn/);
    const duplicateTimes = {
      ...tracks.table,
      columns: tracks.table.columns.map(column =>
        column.name === 'observed_at'
          ? {...column, values: Array(tracks.table.rowCount).fill(1767225600000)}
          : column
      )
    };
    expect(() => profileKeplerNativeDataset(duplicateTimes, tracks.options, tracks.shape!)).toThrow(
      /strictly increasing timestamps/
    );
    expect(() =>
      profileKeplerNativeDataset(tracks.table, tracks.options, {
        ...tracks.shape!,
        timestampEncoding: 'fortnight'
      } as unknown as KeplerNativeDatasetShape)
    ).toThrow(/timestampEncoding must be/);
    const lines = readGallerySample('trips');
    const oversized = {
      ...lines.table,
      columns: lines.table.columns.map(column =>
        column.name === 'geometry'
          ? {
              ...column,
              values: Array.from({length: lines.table.rowCount}, () =>
                JSON.stringify({
                  type: 'LineString',
                  coordinates: Array.from({length: 300}, (_, vertex) => [
                    -74,
                    40,
                    0,
                    1767225600 + vertex
                  ])
                })
              )
            }
          : column
      )
    };
    expect(() => profileKeplerNativeDataset(oversized, lines.options, lines.shape!)).toThrow(
      /vertexCount.*2048/
    );
    const binding = getCatalogBinding('trajectory-thinning')!;
    const multiple = evaluateKeplerOperationApplicability(
      binding,
      resolveKeplerOperationDescriptor(binding).descriptor,
      [profileGallerySample('trips'), profileGallerySample('trip-observations')]
    );
    expect(multiple.status).toBe('needs-mapping');
  });

  it('penalizes surface-sample counts without rewarding ordinary counts or penalizing mapped values', () => {
    const profile = profileGallerySample('points-time');
    const result = suggest([profile]);
    expect(getKey(result.suggestions[0])).toBe('idw-surface');
    expect(result.suggestions[0].status).toBe('ready');
    expect(result.suggestions[0].applicability.warnings.join(' ')).toMatch(/reading/);
    const count = [...result.suggestions, ...Object.values(result.more).flat()].find(
      entry => entry.bindingId === 'cell-aggregation'
    )!;
    expect(count.applicability.fieldBindings).toEqual({});
    expect(count.scoreTerms['rowSemanticsMismatch']).toBe(-35);
    const emerging = [...result.suggestions, ...Object.values(result.more).flat()].find(
      entry => getKey(entry) === 'emerging-hot-spots'
    )!;
    expect(emerging.scoreTerms['rowSemanticsMismatch']).toBe(-35);
    expect(emerging.scoreTerms['timeColumn']).toBe(0);
    expect(emerging.score).toBeLessThan(result.suggestions[0].score);
    expect(emerging.applicability.warnings.some(warning => /rows/.test(warning))).toBe(true);
    const events = {
      ...profile,
      rowSemantics: {value: 'events' as const, source: 'declared' as const}
    };
    expect(
      suggest([events]).suggestions.find(entry => entry.bindingId === 'cell-aggregation')!
        .scoreTerms['semanticsMatch']
    ).toBe(0);
    expect(
      suggest([events]).suggestions.find(entry => getKey(entry) === 'emerging-hot-spots')!
        .scoreTerms['timeColumn']
    ).toBe(12);
    const binding = getCatalogBinding('cell-aggregation')!;
    const mapped = evaluateKeplerOperationApplicability(
      binding,
      resolveKeplerOperationDescriptor(binding).descriptor,
      [profile],
      {fieldBindings: {valueProperty: 'reading'}}
    );
    expect(mapped.status).not.toBe('blocked');
    expect(mapped.requirements.some(requirement => requirement.id === 'row-semantics')).toBe(false);
  });

  it('pairs actual points and districts with stable input assignments independent of load order', () => {
    const profiles = [profileGallerySample('points-time'), profileGallerySample('districts')];
    const result = suggest(profiles);
    const count = result.suggestions.slice(0, 3).find(entry => getKey(entry) === 'count-by-area');
    expect(count).toBeDefined();
    expect(count!.applicability.inputAssignments).toEqual({
      points: 'points-time',
      polygons: 'districts'
    });
    expect(count!.scoreTerms['extraDatasets']).toBeGreaterThan(0);
    expect(suggestKeplerAnalyses([...KEPLER_CATALOG_BINDINGS], [...profiles].reverse())).toEqual(
      result
    );
  });
  it('pairs the network and sensor fixes without changing order on a fresh registry', () => {
    const profiles = [
      profileGallerySample('directed-network'),
      profileGallerySample('points-time')
    ];
    const result = suggest(profiles);
    const matching = result.suggestions.find(entry => getKey(entry) === 'network-map-matching');
    expect(matching).toBeDefined();
    expect(matching!.datasetIds).toEqual(['directed-network', 'points-time']);
    expect(suggestKeplerAnalyses([...KEPLER_CATALOG_BINDINGS], [...profiles].reverse())).toEqual(
      result
    );
  });
});
