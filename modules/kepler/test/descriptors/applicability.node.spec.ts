// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {writeFileSync} from 'node:fs';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {profileKeplerDatasetNetwork} from '../../src/profile/dataset-network-profile';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {GPUAnalysisColumnProfile} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {describe, expect, it} from 'vitest';
import type {CatalogOperationBinding} from '../../src/analysis/catalog-binding-types';
import {
  getCatalogBinding,
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters
} from '../../src/analysis/catalog-bindings';
import {
  evaluateKeplerOperationApplicability,
  getKeplerPrimaryQuestion,
  KEPLER_PREDICTOR_ROLE,
  KEPLER_RESPONSE_ROLE,
  isKeplerDatasetCompatibleWithInput,
  KEPLER_CURATED_ANALYSIS_ORDER,
  KEPLER_CURATED_BONUS,
  KEPLER_MAXIMUM_DATASETS_PER_INPUT,
  KEPLER_DATASET_KIND_INPUT_KINDS,
  listKeplerApplicableBindings,
  suggestKeplerAnalyses,
  type KeplerAnalysisSuggestion
} from '../../src/descriptors/applicability';
import {
  deriveKeplerOperationDescriptor,
  KEPLER_OPERATION_DESCRIPTORS,
  getKeplerOperationDescriptor,
  resolveKeplerOperationDescriptor
} from '../../src/descriptors/operation-descriptors';
import type {
  KeplerOperationDescriptor,
  KeplerOperationFieldDescriptor
} from '../../src/descriptors/operation-descriptor-types';
import {KEPLER_CURATED_ANALYSES} from '../../src/curated/curated-analyses';
import type {KeplerDatasetProfile} from '../../src/profile/dataset-profile-types';
import {
  createTestPointCollection,
  buildSampleProfile,
  buildSmallPointProfile,
  buildTableProfile,
  withRowCount
} from './engine-test-profiles';

import {profileGallerySample} from './gallery-sample-profiles';

function getKey(suggestion: KeplerAnalysisSuggestion): string {
  return suggestion.curatedAnalysisId ?? suggestion.bindingId;
}

function evaluate(
  bindingId: string,
  datasets: readonly KeplerDatasetProfile[],
  options?: Parameters<typeof evaluateKeplerOperationApplicability>[3]
) {
  const binding = getCatalogBinding(bindingId)!;
  return evaluateKeplerOperationApplicability(
    binding,
    resolveKeplerOperationDescriptor(binding).descriptor,
    datasets,
    options
  );
}

it('column-only validation discovery admits geometry-free numeric rows', () => {
  const profile = buildTableProfile(
    'forecasts',
    ['observed', 'predicted', 'variance', 'member1', 'member2', 'pvalue'].map(
      (name): GPUAnalysisColumnProfile => ({
        name,
        type: 'number',
        rowCount: 2,
        nullCount: 0,
        minimum: 0.01,
        maximum: 3,
        distinctCount: 2
      })
    ),
    2
  );
  for (const id of ['validation-gaussian-scores', 'validation-ensemble-scores', 'validation-fdr']) {
    const result = evaluate(id, [profile], {overrides: {}});
    expect(result.inputAssignments['points']).toBe('forecasts');
    expect(result.missing.map(item => item.what).join(' ')).not.toMatch(/geometry|point positions/);
    expect(result.outputKinds[id]).toBe('attributes');
  }
});

// ---------------------------------------------------------------------------------------------
// Suggestion fixtures. This table was written BEFORE the engine was run: each row is what a
// careful analyst would expect to see near the top for that situation, and the one thing that
// would be wrong to lead with. Keys are a curated analysis id or, for a bare operation, a binding
// id. `topThree`: every key must be among the first three suggestions. `topSix`: every key must be
// among the first six. `first`: the first suggestion must be this key. `notFirst`: the first
// suggestion must not be any of these. Where the engine disagreed with a row, the row was judged
// against the engine and the verdict is in the report, not in a silent edit of the table.
// ---------------------------------------------------------------------------------------------

type Situation = {
  name: string;
  datasets: () => KeplerDatasetProfile[];
  topThree?: string[];
  topSix?: string[];
  available?: string[];
  first?: string;
  notFirst?: string[];
  /** Judged separately by the tests below the table. */
  note: string;
};

const FIXTURES: Situation[] = [
  {
    name: 'incident points only',
    datasets: () => [buildSampleProfile('incidents', {keepColumns: ['severity']})],
    topThree: ['hot-spot-map', 'density-surface', 'clusters-and-footprints'],
    notFirst: ['knox-test'],
    note: 'events with one severity column: where are they, are the hot spots real, which groups'
  },
  {
    // Emerging hot spots now reads ISO times directly; it uses the event chronology,
    // while Knox still needs numeric epochs. Static clustering remains available below.
    name: 'incident points with an ISO timestamp',
    datasets: () => [buildSampleProfile('incidents')],
    first: 'emerging-hot-spots',
    topThree: ['emerging-hot-spots', 'density-surface', 'hot-spot-map'],
    note: 'timestamped events support emerging hot spots as well as static density and significance'
  },
  {
    name: 'incident points with an epoch-seconds time column',
    datasets: () => [
      buildSampleProfile('incidents', {
        columns: buildSampleProfile('incidents').columns.map(column =>
          column.name === 'observed'
            ? {...column, name: 'observed_epoch', type: 'integer' as const, integral: true}
            : column
        )
      })
    ],
    // Numeric event times also support the curated temporal analysis.
    topThree: ['emerging-hot-spots', 'density-surface', 'hot-spot-map'],
    // The curated temporal analysis displaces the bare Knox test from the six-slot list;
    // Knox must still be offered through Browse all for numeric event times.
    available: ['knox-test'],
    note: 'emerging hot spots uses chronology; numeric Knox remains available in Browse all'
  },
  {
    name: 'rain gauges with a measured value',
    datasets: () => [buildSampleProfile('gauges')],
    first: 'idw-surface',
    // Both interpolators lead: inverse distance is the plain answer, kriging with diagnostics the
    // careful one. The bare kriging operation is replaced by its curated analysis.
    topThree: ['idw-surface', 'kriging-with-diagnostics'],
    notFirst: ['hot-spot-map', 'hotspot-gi-star', 'density-surface'],
    note: 'rows are samples of a surface that exists everywhere; the question is the value between gauges. A hot spot map of the raw readings would map where the gauges are'
  },
  {
    name: 'districts with cases and population',
    datasets: () => [
      buildSampleProfile('districts', {keepColumns: ['district', 'cases', 'population']})
    ],
    // The smoothed-rate analyses were built for exactly this data. Smoothing alone is a step
    // toward them, so it is only expected within six.
    topThree: ['rate-cluster-map', 'excess-risk'],
    topSix: ['empirical-bayes-rates'],
    notFirst: ['hotspot-gi-star', 'cluster-and-outlier-map', 'local-moran-statistics'],
    note: 'two counts and no rate: clustering the smoothed rate is the analysis that fits; clustering of raw counts would map population'
  },
  {
    name: 'districts with one numeric rate column',
    datasets: () => [buildSampleProfile('districts', {keepColumns: ['district', 'rate']})],
    topThree: ['cluster-and-outlier-map'],
    topSix: ['hotspot-gi-star'],
    notFirst: ['geometry-validity'],
    note: 'a rate on areas: cluster and outlier map first, hot and cold spots as the other view'
  },
  {
    name: 'points plus districts',
    datasets: () => [
      buildSampleProfile('incidents', {keepColumns: ['severity']}),
      buildSampleProfile('districts', {keepColumns: ['district', 'rate']})
    ],
    topThree: ['count-by-area'],
    note: 'a pair rule: counting points by area uses both datasets and outranks the single-dataset analyses'
  },
  {
    name: 'a street network alone',
    datasets: () => [buildSampleProfile('streets')],
    topSix: ['network-structure'],
    note: 'nothing about origins can run; how the streets connect is the analysis that fits, and the service-area analyses are one dataset away'
  },
  {
    name: 'a street network with facilities and demand',
    datasets: () => [
      buildSampleProfile('streets'),
      buildSampleProfile('facilities'),
      buildSampleProfile('demand', {keepColumns: ['residents']})
    ],
    topThree: ['drive-time-catchments'],
    notFirst: ['density-surface', 'hot-spot-map'],
    note: 'facilities, demand and streets are exactly the inputs of travel-time access'
  },
  {
    name: 'a table with no geometry',
    datasets: () => [
      buildTableProfile(
        'table',
        [
          {name: 'name', type: 'category', rowCount: 100, nullCount: 0, distinctCount: 100},
          {name: 'amount', type: 'number', rowCount: 100, nullCount: 0, minimum: 1, maximum: 90}
        ],
        100
      )
    ],
    note: 'nothing spatial is ready; the blocked list says a layer or coordinates are needed'
  },
  {
    name: '15 points',
    datasets: () => [buildSmallPointProfile(15)],
    topThree: ['density-surface'],
    notFirst: ['hot-spot-map', 'hotspot-gi-star'],
    note: 'below 30 features significance statistics are unreliable; show density, not significance'
  },
  {
    name: 'a dataset over a binding capacity',
    datasets: () => [withRowCount(buildSampleProfile('incidents'), 250_000)],
    note: 'the hot spot lattice reads at most 100,000 points; it is blocked with the limit named'
  },
  {
    name: 'before and after incident points',
    datasets: () => [buildSampleProfile('before'), buildSampleProfile('after')],
    topThree: ['period-comparison'],
    note: 'two point sets of the same kind: comparing the periods is the analysis that uses both'
  }
];

/** Writes the table of the engine's actual top six per situation, for the report. */
function writeFixtureReport(): void {
  const lines: string[] = [];
  for (const fixture of FIXTURES) {
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, fixture.datasets());
    lines.push(`## ${fixture.name}`);
    for (const [index, suggestion] of result.suggestions.entries()) {
      lines.push(
        `${index + 1}. ${getKey(suggestion)} [${suggestion.status}] ${suggestion.score} ${JSON.stringify(suggestion.scoreTerms)}`
      );
    }
    lines.push(
      `total ${JSON.stringify(result.total)}; blocked: ${result.blocked.map(entry => `${getKey(entry)} (${entry.applicability.missing[0]?.what})`).join(' | ')}`
    );
  }
  if (process.env['WRITE_FIXTURE_REPORT']) {
    writeFileSync(process.env['WRITE_FIXTURE_REPORT'], lines.join('\n'));
  }
}

describe('suggestion fixtures', () => {
  it('does not let a time column outweigh a contradictory guessed row meaning', () => {
    const profile = profileGallerySample('points-time');
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [profile], {maximumCount: 12});
    const candidates = [...result.suggestions, ...Object.values(result.more).flat()];
    const temporal = candidates.find(entry => entry.curatedAnalysisId === 'emerging-hot-spots')!;
    const interpolation = candidates.find(entry => entry.curatedAnalysisId === 'idw-surface')!;
    expect(temporal.applicability.declaredContradiction).toBe(false);
    expect(temporal.scoreTerms['rowSemanticsMismatch']).toBeLessThan(0);
    expect(temporal.scoreTerms['timeColumn']).toBe(0);
    expect(
      temporal.score,
      JSON.stringify({
        meaning: profile.rowSemantics,
        temporal: temporal.scoreTerms,
        interpolation: interpolation.scoreTerms
      })
    ).toBeLessThan(interpolation.score);
    const events = {
      ...profile,
      rowSemantics: {value: 'events' as const, source: 'declared' as const}
    };
    const eventSuggestions = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [events], {
      maximumCount: 12
    });
    const eventTemporal = [
      ...eventSuggestions.suggestions,
      ...Object.values(eventSuggestions.more).flat()
    ].find(entry => entry.curatedAnalysisId === 'emerging-hot-spots')!;
    expect(eventTemporal.scoreTerms['timeColumn']).toBeGreaterThan(0);
  });

  it('ranks a matching two-period comparison above separate time-aware analyses of either period', () => {
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [
      buildSampleProfile('before'),
      buildSampleProfile('after')
    ]);
    const candidates = [...result.suggestions, ...Object.values(result.more).flat()];
    const relevant = candidates.filter(entry =>
      ['period-comparison', 'emerging-hot-spots'].includes(entry.curatedAnalysisId ?? '')
    );
    expect(
      result.suggestions[0]?.curatedAnalysisId,
      JSON.stringify(
        relevant.map(entry => ({id: entry.id, score: entry.score, terms: entry.scoreTerms}))
      )
    ).toBe('period-comparison');
    expect(result.suggestions[0].datasetIds.slice().sort()).toEqual(['after', 'before']);
    for (const candidate of relevant.filter(
      entry => entry.curatedAnalysisId === 'emerging-hot-spots'
    )) {
      expect(result.suggestions[0].score).toBeGreaterThan(candidate.score);
    }
  });

  for (const fixture of FIXTURES) {
    it(`situation: ${fixture.name}`, () => {
      const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, fixture.datasets());
      const keys = result.suggestions.map(getKey);
      for (const key of fixture.topThree ?? []) {
        expect(keys.slice(0, 3), `${key} in the top three of ${keys.join(', ')}`).toContain(key);
      }
      for (const key of fixture.topSix ?? []) {
        expect(keys.slice(0, 6), `${key} in the top six of ${keys.join(', ')}`).toContain(key);
      }
      for (const key of fixture.available ?? []) {
        expect([...keys, ...Object.values(result.more).flat().map(getKey)]).toContain(key);
      }
      if (fixture.first) {
        expect(keys[0]).toBe(fixture.first);
      }
      for (const key of fixture.notFirst ?? []) {
        expect(keys[0], `${key} must not lead`).not.toBe(key);
      }
    });
  }

  it('prints the report table when asked', () => {
    writeFixtureReport();
    expect(FIXTURES.length).toBeGreaterThanOrEqual(10);
  });

  it('shows the counts warning on a hot spot of counts, and does not rank it first', () => {
    const datasets = [
      buildSampleProfile('districts', {keepColumns: ['district', 'cases', 'population']})
    ];
    const applicability = evaluate('hotspot-gi-star', datasets);
    expect(applicability.measurementWarnings.join(' ')).toMatch(/count/);
    expect(applicability.status).toBe('ready');
    expect(applicability.warnings.join(' ')).toMatch(/count/);
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets, {maximumCount: 40});
    const keys = result.suggestions.map(getKey);
    expect(keys[0]).not.toBe('hotspot-gi-star');
    // The raw-count map, where it is offered at all, sits below the smoothed-rate analysis.
    for (const raw of ['cluster-and-outlier-map', 'hotspot-gi-star']) {
      if (keys.includes(raw)) {
        expect(keys.indexOf(raw), raw).toBeGreaterThan(keys.indexOf('rate-cluster-map'));
      }
    }
  });

  it('shows the fewer-than-30 warning for 15 points and none for the full sample', () => {
    const few = evaluate('recipe-hot-spot-lattice', [buildSmallPointProfile(15)]);
    expect(few.warnings.join(' ')).toMatch(/15 rows; at least 30 are needed/);
    const many = evaluate('recipe-hot-spot-lattice', [buildSampleProfile('incidents')]);
    expect(many.warnings.join(' ')).not.toMatch(/at least 30/);
  });

  it('blocks a dataset over capacity with the limit in the message, and keeps it out of the suggestions', () => {
    const datasets = [withRowCount(buildSampleProfile('incidents'), 250_000)];
    const applicability = evaluate('recipe-hot-spot-lattice', datasets);
    expect(applicability.status).toBe('blocked');
    expect(applicability.missing.map(entry => entry.what).join(' ')).toMatch(
      /250,000 rows; at most 100,000/
    );
    expect(applicability.missing.every(entry => entry.remedy)).toBe(true);
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets);
    expect(result.suggestions.map(getKey)).not.toContain('hot-spot-map');
    // The greyed-out rows name the capacity too: the layer is loaded, one limit stops it.
    expect(result.blocked.length).toBeGreaterThan(0);
    const capacityRows = result.blocked.filter(entry =>
      /at most/.test(entry.applicability.missing[0].what)
    );
    expect(capacityRows.length).toBeGreaterThan(0);
    for (const row of capacityRows) {
      expect(row.applicability.missing[0].what).toMatch(/250,000 rows; at most [\d,]+/);
    }
  });

  it('suggests nothing spatial for a table without geometry and says what is needed', () => {
    const table = FIXTURES.find(fixture => fixture.name === 'a table with no geometry')!;
    // Column-only scores now admit this representation; the separate positive fixtures pin
    // their ready state. This negative fixture concerns operations that require geometry.
    const result = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS.filter(binding => !binding.inputs.some(input => input.acceptsTable)),
      table.datasets()
    );
    expect(result.suggestions).toEqual([]);
    expect(result.total.ready).toBe(0);
    // Column-only scoring may need mappings or confirmation even without display geometry.
    expect(result.blocked.length).toBeGreaterThan(0);
    for (const entry of result.blocked) {
      expect(entry.applicability.missing.length).toBeGreaterThan(0);
      const typedInput = getCatalogBinding(entry.bindingId)!.inputs.some(
        input => input.dataKind === 'raster' || input.dataKind === 'table'
      );
      const remedies = entry.applicability.missing.map(missing => missing.remedy).join(' ');
      expect(remedies, entry.bindingId).toMatch(
        typedInput
          ? /raster|text or category column|polygon|zone/i
          : /coordinate columns or a geometry column/
      );
    }
  });

  it('lists service-area analyses as one dataset away once street topology is profiled', () => {
    const source = buildSampleProfile('streets');
    const initial = evaluate('network-service-area', [source]);
    expect(
      initial.requirements.find(requirement => requirement.id === 'maximum-network-edges')?.state
    ).toBe('unknown');
    const refined = {
      ...source,
      network: profileKeplerDatasetNetwork(
        source,
        createKeplerCuratedAnalysisSample('street-grid-facilities')['network'],
        initial.parameters
      )
    };
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [refined]);
    for (const bindingId of ['network-isochrone-polygons', 'network-service-area']) {
      expect(result.blocked.map(entry => entry.bindingId)).toContain(bindingId);
      const entry = {applicability: evaluate(bindingId, [refined])};
      expect(entry.applicability.status).toBe('blocked');
      expect(entry.applicability.missing[0].what).toMatch(/dataset of points/);
      expect(entry.applicability.missing[0].remedy).toMatch(/Load a layer of points/);
      expect(
        entry.applicability.requirements.find(
          requirement => requirement.id === 'maximum-network-edges'
        )?.state
      ).toBe('passed');
    }
  });
});

describe('suggestion properties', () => {
  const situations = FIXTURES.map(fixture => ({fixture, datasets: fixture.datasets()}));

  it('is deterministic: the same arguments give the same result, in the same order', () => {
    for (const {fixture, datasets} of situations) {
      const first = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets);
      const second = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, fixture.datasets());
      expect(JSON.stringify(second), fixture.name).toBe(JSON.stringify(first));
    }
  });

  it('strictly caps each question at a third of the requested budget', () => {
    for (const {fixture, datasets} of situations) {
      for (const maximumCount of [1, 3, 5, 6, 7, 8, 12]) {
        const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets, {maximumCount});
        expect(result.suggestions.length, fixture.name).toBeLessThanOrEqual(maximumCount);
        const counts = new Map<string, number>();
        for (const entry of result.suggestions)
          counts.set(entry.question, (counts.get(entry.question) ?? 0) + 1);
        for (const [question, count] of counts)
          expect(count, `${fixture.name}: ${question}`).toBeLessThanOrEqual(
            Math.max(1, Math.floor(maximumCount / 3))
          );
      }
    }
  });

  it('never offers a curated analysis and its own binding together', () => {
    for (const {fixture, datasets} of situations) {
      const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets, {maximumCount: 40});
      const all = [...result.suggestions, ...result.blocked];
      const curatedBindings = new Set(
        all.filter(entry => entry.curatedAnalysisId).map(entry => entry.bindingId)
      );
      const bare = all.filter(entry => !entry.curatedAnalysisId).map(entry => entry.bindingId);
      for (const id of bare) {
        expect(curatedBindings.has(id), `${fixture.name}: ${id}`).toBe(false);
      }
      expect(new Set(all.map(entry => entry.id)).size).toBe(all.length);
    }
  });

  it('gives every blocked entry exactly one missing item with a remedy', () => {
    for (const {fixture, datasets} of situations) {
      const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets);
      expect(result.blocked.length, fixture.name).toBeLessThanOrEqual(4);
      for (const entry of result.blocked) {
        // Blocked, or needs-mapping for want of one column that nothing plausible fills.
        expect(['blocked', 'needs-mapping']).toContain(entry.status);
        if (entry.status === 'needs-mapping') {
          expect(entry.applicability.hasEmptyRole).toBe(true);
        }
        expect(entry.applicability.missing, `${fixture.name} ${entry.id}`).toHaveLength(1);
        expect(entry.applicability.missing[0].remedy, `${fixture.name} ${entry.id}`).toBeTruthy();
      }
    }
  });

  it('sums the score terms to the score, with every term an integer', () => {
    for (const {fixture, datasets} of situations) {
      const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets);
      for (const entry of [...result.suggestions, ...result.blocked]) {
        const sum = Object.values(entry.scoreTerms).reduce((total, value) => total + value, 0);
        expect(sum, `${fixture.name} ${entry.id}`).toBe(entry.score);
        expect(Object.values(entry.scoreTerms).every(Number.isInteger)).toBe(true);
      }
    }
  });

  it('orders suggestions by score and never includes a blocked or prominence-3 entry', () => {
    for (const {fixture, datasets} of situations) {
      const {suggestions} = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets);
      for (let index = 1; index < suggestions.length; index++) {
        const previous = suggestions[index - 1];
        const current = suggestions[index];
        // Ties fall to prominence, the documented order and dataset ids, never to an alphabet.
        expect(previous.score >= current.score, fixture.name).toBe(true);
      }
      for (const entry of suggestions) {
        expect(entry.status).not.toBe('blocked');
        expect(
          resolveKeplerOperationDescriptor(getCatalogBinding(entry.bindingId)!).descriptor
            .prominence
        ).toBeLessThan(3);
      }
    }
  });

  it('puts a curated analysis above its bare operation by the curated bonus', () => {
    const datasets = [buildSampleProfile('incidents')];
    const curated = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets, {
      maximumCount: 40
    }).suggestions.find(entry => entry.curatedAnalysisId === 'density-surface')!;
    const bare = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, datasets, {
      maximumCount: 40,
      curated: []
    }).suggestions.find(entry => entry.bindingId === 'kernel-density-grid')!;
    expect(curated.score - bare.score).toBe(KEPLER_CURATED_BONUS);
    expect(curated.scoreTerms['curated']).toBe(KEPLER_CURATED_BONUS);
  });

  it('rejects a maximum count that is not a positive integer', () => {
    for (const maximumCount of [0, -1, 2.5, NaN]) {
      expect(() => suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [], {maximumCount})).toThrow(
        /suggestKeplerAnalyses maximumCount must be an integer of at least 1/
      );
    }
  });

  it('returns no suggestions and no blocked list for no datasets, with every analysis blocked', () => {
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, []);
    expect(result.suggestions).toEqual([]);
    expect(result.total.ready + result.total.needsInput).toBe(0);
    expect(result.total.blocked).toBeGreaterThan(100);
    for (const entry of result.blocked) {
      expect(entry.applicability.missing[0].remedy).toMatch(/^Load /);
    }
  });

  it('reports every candidate in the totals, which add up to the candidate count', () => {
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [
      buildSampleProfile('incidents')
    ]);
    const total = result.total.ready + result.total.needsInput + result.total.blocked;
    // One candidate per binding that reads a layer (fixture demonstrations with no input are not
    // candidates), plus one more for each curated analysis beyond the first that shares a binding
    // (the bare operation is replaced by its curated analyses).
    const candidateBindingCount = KEPLER_CATALOG_BINDINGS.filter(
      binding => binding.inputs.length > 0
    ).length;
    expect(candidateBindingCount).toBeGreaterThan(190);
    expect(total).toBeGreaterThanOrEqual(candidateBindingCount);
    expect(total).toBeLessThanOrEqual(candidateBindingCount + 30);
  });
});

describe('input compatibility', () => {
  const points = buildSampleProfile('incidents');
  const polygons = buildSampleProfile('districts');
  const lines = buildSampleProfile('streets');
  const table = buildTableProfile('t', [], 3);

  function canFill(dataset: KeplerDatasetProfile, bindingId: string, inputName: string): boolean {
    const binding = getCatalogBinding(bindingId)!;
    const descriptor = getKeplerOperationDescriptor(bindingId)!;
    const input = binding.inputs.find(candidate => candidate.name === inputName)!;
    const kind = descriptor.inputs.find(candidate => candidate.name === inputName)!.kind;
    return isKeplerDatasetCompatibleWithInput(dataset, input, kind);
  }

  it('follows the table: kind of the dataset against kind and accepted geometry of the input', () => {
    // [dataset, binding, input, expected]
    const cases: [KeplerDatasetProfile, string, string, boolean][] = [
      [points, 'kernel-density-grid', 'points', true],
      [polygons, 'kernel-density-grid', 'points', false],
      [lines, 'kernel-density-grid', 'points', false],
      [lines, 'network-centrality', 'lines', true],
      [points, 'network-centrality', 'lines', false],
      [polygons, 'network-centrality', 'lines', false],
      [polygons, 'recipe-points-in-polygons-choropleth', 'polygons', true],
      [points, 'recipe-points-in-polygons-choropleth', 'polygons', false],
      [points, 'recipe-points-in-polygons-choropleth', 'points', true],
      // Accepts point and polygon although its input kind is polygons.
      [points, 'local-moran-statistics', 'features', true],
      [polygons, 'local-moran-statistics', 'features', true],
      [lines, 'local-moran-statistics', 'features', false],
      // A network input reads line datasets; a regions input follows its accepted geometry.
      [lines, 'nearest-feature-join', 'targets', true],
      [points, 'nearest-feature-join', 'targets', true],
      [polygons, 'nearest-feature-join', 'targets', true],
      [table, 'kernel-density-grid', 'points', false],
      [table, 'local-moran-statistics', 'features', false],
      [table, 'nearest-feature-join', 'targets', false]
    ];
    for (const [dataset, bindingId, inputName, expected] of cases) {
      expect(
        canFill(dataset, bindingId, inputName),
        `${dataset.label} in ${bindingId}.${inputName}`
      ).toBe(expected);
    }
  });

  it('lists no input kind for a table, and treats point columns as point geometry', () => {
    expect(KEPLER_DATASET_KIND_INPUT_KINDS.attributes).toEqual([]);
    const withColumns: KeplerDatasetProfile = {
      ...table,
      pointColumns: {longitude: 'lon', latitude: 'lat'}
    };
    // The kind stays `attributes`, which fills nothing, so point columns alone are not enough: the
    // profile must say the dataset is points.
    expect(canFill(withColumns, 'kernel-density-grid', 'points')).toBe(false);
    expect(canFill({...withColumns, kind: 'points'}, 'kernel-density-grid', 'points')).toBe(true);
  });

  it('restricts an input to one geometry when asked', () => {
    const binding = getCatalogBinding('local-moran-statistics')!;
    const input = binding.inputs[0];
    expect(isKeplerDatasetCompatibleWithInput(points, input, 'polygons', 'polygon')).toBe(false);
    expect(isKeplerDatasetCompatibleWithInput(polygons, input, 'polygons', 'polygon')).toBe(true);
  });
});

describe('evaluateKeplerOperationApplicability', () => {
  it('does not use one dataset for two inputs: the second input is missing', () => {
    const one = evaluate('cell-compare', [buildSampleProfile('before')]);
    expect(one.status).toBe('blocked');
    expect(one.inputAssignments).toEqual({before: 'before', after: undefined});
    expect(one.missing).toHaveLength(1);
    expect(one.missing[0].what).toMatch(/second dataset of points for "/);
    const two = evaluate('cell-compare', [
      buildSampleProfile('before'),
      buildSampleProfile('after')
    ]);
    expect(two.inputAssignments).toEqual({before: 'before', after: 'after'});
    expect(two.status).not.toBe('blocked');
  });

  it('fills the inputs in the way that fills the most, not in loading order', () => {
    // polygons loaded first, points second: points-in-polygons still pairs each correctly.
    const result = evaluate('recipe-points-in-polygons-choropleth', [
      buildSampleProfile('districts'),
      buildSampleProfile('incidents')
    ]);
    expect(result.inputAssignments).toEqual({points: 'incidents', polygons: 'districts'});
  });

  it('warns when a guessed coordinate model contradicts a coordinate requirement', () => {
    const entry = KEPLER_OPERATION_DESCRIPTORS_WITH_GEOGRAPHIC_BLOCK();
    const binding = getCatalogBinding(entry.bindingId)!;
    const input = entry.requirements.find(
      requirement => requirement.id === 'geographic-coordinates' && requirement.severity === 'block'
    )!.input;
    const datasetFor = (source: 'guess' | 'declared') =>
      binding.inputs.map((candidate, index) => ({
        ...(candidate.name === input
          ? buildSampleProfile('incidents', {coordinates: {model: 'planar', source}})
          : buildSampleProfile('facilities')),
        datasetId: `d${index}`
      }));
    const guessed = evaluateKeplerOperationApplicability(binding, entry, datasetFor('guess'));
    const declared = evaluateKeplerOperationApplicability(binding, entry, datasetFor('declared'));
    expect(declared.status).toBe('blocked');
    expect(guessed.status).not.toBe('blocked');
    const confirmation = guessed.requirements.find(
      result => result.id === 'geographic-coordinates'
    )!;
    expect(confirmation.state).toBe('unknown');
    expect(confirmation.evidence).toBe('guess');
    expect(confirmation.remedy).toMatch(/^Confirm the coordinate system/);
    expect(
      guessed.warnings.some(item => /Confirm the coordinate system|confirm them/.test(item))
    ).toBe(true);
  });

  it('yields needs-confirmation when a blocking requirement cannot be settled', () => {
    const result = evaluate('recipe-hot-spot-lattice', [buildSampleProfile('incidents')], {
      unknownRowCount: true
    });
    expect(result.status).toBe('needs-confirmation');
    expect(result.requirements.filter(item => item.state === 'unknown').length).toBeGreaterThan(0);
    expect(result.missing.length).toBeGreaterThan(0);
  });

  it('warns when a proposed required column fits only by type', () => {
    const counts = buildSampleProfile('districts', {keepColumns: ['district', 'cases']});
    const result = evaluate('local-moran-statistics', [counts]);
    expect(result.status).toBe('ready');
    expect(result.fieldBindings['valueProperty']).toBe('cases');
    expect(result.measurementWarnings.join(' ')).toMatch(/cases is a count/);
    expect(result.warnings.join(' ')).toMatch(/Check the guessed column/);
  });

  it('fills a time role only from a column with name or type evidence for it', () => {
    const binding = getCatalogBinding('knox-test')!;
    const real = getKeplerOperationDescriptor('knox-test')!;
    // Only a number, whose sole qualification is its storage type: not proposed, asked for.
    const numbersOnly = evaluate('knox-test', [
      buildSampleProfile('incidents', {keepColumns: ['severity']})
    ]);
    expect(numbersOnly.status).toBe('needs-mapping');
    expect(numbersOnly.hasEmptyRole).toBe(true);
    expect(numbersOnly.fieldBindings['timeProperty']).toBeUndefined();
    expect(numbersOnly.roleCandidates['time']).toEqual(['severity']);
    expect(numbersOnly.missing[0].what).toMatch(/no column is named or typed like a time/);
    // Time only as ISO text, which the binding cannot read: blocked, with the way out.
    const isoText = evaluate('knox-test', [buildSampleProfile('incidents')]);
    expect(isoText.status).toBe('blocked');
    expect(isoText.missing.map(item => item.remedy).join(' | ')).toMatch(
      /Convert the time column to numbers \(for example epoch seconds\)/
    );
    expect(isoText.fieldBindings['timeProperty']).toBeUndefined();
    // A number named like a time is read.
    const epoch = buildSampleProfile('incidents');
    const named = evaluate('knox-test', [
      withColumns(epoch, [
        ...epoch.columns.filter(column => column.name === 'severity'),
        makeNumberColumn('event_epoch', epoch.rowCount, {type: 'integer', integral: true})
      ])
    ]);
    expect(named.fieldBindings['timeProperty']).toBe('event_epoch');
    // A descriptor that reads text timestamps takes the time-typed column over an integer.
    const timeRole = real.inputs[0].fields[0];
    const descriptor: KeplerOperationDescriptor = {
      ...real,
      inputs: [{...real.inputs[0], fields: [{...timeRole, accepts: ['time', 'integer', 'number']}]}]
    };
    const withTime = evaluateKeplerOperationApplicability(binding, descriptor, [
      buildSampleProfile('incidents')
    ]);
    expect(withTime.fieldBindings['timeProperty']).toBe('observed');
    expect(withTime.filledRoles.find(entry => entry.role === 'time')?.columnType).toBe('time');
  });

  it('offers the rate when a count is given to a rate method', () => {
    const counts = evaluate('local-moran-statistics', [
      buildSampleProfile('districts', {keepColumns: ['district', 'cases', 'population']})
    ]);
    expect(counts.warnings.join(' ')).toMatch(/divide cases by population/);
  });

  it('yields needs-mapping when several columns fit equally well, and lets a person settle it', () => {
    const two: KeplerDatasetProfile = {
      ...buildSampleProfile('districts', {keepColumns: ['district']}),
      columns: [
        {name: 'district', type: 'category', rowCount: 144, nullCount: 0, distinctCount: 144},
        {
          name: 'first_reading',
          type: 'number',
          rowCount: 144,
          nullCount: 0,
          minimum: 1,
          maximum: 9,
          measurement: 'rate',
          measurementSource: 'guess',
          measurementConfidence: 'high'
        },
        {
          name: 'second_reading',
          type: 'number',
          rowCount: 144,
          nullCount: 0,
          minimum: 2,
          maximum: 8,
          measurement: 'rate',
          measurementSource: 'guess',
          measurementConfidence: 'high'
        }
      ]
    };
    const unsettled = evaluate('local-moran-statistics', [two]);
    expect(unsettled.status).toBe('needs-mapping');
    expect(unsettled.missing[0].what).toMatch(/Several columns could be/);
    const settled = evaluate('local-moran-statistics', [two], {
      fieldBindings: {valueProperty: 'second_reading'}
    });
    expect(settled.fieldBindings['valueProperty']).toBe('second_reading');
    expect(settled.status).not.toBe('needs-mapping');
    expect(settled.parameters['valueProperty']).toBe('second_reading');
  });

  it('blocks when a column a person chose is not in the dataset', () => {
    const result = evaluate('local-moran-statistics', [buildSampleProfile('districts')], {
      fieldBindings: {valueProperty: 'no_such_column'}
    });
    expect(result.status).toBe('blocked');
    expect(result.missing[0].what).toMatch(/"no_such_column"/);
  });

  it('blocks when a required column has no candidate, naming the type of column', () => {
    const result = evaluate('local-moran-statistics', [
      buildSampleProfile('districts', {keepColumns: ['district']})
    ]);
    expect(result.status).toBe('blocked');
    expect(result.missing[0].what).toMatch(/numeric column for "/);
    expect(result.missing[0].remedy).toMatch(/Add a numeric column/);
  });

  it('adds the binding capacity when the descriptor states none', () => {
    const binding: CatalogOperationBinding = getCatalogBinding('kernel-density-grid')!;
    const derived = deriveKeplerOperationDescriptor(binding);
    const bare = {...derived, requirements: []};
    const big = withRowCount(
      buildSampleProfile('incidents'),
      binding.inputs[0].maximumFeatures + 1
    );
    const result = evaluateKeplerOperationApplicability(binding, bare, [big]);
    expect(result.status).toBe('blocked');
    expect(result.missing[0].what).toContain(
      binding.inputs[0].maximumFeatures.toLocaleString('en-US')
    );
    const fine = evaluateKeplerOperationApplicability(binding, bare, [
      withRowCount(buildSampleProfile('incidents'), binding.inputs[0].maximumFeatures)
    ]);
    expect(fine.status).not.toBe('blocked');
  });

  it('is ready with reasons and normalised parameters for a clear case', () => {
    const result = evaluate('kernel-density-grid', [buildSampleProfile('incidents')]);
    expect(['ready', 'needs-confirmation']).toContain(result.status);
    expect(result.reasons.join(' | ')).toMatch(/points/);
    expect(result.reasons.join(' | ')).toMatch(/500 rows|\d{3} rows/);
    expect(
      normalizeCatalogParameters(getCatalogBinding('kernel-density-grid')!, result.parameters)
    ).toEqual(result.parameters);
    expect(result.inputAssignments).toEqual({points: 'incidents'});
  });

  it('names a time column in the reasons when one is used', () => {
    const binding = getCatalogBinding('knox-test')!;
    const real = getKeplerOperationDescriptor('knox-test')!;
    const timeRole = real.inputs[0].fields[0];
    const result = evaluateKeplerOperationApplicability(
      binding,
      {
        ...real,
        inputs: [
          {...real.inputs[0], fields: [{...timeRole, accepts: ['time', 'integer', 'number']}]}
        ]
      },
      [buildSampleProfile('incidents')]
    );
    expect(result.fieldBindings['timeProperty']).toBe('observed');
    expect(result.reasons.join(' | ')).toMatch(/points with a time column/);
  });

  it('warns about guessed coordinates and about defaults that fell back', () => {
    const {
      spacing: _spacing,
      extent: _extent,
      ...withoutStatistics
    } = buildSampleProfile('incidents');
    const result = evaluate('kernel-density-grid', [withoutStatistics]);
    // A length that cannot be read from the data is not a constant to run with: a question.
    expect(result.status).toBe('needs-mapping');
    expect(result.missing.map(item => item.what).join(' | ')).toMatch(
      /could not be derived from this data/
    );
    expect(result.warnings.join(' | ')).toMatch(/guess/);
  });

  it('copes with a column name that cannot be written into a parameter', () => {
    const profile = buildSampleProfile('districts');
    const long = 'x'.repeat(80);
    const withLong: KeplerDatasetProfile = {
      ...profile,
      columns: [
        ...profile.columns.filter(column => column.name !== 'rate'),
        {name: long, type: 'number', rowCount: 144, nullCount: 0, minimum: 0, maximum: 9}
      ]
    };
    const result = evaluate('local-moran-statistics', [withLong]);
    expect(Object.values(result.fieldBindings)).not.toContain(long);
    expect(
      normalizeCatalogParameters(getCatalogBinding('local-moran-statistics')!, result.parameters)
    ).toEqual(result.parameters);
  });
});

describe('listKeplerApplicableBindings', () => {
  it('groups every non-blocked binding by primary question, in question order, and nothing else', () => {
    const datasets = [buildSampleProfile('incidents')];
    const groups = listKeplerApplicableBindings(KEPLER_CATALOG_BINDINGS, datasets);
    const flat = groups.flatMap(group => group.bindings);
    expect(flat.length).toBeGreaterThan(10);
    expect(flat.every(entry => entry.applicability.status !== 'blocked')).toBe(true);
    const expected = KEPLER_CATALOG_BINDINGS.filter(
      binding =>
        evaluateKeplerOperationApplicability(
          binding,
          resolveKeplerOperationDescriptor(binding).descriptor,
          datasets
        ).status !== 'blocked'
    ).map(binding => binding.id);
    expect(flat.map(entry => entry.bindingId).sort()).toEqual(expected.sort());
    for (const group of groups) {
      expect(group.label).toMatch(/\?$/);
      for (const entry of group.bindings) {
        const binding = getCatalogBinding(entry.bindingId)!;
        const {descriptor, source} = resolveKeplerOperationDescriptor(binding);
        expect(getKeplerPrimaryQuestion(descriptor)).toBe(group.question);
        expect(entry.derived).toBe(source === 'derived');
      }
    }
    expect(listKeplerApplicableBindings(KEPLER_CATALOG_BINDINGS, [])).toEqual([]);
  });

  it('includes bindings that have only a derived descriptor, marked as derived', () => {
    const synthetic: CatalogOperationBinding = {
      ...getCatalogBinding('kernel-density-grid')!,
      id: 'synthetic-density'
    };
    const groups = listKeplerApplicableBindings([synthetic], [buildSampleProfile('incidents')]);
    expect(groups.flatMap(group => group.bindings).map(entry => entry.derived)).toEqual([true]);
  });
});

function KEPLER_OPERATION_DESCRIPTORS_WITH_GEOGRAPHIC_BLOCK() {
  const found = KEPLER_CATALOG_BINDINGS.map(
    binding => resolveKeplerOperationDescriptor(binding).descriptor
  ).find(descriptor =>
    descriptor.requirements.some(
      requirement => requirement.id === 'geographic-coordinates' && requirement.severity === 'block'
    )
  );
  if (!found) {
    throw new Error('No descriptor has a blocking geographic-coordinates requirement');
  }
  return found;
}

// ---------------------------------------------------------------------------------------------
// Contract additions of the fix round, tested with synthetic descriptors on real bindings.
// ---------------------------------------------------------------------------------------------

function makeSyntheticDescriptor(
  bindingId: string,
  inputName: string,
  fields: KeplerOperationFieldDescriptor[],
  changes: Partial<KeplerOperationDescriptor> = {}
): KeplerOperationDescriptor {
  const binding = getCatalogBinding(bindingId)!;
  const base = deriveKeplerOperationDescriptor(binding);
  return {
    ...base,
    inputs: base.inputs.map(input => (input.name === inputName ? {...input, fields} : input)),
    ...changes
  };
}

function makeNumberColumn(
  name: string,
  rowCount: number,
  extra: Partial<GPUAnalysisColumnProfile> = {}
): GPUAnalysisColumnProfile {
  return {
    name,
    type: 'number',
    rowCount,
    nullCount: 0,
    minimum: 1,
    maximum: 9,
    mean: 5,
    standardDeviation: 2,
    ...extra
  };
}

function withColumns(
  profile: KeplerDatasetProfile,
  columns: GPUAnalysisColumnProfile[]
): KeplerDatasetProfile {
  return {...profile, columns};
}

const VALUE_FIELD: KeplerOperationFieldDescriptor = {
  role: 'value',
  label: 'Value',
  accepts: ['number', 'integer'],
  measurement: ['amount'],
  required: true,
  parameter: 'valueProperty'
};

describe('measurementByGeometry', () => {
  const binding = getCatalogBinding('local-moran-statistics')!;
  const descriptor = makeSyntheticDescriptor('local-moran-statistics', 'features', [
    {...VALUE_FIELD, measurementByGeometry: {polygon: ['rate']}}
  ]);
  const revenue = (profile: KeplerDatasetProfile) =>
    withColumns(profile, [
      makeNumberColumn('revenue', profile.rowCount, {
        measurement: 'amount',
        measurementSource: 'declared'
      })
    ]);

  it('replaces the measurement for the geometry of the assigned dataset', () => {
    const polygons = evaluateKeplerOperationApplicability(binding, descriptor, [
      revenue(buildSampleProfile('districts'))
    ]);
    // An amount per polygon mostly measures the size of the polygon: not a preferred column.
    expect(polygons.status).toBe('ready');
    expect(polygons.measurementWarnings.join(' ')).toMatch(
      /revenue is an amount; this method is meant for rates/
    );
    const points = evaluateKeplerOperationApplicability(binding, descriptor, [
      revenue(buildSampleProfile('incidents'))
    ]);
    expect(points.measurementWarnings).toEqual([]);
    expect(points.status).not.toBe('needs-mapping');
    const rate = evaluateKeplerOperationApplicability(binding, descriptor, [
      withColumns(buildSampleProfile('districts'), [
        makeNumberColumn('rate', 144, {measurement: 'rate', measurementSource: 'declared'}),
        makeNumberColumn('revenue', 144, {measurement: 'amount', measurementSource: 'declared'})
      ])
    ]);
    expect(rate.fieldBindings['valueProperty']).toBe('rate');
    expect(rate.measurementWarnings).toEqual([]);
  });
});

describe('notes', () => {
  it("passes a descriptor's notes to the applicability and to the suggestion", () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const notes = ['This form assumes planar distances.', 'The bandwidth is a support radius.'];
    const descriptor = {...getKeplerOperationDescriptor(binding.id)!, notes};
    const result = evaluateKeplerOperationApplicability(binding, descriptor, [
      buildSampleProfile('incidents')
    ]);
    expect(result.notes).toEqual(notes);
    expect(evaluate('kernel-density-grid', [buildSampleProfile('incidents')]).notes).toEqual(
      getKeplerOperationDescriptor('kernel-density-grid')!.notes ?? []
    );
    const suggestions = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [
      buildSampleProfile('incidents')
    ]);
    for (const entry of suggestions.suggestions) {
      expect(entry.notes).toEqual(entry.applicability.notes);
      expect(entry.notes).toEqual(getKeplerOperationDescriptor(entry.bindingId)!.notes ?? []);
    }
  });
});

describe('isolated points and inactive parameters', () => {
  const binding = getCatalogBinding('local-moran-statistics')!;
  const policy = (
    rule: 'outlier-resistant-nearest-neighbor-distance' | 'median-nearest-neighbor-distance'
  ) => ({
    from: 'data' as const,
    input: 'features',
    rule
  });
  const describeBand = (
    rule: 'outlier-resistant-nearest-neighbor-distance' | 'median-nearest-neighbor-distance',
    activeWhen?: {parameter: string; equals: readonly string[]}
  ) => {
    const base = makeSyntheticDescriptor('local-moran-statistics', 'features', [VALUE_FIELD]);
    return {
      ...base,
      parameters: base.parameters.map(parameter =>
        parameter.name === 'distanceBand'
          ? {...parameter, defaultPolicy: policy(rule), ...(activeWhen ? {activeWhen} : {})}
          : parameter
      )
    };
  };
  const districts = (isolated: number) => {
    const profile = buildSampleProfile('districts');
    return {
      ...profile,
      spacing: {
        ...profile.spacing!,
        nearestNeighbor: {...profile.spacing!.nearestNeighbor!, aboveUpperFenceCount: isolated}
      }
    };
  };

  it('warns with the count when the spacing statistics report isolated points', () => {
    const descriptor = describeBand('outlier-resistant-nearest-neighbor-distance');
    const warned = evaluateKeplerOperationApplicability(binding, descriptor, [districts(3)]);
    expect(warned.warnings.join(' | ')).toMatch(/3 of 144 points are isolated/);
    const resolution = warned.resolutions.find(entry => entry.parameter === 'distanceBand')!;
    // The rule's own explanation travels unchanged.
    expect(resolution.resolution?.explanation).toBeTruthy();
    const clean = evaluateKeplerOperationApplicability(binding, descriptor, [districts(0)]);
    expect(clean.warnings.join(' | ')).not.toMatch(/isolated/);
    const median = evaluateKeplerOperationApplicability(
      binding,
      describeBand('median-nearest-neighbor-distance'),
      [districts(3)]
    );
    expect(median.warnings.join(' | ')).not.toMatch(/isolated/);
  });

  it('raises no warning for the fallback of an inactive parameter and does not count it', () => {
    const {spacing: _spacing, extent: _extent, ...noSpacing} = districts(0);
    const activeWhen = {parameter: 'neighborhood', equals: ['distance-band']};
    const descriptor = describeBand('outlier-resistant-nearest-neighbor-distance', activeWhen);
    const neighborhood = binding.parameters.find(parameter => parameter.name === 'neighborhood')!;
    expect(neighborhood.defaultValue).not.toBe('distance-band');
    const inactive = evaluateKeplerOperationApplicability(binding, descriptor, [noSpacing]);
    const entry = inactive.resolutions.find(item => item.parameter === 'distanceBand')!;
    expect(entry.active).toBe(false);
    expect(entry.source).toBe('fallback');
    expect(inactive.missing.map(item => item.what).join(' | ')).not.toMatch(/could not be derived/);
    expect(inactive.warnings.join(' | ')).not.toMatch(/could not be derived/);
    const active = evaluateKeplerOperationApplicability(binding, descriptor, [noSpacing], {
      overrides: {neighborhood: 'distance-band'}
    });
    expect(active.resolutions.find(item => item.parameter === 'distanceBand')?.active).toBe(true);
    expect(active.status).toBe('needs-mapping');
    expect(active.missing.map(item => item.what).join(' | ')).toMatch(/could not be derived/);
  });
});

describe('circular fills, coordinate and identifier columns', () => {
  const REGRESSION_ID = 'regression-ols';
  const regressionBinding = getCatalogBinding(REGRESSION_ID)!;
  const regressionDescriptor = getKeplerOperationDescriptor(REGRESSION_ID)!;
  const evaluateRegression = (columns: GPUAnalysisColumnProfile[]) => {
    const profile = buildSampleProfile('incidents');
    return evaluateKeplerOperationApplicability(regressionBinding, regressionDescriptor, [
      withColumns(
        profile,
        columns.map(column => ({...column, rowCount: profile.rowCount}))
      )
    ]);
  };
  const count = (name: string) =>
    makeNumberColumn(name, 0, {
      type: 'integer',
      integral: true,
      measurement: 'count',
      measurementSource: 'declared',
      minimum: 10,
      // A rate pair needs the denominator to be larger over most of the range.
      maximum: name === 'population' ? 9000 : 900,
      mean: name === 'population' ? 4000 : 400
    });

  it('marks regression roles with the ids response and predictors, in every descriptor that has them', () => {
    const carriers = KEPLER_OPERATION_DESCRIPTORS.filter(descriptor =>
      descriptor.inputs.some(input =>
        input.fields.some(field => field.role === 'response' || field.role === 'predictors')
      )
    );
    expect(carriers.length).toBeGreaterThanOrEqual(4);
    for (const descriptor of carriers) {
      const roles = descriptor.inputs.flatMap(input => input.fields.map(field => field.role));
      // Predictor support compares training and query features without reading a response.
      if (descriptor.bindingId === 'validation-area-of-applicability') {
        expect(roles).toEqual(['predictors', 'predictors']);
        continue;
      }
      expect(roles, descriptor.bindingId).toContain('response');
      expect(roles, descriptor.bindingId).toContain('predictors');
      expect(descriptor.bindingId).toMatch(/regression/);
    }
    expect(KEPLER_RESPONSE_ROLE).toBe('response');
    expect(KEPLER_PREDICTOR_ROLE).toBe('predictors');
  });

  it("does not regress a rate's numerator on its denominator, and asks when nothing else is left", () => {
    const circular = evaluateRegression([count('cases'), count('population')]);
    expect(circular.status).toBe('needs-mapping');
    expect(circular.missing.map(item => item.what).join(' | ')).toMatch(/circular/);
    expect(circular.warnings.join(' | ')).toMatch(/circular/);
    const independent = evaluateRegression([
      count('cases'),
      count('population'),
      makeNumberColumn('income', 0, {measurement: 'amount', measurementSource: 'declared'})
    ]);
    const response = String(independent.fieldBindings['responseProperty']);
    const predictors = String(independent.fieldBindings['predictorProperties']).split(',');
    // The rate's two parts are never on opposite sides of the model.
    for (const [part, partner] of [
      ['cases', 'population'],
      ['population', 'cases']
    ]) {
      if (response === part) {
        expect(predictors).not.toContain(partner);
      }
    }
    expect(response === 'income' || predictors.includes('income')).toBe(true);
    expect(independent.missing.map(item => item.what).join(' | ')).not.toMatch(/circular/);
  });

  it('does not offer a coordinate column to a numeric role while another candidate exists', () => {
    const chosen = evaluateRegression([
      makeNumberColumn('lon', 0),
      makeNumberColumn('lat', 0),
      makeNumberColumn('rent', 0, {measurement: 'amount', measurementSource: 'declared'}),
      makeNumberColumn('income', 0, {measurement: 'amount', measurementSource: 'declared'})
    ]);
    const used = [
      String(chosen.fieldBindings['responseProperty']),
      ...String(chosen.fieldBindings['predictorProperties']).split(',')
    ];
    expect(used).not.toContain('lon');
    expect(used).not.toContain('lat');
    const only = evaluateRegression([
      makeNumberColumn('rent', 0, {measurement: 'amount', measurementSource: 'declared'}),
      makeNumberColumn('lon', 0)
    ]);
    expect(only.status).toBe('needs-mapping');
    expect(only.missing.map(item => item.what).join(' | ')).toMatch(/lon is a coordinate column/);
  });

  it('does not offer an identifier to a numeric role while another candidate exists', () => {
    const profile = buildSampleProfile('districts');
    const binding = getCatalogBinding('local-moran-statistics')!;
    const descriptor = makeSyntheticDescriptor('local-moran-statistics', 'features', [
      {...VALUE_FIELD, measurement: ['amount', 'rate']}
    ]);
    const id = makeNumberColumn('site_code', 144, {
      type: 'integer',
      measurement: 'identifier',
      measurementSource: 'declared'
    });
    const score = makeNumberColumn('score', 144, {
      measurement: 'amount',
      measurementSource: 'declared'
    });
    const both = evaluateKeplerOperationApplicability(binding, descriptor, [
      withColumns(profile, [id, score])
    ]);
    expect(both.fieldBindings['valueProperty']).toBe('score');
    const only = evaluateKeplerOperationApplicability(binding, descriptor, [
      withColumns(profile, [id])
    ]);
    expect(only.status).toBe('ready');
    expect(only.measurementWarnings.join(' ')).toMatch(/identifier/);
  });
});

// ---------------------------------------------------------------------------------------------
// Fix round, part 2.
// ---------------------------------------------------------------------------------------------

describe('mixed geometry and chosen columns', () => {
  it('blocks a layer that mixes geometry families the input does not read', () => {
    const profile: KeplerDatasetProfile = {
      ...buildSampleProfile('incidents'),
      mixedGeometry: {point: 500, line: 40, polygon: 0}
    };
    const result = evaluate('kernel-density-grid', [profile]);
    expect(result.status).toBe('blocked');
    expect(result.missing.map(item => item.what).join(' | ')).toMatch(
      /mixes geometry kinds: 40 lines/
    );
    expect(result.missing.map(item => item.remedy).join(' | ')).toMatch(
      /Split or filter the layer/
    );
    // An input that reads every family present is fine.
    expect(evaluate('geographic-distribution', [profile]).status).not.toBe('blocked');
    expect(
      evaluate('kernel-density-grid', [
        {...profile, mixedGeometry: {point: 540, line: 0, polygon: 0}}
      ]).status
    ).not.toBe('blocked');
  });

  it('checks a column the person chose against the role: type blocks, meaning stays visible', () => {
    const base = buildSampleProfile('gauges');
    const profile = withColumns(base, [
      makeNumberColumn('rainfall', base.rowCount, {
        measurement: 'amount',
        measurementSource: 'declared'
      }),
      {
        name: 'station_id',
        type: 'category',
        rowCount: base.rowCount,
        nullCount: 0,
        distinctCount: 400
      },
      makeNumberColumn('site_code', base.rowCount, {
        type: 'integer',
        measurement: 'identifier',
        measurementSource: 'declared'
      })
    ]);
    const text = evaluate('idw-grid', [profile], {fieldBindings: {valueProperty: 'station_id'}});
    expect(text.status).toBe('blocked');
    expect(text.missing[0].what).toMatch(
      /station_id holds category values, but this role reads numeric columns/
    );
    // The same through an override of the column parameter.
    const viaOverride = evaluate('idw-grid', [profile], {overrides: {valueProperty: 'station_id'}});
    expect(viaOverride.status).toBe('blocked');
    // An identifier is a number, so it is allowed: the warning stays, marked as the person's choice.
    const identifier = evaluate('idw-grid', [profile], {
      fieldBindings: {valueProperty: 'site_code'}
    });
    expect(identifier.status).not.toBe('blocked');
    expect(identifier.acknowledgedWarnings.join(' ')).toMatch(
      /identifier.*kept because you chose it/
    );
    expect(identifier.warnings).toEqual(
      expect.arrayContaining([...identifier.acknowledgedWarnings])
    );
    expect(identifier.measurementWarnings).toEqual([]);
    const fine = evaluate('idw-grid', [profile], {fieldBindings: {valueProperty: 'rainfall'}});
    expect(fine.acknowledgedWarnings).toEqual([]);
  });

  it('keeps a circular pair the person chose, with the warning', () => {
    const result = evaluate(
      'regression-ols',
      [
        withColumns(buildSampleProfile('incidents'), [
          makeNumberColumn('cases', 600, {
            type: 'integer',
            measurement: 'count',
            measurementSource: 'declared',
            mean: 400,
            maximum: 900
          }),
          makeNumberColumn('population', 600, {
            type: 'integer',
            measurement: 'count',
            measurementSource: 'declared',
            mean: 4000,
            maximum: 9000
          })
        ])
      ],
      {fieldBindings: {responseProperty: 'cases', predictorProperties: 'population'}}
    );
    expect(result.acknowledgedWarnings.join(' ')).toMatch(
      /two parts of a rate.*kept because you chose them/
    );
  });
});

describe('columns that are not filled unasked', () => {
  it('leaves an optional weight empty and lists the candidates', () => {
    const base = buildSampleProfile('incidents');
    const profile = withColumns(base, [
      makeNumberColumn('depth', base.rowCount, {
        measurement: 'amount',
        measurementSource: 'declared'
      }),
      makeNumberColumn('vessel_id', base.rowCount, {
        type: 'integer',
        measurement: 'identifier',
        measurementSource: 'declared'
      })
    ]);
    const result = evaluate('kernel-density-grid', [profile]);
    expect(result.fieldBindings['weightProperty']).toBeUndefined();
    expect(result.roleCandidates['weight']).toEqual(['depth']);
    expect(result.filledRoles.filter(entry => entry.role === 'weight')).toEqual([]);
    // The person's own choice is used.
    expect(
      evaluate('kernel-density-grid', [profile], {fieldBindings: {weightProperty: 'depth'}})
        .fieldBindings['weightProperty']
    ).toBe('depth');
  });

  it('warns without blocking when a required column is only a low-confidence guess', () => {
    const base = buildSampleProfile('gauges');
    const withLevel = (confidence: 'low' | 'high', source: 'guess' | 'declared') =>
      withColumns(base, [
        makeNumberColumn('reading', base.rowCount, {
          measurement: 'amount',
          measurementSource: source,
          measurementConfidence: confidence
        })
      ]);
    const low = evaluate('idw-grid', [withLevel('low', 'guess')]);
    expect(low.status).toBe('ready');
    expect(low.warnings.join(' ')).toMatch(
      /reading (looks like an amount \(a guess|is only a guess)/
    );
    expect(
      low.guessedFacts.some(fact => fact.kind === 'measurement' && fact.column === 'reading')
    ).toBe(true);
    expect(low.filledRoles.find(entry => entry.role === 'value')?.confident).toBe(false);
    expect(evaluate('idw-grid', [withLevel('high', 'guess')]).status).not.toBe('needs-mapping');
    expect(evaluate('idw-grid', [withLevel('low', 'declared')]).status).not.toBe('needs-mapping');
  });

  it('fills cases and population together from the best plausible pair, or not at all', () => {
    const district = (columns: GPUAnalysisColumnProfile[]) =>
      withColumns(
        buildSampleProfile('districts'),
        columns.map(column => ({...column, rowCount: 144}))
      );
    const count = (name: string, mean: number, maximum: number) =>
      makeNumberColumn(name, 144, {
        type: 'integer',
        measurement: 'count',
        measurementSource: 'declared',
        mean,
        maximum
      });
    const pair = evaluate('curated-rate-cluster-map', [
      district([count('population', 4000, 9000), count('cases', 400, 900)])
    ]);
    expect(pair.fieldBindings['casesProperty']).toBe('cases');
    expect(pair.fieldBindings['populationProperty']).toBe('population');
    // Repeated visits per resident can exceed one; only an explicitly named proportion requires
    // its numerator not to exceed the denominator.
    const none = evaluate('curated-rate-cluster-map', [
      district([count('residents', 100, 300), count('proportion_visits', 5000, 9000)])
    ]);
    expect(none.status).toBe('needs-mapping');
    expect(none.hasEmptyRole).toBe(true);
    expect(none.fieldBindings['casesProperty']).toBeUndefined();
    expect(none.fieldBindings['populationProperty']).toBeUndefined();
    expect(none.missing[0].remedy).toMatch(/Add a count of cases and a population column/);
  });

  it('does not take a one-way flag or a cost from a column named like neither', () => {
    const base = buildSampleProfile('streets');
    const profile = withColumns(base, [
      {name: 'highway', type: 'category', rowCount: base.rowCount, nullCount: 0, distinctCount: 6},
      makeNumberColumn('width_m', base.rowCount),
      makeNumberColumn('length_m', base.rowCount, {
        measurement: 'amount',
        measurementSource: 'declared'
      })
    ]);
    const result = evaluate('network-centrality', [profile]);
    expect(result.fieldBindings['onewayProperty']).toBeUndefined();
    // Length mode does not read a property-cost column.
    expect(result.fieldBindings['forwardCostProperty']).toBeUndefined();
    const named = evaluate('network-centrality', [
      withColumns(base, [
        {name: 'oneway', type: 'category', rowCount: base.rowCount, nullCount: 0, distinctCount: 3}
      ])
    ]);
    expect(named.fieldBindings['onewayProperty']).toBe('oneway');
  });
});

describe('row semantics, ties and questions', () => {
  it('warns for unknown mandatory row meaning but blocks a failed declared meaning', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const mandatory: KeplerOperationDescriptor = {
      ...descriptor,
      requirements: [
        ...descriptor.requirements.filter(requirement => requirement.id !== 'row-semantics'),
        {id: 'row-semantics', input: 'points', severity: 'block', expects: ['events']}
      ]
    };
    const profile = {...buildSampleProfile('incidents'), rowSemantics: undefined};
    const unknown = evaluateKeplerOperationApplicability(binding, mandatory, [profile]);
    expect(unknown.status).toBe('ready');
    expect(
      unknown.requirements.find(requirement => requirement.id === 'row-semantics')
    ).toMatchObject({state: 'unknown', severity: 'block'});
    expect(unknown.warnings.join(' ')).toMatch(/rows.*not known/);
    const failed = evaluateKeplerOperationApplicability(binding, mandatory, [
      {...profile, rowSemantics: {value: 'surface-samples', source: 'declared'}}
    ]);
    expect(failed.status).toBe('blocked');
    expect(failed.requirements.find(requirement => requirement.id === 'row-semantics')?.state).toBe(
      'failed'
    );
  });
  const guessedSemantics = (
    profile: KeplerDatasetProfile,
    value: 'events' | 'objects' | 'surface-samples',
    source: 'guess' | 'declared'
  ): KeplerDatasetProfile => ({
    ...profile,
    rowSemantics: {value, source, ...(source === 'guess' ? {confidence: 'low' as const} : {})}
  });

  it('keeps guessed row meaning answerable without blocking the analysis', () => {
    const profile = guessedSemantics(buildSampleProfile('gauges'), 'surface-samples', 'guess');
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [profile], {maximumCount: 12});
    const density = result.suggestions.find(
      entry => entry.curatedAnalysisId === 'density-surface'
    )!;
    expect(density.status).toBe('ready');
    expect(density.applicability.declaredContradiction).toBe(false);
    expect(density.scoreTerms['rowSemanticsMismatch']).toBe(-35);
    expect(density.applicability.warnings.join(' ')).toMatch(/rows|surface-samples/);
    const interpolation = result.suggestions.find(
      entry => entry.curatedAnalysisId === 'idw-surface'
    )!;
    expect(interpolation.scoreTerms['semanticsMatch']).toBe(4);
    const question = result.questions.find(entry => entry.id === 'row-semantics')!;
    expect(question.datasetId).toBe(profile.datasetId);
    expect(question.options).toEqual(['events', 'objects', 'surface-samples']);
    expect(question.affects).toContain(density.id);
    expect(question.affects).toContain(interpolation.id);
  });

  it('keeps an analysis that contradicts the DECLARED meaning out of the list, but under Browse all', () => {
    const profile = guessedSemantics(buildSampleProfile('gauges'), 'surface-samples', 'declared');
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [profile], {maximumCount: 40});
    expect(result.suggestions.map(getKey)).not.toContain('density-surface');
    const density = evaluate('kernel-density-grid', [profile]);
    expect(density.status).toBe('needs-confirmation');
    expect(density.declaredContradiction).toBe(true);
    expect(density.missing[0].what).toMatch(/You declared what these rows are/);
    const browse = listKeplerApplicableBindings(KEPLER_CATALOG_BINDINGS, [profile])
      .flatMap(group => group.bindings)
      .map(entry => entry.bindingId);
    expect(browse).toContain('kernel-density-grid');
  });

  it('breaks a tie by the documented order, never by an id, whatever the load order', () => {
    const result = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS,
      [buildSampleProfile('incidents')],
      {
        maximumCount: 12
      }
    );
    const density = result.suggestions.find(
      entry => entry.curatedAnalysisId === 'density-surface'
    )!;
    const hotSpot = result.suggestions.find(entry => entry.curatedAnalysisId === 'hot-spot-map')!;
    expect(density.score).toBe(hotSpot.score);
    // hot-spot-map sorts before density-surface alphabetically; the documented order decides.
    expect(KEPLER_CURATED_ANALYSIS_ORDER['density-surface']).toBeLessThan(
      KEPLER_CURATED_ANALYSIS_ORDER['hot-spot-map']
    );
    expect(result.suggestions.indexOf(density)).toBeLessThan(result.suggestions.indexOf(hotSpot));
    for (const analysis of KEPLER_CURATED_ANALYSES) {
      expect(KEPLER_CURATED_ANALYSIS_ORDER[analysis.id], analysis.id).toBeGreaterThanOrEqual(0);
    }
    const a = buildSampleProfile('incidents', {});
    const b = {...buildSampleProfile('before'), datasetId: 'zz-before', label: 'zz-before'};
    const forward = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [a, b]);
    const backward = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [b, a]);
    expect(JSON.stringify(backward)).toBe(JSON.stringify(forward));
  });

  it('offers paired change before a repeat temporal analysis, retaining repeats in Browse all', () => {
    const profiles = [buildSampleProfile('before'), buildSampleProfile('after')];
    for (const maximumCount of [6, 7, 8]) {
      const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, profiles, {maximumCount});
      expect(result.suggestions.slice(0, 3).map(getKey)).toContain('period-comparison');
      expect(
        result.suggestions.filter(entry => getKey(entry) === 'emerging-hot-spots')
      ).toHaveLength(1);
      const all = [...result.suggestions, ...Object.values(result.more).flat()];
      expect(all.filter(entry => getKey(entry) === 'emerging-hot-spots')).toHaveLength(2);
      expect(new Set(all.map(entry => entry.id)).size).toBe(all.length);
      expect(
        suggestKeplerAnalyses([...KEPLER_CATALOG_BINDINGS], [...profiles].reverse(), {maximumCount})
      ).toEqual(result);
    }
    // Repeats can fill spare slots when no distinct analysis is available; the family cap holds.
    const result = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS.filter(binding => binding.id === 'kernel-density-grid'),
      profiles
    );
    expect(result.suggestions).toHaveLength(2);
    expect(result.suggestions.every(entry => getKey(entry) === 'density-surface')).toBe(true);
  });

  it('offers each dataset id even when size, extent and schema match', () => {
    const first = buildSampleProfile('incidents');
    const second = {
      ...buildSampleProfile('before'),
      datasetId: 'second-layer',
      label: 'second layer'
    };
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [first, second], {
      maximumCount: 40
    });
    // Distinct analyses lead; the per-layer alternatives remain offered through Browse all.
    const density = [...result.suggestions, ...Object.values(result.more).flat()].filter(
      entry => entry.curatedAnalysisId === 'density-surface'
    );
    expect(density.map(entry => entry.datasetIds[0]).sort()).toEqual(['incidents', 'second-layer']);
    expect(density.map(entry => entry.id).sort()).toEqual([
      'curated:density-surface@incidents',
      'curated:density-surface@second-layer'
    ]);
    const twin = {...first, datasetId: 'incidents-copy', label: 'copy'};
    const twins = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [first, twin], {maximumCount: 40});
    expect(
      [...twins.suggestions, ...Object.values(twins.more).flat()].filter(
        entry => entry.curatedAnalysisId === 'density-surface'
      )
    ).toHaveLength(2);
    expect(twins.dropped.join(' ')).not.toMatch(/holds the same rows as/);
  });
});

describe('layers that do not belong together', () => {
  const evaluateOverlap = (profiles: readonly KeplerDatasetProfile[]) => {
    const binding = getCatalogBinding('cell-compare')!;
    return evaluateKeplerOperationApplicability(
      binding,
      {...getKeplerOperationDescriptor(binding.id)!, overlappingInputs: [['before', 'after']]},
      profiles
    );
  };
  const moved = (
    profile: KeplerDatasetProfile,
    id: string,
    offset: number
  ): KeplerDatasetProfile => ({
    ...profile,
    datasetId: id,
    label: id,
    extent: profile.extent
      ? [
          profile.extent[0] + offset,
          profile.extent[1],
          profile.extent[2] + offset,
          profile.extent[3]
        ]
      : profile.extent
  });

  it('blocks a pairing of layers that do not overlap, and of a layer with itself as two states', () => {
    const near = buildSampleProfile('before');
    const far = moved(buildSampleProfile('after'), 'after-far', 40);
    const apart = evaluateOverlap([near, far]);
    expect(apart.status).toBe('blocked');
    expect(apart.missing[0].what).toMatch(/do not overlap in space/);
    const twin = {...near, label: 'before again'};
    const same = evaluateOverlap([near, twin]);
    expect(same.status).toBe('blocked');
    expect(same.missing.map(entry => entry.what).join(' ')).toMatch(/dataset of points/);
    const fine = evaluateOverlap([near, buildSampleProfile('after')]);
    expect(fine.status).not.toBe('blocked');
  });

  it('judges overlap across longitude 180 with the antimeridian convention', () => {
    const base = buildSampleProfile('before');
    const across = (id: string, west: number, east: number): KeplerDatasetProfile => ({
      ...base,
      datasetId: id,
      label: id,
      extent: [west, 60, east, 61]
    });
    const result = evaluateOverlap([
      across('west-side', 179.5, -179.5),
      across('east-side', -179.9, -179)
    ]);
    expect(result.missing.map(item => item.what).join(' ')).not.toMatch(/do not overlap/);
    const apart = evaluateOverlap([across('a', 179.5, -179.5), across('b', 100, 101)]);
    expect(apart.missing.map(item => item.what).join(' ')).toMatch(/do not overlap/);
  });

  it('bounds the assignment search and says what was not tried', () => {
    const many = Array.from({length: 40}, (_, index) => ({
      ...buildSampleProfile('facilities'),
      datasetId: `layer-${String(index).padStart(2, '0')}`,
      label: `layer ${index}`
    }));
    const started = performance.now();
    const result = evaluate('location-allocation', many);
    expect(performance.now() - started).toBeLessThan(1500);
    expect(result.warnings.join(' | ')).toMatch(/more loaded layers fit .* and were not tried/);
    expect(KEPLER_MAXIMUM_DATASETS_PER_INPUT).toBe(6);
  });
});

describe('the one-step list and the slot rule', () => {
  it('lists an analysis that lacks one column, with the remedy naming the kind of column', () => {
    const profile = buildSampleProfile('districts', {keepColumns: ['district', 'cases']});
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [profile]);
    const row = result.blocked.find(entry => entry.curatedAnalysisId === 'rate-cluster-map');
    expect(row).toBeDefined();
    expect(row!.status).toBe('needs-mapping');
    expect(row!.applicability.missing).toHaveLength(1);
    expect(row!.applicability.missing[0].remedy).toMatch(
      /Add a count of cases and a population column/
    );
    // It has no proposal, so it is not offered as a suggestion.
    expect(result.suggestions.map(entry => entry.curatedAnalysisId)).not.toContain(
      'rate-cluster-map'
    );
    // The list is about the data that is loaded: every row already uses it.
    for (const entry of result.blocked) {
      expect(entry.datasetIds.length).toBeGreaterThan(0);
    }
  });

  it("returns what a question's share left out in `more`, best first", () => {
    const result = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS,
      [buildSampleProfile('incidents')],
      {
        maximumCount: 3
      }
    );
    for (const [question, held] of Object.entries(result.more)) {
      const shown = result.suggestions.map(entry => entry.id);
      for (const entry of held) {
        expect(shown, question).not.toContain(entry.id);
        expect(entry.question).toBe(question);
      }
      expect(held.map(entry => entry.score)).toEqual(
        [...held.map(entry => entry.score)].sort((x, y) => y - x)
      );
    }
  });
});

function isRequiredField(field: KeplerOperationFieldDescriptor): boolean {
  return field.required === true;
}

describe('coordinates, engine and cost', () => {
  const lowGuess = (profile: KeplerDatasetProfile): KeplerDatasetProfile => ({
    ...profile,
    coordinates: {
      model: 'planar',
      source: 'guess',
      confidence: 'low',
      reason: 'small extent near the origin'
    }
  });

  it('warns about a low-confidence coordinate guess without blocking metric analysis', () => {
    // Keep the guessed model and the spacing calculation consistent; a model change needs
    // re-profiling and is covered separately below.
    const profile = lowGuess(
      buildSampleProfile('incidents', {
        coordinates: {model: 'planar', source: 'guess'}
      })
    );
    const metric = evaluate('kernel-density-grid', [profile]);
    expect(metric.status).toBe('ready');
    expect(metric.warnings.join(' ')).toMatch(/coordinate.*guess|guess.*coordinate/);
    expect(metric.resolutions.find(entry => entry.parameter === 'coordinateSystem')?.guessed).toBe(
      true
    );
    expect(metric.guessedFacts.some(fact => fact.kind === 'coordinate-model')).toBe(true);
    // An analysis that measures nothing in the coordinates has nothing to confirm.
    const quiet = KEPLER_CATALOG_BINDINGS.find(binding => {
      const descriptor = resolveKeplerOperationDescriptor(binding).descriptor;
      return (
        binding.inputs.length === 1 &&
        binding.inputs[0].accepts.includes('point') &&
        !descriptor.requirements.some(requirement => requirement.id.endsWith('coordinates')) &&
        !descriptor.parameters.some(
          parameter =>
            parameter.defaultPolicy.from === 'data' &&
            binding.parameters.find(spec => spec.name === parameter.name)?.unit !== undefined
        ) &&
        descriptor.inputs[0].fields.every(field => !isRequiredField(field))
      );
    });
    expect(
      quiet,
      'a point operation with no coordinate requirement and no length default'
    ).toBeDefined();
    expect(evaluate(quiet!.id, [profile]).missing.map(item => item.remedy)).not.toContain(
      'Confirm the coordinate system'
    );
    const high = evaluate('kernel-density-grid', [
      {...profile, coordinates: {...profile.coordinates!, confidence: 'high'}}
    ]);
    expect(high.missing.map(item => item.remedy)).not.toContain('Confirm the coordinate system');
  });

  it('blocks a declaration that the extent contradicts', () => {
    const profile: KeplerDatasetProfile = {
      ...buildSampleProfile('incidents'),
      coordinates: {model: 'geographic', source: 'declared'},
      extent: [500000, 4000000, 510000, 4010000]
    };
    const result = evaluate('kernel-density-grid', [profile]);
    expect(result.status).toBe('blocked');
    expect(result.missing.map(item => item.what).join(' ')).toMatch(
      /declared as longitude and latitude, but its extent/
    );
  });

  it('blocks a GPU-only operation on the CPU and a pairwise analysis beyond the CPU row limit', () => {
    const gpuOnly = KEPLER_CATALOG_BINDINGS.find(binding => binding.engine === 'gpu')!;
    const result = evaluateKeplerOperationApplicability(
      gpuOnly,
      deriveKeplerOperationDescriptor(gpuOnly),
      [buildSampleProfile('incidents'), buildSampleProfile('districts')],
      {executionEngine: 'cpu'}
    );
    expect(result.status).toBe('blocked');
    expect(result.missing.map(item => item.what).join(' ')).toMatch(/needs a GPU device/);
    const big = withRowCount(buildSampleProfile('incidents'), 8000);
    const onCpu = evaluate('density-clustering', [big], {executionEngine: 'cpu'});
    expect(onCpu.status).not.toBe('blocked');
    expect(
      evaluate('density-clustering', [withRowCount(big, 20000)], {executionEngine: 'cpu'}).status
    ).not.toBe('blocked');
    expect(
      evaluate('density-clustering', [withRowCount(big, 20001)], {executionEngine: 'cpu'}).status
    ).toBe('blocked');
    expect(
      evaluate('density-clustering', [big])
        .missing.map(item => item.what)
        .join(' ')
    ).not.toMatch(/CPU engine/);
    // Contiguity weights need no pair work.
    expect(
      evaluate('local-moran-statistics', [withRowCount(buildSampleProfile('districts'), 8000)], {
        executionEngine: 'cpu',
        overrides: {neighborhood: 'queen'}
      })
        .missing.map(item => item.what)
        .join(' ')
    ).not.toMatch(/CPU engine/);
    const wide = evaluate(
      'local-moran-statistics',
      [withRowCount(buildSampleProfile('districts'), 8000)],
      {
        executionEngine: 'cpu',
        overrides: {neighborhood: 'knn'}
      }
    );
    expect(wide.missing.map(item => item.what).join(' ')).toMatch(
      /at most 6,324 on the CPU engine/
    );
  });

  it('never suggests an operation with no inputs, and remembers an evaluation per profile object', () => {
    const noInputs: CatalogOperationBinding = {
      ...getCatalogBinding('kernel-density-grid')!,
      id: 'synthetic-no-input',
      inputs: []
    };
    const descriptor = deriveKeplerOperationDescriptor(noInputs);
    const result = evaluateKeplerOperationApplicability(noInputs, descriptor, [
      buildSampleProfile('incidents')
    ]);
    expect(result.status).toBe('blocked');
    expect(result.missing[0].what).toMatch(/reads no layer/);
    const suggested = suggestKeplerAnalyses([noInputs], [buildSampleProfile('incidents')]);
    expect(suggested.suggestions).toEqual([]);
    expect(suggested.blocked).toEqual([]);
    // Memoised: the same arguments give the same object; another option or profile does not.
    const profile = buildSampleProfile('incidents');
    const first = evaluate('kernel-density-grid', [profile]);
    expect(evaluate('kernel-density-grid', [profile])).toBe(first);
    expect(evaluate('kernel-density-grid', [profile], {executionEngine: 'cpu'})).not.toBe(first);
    expect(evaluate('kernel-density-grid', [{...profile}])).not.toBe(first);
    // Nested options are part of the key: two different chosen columns are two evaluations.
    const gauges = buildSampleProfile('gauges');
    expect(evaluate('idw-grid', [gauges], {fieldBindings: {valueProperty: 'a'}})).not.toBe(
      evaluate('idw-grid', [gauges], {fieldBindings: {valueProperty: 'b'}})
    );
    expect(evaluate('idw-grid', [gauges], {overrides: {cellSize: 1, padding: 2}})).toBe(
      evaluate('idw-grid', [gauges], {overrides: {padding: 2, cellSize: 1}})
    );
  });
});

describe('second review engine regressions', () => {
  const positions: [number, number][] = [
    [1000, 1000],
    [1100, 1000],
    [1000, 1100],
    [1100, 1100]
  ];
  const makeProfile = (
    datasetId: string,
    properties: (index: number) => Record<string, unknown> = () => ({})
  ) =>
    profileKeplerFeatureCollection(createTestPointCollection(positions, properties), {
      datasetId,
      label: datasetId,
      coordinateSystem: 'planar',
      declarations: {
        rowSemantics: 'objects',
        measurement: Object.fromEntries(
          Object.keys(properties(0)).map(name => [name, 'amount' as const])
        )
      }
    });

  it('requires a compatible common column for roles feeding one value parameter', () => {
    const before = makeProfile('before', index => ({amount: 5 + index}));
    const after = makeProfile('after', index => ({total: 15 + index}));
    const missing = evaluate('cell-compare', [before, after], {overrides: {statistic: 'sum'}});
    expect(missing.status).toBe('needs-mapping');
    expect(missing.missing.map(entry => entry.what).join(' ')).toMatch(
      /common column.*valueProperty/
    );
    expect(missing.fieldBindings['valueProperty']).toBeUndefined();
    const shared = evaluate(
      'cell-compare',
      [
        makeProfile('before', index => ({amount: index, reading: 5 + index})),
        makeProfile('after', index => ({total: index, reading: 15 + index}))
      ],
      {overrides: {statistic: 'sum'}}
    );
    expect(shared.fieldBindings['valueProperty']).toBe('reading');
    expect(shared.status, JSON.stringify(shared.missing)).toBe('ready');
    const incompatible = makeProfile('after', index => ({amount: `text-${index}`}));
    expect(
      evaluate('cell-compare', [before, incompatible], {overrides: {statistic: 'sum'}}).status
    ).not.toBe('ready');
  });

  it('returns tied complete dataset assignments for an analyst choice', () => {
    const profiles = [makeProfile('alpha'), makeProfile('beta')];
    const result = evaluate('recipe-straight-line-catchments', profiles);
    expect(result.status).toBe('needs-mapping');
    expect(result.inputAssignmentAlternatives).toHaveLength(2);
    expect(result.inputAssignmentAlternatives).toContainEqual({
      facilities: 'alpha',
      observations: 'beta'
    });
    expect(result.inputAssignmentAlternatives).toContainEqual({
      facilities: 'beta',
      observations: 'alpha'
    });
    const chosen = evaluate('recipe-straight-line-catchments', profiles, {
      inputAssignments: {facilities: 'beta', observations: 'alpha'}
    });
    expect(chosen.status).toBe('ready');
    expect(chosen.inputAssignments).toEqual({facilities: 'beta', observations: 'alpha'});
  });

  it('uses curated whole-word slot names and evaluates chosen fields across complete assignments', async () => {
    const sample = createKeplerCuratedAnalysisSample('street-grid-facilities');
    const profiles = Object.entries(sample).map(([slot, collection]) =>
      profileKeplerFeatureCollection(collection, {
        datasetId: slot,
        label: slot,
        coordinateSystem: 'EPSG:4326'
      })
    );
    const correct = evaluate('network-accessibility', profiles, {executionEngine: 'cpu'});
    expect(correct.inputAssignments).toEqual({
      lines: 'network',
      origins: 'facilities',
      opportunities: 'demand'
    });
    expect(correct.status).toBe('needs-confirmation');
    const output = await runCatalogOperationAsync(
      getCatalogBinding('network-accessibility')!,
      correct.parameters,
      {lines: sample['network'], origins: sample['facilities'], opportunities: sample['demand']},
      {engine: 'cpu'}
    );
    expect(output.tables[0].rowCount).toBe(3);
    const renamed = profiles.map(profile => ({
      ...profile,
      datasetId:
        profile.datasetId === 'facilities'
          ? 'alpha'
          : profile.datasetId === 'demand'
            ? 'beta'
            : profile.datasetId,
      label: 'anonymous'
    }));
    const weighted = evaluate('network-accessibility', renamed, {
      executionEngine: 'cpu',
      fieldBindings: {weightProperty: 'residents'}
    });
    expect(weighted.status).toBe('needs-confirmation');
    expect(weighted.inputAssignments['opportunities']).toBe('beta');
  });

  it('blocks declared coordinate disagreement and gives the transformation remedy', () => {
    const planar = makeProfile('facilities');
    const geographic = profileKeplerFeatureCollection(
      createTestPointCollection([
        [-71, 42],
        [-70.99, 42.01]
      ]),
      {
        datasetId: 'observations',
        label: 'observations',
        coordinateSystem: 'EPSG:4326'
      }
    );
    const result = evaluate('recipe-straight-line-catchments', [planar, geographic]);
    expect(result.status).toBe('blocked');
    expect(
      result.missing.some(
        entry =>
          /different coordinate models/.test(entry.what) && /Transform/.test(entry.remedy ?? '')
      )
    ).toBe(true);
  });

  it('checks chosen minimum, maximum and distinctness before parameter construction', () => {
    const gauges = buildSampleProfile('gauges');
    for (const chosen of [[], ['rainfall', 'temperature'], ['rainfall', 'rainfall']]) {
      const result = evaluate('idw-grid', [gauges], {fieldBindings: {valueProperty: chosen}});
      expect(result.status).toBe('needs-mapping');
      expect(result.missing.map(entry => entry.what).join(' ')).toMatch(
        /valueProperty.*1 to 1 distinct columns/
      );
      expect(result.fieldBindings['valueProperty']).toBeUndefined();
    }
    const regression = buildSampleProfile('districts');
    const field = getKeplerOperationDescriptor(
      'curated-regression-with-diagnostics'
    )!.inputs[0].fields.find(entry => entry.role === 'predictors')!;
    for (const chosen of [
      [],
      ['rate', 'rate'],
      Array.from({length: field.multiple!.maximum + 1}, (_, index) => `column${index}`)
    ]) {
      const result = evaluate('curated-regression-with-diagnostics', [regression], {
        fieldBindings: {predictorProperties: chosen}
      });
      expect(result.status).not.toBe('ready');
      expect(result.missing.map(entry => entry.what).join(' ')).toMatch(
        /predictorProperties.*distinct columns/
      );
    }
  });

  it('enforces a multiple-role minimum when an optional role is supplied', () => {
    const binding = getCatalogBinding('idw-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const optionalMultiple = {
      ...descriptor,
      inputs: descriptor.inputs.map(input => ({
        ...input,
        fields: input.fields.map(field => ({
          ...field,
          required: false,
          multiple: {minimum: 2, maximum: 3}
        }))
      }))
    };
    const result = evaluateKeplerOperationApplicability(
      binding,
      optionalMultiple,
      [buildSampleProfile('gauges')],
      {fieldBindings: {valueProperty: ['rainfall']}}
    );
    expect(result.status).toBe('needs-mapping');
    expect(result.missing.map(entry => entry.what).join(' ')).toMatch(/2 to 3 distinct columns/);
  });

  it('keeps independent periods with identical metadata and permits disjoint catchments', () => {
    const before = makeProfile('before', index => ({amount: index}));
    const after = makeProfile('after', index => ({amount: 10 + index}));
    const comparison = evaluate('cell-compare', [before, after], {overrides: {statistic: 'sum'}});
    expect(comparison.status).toBe('ready');
    const observations = {
      ...after,
      datasetId: 'observations',
      label: 'observations',
      extent: [2000, 1000, 2100, 1100] as const
    };
    const catchment = evaluate('recipe-straight-line-catchments', [
      {...before, datasetId: 'facilities'},
      observations
    ]);
    expect(catchment.status).toBe('ready');
    expect(catchment.missing.map(entry => entry.what).join(' ')).not.toMatch(/overlap/);
  });

  it('does not evaluate inactive requirements, including automatically filled costs', () => {
    const binding = getCatalogBinding('network-build-graph')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const gated = {
      ...descriptor,
      requirements: descriptor.requirements.map(requirement =>
        requirement.role?.includes('cost')
          ? {...requirement, activeWhen: {parameter: 'costMode', equals: ['property']}}
          : requirement
      )
    };
    const sample = createKeplerCuratedAnalysisSample('street-grid-facilities')['network'];
    const collection = {
      ...sample,
      features: sample.features.map((feature, index) => ({
        ...feature,
        properties: {...feature.properties, cost: index === 0 ? null : 100}
      }))
    };
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'network',
      label: 'network',
      coordinateSystem: 'EPSG:4326'
    });
    const length = evaluateKeplerOperationApplicability(binding, gated, [profile], {
      overrides: {costMode: 'length'}
    });
    expect(length.status).toBe('needs-confirmation');
    expect(length.requirements.some(requirement => requirement.role?.includes('cost'))).toBe(false);
    const property = evaluateKeplerOperationApplicability(binding, gated, [profile], {
      overrides: {costMode: 'property'}
    });
    expect(property.status).toBe('needs-confirmation');
    expect(
      property.requirements.some(
        requirement =>
          requirement.role?.includes('cost') &&
          requirement.state === 'failed' &&
          requirement.severity === 'warn'
      )
    ).toBe(true);
  });

  it('evaluates conditional requirements against parameters derived from the mapped field', () => {
    const binding = getCatalogBinding('network-build-graph')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const measured: KeplerOperationDescriptor = {
      ...descriptor,
      parameters: descriptor.parameters.map(parameter =>
        parameter.name === 'costMode'
          ? {
              ...parameter,
              defaultPolicy: {
                from: 'field-measurement',
                input: 'lines',
                role: 'forward-cost',
                values: {count: 'property'}
              }
            }
          : parameter
      ),
      requirements: descriptor.requirements.map(requirement =>
        requirement.role?.includes('cost')
          ? {...requirement, activeWhen: {parameter: 'costMode', equals: ['property']}}
          : requirement
      )
    };
    const collection = {
      type: 'FeatureCollection' as const,
      features: [0, 1].map(index => ({
        type: 'Feature' as const,
        geometry: {
          type: 'LineString' as const,
          coordinates: [
            [1000 + index * 100, 1000],
            [1100 + index * 100, 1000]
          ]
        },
        properties: {cost: index === 0 ? null : 10}
      }))
    };
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'lines',
      label: 'lines',
      coordinateSystem: 'planar',
      declarations: {measurement: {cost: 'count'}}
    });
    const result = evaluateKeplerOperationApplicability(binding, measured, [profile]);
    expect(result.parameters['costMode']).toBe('property');
    expect(result.status).toBe('needs-confirmation');
    expect(
      result.requirements.some(
        requirement => requirement.id === 'no-missing-values' && requirement.state === 'failed'
      )
    ).toBe(true);
  });

  it('evaluates requirements after CPU grid caps resolve their controlling parameter', () => {
    const binding = getCatalogBinding('kernel-density-grid')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const profile = withRowCount(
      buildSampleProfile('incidents'),
      binding.inputs[0].maximumFeatures
    );
    const current = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
      executionEngine: 'cpu'
    });
    expect(
      current.resolutions.find(resolution => resolution.parameter === 'cellSize')?.clamped
    ).toBeDefined();
    const gated: KeplerOperationDescriptor = {
      ...descriptor,
      requirements: [
        ...descriptor.requirements,
        {
          id: 'maximum-rows',
          input: 'points',
          severity: 'block',
          limit: profile.rowCount - 1,
          activeWhen: {parameter: 'cellSize', equals: [Number(current.parameters['cellSize'])]}
        }
      ]
    };
    const result = evaluateKeplerOperationApplicability(binding, gated, [profile], {
      executionEngine: 'cpu'
    });
    expect(result.parameters['cellSize']).toBe(current.parameters['cellSize']);
    expect(result.status).toBe('blocked');
  });

  it('requires confirmation for the actual catchment seed collapse when safe refinement exceeds the cap', () => {
    const observations = profileKeplerFeatureCollection(
      createTestPointCollection(
        Array.from({length: 64}, (_, index) => [
          1000 + (index % 8) * 10000,
          1000 + Math.floor(index / 8) * 10000
        ])
      ),
      {
        datasetId: 'observations',
        label: 'observations',
        coordinateSystem: 'planar'
      }
    );
    const result = evaluate('recipe-straight-line-catchments', [
      makeProfile('facilities'),
      observations
    ]);
    expect(result.status).toBe('needs-confirmation');
    expect(result.missing.map(entry => entry.what).join(' ')).toMatch(
      /seed collisions.*at least 3/
    );
  });

  it('asks about a guessed row meaning even when all current leaders fit it', () => {
    const data = createTestPointCollection(
      Array.from({length: 60}, (_, index) => [
        1000 + (index % 10) * 100,
        1000 + Math.floor(index / 10) * 100
      ])
    );
    const profile = profileKeplerFeatureCollection(data, {
      datasetId: 'anonymous',
      label: 'anonymous',
      coordinateSystem: 'planar'
    });
    expect(profile.rowSemantics?.value).toBe('events');
    const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [profile], {
      executionEngine: 'cpu'
    });
    expect(
      result.questions.some(
        question => question.id === 'row-semantics' && question.datasetId === 'anonymous'
      )
    ).toBe(true);
    const settled = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS,
      [{...profile, rowSemantics: {value: 'objects', source: 'declared'}}],
      {executionEngine: 'cpu'}
    );
    expect(settled.suggestions[0].id).not.toBe(result.suggestions[0].id);
  });

  it('invalidates cached results after in-place profile facts change', () => {
    const profile = {
      ...buildSampleProfile('incidents'),
      rowSemantics: {value: 'surface-samples' as const, source: 'declared' as const}
    } as KeplerDatasetProfile;
    const first = evaluate('kernel-density-grid', [profile]);
    profile.rowSemantics = {value: 'events', source: 'declared'};
    const changed = evaluate('kernel-density-grid', [profile]);
    expect(first.status).toBe('needs-confirmation');
    expect(changed.status).toBe('ready');
    expect(changed).not.toBe(first);
    expect(evaluate('kernel-density-grid', [profile])).toBe(changed);
  });

  it('invalidates cached suggestions after an in-place row declaration changes the leaders', () => {
    const profile = {...buildSampleProfile('incidents')};
    const bindings = [...KEPLER_CATALOG_BINDINGS];
    const first = suggestKeplerAnalyses(bindings, [profile]);
    profile.rowSemantics = {value: 'objects', source: 'declared'};
    const changed = suggestKeplerAnalyses(bindings, [profile]);
    expect(changed.suggestions[0].id).not.toBe(first.suggestions[0].id);
    expect(changed.suggestions[0].curatedAnalysisId).toBe('clusters-and-footprints');
  });

  it('resolves output kindFrom from the assigned input kind', () => {
    const binding = getCatalogBinding('geometry-measures')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const inherited = {
      ...descriptor,
      outputs: descriptor.outputs.map(output => ({...output, kindFrom: binding.inputs[0].name}))
    };
    for (const profile of [
      buildSampleProfile('incidents'),
      buildSampleProfile('streets'),
      buildSampleProfile('districts')
    ]) {
      const result = evaluateKeplerOperationApplicability(binding, inherited, [profile]);
      expect(result.outputKinds[descriptor.outputs[0].tableId]).toBe(profile.kind);
    }
  });
});

describe('coordinate changes and duplicate-heavy clustering defaults', () => {
  it('derives active length defaults in the final coordinate model', () => {
    const collection = createTestPointCollection([
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1]
    ]);
    for (const [profileSystem, runSystem] of [
      ['EPSG:4326', 'planar'],
      ['planar', 'EPSG:4326']
    ] as const) {
      const profile = profileKeplerFeatureCollection(collection, {
        datasetId: 'points',
        label: 'points',
        coordinateSystem: profileSystem,
        declarations: {rowSemantics: 'events'}
      });
      const result = evaluate('kernel-density-grid', [profile], {
        overrides: {coordinateSystem: runSystem}
      });
      expect(result.status).toBe('ready');
      // The choice stands, and the contradiction with the declared layer is stated.
      expect(result.warnings.join(' ')).toMatch(
        runSystem === 'planar'
          ? /declared as longitude and latitude, but this run is set to planar/
          : /declared as planar coordinates, but this run is set to longitude and latitude/
      );
      const matching = evaluate('kernel-density-grid', [profile], {
        overrides: {coordinateSystem: profileSystem}
      });
      expect(matching.warnings.join(' ')).not.toMatch(/but this run is set to/);
      expect(result.resolutions.find(entry => entry.parameter === 'bandwidth')).toMatchObject({
        source: 'data',
        coordinateModel: runSystem === 'planar' ? 'planar' : 'geographic'
      });
      const explicit = evaluate('kernel-density-grid', [profile], {
        overrides: {coordinateSystem: runSystem, cellSize: 1, bandwidth: 2}
      });
      expect(explicit.resolutions.find(entry => entry.parameter === 'bandwidth')?.source).toBe(
        'override'
      );
      expect(explicit.missing.map(entry => entry.what).join(' ')).not.toMatch(
        /could not be derived/
      );
    }
  });

  it('ignores an incompatible length fallback when a k-nearest run does not use it', () => {
    const collection = createTestPointCollection(
      Array.from({length: 40}, (_, index) => [(index % 8) * 0.01, Math.floor(index / 8) * 0.01]),
      index => ({value: index + 1})
    );
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'features',
      label: 'features',
      coordinateSystem: 'EPSG:4326',
      declarations: {rowSemantics: 'objects', measurement: {value: 'amount'}}
    });
    const result = evaluate('hotspot-gi-star', [profile], {
      fieldBindings: {valueProperty: 'value'},
      overrides: {coordinateSystem: 'planar', neighborhood: 'knn'}
    });
    const band = result.resolutions.find(entry => entry.parameter === 'distanceBand')!;
    expect(band.source).toBe('data');
    expect(band.active).toBe(false);
    expect(result.status).toBe('ready');
    expect(result.missing.map(entry => entry.what).join(' ')).not.toMatch(/could not be derived/);
    expect(result.warnings.join(' ')).not.toMatch(
      /could not be derived|read from the data with that in mind/
    );
  });

  it('keeps distinct-location fallback radii data-derived and ready at three scales', () => {
    const normalizedRadii: number[] = [];
    for (const scale of [0.001, 1, 1000]) {
      const collection = createTestPointCollection(
        Array.from({length: 400}, (_, index) => [
          (Math.floor(index / 10) % 8) * 100 * scale,
          Math.floor(Math.floor(index / 10) / 8) * 100 * scale
        ])
      );
      const profile = profileKeplerFeatureCollection(collection, {
        datasetId: 'repeated',
        label: 'repeated',
        coordinateSystem: 'planar',
        declarations: {rowSemantics: 'events'}
      });
      const result = evaluate('density-clustering', [profile]);
      const entry = result.resolutions.find(candidate => candidate.parameter === 'epsilon')!;
      expect(entry.source).toBe('data');
      expect(entry.resolution?.rule).toBe('median-nearest-neighbor-distance');
      expect(result.status).toBe('ready');
      normalizedRadii.push(Number(result.parameters['epsilon']) / scale);
    }
    expect(normalizedRadii[0]).toBeCloseTo(normalizedRadii[1], 6);
    expect(normalizedRadii[2]).toBeCloseTo(normalizedRadii[1], 6);
  });
});

describe('third review engine regressions', () => {
  const positions = [
    [1, 1],
    [4, 1],
    [1, 4],
    [4, 4]
  ] as const;
  const period = (datasetId: string, coordinateSystem: 'planar' | 'EPSG:4326') =>
    profileKeplerFeatureCollection(createTestPointCollection(positions), {
      datasetId,
      label: datasetId,
      coordinateSystem,
      declarations: {rowSemantics: 'events'}
    });

  it('checks declared models for spatial comparisons without a coordinate parameter', () => {
    const result = evaluate(
      'cell-compare',
      [period('before', 'planar'), period('after', 'EPSG:4326')],
      {
        inputAssignments: {before: 'before', after: 'after'},
        overrides: {cellSize: 1}
      }
    );
    expect(result.status).toBe('blocked');
    expect(result.missing).toContainEqual(
      expect.objectContaining({
        what: expect.stringContaining('different coordinate models'),
        remedy: expect.stringContaining('Transform')
      })
    );
  });

  it('exempts inputs described as independent attribute tables from spatial compatibility checks', () => {
    const binding = getCatalogBinding('cell-compare')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const independentTables: KeplerOperationDescriptor = {
      ...descriptor,
      inputs: descriptor.inputs.map(input => ({...input, kind: 'attributes', fields: []})),
      parameters: [],
      requirements: []
    };
    const result = evaluateKeplerOperationApplicability(
      binding,
      independentTables,
      [period('before', 'planar'), period('after', 'EPSG:4326')],
      {
        inputAssignments: {before: 'before', after: 'after'}
      }
    );
    expect(result.status).toBe('ready');
    expect(result.missing).toEqual([]);
  });

  it('does not validate a shared value mapping when count settings leave that role inactive', () => {
    const result = evaluate(
      'cell-compare',
      [period('before', 'planar'), period('after', 'planar')],
      {
        inputAssignments: {before: 'before', after: 'after'},
        fieldBindings: {valueProperty: 'absent-column'},
        overrides: {statistic: 'count', cellSize: 1}
      }
    );
    expect(result.status).toBe('ready');
    expect(result.fieldBindings['valueProperty']).toBeUndefined();
    expect(result.measurementWarnings).toEqual([]);
    expect(result.reasons.join(' ')).toContain('absent-column chosen');
    expect(result.reasons.join(' ')).toContain('not used with the current settings');
  });

  it('warns about guessed spatial models without treating them as conflicting declarations', () => {
    const after = period('after', 'EPSG:4326');
    const result = evaluate(
      'cell-compare',
      [
        period('before', 'planar'),
        {
          ...after,
          coordinates: {model: 'geographic', source: 'guess', confidence: 'high'}
        }
      ],
      {inputAssignments: {before: 'before', after: 'after'}, overrides: {cellSize: 1}}
    );
    expect(result.status).toBe('ready');
    expect(result.warnings.join(' ')).toMatch(/coordinates.*guess|coordinate system.*guess/);
    expect(result.guessedFacts).toContainEqual({
      kind: 'coordinate-model',
      datasetId: 'after',
      effect: 'unknown'
    });
    expect(result.missing.map(entry => entry.what).join(' ')).not.toContain(
      'different coordinate models'
    );
  });

  it('ignores an inactive chosen cost column, including its type, requirements and warnings', () => {
    const profile = profileKeplerFeatureCollection(
      {
        type: 'FeatureCollection',
        features: [
          {
            geometry: {
              type: 'LineString',
              coordinates: [
                [230000, 6100000],
                [230250, 6100000]
              ]
            },
            properties: {cost: 'temporarily closed'}
          }
        ]
      },
      {datasetId: 'lines', label: 'lines', coordinateSystem: 'planar'}
    );
    const baseline = evaluate('network-build-graph', [profile], {overrides: {costMode: 'length'}});
    for (const chosen of ['cost', 'absent-column', 'unused-name'.repeat(20)]) {
      const result = evaluate('network-build-graph', [profile], {
        overrides: {costMode: 'length'},
        fieldBindings: {forwardCostProperty: chosen}
      });
      expect(result.status).toBe('needs-confirmation');
      expect(result.fieldBindings['forwardCostProperty']).toBeUndefined();
      expect(result.filledRoles.some(entry => entry.role === 'forward-cost')).toBe(false);
      expect(result.requirements.some(entry => entry.role === 'forward-cost')).toBe(false);
      expect(result.warnings).toEqual(baseline.warnings);
      expect(result.measurementWarnings).toEqual(baseline.measurementWarnings);
      expect(result.reasons.join(' ')).toContain(`${chosen} chosen`);
      expect(result.reasons.join(' ')).toContain('not used with the current settings');
      const overridden = evaluate('network-build-graph', [profile], {
        overrides: {costMode: 'length', forwardCostProperty: chosen}
      });
      expect(overridden.status).toBe('needs-confirmation');
      expect(overridden.parameters['forwardCostProperty']).toBe('');
    }
    expect(
      evaluate('network-build-graph', [profile], {
        overrides: {costMode: 'property'},
        fieldBindings: {forwardCostProperty: 'cost'}
      }).status
    ).toBe('blocked');
  });

  it('requires mapping when a sampled or signed sum cannot bound dot allocation', () => {
    const base = buildSampleProfile('districts');
    for (const statistics of [{sampleSize: 32, minimum: 100}, {minimum: -800000}]) {
      const profile = {
        ...base,
        columns: [
          {
            name: 'amount',
            type: 'number' as const,
            rowCount: 200,
            nullCount: 0,
            mean: 500,
            measurement: 'amount' as const,
            measurementSource: 'declared' as const,
            ...statistics
          }
        ]
      };
      const result = evaluate('geometry-dot-density', [profile], {
        fieldBindings: {valueProperty: 'amount'}
      });
      expect(['needs-mapping', 'blocked']).toContain(result.status);
      const resolution = result.resolutions.find(entry => entry.parameter === 'dotValue')!;
      expect(resolution.source).toBe('fallback');
      expect(resolution.reason).toContain(
        'sampleSize' in statistics ? 'sampled' : 'negative values'
      );
      expect(result.missing.map(entry => entry.what).join(' ')).toContain(resolution.reason);
      const explicit = evaluate('geometry-dot-density', [profile], {
        fieldBindings: {valueProperty: 'amount'},
        overrides: {dotValue: 10000}
      });
      expect(explicit.resolutions.find(entry => entry.parameter === 'dotValue')?.source).toBe(
        'override'
      );
    }
  });
});

it('uses cross-parameter OR for role requirements and preflight requirements', () => {
  const binding = getCatalogBinding('point-density-grid')!;
  const base = getKeplerOperationDescriptor(binding.id)!;
  const condition = {
    anyOf: [
      {parameter: 'statistic', equals: ['sum']},
      {parameter: 'includeEmptyCells', equals: [false]}
    ]
  };
  const descriptor: KeplerOperationDescriptor = {
    ...base,
    inputs: base.inputs.map(input => ({
      ...input,
      fields: input.fields.map(field => ({...field, required: condition}))
    })),
    requirements: [
      ...base.requirements,
      {
        id: 'minimum-rows',
        input: 'points',
        severity: 'block',
        limit: 1000000000,
        activeWhen: condition
      }
    ]
  };
  const profile = {...buildSampleProfile('incidents'), columns: []};
  for (const [statistic, includeEmptyCells, required] of [
    ['count', true, false],
    ['sum', true, true],
    ['count', false, true]
  ] as const) {
    const assessment = evaluateKeplerOperationApplicability(binding, descriptor, [profile], {
      overrides: {statistic, includeEmptyCells}
    });
    expect(
      assessment.requirements.some(
        requirement => requirement.id === 'minimum-rows' && requirement.state === 'failed'
      )
    ).toBe(required);
    expect(assessment.missing.some(missing => missing.what.includes('column'))).toBe(required);
  }
});
