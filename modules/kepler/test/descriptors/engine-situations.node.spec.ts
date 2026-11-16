// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {writeFileSync} from 'node:fs';
import {describe, expect, it} from 'vitest';
import {
  getCatalogBinding,
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../../src/analysis/catalog-geojson';
import {
  evaluateKeplerOperationApplicability,
  suggestKeplerAnalyses,
  type KeplerAnalysisSuggestion
} from '../../src/descriptors/applicability';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {createTestPointCollection} from './engine-test-profiles';
import {getKeplerOperationDescriptor} from '../../src/descriptors/operation-descriptors';
// 32 datasets built with the real profiler by an independent reviewer, each with an analyst's
// expectation written before the engine was run.
import {buildSituations, type Situation} from './engine-situation-fixtures';

const situations: Situation[] = buildSituations();

function getKey(suggestion: KeplerAnalysisSuggestion): string {
  return suggestion.curatedAnalysisId ?? suggestion.bindingId;
}

describe('review situations', () => {
  it('preserves strict diversity across all review situations', () => {
    for (const situation of situations) {
      const result = suggestKeplerAnalyses(
        KEPLER_CATALOG_BINDINGS,
        situation.datasets.map(dataset => dataset.profile)
      );
      const counts = new Map<string, number>();
      for (const entry of result.suggestions)
        counts.set(entry.question, (counts.get(entry.question) ?? 0) + 1);
      for (const [question, count] of counts)
        expect(count, `${situation.id}: ${question}`).toBeLessThanOrEqual(2);
      expect(result.suggestions.length).toBeLessThanOrEqual(6);
    }
  });

  it('offers temporal event analysis for incident dates while keeping static answers and diversity', () => {
    const situation = situations.find(entry => entry.id === 'S03-crime-incidents')!;
    const result = suggestKeplerAnalyses(
      KEPLER_CATALOG_BINDINGS,
      situation.datasets.map(dataset => dataset.profile)
    );
    const top = result.suggestions.slice(0, 3);
    expect(top.map(getKey)).toContain('emerging-hot-spots');
    expect(top.some(entry => ['density-surface', 'hot-spot-map'].includes(getKey(entry)))).toBe(
      true
    );
    expect(top.some(entry => situation.expectation.first.includes(getKey(entry)))).toBe(true);
    const emerging = top.find(entry => getKey(entry) === 'emerging-hot-spots')!;
    expect(emerging.applicability.fieldBindings['timeProperty']).toBe('date');
    expect(emerging.scoreTerms['rowSemanticsMismatch']).toBeCloseTo(0);
    expect(emerging.scoreTerms['timeColumn']).toBe(12);
  });

  it('writes the table', () => {
    const lines: string[] = [];
    for (const situation of situations) {
      const profiles = situation.datasets.map(dataset => dataset.profile);
      const started = performance.now();
      const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, profiles);
      const elapsed = Math.round(performance.now() - started);
      const keys = result.suggestions.map(getKey);
      const top = result.suggestions.slice(0, 3);
      const {first, never} = situation.expectation;
      const firstOk = first.length === 0 || first.some(key => keys.slice(0, 3).includes(key));
      const neverOk = never.every(key => !keys.slice(0, 3).includes(key));
      lines.push(
        `${situation.id} [${elapsed} ms] first:${firstOk ? 'ok' : 'MISS'} never:${neverOk ? 'ok' : 'HIT'} top3: ${top.map(entry => `${getKey(entry)}${entry.datasetIds.length > 0 && profiles.length > 1 ? '@' + entry.datasetIds.join('+') : ''} [${entry.status}]`).join(' | ')} || expect ${first.join('/')} never ${never.join('/')} || q=${result.questions.map(question => question.id).join(',')}`
      );
    }
    if (process.env['WRITE_SITUATION_REPORT']) {
      writeFileSync(process.env['WRITE_SITUATION_REPORT'], lines.join('\n'));
    }
    expect(lines.length).toBe(situations.length);
  });
});

// Situations whose suggestions are run through the catalog: realistic layers of 12 to 3,000 rows.
const RUN_SITUATION_IDS = [
  'S01-customers-csv',
  'S03-crime-incidents',
  'S04-taxi-trips',
  'S05-weather-stations',
  'S06-ais-vessels',
  'S07-earthquakes',
  'S08-parcels',
  'S09-roads',
  'S10-bus-stops-routes',
  'S12-twelve-points',
  'S15-antimeridian',
  'S19-districts-cases-population',
  'S20-districts-rate-population',
  'S22-rain-gauges',
  'S26-site-metres-0-50',
  'S31-crash-counts-at-points',
  'S32-counties-small'
];

type RunVerdict = {suggestion: string; problem?: string};

/** Class schemes whose single class (not significant) can be the honest answer. */
const INFERENTIAL_CLASS_SCHEMES: readonly string[] = [
  'hot-spot-confidence-bins',
  'local-moran-clusters',
  'local-geary-classes',
  'significance-flag'
];

/**
 * Ready suggestions that still throw or return nothing, with the change they wait for. Each needs
 * a descriptor or binding change that is not the engine's: the test fails when one of them
 * starts to pass (remove the entry) and when any other suggestion fails.
 */
const KNOWN_DESCRIPTOR_ISSUES: Readonly<
  Record<string, readonly {suggestion: string; wait: string}[]>
> = {
  'S12-twelve-points': [
    {
      suggestion: 'binding:cell-aggregation',
      wait: 'cell-aggregation should state a minimum-rows warning (about 30): 12 points give one point per cell and a constant count'
    }
  ]
};

/** Distinct values of a table column. */
function getDistinctValues(values: readonly (string | number | boolean | null)[]): Set<unknown> {
  return new Set(values.filter(value => value !== null));
}

/**
 * Runs one suggestion on the CPU reference engine and returns what is wrong with the result, or
 * undefined when it ran and gave a non-degenerate answer: a throw, no table, an empty table, an
 * output column (the one the map colours by) with one value only, or numbers that are all zero.
 */
function runSuggestion(
  suggestion: KeplerAnalysisSuggestion,
  collections: ReadonlyMap<string, CatalogFeatureCollection>
): string | undefined {
  const binding = getCatalogBinding(suggestion.bindingId)!;
  if (!binding.run) {
    return undefined;
  }
  const inputs: Record<string, CatalogFeatureCollection> = {};
  for (const input of binding.inputs) {
    const datasetId = suggestion.applicability.inputAssignments[input.name];
    const collection = datasetId === undefined ? undefined : collections.get(datasetId);
    if (!collection) {
      return `input ${input.name} has no collection`;
    }
    inputs[input.name] = collection;
  }
  let output;
  try {
    output = binding.run(
      inputs,
      normalizeCatalogParameters(binding, suggestion.applicability.parameters)
    );
  } catch (error) {
    return `threw: ${error instanceof Error ? error.message : String(error)}`;
  }
  if (output.tables.length === 0 || output.tables.every(table => table.rowCount === 0)) {
    return 'returned no rows';
  }
  const descriptor = getKeplerOperationDescriptor(binding.id)!;
  for (const outputDescriptor of descriptor.outputs) {
    const column = outputDescriptor.presentation?.column;
    const table = output.tables.find(candidate => candidate.id === outputDescriptor.tableId);
    if (!column || !table || table.rowCount < 5 || !(column in table.columns)) {
      continue;
    }
    const values = table.columns[column];
    // A significance map with no significant cell is a finding, not a failure: the data may hold
    // no pattern. Class schemes that are memberships (clusters, noise) must still vary.
    const scheme = outputDescriptor.presentation?.classScheme;
    if (scheme !== undefined && INFERENTIAL_CLASS_SCHEMES.includes(scheme)) {
      continue;
    }
    const distinct = getDistinctValues(values);
    if (distinct.size < 2) {
      return `${outputDescriptor.tableId}.${column} has a single value (${String([...distinct][0])}) in ${table.rowCount} rows`;
    }
    if (values.every(value => value === 0 || value === null)) {
      return `${outputDescriptor.tableId}.${column} is all zeros`;
    }
  }
  return undefined;
}

describe('ready suggestions run', () => {
  it('runs the top three ready suggestions of seventeen realistic situations without a throw, an empty table or a degenerate result', () => {
    const verdicts: Record<string, RunVerdict[]> = {};
    let ran = 0;
    for (const situation of situations.filter(entry => RUN_SITUATION_IDS.includes(entry.id))) {
      const profiles = situation.datasets.map(dataset => dataset.profile);
      const collections = new Map(
        situation.datasets.map(dataset => [dataset.profile.datasetId, dataset.collection])
      );
      // The CPU reference engine is what a test (and a host without a GPU) runs: its work caps apply.
      const result = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, profiles, {
        executionEngine: 'cpu',
        maximumCount: 12
      });
      const ready = result.suggestions.filter(entry => entry.status === 'ready').slice(0, 3);
      for (const suggestion of ready) {
        const problem = runSuggestion(suggestion, collections);
        ran++;
        if (problem) {
          verdicts[situation.id] = [
            ...(verdicts[situation.id] ?? []),
            {suggestion: suggestion.id, problem}
          ];
        }
      }
    }
    expect(ran).toBeGreaterThanOrEqual(12 * 2);
    const known = Object.fromEntries(
      Object.entries(KNOWN_DESCRIPTOR_ISSUES).map(([id, entries]) => [
        id,
        entries.map(entry => entry.suggestion)
      ])
    );
    const found = Object.fromEntries(
      Object.entries(verdicts).map(([id, entries]) => [id, entries.map(entry => entry.suggestion)])
    );
    expect(found).toEqual(known);
  });
});

it.skipIf(!process.env['WRITE_F7_SITUATIONS'])(
  'replays every rv8 top-three suggestion through the CPU binding',
  async () => {
    const records = [];
    for (const situation of situations) {
      const suggestions = suggestKeplerAnalyses(
        KEPLER_CATALOG_BINDINGS,
        situation.datasets.map(dataset => dataset.profile),
        {executionEngine: 'cpu'}
      );
      const top = [];
      for (const suggestion of suggestions.suggestions.slice(0, 3)) {
        const problems: string[] = [];
        if (suggestion.status === 'ready') {
          const inputs = Object.fromEntries(
            Object.entries(suggestion.applicability.inputAssignments).map(([input, datasetId]) => [
              input,
              situation.datasets.find(dataset => dataset.profile.datasetId === datasetId)!
                .collection
            ])
          );
          try {
            const result = await runCatalogOperationAsync(
              getCatalogBinding(suggestion.bindingId)!,
              suggestion.applicability.parameters,
              inputs,
              {engine: 'cpu'}
            );
            if (result.tables.length === 0 || result.tables.every(table => table.rowCount === 0)) {
              problems.push('returned no rows');
            }
            for (const output of getKeplerOperationDescriptor(suggestion.bindingId)!.outputs) {
              const table = result.tables.find(entry => entry.id === output.tableId);
              const column = output.presentation?.column;
              if (
                !table ||
                !column ||
                table.rowCount < 5 ||
                !table.columns[column] ||
                (output.presentation?.classScheme &&
                  INFERENTIAL_CLASS_SCHEMES.includes(output.presentation.classScheme))
              ) {
                continue;
              }
              const values = getDistinctValues(table.columns[column]);
              if (values.size < 2) {
                problems.push(
                  `${table.id}.${column} has one value (${String([...values][0])}) in ${table.rowCount} rows`
                );
              }
            }
            problems.push(
              ...result.diagnostics
                .filter(diagnostic => diagnostic.severity === 'error')
                .map(diagnostic => diagnostic.message)
            );
          } catch (error) {
            problems.push(`threw: ${error instanceof Error ? error.message : String(error)}`);
          }
        }
        top.push({
          id: getKey(suggestion),
          datasets: suggestion.datasetIds,
          status: suggestion.status,
          problems
        });
      }
      records.push({
        id: situation.id,
        expectedFirst: situation.expectation.first,
        never: situation.expectation.never,
        firstPresent:
          situation.expectation.first.length === 0 ||
          top.some(suggestion => situation.expectation.first.includes(suggestion.id)),
        neverAbsent: top.every(suggestion => !situation.expectation.never.includes(suggestion.id)),
        top
      });
    }
    writeFileSync(process.env['WRITE_F7_SITUATIONS']!, JSON.stringify(records, null, 2));
    expect(records).toHaveLength(situations.length);
  },
  1200000
);

describe('ready outcomes that need an explanation', () => {
  it('warns that the nationwide default can connect every point into one cluster', async () => {
    // Exact C28 coordinates from the independent c6 probe.
    const collection: CatalogFeatureCollection = {
      type: 'FeatureCollection',
      features: Array.from({length: 80}, (_, index) => ({
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [-121 + (index % 16) * 3, 29 + Math.floor(index / 16) * 4]
        },
        properties: {event_id: 5600 + index}
      }))
    };
    const profile = profileKeplerFeatureCollection(collection, {
      datasetId: 'national-dispatches',
      label: 'national dispatches',
      coordinateSystem: 'EPSG:4326',
      declarations: {rowSemantics: 'events'}
    });
    const suggestion = suggestKeplerAnalyses(KEPLER_CATALOG_BINDINGS, [profile], {
      executionEngine: 'cpu',
      maximumCount: 20
    }).suggestions.find(entry => entry.curatedAnalysisId === 'clusters-and-footprints')!;
    expect(suggestion.status).toBe('ready');
    const result = await runCatalogOperationAsync(
      getCatalogBinding(suggestion.bindingId)!,
      suggestion.applicability.parameters,
      {points: collection},
      {engine: 'cpu'}
    );
    expect(new Set(result.tables[0].columns['cluster']).size).toBe(1);
    expect(suggestion.applicability.warnings.join(' ')).toMatch(/one cluster.*reduce.*epsilon/i);
  });

  it('warns that counting the same stations ignores changed measurements and yields zeros', async () => {
    // Exact C21 positions: two groups from the reviewer's deterministic generator, seed 727.
    let randomState = 727;
    const nextFraction = () => {
      randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
      return randomState / 4294967296;
    };
    const before = createTestPointCollection(
      Array.from({length: 84}, (_, index) => {
        const group = index < 84 * 0.65 ? 0 : 1;
        const east = group * 1800 + nextFraction() * 250;
        const north = group * 1200 + nextFraction() * 250;
        return [-71.12 + east / 83000, 42.3 + north / 111320];
      })
    );
    const after = {
      ...before,
      features: before.features.map((feature, index) => ({
        ...feature,
        properties: {cases: 3 + (index % 11)}
      }))
    };
    const earlier = {
      ...before,
      features: before.features.map((feature, index) => ({
        ...feature,
        properties: {cases: index % 7}
      }))
    };
    const profiles = [earlier, after].map((collection, index) =>
      profileKeplerFeatureCollection(collection, {
        datasetId: index === 0 ? 'before-2024' : 'after-2025',
        label: index === 0 ? 'before 2024' : 'after 2025',
        coordinateSystem: 'EPSG:4326',
        declarations: {rowSemantics: 'objects'}
      })
    );
    const binding = getCatalogBinding('cell-compare')!;
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    const options = {inputAssignments: {before: 'before-2024', after: 'after-2025'}};
    const count = evaluateKeplerOperationApplicability(binding, descriptor, profiles, options);
    expect(count.status).toBe('ready');
    const result = await runCatalogOperationAsync(
      binding,
      count.parameters,
      {before: earlier, after},
      {engine: 'cpu'}
    );
    expect(result.tables[0].columns['change'].every(value => value === 0)).toBe(true);
    expect(count.warnings.join(' ')).toMatch(
      /identical.*locations.*zero change.*statistic.*sum.*valueProperty/i
    );
    const sum = evaluateKeplerOperationApplicability(binding, descriptor, profiles, {
      ...options,
      overrides: {statistic: 'sum'},
      fieldBindings: {valueProperty: 'cases'}
    });
    expect(sum.warnings.join(' ')).not.toMatch(/zero change/);
    const measured = await runCatalogOperationAsync(
      binding,
      sum.parameters,
      {before: earlier, after},
      {engine: 'cpu'}
    );
    expect(
      measured.tables[0].columns['change'].some(value => typeof value === 'number' && value !== 0)
    ).toBe(true);
    const events = evaluateKeplerOperationApplicability(
      binding,
      descriptor,
      profiles.map(profile => ({...profile, rowSemantics: {value: 'events', source: 'declared'}})),
      options
    );
    expect(events.warnings.join(' ')).not.toMatch(/zero change/);
  });
});
