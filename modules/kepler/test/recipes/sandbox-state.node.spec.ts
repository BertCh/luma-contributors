// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  parseKeplerSandboxState,
  rebindKeplerSandboxState,
  restoreKeplerSandboxState,
  serializeKeplerSandboxState,
  type KeplerSandboxState
} from '../../src/recipes/sandbox-state';

function createState(): KeplerSandboxState {
  const dataset = {datasetId: 'observations', revision: 3};
  return {
    schemaVersion: 1,
    sources: [
      {
        dataset,
        reference: {
          kind: 'fixture',
          id: 'observations',
          version: 1,
          parameters: {seed: 42, rowCount: 100}
        },
        schema: [{name: 'value', type: 'Float32', nullable: true}],
        normalization: {float64: {mode: 'exact'}}
      }
    ],
    nativeConfig: {
      version: 'v1',
      config: {visState: {filters: [{name: ['value'], value: [2, 8]}]}}
    },
    selections: [
      {source: dataset, revision: 7, value: {filters: [{column: 'value', minimum: 2, maximum: 8}]}}
    ],
    analyses: [
      {
        schemaVersion: 1,
        algorithmVersion: 1,
        id: 'analysis',
        source: dataset,
        population: 'selected',
        selectionRevision: 7,
        backendPolicy: 'require-gpu',
        operation: {kind: 'standardize', column: 'value', method: 'z-score', variance: 'sample'}
      }
    ]
  };
}

test('sandbox round-trips detached native configuration, source policy, selection and analysis', () => {
  const original = createState();
  const serialized = serializeKeplerSandboxState(original);
  const restored = restoreKeplerSandboxState(
    serialized,
    () => original.sources[0],
    () => 7
  );
  expect(restored).toEqual(original);
  expect(restored.nativeConfig).not.toBe(original.nativeConfig);
  expect(restored.sources[0].normalization).not.toBe(original.sources[0].normalization);
});

test.each([
  new Uint32Array([1]),
  new Map([['buffer', 1]]),
  new Date(),
  1n,
  Number.NaN,
  () => 1,
  undefined
])('sandbox rejects non-JSON/native handles rather than silently serializing them: %s', value => {
  expect(() => parseKeplerSandboxState({...createState(), nativeConfig: value})).toThrow();
});

test('sandbox rejects internally stale source and selection references', () => {
  const state = createState();
  expect(() =>
    parseKeplerSandboxState({
      ...state,
      selections: [{...state.selections[0], source: {datasetId: 'observations', revision: 4}}]
    })
  ).toThrow(/stale dataset/);
  expect(() =>
    parseKeplerSandboxState({...state, analyses: [{...state.analyses[0], selectionRevision: 6}]})
  ).toThrow(/stale selection/);
  expect(() => parseKeplerSandboxState({...state, schemaVersion: 2})).toThrow(/version/);
  expect(() =>
    parseKeplerSandboxState({...state, sources: [...state.sources, ...state.sources]})
  ).toThrow(/Duplicate/);
});

test('ordinary restore rejects current revisions that differ from the saved document', () => {
  const state = createState();
  const serialized = serializeKeplerSandboxState(state);
  expect(() =>
    restoreKeplerSandboxState(
      serialized,
      () => ({...state.sources[0], dataset: {datasetId: 'observations', revision: 4}}),
      () => 7
    )
  ).toThrow(/dataset revision/);
  expect(() =>
    restoreKeplerSandboxState(
      serialized,
      () => state.sources[0],
      () => 8
    )
  ).toThrow(/selection revision/);
});

test('source matching treats omitted normalization options consistently with explicit undefined', () => {
  const state = createState();
  expect(
    restoreKeplerSandboxState(
      serializeKeplerSandboxState(state),
      () => ({
        ...state.sources[0],
        normalization: {...state.sources[0].normalization, columns: undefined}
      }),
      () => 7
    )
  ).toEqual(state);
});

test('explicit reimport rebinds recipes only after matching source/schema/policy and replaying selections', () => {
  const state = createState();
  const source = {...state.sources[0], dataset: {datasetId: 'observations', revision: 5}};
  expect(() =>
    rebindKeplerSandboxState(state, [{source, restoredSelectionRevision: 2}], {
      allowReimport: false
    })
  ).toThrow(/explicit/);
  expect(() => rebindKeplerSandboxState(state, [{source}], {allowReimport: true})).toThrow(
    /revision/
  );
  const restored = rebindKeplerSandboxState(state, [{source, restoredSelectionRevision: 2}], {
    allowReimport: true
  });
  expect(restored.analyses[0].source.revision).toBe(5);
  expect(restored.analyses[0].selectionRevision).toBe(2);
  expect(restored.selections[0].revision).toBe(2);
  expect(restored.nativeConfig).toEqual(state.nativeConfig);
  expect(state.analyses[0].source.revision).toBe(3);
  for (const changed of [
    {...source, reference: {...source.reference, parameters: {seed: 43, rowCount: 100}}},
    {...source, schema: [{name: 'value', type: 'Float64', nullable: true}]},
    {
      ...source,
      normalization: {
        float64: {mode: 'round' as const, maximumAbsoluteError: 1, maximumRelativeError: 0}
      }
    }
  ])
    expect(() =>
      rebindKeplerSandboxState(state, [{source: changed, restoredSelectionRevision: 2}], {
        allowReimport: true
      })
    ).toThrow(/does not match/);
});

test('normalization round-trips categorical code order and exact time units', () => {
  const state = createState();
  state.sources[0].normalization = {
    columns: ['value'],
    float64: {mode: 'round', maximumAbsoluteError: 0.001, maximumRelativeError: 0.0001},
    dictionaries: {category: {ordered: false, values: ['South', 'North', '']}},
    timeColumns: {time: 'nanosecond'}
  };
  expect(
    parseKeplerSandboxState(JSON.parse(serializeKeplerSandboxState(state))).sources[0].normalization
  ).toEqual(state.sources[0].normalization);
  expect(() =>
    parseKeplerSandboxState({
      ...state,
      sources: [
        {
          ...state.sources[0],
          normalization: {
            float64: {mode: 'round', maximumAbsoluteError: -1, maximumRelativeError: 0}
          }
        }
      ]
    })
  ).toThrow(/budget/);
});
