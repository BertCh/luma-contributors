// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetOpsTable} from '@kepler.gl/table';
import {runSuitability, wrapTo} from '@kepler.gl/actions';
import {expect, test} from 'vitest';
import {
  getResidentSuitabilityDerivedDataset,
  parseResidentSuitabilityLineage
} from '../../../examples/deck/kepler-compute/src/resident-suitability-lineage';
import {
  getResidentSuitabilityRequest,
  getResidentSuitabilityRequestKey,
  getResidentSuitabilityStatus,
  RESIDENT_SUITABILITY_STATUS
} from '../../../examples/deck/kepler-compute/src/resident-suitability-actions';
import {assertResidentSuitabilityStatistics} from '../../../examples/deck/kepler-compute/src/resident-suitability-pack';
import {
  computeNativeSuitabilityScores,
  createResidentSuitabilityResultPlan,
  getResidentSuitabilityEffectiveWeights,
  RESIDENT_SUITABILITY_KERNEL_ROW_LIMIT,
  type ResidentSuitabilityOperation,
  type ResidentSuitabilitySourceMetadata
} from '../../../examples/deck/kepler-compute/src/resident-suitability-plan';

// Load the pinned native CPU implementation without its unrelated rendering barrels.
const {suitabilityDataset} = await import(
  // @ts-expect-error Direct implementation file has no declaration entry.
  new URL(
    '../../../node_modules/@kepler.gl/table/dist/esm/dataset-ops/suitability.js',
    import.meta.url
  ).pathname
);
const {RowDataContainer} = await import(
  // @ts-expect-error Direct implementation file has no declaration entry.
  new URL('../../../node_modules/@kepler.gl/utils/dist/esm/row-data-container.js', import.meta.url)
    .pathname
);

const names = ['id', 'cost', 'access', 'label'];
const types = ['integer', 'real', 'real', 'string'];
const rows: (number | string | null)[][] = [
  [1, 10, 3, 'a'],
  [2, 20, 1, 'b'],
  [3, 15, 7, 'c'],
  [4, null, null, 'd'],
  [5, 40, 5, 'e'],
  [6, 25, 9, 'f']
];
const operation: ResidentSuitabilityOperation = {
  id: 'op',
  dataId: 'data',
  weights: {cost: 0.25, access: 1},
  weightStandardization: 'normalize',
  dataStandardization: 'range',
  outputFieldName: 'score',
  resultId: 'result',
  resultLabel: 'Suitability result'
};
const nativeTable = (): DatasetOpsTable =>
  ({
    id: 'data',
    label: 'Data',
    color: [1, 2, 3],
    fields: names.map((name, index) => ({
      name,
      type: types[index],
      analyzerType: types[index].toUpperCase(),
      fieldIdx: index,
      id: name,
      format: ''
    })),
    dataContainer: new RowDataContainer({rows})
  }) as unknown as DatasetOpsTable;
const metadata: ResidentSuitabilitySourceMetadata = {
  datasetId: 'data',
  label: 'Data',
  revision: 3,
  rowCount: rows.length,
  fields: names.map((name, index) => ({
    name,
    type: types[index],
    analyzerType: types[index].toUpperCase()
  })),
  columns: {
    id: {format: 'sint32'},
    cost: {format: 'float32'},
    access: {format: 'float32'},
    label: {format: 'uint32'}
  },
  dictionaries: {label: {values: ['a', 'b', 'c', 'd', 'e', 'f'], ordered: false}}
};

/**
 * Emulates the resident path on the CPU with the independent native-semantics oracle: standardize
 * each column over its own defined values, then apply the planned effective weights.
 */
function scoreResident(change: Partial<ResidentSuitabilityOperation>) {
  const plan = createResidentSuitabilityResultPlan({...operation, ...change}, metadata);
  const columns = plan.scoring.indicators.map(({column}) =>
    rows.map(row => row[names.indexOf(column)] as number | null)
  );
  const effectiveWeights = getResidentSuitabilityEffectiveWeights(
    plan.scoring,
    plan.provenance.weightStandardization
  );
  const scores = computeNativeSuitabilityScores(
    columns,
    // The oracle applies weight standardization itself; raw weights pass through unchanged.
    effectiveWeights,
    {dataStandardization: plan.provenance.dataStandardization, weightStandardization: 'raw'}
  );
  return {
    plan,
    values: scores.map(score => score ?? 0),
    validity: scores.map(score => (score === null ? 0 : 1))
  };
}

test.each([
  ['range', 'normalize'],
  ['range', 'raw'],
  ['zScore', 'normalize'],
  ['zScore', 'raw'],
  ['raw', 'normalize'],
  ['raw', 'raw']
] as const)('%s data and %s weights reproduce the pinned native result within Float32', (dataStandardization, weightStandardization) => {
  const config = {dataStandardization, weightStandardization};
  const resident = scoreResident(config);
  const native = suitabilityDataset(nativeTable(), {
    weights: operation.weights,
    outputFieldName: 'score',
    resultId: 'result',
    label: 'Suitability result',
    operationId: 'op',
    ...config
  });
  const nativeRows = native.data.rows as unknown[][];
  expect(resident.plan.provenance.fields).toEqual(native.data.fields);
  expect(resident.plan.derivedDataset).toEqual(native.metadata.derivedDataset);
  expect(resident.plan.columns.map(column => column.outputColumn)).toEqual(
    native.data.fields.slice(0, -1).map((field: {name: string}) => field.name)
  );
  const nativeScores = nativeRows.map(row => row[row.length - 1] as number | null);
  expect(nativeScores[3]).toBeNull();
  for (const [row, expected] of nativeScores.entries()) {
    if (expected === null) {
      expect(resident.validity[row]).toBe(0);
      expect(resident.values[row]).toBe(0);
    } else {
      expect(resident.validity[row]).toBe(1);
      expect(resident.values[row]).toBeCloseTo(expected, 5);
    }
  }
});

const statistics = (change: Record<string, unknown> = {}) => ({
  names: ['a', 'b'],
  formats: ['float32', 'sint32'] as ('float32' | 'sint32')[],
  extents: new Float32Array([1, 5, -3, 9]),
  counts: new Uint32Array([4, 4]),
  mixedMissingRows: 0,
  ...change
});

test('GPU statistics accept varied columns and empty defined sets', () => {
  expect(() => assertResidentSuitabilityStatistics(statistics())).not.toThrow();
  expect(() =>
    assertResidentSuitabilityStatistics(
      statistics({counts: new Uint32Array([0, 0]), extents: new Float32Array(4)})
    )
  ).not.toThrow();
});

test('mixed missing rows, inexact integers and constant columns are refused', () => {
  expect(() => assertResidentSuitabilityStatistics(statistics({mixedMissingRows: 1}))).toThrow(
    /different rows/
  );
  expect(() =>
    assertResidentSuitabilityStatistics(
      statistics({extents: new Float32Array([1, 5, -3, 16_777_216])})
    )
  ).toThrow(/"b" has integers/);
  // Float32 columns may hold any magnitude.
  expect(() =>
    assertResidentSuitabilityStatistics(statistics({extents: new Float32Array([1, 3e9, -3, 9])}))
  ).not.toThrow();
  expect(() =>
    assertResidentSuitabilityStatistics(statistics({extents: new Float32Array([2, 2, -3, 9])}))
  ).toThrow(/"a" is constant/);
  expect(() =>
    assertResidentSuitabilityStatistics(statistics({counts: new Uint32Array([4])}))
  ).toThrow(/do not match/);
});

test('effective weights normalize by the total or stay raw', () => {
  const {scoring} = createResidentSuitabilityResultPlan(operation, metadata);
  expect(getResidentSuitabilityEffectiveWeights(scoring, 'normalize')).toEqual([0.2, 0.8]);
  expect(getResidentSuitabilityEffectiveWeights(scoring, 'raw')).toEqual([0.25, 1]);
});

test('result plans name columns like native and reject unsupported shapes with guidance', () => {
  const plan = createResidentSuitabilityResultPlan(
    {...operation, outputFieldName: 'cost', columns: ['id', 'cost']},
    metadata
  );
  expect(plan.provenance.fields.map(field => field.name)).toEqual(['id', 'cost', 'cost_1']);
  expect(plan.indicatorFormats).toEqual(['float32', 'float32']);
  const reject = (
    change: Partial<ResidentSuitabilityOperation>,
    pattern: RegExp,
    source = metadata
  ) =>
    expect(() => createResidentSuitabilityResultPlan({...operation, ...change}, source)).toThrow(
      pattern
    );
  reject({dataStandardization: 'bogus' as never}, /Unknown suitability data standardization/);
  reject({resultId: 'data'}, /separate result/);
  reject({}, /1 to 16777216 rows/, {...metadata, rowCount: 0});
  // The former 262144-row cap is gone; only the kernel capacity bounds the input.
  expect(() =>
    createResidentSuitabilityResultPlan(operation, {...metadata, rowCount: 262_145})
  ).not.toThrow();
  expect(() =>
    createResidentSuitabilityResultPlan(operation, {
      ...metadata,
      rowCount: RESIDENT_SUITABILITY_KERNEL_ROW_LIMIT
    })
  ).not.toThrow();
  reject({}, /1 to 16777216 rows/, {
    ...metadata,
    rowCount: RESIDENT_SUITABILITY_KERNEL_ROW_LIMIT + 1
  });
  reject({}, /needs prepared Float32/, {
    ...metadata,
    columns: {...metadata.columns, cost: {format: 'float64'}}
  });
  reject({}, /cannot carry column label/, {
    ...metadata,
    dictionaries: {label: {values: ['a'], ordered: true}}
  });
  reject({}, /at most 32 output/, {
    ...metadata,
    fields: [
      ...metadata.fields,
      ...Array.from({length: 40}, (_, index) => ({name: `extra${index}`, type: 'real'}))
    ],
    columns: {
      ...metadata.columns,
      ...Object.fromEntries(
        Array.from({length: 40}, (_, index) => [`extra${index}`, {format: 'float32'}])
      )
    }
  });
  expect(() => scoreResident({weights: {cost: 1}, columns: ['cost']})).not.toThrow();
});

test('categorical weighted fields are reclassified by dictionary position or supplied scores', () => {
  const categorical: ResidentSuitabilitySourceMetadata = {
    ...metadata,
    columns: {...metadata.columns, label: {format: 'uint32'}},
    dictionaries: {label: {values: ['a', 'b', 'c'], ordered: false}}
  };
  const byPosition = createResidentSuitabilityResultPlan(
    {...operation, weights: {cost: 1, label: 1}, columns: ['cost']},
    categorical
  );
  expect(byPosition.indicatorReclassifications).toEqual([null, [0, 1, 2, null]]);
  expect(byPosition.indicatorFormats[1]).toBe('uint32');
  expect(byPosition.scoring.notes.join(' ')).toMatch(/dictionary position/);
  const supplied = createResidentSuitabilityResultPlan(
    {
      ...operation,
      weights: {cost: 1, label: 1},
      columns: ['cost'],
      categoryScores: {label: {a: 10, c: 30}}
    },
    categorical
  );
  expect(supplied.indicatorReclassifications[1]).toEqual([10, null, 30, null]);
  expect(supplied.provenance.reclassifications).toEqual([
    {column: 'label', scores: [10, null, 30, null]}
  ]);
  expect(parseResidentSuitabilityLineage(JSON.parse(JSON.stringify(supplied.provenance)))).toEqual(
    supplied.provenance
  );
  expect(() =>
    createResidentSuitabilityResultPlan(
      {...operation, weights: {label: 1}, columns: ['cost']},
      {...categorical, dictionaries: {label: {values: ['a'], ordered: true}}}
    )
  ).toThrow(/unordered prepared string dictionary/);
});

test('lineage round trips, rejects tampering and reports native derived dataset metadata', () => {
  const {provenance} = createResidentSuitabilityResultPlan(operation, metadata);
  const saved = parseResidentSuitabilityLineage(JSON.parse(JSON.stringify(provenance)));
  expect(saved).toEqual(provenance);
  expect(getResidentSuitabilityDerivedDataset(saved)).toEqual({
    type: 'suitability',
    sourceDataIds: ['data'],
    operationId: 'op'
  });
  const tamper = (change: Record<string, unknown>) =>
    expect(() => parseResidentSuitabilityLineage({...provenance, ...change})).toThrow();
  tamper({schemaVersion: 2});
  tamper({scorePolicy: 'other'});
  tamper({dataStandardization: 'bogus'});
  tamper({scoreColumn: 'missing'});
  tamper({indicators: [{column: 'cost', weight: 2}]});
  tamper({fields: provenance.fields.slice(1)});
  tamper({source: {datasetId: 'data', revision: -1}});
  expect(() => parseResidentSuitabilityLineage(null)).toThrow();
});

test('native Suitability requests snapshot resident configuration and ignore other maps', () => {
  const state = {
    visState: {
      suitabilities: [
        {
          ...operation,
          weightStandardization: 'other',
          dataStandardization: undefined,
          columns: ['id']
        },
        {...operation, id: 'ordinary', dataId: 'ordinary'}
      ]
    }
  };
  const resident = (id: string) => id === 'data';
  const request = getResidentSuitabilityRequest(state, runSuitability('op'), resident)!;
  // Native treats an unknown weight method as unstandardized weights and an absent data method as range.
  expect(request).toMatchObject({
    weightStandardization: 'raw',
    dataStandardization: 'range',
    columns: ['id']
  });
  expect(
    getResidentSuitabilityRequest(state, runSuitability('ordinary'), resident)
  ).toBeUndefined();
  expect(
    getResidentSuitabilityRequest(state, wrapTo('another-map', runSuitability('op')), resident)
  ).toBeUndefined();
  expect(
    getResidentSuitabilityRequest(state, wrapTo('compute', runSuitability('op')), resident)
  ).toBeDefined();
  const edited = {
    visState: {suitabilities: [{...state.visState.suitabilities[0], weights: {cost: 0.5}}]}
  };
  expect(
    getResidentSuitabilityRequestKey(
      getResidentSuitabilityRequest(edited, runSuitability('op'), resident)!
    )
  ).not.toBe(getResidentSuitabilityRequestKey(request));
  expect(
    getResidentSuitabilityStatus({
      type: RESIDENT_SUITABILITY_STATUS,
      id: 'op',
      requestId: 1,
      phase: 'running'
    })
  ).toMatchObject({id: 'op', phase: 'running'});
  expect(getResidentSuitabilityStatus({type: 'other'})).toBeUndefined();
});

test('missing-value policy: refuse by default, exclude and renormalize on request', () => {
  expect(() =>
    assertResidentSuitabilityStatistics(statistics({mixedMissingRows: 3, missingValues: 'refuse'}))
  ).toThrow(/choose a missing-value policy/);
  for (const missingValues of ['exclude', 'renormalize'] as const)
    expect(() =>
      assertResidentSuitabilityStatistics(statistics({mixedMissingRows: 3, missingValues}))
    ).not.toThrow();
  expect(createResidentSuitabilityResultPlan(operation, metadata).scoring.missingValuePolicy).toBe(
    'refuse'
  );
  expect(() =>
    createResidentSuitabilityResultPlan({...operation, missingValues: 'bogus' as never}, metadata)
  ).toThrow(/missing-value policy/);
  const renormalized = createResidentSuitabilityResultPlan(
    {...operation, missingValues: 'renormalize'},
    metadata
  );
  expect(renormalized.provenance.scorePolicy).toBe(
    'float32-composite-score-renormalized-defined-weights'
  );
  expect(renormalized.scoring.notes.join(' ')).toMatch(/rescaled to the full weight total/);
  expect(
    parseResidentSuitabilityLineage(JSON.parse(JSON.stringify(renormalized.provenance)))
  ).toEqual(renormalized.provenance);
  const excluded = createResidentSuitabilityResultPlan(
    {...operation, missingValues: 'exclude'},
    metadata
  );
  expect(excluded.provenance.missingValues).toBe('exclude');
  expect(() =>
    parseResidentSuitabilityLineage({
      ...renormalized.provenance,
      scorePolicy: 'float32-composite-score-aligned-missing-values-null'
    })
  ).toThrow();
});

test('the reference scorer renormalizes defined weights to the full total', () => {
  const columns = [
    [0, 10, null],
    [10, null, 20]
  ];
  const options = {dataStandardization: 'range', weightStandardization: 'normalize'} as const;
  // Range standardization maps column a to [0, 1, null] and b to [0, null, 1].
  expect(computeNativeSuitabilityScores(columns, [1, 3], options)).toEqual([0, null, null]);
  const renormalized = computeNativeSuitabilityScores(columns, [1, 3], {
    ...options,
    missingValues: 'renormalize'
  });
  expect(renormalized[0]).toBe(0);
  // Only column a is defined: 1 * 0.25 over defined weight 0.25, scaled to the total 1.
  expect(renormalized[1]).toBeCloseTo(1, 12);
  // Only column b is defined: 1 * 0.75 over defined weight 0.75, scaled to the total 1.
  expect(renormalized[2]).toBeCloseTo(1, 12);
});
