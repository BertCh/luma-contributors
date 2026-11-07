// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetOpsTable, GroupByDatasetConfig} from '@kepler.gl/table';
import {expect, test, vi} from 'vitest';
import {
  createResidentGroupByPlan,
  parseResidentGroupByProvenance,
  type ResidentGroupBySourceMetadata
} from '../../../examples/deck/kepler-compute/src/resident-group-by-plan';

// Load the pinned native CPU implementation without its unrelated rendering barrel.
// The aggregation function remains the actual installed Kepler implementation.
vi.mock('@kepler.gl/utils', async () =>
  vi.importActual(
    new URL('../../../node_modules/@kepler.gl/utils/dist/esm/aggregation.js', import.meta.url)
      .pathname
  )
);
const {groupByDataset} = await vi.importActual<
  Pick<typeof import('@kepler.gl/table'), 'groupByDataset'>
>(
  new URL(
    '../../../node_modules/@kepler.gl/table/dist/esm/dataset-ops/group-by.js',
    import.meta.url
  ).pathname
);
const {RowDataContainer} = await vi.importActual<{
  RowDataContainer: new (options: {rows: unknown[][]}) => DatasetOpsTable['dataContainer'];
}>(
  new URL('../../../node_modules/@kepler.gl/utils/dist/esm/row-data-container.js', import.meta.url)
    .pathname
);

const fields: DatasetOpsTable['fields'] = [
  {name: 'amount_sum', type: 'string', analyzerType: 'STRING', fieldIdx: 0, id: 'key', format: ''},
  {name: 'amount', type: 'real', analyzerType: 'FLOAT', fieldIdx: 1, id: 'amount', format: ''},
  {
    name: 'nullable',
    type: 'string',
    analyzerType: 'STRING',
    fieldIdx: 2,
    id: 'nullable',
    format: ''
  }
];
const rows = [
  ['b', 2, null],
  [null, 8, 'defined'],
  ['a', null, null],
  ['b', 6, null],
  [undefined, null, null]
];
const source: ResidentGroupBySourceMetadata = {
  datasetId: 'source',
  revision: 7,
  rowCount: rows.length,
  fields,
  columns: {amount_sum: {format: 'uint32'}, amount: {format: 'float32'}},
  dictionaries: {amount_sum: {values: ['a', 'b', 'unused'], ordered: false}},
  nullCounts: {amount_sum: [1, 1], amount: [0, 2]}
};
const operation = {
  id: 'operation',
  dataId: 'source',
  fieldName: 'amount_sum',
  aggregations: {amount: 'sum', nullable: 'count'},
  resultId: 'result',
  resultLabel: 'Grouped values'
};
function nativeResult(aggregations: GroupByDatasetConfig['aggregations']) {
  return groupByDataset(
    {
      id: source.datasetId,
      label: 'Source',
      color: [1, 2, 3],
      fields,
      dataContainer: new RowDataContainer({rows})
    },
    {
      fieldName: 'amount_sum',
      aggregations,
      resultId: operation.resultId,
      label: operation.resultLabel,
      operationId: operation.id
    }
  );
}

test('admitted schema and lineage match pinned native Group By, including collision and null count semantics', () => {
  const plan = createResidentGroupByPlan(operation, source);
  const native = nativeResult({amount: 'sum', nullable: 'count'});
  expect(plan.provenance.fields).toEqual(native.data.fields);
  expect(plan.derivedDataset).toEqual(native.metadata?.derivedDataset);
  expect(plan.resultId).toBe(native.info.id);
  expect(plan.resultLabel).toBe(native.info.label);
  expect(plan.settings).toEqual({
    keyColumn: 'amount_sum',
    aggregations: [
      {column: 'amount', operation: 'sum', name: 'amount_sum_1'},
      {column: 'nullable', operation: 'count', name: 'nullable_count'}
    ],
    population: 'all',
    numericalPolicy: 'float32'
  });
  // Native first source occurrence, one null/undefined group, count includes null measures.
  expect(native.data.rows).toEqual([
    ['b', 8, 2],
    [null, 8, 2],
    ['a', null, 1]
  ]);
  expect(plan.keyCapacity).toBe(4); // unused source labels still consume admission slots
  expect(plan.provenance.rowOrder).toBe('first-source-occurrence');
});

test('native distinct keys and average/minimum/maximum schemas remain compatible', () => {
  const distinct = createResidentGroupByPlan({...operation, aggregations: {}}, source);
  expect(distinct.provenance.fields).toEqual(nativeResult({}).data.fields);
  expect(nativeResult({}).data.rows).toEqual([['b'], [null], ['a']]);
  for (const technique of ['average', 'minimum', 'maximum'] as const) {
    const plan = createResidentGroupByPlan(
      {...operation, aggregations: {amount: technique}},
      source
    );
    expect(plan.provenance.fields).toEqual(nativeResult({amount: technique}).data.fields);
    expect(plan.settings.aggregations[0]?.operation).toBe(
      {average: 'mean', minimum: 'min', maximum: 'max'}[technique]
    );
  }
});

test('metadata admission rejects unsupported keys, methods and bounds before source access', () => {
  expect(() =>
    createResidentGroupByPlan({...operation, aggregations: {nullable: 'countUnique'}}, source)
  ).toThrow(/countUnique.*SQL/);
  expect(() =>
    createResidentGroupByPlan({...operation, aggregations: {nullable: 'sum'}}, source)
  ).toThrow(/Float32.*SQL/);
  expect(() => createResidentGroupByPlan(operation, {...source, rowCount: 67_108_865})).toThrow(
    /row count/
  );
  expect(() =>
    createResidentGroupByPlan(operation, {
      ...source,
      dictionaries: {amount_sum: {values: Array(4_194_304).fill('a'), ordered: false}}
    })
  ).toThrow(/4194304/);
  expect(() =>
    createResidentGroupByPlan(operation, {
      ...source,
      dictionaries: {amount_sum: {values: ['é'.repeat(16_777_217)], ordered: false}}
    })
  ).toThrow(/32 MiB/);
  expect(() =>
    createResidentGroupByPlan(operation, {...source, columns: {amount_sum: {format: 'float32'}}})
  ).toThrow(/string dictionary/);
});

test('saved provenance is detached, preserves source revision and rejects policy or schema drift', () => {
  const plan = createResidentGroupByPlan(operation, source);
  const saved = JSON.parse(JSON.stringify(plan.provenance));
  const parsed = parseResidentGroupByProvenance(saved);
  expect(parsed).toEqual(plan.provenance);
  saved.fields[0].name = 'mutated';
  expect(parsed.fields[0]?.name).toBe('amount_sum');
  expect(parsed.source).toEqual({datasetId: 'source', revision: 7});
  expect(() => parseResidentGroupByProvenance({...parsed, population: 'selected'})).toThrow(
    /policy/
  );
  expect(() =>
    parseResidentGroupByProvenance({...parsed, fields: [...parsed.fields].reverse()})
  ).toThrow(/schema/);
  expect(() =>
    parseResidentGroupByProvenance({
      ...parsed,
      aggregations: [{column: 'amount', technique: 'sum', name: 'amount_sum'}]
    })
  ).toThrow(/names/);
});
