// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DatasetOpsTable} from '@kepler.gl/table';
import {expect, test} from 'vitest';
import {
  createResidentJoinPlan,
  makeResidentJoinNumericKey,
  planResidentJoinPairs,
  RESIDENT_JOIN_NO_ROW,
  type ResidentJoinSourceMetadata,
  type ResidentJoinType
} from '../../../examples/deck/kepler-compute/src/resident-join-plan';

// Load the pinned native CPU implementation without its unrelated rendering barrels.
const {joinDatasets} = await import(
  // @ts-expect-error Direct implementation file has no declaration entry.
  new URL('../../../node_modules/@kepler.gl/table/dist/esm/dataset-ops/join.js', import.meta.url)
    .pathname
);
const {RowDataContainer} = await import(
  // @ts-expect-error Direct implementation file has no declaration entry.
  new URL('../../../node_modules/@kepler.gl/utils/dist/esm/row-data-container.js', import.meta.url)
    .pathname
);

const leftLabels = ['a', 'b', '', 'unused', 'c', 'z'];
const rightLabels = ['c', 'b', 'a', '', 'x'];
const leftKeys = ['b', null, 'a', '', 'b', 'z', 'c', undefined];
const rightKeys = ['a', 'b', null, 'a', 'x', 'b', '', 'c'];
const leftAmounts = [1, 2, 3, 4, 5, 6, 7, 8];
const rightWeights = [10, 20, 30, 40, 50, 60, 70, 80];

const fieldsFor = (key: string, value: string): DatasetOpsTable['fields'] => [
  {name: key, type: 'string', analyzerType: 'STRING', fieldIdx: 0, id: key, format: ''},
  {name: value, type: 'real', analyzerType: 'FLOAT', fieldIdx: 1, id: value, format: ''}
];
function table(id: string, keys: unknown[], values: number[], key: string, value: string) {
  return {
    id,
    label: id === 'right' ? 'Right data' : 'Left data',
    color: [1, 2, 3],
    fields: fieldsFor(key, value),
    dataContainer: new RowDataContainer({rows: keys.map((k, index) => [k, values[index]])})
  } as unknown as DatasetOpsTable;
}
function codes(keys: unknown[], labels: string[]) {
  const validity = new Uint32Array(keys.map(key => (key === null || key === undefined ? 0 : 1)));
  return {
    codes: new Uint32Array(keys.map(key => Math.max(0, labels.indexOf(key as string)))),
    validity
  };
}
function metadata(
  datasetId: string,
  label: string,
  key: string,
  value: string,
  labels: string[],
  rowCount: number
): ResidentJoinSourceMetadata {
  return {
    datasetId,
    label,
    revision: 1,
    rowCount,
    fields: fieldsFor(key, value),
    columns: {[key]: {format: 'uint32'}, [value]: {format: 'float32'}},
    dictionaries: {[key]: {values: labels, ordered: false}},
    nullCounts: {}
  };
}

test.each([
  'LEFT',
  'INNER',
  'FULL'
] as ResidentJoinType[])('%s row pairs, order and null/empty/duplicate keys match pinned native Join', joinType => {
  const left = codes(leftKeys, leftLabels);
  const right = codes(rightKeys, rightLabels);
  const pairs = planResidentJoinPairs({
    joinType,
    leftCodes: left.codes,
    leftValidity: left.validity,
    leftLabels,
    rightCodes: right.codes,
    rightValidity: right.validity,
    rightLabels,
    maximumRows: 1000
  });
  const native = joinDatasets(
    table('left', leftKeys, leftAmounts, 'key', 'amount'),
    table('right', rightKeys, rightWeights, 'key', 'weight'),
    {leftField: 'key', rightField: 'key', type: joinType, resultId: 'out', label: 'out'}
  );
  const actual = Array.from({length: pairs.rowCount}, (_, row) => {
    const leftRow = pairs.leftRows[row];
    const rightRow = pairs.rightRows[row];
    return [
      leftRow === RESIDENT_JOIN_NO_ROW ? null : (leftKeys[leftRow] ?? null),
      leftRow === RESIDENT_JOIN_NO_ROW ? null : leftAmounts[leftRow],
      rightRow === RESIDENT_JOIN_NO_ROW ? null : (rightKeys[rightRow] ?? null),
      rightRow === RESIDENT_JOIN_NO_ROW ? null : rightWeights[rightRow]
    ];
  });
  // Native emits undefined for an undefined left key; the resident path reports null.
  expect(actual).toEqual(native.data.rows.map((row: unknown[]) => row.map(value => value ?? null)));
});

test('output schema, collision naming and lineage match pinned native Join', () => {
  const leftMetadata = metadata('left', 'Left data', 'key', 'amount', leftLabels, 8);
  const rightMetadata = metadata('right', 'Right data', 'key', 'weight', rightLabels, 8);
  const plan = createResidentJoinPlan(
    {
      id: 'join',
      leftDataId: 'left',
      rightDataId: 'right',
      leftField: 'key',
      rightField: 'key',
      type: 'FULL',
      resultId: 'out',
      resultLabel: 'out'
    },
    leftMetadata,
    rightMetadata
  );
  const native = joinDatasets(
    table('left', leftKeys, leftAmounts, 'key', 'amount'),
    table('right', rightKeys, rightWeights, 'key', 'weight'),
    {
      leftField: 'key',
      rightField: 'key',
      type: 'FULL',
      resultId: 'out',
      label: 'out',
      operationId: 'join'
    }
  );
  expect(plan.provenance.fields).toEqual(native.data.fields);
  expect(plan.derivedDataset).toEqual(native.metadata.derivedDataset);
  expect(plan.columns.map(column => column.outputColumn)).toEqual([
    'key',
    'amount',
    'Right data_key',
    'Right data_weight'
  ]);
  // Selected columns keep the requested order and drop unknown names like native.
  const selected = createResidentJoinPlan(
    {
      id: 'join',
      leftDataId: 'left',
      rightDataId: 'right',
      leftField: 'key',
      rightField: 'key',
      type: 'LEFT',
      leftColumns: ['amount', 'missing'],
      rightColumns: ['weight'],
      resultId: 'out',
      resultLabel: 'out'
    },
    leftMetadata,
    rightMetadata
  );
  expect(selected.provenance.fields.map(field => field.name)).toEqual([
    'amount',
    'Right data_weight'
  ]);
});

test('admission rejects unsupported configurations with SQL guidance', () => {
  const leftMetadata = metadata('left', 'Left data', 'key', 'amount', leftLabels, 8);
  const rightMetadata = metadata('right', 'Right data', 'key', 'weight', rightLabels, 8);
  const operation = {
    id: 'join',
    leftDataId: 'left',
    rightDataId: 'right',
    leftField: 'key',
    rightField: 'key',
    type: 'LEFT',
    resultId: 'out',
    resultLabel: 'out'
  };
  expect(() =>
    createResidentJoinPlan({...operation, leftField: 'amount'}, leftMetadata, rightMetadata)
  ).toThrow(/both be string dictionaries.*SQL/);
  expect(() =>
    createResidentJoinPlan({...operation, type: 'RIGHT'}, leftMetadata, rightMetadata)
  ).toThrow(/RIGHT.*SQL/);
  expect(() =>
    createResidentJoinPlan({...operation, implementation: 'spatial'}, leftMetadata, rightMetadata)
  ).toThrow(/Spatial Join.*SQL/);
  expect(() =>
    createResidentJoinPlan(operation, {...leftMetadata, rowCount: 0}, rightMetadata)
  ).toThrow(/at least one row.*SQL/);
  expect(() =>
    createResidentJoinPlan(
      {...operation, leftColumns: [], rightColumns: []},
      leftMetadata,
      rightMetadata
    )
  ).toThrow(/at least one output column/);
  expect(() =>
    planResidentJoinPairs({
      joinType: 'FULL',
      leftCodes: new Uint32Array([0, 0, 0]),
      leftLabels: ['a'],
      rightCodes: new Uint32Array([0, 0, 0]),
      rightLabels: ['a'],
      maximumRows: 8
    })
  ).toThrow(/exceeds 8 rows.*SQL/);
});

test('admission has no side row or output row cap', () => {
  const leftMetadata = metadata('left', 'Left data', 'key', 'amount', leftLabels, 20_000_000);
  const rightMetadata = metadata('right', 'Right data', 'key', 'weight', rightLabels, 5_000_000);
  const plan = createResidentJoinPlan(
    {
      id: 'join',
      leftDataId: 'left',
      rightDataId: 'right',
      leftField: 'key',
      rightField: 'key',
      type: 'FULL',
      resultId: 'out',
      resultLabel: 'out'
    },
    leftMetadata,
    rightMetadata
  );
  expect(plan.left.rowCount).toBe(20_000_000);
  expect('outputRowLimit' in plan).toBe(false);
  // Oracle callers may omit the bound.
  expect(
    planResidentJoinPairs({
      joinType: 'INNER',
      leftCodes: new Uint32Array(600),
      leftLabels: ['a'],
      rightCodes: new Uint32Array(600),
      rightLabels: ['a']
    }).rowCount
  ).toBe(360_000);
});

test('numeric Int32 and integral Float32 keys match native String(value) semantics', () => {
  const numericLeft = [1, 2, 3, null, 2, 10, -0, 0];
  const numericRight = [2, 2, 3, 10, 1, null, 0, 7];
  const toWords = (values: (number | null)[], format: 'sint32' | 'float32') => {
    const words = new Uint32Array(values.length);
    for (const [index, value] of values.entries()) {
      if (format === 'float32') new Float32Array(words.buffer)[index] = value ?? 0;
      else new Int32Array(words.buffer)[index] = value ?? 0;
    }
    return words;
  };
  const validity = (values: (number | null)[]) =>
    new Uint32Array(values.map(value => (value === null ? 0 : 1)));
  const leftKey = makeResidentJoinNumericKey(
    toWords(numericLeft, 'sint32'),
    'sint32',
    validity(numericLeft)
  );
  const rightKey = makeResidentJoinNumericKey(
    toWords(numericRight, 'float32'),
    'float32',
    validity(numericRight)
  );
  for (const joinType of ['LEFT', 'INNER', 'FULL'] as ResidentJoinType[]) {
    const pairs = planResidentJoinPairs({
      joinType,
      leftCodes: leftKey.codes,
      leftValidity: validity(numericLeft),
      leftLabels: leftKey.labels,
      rightCodes: rightKey.codes,
      rightValidity: validity(numericRight),
      rightLabels: rightKey.labels,
      maximumRows: 1000
    });
    const native = joinDatasets(
      table('left', numericLeft, leftAmounts, 'key', 'amount'),
      table('right', numericRight, rightWeights, 'key', 'weight'),
      {leftField: 'key', rightField: 'key', type: joinType, resultId: 'out', label: 'out'}
    );
    const actual = Array.from({length: pairs.rowCount}, (_, row) => [
      pairs.leftRows[row] === RESIDENT_JOIN_NO_ROW ? null : leftAmounts[pairs.leftRows[row]],
      pairs.rightRows[row] === RESIDENT_JOIN_NO_ROW ? null : rightWeights[pairs.rightRows[row]]
    ]);
    expect(actual).toEqual(
      native.data.rows.map((row: unknown[]) => [row[1] ?? null, row[3] ?? null])
    );
  }
  expect(() =>
    makeResidentJoinNumericKey(new Uint32Array(new Float32Array([0.5]).buffer), 'float32')
  ).toThrow(/integers.*SQL/);
});
