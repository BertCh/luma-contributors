// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  canonicalizeResidentJoinKeys,
  planResidentJoinPairsOnGpu
} from '../../../examples/deck/kepler-compute/src/resident-join-gpu-reference';
import {
  createResidentJoinPlan,
  makeResidentJoinDictionaryKeyTables,
  planResidentJoinPairs,
  type ResidentJoinSourceMetadata
} from '../../../examples/deck/kepler-compute/src/resident-join-plan';
import {parseResidentJoinLineage} from '../../../examples/deck/kepler-compute/src/resident-join-lineage';
import {describeResidentJoinPolicy} from '../../../examples/deck/kepler-compute/src/analysis2-policy-binding';

const labels = ['a', 'b', 'c'];
const leftCodes = Uint32Array.from([0, 1, 2, 0]);
const rightCodes = Uint32Array.from([1, 0, 0, 1, 1]);

function runGpu(
  joinType: 'LEFT' | 'INNER' | 'FULL',
  duplicateRightMatches: 'all' | 'first' | 'last'
) {
  const tables = makeResidentJoinDictionaryKeyTables(labels, labels);
  const left = canonicalizeResidentJoinKeys({words: leftCodes, mode: 'string', table: tables.left});
  const right = canonicalizeResidentJoinKeys({
    words: rightCodes,
    mode: 'string',
    table: tables.right
  });
  return planResidentJoinPairsOnGpu({
    joinType,
    leftKey: left.key,
    leftOk: left.ok,
    rightKey: right.key,
    rightOk: right.ok,
    duplicateRightMatches
  });
}

test('first and last duplicate policies agree between the oracle and the GPU mirror', () => {
  for (const joinType of ['LEFT', 'INNER', 'FULL'] as const)
    for (const duplicateRightMatches of ['all', 'first', 'last'] as const) {
      const expected = planResidentJoinPairs({
        joinType,
        leftCodes,
        leftLabels: labels,
        rightCodes,
        rightLabels: labels,
        duplicateRightMatches
      });
      const actual = runGpu(joinType, duplicateRightMatches);
      expect([...actual.leftRows]).toEqual([...expected.leftRows]);
      expect([...actual.rightRows]).toEqual([...expected.rightRows]);
    }
  const first = runGpu('INNER', 'first');
  // Left rows a, b, a join the first duplicate right rows: a is right rows 1 and 2, b is 0, 3, 4.
  expect([...first.leftRows]).toEqual([0, 1, 3]);
  expect([...first.rightRows]).toEqual([1, 0, 1]);
  const last = runGpu('INNER', 'last');
  expect([...last.rightRows]).toEqual([2, 4, 2]);
});

function source(datasetId: string, label: string, names: string[]): ResidentJoinSourceMetadata {
  return {
    datasetId,
    label,
    revision: 1,
    rowCount: 3,
    fields: names.map(name => ({name, type: name === 'key' ? 'string' : 'real'})),
    columns: Object.fromEntries(
      names.map(name => [name, {format: name === 'key' ? 'uint32' : 'float32'}])
    ),
    dictionaries: {key: {values: ['a', 'b'], ordered: false}},
    nullCounts: {}
  };
}

test('right column suffix policy names collisions and can drop the right key', () => {
  const left = source('left', 'Left', ['key', 'value']);
  const right = source('right', 'Right', ['key', 'value', 'extra']);
  const operation = {
    id: 'join',
    leftDataId: 'left',
    rightDataId: 'right',
    leftField: 'key',
    rightField: 'key',
    type: 'LEFT',
    resultId: 'result',
    resultLabel: 'Result'
  };
  const native = createResidentJoinPlan(operation, left, right);
  expect(native.columns.map(column => column.outputColumn)).toEqual([
    'key',
    'value',
    'Right_key',
    'Right_value',
    'Right_extra'
  ]);
  const suffixed = createResidentJoinPlan({...operation, rightColumnSuffix: '_right'}, left, right);
  expect(suffixed.columns.map(column => column.outputColumn)).toEqual([
    'key',
    'value',
    'key_right',
    'value_right',
    'extra'
  ]);
  const dropped = createResidentJoinPlan(
    {...operation, rightColumnSuffix: '_right', dropRightKey: true, duplicateRightMatches: 'first'},
    left,
    right
  );
  expect(dropped.columns.map(column => column.outputColumn)).toEqual([
    'key',
    'value',
    'value_right',
    'extra'
  ]);
  expect(dropped.duplicateRightMatches).toBe('first');
  expect(describeResidentJoinPolicy(dropped)).toMatch(/first duplicate.*"_right"/);
  const saved = parseResidentJoinLineage(JSON.parse(JSON.stringify(dropped.provenance)));
  expect(saved).toMatchObject({duplicateRightMatches: 'first', rightColumnSuffix: '_right'});
  expect(() =>
    createResidentJoinPlan({...operation, duplicateRightMatches: 'any' as never}, left, right)
  ).toThrow(/duplicate policy/);
});

test('spatial join plans refuse unsupported declared coordinate systems', async () => {
  const {createResidentSpatialJoinPlan} = await import(
    '../../../examples/deck/kepler-compute/src/resident-spatial-join-plan'
  );
  const operation = {
    id: 'spatial',
    leftDataId: 'left',
    rightDataId: 'right',
    leftGeo: {latField: 'lat', lngField: 'lng'},
    rightGeo: {fieldName: 'geometry'},
    aggregations: {},
    resultId: 'result',
    resultLabel: 'Result',
    leftCrs: 'EPSG:3857',
    rightCrs: 'EPSG:4326'
  } as unknown as Parameters<typeof createResidentSpatialJoinPlan>[0];
  const metadata = (datasetId: string) =>
    ({
      datasetId,
      label: datasetId,
      revision: 1,
      rowCount: 2,
      fields: [],
      columns: {},
      dictionaries: {},
      nullCounts: {}
    }) as never;
  // EPSG:3857 against EPSG:4326 is reprojected on the GPU, so it passes the CRS check and only
  // fails later on the empty fixture metadata; other declared systems are refused up front.
  expect(() =>
    createResidentSpatialJoinPlan(operation, metadata('left'), metadata('right'))
  ).not.toThrow(/CRS|EPSG/);
  expect(() =>
    createResidentSpatialJoinPlan(
      {...operation, rightCrs: 'EPSG:2154'},
      metadata('left'),
      metadata('right')
    )
  ).toThrow(/Spatial Join runs on EPSG:4326, EPSG:3857, UTM/);
});
