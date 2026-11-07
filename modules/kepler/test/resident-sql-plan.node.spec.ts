// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  CompiledGPUDataFrameAggregation,
  CompiledGPUDataFrameGlobalSort,
  CompiledGPUDataFrameGroupedAggregation,
  CompiledGPUDataFrameJoin,
  CompiledGPUDataFrameQuery
} from '@luma-contributors/graph/gpu-dataframe';
import {
  planResidentSqlBatchGathers,
  planResidentSqlMaterialization
} from '../../../examples/deck/kepler-compute/src/resident-sql-plan';

function makeFake(prototype: object, format = 'float32', rowCount = 3) {
  const vector = {format, data: [{}]};
  return Object.assign(Object.create(prototype), {
    table: {numRows: rowCount, schema: {fields: [{name: 'a'}]}, gpuVectors: {a: vector}},
    validity: {}
  });
}

test('batch gather planning preserves chunk boundaries and drops empty batches', () => {
  expect(planResidentSqlBatchGathers([2, 0, 5])).toEqual({
    gathers: [
      {sourceChunkIndex: 0, rowCount: 2},
      {sourceChunkIndex: 2, rowCount: 5}
    ],
    rowCount: 7
  });
  expect(planResidentSqlBatchGathers([0, 0]).rowCount).toBe(0);
});

test('materialization planning keeps aggregates, joins and sorts GPU-resident', () => {
  expect(planResidentSqlMaterialization(makeFake(CompiledGPUDataFrameQuery.prototype))).toEqual({
    kind: 'batches'
  });
  expect(planResidentSqlMaterialization(makeFake(CompiledGPUDataFrameJoin.prototype))).toEqual({
    kind: 'batches'
  });
  expect(
    planResidentSqlMaterialization(makeFake(CompiledGPUDataFrameGlobalSort.prototype))
  ).toEqual({kind: 'global-sort'});
  expect(
    planResidentSqlMaterialization(makeFake(CompiledGPUDataFrameAggregation.prototype, 'uint32', 1))
  ).toEqual({kind: 'dense', rowCount: 1});
  expect(
    planResidentSqlMaterialization(
      makeFake(CompiledGPUDataFrameGroupedAggregation.prototype, 'float32', 12)
    )
  ).toEqual({kind: 'dense', rowCount: 12});
});

test('materialization planning states the readback reason for unsupported columns', () => {
  const plan = planResidentSqlMaterialization(
    makeFake(CompiledGPUDataFrameQuery.prototype, 'float16')
  );
  expect(plan).toMatchObject({kind: 'readback'});
  expect((plan as {reason: string}).reason).toContain('"a"');
});
