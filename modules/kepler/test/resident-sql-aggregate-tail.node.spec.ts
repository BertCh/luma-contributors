// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  makeResidentSqlAggregateTailSql,
  splitResidentSqlAggregateTail
} from '../../../examples/deck/kepler-compute/src/resident-sql-aggregate-tail';
import {
  rewriteResidentSql,
  type ResidentSqlTableInfo
} from '../../../examples/deck/kepler-compute/src/resident-sql-strings';

test('aggregate ORDER BY and LIMIT split into a second stage', () => {
  const tail = splitResidentSqlAggregateTail(
    'SELECT region, COUNT(*) AS trip_count FROM joined GROUP BY region ORDER BY trip_count DESC LIMIT 3'
  )!;
  expect(tail.aggregateSql).toBe(
    'SELECT region, COUNT(*) AS trip_count FROM joined GROUP BY region'
  );
  expect(tail.orderBy).toEqual({column: 'trip_count', direction: 'DESC'});
  expect(tail.limit).toBe(3);
  expect(makeResidentSqlAggregateTailSql(tail, 'stage')).toBe(
    'SELECT * FROM stage ORDER BY trip_count DESC LIMIT 3'
  );
});

test('only output columns may be ordered; plain selects are not split', () => {
  expect(
    splitResidentSqlAggregateTail('SELECT g, COUNT(*) AS c FROM t GROUP BY g ORDER BY COUNT(*)')
  ).toBeNull();
  expect(splitResidentSqlAggregateTail('SELECT id FROM t ORDER BY id LIMIT 2')).toBeNull();
  expect(splitResidentSqlAggregateTail('SELECT SUM(x) AS s FROM t LIMIT 1;')?.limit).toBe(1);
});

test('mixed float and integer arithmetic promotes the integer column', () => {
  const table: ResidentSqlTableInfo = {
    name: 'trips',
    columns: {id: {format: 'sint32'}, fare: {format: 'float32'}, capacity: {format: 'sint32'}}
  };
  const rewrite = rewriteResidentSql('SELECT id, fare * capacity AS load FROM trips', name =>
    name === 'trips' ? table : undefined
  )!;
  expect(rewrite.derived.some(column => column.kind === 'cast')).toBe(true);
  expect(rewrite.sql).not.toContain('"capacity"');
});
