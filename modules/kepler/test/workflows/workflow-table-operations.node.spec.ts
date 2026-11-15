// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import type {CatalogColumnType, CatalogOutputTable} from '../../src/analysis/catalog-geojson';
import {
  getWorkflowColumnNames,
  getWorkflowFeatures,
  makeWorkflowData,
  makeWorkflowDataFromFeatures
} from '../../src/workflows/workflow-data';
import {getWorkflowTableOperations} from '../../src/workflows/workflow-table-operations';
import {
  WorkflowError,
  type WorkflowData,
  type WorkflowOperationOutput,
  type WorkflowRunContext
} from '../../src/workflows/workflow-types';
import {makeFixtureHost, makeFixturePoints} from './workflow-fixtures';

const OPERATIONS = getWorkflowTableOperations();
const CONTEXT: WorkflowRunContext = {
  device: null,
  engine: 'cpu',
  host: makeFixtureHost(),
  signal: new AbortController().signal,
  workflowId: 'table-spec',
  stepId: 'step'
};
const POINT = '{"type":"Point","coordinates":[1,2]}';

function makeData(
  columns: CatalogOutputTable['columns'],
  columnTypes: CatalogOutputTable['columnTypes'],
  options: {geometryColumn?: string; identityColumn?: string} = {}
): WorkflowData {
  return makeWorkflowData(
    {
      id: 'input',
      label: 'Input',
      rowCount: Object.values(columns)[0]?.length ?? 0,
      columns,
      columnTypes,
      ...(options.geometryColumn !== undefined ? {geometryColumn: options.geometryColumn} : {})
    },
    options.identityColumn !== undefined ? {identityColumn: options.identityColumn} : {}
  );
}
function getOperation(name: string) {
  const operation = OPERATIONS.find(candidate => candidate.id === `table-${name}`);
  if (!operation) throw new Error(`Missing operation ${name}`);
  return operation;
}

// Every successful run checks both value immutability and array independence for every input.
async function run(
  name: string,
  inputs: Record<string, WorkflowData>,
  raw: Record<string, unknown> = {}
): Promise<WorkflowOperationOutput> {
  const snapshot = structuredClone(inputs);
  const operation = getOperation(name);
  const result = await operation.run(CONTEXT, inputs, operation.normalizeParameters(raw));
  expect(inputs).toEqual(snapshot);
  for (const output of Object.values(result.outputs)) {
    for (const input of Object.values(inputs)) {
      expect(output.table.columns).not.toBe(input.table.columns);
      expect(output.table.columnTypes).not.toBe(input.table.columnTypes);
      for (const values of Object.values(output.table.columns)) {
        for (const sourceValues of Object.values(input.table.columns))
          expect(values).not.toBe(sourceValues);
      }
    }
    expect(Number.isInteger(output.table.rowCount)).toBe(true);
    expect(output.table.rowCount).toBeGreaterThanOrEqual(0);
    for (const values of Object.values(output.table.columns)) {
      expect(values.length).toBe(output.table.rowCount);
    }
    if (Object.keys(output.table.columns).length === 0) {
      expect(['select-columns', 'drop-geometry']).toContain(name);
      expect(output.table.rowCount).toBe(inputs['rows'].table.rowCount);
    }
  }
  expect(result.engine).toBe('cpu-reference');
  expect(result.summary).toEqual({
    inputRows: Object.values(inputs).reduce((total, data) => total + data.table.rowCount, 0),
    outputRows: Object.values(result.outputs)[0].table.rowCount
  });
  return result;
}

function expectColumns(
  result: WorkflowOperationOutput,
  columns: CatalogOutputTable['columns'],
  types: CatalogOutputTable['columnTypes']
): WorkflowData {
  const output = Object.values(result.outputs)[0];
  expect(Object.keys(output.table.columns)).toEqual(Object.keys(columns));
  expect(output.table.columns).toEqual(columns);
  expect(output.table.columnTypes).toEqual(types);
  return output;
}

async function expectRefusal(
  name: string,
  inputs: Record<string, WorkflowData>,
  raw: Record<string, unknown>,
  parameter: string,
  message?: string
): Promise<void> {
  const snapshot = structuredClone(inputs);
  try {
    const operation = getOperation(name);
    await operation.run(CONTEXT, inputs, operation.normalizeParameters(raw));
    expect.fail('Expected a WorkflowError');
  } catch (error) {
    expect(error).toBeInstanceOf(WorkflowError);
    expect(error).toMatchObject({code: 'parameter-out-of-range', details: {parameter}});
    if (message) expect((error as WorkflowError).message).toBe(message);
  }
  expect(inputs).toEqual(snapshot);
}

describe('workflow table operations', () => {
  test('declares nine bounded builtins with the contracted ports', () => {
    expect(OPERATIONS.map(operation => operation.id)).toEqual([
      'table-join-attributes',
      'table-filter-rows',
      'table-calculate-column',
      'table-select-columns',
      'table-sort-rows',
      'table-summarize',
      'table-concatenate',
      'table-points-from-columns',
      'table-drop-geometry'
    ]);
    expect(OPERATIONS.map(operation => operation.outputs[0].name)).toEqual([
      'joined',
      'filtered',
      'calculated',
      'selected',
      'sorted',
      'summary',
      'combined',
      'points',
      'table'
    ]);
    for (const operation of OPERATIONS) {
      expect(operation).toMatchObject({
        category: 'table',
        origin: 'builtin',
        engine: 'cpu-reference',
        version: '1'
      });
      for (const port of operation.inputs)
        expect(port).toMatchObject({
          accepts: ['point', 'line', 'polygon', 'features', 'table'],
          maximumRows: 100000,
          maximumVertices: 2000000
        });
      expect(operation.outputs[0].maximumRows).toBe(100000);
    }
  });

  test('joins numeric and string keys, first duplicate wins, null never matches', async () => {
    const target = makeData(
      {id: [3, 4, null, 9], geometry: [POINT, POINT, POINT, POINT]},
      {id: 'int32', geometry: 'string'},
      {identityColumn: 'id', geometryColumn: 'geometry'}
    );
    const attributes = makeData(
      {
        source_id: ['3', '3', '4', null],
        score: [8, 99, 7, 100],
        geometry: [POINT, POINT, POINT, POINT]
      },
      {source_id: 'string', score: 'float64', geometry: 'string'},
      {geometryColumn: 'geometry'}
    );
    const result = await run('join-attributes', {target, attributes});
    const output = expectColumns(
      result,
      {id: [3, 4, null, 9], geometry: [POINT, POINT, POINT, POINT], score: [8, 7, null, null]},
      {id: 'int32', geometry: 'string', score: 'float64'}
    );
    expect(output).toMatchObject({
      identityColumn: 'id',
      kind: 'point',
      table: {geometryColumn: 'geometry'}
    });
    expect(result.diagnostics).toEqual([
      {
        code: 'duplicate-attribute-key',
        severity: 'warning',
        count: 1,
        message: 'Join attributes: 1 duplicate attribute keys; the first row wins.'
      },
      {
        code: 'unmatched-target-rows',
        severity: 'warning',
        count: 2,
        message: 'Join attributes: 2 target rows have no matching attributes.'
      }
    ]);
    const dropped = await run(
      'join-attributes',
      {target, attributes},
      {unmatched: 'drop', prefix: 'added_', columns: 'score, geometry'}
    );
    expectColumns(
      dropped,
      {id: [3, 4], geometry: [POINT, POINT], added_score: [8, 7]},
      {id: 'int32', geometry: 'string', added_score: 'float64'}
    );
    expect(dropped.diagnostics.map(diagnostic => [diagnostic.code, diagnostic.count])).toEqual([
      ['duplicate-attribute-key', 1],
      ['unmatched-target-rows', 2]
    ]);
  });

  test('joins by retained identity after filtering without using row positions', async () => {
    const original = makeFixturePoints(4);
    const filtered = (await run('filter-rows', {rows: original}, {expression: 'value >= 3'}))
      .outputs['filtered'];
    const attributes = makeData(
      {source_id: ['p3', 'p0', 'p2'], score: [30, 0, 20]},
      {source_id: 'string', score: 'int32'}
    );
    const result = await run('join-attributes', {target: filtered, attributes});
    expect(result.outputs['joined'].table.columns['point_id']).toEqual(['p2', 'p3']);
    expect(result.outputs['joined'].table.columns['score']).toEqual([20, 30]);
    expect(result.outputs['joined'].identityColumn).toBe('point_id');
  });

  test('joins 50000 rows within two seconds using a hash lookup', async () => {
    const target = makeData(
      {id: Array.from({length: 50000}, (_, index) => index)},
      {id: 'int32'},
      {identityColumn: 'id'}
    );
    const attributes = makeData(
      {
        source_id: Array.from({length: 50000}, (_, index) => String(49999 - index)),
        score: Array.from({length: 50000}, (_, index) => (49999 - index) * 2)
      },
      {source_id: 'string', score: 'int32'}
    );
    const start = performance.now();
    const result = await run('join-attributes', {target, attributes});
    const elapsed = performance.now() - start;
    expect(result.outputs['joined'].table.columns['score']).toEqual(
      Array.from({length: 50000}, (_, index) => index * 2)
    );
    expect(elapsed).toBeLessThan(2000);
  });

  test('join refuses missing identity and collisions; prefix resolves collisions', async () => {
    const target = makeData({id: [1], score: [2]}, {id: 'int32', score: 'int32'});
    const attributes = makeData({source_id: [1], score: [5]}, {source_id: 'int32', score: 'int32'});
    await expectRefusal('join-attributes', {target, attributes}, {}, 'targetKey');
    await expectRefusal('join-attributes', {target, attributes}, {targetKey: 'id'}, 'prefix');
    expectColumns(
      await run('join-attributes', {target, attributes}, {targetKey: 'id', prefix: 'new_'}),
      {id: [1], score: [2], new_score: [5]},
      {id: 'int32', score: 'int32', new_score: 'int32'}
    );
  });

  test('filter preserves types, truthy rows and null diagnostics', async () => {
    const rows = makeData(
      {
        id: ['a', 'b', 'c', 'd', 'e'],
        value: [2, 0, null, -1, 1],
        geometry: [POINT, POINT, POINT, POINT, POINT]
      },
      {id: 'string', value: 'float64', geometry: 'string'},
      {identityColumn: 'id', geometryColumn: 'geometry'}
    );
    const result = await run('filter-rows', {rows}, {expression: 'value'});
    expectColumns(
      result,
      {id: ['a', 'd', 'e'], value: [2, -1, 1], geometry: [POINT, POINT, POINT]},
      {id: 'string', value: 'float64', geometry: 'string'}
    );
    expect(result.diagnostics).toEqual([
      {
        code: 'null-condition',
        severity: 'info',
        count: 1,
        message: 'Filter rows: Expression evaluated to null for 1 rows.'
      }
    ]);
    expect(result.outputs['filtered'].identityColumn).toBe('id');
  });

  test.each([
    ['value * 2', [4, null, 0], 'float64'],
    ['value > 0', [true, false, false], 'boolean'],
    ['if(isnull(value), null, if(value, "yes", 3))', ['yes', null, '3'], 'string'],
    ['null', [null, null, null], 'float64']
  ] as const)('calculate infers type for %s', async (expression, values, type) => {
    const rows = makeData(
      {id: [1, 2, 3], value: [2, null, 0]},
      {id: 'int32', value: 'float64'},
      {identityColumn: 'id'}
    );
    const result = await run('calculate-column', {rows}, {columnName: 'derived', expression});
    expectColumns(
      result,
      {id: [1, 2, 3], value: [2, null, 0], derived: [...values]},
      {id: 'int32', value: 'float64', derived: type}
    );
    expect(result.diagnostics).toEqual([]);
    expect(result.outputs['calculated'].identityColumn).toBe('id');
  });

  test('calculate replaces a column with counted diagnostic and protects geometry', async () => {
    const rows = makeFixturePoints(2);
    const result = await run(
      'calculate-column',
      {rows},
      {columnName: 'value', expression: 'value + 1'}
    );
    expectColumns(result, {...rows.table.columns, value: [1, 2.5]}, rows.table.columnTypes);
    expect(result.diagnostics).toEqual([
      {
        code: 'column-replaced',
        severity: 'info',
        count: 2,
        message: 'Calculate column: Column "value" was replaced for 2 rows.'
      }
    ]);
    await expectRefusal(
      'calculate-column',
      {rows},
      {columnName: 'geometry', expression: '1'},
      'columnName'
    );
  });

  test('select keeps geometry, renames simultaneously, follows identity, and drops identity', async () => {
    const rows = makeFixturePoints(2);
    const result = await run(
      'select-columns',
      {rows},
      {columns: 'point_id, value', mode: 'keep', rename: 'point_id=key, value=number'}
    );
    expectColumns(
      result,
      {key: ['p0', 'p1'], number: [0, 1.5], geometry: [...rows.table.columns['geometry']]},
      {key: 'string', number: 'float64', geometry: 'string'}
    );
    expect(result.outputs['selected'].identityColumn).toBe('key');
    const dropped = await run(
      'select-columns',
      {rows},
      {mode: 'drop', columns: 'point_id, geometry'}
    );
    expectColumns(
      dropped,
      {value: [0, 1.5], geometry: [...rows.table.columns['geometry']]},
      {value: 'float64', geometry: 'string'}
    );
    expect(dropped.outputs['selected'].identityColumn).toBeUndefined();
    expectColumns(
      await run('select-columns', {rows}, {}),
      {geometry: [...rows.table.columns['geometry']]},
      {geometry: 'string'}
    );
    const swapped = await run(
      'select-columns',
      {rows},
      {columns: 'value, point_id', rename: 'point_id=value,value=point_id'}
    );
    expectColumns(
      swapped,
      {value: ['p0', 'p1'], point_id: [0, 1.5], geometry: [...rows.table.columns['geometry']]},
      {value: 'string', point_id: 'float64', geometry: 'string'}
    );
  });

  test.each([
    'a',
    'a=',
    '=b',
    'a=b=c',
    'a=b,',
    'value=a,value=b'
  ])('select refuses malformed rename %s', async rename => {
    await expectRefusal('select-columns', {rows: makeFixturePoints(2)}, {rename}, 'rename');
  });
  test('select does not transfer identity from a dropped column to an unrelated column', async () => {
    const rows = makeFixturePoints(1);
    const result = await run(
      'select-columns',
      {rows},
      {columns: 'value', rename: 'point_id=value'}
    );
    expect(result.outputs['selected'].identityColumn).toBeUndefined();
    expectColumns(
      result,
      {value: [0], geometry: [...rows.table.columns['geometry']]},
      {value: 'float64', geometry: 'string'}
    );
  });
  test('select refuses unknown names, geometry renaming and duplicate result names', async () => {
    const rows = makeFixturePoints(1);
    for (const rename of ['missing=new', 'geometry=shape', 'value=point_id', 'value=geometry']) {
      await expectRefusal('select-columns', {rows}, {mode: 'drop', rename}, 'rename');
    }
  });

  test.each([
    ['ascending', [4, 1, 2, 0, 3, 5, 6]],
    ['descending', [0, 1, 2, 4, 3, 5, 6]]
  ])('sort is numeric and stable with invalids last in %s order', async (order, expected) => {
    const rows = makeData(
      {id: [0, 1, 2, 3, 4, 5, 6], value: [10, 2, 2, null, -1, NaN, Infinity]},
      {id: 'int32', value: 'float64'},
      {identityColumn: 'id'}
    );
    const result = await run('sort-rows', {rows}, {column: 'value', order});
    const indices = expected as number[];
    expectColumns(
      result,
      {id: indices, value: indices.map(index => rows.table.columns['value'][index])},
      {id: 'int32', value: 'float64'}
    );
    expect(result.outputs['sorted'].identityColumn).toBe('id');
    const limited = await run('sort-rows', {rows}, {column: 'value', order, limit: 2});
    expect(limited.outputs['sorted'].table.columns['id']).toEqual(indices.slice(0, 2));
  });
  test.each([
    [['a', 'Z', 'ä', 'A'], 'string', [3, 1, 0, 2]],
    [[true, false, false, true], 'boolean', [1, 2, 0, 3]]
  ] as const)('sort uses code units or boolean order', async (values, type, indices) => {
    const rows = makeData({id: [0, 1, 2, 3], value: [...values]}, {id: 'int32', value: type});
    expectColumns(
      await run('sort-rows', {rows}, {column: 'value'}),
      {id: [...indices], value: indices.map(index => values[index])},
      {id: 'int32', value: type}
    );
  });

  test('summarise groups in first-seen order and ignores non-finite values', async () => {
    const rows = makeData(
      {group: ['b', 'a', 'b', 'c', 'a', 'c'], value: [2, null, 4, Infinity, 8, NaN]},
      {group: 'string', value: 'float64'}
    );
    const result = await run('summarize', {rows}, {groupBy: 'group', valueColumn: 'value'});
    const output = expectColumns(
      result,
      {
        group: ['b', 'a', 'c'],
        row_count: [2, 2, 2],
        value_count: [2, 1, 0],
        value_sum: [6, 8, null],
        value_mean: [3, 8, null],
        value_minimum: [2, 8, null],
        value_maximum: [4, 8, null]
      },
      {
        group: 'string',
        row_count: 'int32',
        value_count: 'int32',
        value_sum: 'float64',
        value_mean: 'float64',
        value_minimum: 'float64',
        value_maximum: 'float64'
      }
    );
    expect(output.kind).toBe('table');
    expect(output.table.geometryColumn).toBeUndefined();
    expectColumns(await run('summarize', {rows}, {}), {row_count: [6]}, {row_count: 'int32'});
  });
  test('summarise compensates cancellation and avoids intermediate overflow', async () => {
    const rows = makeData({value: [1e16, 1, -1e16]}, {value: 'float64'});
    const result = await run('summarize', {rows}, {valueColumn: 'value'});
    expect(result.outputs['summary'].table.columns['value_sum']).toEqual([1]);
    expect(result.outputs['summary'].table.columns['value_mean']).toEqual([1 / 3]);
    const large = await run(
      'summarize',
      {rows: makeData({value: [1e308, 1e308, -1e308]}, {value: 'float64'})},
      {valueColumn: 'value'}
    );
    expect(large.outputs['summary'].table.columns['value_sum']).toEqual([1e308]);
    expect(large.outputs['summary'].table.columns['value_mean']).toEqual([1e308 / 3]);
  });
  test.each([
    [[1e308, 1, -1e308], 1],
    [[1e308, 1e308, 1, -1e308, -1e308], 1],
    [[1e308, 1e-308, -1e308], 1e-308],
    [[Number.MAX_VALUE, -Number.MAX_VALUE, Number.MIN_VALUE], Number.MIN_VALUE]
  ])('summarise retains tiny residuals of extreme finite values', async (values, sum) => {
    const result = await run(
      'summarize',
      {rows: makeData({value: values as number[]}, {value: 'float64'})},
      {valueColumn: 'value'}
    );
    expect(result.outputs['summary'].table.columns['value_sum']).toEqual([sum]);
  });
  test('summarise handles empty input and composite null groups without collisions', async () => {
    const empty = makeData({value: []}, {value: 'float64'});
    const result = await run('summarize', {rows: empty}, {valueColumn: 'value'});
    expectColumns(
      result,
      {
        row_count: [0],
        value_count: [0],
        value_sum: [null],
        value_mean: [null],
        value_minimum: [null],
        value_maximum: [null]
      },
      {
        row_count: 'int32',
        value_count: 'int32',
        value_sum: 'float64',
        value_mean: 'float64',
        value_minimum: 'float64',
        value_maximum: 'float64'
      }
    );
    expectColumns(
      await run('summarize', {rows: empty}, {groupBy: 'value'}),
      {value: [], row_count: []},
      {value: 'float64', row_count: 'int32'}
    );
    const rows = makeData(
      {first: [null, 'null', 'a,b', 'a'], second: ['x', 'x', 'c', 'b,c']},
      {first: 'string', second: 'string'},
      {identityColumn: 'first'}
    );
    const grouped = await run('summarize', {rows}, {groupBy: 'first, second'});
    expectColumns(
      grouped,
      {...rows.table.columns, row_count: [1, 1, 1, 1]},
      {...rows.table.columnTypes, row_count: 'int32'}
    );
    expect(grouped.outputs['summary'].identityColumn).toBe('first');
  });
  test('summarise refuses a nonnumeric value or aggregate name collision', async () => {
    const rows = makeData({label: ['a'], row_count: [3]}, {label: 'string', row_count: 'int32'});
    await expectRefusal('summarize', {rows}, {valueColumn: 'label'}, 'valueColumn');
    await expectRefusal('summarize', {rows}, {groupBy: 'row_count'}, 'groupBy');
  });

  test('concatenate unions columns in first-seen order, converts conflicts, fills missing values', async () => {
    const first = makeData(
      {id: [1], geometry: [POINT], value: [2], flag: [true]},
      {id: 'int32', geometry: 'string', value: 'float64', flag: 'boolean'},
      {geometryColumn: 'geometry', identityColumn: 'id'}
    );
    const second = makeData(
      {id: ['two'], value: ['three'], extra: [8]},
      {id: 'string', value: 'string', extra: 'int32'},
      {identityColumn: 'id'}
    );
    const result = await run('concatenate', {first, second});
    expectColumns(
      result,
      {
        id: ['1', 'two'],
        geometry: [POINT, null],
        value: ['2', 'three'],
        flag: [true, null],
        extra: [null, 8],
        source_table: ['first', 'second']
      },
      {
        id: 'string',
        geometry: 'string',
        value: 'string',
        flag: 'boolean',
        extra: 'int32',
        source_table: 'string'
      }
    );
    expect(result.outputs['combined']).toMatchObject({
      identityColumn: 'id',
      kind: 'point',
      table: {geometryColumn: 'geometry'}
    });
  });
  test('concatenate merges different geometry names and preserves second-only geometry', async () => {
    const first = makeData(
      {id: [1], shape: [POINT]},
      {id: 'int32', shape: 'string'},
      {geometryColumn: 'shape'}
    );
    const second = makeData(
      {geometry: [POINT], id: [2]},
      {geometry: 'string', id: 'int32'},
      {geometryColumn: 'geometry'}
    );
    const combined = expectColumns(
      await run('concatenate', {first, second}),
      {id: [1, 2], shape: [POINT, POINT], source_table: ['first', 'second']},
      {id: 'int32', shape: 'string', source_table: 'string'}
    );
    expect(getWorkflowFeatures(combined).features.map(feature => feature.geometry)).toEqual([
      JSON.parse(POINT),
      JSON.parse(POINT)
    ]);
    const table = makeData({id: [3]}, {id: 'int32'});
    expectColumns(
      await run('concatenate', {first: table, second}),
      {id: [3, 2], geometry: [null, POINT], source_table: ['first', 'second']},
      {id: 'int32', geometry: 'string', source_table: 'string'}
    );
  });
  test('concatenate enforces combined capacity and permits the exact limit', async () => {
    const first = makeData({value: Array<number>(50000).fill(1)}, {value: 'int32'});
    const second = makeData({value: Array<number>(50001).fill(2)}, {value: 'int32'});
    await expect(
      getOperation('concatenate').run(CONTEXT, {first, second}, {})
    ).rejects.toMatchObject({code: 'capacity-exceeded'});
    expect(
      (await run('concatenate', {first, second: first})).outputs['combined'].table.rowCount
    ).toBe(100000);
  });

  test('points retain bounds and drop null, non-finite and out-of-range coordinates', async () => {
    const rows = makeData(
      {
        id: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        longitude: [-180, 180, null, Infinity, 181, 0, 0, 0],
        latitude: [-90, 90, 0, 0, 0, -91, NaN, 12]
      },
      {id: 'string', longitude: 'float64', latitude: 'float64'},
      {identityColumn: 'id'}
    );
    const result = await run(
      'points-from-columns',
      {rows},
      {longitudeColumn: 'longitude', latitudeColumn: 'latitude'}
    );
    const output = expectColumns(
      result,
      {
        id: ['a', 'b', 'h'],
        longitude: [-180, 180, 0],
        latitude: [-90, 90, 12],
        geometry: [
          '{"type":"Point","coordinates":[-180,-90]}',
          '{"type":"Point","coordinates":[180,90]}',
          '{"type":"Point","coordinates":[0,12]}'
        ]
      },
      {id: 'string', longitude: 'float64', latitude: 'float64', geometry: 'string'}
    );
    expect(output).toMatchObject({
      kind: 'point',
      identityColumn: 'id',
      table: {geometryColumn: 'geometry'}
    });
    expect(result.diagnostics).toEqual([
      {
        code: 'invalid-coordinate',
        severity: 'warning',
        count: 5,
        message: 'Points from columns: 5 rows have null, non-finite or out-of-range coordinates.'
      }
    ]);
  });
  test('points replace existing geometry and protect coordinate types and distinctness', async () => {
    const rows = makeData(
      {longitude: [1], latitude: [2], shape: [POINT]},
      {longitude: 'int32', latitude: 'int32', shape: 'string'},
      {geometryColumn: 'shape'}
    );
    expectColumns(
      await run(
        'points-from-columns',
        {rows},
        {longitudeColumn: 'longitude', latitudeColumn: 'latitude'}
      ),
      {longitude: [1], latitude: [2], geometry: [POINT]},
      {longitude: 'int32', latitude: 'int32', geometry: 'string'}
    );
    await expectRefusal(
      'points-from-columns',
      {rows},
      {longitudeColumn: 'shape', latitudeColumn: 'latitude'},
      'longitudeColumn'
    );
    await expectRefusal(
      'points-from-columns',
      {rows},
      {longitudeColumn: 'longitude', latitudeColumn: 'shape'},
      'latitudeColumn'
    );
    await expectRefusal(
      'points-from-columns',
      {rows},
      {longitudeColumn: 'latitude', latitudeColumn: 'latitude'},
      'latitudeColumn'
    );
  });
  test('drop geometry retains remaining columns and identity and works without geometry', async () => {
    const rows = makeFixturePoints(2);
    const result = await run('drop-geometry', {rows});
    const output = expectColumns(
      result,
      {point_id: ['p0', 'p1'], value: [0, 1.5]},
      {point_id: 'string', value: 'float64'}
    );
    expect(output).toMatchObject({kind: 'table', identityColumn: 'point_id'});
    expect(output.table.geometryColumn).toBeUndefined();
    expectColumns(
      await run('drop-geometry', {rows: output}),
      output.table.columns,
      output.table.columnTypes
    );
  });

  test.each([
    ['join-attributes', 'targetKey', {targetKey: 'missing'}],
    ['join-attributes', 'attributesKey', {attributesKey: 'missing'}],
    ['join-attributes', 'columns', {columns: 'missing'}],
    ['filter-rows', 'expression', {expression: 'missing > 1'}],
    ['calculate-column', 'expression', {columnName: 'new', expression: 'missing'}],
    ['select-columns', 'columns', {columns: 'missing'}],
    ['sort-rows', 'column', {column: 'missing'}],
    ['summarize', 'groupBy', {groupBy: 'missing'}],
    ['summarize', 'valueColumn', {valueColumn: 'missing'}],
    [
      'points-from-columns',
      'longitudeColumn',
      {longitudeColumn: 'missing', latitudeColumn: 'value'}
    ],
    ['points-from-columns', 'latitudeColumn', {longitudeColumn: 'value', latitudeColumn: 'missing'}]
  ] as const)('%s refuses unknown column in %s with parameter detail', async (name, parameter, raw) => {
    const rows = makeFixturePoints(1);
    const attributes = makeData(
      {source_id: ['p0'], score: [2]},
      {source_id: 'string', score: 'int32'}
    );
    await expectRefusal(name, {rows, target: rows, attributes}, raw, parameter);
  });
  test('missing column message is case-sensitive, labelled, and bounded to twelve names', async () => {
    const rows = makeData({a: [1], b: [2], c: [3]}, {a: 'int32', b: 'int32', c: 'int32'});
    await expectRefusal(
      'sort-rows',
      {rows},
      {column: 'A'},
      'column',
      'Sort rows: Column names column "A", which the input does not have. Columns: a, b, c.'
    );
    const wide = makeData(
      Object.fromEntries(Array.from({length: 15}, (_, index) => [`column${index}`, [1]])),
      Object.fromEntries(
        Array.from({length: 15}, (_, index) => [`column${index}`, 'int32' as CatalogColumnType])
      )
    );
    await expectRefusal(
      'sort-rows',
      {rows: wide},
      {column: 'missing'},
      'column',
      `Sort rows: Column names column "missing", which the input does not have. Columns: ${Array.from({length: 12}, (_, index) => `column${index}`).join(', ')}.`
    );
  });
  test.each([
    'filter-rows',
    'calculate-column'
  ])('%s parses and rejects blank and invalid expressions during normalization', name => {
    for (const expression of ['', ' ', 'value +', 'unknownFunction(value)']) {
      expect(() =>
        getOperation(name).normalizeParameters({
          expression,
          ...(name === 'calculate-column' ? {columnName: 'new'} : {})
        })
      ).toThrowError(WorkflowError);
      try {
        getOperation(name).normalizeParameters({
          expression,
          ...(name === 'calculate-column' ? {columnName: 'new'} : {})
        });
      } catch (error) {
        expect(error).toMatchObject({
          code: 'parameter-out-of-range',
          details: {parameter: 'expression'}
        });
        expect((error as WorkflowError).message).toContain('Expression:');
      }
    }
  });
  test.each([
    ['calculate-column', 'columnName', {columnName: '', expression: '1'}],
    ['calculate-column', 'columnName', {columnName: ' ', expression: '1'}],
    ['calculate-column', 'columnName', {columnName: 'a'.repeat(65), expression: '1'}],
    ['select-columns', 'rename', {rename: 'a'.repeat(513)}],
    ['select-columns', 'mode', {mode: 'invalid'}],
    ['join-attributes', 'unmatched', {unmatched: 'invalid'}],
    ['join-attributes', 'prefix', {prefix: 'a'.repeat(65)}],
    ['sort-rows', 'order', {order: 'invalid'}],
    ['sort-rows', 'limit', {limit: -1}],
    ['sort-rows', 'limit', {limit: 100001}],
    ['sort-rows', 'limit', {limit: 1.5}]
  ] as const)('%s refuses parameter %s outside limits', async (name, parameter, raw) => {
    const rows = makeFixturePoints(1);
    await expectRefusal(name, {rows, target: rows, attributes: rows}, raw, parameter);
  });
  test('calculate accepts 64 characters and sort coerces a whole-number string', async () => {
    const rows = makeFixturePoints(1);
    const name = 'a'.repeat(64);
    expect(
      (await run('calculate-column', {rows}, {columnName: name, expression: '1'})).outputs[
        'calculated'
      ].table.columns[name]
    ).toEqual([1]);
    expect(
      getOperation('sort-rows').normalizeParameters({column: 'value', limit: '2'})['limit']
    ).toBe(2);
  });
  test.each(OPERATIONS)('$id checks cancellation before inputs and parameters', async operation => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      operation.run({...CONTEXT, signal: controller.signal}, {}, {})
    ).rejects.toMatchObject({code: 'cancelled', details: {stepId: 'step'}});
  });
  test.each(OPERATIONS)('$id rejects unknown parameters', operation => {
    expect(() => operation.normalizeParameters({unrecognised: 'value'})).toThrowError(
      expect.objectContaining({code: 'unknown-parameter', details: {parameter: 'unrecognised'}})
    );
  });
  test('column creation and joins preserve prototype-like names as ordinary columns', async () => {
    const rows = makeData({id: [1]}, {id: 'int32'}, {identityColumn: 'id'});
    const result = await run(
      'calculate-column',
      {rows},
      {columnName: '__proto__', expression: 'id * 2'}
    );
    expectColumns(
      result,
      Object.fromEntries([
        ['id', [1]],
        ['__proto__', [2]]
      ]),
      Object.fromEntries([
        ['id', 'int32'],
        ['__proto__', 'float64']
      ])
    );
    const attributes = makeData(
      Object.fromEntries([
        ['source_id', [1]],
        ['__proto__', [3]]
      ]),
      Object.fromEntries([
        ['source_id', 'int32'],
        ['__proto__', 'int32']
      ])
    );
    expectColumns(
      await run('join-attributes', {target: rows, attributes}),
      Object.fromEntries([
        ['id', [1]],
        ['__proto__', [3]]
      ]),
      Object.fromEntries([
        ['id', 'int32'],
        ['__proto__', 'int32']
      ])
    );
    const second = makeData({toString: [2]}, {toString: 'int32' as const});
    expectColumns(
      await run('concatenate', {first: rows, second}),
      {
        id: [1, null],
        toString: [null, 2],
        source_table: ['first', 'second'],
        workflow_identity: ['1', '0']
      },
      {id: 'int32', toString: 'int32' as const, source_table: 'string', workflow_identity: 'string'}
    );
  });
  test('missing input and excessive input rows are refused', async () => {
    await expect(getOperation('drop-geometry').run(CONTEXT, {}, {})).rejects.toMatchObject({
      code: 'input-required',
      details: {port: 'rows'}
    });
    const rows = makeData({value: []}, {value: 'int32'});
    rows.table.rowCount = 100001;
    await expect(getOperation('drop-geometry').run(CONTEXT, {rows}, {})).rejects.toMatchObject({
      code: 'capacity-exceeded',
      details: {port: 'rows'}
    });
  });
  test('F1 reconciles different geometry column names into spatial features', async () => {
    const first = makeData({shape: [POINT]}, {shape: 'string'}, {geometryColumn: 'shape'});
    const second = makeData(
      {geometry: [POINT]},
      {geometry: 'string'},
      {geometryColumn: 'geometry'}
    );
    const result = await run('concatenate', {first, second});
    const output = expectColumns(
      result,
      {shape: [POINT, POINT], source_table: ['first', 'second']},
      {shape: 'string', source_table: 'string'}
    );
    expect(getWorkflowFeatures(output).features.map(feature => feature.geometry)).toEqual([
      JSON.parse(POINT),
      JSON.parse(POINT)
    ]);
  });
  test.each([
    'first',
    'second'
  ] as const)('F1 refuses a geometry/attribute collision in %s without losing data', async attributePort => {
    const geometry = makeData({shape: [POINT]}, {shape: 'string'}, {geometryColumn: 'shape'});
    const attributes = makeData({shape: ['ordinary attribute']}, {shape: 'string'});
    const inputs =
      attributePort === 'first'
        ? {first: attributes, second: geometry}
        : {first: geometry, second: attributes};
    const snapshot = structuredClone(inputs);
    await expect(getOperation('concatenate').run(CONTEXT, inputs, {})).rejects.toMatchObject({
      code: 'incompatible-port',
      details: {port: attributePort},
      message: `Concatenate: ${attributePort === 'first' ? 'First' : 'Second'} column "shape" is an attribute but is also the output geometry column; rename the attribute before concatenating.`
    });
    expect(inputs).toEqual(snapshot);
  });
  test.each([
    'first',
    'second'
  ] as const)('F2 refuses reserved source_table geometry on %s', async port => {
    const geometry = makeData(
      {source_table: [POINT]},
      {source_table: 'string'},
      {geometryColumn: 'source_table'}
    );
    const rows = makeData({value: [1]}, {value: 'int32'});
    const inputs =
      port === 'first' ? {first: geometry, second: rows} : {first: rows, second: geometry};
    const snapshot = structuredClone(inputs);
    await expect(getOperation('concatenate').run(CONTEXT, inputs, {})).rejects.toMatchObject({
      code: 'incompatible-port',
      details: {port},
      message: `Concatenate: ${port === 'first' ? 'First' : 'Second'} column "source_table" is reserved for output provenance; rename the column before concatenating.`
    });
    expect(inputs).toEqual(snapshot);
  });
  test.each([
    true,
    false
  ])('F3 reconciles different identities with unrelated attribute present: %s', async includeUnrelated => {
    const first = makeData(
      {left_id: ['a'], geometry: [POINT]},
      {left_id: 'string', geometry: 'string'},
      {geometryColumn: 'geometry', identityColumn: 'left_id'}
    );
    const second = makeData(
      {right_id: ['b'], ...(includeUnrelated ? {left_id: ['unrelated']} : {}), geometry: [POINT]},
      {
        right_id: 'string',
        ...(includeUnrelated ? {left_id: 'string' as const} : {}),
        geometry: 'string'
      },
      {geometryColumn: 'geometry', identityColumn: 'right_id'}
    );
    const result = await run('concatenate', {first, second});
    const output = result.outputs['combined'];
    expect(output.identityColumn).toBe('workflow_identity');
    expect(output.table.columnTypes['workflow_identity']).toBe('string');
    expect(output.table.columns['workflow_identity']).toEqual(['a', 'b']);
    expect(output.table.columns['left_id']).toEqual(['a', includeUnrelated ? 'unrelated' : null]);
    expect(output.table.columns['right_id']).toEqual([null, 'b']);
    expect(getWorkflowFeatures(output).features.map(feature => feature.id)).toEqual(['a', 'b']);
  });
  test.each([
    'ascending',
    'descending'
  ] as const)('F4 sorts mixed GeoJSON IDs by declared string type in %s order', async order => {
    const rows = makeWorkflowDataFromFeatures(
      {
        type: 'FeatureCollection',
        features: [2, '10', 1].map(id => ({
          type: 'Feature',
          id,
          properties: {},
          geometry: JSON.parse(POINT)
        }))
      },
      {id: 'source', label: 'Source'}
    );
    const result = await run('sort-rows', {rows}, {column: 'feature_id', order});
    const output = result.outputs['sorted'];
    expect(output.table.columnTypes['feature_id']).toBe('string');
    expect(output.table.columns['feature_id']).toEqual(
      order === 'ascending' ? [1, '10', 2] : [2, '10', 1]
    );
    expect(getWorkflowFeatures(output).features.map(feature => feature.id)).toEqual(
      order === 'ascending' ? [1, '10', 2] : [2, '10', 1]
    );
  });
  // CONTRACT REQUEST: workflow-data must expose a column-order setter; its cache is private.
  test.fails('F5 preserves first-seen order for integer-like columns through the data API', async () => {
    const first = makeData({'10': [1], name: ['first']}, {'10': 'int32', name: 'string'});
    const second = makeData({'2': [2]}, {'2': 'int32'});
    const output = (await run('concatenate', {first, second})).outputs['combined'];
    expect(getWorkflowColumnNames(output)).toEqual(['10', 'name', '2', 'source_table']);
  });
  test('F6 clears empty-string geometry metadata after dropping geometry', async () => {
    const rows = makeData(
      {'': [POINT], value: [1]},
      {'': 'string', value: 'int32'},
      {geometryColumn: ''}
    );
    expect(rows.table.geometryColumn).toBe('');
    const result = await run('drop-geometry', {rows});
    const output = expectColumns(result, {value: [1]}, {value: 'int32'});
    expect(output.kind).toBe('table');
    expect(output.table.geometryColumn).toBeUndefined();
  });

  test('select and drop geometry preserve exact row counts with no remaining columns', async () => {
    const rows = makeData({value: [1, 2, 3]}, {value: 'int32'});
    const selected = (await run('select-columns', {rows})).outputs['selected'];
    expect(selected.table.columns).toEqual({});
    expect(selected.table.columnTypes).toEqual({});
    expect(selected.table.rowCount).toBe(3);
    const geometries = makeData(
      {geometry: [POINT, POINT]},
      {geometry: 'string'},
      {geometryColumn: 'geometry'}
    );
    const dropped = (await run('drop-geometry', {rows: geometries})).outputs['table'];
    expect(dropped.table.columns).toEqual({});
    expect(dropped.table.columnTypes).toEqual({});
    expect(dropped.table.rowCount).toBe(2);
    const empty = await run('drop-geometry', {rows: makeData({}, {})});
    expect(empty.outputs['table'].table.rowCount).toBe(0);
  });
  test('F1 preserves attributes named for the other input geometry column', async () => {
    const first = makeData(
      {shape: [POINT], geometry: ['attribute']},
      {shape: 'string', geometry: 'string'},
      {geometryColumn: 'shape'}
    );
    const second = makeData(
      {geometry: [POINT]},
      {geometry: 'string'},
      {geometryColumn: 'geometry'}
    );
    const result = await run('concatenate', {first, second});
    const output = expectColumns(
      result,
      {shape: [POINT, POINT], geometry: ['attribute', null], source_table: ['first', 'second']},
      {shape: 'string', geometry: 'string', source_table: 'string'}
    );
    expect(getWorkflowFeatures(output).features.map(feature => feature.geometry)).toEqual([
      JSON.parse(POINT),
      JSON.parse(POINT)
    ]);
  });
  test.each([
    'first',
    'second'
  ] as const)('F2 refuses reserved source_table attributes on %s', async port => {
    const attributes = makeData({source_table: ['keep me']}, {source_table: 'string'});
    const rows = makeData({value: [1]}, {value: 'int32'});
    const inputs =
      port === 'first' ? {first: attributes, second: rows} : {first: rows, second: attributes};
    const snapshot = structuredClone(inputs);
    await expect(getOperation('concatenate').run(CONTEXT, inputs, {})).rejects.toMatchObject({
      code: 'incompatible-port',
      details: {port}
    });
    expect(inputs).toEqual(snapshot);
  });
  test('F3 generated identity avoids collisions and supports a downstream identity join', async () => {
    const first = makeData(
      {left_id: ['a'], workflow_identity: ['keep first']},
      {left_id: 'string', workflow_identity: 'string'},
      {identityColumn: 'left_id'}
    );
    const second = makeData(
      {right_id: ['b'], workflow_identity_2: ['keep second']},
      {right_id: 'string', workflow_identity_2: 'string'},
      {identityColumn: 'right_id'}
    );
    const combined = (await run('concatenate', {first, second})).outputs['combined'];
    expect(combined.identityColumn).toBe('workflow_identity_3');
    expect(combined.table.columns['workflow_identity']).toEqual(['keep first', null]);
    expect(combined.table.columns['workflow_identity_2']).toEqual([null, 'keep second']);
    expect(combined.table.columns['workflow_identity_3']).toEqual(['a', 'b']);
    const attributes = makeData(
      {source_id: ['b', 'a'], score: [2, 1]},
      {source_id: 'string', score: 'int32'}
    );
    const joined = await run('join-attributes', {target: combined, attributes});
    expect(joined.outputs['joined'].table.columns['score']).toEqual([1, 2]);
    expect(joined.diagnostics).toEqual([]);
  });
  test('F3 reconciles missing identity using source row indices and preserves null identities', async () => {
    const first = makeData({id: ['a', null]}, {id: 'string'}, {identityColumn: 'id'});
    const second = makeData({id: ['unrelated', 'also unrelated']}, {id: 'string'});
    const output = (await run('concatenate', {first, second})).outputs['combined'];
    expect(output.identityColumn).toBe('workflow_identity');
    expect(output.table.columns['workflow_identity']).toEqual(['a', null, '0', '1']);
    expect(output.table.columns['id']).toEqual(['a', null, 'unrelated', 'also unrelated']);
  });
  test('F6 concatenates an empty-string geometry column without losing either geometry', async () => {
    const first = makeData({'': [POINT]}, {'': 'string'}, {geometryColumn: ''});
    const second = makeData({shape: [POINT]}, {shape: 'string'}, {geometryColumn: 'shape'});
    const output = (await run('concatenate', {first, second})).outputs['combined'];
    expect(output.table.geometryColumn).toBe('');
    expect(output.table.columns['']).toEqual([POINT, POINT]);
    expect(getWorkflowFeatures(output).features.map(feature => feature.geometry)).toEqual([
      JSON.parse(POINT),
      JSON.parse(POINT)
    ]);
  });
});
