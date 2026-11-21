// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {
  makeGPUAnalyticsTableFromArrowTable,
  makeArrowTableFromGPUAnalyticsTable
} from '@luma.gl/graph/arrow';
import {KeplerComputeSession} from '../src/session/compute-session';
import {
  ResidentSqlGpuRuntime,
  materializeResidentSqlArrow
} from '../../../examples/deck/kepler-compute/src/resident-sql-gpu-runtime';

/**
 * An ordinary empty object whose static type is the map it is passed as. An object literal `{}`
 * is typed with `Object.prototype.constructor`, which collides with a column named "constructor".
 */
function createOrdinaryEmptyMap<MapType extends object>(): MapType {
  return {} as MapType;
}

test('SQL Arrow materialization preserves constructor, nullable and dictionary columns', async () => {
  const device = new NullDevice({});
  const source = new arrow.Table({
    constructor: arrow.vectorFromArray([42, 42, 42], new arrow.Int32()),
    score: arrow.vectorFromArray([5, null, 7], new arrow.Int32()),
    label: arrow.vectorFromArray(
      ['a', 'b', 'a'],
      new arrow.Dictionary(new arrow.Utf8(), new arrow.Int32())
    )
  });
  const resident = makeGPUAnalyticsTableFromArrowTable(device, source);
  // Empty ordinary maps have inherited constructor entries in the installed package.
  await expect(
    makeArrowTableFromGPUAnalyticsTable({
      table: resident.table,
      validity: createOrdinaryEmptyMap(),
      dictionaries: createOrdinaryEmptyMap()
    })
  ).rejects.toThrow();
  const result = await materializeResidentSqlArrow(resident);
  expect(Array.from(result.getChild('constructor')!)).toEqual([42, 42, 42]);
  expect(Array.from(result.getChild('score')!)).toEqual([5, null, 7]);
  expect(Array.from(result.getChild('label')!)).toEqual(['a', 'b', 'a']);
  const emptyMetadata = await materializeResidentSqlArrow({
    table: resident.table,
    validity: resident.validity
  });
  expect(emptyMetadata.getChild('constructor')!.get(0)).toBe(42);
  const nonnullable = new arrow.Table(
    new arrow.Schema([new arrow.Field('constructor', new arrow.Int32(), false)]),
    {constructor: arrow.vectorFromArray([42], new arrow.Int32())}
  );
  const plain = makeGPUAnalyticsTableFromArrowTable(device, nonnullable);
  const emptyMaps = await materializeResidentSqlArrow({
    table: plain.table,
    validity: createOrdinaryEmptyMap(),
    dictionaries: createOrdinaryEmptyMap()
  });
  expect(emptyMaps.getChild('constructor')!.get(0)).toBe(42);
  device.destroy();
});

for (const nullable of [false, true])
  test('constructor registers and materializes with nullable=' + nullable, async () => {
    const device = new NullDevice({});
    const source = new arrow.Table(
      new arrow.Schema([new arrow.Field('constructor', new arrow.Int32(), nullable)]),
      {constructor: arrow.vectorFromArray(nullable ? [42, null] : [42, 43], new arrow.Int32())}
    );
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 16 * 1024 * 1024});
    const imported = session.importArrowDataset({id: 'legal-columns', table: source});
    const runtime = new ResidentSqlGpuRuntime(session);
    expect(runtime.getRegistration().tables).toEqual([
      {name: 'legal-columns', datasetId: 'legal-columns', columnCount: 1},
      {name: 'legal_columns', datasetId: 'legal-columns', columnCount: 1}
    ]);
    const result = await materializeResidentSqlArrow(imported.value.analytics);
    expect(Array.from(result.getChild('constructor')!)).toEqual(nullable ? [42, null] : [42, 43]);
    runtime.destroy();
    await session.destroy(Promise.resolve());
    device.destroy();
  });
