// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {KeplerComputeSession, type KeplerMemoryReservation} from '../src';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {createResidentGroupByPlan} from '../../../examples/deck/kepler-compute/src/resident-group-by-plan';
import {
  executeResidentGroupBy,
  type ResidentGroupByResult
} from '../../../examples/deck/kepler-compute/src/resident-group-by-runtime';

test('native Group By adopts all-source aggregates with original dictionary codes, independent rows and reversible ownership', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const labels = ['a', 'b', 'unused'];
  const dictionary = arrow.vectorFromArray(labels, new arrow.Utf8());
  const dictionaryType = new arrow.Dictionary(new arrow.Utf8(), new arrow.Uint32(), 0, false);
  const schema = new arrow.Schema([
    new arrow.Field('amount_sum', dictionaryType, true),
    new arrow.Field('amount', new arrow.Float32(), true),
    new arrow.Field('nullable', new arrow.Float32(), true),
    new arrow.Field('row', new arrow.Float32(), false)
  ]);
  const key = (codes: number[], validity: number) =>
    arrow.makeVector(
      arrow.makeData({
        type: dictionaryType,
        data: Uint32Array.from(codes),
        length: codes.length,
        nullCount: 1,
        nullBitmap: Uint8Array.of(validity),
        dictionary
      })
    );
  const first = new arrow.Table(schema, {
    amount_sum: key([1, 0], 0b01),
    amount: arrow.vectorFromArray([2, 8], new arrow.Float32()),
    nullable: arrow.vectorFromArray([null, 5], new arrow.Float32()),
    row: arrow.makeVector(new Float32Array([0, 1]))
  });
  const second = new arrow.Table(schema, {
    amount_sum: key([0, 1, 0], 0b011),
    amount: arrow.vectorFromArray([null, 6, null], new arrow.Float32()),
    nullable: arrow.vectorFromArray([null, null, null], new arrow.Float32()),
    row: arrow.makeVector(new Float32Array([2, 3, 4]))
  });
  const source = session.importArrowDataset({
    id: 'group-source',
    table: first.concat(second),
    normalization: {dictionaries: {amount_sum: {values: labels, ordered: false}}}
  });
  const fields = [
    {
      name: 'amount_sum',
      type: 'string',
      analyzerType: 'STRING',
      fieldIdx: 0,
      id: 'key',
      format: ''
    },
    {name: 'amount', type: 'real', analyzerType: 'FLOAT', fieldIdx: 1, id: 'amount', format: ''},
    {name: 'nullable', type: 'real', analyzerType: 'FLOAT', fieldIdx: 2, id: 'nullable', format: ''}
  ];
  const operation = {
    id: 'group-operation',
    dataId: source.id,
    fieldName: 'amount_sum',
    aggregations: {amount: 'sum', nullable: 'count'},
    resultId: 'group-result',
    resultLabel: 'Grouped values'
  };
  const plan = createResidentGroupByPlan(operation, {
    datasetId: source.id,
    revision: source.revision,
    rowCount: source.rowCount,
    fields,
    columns: source.value.columns,
    dictionaries: source.value.prepared.dictionaries,
    nullCounts: source.value.analytics.nullCounts
  });
  // These expectations match the pinned upstream oracle in resident-group-by-plan.node.spec.ts.
  // Keep native rendering barrels out of this GPU runtime/adoption fixture.
  const expectedRows = [
    ['b', 8, 2],
    [null, 8, 2],
    ['a', null, 1]
  ];
  const expectedFields = [
    {name: 'amount_sum', type: 'string', analyzerType: 'STRING'},
    {name: 'amount_sum_1', type: 'real', analyzerType: 'FLOAT'},
    {name: 'nullable_count', type: 'integer', analyzerType: 'integer'}
  ];
  const expectedDerivedDataset = {
    type: 'groupBy',
    sourceDataIds: [source.id],
    operationId: operation.id
  };
  const explorer = session.openExplorer(source.id, {
    workingMemoryBudgetBytes: 8 * 1024 * 1024,
    dimensions: [{id: 'row', column: 'row', kind: 'range'}]
  });
  const sourceChunks = Object.values(source.value.columns).flatMap(vector => vector.data);
  const sourceBuffers = [
    ...new Set(
      sourceChunks.map(chunk =>
        chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer
      )
    )
  ];
  const sourceWrites = sourceBuffers.map(buffer => vi.spyOn(buffer, 'write'));
  const sourceReads = sourceBuffers.map(buffer => vi.spyOn(buffer, 'readAsync'));
  const stagingBytes = Math.max(16, plan.keyCapacity * 4 * (3 + plan.measures.length * 2));
  const stagingReservations: KeplerMemoryReservation[] = [];
  const stagingReadSpies: Array<{mockRestore(): void}> = [];
  let executing = false;
  let stagedReads = 0;
  const reserveMemory = session.registry.reserveMemory.bind(session.registry);
  const reservationSpy = vi
    .spyOn(session.registry, 'reserveMemory')
    .mockImplementation((bytes, kind) => {
      const reservation = reserveMemory(bytes, kind);
      if (kind === 'gpu' && bytes === stagingBytes) stagingReservations.push(reservation);
      return reservation;
    });
  const createDeviceBuffer = device.createBuffer.bind(device);
  const createBuffer = vi.spyOn(device, 'createBuffer').mockImplementation(props => {
    const buffer = createDeviceBuffer(props);
    if (props.id?.startsWith('group-by-')) {
      expect(stagingReservations.at(-1)?.isActive).toBe(true);
      const read = buffer.readAsync.bind(buffer);
      stagingReadSpies.push(
        vi.spyOn(buffer, 'readAsync').mockImplementation(async (offset, length) => {
          const staging = executing ? stagingReservations.at(-1) : undefined;
          if (executing) {
            expect(staging?.isActive).toBe(true);
            expect(session.registry.memoryUsage.reservedBytes.gpu).toBeGreaterThanOrEqual(
              stagingBytes
            );
            stagedReads++;
          }
          try {
            return await read(offset, length);
          } finally {
            if (staging) expect(staging.isActive).toBe(true);
          }
        })
      );
    }
    return buffer;
  });
  const adopt = session.adoptArrowDataset.bind(session);
  const adoption = vi.spyOn(session, 'adoptArrowDataset').mockImplementation(options => {
    const allocationsBefore = createBuffer.mock.calls.length;
    expect(stagingReservations.at(-1)?.isActive).toBe(false);
    const adopted = adopt(options);
    expect(createBuffer.mock.calls.length).toBe(allocationsBefore);
    for (const [name, vector] of Object.entries(options.columns))
      expect(adopted?.value.columns[name].data[0].buffer).toBe(vector.data[0].buffer);
    return adopted;
  });
  const results: ResidentGroupByResult[] = [];
  const destructionSpies: Array<{mockRestore(): void}> = [];
  const execute = async (extra: {signal?: AbortSignal; isCurrent?: () => boolean} = {}) => {
    executing = true;
    try {
      return await executeResidentGroupBy({
        session,
        source: {datasetId: source.id, revision: source.revision},
        plan,
        isCurrent: () => true,
        ...extra
      });
    } finally {
      executing = false;
    }
  };
  const readRows = (table: arrow.Table) =>
    Array.from({length: table.numRows}, (_, row) =>
      table.schema.fields.map(field => table.getChild(field.name)!.get(row))
    );
  const outputBuffers = (dataset: ResidentGroupByResult['dataset']) => [
    ...new Set(
      [...Object.values(dataset.value.columns), ...Object.values(dataset.value.analytics.validity)]
        .filter(vector => vector !== undefined)
        .flatMap(vector => vector.data.map(chunk => chunk.buffer))
    )
  ];
  try {
    explorer.setRange('row', 0, 0);
    const encoder = device.createCommandEncoder();
    explorer.encode(encoder);
    device.submit(encoder.finish());
    const aborted = new AbortController();
    aborted.abort();
    const allocationsBeforeAbort = createBuffer.mock.calls.length;
    await expect(execute({signal: aborted.signal})).rejects.toThrow(/stale|cancel/i);
    await expect(execute({isCurrent: () => false})).rejects.toThrow(/stale|cancel/i);
    expect(createBuffer.mock.calls.length).toBe(allocationsBeforeAbort);
    expect(session.registry.getDataset(plan.resultId)).toBeUndefined();

    const rolledBack = await execute();
    results.push(rolledBack);
    expect(rolledBack.isCurrent()).toBe(true);
    expect(readRows(rolledBack.dataset.value.source)).toEqual(expectedRows);
    const rollbackBuffers = outputBuffers(rolledBack.dataset);
    const rollbackDestruction = rollbackBuffers.map(buffer => vi.spyOn(buffer, 'destroy'));
    destructionSpies.push(...rollbackDestruction);
    await rolledBack.destroy();
    await rolledBack.destroy();
    expect(rolledBack.isCurrent()).toBe(false);
    expect(session.registry.getDataset(plan.resultId)).toBeUndefined();
    for (const destroy of rollbackDestruction) expect(destroy).toHaveBeenCalledTimes(1);
    for (const buffer of sourceBuffers) expect(buffer.destroyed).toBe(false);

    const result = await execute();
    results.push(result);
    const dataset = result.dataset;
    expect(readRows(dataset.value.source)).toEqual(expectedRows);
    expect(result.provenance.fields).toEqual(expectedFields);
    expect(dataset.value.source.schema.fields.map(field => field.name)).toEqual([
      'amount_sum',
      'amount_sum_1',
      'nullable_count'
    ]);
    expect(result.provenance).toMatchObject({
      population: 'all',
      source: {datasetId: source.id, revision: source.revision},
      rowOrder: 'first-source-occurrence'
    });
    expect(dataset.source?.metadata?.['derivedDataset']).toEqual(expectedDerivedDataset);
    expect(dataset.rowCount).toBe(3);
    expect(session.registry.resolveSourceRow(dataset, 2)).toMatchObject({
      datasetId: plan.resultId,
      sourceRowIndex: 2
    });
    expect(source.value.columns['amount'].data.map(chunk => chunk.length)).toEqual([2, 3]);
    expect(dataset.value.prepared.dictionaries['amount_sum'].values).toEqual(labels);
    const codesBuffer = dataset.value.columns['amount_sum'].data[0].buffer;
    const codesBytes = await codesBuffer.readAsync(0, 12);
    expect(Array.from(new Uint32Array(codesBytes.buffer, codesBytes.byteOffset, 3))).toEqual([
      1, 3, 0
    ]);
    const measureValidity = dataset.value.analytics.validity['amount_sum_1']!;
    const validityBytes = await measureValidity.data[0].buffer.readAsync(0, 12);
    expect(Array.from(new Uint32Array(validityBytes.buffer, validityBytes.byteOffset, 3))).toEqual([
      1, 1, 0
    ]);
    const exported = arrow.tableFromIPC(arrow.tableToIPC(dataset.value.source));
    expect(readRows(exported)).toEqual(expectedRows);
    expect(JSON.parse(exported.schema.metadata.get('luma.gl.groupBy')!)).toEqual(result.provenance);
    const preparedExport = arrow.tableFromIPC(arrow.tableToIPC(dataset.value.prepared.table));
    expect(Array.from(preparedExport.getChild('amount_sum')!.data[0].dictionary!)).toEqual(labels);
    expect(readRows(preparedExport)).toEqual(expectedRows);
    expect(adoption).toHaveBeenCalledTimes(2);
    expect(stagingReservations).toHaveLength(2);
    expect(stagingReservations.every(reservation => !reservation.isActive)).toBe(true);
    // Per run: status, keys, first source rows and a values/validity pair per measure (key validity is derived from codes).
    expect(stagedReads).toBe(2 * (3 + plan.measures.length * 2));
    for (const read of sourceReads) expect(read).not.toHaveBeenCalled();
    for (const write of sourceWrites) expect(write).not.toHaveBeenCalled();

    const adoptedBuffers = outputBuffers(dataset);
    const adoptedDestruction = adoptedBuffers.map(buffer => vi.spyOn(buffer, 'destroy'));
    destructionSpies.push(...adoptedDestruction);
    result.transfer();
    await result.destroy();
    expect(session.registry.getDataset(plan.resultId)?.revision).toBe(dataset.revision);
    for (const destroy of adoptedDestruction) expect(destroy).not.toHaveBeenCalled();
    // Simulated session loss retains the exact Arrow/dictionary recipe for replacement-device recovery.
    await session.handleDeviceLoss();
    const recovery = session.registry
      .getRecoverySources()
      .find(entry => entry.id === plan.resultId)!;
    expect(recovery.source.value).toBe(dataset.value.source);
    expect(recovery.source.metadata?.['normalization']).toEqual({
      dictionaries: {amount_sum: {values: labels, ordered: false}}
    });
    expect(recovery.source.metadata?.['groupBy']).toEqual(result.provenance);
    expect(recovery.resources.some(resource => resource.value === dataset.value.prepared)).toBe(
      true
    );
    for (const destroy of adoptedDestruction) expect(destroy).toHaveBeenCalledTimes(1);
  } finally {
    for (const result of results) await result.destroy();
    adoption.mockRestore();
    createBuffer.mockRestore();
    reservationSpy.mockRestore();
    for (const spy of stagingReadSpies) spy.mockRestore();
    for (const spy of [...sourceReads, ...sourceWrites, ...destructionSpies]) spy.mockRestore();
    const fence = device.createFence();
    try {
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
  }
  expect(session.registry.memoryUsage.totalBytes).toBe(0);
});
