// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {runGroupBy, wrapTo} from '@kepler.gl/actions';
import {
  getResidentGroupByRequest,
  getResidentGroupByRequestKey,
  updateResidentGroupByStatus,
  ResidentGroupByRequests,
  type ResidentGroupByRequest,
  type ResidentGroupByStatus
} from '../../../examples/deck/kepler-compute/src/resident-action-guards';

const request: ResidentGroupByRequest = {
  id: 'operation',
  dataId: 'resident',
  fieldName: 'category',
  aggregations: {value: 'sum'},
  resultId: 'result',
  resultLabel: 'Grouped values'
};
const resident = (id: string) => id === 'resident';

test('async status preserves editable operations and ignores stale completion tokens', () => {
  const pending = updateResidentGroupByStatus([request], {
    id: request.id,
    requestId: 1,
    phase: 'running'
  });
  const newer = updateResidentGroupByStatus(pending, {
    id: request.id,
    requestId: 2,
    phase: 'running'
  });
  expect(
    updateResidentGroupByStatus(newer, {id: request.id, requestId: 1, phase: 'complete'})
  ).toEqual(newer);
  const failed = updateResidentGroupByStatus(newer, {
    id: request.id,
    requestId: 2,
    phase: 'failed',
    error: 'Use SQL'
  });
  expect(failed[0]).toMatchObject({...request, error: 'Use SQL', residentStatus: 'failed'});
  expect(
    updateResidentGroupByStatus(failed, {id: request.id, requestId: 2, phase: 'complete'})
  ).toEqual([]);
});

test('native Group By snapshots only its own resident configuration and retains column order', () => {
  const aggregations = {value: 'sum', other: 'count'};
  const state = {
    visState: {
      groupBys: [
        {...request, aggregations},
        {...request, id: 'ordinary', dataId: 'ordinary'}
      ]
    }
  };
  const captured = getResidentGroupByRequest(
    state,
    wrapTo('compute', runGroupBy(request.id)),
    resident
  )!;
  aggregations.value = 'average';
  expect(captured.aggregations['value']).toBe('sum');
  expect(getResidentGroupByRequest(state, runGroupBy('ordinary'), resident)).toBeUndefined();
  expect(
    getResidentGroupByRequest(state, wrapTo('another', runGroupBy(request.id)), resident)
  ).toBeUndefined();
  expect(getResidentGroupByRequestKey(captured)).not.toBe(
    getResidentGroupByRequestKey({...captured, aggregations: {other: 'count', value: 'sum'}})
  );
});

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}

function setup(
  execute: (
    request: ResidentGroupByRequest,
    signal: AbortSignal,
    current: () => boolean
  ) => Promise<string>
) {
  const statuses: ResidentGroupByStatus[] = [];
  const attach = vi.fn(async (_result: string, _isCurrent: () => boolean) => undefined);
  const transfer = vi.fn();
  const discard = vi.fn(async (_result: string) => undefined);
  const cleanupError = vi.fn();
  const controller = new ResidentGroupByRequests({
    execute,
    attach,
    transfer,
    discard,
    status: status => statuses.push(status),
    cleanupError
  });
  return {controller, statuses, attach, transfer, discard, cleanupError};
}

test('Group By starts after dispatch, attaches then transfers, and removes its operation only on success', async () => {
  const execution = vi.fn(async () => 'owned-result');
  const host = setup(execution);
  const task = host.controller.run(request, () => true);
  expect(execution).not.toHaveBeenCalled();
  expect(host.statuses.map(status => status.phase)).toEqual(['running']);
  await task;
  expect(host.attach).toHaveBeenCalledWith('owned-result', expect.any(Function));
  expect(host.transfer).toHaveBeenCalledWith('owned-result');
  expect(host.discard).not.toHaveBeenCalled();
  expect(host.statuses.map(status => status.phase)).toEqual(['running', 'complete']);
});

test('superseded native runs cannot attach or discard a newer result', async () => {
  const first = deferred<string>();
  let attempts = 0;
  const host = setup(async () => (++attempts === 1 ? first.promise : 'new-result'));
  const oldTask = host.controller.run(request, () => true);
  await Promise.resolve();
  await host.controller.run(request, () => true);
  first.resolve('old-result');
  await oldTask;
  expect(host.attach.mock.calls.map(call => call[0])).toEqual(['new-result']);
  expect(host.transfer.mock.calls.map(call => call[0])).toEqual(['new-result']);
  expect(host.discard.mock.calls.map(call => call[0])).toEqual(['old-result']);
  expect(host.statuses.filter(status => status.phase === 'complete')).toHaveLength(1);
});

test('source replacement, native editing and removal abort pending work and discard late results', async () => {
  const pending = deferred<string>();
  let current = true;
  let signal: AbortSignal | undefined;
  const host = setup(async (_request, operationSignal) => {
    signal = operationSignal;
    return pending.promise;
  });
  const task = host.controller.run(request, () => current);
  await Promise.resolve();
  current = false;
  host.controller.reconcile();
  expect(signal?.aborted).toBe(true);
  pending.resolve('stale-source-result');
  await task;
  expect(host.attach).not.toHaveBeenCalled();
  expect(host.discard).toHaveBeenCalledWith('stale-source-result');
  expect(host.statuses.map(status => status.phase)).toEqual(['running', 'cancelled']);
});

test('unsupported execution and attachment failures stay editable; partially attached results retire', async () => {
  const host = setup(async () => {
    throw new Error('Use SQL for this aggregation');
  });
  await host.controller.run(request, () => true);
  expect(host.statuses.at(-1)).toMatchObject({
    phase: 'failed',
    error: 'Use SQL for this aggregation'
  });
  const attachment = setup(async () => 'partial-result');
  attachment.attach.mockRejectedValueOnce(new Error('Source changed while attaching'));
  await attachment.controller.run(request, () => true);
  expect(attachment.transfer).not.toHaveBeenCalled();
  expect(attachment.discard).toHaveBeenCalledWith('partial-result');
  expect(attachment.statuses.at(-1)?.phase).toBe('failed');
});

test('source changes during attachment discard only the captured result and never commit it', async () => {
  let current = true;
  const attaching = deferred<void>();
  const started = deferred<void>();
  const host = setup(async () => 'old-owner-result');
  host.attach.mockImplementationOnce(async () => {
    started.resolve();
    await attaching.promise;
  });
  const task = host.controller.run(request, () => current);
  await started.promise;
  current = false;
  host.controller.reconcile();
  attaching.resolve();
  await task;
  expect(host.transfer).not.toHaveBeenCalled();
  expect(host.discard).toHaveBeenCalledExactlyOnceWith('old-owner-result');
});

test('completed results remain immutable snapshots after their historical source disappears', async () => {
  let sourceCurrent = true;
  const host = setup(async () => 'transferred-snapshot');
  await host.controller.run(request, () => sourceCurrent);
  sourceCurrent = false;
  host.controller.reconcile();
  expect(host.transfer).toHaveBeenCalledExactlyOnceWith('transferred-snapshot');
  expect(host.discard).not.toHaveBeenCalled();
  expect(host.statuses.map(status => status.phase)).toEqual(['running', 'complete']);
});
