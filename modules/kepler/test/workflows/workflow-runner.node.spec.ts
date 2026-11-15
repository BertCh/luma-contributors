// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {describe, expect, test, vi} from 'vitest';
import {KeplerAllocationError} from '../../src/runtime/dataset-registry';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import type {WorkflowRunnerOptions} from '../../src/workflows/workflow-runner';
import {WorkflowError} from '../../src/workflows/workflow-types';
import type {
  WorkflowData,
  WorkflowDocument,
  WorkflowOperation,
  WorkflowOperationOutput,
  WorkflowOperationRegistry,
  WorkflowPublishRequest
} from '../../src/workflows/workflow-types';
import {
  makeFixtureDocument,
  makeFixtureHost,
  makeFixturePoints,
  makeFixtureRegistry,
  makeFixtureTable
} from './workflow-fixtures';

function makeDeferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(complete => {
    resolve = complete;
  });
  return {promise, resolve};
}

async function flushMicrotasks(): Promise<void> {
  for (let index = 0; index < 30; index++) await Promise.resolve();
}

function replaceOperations(
  registry: WorkflowOperationRegistry,
  replacements: Partial<WorkflowOperation>[]
): WorkflowOperationRegistry {
  const operations = registry.list().map(operation => ({
    ...operation,
    ...replacements.find(replacement => replacement.id === operation.id)
  }));
  return {
    list: () => operations,
    get: operationId => operations.find(operation => operation.id === operationId)
  };
}

function makeOutput(data = makeFixturePoints(4)): WorkflowOperationOutput {
  return {
    outputs: {data},
    diagnostics: [],
    summary: {},
    engine: 'host',
    dispatchedContributors: []
  };
}

function createFixture(options: Partial<WorkflowRunnerOptions> = {}) {
  const calls: string[] = [];
  const host = makeFixtureHost();
  const registry = makeFixtureRegistry(calls);
  let clock = 0;
  const runner = new WorkflowRunner({host, registry, now: () => ++clock, ...options});
  const document = makeFixtureDocument();
  runner.setDocument(document);
  return {runner, calls, host, registry, document};
}

function readStatuses(runner: WorkflowRunner): string[] {
  const state = runner.getState();
  return state.order.map(stepId => state.steps[stepId].status);
}

function editFactor(document: WorkflowDocument, factor: number): WorkflowDocument {
  const edited = structuredClone(document);
  edited.steps[1].parameters['factor'] = {from: 'static', value: factor};
  return edited;
}

function makeControlledFixture() {
  const entered = makeDeferred();
  const release = makeDeferred();
  const calls: string[] = [];
  const registry = makeFixtureRegistry(calls);
  const original = registry.get('fixture-scale')!;
  let invocation = 0;
  const controlled = replaceOperations(registry, [
    {
      id: original.id,
      run: async (...arguments_) => {
        if (++invocation === 1) {
          entered.resolve();
          await release.promise;
        }
        return original.run(...arguments_);
      }
    }
  ]);
  const fixture = createFixture({registry: controlled});
  return {...fixture, calls, entered, release};
}

describe('WorkflowRunner results and provenance', () => {
  test('runs the fixture chain and retains immutable, stable snapshots', async () => {
    const {runner, calls} = createFixture();
    const initial = runner.getState();
    expect(runner.getState()).toBe(initial);
    expect(readStatuses(runner)).toEqual(['ready', 'ready', 'ready']);
    const record = await runner.run();
    expect(record.runId).toBe('fixture-workflow-1');
    expect(record.status).toBe('done');
    expect(calls).toEqual(['fixture-source:source', 'fixture-scale:scale', 'fixture-count:count']);
    expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
    const state = runner.getState();
    expect(state.lastRun).toBe(record);
    expect(record.steps.map(step => [step.stepId, step.status, step.cacheHit])).toEqual([
      ['source', 'done', false],
      ['scale', 'done', false],
      ['count', 'done', false]
    ]);
    expect(state.steps['scale'].result?.outputs['scaled'].table.columns['value']).toEqual([
      0, 4.5, 9, 13.5
    ]);
    expect(state.steps['count'].result?.outputs['count'].table.columns['rows']).toEqual([4]);
    expect(record.steps[1].inputs).toEqual([
      {
        port: 'points',
        stepId: 'source',
        output: 'data',
        key: state.steps['source'].key,
        rowCount: 4
      }
    ]);
    expect(record.steps[1].outputRows).toEqual({scaled: 4});
    expect(record.steps[1].engine).toBe('cpu-reference');
    expect(record.steps[1].fallbackReason).toBe('No GPU device supplied');
    expect(record.steps[1].dispatchedContributors).toEqual([]);
    expect(record.startedAt).toBe(1);
    expect(record.durationMilliseconds).toBe(13);
    expect(record.steps.map(step => step.durationMilliseconds)).toEqual([3, 3, 3]);
    expect(state.steps['scale'].result?.durationMilliseconds).toBe(1);
    expect(state.steps['scale'].result?.completedAt).toBe(8);
    expect(state.steps['scale'].result?.parameters).toEqual({factor: 3, fail: false});
    expect(state.steps['scale'].issues.some(issue => issue.code === 'gpu-fallback')).toBe(false);
    expect(Object.isFrozen(state)).toBe(true);
    expect(
      Object.isFrozen(state.steps['scale'].result?.outputs['scaled'].table.columns['value'])
    ).toBe(true);
    expect(() => {
      state.steps['source'].status = 'error';
    }).toThrow(TypeError);
    expect(initial.steps['source'].status).toBe('ready');
    expect(runner.getState()).toBe(state);
    await runner.destroy();
  });

  test('edits invalidate only dependents, undo restores the node cache, and external revisions invalidate all', async () => {
    const {runner, calls, document, host} = createFixture();
    await runner.run();
    const originalKeys = runner.getState().order.map(stepId => runner.getState().steps[stepId].key);
    runner.setDocument(editFactor(document, 5));
    expect(readStatuses(runner)).toEqual(['done', 'stale', 'stale']);
    calls.length = 0;
    await runner.run();
    expect(calls).toEqual(['fixture-scale:scale', 'fixture-count:count']);
    calls.length = 0;
    runner.setDocument(document);
    expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
    expect(runner.getState().order.map(stepId => runner.getState().steps[stepId].key)).toEqual(
      originalKeys
    );
    expect((await runner.run()).steps).toEqual([]);
    expect(calls).toEqual([]);
    host.revision++;
    runner.refreshExternal();
    expect(readStatuses(runner)).toEqual(['stale', 'stale', 'stale']);
    await runner.destroy();
  });

  test('runs only the target and upstream; force reruns done steps', async () => {
    const {runner, calls} = createFixture();
    const first = await runner.run({target: 'scale'});
    expect(first.target).toBe('scale');
    expect(first.steps.map(step => step.stepId)).toEqual(['source', 'scale']);
    expect(readStatuses(runner)).toEqual(['done', 'done', 'ready']);
    calls.length = 0;
    const second = await runner.run({target: 'scale', force: true});
    expect(second.runId).toBe('fixture-workflow-2');
    expect(calls).toEqual(['fixture-source:source', 'fixture-scale:scale']);
    expect(second.steps.every(step => !step.cacheHit)).toBe(true);
    await runner.destroy();
  });

  test('undo restores observed geometry kinds before deriving downstream statuses', async () => {
    const base = makeFixtureRegistry();
    const registry = replaceOperations(base, [
      {
        id: 'fixture-source',
        run: async (_context, _inputs, parameters) => {
          if (parameters['count'] === 4) return makeOutput();
          return makeOutput({
            kind: 'line',
            table: makeFixtureTable('lines', [
              {
                id: 'a',
                value: 1,
                geometry: JSON.stringify({
                  type: 'LineString',
                  coordinates: [
                    [0, 0],
                    [1, 1]
                  ]
                })
              }
            ])
          });
        }
      }
    ]);
    const {runner, document} = createFixture({registry});
    await runner.run();
    const edited = structuredClone(document);
    edited.steps[0].parameters['count'] = {from: 'static', value: 2};
    runner.setDocument(edited);
    expect((await runner.run()).status).toBe('error');
    runner.setDocument(document);
    expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
    expect(runner.getState().steps['source'].result?.outputs['data'].kind).toBe('point');
    await runner.destroy();
  });

  test('rejects unknown and disabled targets without executing', async () => {
    const {runner, calls, document} = createFixture();
    expect(await runner.run({target: 'missing'})).toMatchObject({status: 'error', steps: []});
    document.steps[1].disabled = true;
    runner.setDocument(document);
    expect(await runner.run({target: 'scale'})).toMatchObject({status: 'error', steps: []});
    expect(await runner.run({target: 'count'})).toMatchObject({status: 'error', steps: []});
    expect(calls).toEqual([]);
    await runner.destroy();
  });

  test('blocks failed dependents and continues independent branches', async () => {
    const {runner, calls, document} = createFixture();
    document.steps[1].parameters['fail'] = {from: 'static', value: true};
    document.steps.push({
      ...structuredClone(document.steps[2]),
      id: 'independent',
      inputs: {rows: {stepId: 'source', output: 'data'}},
      position: {x: 600, y: 0}
    });
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe('error');
    expect(record.steps.map(step => [step.stepId, step.status])).toEqual([
      ['source', 'done'],
      ['scale', 'error'],
      ['count', 'skipped'],
      ['independent', 'done']
    ]);
    expect(calls).toEqual([
      'fixture-source:source',
      'fixture-scale:scale',
      'fixture-count:independent'
    ]);
    expect(runner.getState().steps['scale'].error).toMatchObject({
      code: 'operation-failed',
      message: 'Fixture scale: asked to fail'
    });
    await runner.destroy();
  });

  test('keeps results visible when a step is disabled or no longer configured', async () => {
    const {runner, document} = createFixture();
    await runner.run();
    const result = runner.getState().steps['scale'].result;
    document.steps[1].disabled = true;
    runner.setDocument(document);
    expect(runner.getState().steps['scale']).toMatchObject({status: 'disabled', result});
    expect(runner.getState().steps['count'].status).toBe('blocked');
    document.steps[1].disabled = false;
    document.steps[1].operationId = 'unavailable';
    runner.setDocument(document);
    expect(runner.getState().steps['scale']).toMatchObject({status: 'not-configured', result});
    await runner.destroy();
  });

  test('clearResults drops a step cache; removing and restoring a step does not resurrect its result', async () => {
    const {runner, document, calls} = createFixture();
    await runner.run();
    runner.clearResults('scale');
    expect(readStatuses(runner)).toEqual(['done', 'ready', 'done']);
    calls.length = 0;
    await runner.run();
    expect(calls).toEqual(['fixture-scale:scale']);
    const removed = structuredClone(document);
    removed.steps.splice(1);
    runner.setDocument(removed);
    runner.setDocument(document);
    expect(readStatuses(runner)).toEqual(['done', 'ready', 'ready']);
    runner.clearResults();
    expect(readStatuses(runner)).toEqual(['ready', 'ready', 'ready']);
    await runner.destroy();
  });

  test('copies the document so caller mutation cannot alter an active workflow', async () => {
    const {runner, document} = createFixture();
    document.steps[1].parameters['factor'] = {from: 'static', value: 20};
    await runner.run();
    expect(runner.getState().steps['scale'].result?.parameters['factor']).toBe(3);
    expect(Object.isFrozen(runner.document?.steps)).toBe(true);
    await runner.destroy();
  });

  test('a content cache hit is recorded and published without invoking the operation', async () => {
    const {runner, calls, document, host} = createFixture();
    await runner.run();
    calls.length = 0;
    document.steps.push({
      ...structuredClone(document.steps[0]),
      id: 'copy',
      display: ['data'],
      position: {x: 800, y: 0}
    });
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.steps).toHaveLength(1);
    expect(record.steps[0]).toMatchObject({
      stepId: 'copy',
      status: 'done',
      cacheHit: true,
      inputs: [],
      outputRows: {data: 4}
    });
    expect(calls).toEqual([]);
    expect(host.published[1]).toMatchObject({stepId: 'copy', output: 'data', rowCount: 4});
    expect(runner.getState().steps['copy'].result?.cacheHit).toBe(true);
    await runner.destroy();
  });

  test('a failed force run retains its previous success and exposes only the current error', async () => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-scale')!;
    let fail = false;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => {
          if (fail) throw new WorkflowError('operation-failed', 'Transient failure');
          return original.run(...arguments_);
        }
      }
    ]);
    const {runner, document} = createFixture({registry});
    await runner.run();
    const result = runner.getState().steps['scale'].result;
    fail = true;
    await runner.run({force: true});
    await runner.run({force: true});
    expect(runner.getState().steps['scale'].status).toBe('error');
    expect(runner.getState().steps['scale'].result).toBe(result);
    expect(
      runner.getState().steps['scale'].issues.filter(issue => issue.severity === 'error')
    ).toHaveLength(1);
    runner.setDocument(editFactor(document, 5));
    expect(runner.getState().steps['scale'].status).toBe('stale');
    expect(runner.getState().steps['scale'].error).toBeUndefined();
    await runner.destroy();
  });
});

describe('WorkflowRunner input and output gates', () => {
  test('row capacity belongs to the consumer, names the limit, and prevents execution', async () => {
    const {runner, calls, document} = createFixture();
    document.steps[0].parameters['count'] = {from: 'static', value: 11};
    document.steps[1].label = 'Scale points';
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe('error');
    expect(record.steps[1].error).toMatchObject({
      code: 'capacity-exceeded',
      stepId: 'scale',
      port: 'points',
      message:
        'Scale points: Points allows at most 10 rows; got 11. Add a filter or sort-with-limit step before it.'
    });
    expect(calls).toEqual(['fixture-source:source']);
    await runner.destroy();
  });

  test('vertex capacity is checked before the consumer executes', async () => {
    const calls: string[] = [];
    const registry = replaceOperations(makeFixtureRegistry(calls), [
      {
        id: 'fixture-scale',
        inputs: [
          {name: 'points', label: 'Points', accepts: ['point'], maximumRows: 10, maximumVertices: 3}
        ]
      }
    ]);
    const {runner} = createFixture({registry});
    const record = await runner.run();
    expect(record.steps[1].error).toMatchObject({
      code: 'capacity-exceeded',
      port: 'points',
      message:
        'Fixture scale: Points allows at most 3 vertices; got 4. Add a filter or sort-with-limit step before it.'
    });
    expect(calls).toEqual(['fixture-source:source']);
    await runner.destroy();
  });

  test.each([
    'table',
    'line',
    'features'
  ] as const)('rejects actual %s at a point input', async kind => {
    const data: WorkflowData =
      kind === 'table'
        ? {kind: 'table', table: makeFixtureTable('table', [{id: 'a', value: 1}])}
        : {
            kind,
            table: makeFixtureTable('geometry', [
              {
                id: 'a',
                value: 1,
                geometry: JSON.stringify({
                  type: 'LineString',
                  coordinates: [
                    [0, 0],
                    [1, 1]
                  ]
                })
              },
              ...(kind === 'features'
                ? [
                    {
                      id: 'b',
                      value: 2,
                      geometry: JSON.stringify({type: 'Point', coordinates: [0, 0]})
                    }
                  ]
                : [])
            ])
          };
    const registry = replaceOperations(makeFixtureRegistry(), [
      {id: 'fixture-source', run: async () => makeOutput(data)}
    ]);
    const {runner} = createFixture({registry});
    const record = await runner.run();
    expect(record.steps[1].error).toMatchObject({
      code: kind === 'table' ? 'geometry-required' : 'geometry-kind-not-accepted',
      port: 'points',
      message:
        kind === 'table'
          ? 'Fixture scale: Points requires geometry (point); got table.'
          : `Fixture scale: Points accepts point; got ${kind}.`
    });
    await runner.destroy();
  });

  test('checks table geometry even when a producer mislabels the data kind', async () => {
    const data: WorkflowData = {
      kind: 'point',
      table: makeFixtureTable('table', [{id: 'a', value: 1}])
    };
    const {runner} = createFixture({
      registry: replaceOperations(makeFixtureRegistry(), [
        {id: 'fixture-source', run: async () => makeOutput(data)}
      ])
    });
    expect((await runner.run()).steps[1].error?.code).toBe('geometry-required');
    await runner.destroy();
  });

  test('accepts geometry at an input accepting table', async () => {
    const registry = replaceOperations(makeFixtureRegistry(), [
      {
        id: 'fixture-scale',
        inputs: [{name: 'points', label: 'Rows', accepts: ['table'], maximumRows: 10}]
      }
    ]);
    const {runner} = createFixture({registry});
    expect((await runner.run()).status).toBe('done');
    await runner.destroy();
  });

  test.each([
    'point',
    'features',
    'table'
  ] as const)('rejects an empty %s input before geometry or capacity checks', async kind => {
    const calls: string[] = [];
    const table = makeFixtureTable('empty', []);
    const data: WorkflowData = {
      kind,
      table:
        kind === 'table'
          ? table
          : {
              ...table,
              geometryColumn: 'geometry',
              columns: {...table.columns, geometry: []},
              columnTypes: {...table.columnTypes, geometry: 'string'}
            }
    };
    const registry = replaceOperations(makeFixtureRegistry(calls), [
      {
        id: 'fixture-source',
        run: async context => {
          calls.push(`fixture-source:${context.stepId}`);
          return makeOutput(data);
        }
      }
    ]);
    const {runner, document} = createFixture({registry});
    document.steps[0].label = 'Filtered clusters';
    document.steps[1].label = 'Cluster outlines';
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe('error');
    expect(record.steps.map(step => step.status)).toEqual(['done', 'error', 'skipped']);
    expect(record.steps[1].error).toMatchObject({
      severity: 'error',
      code: 'input-required',
      stepId: 'scale',
      port: 'points',
      message:
        'Cluster outlines: "Points" received no rows from Filtered clusters. Check that step\'s result or its parameters.'
    });
    expect(calls).toEqual(['fixture-source:source']);
    expect(runner.getState().steps['scale'].result).toBeUndefined();
    await runner.destroy();
  });

  test('rejects an empty table at a table input and uses operation titles without labels', async () => {
    const calls: string[] = [];
    const registry = replaceOperations(makeFixtureRegistry(calls), [
      {
        id: 'fixture-source',
        run: async () => makeOutput({kind: 'table', table: makeFixtureTable('empty', [])})
      }
    ]);
    const {runner, document} = createFixture({registry});
    document.steps.splice(1, 1);
    document.steps[1].inputs = {rows: {stepId: 'source', output: 'data'}};
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe('error');
    expect(record.steps[1].error).toMatchObject({
      code: 'input-required',
      stepId: 'count',
      port: 'rows',
      message:
        'Fixture count: "Rows" received no rows from Fixture source. Check that step\'s result or its parameters.'
    });
    expect(calls).toEqual([]);
    await runner.destroy();
  });

  test('allows row and vertex capacities exactly at their limit', async () => {
    const {runner, document} = createFixture();
    document.steps[0].parameters['count'] = {from: 'static', value: 10};
    runner.setDocument(document);
    expect((await runner.run()).status).toBe('done');
    await runner.destroy();
  });

  test.each(['missing', 'rows'] as const)('checks %s output before storing it', async failure => {
    const registry = replaceOperations(makeFixtureRegistry(), [
      {
        id: 'fixture-source',
        run: async () =>
          failure === 'missing' ? {...makeOutput(), outputs: {}} : makeOutput(makeFixturePoints(51))
      }
    ]);
    const {runner} = createFixture({registry});
    const record = await runner.run();
    expect(record.steps[0].error).toMatchObject({
      code: failure === 'missing' ? 'operation-failed' : 'capacity-exceeded',
      port: 'data',
      message:
        failure === 'missing'
          ? 'Fixture source: missing output Data.'
          : 'Fixture source: Data allows at most 50 rows; got 51.'
    });
    expect(runner.getState().steps['source'].result).toBeUndefined();
    expect(record.steps.map(step => step.status)).toEqual(['error', 'skipped', 'skipped']);
    await runner.destroy();
  });

  test.each([
    [new WorkflowError('dataset-read-failed', 'cannot read'), 'dataset-read-failed'],
    [new KeplerAllocationError('capacity', {message: 'Rows exceeds 12'}), 'capacity-exceeded'],
    [new KeplerAllocationError('memory-budget'), 'operation-failed'],
    ['unexpected value', 'operation-failed']
  ] as const)('maps thrown errors and prefixes the label (%s)', async (error, code) => {
    const registry = replaceOperations(makeFixtureRegistry(), [
      {
        id: 'fixture-source',
        run: async () => {
          throw error;
        }
      }
    ]);
    const {runner, document} = createFixture({registry});
    document.steps[0].label = 'Read points';
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.steps[0].error?.code).toBe(code);
    expect(record.steps[0].error?.message.startsWith('Read points: ')).toBe(true);
    await runner.destroy();
  });
});

describe('WorkflowRunner fallback policy', () => {
  test('cache hits apply fallback policy for the consuming step', async () => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-scale')!;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => ({
          ...(await original.run(...arguments_)),
          engine: 'cpu-reference',
          fallbackReason: 'GPU limit is 2 rows'
        })
      }
    ]);
    const host = makeFixtureHost();
    host.getDevice = () => ({}) as Device;
    const {runner, document} = createFixture({host, registry});
    await runner.run();
    const copy: WorkflowDocument['steps'][number] = {
      ...structuredClone(document.steps[1]),
      id: 'copy',
      label: 'Copied scale',
      display: [],
      onFallback: 'stop',
      position: {x: 800, y: 0}
    };
    document.steps.push(copy);
    runner.setDocument(document);
    expect((await runner.run()).steps[0].error).toMatchObject({
      code: 'gpu-fallback',
      message: 'Copied scale: GPU fallback: GPU limit is 2 rows'
    });
    expect(runner.getState().steps['copy'].result).toBeUndefined();
    copy.onFallback = 'warn';
    runner.setDocument(document);
    expect((await runner.run()).steps[0]).toMatchObject({status: 'done', cacheHit: true});
    expect(
      runner.getState().steps['copy'].issues.filter(issue => issue.code === 'gpu-fallback')
    ).toEqual([
      {
        severity: 'warning',
        code: 'gpu-fallback',
        message: 'Copied scale: GPU fallback: GPU limit is 2 rows',
        stepId: 'copy'
      }
    ]);
    await runner.destroy();
  });
  test.each(['warn', 'stop', 'cpu', 'no-device'] as const)('handles %s fallback', async policy => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-scale')!;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => ({
          ...(await original.run(...arguments_)),
          fallbackReason: 'GPU limit is 2 rows',
          engine: 'cpu-reference'
        })
      }
    ]);
    const host = makeFixtureHost();
    if (policy !== 'no-device') host.getDevice = () => ({}) as Device;
    const {runner, document} = createFixture({host, registry});
    if (policy === 'cpu') document.steps[1].engine = 'cpu';
    if (policy === 'stop') document.steps[1].onFallback = 'stop';
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe(policy === 'stop' ? 'error' : 'done');
    const issue = runner
      .getState()
      .steps['scale'].issues.find(entry => entry.code === 'gpu-fallback');
    if (policy === 'warn' || policy === 'stop') {
      expect(issue).toMatchObject({
        severity: policy === 'warn' ? 'warning' : 'error',
        code: 'gpu-fallback',
        message: 'Fixture scale: GPU fallback: GPU limit is 2 rows'
      });
    } else expect(issue).toBeUndefined();
    if (policy === 'stop') expect(runner.getState().steps['scale'].result).toBeUndefined();
    await runner.destroy();
  });
});

describe('WorkflowRunner cancellation and edits during execution', () => {
  test('cancel discards the pending result, records cancellation, and preserves completed upstream', async () => {
    const {runner, entered, release, calls} = makeControlledFixture();
    const running = runner.run();
    await entered.promise;
    expect(readStatuses(runner)).toEqual(['done', 'running', 'queued']);
    runner.cancel();
    release.resolve();
    const record = await running;
    expect(record.status).toBe('cancelled');
    expect(record.steps.map(step => step.status)).toEqual(['done', 'cancelled', 'skipped']);
    expect(readStatuses(runner)).toEqual(['done', 'ready', 'ready']);
    expect(runner.getState().steps['scale'].error).toBeUndefined();
    expect(calls).toEqual(['fixture-source:source', 'fixture-scale:scale']);
    await runner.destroy();
  });

  test('a replacement run waits for the cancelled operation to settle', async () => {
    const {runner, entered, release, calls} = makeControlledFixture();
    const first = runner.run();
    await entered.promise;
    const second = runner.run();
    await flushMicrotasks();
    expect(calls).toEqual(['fixture-source:source']);
    release.resolve();
    expect((await first).status).toBe('cancelled');
    expect((await second).status).toBe('done');
    expect(calls).toEqual([
      'fixture-source:source',
      'fixture-scale:scale',
      'fixture-scale:scale',
      'fixture-count:count'
    ]);
    expect(runner.getState().lastRun?.runId).toBe('fixture-workflow-2');
    await runner.destroy();
  });

  test('multiple pending replacements cancel all but the newest request', async () => {
    const {runner, entered, release} = makeControlledFixture();
    const first = runner.run();
    await entered.promise;
    const second = runner.run();
    const third = runner.run();
    release.resolve();
    expect((await first).status).toBe('cancelled');
    expect(await second).toMatchObject({status: 'cancelled', steps: []});
    expect((await third).status).toBe('done');
    await runner.destroy();
  });

  test('an edit during execution leaves snapshot results stale and the next run uses the edit', async () => {
    const {runner, entered, release, document} = makeControlledFixture();
    const running = runner.run();
    await entered.promise;
    const oldKey = runner.getState().steps['scale'].key;
    runner.setDocument(editFactor(document, 7));
    expect(runner.getState().steps['scale'].status).toBe('ready');
    release.resolve();
    const record = await running;
    expect(record.status).toBe('done');
    expect(readStatuses(runner)).toEqual(['done', 'stale', 'stale']);
    expect(runner.getState().steps['scale'].result?.key).toBe(oldKey);
    expect(runner.getState().steps['scale'].result?.parameters['factor']).toBe(3);
    await runner.run();
    expect(runner.getState().steps['scale'].result?.parameters['factor']).toBe(7);
    await runner.destroy();
  });

  test('aborted operation errors are cancellation rather than step failure', async () => {
    const entered = makeDeferred();
    const registry = replaceOperations(makeFixtureRegistry(), [
      {
        id: 'fixture-source',
        run: async context => {
          entered.resolve();
          await new Promise<void>((_resolve, reject) =>
            context.signal.addEventListener('abort', () => reject(new Error('aborted')), {
              once: true
            })
          );
          return makeOutput();
        }
      }
    ]);
    const {runner} = createFixture({registry});
    const running = runner.run();
    await entered.promise;
    runner.cancel();
    expect((await running).steps[0].status).toBe('cancelled');
    expect(runner.getState().steps['source'].error).toBeUndefined();
    await runner.destroy();
  });
});

describe('WorkflowRunner display lifecycle', () => {
  test('publishes labels, replaces with previousDatasetId, retains stale display, then retracts', async () => {
    const host = makeFixtureHost();
    const requests: WorkflowPublishRequest[] = [];
    const publish = host.publishOutput!;
    host.publishOutput = async (request, signal) => {
      requests.push(request);
      return publish(request, signal);
    };
    const {runner, document} = createFixture({host});
    await runner.run();
    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      workflowId: document.id,
      workflowTitle: document.title,
      stepId: 'scale',
      output: 'scaled',
      label: 'Fixture scale · Scaled',
      previousDatasetId: undefined
    });
    expect(runner.getState().steps['scale'].published?.['scaled']).toEqual({
      datasetId: 'published-1',
      label: 'Fixture scale · Scaled',
      key: runner.getState().steps['scale'].key
    });
    const edited = editFactor(document, 6);
    runner.setDocument(edited);
    expect(runner.getState().steps['scale'].published?.['scaled'].datasetId).toBe('published-1');
    expect(host.retracted).toEqual([]);
    await runner.run();
    expect(requests).toHaveLength(2);
    expect(requests[1].previousDatasetId).toBe('published-1');
    edited.steps[1].display = [];
    runner.setDocument(edited);
    await flushMicrotasks();
    expect(host.retracted).toEqual(['published-1']);
    expect(runner.getState().steps['scale'].published).toBeUndefined();
    await runner.destroy();
  });

  test('turning display on for a done step publishes without an operation run', async () => {
    const {runner, document, host, calls} = createFixture();
    document.steps[1].display = [];
    runner.setDocument(document);
    await runner.run();
    expect(host.published).toEqual([]);
    calls.length = 0;
    document.steps[1].display = ['scaled'];
    runner.setDocument(document);
    await flushMicrotasks();
    expect(host.published).toEqual([
      {stepId: 'scale', output: 'scaled', datasetId: 'published-1', rowCount: 4}
    ]);
    expect(calls).toEqual([]);
    await runner.destroy();
  });

  test('publishes restored cache results when they are run', async () => {
    const {runner, document, host} = createFixture();
    await runner.run();
    runner.setDocument(editFactor(document, 8));
    await runner.run();
    const hidden = structuredClone(document);
    hidden.steps[1].display = [];
    runner.setDocument(hidden);
    await flushMicrotasks();
    hidden.steps[1].display = ['scaled'];
    runner.setDocument(hidden);
    await runner.run();
    await flushMicrotasks();
    expect(host.published).toHaveLength(3);
    expect(runner.getState().steps['scale'].result?.cacheHit).toBe(true);
    expect(runner.getState().steps['scale'].published?.['scaled'].key).toBe(
      runner.getState().steps['scale'].key
    );
    await runner.destroy();
  });

  test('publish failures are warnings and retract failures become document warnings', async () => {
    const host = makeFixtureHost();
    let attempts = 0;
    host.publishOutput = async () => {
      attempts++;
      throw new Error('Map unavailable');
    };
    const {runner, document} = createFixture({host});
    expect((await runner.run()).status).toBe('done');
    await flushMicrotasks();
    expect(attempts).toBe(1);
    expect(runner.getState().steps['scale'].issues).toContainEqual({
      severity: 'warning',
      code: 'publish-failed',
      message: 'Fixture scale: could not display Scaled: Map unavailable',
      stepId: 'scale',
      port: 'scaled'
    });
    expect(runner.getState().steps['scale'].status).toBe('done');
    host.publishOutput = async request => ({datasetId: 'dataset', label: request.label});
    await runner.run({force: true});
    host.retractOutput = async () => {
      throw new Error('cannot retract');
    };
    document.steps[1].display = [];
    runner.setDocument(document);
    await flushMicrotasks();
    expect(runner.getState().issues).toContainEqual({
      severity: 'warning',
      code: 'publish-failed',
      message: 'Fixture workflow: could not retract displayed dataset dataset: cannot retract'
    });
    await runner.destroy();
  });

  test('display is ignored without host support and removing a displayed step retracts it', async () => {
    const host = makeFixtureHost();
    host.publishOutput = undefined;
    const {runner} = createFixture({host});
    expect((await runner.run()).status).toBe('done');
    expect(host.published).toEqual([]);
    await runner.destroy();
    const fixture = createFixture();
    await fixture.runner.run();
    fixture.document.steps.splice(1);
    fixture.runner.setDocument(fixture.document);
    await flushMicrotasks();
    expect(fixture.host.retracted).toEqual(['published-1']);
    await fixture.runner.destroy();
  });

  test('retracts an in-flight publication whose flag was removed', async () => {
    const entered = makeDeferred();
    const release = makeDeferred();
    const host = makeFixtureHost();
    host.publishOutput = async request => {
      entered.resolve();
      await release.promise;
      return {datasetId: 'late', label: request.label};
    };
    const {runner, document} = createFixture({host});
    const running = runner.run();
    await entered.promise;
    document.steps[1].display = [];
    runner.setDocument(document);
    release.resolve();
    await running;
    expect(host.retracted).toEqual(['late']);
    expect(runner.getState().steps['scale'].published).toBeUndefined();
    await runner.destroy();
  });
});

describe('WorkflowRunner cache, subscriptions and lifetime', () => {
  test.each([
    {maximumCachedResults: 0},
    {maximumCachedRows: 3}
  ])('evicts non-latest results by its configured limit (%j)', async limits => {
    const {runner, document} = createFixture(limits);
    await runner.run();
    runner.setDocument(editFactor(document, 4));
    await runner.run();
    runner.setDocument(document);
    expect(runner.getState().steps['source'].status).toBe('done');
    expect(runner.getState().steps['scale'].status).toBe('stale');
    await runner.destroy();
  });

  test('evicts least recently used entries first and protects every latest result', async () => {
    const {runner, document} = createFixture({maximumCachedResults: 2});
    await runner.run();
    const second = editFactor(document, 4);
    runner.setDocument(second);
    await runner.run();
    runner.setDocument(document);
    expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
    runner.setDocument(editFactor(document, 5));
    await runner.run();
    runner.setDocument(second);
    expect(readStatuses(runner)).toEqual(['done', 'stale', 'stale']);
    runner.setDocument(document);
    expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
    await runner.destroy();
  });

  test('queues listener notifications without re-entering listeners and increments revisions once', () => {
    const {runner, document} = createFixture();
    let depth = 0;
    let maximumDepth = 0;
    const revisions: number[] = [];
    let edited = false;
    const unsubscribe = runner.subscribe(() => {
      depth++;
      maximumDepth = Math.max(maximumDepth, depth);
      revisions.push(runner.getState().revision);
      if (!edited) {
        edited = true;
        const disabled = structuredClone(document);
        disabled.steps[0].disabled = true;
        runner.setDocument(disabled);
        expect(runner.getState().steps['source'].status).toBe('disabled');
      }
      depth--;
    });
    const before = runner.getState().revision;
    const unknown = structuredClone(document);
    unknown.steps[2].operationId = 'unknown';
    runner.setDocument(unknown);
    expect(revisions).toEqual([before + 1, before + 2]);
    expect(maximumDepth).toBe(1);
    const snapshot = runner.getState();
    runner.refreshExternal();
    expect(runner.getState()).toBe(snapshot);
    unsubscribe();
    runner.setDocument(document);
    expect(revisions).toHaveLength(2);
  });

  test('destroy retracts, cancels, freezes the final snapshot and makes every method a no-op', async () => {
    const {runner, host, document} = createFixture();
    await runner.run();
    const snapshot = runner.getState();
    let notifications = 0;
    runner.subscribe(() => {
      notifications++;
    });
    await runner.destroy();
    expect(host.retracted).toEqual(['published-1']);
    runner.setDocument(editFactor(document, 20));
    runner.refreshExternal();
    runner.clearResults();
    runner.cancel();
    const unsubscribe = runner.subscribe(() => {
      notifications++;
    });
    unsubscribe();
    await runner.run();
    await runner.destroy();
    expect(notifications).toBe(0);
    expect(runner.getState()).toBe(snapshot);
    expect(host.retracted).toEqual(['published-1']);
  });

  test('destroy while an operation is in flight cannot resurrect results or publications', async () => {
    const {runner, entered, release, host} = makeControlledFixture();
    const running = runner.run();
    await entered.promise;
    const snapshot = runner.getState();
    await runner.destroy();
    release.resolve();
    expect((await running).status).toBe('cancelled');
    expect(runner.getState()).toBe(snapshot);
    expect(host.published).toEqual([]);
  });
});

describe('WorkflowRunner auto-run debounce', () => {
  test('debounces edits, refreshes external revisions and cancels the timer on destroy', async () => {
    vi.useFakeTimers();
    try {
      const {runner, document, calls, host} = createFixture({autoRunDelayMilliseconds: 100});
      document.settings.autoRun = true;
      runner.setDocument(document);
      await vi.advanceTimersByTimeAsync(60);
      expect(calls).toEqual([]);
      runner.setDocument(editFactor(document, 5));
      await vi.advanceTimersByTimeAsync(60);
      expect(calls).toEqual([]);
      await vi.advanceTimersByTimeAsync(40);
      expect(calls).toEqual([
        'fixture-source:source',
        'fixture-scale:scale',
        'fixture-count:count'
      ]);
      expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
      host.revision++;
      runner.refreshExternal();
      await vi.advanceTimersByTimeAsync(100);
      expect(calls).toHaveLength(6);
      runner.setDocument(editFactor(document, 6));
      await runner.destroy();
      await vi.advanceTimersByTimeAsync(100);
      expect(calls).toHaveLength(6);
    } finally {
      vi.useRealTimers();
    }
  });

  test('an edit during a run schedules one rerun only after the running step settles', async () => {
    vi.useFakeTimers();
    try {
      const {runner, entered, release, document, calls} = makeControlledFixture();
      document.settings.autoRun = true;
      runner.setDocument(document);
      await vi.advanceTimersByTimeAsync(400);
      await entered.promise;
      runner.setDocument(editFactor(document, 9));
      await vi.advanceTimersByTimeAsync(1000);
      expect(calls).toEqual(['fixture-source:source']);
      release.resolve();
      await flushMicrotasks();
      await vi.advanceTimersByTimeAsync(399);
      expect(calls).toHaveLength(3);
      await vi.advanceTimersByTimeAsync(1);
      expect(calls).toHaveLength(5);
      expect(runner.getState().steps['scale'].result?.parameters['factor']).toBe(9);
      await runner.destroy();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('WorkflowRunner independent review regressions', () => {
  test('F1: hiding and re-showing an in-flight replacement retains a live publication', async () => {
    const entered = makeDeferred();
    const release = makeDeferred();
    const host = makeFixtureHost();
    const liveDatasets = new Set<string>();
    const requests: WorkflowPublishRequest[] = [];
    const publish = host.publishOutput!;
    host.publishOutput = async (request, signal) => {
      requests.push(request);
      if (requests.length === 2) {
        entered.resolve();
        await release.promise;
      }
      const output = await publish(request, signal);
      liveDatasets.add(output.datasetId);
      return output;
    };
    host.retractOutput = async datasetId => {
      host.retracted.push(datasetId);
      liveDatasets.delete(datasetId);
    };
    const {runner, document} = createFixture({host});
    await runner.run();
    const changed = editFactor(document, 4);
    runner.setDocument(changed);
    const running = runner.run();
    await entered.promise;
    changed.steps[1].display = [];
    runner.setDocument(changed);
    changed.steps[1].display = ['scaled'];
    runner.setDocument(changed);
    release.resolve();
    expect((await running).status).toBe('done');
    await flushMicrotasks();
    const published = runner.getState().steps['scale'].published?.['scaled'];
    expect(published).toMatchObject({
      datasetId: 'published-1',
      key: runner.getState().steps['scale'].key
    });
    expect([...liveDatasets]).toEqual(['published-1']);
    expect(host.retracted).toEqual([]);
    expect(requests).toHaveLength(2);
    expect(requests[1].previousDatasetId).toBe('published-1');
    await runner.destroy();
    expect([...liveDatasets]).toEqual([]);
    expect(host.retracted).toEqual(['published-1']);
  });

  test('F2: auto-run replans new observed kinds and runs newly unblocked consumers', async () => {
    vi.useFakeTimers();
    const calls: string[] = [];
    const base = makeFixtureRegistry(calls);
    const original = base.get('fixture-source')!;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => {
          const output = await original.run(...arguments_);
          if (arguments_[2]['count'] === 4)
            output.outputs['data'] = {
              kind: 'line',
              table: makeFixtureTable('lines', [
                {
                  id: 'a',
                  value: 1,
                  geometry: JSON.stringify({
                    type: 'LineString',
                    coordinates: [
                      [0, 0],
                      [1, 1]
                    ]
                  })
                }
              ])
            };
          return output;
        }
      }
    ]);
    const {runner, document} = createFixture({registry, autoRunDelayMilliseconds: 10});
    try {
      expect((await runner.run()).status).toBe('error');
      document.steps[0].parameters['count'] = {from: 'static', value: 3};
      document.settings.autoRun = true;
      runner.setDocument(document);
      calls.length = 0;
      await vi.advanceTimersByTimeAsync(100);
      expect(runner.getState().steps['source'].result?.outputs['data'].kind).toBe('point');
      expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
      expect(calls).toEqual([
        'fixture-source:source',
        'fixture-scale:scale',
        'fixture-count:count'
      ]);
      expect(runner.getState().lastRun?.steps.map(step => step.stepId)).toEqual(['scale', 'count']);
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      await runner.destroy();
      vi.useRealTimers();
    }
  });

  test('F3: undo cannot restore a fallback under the stop policy', async () => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-scale')!;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => ({
          ...(await original.run(...arguments_)),
          engine: 'cpu-reference',
          fallbackReason: 'GPU capacity exceeded'
        })
      }
    ]);
    const host = makeFixtureHost();
    host.getDevice = () => ({}) as Device;
    const {runner, document} = createFixture({host, registry});
    await runner.run();
    runner.setDocument(editFactor(document, 4));
    await runner.run();
    const retainedResult = runner.getState().steps['scale'].result;
    document.steps[1].onFallback = 'stop';
    runner.setDocument(document);
    expect(runner.getState().steps['scale'].status).toBe('error');
    expect(runner.getState().steps['scale'].result).toBe(retainedResult);
    expect(runner.getState().steps['scale'].error).toMatchObject({
      severity: 'error',
      code: 'gpu-fallback',
      message: 'Fixture scale: GPU fallback: GPU capacity exceeded'
    });
    const record = await runner.run();
    expect(record.status).toBe('error');
    expect(record.steps.map(step => [step.stepId, step.status])).toEqual([['scale', 'error']]);
    expect(runner.getState().steps['count'].result?.cacheHit).toBe(true);
    expect(runner.getState().steps['scale'].result).toBe(retainedResult);
    await runner.destroy();
  });

  test.each([
    42,
    null,
    {
      toString: () => {
        throw new Error('unreadable');
      }
    }
  ])('F4: safely normalizes a non-string Error.message (case %#) and recovers', async message => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-source')!;
    let failures = 0;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => {
          if (++failures <= 2) throw Object.assign(new Error(), {message});
          return original.run(...arguments_);
        }
      }
    ]);
    const {runner} = createFixture({registry});
    try {
      for (let index = 0; index < 2; index++) {
        const record = await runner.run();
        expect(record.status).toBe('error');
        expect(record.steps[0].error).toMatchObject({
          code: 'operation-failed',
          message: `Fixture source: ${typeof message === 'object' && message !== null ? 'Operation failed with an unreadable error.' : String(message)}`
        });
        expect(runner.getState().running).toBe(false);
        expect(runner.getState().lastRun).toBe(record);
      }
      expect((await runner.run()).status).toBe('done');
      expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
    } finally {
      await runner.destroy();
    }
  });

  test('F5: listener cancellation on running prevents invocation', async () => {
    const signalsAtInvocation: boolean[] = [];
    const registry = replaceOperations(makeFixtureRegistry(), [
      {
        id: 'fixture-source',
        run: async context => {
          signalsAtInvocation.push(context.signal.aborted);
          return makeOutput();
        }
      }
    ]);
    const {runner} = createFixture({registry});
    let cancelled = false;
    runner.subscribe(() => {
      if (!cancelled && runner.getState().steps['source'].status === 'running') {
        cancelled = true;
        runner.cancel();
      }
    });
    const record = await runner.run();
    expect(cancelled).toBe(true);
    expect(signalsAtInvocation).toEqual([]);
    expect(record.status).toBe('cancelled');
    expect(record.steps.map(step => step.status)).toEqual(['cancelled', 'skipped', 'skipped']);
    expect(runner.getState().running).toBe(false);
    await runner.destroy();
  });

  test.each([
    5, 7
  ])('F6: total cache row budget %s includes protected latest rows', async maximumCachedRows => {
    const {runner, document} = createFixture({maximumCachedRows});
    document.steps.splice(1);
    document.steps[0].parameters['count'] = {from: 'static', value: 3};
    runner.setDocument(document);
    await runner.run();
    const changed = structuredClone(document);
    changed.steps[0].parameters['count'] = {from: 'static', value: 4};
    runner.setDocument(changed);
    await runner.run();
    expect(runner.getState().steps['source'].result?.outputs['data'].table.rowCount).toBe(4);
    runner.setDocument(document);
    expect(readStatuses(runner)).toEqual([maximumCachedRows === 5 ? 'stale' : 'done']);
    expect(runner.getState().steps['source'].result?.outputs['data'].table.rowCount).toBe(
      maximumCachedRows === 5 ? 4 : 3
    );
    await runner.destroy();
  });

  test.each([
    NaN,
    -1,
    -Infinity,
    Infinity,
    1.5,
    Number.MAX_SAFE_INTEGER + 1
  ])('F7: invalid output row count %s is never retained', async rowCount => {
    const registry = replaceOperations(makeFixtureRegistry(), [
      {
        id: 'fixture-source',
        run: async () => {
          const output = makeOutput();
          output.outputs['data'].table.rowCount = rowCount;
          return output;
        }
      }
    ]);
    const {runner, document} = createFixture({registry});
    document.steps.splice(1);
    runner.setDocument(document);
    const record = await runner.run();
    expect(record.status).toBe('error');
    expect(record.steps[0].error).toMatchObject({
      code: 'operation-failed',
      port: 'data',
      message: `Fixture source: Data row count must be a nonnegative safe integer; got ${rowCount}.`
    });
    expect(runner.getState().steps['source'].result).toBeUndefined();
    expect(runner.getState().steps['source'].status).toBe('error');
    await runner.destroy();
  });

  test('F8: cancellation before execution is observable as lastRun and a later run succeeds', async () => {
    const {runner, calls} = createFixture();
    const running = runner.run();
    runner.cancel();
    const record = await running;
    expect(record).toMatchObject({status: 'cancelled', steps: [], runId: 'fixture-workflow-1'});
    expect(runner.getState().lastRun).toBe(record);
    expect(runner.getState().running).toBe(false);
    expect(calls).toEqual([]);
    expect((await runner.run()).status).toBe('done');
    await runner.destroy();
  });
});

describe('WorkflowRunner review boundary coverage', () => {
  test('F1: hiding an in-flight replacement permanently retracts its live dataset once', async () => {
    const entered = makeDeferred();
    const release = makeDeferred();
    const host = makeFixtureHost();
    const liveDatasets = new Set<string>();
    const publish = host.publishOutput!;
    let publications = 0;
    host.publishOutput = async (request, signal) => {
      if (++publications === 2) {
        entered.resolve();
        await release.promise;
      }
      const output = await publish(request, signal);
      liveDatasets.add(output.datasetId);
      return output;
    };
    host.retractOutput = async datasetId => {
      host.retracted.push(datasetId);
      liveDatasets.delete(datasetId);
    };
    const {runner, document} = createFixture({host});
    await runner.run();
    const changed = editFactor(document, 4);
    runner.setDocument(changed);
    const running = runner.run();
    await entered.promise;
    changed.steps[1].display = [];
    runner.setDocument(changed);
    release.resolve();
    await running;
    await flushMicrotasks();
    expect([...liveDatasets]).toEqual([]);
    expect(host.retracted).toEqual(['published-1']);
    expect(runner.getState().steps['scale'].published).toBeUndefined();
    await runner.destroy();
    expect(host.retracted).toEqual(['published-1']);
  });

  test('F3: changing the policy of the latest fallback also enforces refusal', async () => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-scale')!;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => ({
          ...(await original.run(...arguments_)),
          engine: 'cpu-reference',
          fallbackReason: 'GPU capacity exceeded'
        })
      }
    ]);
    const host = makeFixtureHost();
    host.getDevice = () => ({}) as Device;
    const {runner, document} = createFixture({host, registry});
    await runner.run();
    document.steps[1].onFallback = 'stop';
    runner.setDocument(document);
    expect(runner.getState().steps['scale'].status).toBe('error');
    expect((await runner.run()).status).toBe('error');
    document.steps[1].onFallback = 'warn';
    runner.setDocument(document);
    expect(runner.getState().steps['scale'].status).toBe('done');
    expect(runner.getState().steps['scale'].error).toBeUndefined();
    expect(
      runner.getState().steps['scale'].issues.filter(issue => issue.code === 'gpu-fallback')
    ).toHaveLength(1);
    await runner.destroy();
  });

  test('F4: a failure inside issue normalization still finalizes the task and releases the tail', async () => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-source')!;
    let invocation = 0;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => {
          if (++invocation === 1) {
            const failure = new WorkflowError('dataset-read-failed', 'Original failure');
            Object.defineProperty(failure, 'details', {
              get: () => {
                throw new Error('Cannot normalize details');
              }
            });
            throw failure;
          }
          return original.run(...arguments_);
        }
      }
    ]);
    const {runner} = createFixture({registry});
    const record = await runner.run();
    expect(record.status).toBe('error');
    expect(record.steps[0].error).toMatchObject({
      code: 'operation-failed',
      message: 'Fixture source: Cannot normalize details'
    });
    expect(runner.getState().running).toBe(false);
    expect(runner.getState().runningStepId).toBeUndefined();
    expect(runner.getState().lastRun).toBe(record);
    expect((await runner.run()).status).toBe('done');
    await runner.destroy();
  });

  test('F5: listener destruction on running prevents operation invocation and preserves its last snapshot', async () => {
    const {runner, calls} = createFixture();
    let destroyed: Promise<void> | undefined;
    runner.subscribe(() => {
      if (runner.getState().steps['source'].status === 'running' && !destroyed)
        destroyed = runner.destroy();
    });
    const running = runner.run();
    await flushMicrotasks();
    const snapshot = runner.getState();
    expect((await running).status).toBe('cancelled');
    await destroyed;
    expect(calls).toEqual([]);
    expect(runner.getState()).toBe(snapshot);
  });

  test('F6: excess latest rows are protected while all older results are evicted', async () => {
    const {runner, document} = createFixture({maximumCachedRows: 2});
    document.steps.splice(1);
    document.steps[0].parameters['count'] = {from: 'static', value: 3};
    runner.setDocument(document);
    await runner.run();
    expect(runner.getState().steps['source'].status).toBe('done');
    const changed = structuredClone(document);
    changed.steps[0].parameters['count'] = {from: 'static', value: 4};
    runner.setDocument(changed);
    await runner.run();
    expect(runner.getState().steps['source'].status).toBe('done');
    runner.setDocument(document);
    expect(runner.getState().steps['source'].status).toBe('stale');
    expect(runner.getState().steps['source'].result?.outputs['data'].table.rowCount).toBe(4);
    await runner.destroy();
  });

  test.each([0, 50])('F7: valid output row-count boundary %s succeeds', async count => {
    const {runner, document} = createFixture({
      registry: replaceOperations(makeFixtureRegistry(), [
        {id: 'fixture-source', run: async () => makeOutput(makeFixturePoints(count))}
      ])
    });
    document.steps.splice(1);
    runner.setDocument(document);
    expect((await runner.run()).status).toBe('done');
    expect(runner.getState().steps['source'].result?.outputs['data'].table.rowCount).toBe(count);
    await runner.destroy();
  });

  test('F8: destruction before execution preserves the final snapshot without publishing cancellation', async () => {
    const {runner, calls} = createFixture();
    const running = runner.run();
    const snapshot = runner.getState();
    await runner.destroy();
    expect((await running).status).toBe('cancelled');
    expect(runner.getState()).toBe(snapshot);
    expect(runner.getState().lastRun).toBeUndefined();
    expect(calls).toEqual([]);
  });
});

describe('WorkflowRunner review publication and snapshot stability', () => {
  test('F1: destroying during a replacement retracts the reused live dataset once', async () => {
    const entered = makeDeferred();
    const release = makeDeferred();
    const host = makeFixtureHost();
    const liveDatasets = new Set<string>();
    const publish = host.publishOutput!;
    let publications = 0;
    host.publishOutput = async (request, signal) => {
      if (++publications === 2) {
        entered.resolve();
        await release.promise;
      }
      const output = await publish(request, signal);
      liveDatasets.add(output.datasetId);
      return output;
    };
    host.retractOutput = async datasetId => {
      host.retracted.push(datasetId);
      liveDatasets.delete(datasetId);
    };
    const {runner, document} = createFixture({host});
    await runner.run();
    runner.setDocument(editFactor(document, 4));
    const running = runner.run();
    await entered.promise;
    const snapshot = runner.getState();
    const destroying = runner.destroy();
    release.resolve();
    await destroying;
    expect((await running).status).toBe('cancelled');
    expect([...liveDatasets]).toEqual([]);
    expect(host.retracted).toEqual(['published-1']);
    expect(runner.getState()).toBe(snapshot);
  });

  test('F3: refreshing an unchanged fallback refusal preserves snapshot identity', async () => {
    const base = makeFixtureRegistry();
    const original = base.get('fixture-scale')!;
    const registry = replaceOperations(base, [
      {
        id: original.id,
        run: async (...arguments_) => ({
          ...(await original.run(...arguments_)),
          engine: 'cpu-reference',
          fallbackReason: 'GPU capacity exceeded'
        })
      }
    ]);
    const host = makeFixtureHost();
    host.getDevice = () => ({}) as Device;
    const {runner, document} = createFixture({host, registry});
    await runner.run();
    document.steps[1].onFallback = 'stop';
    runner.setDocument(document);
    const snapshot = runner.getState();
    runner.refreshExternal();
    expect(runner.getState()).toBe(snapshot);
    expect(snapshot.steps['scale'].error?.code).toBe('gpu-fallback');
    await runner.destroy();
  });
});

describe('WorkflowRunner display styles', () => {
  test('publishes style changes and removal without changing keys or running done steps', async () => {
    const host = makeFixtureHost();
    const requests: WorkflowPublishRequest[] = [];
    const publish = host.publishOutput!;
    host.publishOutput = async (request, signal) => {
      requests.push(request);
      return publish(request, signal);
    };
    const {runner, document, calls} = createFixture({host});
    document.steps[1].displayStyle = {
      scaled: {colorColumn: 'value', colorScale: 'quantile', palette: 'sequential'}
    };
    runner.setDocument(document);
    await runner.run();
    expect(requests).toHaveLength(1);
    expect(requests[0].style).toEqual({
      colorColumn: 'value',
      colorScale: 'quantile',
      palette: 'sequential'
    });
    const keys = runner.getState().order.map(stepId => runner.getState().steps[stepId].key);
    const result = runner.getState().steps['scale'].result;
    calls.length = 0;
    document.steps[1].displayStyle = {
      scaled: {palette: 'diverging', colorScale: 'quantize', colorColumn: 'value'}
    };
    runner.setDocument(document);
    await flushMicrotasks();
    expect(requests).toHaveLength(2);
    expect(requests[1]).toMatchObject({
      previousDatasetId: 'published-1',
      style: {colorColumn: 'value', colorScale: 'quantize', palette: 'diverging'}
    });
    expect(runner.getState().steps['scale'].published?.['scaled']).toMatchObject({
      style: requests[1].style
    });
    expect(runner.getState().order.map(stepId => runner.getState().steps[stepId].key)).toEqual(
      keys
    );
    expect(runner.getState().steps['scale'].result).toBe(result);
    const snapshot = runner.getState();
    document.steps[1].displayStyle = {
      scaled: {colorColumn: 'value', colorScale: 'quantize', palette: 'diverging'}
    };
    runner.setDocument(document);
    await flushMicrotasks();
    expect(requests).toHaveLength(2);
    expect(runner.getState()).toBe(snapshot);
    document.steps[1].displayStyle = undefined;
    runner.setDocument(document);
    await flushMicrotasks();
    expect(requests).toHaveLength(3);
    expect(requests[2].previousDatasetId).toBe('published-1');
    expect(requests[2].style).toBeUndefined();
    expect(runner.getState().steps['scale'].published?.['scaled']).not.toHaveProperty('style');
    expect(readStatuses(runner)).toEqual(['done', 'done', 'done']);
    expect(calls).toEqual([]);
    await runner.destroy();
  });

  test('keeps a stale publication until the next run applies its new style', async () => {
    const host = makeFixtureHost();
    const requests: WorkflowPublishRequest[] = [];
    const publish = host.publishOutput!;
    host.publishOutput = async (request, signal) => {
      requests.push(request);
      return publish(request, signal);
    };
    const {runner, document, calls} = createFixture({host});
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'sequential'}};
    runner.setDocument(document);
    await runner.run();
    const published = runner.getState().steps['scale'].published?.['scaled'];
    calls.length = 0;
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'diverging'}};
    runner.setDocument(document);
    const changed = editFactor(document, 4);
    runner.setDocument(changed);
    await flushMicrotasks();
    expect(readStatuses(runner)).toEqual(['done', 'stale', 'stale']);
    expect(requests).toHaveLength(1);
    expect(calls).toEqual([]);
    expect(runner.getState().steps['scale'].published?.['scaled']).toEqual(published);
    await runner.run();
    expect(requests).toHaveLength(2);
    expect(requests[1]).toMatchObject({
      previousDatasetId: 'published-1',
      style: {colorColumn: 'value', palette: 'diverging'}
    });
    expect(calls).toEqual(['fixture-scale:scale', 'fixture-count:count']);
    expect(runner.getState().steps['scale'].published?.['scaled']).toMatchObject({
      key: runner.getState().steps['scale'].key,
      style: requests[1].style
    });
    await runner.destroy();
  });

  test('does not deduplicate a new style against an in-flight publication of the same result', async () => {
    const entered = makeDeferred();
    const release = makeDeferred();
    const host = makeFixtureHost();
    const requests: WorkflowPublishRequest[] = [];
    const publish = host.publishOutput!;
    host.publishOutput = async (request, signal) => {
      requests.push(request);
      if (requests.length === 1) {
        entered.resolve();
        await release.promise;
      }
      return publish(request, signal);
    };
    const {runner, document, calls} = createFixture({host});
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'sequential'}};
    runner.setDocument(document);
    const running = runner.run();
    await entered.promise;
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'diverging'}};
    runner.setDocument(document);
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'categorical'}};
    runner.setDocument(document);
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'diverging'}};
    runner.setDocument(document);
    release.resolve();
    expect((await running).status).toBe('done');
    await flushMicrotasks();
    expect(requests.map(request => request.style?.palette)).toEqual(['sequential', 'diverging']);
    expect(requests[1].previousDatasetId).toBe('published-1');
    expect(runner.getState().steps['scale'].published?.['scaled']).toMatchObject({
      style: {colorColumn: 'value', palette: 'diverging'}
    });
    expect(calls).toEqual(['fixture-source:source', 'fixture-scale:scale', 'fixture-count:count']);
    await runner.destroy();
  });
});

describe('WorkflowRunner second-wave review regressions', () => {
  test('F1: a queued style retry after a failed replacement cannot publish a stale result', async () => {
    const host = makeFixtureHost();
    const requests: WorkflowPublishRequest[] = [];
    const publish = host.publishOutput!;
    host.publishOutput = async (request, signal) => {
      requests.push(request);
      if (requests.length === 2) throw new Error('Replacement failed');
      return publish(request, signal);
    };
    const {runner, document, calls} = createFixture({host});
    await runner.run();
    const published = runner.getState().steps['scale'].published?.['scaled'];
    const changed = editFactor(document, 4);
    runner.setDocument(changed);
    expect((await runner.run()).status).toBe('done');
    expect(requests).toHaveLength(2);
    expect(runner.getState().steps['scale'].issues).toContainEqual({
      severity: 'warning',
      code: 'publish-failed',
      message: 'Fixture scale: could not display Scaled: Replacement failed',
      stepId: 'scale',
      port: 'scaled'
    });
    calls.length = 0;
    changed.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'diverging'}};
    runner.setDocument(changed);
    const editedAgain = editFactor(changed, 5);
    runner.setDocument(editedAgain);
    await flushMicrotasks();
    expect(runner.getState().steps['scale'].status).toBe('stale');
    expect(requests).toHaveLength(2);
    expect(calls).toEqual([]);
    expect(runner.getState().steps['scale'].published?.['scaled']).toEqual(published);
    expect(runner.getState().steps['scale'].result?.parameters['factor']).toBe(4);
    expect((await runner.run()).status).toBe('done');
    expect(requests).toHaveLength(3);
    expect(requests[2]).toMatchObject({
      previousDatasetId: 'published-1',
      style: {colorColumn: 'value', palette: 'diverging'}
    });
    expect(requests[2].data.table.columns['value']).toEqual([0, 7.5, 15, 22.5]);
    expect(calls).toEqual(['fixture-scale:scale', 'fixture-count:count']);
    await runner.destroy();
  });

  test('F6: restoring the successfully published style clears a failed-style warning without republishing', async () => {
    const host = makeFixtureHost();
    const requests: WorkflowPublishRequest[] = [];
    const publish = host.publishOutput!;
    host.publishOutput = async (request, signal) => {
      requests.push(request);
      if (request.style?.palette === 'diverging') throw new Error('diverging failed');
      return publish(request, signal);
    };
    const {runner, document, calls} = createFixture({host});
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'sequential'}};
    runner.setDocument(document);
    await runner.run();
    const result = runner.getState().steps['scale'].result;
    const published = runner.getState().steps['scale'].published?.['scaled'];
    calls.length = 0;
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'diverging'}};
    runner.setDocument(document);
    await flushMicrotasks();
    expect(
      runner.getState().steps['scale'].issues.filter(issue => issue.code === 'publish-failed')
    ).toEqual([
      {
        severity: 'warning',
        code: 'publish-failed',
        message: 'Fixture scale: could not display Scaled: diverging failed',
        stepId: 'scale',
        port: 'scaled'
      }
    ]);
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value', palette: 'sequential'}};
    runner.setDocument(document);
    await flushMicrotasks();
    expect(requests.map(request => request.style?.palette)).toEqual(['sequential', 'diverging']);
    expect(
      runner.getState().steps['scale'].issues.filter(issue => issue.code === 'publish-failed')
    ).toEqual([]);
    expect(runner.getState().steps['scale'].status).toBe('done');
    expect(runner.getState().steps['scale'].published?.['scaled']).toEqual(published);
    expect(runner.getState().steps['scale'].result).toBe(result);
    expect(calls).toEqual([]);
    await runner.destroy();
  });
});
