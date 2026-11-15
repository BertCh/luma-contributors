// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {normalizeWorkflowParameters} from '../../src/workflows/workflow-parameters';
import {planWorkflow} from '../../src/workflows/workflow-plan';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {getWorkflowSourceOperations} from '../../src/workflows/workflow-source-operations';
import {
  getWorkflowSuggestions,
  type WorkflowSuggestion
} from '../../src/workflows/workflow-suggestions';
import {getWorkflowTableOperations} from '../../src/workflows/workflow-table-operations';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowData,
  type WorkflowDocument,
  type WorkflowIssue,
  type WorkflowOperation,
  type WorkflowOperationRegistry,
  type WorkflowRunnerState
} from '../../src/workflows/workflow-types';
import {
  makeFixtureDocument,
  makeFixtureHost,
  makeFixturePoints,
  makeFixtureRegistry
} from './workflow-fixtures';

function makeRegistry(overrides: WorkflowOperation[] = []): WorkflowOperationRegistry {
  const operations = [
    ...makeFixtureRegistry().list(),
    ...getWorkflowTableOperations(),
    ...getWorkflowSourceOperations()
  ];
  return createWorkflowOperationRegistry([
    ...operations.filter(operation => !overrides.some(override => override.id === operation.id)),
    ...overrides
  ]);
}

function modifyOperation(id: string, patch: Partial<WorkflowOperation>): WorkflowOperation {
  const operation = {...makeRegistry().get(id)!, ...patch};
  operation.normalizeParameters = raw =>
    normalizeWorkflowParameters(operation.parameters, raw, {
      operationTitle: operation.title
    });
  return operation;
}

function makeState(outputs: Record<string, Record<string, WorkflowData>>): WorkflowRunnerState {
  return {
    revision: 1,
    running: false,
    order: Object.keys(outputs),
    issues: [],
    steps: Object.fromEntries(
      Object.entries(outputs).map(([stepId, data]) => [
        stepId,
        {
          status: 'done',
          key: 'key',
          issues: [],
          result: {
            key: 'key',
            outputs: data,
            diagnostics: [],
            summary: {},
            engine: 'cpu-reference',
            dispatchedContributors: [],
            parameters: {},
            durationMilliseconds: 0,
            cacheHit: false,
            completedAt: 0
          }
        }
      ])
    )
  };
}

function makeJoinState(): WorkflowRunnerState {
  const target = makeFixturePoints(4);
  return makeState({
    source: {data: target},
    scale: {
      scaled: {
        kind: 'table',
        identityColumn: 'source_id',
        table: {
          id: 'scaled',
          label: 'Attributes',
          rowCount: 4,
          columns: {source_id: target.table.columns['point_id'], value: [10, 20, 30, 40]},
          columnTypes: {source_id: 'string', value: 'float64'}
        }
      }
    }
  });
}

function freezeValue(value: unknown): void {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freezeValue(child);
    Object.freeze(value);
  }
}

function suggest(
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry,
  details: Omit<WorkflowIssue, 'severity' | 'message'> & {message?: string},
  state: WorkflowRunnerState | null = null
): WorkflowSuggestion[] {
  const original = structuredClone(document);
  freezeValue(document);
  const suggestions = getWorkflowSuggestions(document, registry, state, {
    severity: 'error',
    message: 'Fixture issue.',
    ...details
  });
  expect(document).toEqual(original);
  expect(suggestions.length).toBeLessThanOrEqual(3);
  expect(new Set(suggestions.map(suggestion => suggestion.id)).size).toBe(suggestions.length);
  for (const suggestion of suggestions) {
    expect(suggestion.label.length).toBeLessThanOrEqual(48);
    expect(suggestion.id).toContain(`:${details.stepId}`);
  }
  return suggestions;
}

function apply(
  suggestion: WorkflowSuggestion,
  document: WorkflowDocument,
  registry: WorkflowOperationRegistry
): WorkflowDocument {
  const original = structuredClone(document);
  const result = suggestion.apply(document, registry);
  expect(document).toEqual(original);
  expect(result).not.toBe(document);
  return result;
}

function expectSeparatedSteps(document: WorkflowDocument): void {
  for (const [index, first] of document.steps.entries()) {
    for (const second of document.steps.slice(index + 1)) {
      const horizontalDistance = Math.abs(first.position.x - second.position.x);
      const verticalDistance = Math.abs(first.position.y - second.position.y);
      expect(
        horizontalDistance >= 220 || verticalDistance >= 100,
        `${first.id} overlaps ${second.id}`
      ).toBe(true);
    }
  }
}

const registry = makeRegistry();

describe('workflow issue suggestions', () => {
  test.each([
    'capacity-exceeded',
    'capacity-may-exceed'
  ] as const)('%s inserts a row limit or editable filter and makes room downstream', async code => {
    const document = makeFixtureDocument();
    const suggestions = suggest(document, registry, {code, stepId: 'scale', port: 'points'});
    expect(suggestions.map(suggestion => suggestion.label)).toEqual([
      'Insert Sort rows with a limit',
      'Insert Filter rows'
    ]);
    for (const [index, suggestion] of suggestions.entries()) {
      const result = apply(suggestion, document, registry);
      const inserted = result.steps[3];
      expect(result.steps).toHaveLength(4);
      expect(inserted.operationId).toBe(index === 0 ? 'table-sort-rows' : 'table-filter-rows');
      expect(inserted.position).toEqual({x: 280, y: 0});
      expect(result.steps.slice(0, 3).map(step => step.position)).toEqual([
        {x: 0, y: 0},
        {x: 560, y: 0},
        {x: 840, y: 0}
      ]);
      expectSeparatedSteps(result);
      expect(inserted.inputs).toEqual({rows: {stepId: 'source', output: 'data'}});
      expect(inserted.parameters).toEqual(
        index === 0
          ? {
              column: {from: 'static', value: 'point_id'},
              limit: {from: 'static', value: 10}
            }
          : {expression: {from: 'static', value: 'true'}}
      );
      expect(result.steps[1].inputs).toEqual({
        points: {
          stepId: inserted.id,
          output: index === 0 ? 'sorted' : 'filtered'
        }
      });
      expect(planWorkflow(result, {registry}).steps[inserted.id].issues).toEqual([]);
      if (index === 0) {
        const output = await registry.get(inserted.operationId)!.run(
          {
            device: null,
            engine: 'cpu',
            host: makeFixtureHost(),
            signal: new AbortController().signal,
            workflowId: document.id,
            stepId: inserted.id
          },
          {rows: makeFixturePoints(15)},
          {column: 'point_id', limit: 10}
        );
        expect(output.outputs['sorted'].table.rowCount).toBe(10);
      }
    }
  });

  test('row repairs move all transitive descendants and leave other producer branches in place', () => {
    const document = makeFixtureDocument();
    document.steps.push(
      {
        id: 'last',
        operationId: 'fixture-count',
        parameters: {},
        inputs: {rows: {stepId: 'count', output: 'count'}},
        position: {x: 840, y: 0}
      },
      {
        id: 'sibling',
        operationId: 'fixture-scale',
        parameters: {},
        inputs: {points: {stepId: 'source', output: 'data'}},
        position: {x: 280, y: 300}
      }
    );
    const suggestions = suggest(document, registry, {
      code: 'capacity-exceeded',
      stepId: 'scale',
      port: 'points'
    });
    expect(suggestions).toHaveLength(2);
    for (const suggestion of suggestions) {
      const result = apply(suggestion, document, registry);
      expect(result.steps.map(step => [step.id, step.position])).toEqual([
        ['source', {x: 0, y: 0}],
        ['scale', {x: 560, y: 0}],
        ['count', {x: 840, y: 0}],
        ['last', {x: 1120, y: 0}],
        ['sibling', {x: 280, y: 300}],
        [result.steps[5].id, {x: 280, y: 0}]
      ]);
      expectSeparatedSteps(result);
    }
  });

  test('row repairs use the consumer position at application time', () => {
    const document = makeFixtureDocument();
    const [suggestion] = suggest(document, registry, {
      code: 'capacity-exceeded',
      stepId: 'scale',
      port: 'points'
    });
    const current = structuredClone(document);
    current.steps[1].position = {x: 400, y: 200};
    current.steps[2].position = {x: 680, y: 200};
    const result = apply(suggestion, current, registry);
    expect(result.steps.map(step => step.position)).toEqual([
      {x: 0, y: 0},
      {x: 680, y: 200},
      {x: 960, y: 200},
      {x: 400, y: 200}
    ]);
    expectSeparatedSteps(result);
  });

  test('row limiting falls back to latest columns when the output declares no identity', () => {
    const source = modifyOperation('fixture-source', {
      outputs: [
        {
          name: 'data',
          label: 'Data',
          kind: 'features',
          identityColumns: [],
          maximumRows: 50
        }
      ]
    });
    const currentRegistry = makeRegistry([source]);
    const document = makeFixtureDocument();
    const state = makeState({source: {data: makeFixturePoints(15)}});
    const [suggestion] = suggest(
      document,
      currentRegistry,
      {
        code: 'capacity-exceeded',
        stepId: 'scale',
        port: 'points'
      },
      state
    );
    const result = apply(suggestion, document, currentRegistry);
    expect(result.steps[3].parameters['column']).toEqual({from: 'static', value: 'point_id'});
    expect(
      suggest(document, currentRegistry, {
        code: 'capacity-exceeded',
        stepId: 'scale',
        port: 'points'
      }).map(candidate => candidate.label)
    ).toEqual(['Insert Filter rows']);
  });

  test('geometry required joins producer attributes with explicit keys and collision-free names', async () => {
    const attributes = modifyOperation('fixture-scale', {
      outputs: [
        {
          name: 'scaled',
          label: 'Attributes',
          kind: 'table',
          identityColumns: ['source_id'],
          maximumRows: 10
        }
      ]
    });
    const currentRegistry = makeRegistry([attributes]);
    const document = makeFixtureDocument();
    document.steps[2].operationId = 'fixture-scale';
    document.steps[2].inputs = {points: {stepId: 'scale', output: 'scaled'}};
    document.steps[0].label = 'Original features';
    const state = makeJoinState();
    state.steps['source'].result!.outputs['data'].table.columns['attributes_value'] = [5, 5, 5, 5];
    state.steps['source'].result!.outputs['data'].table.columnTypes['attributes_value'] = 'float64';
    const [suggestion] = suggest(
      document,
      currentRegistry,
      {
        code: 'geometry-required',
        stepId: 'count',
        port: 'points'
      },
      state
    );
    expect(suggestion.label).toBe('Join back to Original features');
    const result = apply(suggestion, document, currentRegistry);
    expect(result.steps[3]).toMatchObject({
      operationId: 'table-join-attributes',
      position: {x: 280, y: 150},
      inputs: {
        target: {stepId: 'source', output: 'data'},
        attributes: {stepId: 'scale', output: 'scaled'}
      }
    });
    expect(result.steps[3].parameters).toEqual({
      targetKey: {from: 'static', value: 'point_id'},
      attributesKey: {from: 'static', value: 'source_id'},
      columns: {from: 'static', value: ''},
      prefix: {from: 'static', value: 'attributes2_'},
      unmatched: {from: 'static', value: 'keep'}
    });
    const joined = await currentRegistry.get('table-join-attributes')!.run(
      {
        device: null,
        engine: 'cpu',
        host: makeFixtureHost(),
        signal: new AbortController().signal,
        workflowId: document.id,
        stepId: result.steps[3].id
      },
      {
        target: state.steps['source'].result!.outputs['data'],
        attributes: state.steps['scale'].result!.outputs['scaled']
      },
      planWorkflow(result, {registry: currentRegistry}).steps[result.steps[3].id].parameters!
    );
    expect(joined.outputs['joined'].table.columns['attributes2_value']).toEqual([10, 20, 30, 40]);
    expect(joined.outputs['joined'].table.columns['attributes_value']).toEqual([5, 5, 5, 5]);
    expect(joined.outputs['joined'].table.columns['value']).toEqual([0, 1.5, 3, 4.5]);
    expect(joined.outputs['joined'].kind).toBe('point');
    expect(result.steps[2].inputs['points']).toEqual({
      stepId: result.steps[3].id,
      output: 'joined'
    });
    expect(result.steps.slice(0, 3).map(step => step.position)).toEqual(
      document.steps.map(step => step.position)
    );
    expectSeparatedSteps(result);
  });

  test('F3: joining a summary grouped by identity runs the repaired chain to done', async () => {
    const document = makeFixtureDocument();
    document.steps.splice(1, 0, {
      id: 'summary',
      operationId: 'table-summarize',
      inputs: {rows: {stepId: 'source', output: 'data'}},
      parameters: {groupBy: {from: 'static', value: 'point_id'}},
      position: {x: 280, y: 0}
    });
    document.steps[2].inputs = {points: {stepId: 'summary', output: 'summary'}};
    document.steps[2].position.x = 560;
    document.steps[3].position.x = 840;
    const runner = new WorkflowRunner({host: makeFixtureHost(), registry, now: () => 0});
    try {
      runner.setDocument(document);
      expect((await runner.run()).status).toBe('error');
      const state = runner.getState();
      expect(state.steps['scale'].error?.code).toBe('geometry-required');
      const [suggestion] = suggest(
        document,
        registry,
        {
          code: 'geometry-required',
          stepId: 'scale',
          port: 'points'
        },
        state
      );
      expect(suggestion.label).toBe('Join back to Fixture source');
      const result = apply(suggestion, document, registry);
      runner.setDocument(result);
      const record = await runner.run();
      expect(record.status, JSON.stringify(record.steps.flatMap(step => step.error ?? []))).toBe(
        'done'
      );
      expect(runner.getState().steps['scale'].status).toBe('done');
      expect(runner.getState().steps['count'].status).toBe('done');
      expect(
        runner.getState().steps[result.steps[4].id].result?.outputs['joined'].table.columns[
          'row_count'
        ]
      ).toEqual([1, 1, 1, 1]);
    } finally {
      await runner.destroy();
    }
  });

  test.each([
    'missing results',
    'missing attributes key',
    'missing target key',
    'stale attributes',
    'stale target',
    'wrong geometry',
    'too many target rows'
  ])('join repairs are unavailable with %s', condition => {
    const currentRegistry = makeRegistry([
      modifyOperation('fixture-scale', {
        outputs: [
          {
            name: 'scaled',
            label: 'Attributes',
            kind: 'table',
            identityColumns: ['source_id'],
            maximumRows: 10
          }
        ]
      })
    ]);
    const document = makeFixtureDocument();
    document.steps[2].operationId = 'fixture-scale';
    document.steps[2].inputs = {points: {stepId: 'scale', output: 'scaled'}};
    const state = makeJoinState();
    if (condition === 'missing attributes key') {
      delete state.steps['scale'].result!.outputs['scaled'].table.columns['source_id'];
    } else if (condition === 'missing target key') {
      delete state.steps['source'].result!.outputs['data'].table.columns['point_id'];
    } else if (condition === 'stale attributes') {
      state.steps['scale'].key = 'new-key';
      state.steps['scale'].status = 'stale';
    } else if (condition === 'stale target') {
      state.steps['source'].key = 'new-key';
      state.steps['source'].status = 'stale';
    } else if (condition === 'wrong geometry') {
      state.steps['source'].result!.outputs['data'].kind = 'polygon';
    } else if (condition === 'too many target rows') {
      state.steps['source'].result!.outputs['data'] = makeFixturePoints(11);
    }
    expect(
      suggest(
        document,
        currentRegistry,
        {
          code: 'geometry-required',
          stepId: 'count',
          port: 'points'
        },
        condition === 'missing results' ? null : state
      )
    ).toEqual([]);
  });

  test('join repairs refuse application after upstream parameters change', () => {
    const currentRegistry = makeRegistry([
      modifyOperation('fixture-scale', {
        outputs: [
          {
            name: 'scaled',
            label: 'Attributes',
            kind: 'table',
            identityColumns: ['source_id'],
            maximumRows: 10
          }
        ]
      })
    ]);
    const document = makeFixtureDocument();
    document.steps[2].operationId = 'fixture-scale';
    document.steps[2].inputs = {points: {stepId: 'scale', output: 'scaled'}};
    const [suggestion] = suggest(
      document,
      currentRegistry,
      {
        code: 'geometry-required',
        stepId: 'count',
        port: 'points'
      },
      makeJoinState()
    );
    const current = structuredClone(document);
    current.steps[1].parameters['factor'] = {from: 'static', value: 5};
    expect(() => suggestion.apply(current, currentRegistry)).toThrow(
      expect.objectContaining({
        code: 'upstream-not-configured',
        details: {stepId: 'count', port: 'points'}
      })
    );
    expect(current.steps).toHaveLength(3);
    expect(current.steps[2].inputs).toEqual({points: {stepId: 'scale', output: 'scaled'}});
  });

  test('missing inputs offer the two nearest compatible outputs and refuse descendants', () => {
    const document = makeFixtureDocument();
    document.steps[1].inputs = {};
    document.steps.push({
      id: 'descendant',
      operationId: 'fixture-scale',
      parameters: {},
      inputs: {points: {stepId: 'scale', output: 'scaled'}},
      position: {x: 281, y: 0}
    });
    document.steps.push(
      ...['near', 'far', 'farthest'].map((id, index) => ({
        id,
        label: id,
        operationId: 'fixture-source',
        inputs: {},
        parameters: {},
        position: {x: 300 + index * 100, y: 0}
      }))
    );
    const pointSource = modifyOperation('fixture-source', {
      outputs: [
        {
          name: 'data',
          label: 'Data',
          kind: 'point',
          identityColumns: [],
          maximumRows: 50
        }
      ]
    });
    const currentRegistry = makeRegistry([pointSource]);
    const suggestions = suggest(document, currentRegistry, {
      code: 'input-required',
      stepId: 'scale',
      port: 'points'
    });
    expect(suggestions.map(suggestion => suggestion.label)).toEqual([
      'Connect near · Data',
      'Connect far · Data'
    ]);
    for (const [index, suggestion] of suggestions.entries()) {
      const result = apply(suggestion, document, currentRegistry);
      expect(result.steps[1].inputs).toEqual({
        points: {stepId: index === 0 ? 'near' : 'far', output: 'data'}
      });
    }
  });

  test('missing inputs can add a GeoJSON source when its defaults plan safely', () => {
    const source = modifyOperation('source-geojson', {parameters: []});
    const currentRegistry = makeRegistry([source]);
    const document = makeFixtureDocument();
    document.steps = [document.steps[1]];
    document.steps[0].inputs = {};
    const [suggestion] = suggest(document, currentRegistry, {
      code: 'input-required',
      stepId: 'scale',
      port: 'points'
    });
    expect(suggestion.label).toBe('Add a GeoJSON source');
    const result = apply(suggestion, document, currentRegistry);
    expect(result.steps[1]).toMatchObject({
      operationId: 'source-geojson',
      position: {x: 0, y: 150}
    });
    expect(result.steps[0].inputs).toEqual({points: {stepId: result.steps[1].id, output: 'data'}});
    expect(result.steps[0].position).toEqual(document.steps[0].position);
    expectSeparatedSteps(result);
  });

  test.each([
    'source',
    'join'
  ] as const)('%s repairs sit below the lowest current feeder without moving steps', kind => {
    const originalScale = registry.get('fixture-scale')!;
    const scale = modifyOperation('fixture-scale', {
      inputs: [
        ...originalScale.inputs,
        {
          name: 'context',
          label: 'Context',
          accepts: ['table'],
          maximumRows: 100,
          optional: true
        }
      ],
      ...(kind === 'join'
        ? {
            outputs: [
              {
                name: 'scaled',
                label: 'Attributes',
                kind: 'table',
                identityColumns: ['source_id'],
                maximumRows: 10
              }
            ]
          }
        : {})
    });
    const currentRegistry = makeRegistry([
      scale,
      modifyOperation('source-geojson', {parameters: []})
    ]);
    const document = makeFixtureDocument();
    if (kind === 'join') {
      document.steps[2].operationId = 'fixture-scale';
      document.steps[2].inputs = {points: {stepId: 'scale', output: 'scaled'}};
    } else {
      document.steps[1].inputs = {};
    }
    const consumer = document.steps[kind === 'join' ? 2 : 1];
    consumer.inputs['context'] = {stepId: 'lower-source', output: 'data'};
    document.steps.push({
      id: 'lower-source',
      operationId: 'fixture-source',
      parameters: {},
      inputs: {},
      position: {x: 0, y: 300}
    });
    const [suggestion] = suggest(
      document,
      currentRegistry,
      {
        code: kind === 'join' ? 'geometry-required' : 'input-required',
        stepId: consumer.id,
        port: 'points'
      },
      kind === 'join' ? makeJoinState() : null
    );
    expect(suggestion.label).toBe(
      kind === 'join' ? 'Join back to Fixture source' : 'Add a GeoJSON source'
    );
    const current = structuredClone(document);
    current.steps[3].position.y = 450;
    const result = apply(suggestion, current, currentRegistry);
    expect(result.steps[4].position).toEqual({x: consumer.position.x - 280, y: 600});
    expect(result.steps.slice(0, 4).map(step => step.position)).toEqual(
      current.steps.map(step => step.position)
    );
    expectSeparatedSteps(result);
  });

  test('incompatible ports can be disconnected when optional', () => {
    const scale = modifyOperation('fixture-scale', {
      inputs: [
        {
          name: 'points',
          label: 'Points',
          accepts: ['point'],
          maximumRows: 10,
          optional: true
        }
      ]
    });
    const source = modifyOperation('fixture-source', {
      outputs: [
        {
          name: 'data',
          label: 'Data',
          kind: 'polygon',
          maximumRows: 50,
          identityColumns: []
        }
      ]
    });
    const currentRegistry = makeRegistry([scale, source]);
    const document = makeFixtureDocument();
    const [suggestion] = suggest(document, currentRegistry, {
      code: 'incompatible-port',
      stepId: 'scale',
      port: 'points'
    });
    expect(suggestion.label).toBe('Disconnect "Points"');
    expect(apply(suggestion, document, currentRegistry).steps[1].inputs).toEqual({});
    expect(
      suggest(document, makeRegistry([source]), {
        code: 'incompatible-port',
        stepId: 'scale',
        port: 'points'
      })
    ).toEqual([]);
  });

  test('a missing workflow input resets the parameter binding to its default', () => {
    const document = makeFixtureDocument();
    document.steps[1].parameters['factor'] = {from: 'input', inputId: 'missing'};
    const [suggestion] = suggest(document, registry, {
      code: 'unknown-workflow-input',
      stepId: 'scale',
      parameter: 'factor'
    });
    expect(suggestion.label).toBe('Reset "Factor" to its default');
    const result = apply(suggestion, document, registry);
    expect(result.steps[1].parameters).toEqual({});
    expect(planWorkflow(result, {registry}).steps['scale'].parameters?.['factor']).toBe(2);
  });

  test.each([
    [-5, 0],
    [101, 100]
  ])('out of range %s clamps to %s', (value, bound) => {
    const document = makeFixtureDocument();
    document.steps[1].parameters['factor'] = {from: 'static', value};
    const [suggestion] = suggest(document, registry, {
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'factor'
    });
    expect(suggestion.label).toBe(`Set "Factor" to ${bound}`);
    expect(apply(suggestion, document, registry).steps[1].parameters['factor']).toEqual({
      from: 'static',
      value: bound
    });
  });

  test('unknown columns prefer case-insensitive equality then edit distance, limited to three', () => {
    const document = makeFixtureDocument();
    document.steps = [
      document.steps[0],
      {
        id: 'sort',
        operationId: 'table-sort-rows',
        inputs: {rows: {stepId: 'source', output: 'data'}},
        parameters: {column: {from: 'static', value: 'VALUE'}},
        position: {x: 280, y: 0}
      }
    ];
    const data = makeFixturePoints(1);
    data.table.columns['values'] = [1];
    data.table.columns['unrelated'] = [2];
    const state = makeState({source: {data}});
    const details = {
      code: 'parameter-out-of-range' as const,
      stepId: 'sort',
      parameter: 'column',
      message: 'Sort rows: Column VALUE does not exist.'
    };
    const suggestions = suggest(document, registry, details, state);
    expect(suggestions.map(suggestion => suggestion.label)).toEqual([
      'Use column "value"',
      'Use column "values"'
    ]);
    for (const [index, suggestion] of suggestions.entries())
      expect(apply(suggestion, document, registry).steps[1].parameters['column']).toEqual({
        from: 'static',
        value: index === 0 ? 'value' : 'values'
      });
    expect(suggest(document, registry, details)).toEqual([]);
    expect(
      suggest(document, registry, {...details, message: 'Sort rows: Column is invalid.'}, state)
    ).toEqual([]);
  });

  test('column repairs resolve workflow bindings and refuse names beyond edit distance three', () => {
    const document = makeFixtureDocument();
    document.inputs.push({id: 'column-name', label: 'Column', kind: 'string', value: 'abcde'});
    document.steps = [
      document.steps[0],
      {
        id: 'sort',
        operationId: 'table-sort-rows',
        inputs: {rows: {stepId: 'source', output: 'data'}},
        parameters: {column: {from: 'input', inputId: 'column-name'}},
        position: {x: 280, y: 0}
      }
    ];
    const data = makeFixturePoints(1);
    data.table.columns = {abxyz: [1], axyzz: [2]};
    const [suggestion, ...remaining] = suggest(
      document,
      registry,
      {
        code: 'parameter-out-of-range',
        stepId: 'sort',
        parameter: 'column',
        message: 'Sort rows: Column abcde does not exist.'
      },
      makeState({source: {data}})
    );
    expect(remaining).toEqual([]);
    expect(suggestion.label).toBe('Use column "abxyz"');
    expect(apply(suggestion, document, registry).steps[1].parameters['column']).toEqual({
      from: 'static',
      value: 'abxyz'
    });
  });

  test('integer bounds are repaired, while workflow-bound numbers are left unchanged', () => {
    const document = makeFixtureDocument();
    document.steps[0].parameters['count'] = {from: 'static', value: 51};
    const [suggestion] = suggest(document, registry, {
      code: 'parameter-out-of-range',
      stepId: 'source',
      parameter: 'count'
    });
    expect(suggestion.label).toBe('Set "Point count" to 50');
    expect(apply(suggestion, document, registry).steps[0].parameters['count']).toEqual({
      from: 'static',
      value: 50
    });
    const bound = makeFixtureDocument();
    bound.inputs.push({id: 'factor', label: 'Factor', kind: 'number', value: -5});
    bound.steps[1].parameters['factor'] = {from: 'input', inputId: 'factor'};
    expect(
      suggest(bound, registry, {
        code: 'parameter-out-of-range',
        stepId: 'scale',
        parameter: 'factor'
      })
    ).toEqual([]);
  });

  test('upstream disabled enables the nearest disabled ancestor', () => {
    const document = makeFixtureDocument();
    document.steps[0].disabled = true;
    document.steps[1].disabled = true;
    const [suggestion] = suggest(document, registry, {code: 'upstream-disabled', stepId: 'count'});
    expect(suggestion.label).toBe('Enable Fixture scale');
    const result = apply(suggestion, document, registry);
    expect(result.steps[1].disabled).toBe(false);
    expect(result.steps[0].disabled).toBe(true);
  });

  test('unknown operations can be removed without introducing downstream errors', () => {
    const document = makeFixtureDocument();
    document.steps[2].operationId = 'missing';
    const [suggestion] = suggest(document, registry, {code: 'unknown-operation', stepId: 'count'});
    expect(suggestion.label).toBe('Remove this step');
    expect(apply(suggestion, document, registry).steps.map(step => step.id)).toEqual([
      'source',
      'scale'
    ]);
    const connected = makeFixtureDocument();
    connected.steps[1].operationId = 'missing';
    expect(suggest(connected, registry, {code: 'unknown-operation', stepId: 'scale'})).toEqual([]);
  });

  test('GPU fallback changes stop to warn', () => {
    const document = makeFixtureDocument();
    document.steps[1].onFallback = 'stop';
    const [suggestion] = suggest(document, registry, {code: 'gpu-fallback', stepId: 'scale'});
    expect(suggestion.label).toBe('Continue on the CPU engine when the GPU cannot …');
    expect(apply(suggestion, document, registry).steps[1].onFallback).toBe('warn');
    expect(
      suggest(makeFixtureDocument(), registry, {code: 'gpu-fallback', stepId: 'scale'})
    ).toEqual([]);
  });

  test('labels truncate the embedded step name and IDs remain deterministic', () => {
    const document = makeFixtureDocument();
    document.steps[1].label = 'A'.repeat(80);
    document.steps[1].disabled = true;
    const details = {code: 'upstream-disabled' as const, stepId: 'count'};
    const first = suggest(document, registry, details);
    const second = suggest(document, registry, details);
    expect(first[0].label).toBe(`Enable ${'A'.repeat(40)}…`);
    expect(first.map(suggestion => suggestion.id)).toEqual(second.map(suggestion => suggestion.id));
    expect(first[0].id).toBe('enable:count:scale');
  });

  test('unsafe or unsupported repairs are omitted, including an unconfigured real source', () => {
    const document = makeFixtureDocument();
    document.steps = [document.steps[1]];
    document.steps[0].inputs = {};
    expect(
      suggest(document, registry, {code: 'input-required', stepId: 'scale', port: 'points'})
    ).toEqual([]);
    expect(suggest(document, registry, {code: 'input-required', port: 'points'})).toEqual([]);
    expect(
      suggest(document, registry, {code: 'input-required', stepId: 'missing', port: 'points'})
    ).toEqual([]);
    expect(suggest(document, registry, {code: 'operation-failed', stepId: 'scale'})).toEqual([]);
    const full = makeFixtureDocument();
    while (full.steps.length < WORKFLOW_LIMITS.maximumSteps)
      full.steps.push({
        ...full.steps[0],
        id: `extra-${full.steps.length}`
      });
    expect(
      suggest(full, registry, {code: 'capacity-exceeded', stepId: 'scale', port: 'points'})
    ).toEqual([]);
  });

  test('application rechecks the current registry and reports the exact introduced error', () => {
    const document = makeFixtureDocument();
    const [suggestion] = suggest(document, registry, {
      code: 'capacity-exceeded',
      stepId: 'scale',
      port: 'points'
    });
    const invalidSort = modifyOperation('table-sort-rows', {
      inputs: [
        {name: 'rows', label: 'Rows', accepts: ['features'], maximumRows: 100},
        {name: 'extra', label: 'Extra', accepts: ['table'], maximumRows: 100}
      ]
    });
    try {
      suggestion.apply(document, makeRegistry([invalidSort]));
      throw new Error('Expected repair refusal');
    } catch (error) {
      expect(error).toBeInstanceOf(WorkflowError);
      expect((error as WorkflowError).code).toBe('input-required');
      expect((error as WorkflowError).details.port).toBe('extra');
    }
    expect(document.steps).toHaveLength(3);
  });
});
