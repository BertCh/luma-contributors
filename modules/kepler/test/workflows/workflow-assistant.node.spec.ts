// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {
  describeWorkflowOperationForAssistant,
  describeWorkflowForAssistant,
  getWorkflowAssistantTools,
  listWorkflowsForAssistant,
  listWorkflowOperationsForAssistant,
  parseWorkflowDraft,
  parseWorkflowInputChanges,
  parseWorkflowRunRequest
} from '../../src/workflows/workflow-assistant';
import {
  parseWorkflowDocument,
  serializeWorkflowDocument
} from '../../src/workflows/workflow-document';
import * as workflowExport from '../../src/workflows/workflow-export';
import {planWorkflow} from '../../src/workflows/workflow-plan';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
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

function makeRegistry(operations: readonly WorkflowOperation[]): WorkflowOperationRegistry {
  return {
    list: () => operations,
    get: operationId => operations.find(operation => operation.id === operationId)
  };
}

function makeDraftArguments() {
  return {
    title: 'Fixture workflow',
    description: 'Scale points and count rows.',
    steps: [
      {id: 'source', operationId: 'fixture-source', parameters: {count: 4}},
      {
        id: 'scale',
        operationId: 'fixture-scale',
        label: 'Scale values',
        inputs: {points: 'source.data'},
        parameters: {factor: 3},
        display: true
      },
      {id: 'count', operationId: 'fixture-count', inputs: {rows: 'scale.scaled'}}
    ]
  };
}

function expectMalformed(action: () => unknown, path: string): void {
  expect(action).toThrow(WorkflowError);
  expect(action).toThrow(
    expect.objectContaining({code: 'malformed', message: expect.stringContaining(path)})
  );
}

function makeSavedDocument() {
  const document = makeFixtureDocument();
  document.description = 'Saved analysis.';
  document.steps[1].label = 'Scale values';
  document.inputs = [
    {
      id: 'factor',
      label: 'Distance',
      kind: 'number',
      value: 3,
      unit: 'meters',
      minimum: 0,
      maximum: 100
    },
    {id: 'mode', label: 'Mode', kind: 'enum', value: 'first', options: ['first', 'last']},
    {id: 'bands', label: 'Bands', kind: 'number-list', value: [1, 2]},
    {id: 'enabled', label: 'Enabled', kind: 'boolean', value: false}
  ];
  return document;
}

describe('saved workflows as assistant tools', () => {
  test('publishes saved-workflow schemas and explicit approval requirements', () => {
    const tools = getWorkflowAssistantTools().slice(3);
    expect(tools.map(tool => tool.parameters)).toEqual([
      {type: 'object', properties: {}, required: [], additionalProperties: false},
      {
        type: 'object',
        properties: {workflowId: {type: 'string'}},
        required: ['workflowId'],
        additionalProperties: false
      },
      {
        type: 'object',
        properties: {
          workflowId: {type: 'string'},
          inputs: {
            type: 'object',
            additionalProperties: {
              anyOf: [
                {type: 'string', maxLength: WORKFLOW_LIMITS.maximumTextCharacters},
                {type: 'number'},
                {type: 'boolean'},
                {type: 'array', items: {type: 'number'}}
              ]
            }
          }
        },
        required: ['workflowId', 'inputs'],
        additionalProperties: false
      },
      {
        type: 'object',
        properties: {workflowId: {type: 'string'}, target: {type: 'string'}},
        required: ['workflowId'],
        additionalProperties: false
      }
    ]);
    expect(tools[2].description).toContain('only after the user approves');
    expect(tools[3].description).toContain('only after the user approves that exact call');
    expect(tools[3].description).toContain('per-step summary, not the data');
  });

  test('lists saved workflow metadata and describes its full input contract without state', () => {
    const document = makeSavedDocument();
    const registry = makeFixtureRegistry();
    expect(listWorkflowsForAssistant([], registry)).toEqual({workflows: []});
    expect(listWorkflowsForAssistant([document], registry)).toEqual({
      workflows: [
        {
          id: document.id,
          title: 'Fixture workflow',
          description: 'Saved analysis.',
          stepCount: 3,
          inputs: document.inputs
        }
      ]
    });
    const expected = {
      id: document.id,
      title: document.title,
      description: document.description,
      outline: workflowExport.describeWorkflow(document, registry),
      inputs: document.inputs
    };
    expect(describeWorkflowForAssistant([document], registry, {workflowId: document.id})).toEqual(
      expected
    );
    expect(
      describeWorkflowForAssistant([document], registry, {workflowId: document.id}, null)
    ).toEqual(expected);
    expect(JSON.parse(JSON.stringify(expected))).toEqual(expected);
  });

  test('describes each step with status, output row counts and its first issue without data', () => {
    const document = makeSavedDocument();
    const state: WorkflowRunnerState = {
      revision: 1,
      running: false,
      order: ['source', 'scale', 'count'],
      issues: [],
      steps: {
        source: {
          status: 'done',
          key: 'source-key',
          issues: [],
          result: {
            key: 'source-key',
            outputs: {data: makeFixturePoints(4)},
            diagnostics: [],
            summary: {},
            engine: 'host',
            dispatchedContributors: [],
            parameters: {count: 4},
            durationMilliseconds: 0,
            cacheHit: false,
            completedAt: 0
          }
        },
        scale: {
          status: 'error',
          key: 'scale-key',
          issues: [
            {
              code: 'capacity-may-exceed',
              severity: 'info',
              message: 'Scale values: input may exceed 10 rows.'
            }
          ],
          error: {
            code: 'operation-failed',
            severity: 'error',
            message: 'Scale values: operation failed.'
          }
        }
      }
    };
    const result = describeWorkflowForAssistant(
      [document],
      makeFixtureRegistry(),
      {workflowId: document.id},
      state
    ) as {steps: unknown[]};
    expect(result.steps).toEqual([
      {
        id: 'source',
        label: 'Fixture source',
        operationId: 'fixture-source',
        status: 'done',
        rows: {data: 4},
        firstIssue: null
      },
      {
        id: 'scale',
        label: 'Scale values',
        operationId: 'fixture-scale',
        status: 'error',
        rows: {},
        firstIssue: 'Scale values: operation failed.'
      },
      {
        id: 'count',
        label: 'Fixture count',
        operationId: 'fixture-count',
        status: null,
        rows: {},
        firstIssue: null
      }
    ]);
    expect(JSON.stringify(result)).not.toContain('point_id');
    state.steps['scale'].error = undefined;
    expect(
      (
        describeWorkflowForAssistant(
          [document],
          makeFixtureRegistry(),
          {workflowId: document.id},
          state
        ) as {steps: {firstIssue: unknown}[]}
      ).steps[1].firstIssue
    ).toBe('Scale values: input may exceed 10 rows.');
  });

  test('proposes only changed input values, preferring ids to labels and leaving the saved document untouched', () => {
    const document = makeSavedDocument();
    document.inputs.push({id: 'alias', label: 'factor', kind: 'number', value: 10});
    const original = JSON.stringify(document);
    const result = parseWorkflowInputChanges([document], {
      workflowId: document.id,
      inputs: {factor: 5, MODE: 'last', Bands: [1, 2], eNaBlEd: true}
    });
    expect(result.workflowId).toBe(document.id);
    expect(result.changes).toEqual([
      {inputId: 'factor', label: 'Distance', previous: 3, next: 5},
      {inputId: 'mode', label: 'Mode', previous: 'first', next: 'last'},
      {inputId: 'enabled', label: 'Enabled', previous: false, next: true}
    ]);
    expect(result.document.inputs.map(input => input.value)).toEqual([5, 'last', [1, 2], true, 10]);
    expect(JSON.stringify(document)).toBe(original);
    expect(parseWorkflowDocument(serializeWorkflowDocument(result.document))).toEqual(
      result.document
    );
    const unchanged = parseWorkflowInputChanges([document], {
      workflowId: document.id,
      inputs: {Distance: 3, Bands: [1, 2]}
    });
    expect(unchanged.changes).toEqual([]);
    expect(unchanged.document).toBe(document);
    const bands = Array.from({length: 65}, (_, index) => index);
    expect(
      parseWorkflowInputChanges([document], {workflowId: document.id, inputs: {bands}}).changes
    ).toEqual([{inputId: 'bands', label: 'Bands', previous: [1, 2], next: bands}]);
  });

  test('rejects ambiguous labels, conflicting aliases, unknown inputs and invalid input values', () => {
    const document = makeSavedDocument();
    document.inputs.push({id: 'other-distance', label: 'distance', kind: 'number', value: 1});
    expect(() =>
      parseWorkflowInputChanges([document], {workflowId: document.id, inputs: {DISTANCE: 5}})
    ).toThrow(
      expect.objectContaining({
        code: 'malformed',
        message:
          'Fixture workflow: Input label "DISTANCE" is ambiguous between "factor", "other-distance"; use an input id.'
      })
    );
    expect(
      parseWorkflowInputChanges([document], {workflowId: document.id, inputs: {factor: 5}}).changes
    ).toEqual([{inputId: 'factor', label: 'Distance', previous: 3, next: 5}]);
    document.inputs.pop();
    for (const [inputs, code, label] of [
      [{factor: 101}, 'parameter-out-of-range', 'Distance'],
      [{factor: NaN}, 'parameter-out-of-range', 'Distance'],
      [{mode: 'middle'}, 'parameter-out-of-range', 'Mode'],
      [{enabled: 'true'}, 'parameter-out-of-range', 'Enabled'],
      [{bands: []}, 'parameter-out-of-range', 'Bands'],
      [{missing: 1}, 'unknown-workflow-input', 'missing'],
      [{factor: 5, Distance: 6}, 'malformed', 'Distance']
    ] as const) {
      expect(() =>
        parseWorkflowInputChanges([document], {workflowId: document.id, inputs})
      ).toThrow(expect.objectContaining({code, message: expect.stringContaining(label)}));
    }
    expect(document.inputs.map(input => input.value)).toEqual([3, 'first', [1, 2], false]);
  });

  test('unknown workflow ids are malformed and list the available ids for every saved-workflow parser', () => {
    const document = makeSavedDocument();
    const registry = makeFixtureRegistry();
    for (const action of [
      () => describeWorkflowForAssistant([document], registry, {workflowId: 'missing'}),
      () => parseWorkflowInputChanges([document], {workflowId: 'missing', inputs: {}}),
      () => parseWorkflowRunRequest([document], {workflowId: 'missing'})
    ]) {
      expect(action).toThrow(
        expect.objectContaining({
          code: 'malformed',
          message: 'Workflow "missing" is not available. Available workflow ids: fixture-workflow.'
        })
      );
    }
    expectMalformed(
      () =>
        describeWorkflowForAssistant([document], registry, {workflowId: document.id, extra: true}),
      'arguments.extra'
    );
    expectMalformed(
      () => parseWorkflowInputChanges([document], {workflowId: document.id, inputs: null}),
      'arguments.inputs'
    );
    expectMalformed(() => parseWorkflowRunRequest([document], null), 'arguments');
  });

  test('validates a run target as a step id without running anything', () => {
    const document = makeSavedDocument();
    const original = JSON.stringify(document);
    expect(parseWorkflowRunRequest([document], {workflowId: document.id})).toEqual({
      workflowId: document.id
    });
    expect(parseWorkflowRunRequest([document], {workflowId: document.id, target: 'scale'})).toEqual(
      {workflowId: document.id, target: 'scale'}
    );
    expect(() =>
      parseWorkflowRunRequest([document], {workflowId: document.id, target: 'Scale values'})
    ).toThrow(expect.objectContaining({code: 'unknown-step', details: {stepId: 'Scale values'}}));
    expectMalformed(
      () => parseWorkflowRunRequest([document], {workflowId: document.id, target: 2}),
      'arguments.target'
    );
    expectMalformed(
      () => parseWorkflowRunRequest([document], {workflowId: document.id, force: true}),
      'arguments.force'
    );
    expect(JSON.stringify(document)).toBe(original);
  });
});

describe('workflow assistant tool contracts', () => {
  test('publishes exactly the seven tools and JSON object schemas', () => {
    const tools = getWorkflowAssistantTools();
    expect(tools.map(tool => tool.name)).toEqual([
      'listAnalysisOperations',
      'describeAnalysisOperation',
      'defineWorkflow',
      'listWorkflows',
      'describeWorkflow',
      'setWorkflowInputs',
      'runWorkflow'
    ]);
    for (const tool of tools) {
      expect(JSON.parse(JSON.stringify(tool.parameters))).toEqual(tool.parameters);
      expect(tool.parameters).toMatchObject({type: 'object', additionalProperties: false});
      expect(tool.description.length).toBeGreaterThan(40);
    }
    expect(tools[0].parameters).toEqual({
      type: 'object',
      properties: {
        query: {type: 'string'},
        category: {
          type: 'string',
          enum: [
            'source',
            'table',
            'geometry',
            'network',
            'aggregation',
            'interpolation',
            'density',
            'clustering',
            'raster',
            'analysis'
          ]
        },
        compatibleWith: {type: 'string', enum: ['point', 'line', 'polygon', 'features', 'table']},
        limit: {type: 'integer', minimum: 1, maximum: 50, default: 25}
      },
      required: [],
      additionalProperties: false
    });
    expect(tools[1].parameters).toEqual({
      type: 'object',
      properties: {operationId: {type: 'string'}},
      required: ['operationId'],
      additionalProperties: false
    });
    expect(tools[2].parameters).toMatchObject({
      required: ['title', 'steps'],
      properties: {
        title: {type: 'string', maxLength: 80},
        description: {type: 'string', maxLength: 500},
        steps: {
          type: 'array',
          minItems: 1,
          maxItems: 64,
          items: {
            type: 'object',
            required: ['id', 'operationId'],
            additionalProperties: false,
            properties: {
              id: {type: 'string'},
              operationId: {type: 'string'},
              label: {type: 'string'},
              inputs: {type: 'object', additionalProperties: {type: 'string'}},
              parameters: {type: 'object', additionalProperties: {anyOf: expect.any(Array)}},
              display: {anyOf: [{type: 'boolean'}, {type: 'array', items: {type: 'string'}}]}
            }
          }
        }
      }
    });
    expect(tools[2].description).toContain('only proposes');
    expect(tools[2].description).toContain('approval');
  });

  test('definitions have independent JSON objects on every call', () => {
    const first = getWorkflowAssistantTools();
    const second = getWorkflowAssistantTools();
    expect(first).toEqual(second);
    expect(first[0].parameters).not.toBe(second[0].parameters);
  });
});

describe('assistant operation discovery', () => {
  test('lists compact contracts with filters and a total before applying the limit', () => {
    const registry = makeFixtureRegistry();
    expect(listWorkflowOperationsForAssistant(registry, {query: 'fixture', limit: 1})).toEqual({
      total: 3,
      operations: [
        {
          id: 'fixture-count',
          title: 'Fixture count',
          category: 'table',
          engine: 'cpu-reference',
          summary: 'Counts rows.',
          inputs: [{name: 'rows', accepts: ['point', 'line', 'polygon', 'table']}],
          outputs: [{name: 'count', kind: 'table'}]
        }
      ]
    });
    expect(
      listWorkflowOperationsForAssistant(registry, {
        query: 'VALUE',
        category: 'geometry',
        compatibleWith: 'point'
      })
    ).toEqual({
      total: 1,
      operations: [
        {
          id: 'fixture-scale',
          title: 'Fixture scale',
          category: 'geometry',
          engine: 'gpu-or-cpu',
          summary: 'Multiplies the value column.',
          inputs: [{name: 'points', accepts: ['point']}],
          outputs: [{name: 'scaled', kind: 'point'}]
        }
      ]
    });
    expect(listWorkflowOperationsForAssistant(registry, {compatibleWith: 'polygon'})).toMatchObject(
      {total: 1, operations: [{id: 'fixture-count'}]}
    );
    expect(listWorkflowOperationsForAssistant(registry, {category: 'network'})).toEqual({
      total: 0,
      operations: []
    });
  });

  test('defaults to 25, accepts 50, and truncates summaries at 160 characters', () => {
    const source = makeFixtureRegistry().get('fixture-source')!;
    const registry = makeRegistry(
      Array.from({length: 60}, (_, index) => ({
        ...source,
        id: `source-${index}`,
        summary: 'x'.repeat(200)
      }))
    );
    expect(listWorkflowOperationsForAssistant(registry, {})).toMatchObject({
      total: 60,
      operations: expect.any(Array)
    });
    const defaults = listWorkflowOperationsForAssistant(registry, {}) as {
      operations: {summary: string}[];
    };
    expect(defaults.operations).toHaveLength(25);
    expect(defaults.operations[0].summary).toBe('x'.repeat(160));
    expect(listWorkflowOperationsForAssistant(registry, {limit: 50})).toMatchObject({
      operations: expect.arrayContaining([expect.objectContaining({id: 'source-49'})])
    });
  });

  test.each([
    [null, 'arguments'],
    [[], 'arguments'],
    [{extra: true}, 'arguments.extra'],
    [{query: 2}, 'arguments.query'],
    [{category: 'invalid'}, 'arguments.category'],
    [{compatibleWith: 'invalid'}, 'arguments.compatibleWith'],
    [{limit: 0}, 'arguments.limit'],
    [{limit: 51}, 'arguments.limit'],
    [{limit: 1.5}, 'arguments.limit'],
    [{limit: '2'}, 'arguments.limit'],
    [{limit: NaN}, 'arguments.limit'],
    [{category: undefined}, 'arguments.category']
  ])('refuses malformed discovery arguments %j', (args, path) => {
    expectMalformed(() => listWorkflowOperationsForAssistant(makeFixtureRegistry(), args), path);
  });

  test('describes the full fixture scale contract exactly', () => {
    expect(
      describeWorkflowOperationForAssistant(makeFixtureRegistry(), {operationId: 'fixture-scale'})
    ).toEqual({
      id: 'fixture-scale',
      title: 'Fixture scale',
      category: 'geometry',
      summary: 'Multiplies the value column.',
      origin: 'builtin',
      version: '1',
      engine: 'gpu-or-cpu',
      inputs: [
        {name: 'points', label: 'Points', accepts: ['point'], maximumRows: 10, maximumVertices: 10}
      ],
      outputs: [
        {
          name: 'scaled',
          label: 'Scaled',
          kind: 'point',
          identityColumns: ['point_id'],
          maximumRows: 10
        }
      ],
      parameters: [
        {
          name: 'factor',
          label: 'Factor',
          kind: 'number',
          minimum: 0,
          maximum: 100,
          defaultValue: 2
        },
        {name: 'fail', label: 'Fail', kind: 'boolean', defaultValue: false}
      ],
      capabilityIds: [],
      contributorNames: [],
      declaredBehavior: ['Fixture operation.'],
      limitation: 'Fixture only.'
    });
  });

  test('preserves units, options and text limits while omitting defaults longer than 80', () => {
    const source = makeFixtureRegistry().get('fixture-source')!;
    const operation: WorkflowOperation = {
      ...source,
      parameters: [
        {
          name: 'distance',
          label: 'Distance',
          kind: 'number',
          unit: 'meters',
          minimum: 0,
          maximum: 100,
          defaultValue: 5
        },
        {
          name: 'mode',
          label: 'Mode',
          kind: 'enum',
          options: ['first', 'last'],
          defaultValue: 'first'
        },
        {
          name: 'short',
          label: 'Short',
          kind: 'text',
          maximumLength: 100,
          defaultValue: 'x'.repeat(80)
        },
        {
          name: 'long',
          label: 'Long',
          kind: 'text',
          maximumLength: 100,
          defaultValue: 'x'.repeat(81)
        }
      ]
    };
    const result = describeWorkflowOperationForAssistant(makeRegistry([operation]), {
      operationId: operation.id
    });
    expect(result).toMatchObject({
      parameters: [
        {unit: 'meters', minimum: 0, maximum: 100, defaultValue: 5},
        {options: ['first', 'last'], defaultValue: 'first'},
        {maximumLength: 100, defaultValue: 'x'.repeat(80)},
        {maximumLength: 100}
      ]
    });
    expect((result as {parameters: unknown[]}).parameters[3]).not.toHaveProperty('defaultValue');
    expect(JSON.parse(JSON.stringify(result))).toEqual(result);
  });

  test('unknown ids suggest up to five search results', () => {
    const registry = makeFixtureRegistry();
    expect(() => describeWorkflowOperationForAssistant(registry, {operationId: 'fixture'})).toThrow(
      expect.objectContaining({
        code: 'unknown-operation',
        message:
          'Operation "fixture" is not available. Try fixture-count, fixture-scale, fixture-source.'
      })
    );
    const source = registry.get('fixture-source')!;
    const many = makeRegistry(
      Array.from({length: 8}, (_, index) => ({...source, id: `suggest-${index}`}))
    );
    expect(() => describeWorkflowOperationForAssistant(many, {operationId: 'suggest'})).toThrow(
      'Try suggest-0, suggest-1, suggest-2, suggest-3, suggest-4.'
    );
    expect(() =>
      describeWorkflowOperationForAssistant(registry, {operationId: 'does-not-exist'})
    ).toThrow(
      expect.objectContaining({
        code: 'unknown-operation',
        message: 'Operation "does-not-exist" is not available.'
      })
    );
  });

  test.each([
    [{}, 'arguments.operationId'],
    [{operationId: 1}, 'arguments.operationId'],
    [{operationId: 'fixture-source', extra: false}, 'arguments.extra']
  ])('validates describe arguments %j', (args, path) => {
    expectMalformed(() => describeWorkflowOperationForAssistant(makeFixtureRegistry(), args), path);
  });
});

describe('assistant workflow drafts', () => {
  test('F1: competing dotted-id interpretations stay unresolved and cannot execute', async () => {
    const registry = makeFixtureRegistry();
    const draft = parseWorkflowDraft(
      {
        title: 'Ambiguous source',
        steps: [
          {id: 'source', operationId: 'fixture-source', parameters: {count: 4}},
          {id: 'source.data', operationId: 'fixture-source', parameters: {count: 9}},
          {id: 'count', operationId: 'fixture-count', inputs: {rows: 'source.data'}}
        ]
      },
      {registry}
    );
    expect(draft.issues).toEqual([
      expect.objectContaining({
        code: 'unknown-output-port',
        severity: 'error',
        stepId: 'count',
        port: 'rows',
        message: expect.stringContaining('ambiguous')
      })
    ]);
    expect(draft.issues[0].message).toContain('source-data.data');
    expect(draft.document.steps[2].inputs).toEqual({
      rows: {stepId: 'source', output: 'unresolved-output'}
    });
    expect(parseWorkflowDocument(serializeWorkflowDocument(draft.document))).toEqual(
      draft.document
    );
    const runner = new WorkflowRunner({registry, host: makeFixtureHost(), now: () => 0});
    try {
      runner.setDocument(draft.document);
      const record = await runner.run();
      expect(record.steps.map(step => step.stepId)).toEqual(['source', 'source-data']);
      expect(runner.getState().steps['count'].status).toBe('not-configured');
      expect(runner.getState().steps['count'].result).toBeUndefined();
      expect((await runner.run({target: 'count'})).status).toBe('error');
    } finally {
      await runner.destroy();
    }
  });

  test('F1: competing qualified interpretations also report ambiguity', () => {
    const fixture = makeFixtureRegistry();
    const source = fixture.get('fixture-source')!;
    const dotted = {
      ...source,
      id: 'dotted-output',
      outputs: [{...source.outputs[0], name: 'data.points'}]
    };
    const draft = parseWorkflowDraft(
      {
        title: 'Ambiguous prefixes',
        steps: [
          {id: 'source', operationId: dotted.id},
          {id: 'source.data', operationId: source.id},
          {id: 'count', operationId: 'fixture-count', inputs: {rows: 'source.data.points'}}
        ]
      },
      {
        registry: makeRegistry([
          dotted,
          {...source, outputs: [{...source.outputs[0], name: 'points'}]},
          fixture.get('fixture-count')!
        ])
      }
    );
    expect(draft.issues.map(issue => issue.code)).toEqual(['unknown-output-port']);
    expect(draft.issues[0].message).toContain('ambiguous');
  });

  test('F2: the planner accepts self-cycles but the strict document contract currently rejects them', () => {
    const registry = makeFixtureRegistry();
    const document = makeFixtureDocument();
    document.steps = [
      {
        id: 'count',
        operationId: 'fixture-count',
        inputs: {rows: {stepId: 'count', output: 'count'}},
        parameters: {},
        position: {x: 0, y: 0}
      }
    ];
    expect(planWorkflow(document, {registry}).issues.map(issue => issue.code)).toEqual(['cycle']);
    expect(
      planWorkflow(document, {registry}).steps['count'].issues.map(issue => issue.code)
    ).toEqual(['cycle']);
    expect(() => serializeWorkflowDocument(document)).toThrow(
      expect.objectContaining({
        code: 'malformed',
        message: expect.stringContaining('a step must not read itself')
      })
    );
    expect(() =>
      parseWorkflowDraft(
        {
          title: 'Self cycle',
          steps: [{id: 'count', operationId: 'fixture-count', inputs: {rows: 'count'}}]
        },
        {registry}
      )
    ).toThrow(
      expect.objectContaining({
        code: 'malformed',
        message: expect.stringContaining('document.steps[0].inputs.rows.stepId')
      })
    );
  });

  test('F3: display forms preserve exact output identifiers with surrounding whitespace', () => {
    const source = makeFixtureRegistry().get('fixture-source')!;
    const operation = {...source, outputs: [{...source.outputs[0], name: ' data '}]};
    const registry = makeRegistry([operation]);
    for (const display of [true, [' data '], [' data ', ' data ']]) {
      const draft = parseWorkflowDraft(
        {title: 'Exact output', steps: [{id: 'source', operationId: source.id, display}]},
        {registry}
      );
      expect(draft.document.steps[0].display).toEqual([' data ']);
      expect(draft.issues).toEqual([]);
      expect(parseWorkflowDocument(serializeWorkflowDocument(draft.document))).toEqual(
        draft.document
      );
    }
    expectMalformed(
      () =>
        parseWorkflowDraft(
          {
            title: 'Blank output',
            steps: [{id: 'source', operationId: source.id, display: ['   ']}]
          },
          {registry}
        ),
      'arguments.steps[0].display[0]'
    );
  });

  test('F4: ambiguity issue replacement performs at most linear predicate work', () => {
    const fixture = makeFixtureRegistry();
    const source = fixture.get('fixture-source')!;
    const registry = makeRegistry([
      {...source, outputs: [...source.outputs, {...source.outputs[0], name: 'second'}]},
      fixture.get('fixture-count')!
    ]);
    const connectionCount = 512;
    const originalSome = Array.prototype.some;
    let replacementComparisons = 0;
    const some = vi.spyOn(Array.prototype, 'some').mockImplementation(function (
      this: unknown[],
      predicate: (value: unknown, index: number, array: unknown[]) => unknown,
      thisArgument?: unknown
    ): boolean {
      const first = this[0] as {code?: unknown; stepId?: unknown} | undefined;
      const replacements =
        this.length === connectionCount &&
        first?.code === 'unknown-output-port' &&
        first.stepId === 'count';
      return originalSome.call(this, (value: unknown, index: number, array: unknown[]) => {
        if (replacements) replacementComparisons++;
        return predicate.call(thisArgument, value, index, array);
      });
    });
    try {
      const draft = parseWorkflowDraft(
        {
          title: 'Many connections',
          steps: [
            {id: 'source', operationId: source.id},
            {
              id: 'count',
              operationId: 'fixture-count',
              inputs: Object.fromEntries(
                Array.from({length: connectionCount}, (_, index) => [`port-${index}`, 'source'])
              )
            }
          ]
        },
        {registry}
      );
      expect(draft.issues.filter(issue => issue.code === 'unknown-output-port')).toHaveLength(
        connectionCount
      );
      expect(replacementComparisons).toBeLessThanOrEqual(connectionCount * 4);
    } finally {
      some.mockRestore();
    }
  });

  test('F4: a 12,000-connection draft keeps every unknown-port and ambiguity issue', () => {
    const fixture = makeFixtureRegistry();
    const source = fixture.get('fixture-source')!;
    const registry = makeRegistry([
      {...source, outputs: [...source.outputs, {...source.outputs[0], name: 'second'}]},
      fixture.get('fixture-count')!
    ]);
    const draft = parseWorkflowDraft(
      {
        title: 'Large draft',
        steps: [
          {id: 'source', operationId: source.id},
          {
            id: 'count',
            operationId: 'fixture-count',
            inputs: Object.fromEntries(
              Array.from({length: 12000}, (_, index) => [`port-${index}`, 'source'])
            )
          }
        ]
      },
      {registry}
    );
    expect(JSON.stringify(serializeWorkflowDocument(draft.document)).length).toBeLessThan(
      WORKFLOW_LIMITS.maximumDocumentCharacters
    );
    expect(draft.issues).toHaveLength(24001);
    expect(draft.issues.filter(issue => issue.code === 'unknown-input-port')).toHaveLength(12000);
    expect(draft.issues.filter(issue => issue.code === 'unknown-output-port')).toHaveLength(12000);
    expect(draft.issues.filter(issue => issue.code === 'input-required')).toHaveLength(1);
    for (const port of ['port-0', 'port-11999']) {
      expect(draft.issues.filter(issue => issue.port === port).map(issue => issue.code)).toEqual([
        'unknown-input-port',
        'unknown-output-port'
      ]);
    }
  });

  test('F5: only the exact exporter dependency sentinel uses the deterministic fallback outline', () => {
    const outline = vi.spyOn(workflowExport, 'describeWorkflow');
    try {
      outline.mockImplementation(() => {
        throw new Error('not implemented');
      });
      const calls: string[] = [];
      const draft = parseWorkflowDraft(makeDraftArguments(), {
        registry: makeFixtureRegistry(calls)
      });
      expect(draft.outline).toBe(
        'Fixture workflow\n1. Fixture source (source)\n2. Scale values (scale): points ← source.data; factor = 3\n3. Fixture count (count): rows ← scale.scaled'
      );
      expect(calls).toEqual([]);
      for (const error of [
        new Error('not implemented yet'),
        new Error('Not implemented'),
        new Error('not  implemented'),
        new Error(' not implemented'),
        new Error('not implemented '),
        new Error('not implemented\n'),
        new Error('not\timplemented'),
        'not implemented'
      ]) {
        outline.mockImplementation(() => {
          throw error;
        });
        let receivedError: unknown;
        try {
          parseWorkflowDraft(makeDraftArguments(), {registry: makeFixtureRegistry()});
        } catch (caughtError) {
          receivedError = caughtError;
        }
        expect(receivedError).toBe(error);
      }
    } finally {
      outline.mockRestore();
    }
  });

  test('proposes and lays out a three-step workflow without running it, and round trips strictly', () => {
    const calls: string[] = [];
    const args = makeDraftArguments();
    const original = JSON.stringify(args);
    const draft = parseWorkflowDraft(args, {
      registry: makeFixtureRegistry(calls),
      takenIds: ['fixture-workflow']
    });
    expect(draft.document).toEqual({
      schemaVersion: 1,
      id: 'fixture-workflow-2',
      title: 'Fixture workflow',
      description: 'Scale points and count rows.',
      inputs: [],
      settings: {engine: 'auto', autoRun: false},
      steps: [
        {
          id: 'source',
          operationId: 'fixture-source',
          inputs: {},
          parameters: {},
          position: {x: 0, y: 0}
        },
        {
          id: 'scale',
          operationId: 'fixture-scale',
          label: 'Scale values',
          inputs: {points: {stepId: 'source', output: 'data'}},
          parameters: {factor: {from: 'static', value: 3}},
          display: ['scaled'],
          position: {x: 280, y: 0}
        },
        {
          id: 'count',
          operationId: 'fixture-count',
          inputs: {rows: {stepId: 'scale', output: 'scaled'}},
          parameters: {},
          position: {x: 560, y: 0}
        }
      ]
    });
    expect(draft.issues.filter(issue => issue.severity === 'error')).toEqual([]);
    expect(draft.outline).toContain('Scale values');
    expect(draft.outline.toLowerCase()).toContain('factor');
    expect(parseWorkflowDocument(serializeWorkflowDocument(draft.document))).toEqual(
      draft.document
    );
    expect(JSON.stringify(args)).toBe(original);
    expect(calls).toEqual([]);
  });

  test('remaps shorthand, forward references and ids that collide after normalisation', () => {
    const draft = parseWorkflowDraft(
      {
        title: 'Title',
        steps: [
          {id: 'Consumer', operationId: 'fixture-count', inputs: {rows: 'Step.One'}},
          {id: 'Step.One', operationId: 'fixture-source'},
          {id: 'Step One', operationId: 'fixture-scale', inputs: {points: 'Step.One.data'}},
          {id: 'last', operationId: 'fixture-count', inputs: {rows: 'Step One.scaled'}}
        ]
      },
      {registry: makeFixtureRegistry()}
    );
    expect(draft.document.steps.map(step => step.id)).toEqual([
      'consumer',
      'step-one',
      'step-one-2',
      'last'
    ]);
    expect(draft.document.steps.map(step => step.inputs)).toEqual([
      {rows: {stepId: 'step-one', output: 'data'}},
      {},
      {points: {stepId: 'step-one', output: 'data'}},
      {rows: {stepId: 'step-one-2', output: 'scaled'}}
    ]);
    expect(draft.issues.filter(issue => issue.severity === 'error')).toEqual([]);
  });

  test('ambiguous shorthand is a useful issue and the document is still serialisable', () => {
    const registry = makeFixtureRegistry();
    const source = registry.get('fixture-source')!;
    const multiple = {
      ...source,
      outputs: [...source.outputs, {...source.outputs[0], name: 'other'}]
    };
    const draft = parseWorkflowDraft(
      {
        title: 'Title',
        steps: [
          {id: 'source', operationId: source.id},
          {id: 'count', operationId: 'fixture-count', label: 'Count rows', inputs: {rows: 'source'}}
        ]
      },
      {registry: makeRegistry([multiple, registry.get('fixture-count')!])}
    );
    expect(draft.issues.filter(issue => issue.code === 'unknown-output-port')).toEqual([
      {
        code: 'unknown-output-port',
        severity: 'error',
        stepId: 'count',
        port: 'rows',
        message:
          'Count rows: Rows reference "source" must name one output as "stepId.output"; the producer declares 2 outputs.'
      }
    ]);
    expect(parseWorkflowDocument(serializeWorkflowDocument(draft.document))).toEqual(
      draft.document
    );
  });

  test('preserves dotted output names when remapping original ids', () => {
    const registry = makeFixtureRegistry();
    const source = registry.get('fixture-source')!;
    const dotted = {...source, outputs: [{...source.outputs[0], name: 'points.data'}]};
    const draft = parseWorkflowDraft(
      {
        title: 'Dotted ports',
        steps: [
          {id: 'Step.One', operationId: source.id},
          {id: 'count', operationId: 'fixture-count', inputs: {rows: 'Step.One.points.data'}}
        ]
      },
      {registry: makeRegistry([dotted, registry.get('fixture-count')!])}
    );
    expect(draft.document.steps[1].inputs).toEqual({
      rows: {stepId: 'step-one', output: 'points.data'}
    });
    expect(draft.issues.filter(issue => issue.severity === 'error')).toEqual([]);
  });

  test('preserves unknown references and cycles as plan issues', () => {
    const registry = makeFixtureRegistry();
    const dangling = parseWorkflowDraft(
      {
        title: 'Missing upstream',
        steps: [{id: 'count', operationId: 'fixture-count', inputs: {rows: 'missing.data'}}]
      },
      {registry}
    );
    expect(dangling.document.steps[0].inputs).toEqual({rows: {stepId: 'missing', output: 'data'}});
    expect(dangling.issues.map(issue => issue.code)).toEqual(['unknown-step']);
    const cycle = parseWorkflowDraft(
      {
        title: 'Cycle',
        steps: [
          {id: 'first', operationId: 'fixture-count', inputs: {rows: 'second'}},
          {id: 'second', operationId: 'fixture-count', inputs: {rows: 'first'}}
        ]
      },
      {registry}
    );
    expect(cycle.issues.map(issue => issue.code)).toEqual(['cycle', 'cycle', 'cycle']);
    expect(parseWorkflowDocument(serializeWorkflowDocument(cycle.document))).toEqual(
      cycle.document
    );
  });

  test('preserves unknown operations and parameters and reports exact configuration codes', () => {
    const draft = parseWorkflowDraft(
      {
        title: 'Invalid configuration',
        steps: [
          {id: 'unknown', operationId: 'no-such-operation', parameters: {kept: 'value'}},
          {id: 'source', operationId: 'fixture-source', parameters: {count: -1, mystery: 7}},
          {id: 'scale', operationId: 'fixture-scale', inputs: {wrong: 'source.wrong'}}
        ]
      },
      {registry: makeFixtureRegistry()}
    );
    expect(
      draft.issues.filter(issue => issue.stepId === 'unknown').map(issue => issue.code)
    ).toEqual(['unknown-operation']);
    expect(
      draft.issues.filter(issue => issue.stepId === 'source').map(issue => issue.code)
    ).toEqual(['unknown-parameter', 'parameter-out-of-range']);
    expect(draft.issues.filter(issue => issue.stepId === 'scale').map(issue => issue.code)).toEqual(
      ['unknown-input-port', 'unknown-output-port', 'input-required']
    );
    expect(draft.document.steps[0].parameters).toEqual({kept: {from: 'static', value: 'value'}});
    expect(draft.document.steps[1].parameters).toEqual({
      count: {from: 'static', value: -1},
      mystery: {from: 'static', value: 7}
    });
  });

  test('keeps an incompatible geometry connection for the planner to reject', () => {
    const registry = makeFixtureRegistry();
    const source = registry.get('fixture-source')!;
    const polygon: WorkflowOperation = {
      ...source,
      outputs: [{...source.outputs[0], kind: 'polygon'}]
    };
    const draft = parseWorkflowDraft(
      {
        title: 'Title',
        steps: [
          {id: 'source', operationId: source.id},
          {id: 'scale', operationId: 'fixture-scale', inputs: {points: 'source'}}
        ]
      },
      {registry: makeRegistry([polygon, registry.get('fixture-scale')!])}
    );
    expect(draft.issues).toEqual([
      expect.objectContaining({
        code: 'incompatible-port',
        stepId: 'scale',
        port: 'points',
        severity: 'error'
      }),
      expect.objectContaining({code: 'capacity-may-exceed'})
    ]);
  });

  test('drops scalar and list defaults and accepts both display forms', () => {
    const registry = makeFixtureRegistry();
    const source = registry.get('fixture-source')!;
    const operation: WorkflowOperation = {
      ...source,
      parameters: [
        ...source.parameters,
        {name: 'numbers', label: 'Numbers', kind: 'number-list', defaultValue: [1, 2]}
      ]
    };
    const draft = parseWorkflowDraft(
      {
        title: 'Title',
        steps: [
          {
            id: 'first',
            operationId: source.id,
            parameters: {count: 4, numbers: [1, 2]},
            display: true
          },
          {id: 'second', operationId: source.id, display: ['data', 'data']},
          {id: 'third', operationId: source.id, display: false}
        ]
      },
      {registry: makeRegistry([operation])}
    );
    expect(draft.document.steps.map(step => step.parameters)).toEqual([{}, {}, {}]);
    expect(draft.document.steps.map(step => step.display)).toEqual([['data'], ['data'], undefined]);
  });

  test('sorts all errors before warnings and informational issues', () => {
    const draft = parseWorkflowDraft(
      {
        title: 'Title',
        steps: [
          {id: 'source', operationId: 'fixture-source', display: ['missing']},
          {
            id: 'scale',
            operationId: 'fixture-scale',
            inputs: {points: 'source'},
            parameters: {factor: -1}
          }
        ]
      },
      {registry: makeFixtureRegistry()}
    );
    expect(draft.issues.map(issue => issue.severity)).toEqual(['error', 'warning', 'info', 'info']);
    expect(draft.issues.map(issue => issue.code)).toEqual([
      'parameter-out-of-range',
      'display-unknown-output',
      'unverified-port',
      'capacity-may-exceed'
    ]);
  });

  test.each([
    [null, 'arguments'],
    [{title: 'Title'}, 'arguments.steps'],
    [{title: 'Title', steps: []}, 'arguments.steps'],
    [{steps: [{id: 'source', operationId: 'fixture-source'}]}, 'arguments.title'],
    [{title: 'Title', steps: [{}]}, 'arguments.steps[0].id'],
    [{title: 'Title', steps: [{id: 'source'}]}, 'arguments.steps[0].operationId'],
    [{title: 'Title', steps: [{id: 1, operationId: 'fixture-source'}]}, 'arguments.steps[0].id'],
    [{title: 'Title', steps: [{id: 'source', operationId: ''}]}, 'arguments.steps[0].operationId'],
    [
      {
        title: 'Title',
        steps: [
          {id: 'same', operationId: 'fixture-source'},
          {id: 'same', operationId: 'fixture-source'}
        ]
      },
      'arguments.steps[1].id'
    ],
    [
      {
        title: 'Title',
        steps: Array.from({length: 65}, (_, index) => ({
          id: `step-${index}`,
          operationId: 'fixture-source'
        }))
      },
      'maximumSteps'
    ],
    [
      {
        title: 'Title',
        steps: [{id: 'source', operationId: 'fixture-source', parameters: {count: NaN}}]
      },
      'arguments.steps[0].parameters.count'
    ],
    [
      {
        title: 'Title',
        steps: [
          {
            id: 'source',
            operationId: 'fixture-source',
            parameters: {count: {from: 'static', value: 2}}
          }
        ]
      },
      'arguments.steps[0].parameters.count'
    ]
  ])('refuses structural problems %j with a path', (args, path) => {
    expectMalformed(() => parseWorkflowDraft(args, {registry: makeFixtureRegistry()}), path);
  });

  test('refuses excessive and control-character titles and trims surrounding whitespace', () => {
    const args = makeDraftArguments();
    for (const title of ['x'.repeat(100000), 'Bad\u0000title', 'Bad\ntitle', '   ']) {
      expectMalformed(
        () => parseWorkflowDraft({...args, title}, {registry: makeFixtureRegistry()}),
        'arguments.title'
      );
    }
    expect(
      parseWorkflowDraft({...args, title: '  Valid title  '}, {registry: makeFixtureRegistry()})
        .document.title
    ).toBe('Valid title');
    expect(
      parseWorkflowDraft({...args, title: 'x'.repeat(80)}, {registry: makeFixtureRegistry()})
        .document.id
    ).toHaveLength(48);
    expectMalformed(
      () =>
        parseWorkflowDraft(
          {...args, description: 'x'.repeat(WORKFLOW_LIMITS.maximumNoteCharacters + 1)},
          {registry: makeFixtureRegistry()}
        ),
      'maximumNoteCharacters'
    );
  });

  test('accepts the step limit and refuses parameter text and list values over document limits', () => {
    const registry = makeFixtureRegistry();
    const draft = parseWorkflowDraft(
      {
        title: 'Maximum steps',
        steps: Array.from({length: 64}, (_, index) => ({
          id: `step-${index}`,
          operationId: 'fixture-source'
        }))
      },
      {registry}
    );
    expect(draft.document.steps).toHaveLength(64);
    expect(draft.issues).toEqual([]);
    for (const [value, limit] of [
      ['x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters + 1), 'maximumTextCharacters'],
      [Array.from({length: 65}, () => 0), 'maximum parameter number-list length']
    ] as const) {
      expectMalformed(
        () =>
          parseWorkflowDraft(
            {
              title: 'Limits',
              steps: [{id: 'source', operationId: 'fixture-source', parameters: {count: value}}]
            },
            {registry}
          ),
        limit
      );
    }
  });

  test('step errors start with the step label or operation title and retain the argument path', () => {
    for (const label of ['My points', undefined]) {
      expect(() =>
        parseWorkflowDraft(
          {
            title: 'Title',
            steps: [
              {
                id: 'source',
                operationId: 'fixture-source',
                ...(label ? {label} : {}),
                parameters: {count: Infinity}
              }
            ]
          },
          {registry: makeFixtureRegistry()}
        )
      ).toThrow(
        expect.objectContaining({
          code: 'malformed',
          message: `${label ?? 'Fixture source'}: arguments.steps[0].parameters.count: must be a string, finite number, boolean or list of finite numbers.`
        })
      );
    }
  });

  test('refuses accessors and prototype keys without evaluating them', () => {
    const getter = vi.fn(() => {
      throw new Error('must never be read');
    });
    const args = Object.defineProperty({}, 'title', {enumerable: true, get: getter});
    expectMalformed(
      () => parseWorkflowDraft(args, {registry: makeFixtureRegistry()}),
      'arguments.title'
    );
    expect(getter).not.toHaveBeenCalled();
    expectMalformed(
      () =>
        parseWorkflowDraft(
          JSON.parse(
            '{"title":"Title","steps":[{"id":"source","operationId":"fixture-source","parameters":{"__proto__":2}}]}'
          ),
          {registry: makeFixtureRegistry()}
        ),
      '__proto__'
    );
    const sparseSteps = new Array(1);
    expectMalformed(
      () =>
        parseWorkflowDraft({title: 'Title', steps: sparseSteps}, {registry: makeFixtureRegistry()}),
      'arguments.steps[0]'
    );
  });

  test('uses the export outline verbatim and propagates exporter failures', () => {
    const outline = vi.spyOn(workflowExport, 'describeWorkflow');
    try {
      const registry = makeFixtureRegistry();
      outline.mockReturnValue('Export outline');
      const draft = parseWorkflowDraft(makeDraftArguments(), {registry});
      expect(draft.outline).toBe('Export outline');
      expect(outline).toHaveBeenCalledExactlyOnceWith(draft.document, registry);
      const error = new WorkflowError('malformed', 'Unexpected outline error');
      outline.mockImplementation(() => {
        throw error;
      });
      let receivedError: unknown;
      try {
        parseWorkflowDraft(makeDraftArguments(), {registry});
      } catch (caughtError) {
        receivedError = caughtError;
      }
      expect(receivedError).toBe(error);
    } finally {
      outline.mockRestore();
    }
  });
});
