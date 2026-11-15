// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {hashWorkflowValue} from '../../src/workflows/workflow-data';
import {parseWorkflowDocument} from '../../src/workflows/workflow-document';
import {
  getWorkflowDownstreamSteps,
  getWorkflowTopologicalOrder,
  getWorkflowUpstreamSteps,
  planWorkflow,
  resolveWorkflowParameters
} from '../../src/workflows/workflow-plan';
import {WORKFLOW_LIMITS, WorkflowError} from '../../src/workflows/workflow-types';
import type {
  WorkflowDocument,
  WorkflowInput,
  WorkflowOperation,
  WorkflowOperationRegistry,
  WorkflowPlan,
  WorkflowStep
} from '../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureHost, makeFixtureRegistry} from './workflow-fixtures';

function getStep(document: WorkflowDocument, stepId: string): WorkflowStep {
  return document.steps.find(step => step.id === stepId)!;
}

function makeRegistry(overrides: Partial<WorkflowOperation>): WorkflowOperationRegistry {
  const operations = makeFixtureRegistry()
    .list()
    .map(operation => (operation.id === overrides.id ? {...operation, ...overrides} : operation));
  return {
    list: () => operations,
    get: operationId => operations.find(operation => operation.id === operationId)
  };
}

function makePlan(document = makeFixtureDocument()): WorkflowPlan {
  return planWorkflow(document, {registry: makeFixtureRegistry()});
}

function readKeys(plan: WorkflowPlan): Record<string, string | null> {
  return Object.fromEntries(plan.order.map(stepId => [stepId, plan.steps[stepId].key]));
}

describe('workflow topological order and graph traversal', () => {
  test('orders dependencies before consumers regardless of positions or document order', () => {
    const document = makeFixtureDocument();
    document.steps.reverse();
    getStep(document, 'count').position.x = -100;
    expect(getWorkflowTopologicalOrder(document)).toEqual({
      order: ['source', 'scale', 'count'],
      cyclic: []
    });
    expect(getWorkflowUpstreamSteps(document, 'count')).toEqual(['source', 'scale']);
    expect(getWorkflowDownstreamSteps(document, 'source')).toEqual(['scale', 'count']);
    expect(getWorkflowUpstreamSteps(document, 'source')).toEqual([]);
    expect(getWorkflowDownstreamSteps(document, 'count')).toEqual([]);
    expect(getWorkflowUpstreamSteps(document, 'missing')).toEqual([]);
    expect(getWorkflowDownstreamSteps(document, 'missing')).toEqual([]);
  });

  test('breaks ties by x, then y, then id, including newly available steps', () => {
    const document = makeFixtureDocument();
    document.steps = ['last', 'bravo', 'alpha', 'first'].map((id, index) => ({
      ...document.steps[0],
      id,
      position: {x: index === 0 ? 1 : 0, y: index === 3 ? -1 : 0}
    }));
    expect(getWorkflowTopologicalOrder(document).order).toEqual([
      'first',
      'alpha',
      'bravo',
      'last'
    ]);
    document.steps[0].inputs = {unused: {stepId: 'alpha', output: 'data'}};
    document.steps[0].position = {x: -1, y: 0};
    expect(getWorkflowTopologicalOrder(document).order).toEqual([
      'first',
      'alpha',
      'last',
      'bravo'
    ]);
  });

  test('ignores missing producers and deduplicates parallel edges for ordering', () => {
    const document = makeFixtureDocument();
    getStep(document, 'source').inputs = {missing: {stepId: 'absent', output: 'data'}};
    getStep(document, 'scale').inputs['second'] = {stepId: 'source', output: 'data'};
    expect(getWorkflowTopologicalOrder(document)).toEqual({
      order: ['source', 'scale', 'count'],
      cyclic: []
    });
    expect(makePlan(document).steps['scale'].upstream).toEqual(['source']);
  });

  test('returns direct neighbours sorted by topological position', () => {
    const document = makeFixtureDocument();
    getStep(document, 'count').inputs['extra'] = {stepId: 'source', output: 'data'};
    const plan = makePlan(document);
    expect(plan.steps['source'].downstream).toEqual(['scale', 'count']);
    expect(plan.steps['count'].upstream).toEqual(['source', 'scale']);
    expect(plan.steps['scale'].downstream).toEqual(['count']);
  });

  test('appends only actual cycle members in id order and excludes self from traversals', () => {
    const document = makeFixtureDocument();
    getStep(document, 'source').inputs = {feedback: {stepId: 'scale', output: 'scaled'}};
    expect(getWorkflowTopologicalOrder(document)).toEqual({
      order: ['count', 'scale', 'source'],
      cyclic: ['scale', 'source']
    });
    expect(getWorkflowUpstreamSteps(document, 'scale')).toEqual(['source']);
    expect(getWorkflowDownstreamSteps(document, 'source')).toEqual(['count', 'scale']);
  });

  test('handles empty documents', () => {
    const document = makeFixtureDocument();
    document.steps = [];
    expect(makePlan(document)).toEqual({order: [], steps: {}, issues: []});
  });

  test('does not enqueue a removed cycle when an acyclic bridge feeds another cycle', () => {
    const document = makeFixtureDocument();
    const scale = document.steps[1];
    document.steps = [
      document.steps[0],
      {...scale, id: 'first-cycle', inputs: {points: {stepId: 'first-cycle', output: 'scaled'}}},
      {...scale, id: 'bridge', inputs: {points: {stepId: 'first-cycle', output: 'scaled'}}},
      {
        ...scale,
        id: 'second-cycle',
        inputs: {
          points: {stepId: 'second-cycle', output: 'scaled'},
          extra: {stepId: 'bridge', output: 'scaled'}
        }
      },
      {
        ...document.steps[2],
        inputs: {rows: {stepId: 'second-cycle', output: 'scaled'}}
      }
    ];
    expect(getWorkflowTopologicalOrder(document)).toEqual({
      order: ['source', 'bridge', 'count', 'first-cycle', 'second-cycle'],
      cyclic: ['first-cycle', 'second-cycle']
    });
    const plan = makePlan(document);
    expect(plan.steps['bridge'].status).toBe('blocked');
    expect(plan.steps['count'].status).toBe('blocked');
  });
});

describe('review regressions', () => {
  test.each([
    {kind: 'number', minimum: 0, maximum: 5, value: 6, requirement: 'at most 5'},
    {kind: 'number', minimum: 5, maximum: 10, value: 4, requirement: 'at least 5'},
    {kind: 'integer', minimum: 0, maximum: 5, value: 6, requirement: 'at most 5'},
    {kind: 'number-list', minimum: 0, maximum: 5, value: [1, 6], requirement: 'at most 5'},
    {kind: 'number-list', minimum: 0, maximum: 5, value: [-1, 3], requirement: 'at least 0'}
  ] as const)('F1 rejects bound $kind values outside workflow input limits: $value', input => {
    const document = makeFixtureDocument();
    document.inputs = [{id: 'shared-factor', label: 'Shared factor', ...input}].map(
      ({requirement: _requirement, ...workflowInput}) => workflowInput
    );
    getStep(document, 'scale').parameters['factor'] = {from: 'input', inputId: 'shared-factor'};
    const registry =
      input.kind === 'number-list'
        ? makeRegistry({
            id: 'fixture-scale',
            parameters: [{name: 'factor', label: 'Factor', kind: 'number-list', defaultValue: [1]}],
            normalizeParameters: raw => ({factor: raw['factor'] as WorkflowInput['value']})
          })
        : makeFixtureRegistry();
    const plan = planWorkflow(parseWorkflowDocument(document), {registry});
    expect(plan.steps['scale'].issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'factor',
      message: `Fixture scale: "Factor" bound to workflow input "Shared factor" must be ${input.requirement}; got ${JSON.stringify(input.value)}.`
    });
    expect(plan.steps['scale'].status).toBe('not-configured');
    expect(plan.steps['scale'].parameters).toBeUndefined();
    expect(plan.steps['scale'].key).toBeNull();
    expect(plan.steps['count'].status).toBe('blocked');
  });

  test.each([0, 5])('F1 accepts inclusive workflow input boundary %s', value => {
    const document = makeFixtureDocument();
    document.inputs = [
      {id: 'factor', label: 'Factor', kind: 'number', minimum: 0, maximum: 5, value}
    ];
    getStep(document, 'scale').parameters['factor'] = {from: 'input', inputId: 'factor'};
    const planned = makePlan(parseWorkflowDocument(document)).steps['scale'];
    expect(planned.parameters).toEqual({factor: value, fail: false});
    expect(planned.status).toBe('ready');
    expect(planned.key).not.toBeNull();
  });

  test('F1 rejects binding a numeric workflow input to a boolean parameter', () => {
    const document = makeFixtureDocument();
    document.inputs = [{id: 'flag', label: 'Shared flag', kind: 'number', value: 1}];
    getStep(document, 'scale').parameters['fail'] = {from: 'input', inputId: 'flag'};
    const planned = makePlan(parseWorkflowDocument(document)).steps['scale'];
    expect(planned.issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'fail',
      message:
        'Fixture scale: "Fail" cannot bind workflow input "Shared flag" of kind number to a boolean parameter.'
    });
    expect(planned.status).toBe('not-configured');
    expect(planned.key).toBeNull();
  });

  test('F1 still enforces operation limits when a workflow input has a wider range', () => {
    const document = makeFixtureDocument();
    document.inputs = [
      {id: 'factor', label: 'Factor', kind: 'number', minimum: 0, maximum: 200, value: 101}
    ];
    getStep(document, 'scale').parameters['factor'] = {from: 'input', inputId: 'factor'};
    const planned = makePlan(parseWorkflowDocument(document)).steps['scale'];
    expect(planned.issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'factor',
      message: 'Fixture scale: Factor is out of range'
    });
    expect(planned.status).toBe('not-configured');
    expect(planned.key).toBeNull();
  });

  test('F1 unused workflow inputs outside their limits do not invalidate steps', () => {
    const document = makeFixtureDocument();
    const expected = readKeys(makePlan(document));
    document.inputs = [
      {id: 'unused', label: 'Unused', kind: 'number', minimum: 0, maximum: 5, value: 6}
    ];
    expect(readKeys(makePlan(parseWorkflowDocument(document)))).toEqual(expected);
  });

  test('F1 validates a bound enum before operation normalization', () => {
    const document = makeFixtureDocument();
    document.inputs = [
      {id: 'mode', label: 'Shared mode', kind: 'enum', options: ['one', 'two'], value: 'one'}
    ];
    getStep(document, 'scale').parameters = {mode: {from: 'input', inputId: 'mode'}};
    let normalizationCalls = 0;
    const registry = makeRegistry({
      id: 'fixture-scale',
      parameters: [{name: 'mode', label: 'Mode', kind: 'string', defaultValue: 'one'}],
      normalizeParameters: raw => {
        normalizationCalls++;
        return {mode: raw['mode'] as string};
      }
    });
    const parsed = parseWorkflowDocument(document);
    expect(planWorkflow(parsed, {registry}).steps['scale'].status).toBe('ready');
    expect(normalizationCalls).toBe(1);
    parsed.inputs[0].value = 'three';
    const planned = planWorkflow(parsed, {registry}).steps['scale'];
    expect(normalizationCalls).toBe(1);
    expect(planned.issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'mode',
      message:
        'Fixture scale: "Mode" bound to workflow input "Shared mode" must be one of "one", "two"; got "three".'
    });
    expect(planned.key).toBeNull();
    expect(planned.status).toBe('not-configured');
  });

  test('F2 ignores inherited output observations, including built-in prototype names', () => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').inputs['points'].output = 'constructor';
    const registry = makeRegistry({
      id: 'fixture-source',
      outputs: [
        {
          name: 'constructor',
          label: 'Constructor',
          kind: 'point',
          identityColumns: [],
          maximumRows: 1
        }
      ]
    });
    const parsed = parseWorkflowDocument(document);
    const expected = planWorkflow(parsed, {registry});
    expect(expected.steps['scale'].status).toBe('ready');
    expect(planWorkflow(parsed, {registry, observedKinds: {source: {}}})).toEqual(expected);
    expect(
      planWorkflow(parsed, {
        registry,
        observedKinds: {source: Object.create({constructor: 'polygon'})}
      })
    ).toEqual(expected);
  });

  test('F2 ignores inherited producer observations but respects own observations', () => {
    const document = parseWorkflowDocument(makeFixtureDocument());
    const registry = makeFixtureRegistry();
    const inherited = Object.create({source: {data: 'polygon'}});
    expect(planWorkflow(document, {registry, observedKinds: inherited})).toEqual(
      planWorkflow(document, {registry})
    );
    expect(
      planWorkflow(document, {registry, observedKinds: {source: {data: 'polygon'}}}).steps['scale']
        .status
    ).toBe('not-configured');
  });

  test.each([
    {name: 'null prototype', makeError: () => Object.create(null)},
    {
      name: 'throwing conversion',
      makeError: () => ({
        toString() {
          throw new Error('Conversion failed');
        }
      })
    },
    {
      name: 'throwing error message',
      makeError: () =>
        Object.defineProperty(new Error(), 'message', {
          get() {
            throw new Error('Message unavailable');
          }
        })
    },
    {
      name: 'throwing prototype trap',
      makeError: () =>
        new Proxy(
          {},
          {
            getPrototypeOf() {
              throw new Error('Prototype unavailable');
            }
          }
        )
    }
  ])('F3 contains normalizer throws with $name', ({makeError}) => {
    const registry = makeRegistry({
      id: 'fixture-scale',
      normalizeParameters: () => {
        throw makeError();
      }
    });
    const document = parseWorkflowDocument(makeFixtureDocument());
    const first = planWorkflow(document, {registry});
    expect(first.steps['scale'].issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      message: 'Fixture scale: parameter normalization failed with an unreadable error.'
    });
    expect(first.steps['scale'].status).toBe('not-configured');
    expect(first.steps['scale'].parameters).toBeUndefined();
    expect(first.steps['scale'].key).toBeNull();
    expect(first.steps['count'].status).toBe('blocked');
    expect(planWorkflow(document, {registry})).toEqual(first);
  });

  test.each([
    {
      title: 'Count points',
      labels: ['Count', 'Factor'],
      message: 'Count points: Factor must be at most 1; got 3.'
    },
    {
      title: 'Scale points',
      labels: ['Factor', 'Factor limit'],
      message: 'Scale points: Factor limit must be at most 1; got 3.'
    },
    {
      title: 'Scale points',
      labels: ['Limit', 'Limitations'],
      message: 'Scale points: Limitations must be at most 1; got 3.'
    },
    {
      title: 'Scale points',
      labels: ['Count', 'Factor'],
      message: 'Scale points: ReCounted Factor must be at most 1; got 3.'
    }
  ])('F4 attributes errors after removing $title and resolving overlapping labels', ({
    title,
    labels,
    message
  }) => {
    const registry = makeRegistry({
      id: 'fixture-scale',
      title,
      parameters: labels.map((label, index) => ({
        name: index === 0 ? 'count' : 'factor',
        label,
        kind: 'number',
        defaultValue: 1
      })),
      normalizeParameters: () => {
        throw new Error(message);
      }
    });
    const planned = planWorkflow(parseWorkflowDocument(makeFixtureDocument()), {registry}).steps[
      'scale'
    ];
    expect(planned.issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'factor',
      message
    });
  });

  test('F4 explicit WorkflowError attribution takes precedence over matching message labels', () => {
    const registry = makeRegistry({
      id: 'fixture-scale',
      normalizeParameters: () => {
        throw new WorkflowError(
          'parameter-out-of-range',
          'Fixture scale: Factor conflicts with Fail.',
          {parameter: 'fail'}
        );
      }
    });
    expect(
      planWorkflow(parseWorkflowDocument(makeFixtureDocument()), {registry}).steps['scale'].issues
    ).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'fail',
      message: 'Fixture scale: Factor conflicts with Fail.'
    });
  });
});

describe('workflow parameter resolution', () => {
  test('merges defaults, static values and workflow inputs while skipping unknown names', () => {
    const document = makeFixtureDocument();
    const step = getStep(document, 'scale');
    const operation = makeFixtureRegistry().get('fixture-scale')!;
    expect(resolveWorkflowParameters(document, step, operation)).toEqual({factor: 3, fail: false});
    document.inputs = [{id: 'multiplier', label: 'Multiplier', kind: 'number', value: 7}];
    step.parameters = {
      factor: {from: 'input', inputId: 'multiplier'},
      unknown: {from: 'static', value: 100}
    };
    expect(resolveWorkflowParameters(document, step, operation)).toEqual({factor: 7, fail: false});
    step.parameters['factor'] = {from: 'input', inputId: 'missing'};
    expect(resolveWorkflowParameters(document, step, operation)).toEqual({factor: 2, fail: false});
    step.parameters['factor'] = {from: 'static', value: 0};
    step.parameters['fail'] = {from: 'static', value: false};
    expect(resolveWorkflowParameters(document, step, operation)).toEqual({factor: 0, fail: false});
  });

  test('copies number-list values so normalisation cannot mutate the document or defaults', () => {
    const document = makeFixtureDocument();
    const operation = {
      ...makeFixtureRegistry().get('fixture-source')!,
      parameters: [
        {name: 'values', label: 'Values', kind: 'number-list' as const, defaultValue: [1]}
      ]
    };
    const first = resolveWorkflowParameters(document, document.steps[0], operation);
    (first['values'] as number[]).push(2);
    expect(operation.parameters[0].defaultValue).toEqual([1]);
    document.steps[0].parameters = {values: {from: 'static', value: [3]}};
    const second = resolveWorkflowParameters(document, document.steps[0], operation);
    (second['values'] as number[]).push(4);
    expect(document.steps[0].parameters['values']).toEqual({from: 'static', value: [3]});
  });
});

describe('workflow planning issues', () => {
  test('reports unresolved geometry and upper-bound capacity as information', () => {
    const plan = makePlan();
    expect(plan.issues).toEqual([]);
    expect(plan.steps['scale'].issues).toEqual([
      {
        severity: 'info',
        code: 'unverified-port',
        stepId: 'scale',
        port: 'points',
        message: 'Fixture scale: "Points" accepts points; "Data" of Fixture source is features.'
      },
      {
        severity: 'info',
        code: 'capacity-may-exceed',
        stepId: 'scale',
        port: 'points',
        message:
          'Fixture scale: "Points" accepts at most 10 rows; "Data" of Fixture source allows up to 50 rows (an upper bound).'
      }
    ]);
    expect(plan.order.map(stepId => plan.steps[stepId].status)).toEqual([
      'ready',
      'ready',
      'ready'
    ]);
  });

  test('unknown operations are not configured and block their consumers', () => {
    const document = makeFixtureDocument();
    getStep(document, 'source').operationId = 'removed';
    const plan = makePlan(document);
    expect(plan.steps['source'].issues).toEqual([
      {
        severity: 'error',
        code: 'unknown-operation',
        stepId: 'source',
        message: 'source: operation "removed" is not available.'
      }
    ]);
    expect(plan.steps['source'].status).toBe('not-configured');
    expect(plan.steps['scale'].status).toBe('blocked');
    expect(plan.steps['count'].status).toBe('blocked');
    expect(readKeys(plan)).toEqual({source: null, scale: null, count: null});
  });

  test.each([
    {
      name: 'unknown input',
      inputs: {mystery: {stepId: 'source', output: 'data'}},
      code: 'unknown-input-port',
      port: 'mystery',
      message: 'Fixture scale: input "mystery" is not declared.'
    },
    {
      name: 'missing step',
      inputs: {points: {stepId: 'missing', output: 'data'}},
      code: 'unknown-step',
      port: 'points',
      message: 'Fixture scale: "Points" refers to missing step "missing".'
    },
    {
      name: 'unknown output',
      inputs: {points: {stepId: 'source', output: 'missing'}},
      code: 'unknown-output-port',
      port: 'points',
      message:
        'Fixture scale: "Points" refers to output "missing" that Fixture source does not declare.'
    },
    {
      name: 'required input',
      inputs: {},
      code: 'input-required',
      port: 'points',
      message: 'Fixture scale: needs an input for "Points".'
    }
  ])('reports $name precisely', ({inputs, code, port, message}) => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').inputs = inputs as WorkflowStep['inputs'];
    const plan = makePlan(document);
    expect(plan.steps['scale'].issues).toContainEqual({
      severity: 'error',
      stepId: 'scale',
      code,
      port,
      message
    });
    expect(plan.steps['scale'].status).toBe('not-configured');
    expect(plan.steps['scale'].key).toBeNull();
    expect(plan.steps['count'].status).toBe('blocked');
  });

  test('allows absent optional inputs', () => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').inputs = {};
    const operation = makeFixtureRegistry().get('fixture-scale')!;
    const registry = makeRegistry({
      id: operation.id,
      inputs: operation.inputs.map(input => ({...input, optional: true}))
    });
    const plan = planWorkflow(document, {registry});
    expect(plan.steps['scale'].issues).toEqual([]);
    expect(plan.steps['scale'].status).toBe('ready');
  });

  test('observed kinds settle compatibility into yes or no using labels', () => {
    const document = makeFixtureDocument();
    getStep(document, 'source').label = 'Source points';
    getStep(document, 'scale').label = 'Scale points';
    const registry = makeFixtureRegistry();
    const accepted = planWorkflow(document, {
      registry,
      observedKinds: {source: {data: 'point'}}
    });
    expect(accepted.steps['scale'].issues.map(issue => issue.code)).toEqual([
      'capacity-may-exceed'
    ]);
    expect(accepted.steps['scale'].status).toBe('ready');
    const refused = planWorkflow(document, {
      registry,
      observedKinds: {source: {data: 'polygon'}}
    });
    expect(refused.steps['scale'].issues[0]).toEqual({
      severity: 'error',
      code: 'incompatible-port',
      stepId: 'scale',
      port: 'points',
      message: 'Scale points: "Points" accepts points; "Data" of Source points is polygons.'
    });
    expect(refused.steps['scale'].status).toBe('not-configured');
    expect(refused.steps['count'].status).toBe('blocked');
    expect(refused.steps['source'].key).toBe(accepted.steps['source'].key);
  });

  test('capacity at the limit produces no warning', () => {
    const operation = makeFixtureRegistry().get('fixture-source')!;
    const registry = makeRegistry({
      id: operation.id,
      outputs: operation.outputs.map(output => ({...output, kind: 'point', maximumRows: 10}))
    });
    expect(planWorkflow(makeFixtureDocument(), {registry}).steps['scale'].issues).toEqual([]);
  });

  test('unknown parameters and workflow inputs use parameter labels', () => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').parameters = {
      mystery: {from: 'static', value: 1},
      factor: {from: 'input', inputId: 'missing'}
    };
    const planned = makePlan(document).steps['scale'];
    expect(planned.issues.filter(issue => issue.severity === 'error')).toEqual([
      {
        severity: 'error',
        code: 'unknown-workflow-input',
        stepId: 'scale',
        parameter: 'factor',
        message: 'Fixture scale: "Factor" refers to missing workflow input "missing".'
      },
      {
        severity: 'error',
        code: 'unknown-parameter',
        stepId: 'scale',
        parameter: 'mystery',
        message: 'Fixture scale: parameter "mystery" is not declared.'
      }
    ]);
    expect(planned.parameters).toEqual({factor: 2, fail: false});
    expect(planned.status).toBe('not-configured');
  });

  test.each([-1, 101])('normalisation failures name the parameter for factor %s', factor => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').label = 'Scale values';
    getStep(document, 'scale').parameters['factor'] = {from: 'static', value: factor};
    const planned = makePlan(document).steps['scale'];
    expect(planned.issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'factor',
      message: 'Scale values: Factor is out of range'
    });
    expect(planned.parameters).toBeUndefined();
    expect(planned.status).toBe('not-configured');
    expect(planned.key).toBeNull();
  });

  test.each([0, 100])('normalisation accepts the factor boundary %s', factor => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').parameters['factor'] = {from: 'static', value: factor};
    expect(makePlan(document).steps['scale'].status).toBe('ready');
  });

  test('preserves WorkflowError codes and parameter details, with one step prefix', () => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').label = 'Scale values';
    const registry = makeRegistry({
      id: 'fixture-scale',
      normalizeParameters: () => {
        throw new WorkflowError('limit-exceeded', 'Scale values: Factor must not exceed 100.', {
          parameter: 'factor'
        });
      }
    });
    expect(planWorkflow(document, {registry}).steps['scale'].issues).toContainEqual({
      severity: 'error',
      code: 'limit-exceeded',
      stepId: 'scale',
      parameter: 'factor',
      message: 'Scale values: Factor must not exceed 100.'
    });
  });

  test('contains non-Error normalisation throws as configuration issues', () => {
    const registry = makeRegistry({
      id: 'fixture-scale',
      normalizeParameters: () => {
        throw 'Factor is invalid';
      }
    });
    expect(planWorkflow(makeFixtureDocument(), {registry}).steps['scale'].issues).toContainEqual({
      severity: 'error',
      code: 'parameter-out-of-range',
      stepId: 'scale',
      parameter: 'factor',
      message: 'Fixture scale: Factor is invalid'
    });
  });

  test('unknown display outputs warn without blocking the step', () => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').display = ['missing'];
    const planned = makePlan(document).steps['scale'];
    expect(planned.issues).toContainEqual({
      severity: 'warning',
      code: 'display-unknown-output',
      stepId: 'scale',
      port: 'missing',
      message: 'Fixture scale: display output "missing" is not declared.'
    });
    expect(planned.status).toBe('ready');
    expect(planned.key).not.toBeNull();
  });

  test('cycles mark only cycle members as not configured and their dependents as blocked', () => {
    const document = makeFixtureDocument();
    getStep(document, 'source').inputs = {feedback: {stepId: 'scale', output: 'scaled'}};
    const plan = makePlan(document);
    expect(plan.issues).toEqual([
      {
        severity: 'error',
        code: 'cycle',
        message: 'Fixture workflow: a cycle connects Fixture scale, Fixture source.'
      }
    ]);
    for (const stepId of ['source', 'scale']) {
      expect(plan.steps[stepId].status).toBe('not-configured');
      expect(plan.steps[stepId].issues).toContainEqual({
        severity: 'error',
        code: 'cycle',
        stepId,
        message: `${stepId === 'source' ? 'Fixture source' : 'Fixture scale'}: cannot read itself, directly or through a cycle.`
      });
    }
    expect(plan.steps['count'].status).toBe('blocked');
    expect(readKeys(plan)).toEqual({count: null, scale: null, source: null});
  });

  test('rejects self reads', () => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').inputs = {points: {stepId: 'scale', output: 'scaled'}};
    expect(getWorkflowTopologicalOrder(document)).toEqual({
      order: ['source', 'count', 'scale'],
      cyclic: ['scale']
    });
    expect(makePlan(document).steps['scale'].issues.map(issue => issue.code)).toContain('cycle');
  });

  test('reports the document step limit exactly', () => {
    const document = makeFixtureDocument();
    document.steps = Array.from({length: WORKFLOW_LIMITS.maximumSteps + 1}, (_, index) => ({
      ...document.steps[0],
      id: `source-${index}`
    }));
    expect(makePlan(document).issues).toEqual([
      {
        severity: 'error',
        code: 'limit-exceeded',
        message: 'Fixture workflow: steps must not exceed 64; got 65.'
      }
    ]);
  });
});

describe('workflow statuses and content keys', () => {
  test('disabling the middle step leaves upstream ready and blocks downstream', () => {
    const document = makeFixtureDocument();
    getStep(document, 'scale').disabled = true;
    const plan = makePlan(document);
    expect(plan.steps['source'].status).toBe('ready');
    expect(plan.steps['source'].key).not.toBeNull();
    expect(plan.steps['scale'].status).toBe('disabled');
    expect(plan.steps['count'].status).toBe('blocked');
    expect(plan.steps['count'].issues).toEqual([
      {
        severity: 'info',
        code: 'upstream-disabled',
        stepId: 'count',
        message: 'Fixture count: upstream step "Fixture scale" is disabled.'
      }
    ]);
    expect(plan.steps['scale'].key).toBeNull();
    expect(plan.steps['count'].key).toBeNull();
  });

  test('finds the nearest disabled or not configured ancestor transitively', () => {
    const document = makeFixtureDocument();
    getStep(document, 'source').disabled = true;
    let plan = makePlan(document);
    expect(plan.steps['count'].issues).toEqual([
      {
        severity: 'info',
        code: 'upstream-disabled',
        stepId: 'count',
        message: 'Fixture count: upstream step "Fixture source" is disabled.'
      }
    ]);
    getStep(document, 'scale').parameters['factor'] = {from: 'static', value: -1};
    plan = makePlan(document);
    expect(plan.steps['count'].issues).toEqual([
      {
        severity: 'info',
        code: 'upstream-not-configured',
        stepId: 'count',
        message: 'Fixture count: upstream step "Fixture scale" is not configured.'
      }
    ]);
    getStep(document, 'scale').disabled = true;
    expect(makePlan(document).steps['scale'].status).toBe('disabled');
  });

  test('is deterministic, pure and never runs operations', () => {
    const document = makeFixtureDocument();
    const snapshot = structuredClone(document);
    const calls: string[] = [];
    const options = {registry: makeFixtureRegistry(calls), host: makeFixtureHost()};
    const first = planWorkflow(document, options);
    expect(planWorkflow(document, options)).toEqual(first);
    expect(document).toEqual(snapshot);
    expect(calls).toEqual([]);
    expect(first.steps['source'].key).toBe(
      hashWorkflowValue(['fixture-source', '1', {count: 4}, 'cpu', 'fixture@1', []])
    );
    expect(first.steps['scale'].key).toBe(
      hashWorkflowValue([
        'fixture-scale',
        '1',
        {factor: 3, fail: false},
        'auto',
        undefined,
        [['points', first.steps['source'].key, 'data']]
      ])
    );
  });

  test('presentation and fallback edits do not change any keys', () => {
    const document = makeFixtureDocument();
    const before = readKeys(makePlan(document));
    document.title = 'Renamed';
    document.description = 'New description';
    document.settings.autoRun = true;
    document.steps.reverse();
    for (const step of document.steps) {
      step.label = `New ${step.id}`;
      step.position = {x: -100, y: 300};
      step.note = 'A note';
      step.display = [];
      step.onFallback = 'stop';
    }
    expect(readKeys(makePlan(document))).toEqual(before);
  });

  test('parameter edits affect exactly the step and its dependents', () => {
    const document = makeFixtureDocument();
    const before = readKeys(makePlan(document));
    getStep(document, 'scale').parameters['factor'] = {from: 'static', value: 4};
    const after = readKeys(makePlan(document));
    expect(after['source']).toBe(before['source']);
    expect(after['scale']).not.toBe(before['scale']);
    expect(after['count']).not.toBe(before['count']);
  });

  test('operation versions affect the operation and its dependents', () => {
    const before = makePlan();
    const after = planWorkflow(makeFixtureDocument(), {
      registry: makeRegistry({id: 'fixture-scale', version: '2'})
    });
    expect(after.steps['source'].key).toBe(before.steps['source'].key);
    expect(after.steps['scale'].key).not.toBe(before.steps['scale'].key);
    expect(after.steps['count'].key).not.toBe(before.steps['count'].key);
  });

  test('host revisions invalidate the source and every dependent', () => {
    const document = makeFixtureDocument();
    const host = makeFixtureHost();
    const registry = makeFixtureRegistry();
    const before = planWorkflow(document, {registry, host});
    host.revision++;
    const after = planWorkflow(document, {registry, host});
    for (const stepId of before.order) {
      expect(after.steps[stepId].key).not.toBe(before.steps[stepId].key);
    }
  });

  test('external revision throws produce a deterministic error key without throwing', () => {
    const document = makeFixtureDocument();
    const registry = makeRegistry({
      id: 'fixture-source',
      getExternalRevisionKey: () => {
        throw new Error('Dataset unavailable');
      }
    });
    const options = {registry, host: makeFixtureHost()};
    const first = planWorkflow(document, options);
    expect(first.steps['source'].status).toBe('ready');
    expect(first.steps['source'].key).toBe(
      hashWorkflowValue(['fixture-source', '1', {count: 4}, 'cpu', 'error', []])
    );
    expect(planWorkflow(document, options)).toEqual(first);
    expect(first.steps['source'].key).not.toBe(
      planWorkflow(document, {registry}).steps['source'].key
    );
  });

  test('engine switches affect only GPU-capable operations directly, then propagate', () => {
    const document = makeFixtureDocument();
    document.steps.push({
      ...getStep(document, 'count'),
      id: 'independent-count',
      inputs: {rows: {stepId: 'source', output: 'data'}}
    });
    const before = makePlan(document);
    document.settings.engine = 'cpu';
    const after = makePlan(document);
    expect(after.steps['source'].key).toBe(before.steps['source'].key);
    expect(after.steps['independent-count'].key).toBe(before.steps['independent-count'].key);
    expect(after.steps['scale'].key).not.toBe(before.steps['scale'].key);
    expect(after.steps['count'].key).not.toBe(before.steps['count'].key);
    expect(before.steps['source'].engine).toBe('cpu');
    expect(before.steps['count'].engine).toBe('cpu');
    expect(before.steps['scale'].engine).toBe('auto');
    expect(after.steps['scale'].engine).toBe('cpu');
    getStep(document, 'scale').engine = 'auto';
    expect(readKeys(makePlan(document))).toEqual(readKeys(before));
  });

  test('workflow inputs affect only bound steps and their dependents', () => {
    const document = makeFixtureDocument();
    document.inputs = [
      {id: 'factor', label: 'Factor', kind: 'number', value: 3},
      {id: 'unused', label: 'Unused', kind: 'number', value: 1}
    ];
    getStep(document, 'scale').parameters['factor'] = {from: 'input', inputId: 'factor'};
    const before = makePlan(document);
    document.inputs[1].value = 9;
    expect(readKeys(makePlan(document))).toEqual(readKeys(before));
    document.inputs[0].value = 4;
    const after = makePlan(document);
    expect(after.steps['source'].key).toBe(before.steps['source'].key);
    expect(after.steps['scale'].key).not.toBe(before.steps['scale'].key);
    expect(after.steps['count'].key).not.toBe(before.steps['count'].key);
    expect(after.steps['scale'].parameters).toEqual({factor: 4, fail: false});
  });

  test('connection keys use input name order regardless of insertion order', () => {
    const document = makeFixtureDocument();
    const operation = makeFixtureRegistry().get('fixture-count')!;
    const registry = makeRegistry({
      id: operation.id,
      inputs: [operation.inputs[0], {...operation.inputs[0], name: 'another', label: 'Another'}]
    });
    const step = getStep(document, 'count');
    step.inputs['another'] = {stepId: 'source', output: 'data'};
    const before = planWorkflow(document, {registry});
    step.inputs = {another: step.inputs['another'], rows: step.inputs['rows']};
    expect(readKeys(planWorkflow(document, {registry}))).toEqual(readKeys(before));
    expect(before.steps['count'].key).toBe(
      hashWorkflowValue([
        'fixture-count',
        '1',
        {},
        'cpu',
        undefined,
        [
          ['another', before.steps['source'].key, 'data'],
          ['rows', before.steps['scale'].key, 'scaled']
        ]
      ])
    );
    step.inputs['another'] = {stepId: 'scale', output: 'scaled'};
    expect(planWorkflow(document, {registry}).steps['count'].key).not.toBe(
      before.steps['count'].key
    );
  });

  test('supports ids that match object prototype properties', () => {
    const document = makeFixtureDocument();
    document.steps[0].id = 'constructor';
    getStep(document, 'scale').inputs['points'].stepId = 'constructor';
    const plan = makePlan(document);
    expect(plan.order).toEqual(['constructor', 'scale', 'count']);
    expect(plan.steps['constructor'].status).toBe('ready');
    expect(plan.steps['scale'].status).toBe('ready');
  });

  test('plans a 64-step chain in under 50 milliseconds', () => {
    const document = makeFixtureDocument();
    const source = document.steps[0];
    const scale = document.steps[1];
    document.steps = [source];
    for (let index = 1; index < WORKFLOW_LIMITS.maximumSteps; index++) {
      document.steps.push({
        ...scale,
        id: `scale-${index}`,
        inputs: {
          points: {
            stepId: index === 1 ? source.id : `scale-${index - 1}`,
            output: index === 1 ? 'data' : 'scaled'
          }
        }
      });
    }
    const registry = makeFixtureRegistry();
    const startedAt = performance.now();
    const plan = planWorkflow(document, {registry});
    const durationMilliseconds = performance.now() - startedAt;
    expect(plan.order).toHaveLength(64);
    expect(plan.issues).toEqual([]);
    expect(plan.order.every(stepId => plan.steps[stepId].status === 'ready')).toBe(true);
    expect(durationMilliseconds).toBeLessThan(50);
  });
});
