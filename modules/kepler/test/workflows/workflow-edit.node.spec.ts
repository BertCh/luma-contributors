// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {serializeWorkflowDocument} from '../../src/workflows/workflow-document';
import {WorkflowHistory} from '../../src/workflows/workflow-history';
import {
  addWorkflowStep,
  bindWorkflowParameter,
  checkWorkflowConnection,
  copyWorkflowSteps,
  connectWorkflowSteps,
  demoteWorkflowParameter,
  disconnectWorkflowInput,
  duplicateWorkflowSteps,
  insertWorkflowSteps,
  layoutWorkflow,
  parseWorkflowClipboard,
  pasteWorkflowSteps,
  moveWorkflowSteps,
  promoteWorkflowParameter,
  removeWorkflowInput,
  removeWorkflowSteps,
  renameWorkflow,
  replaceWorkflowStepOperation,
  setWorkflowDisplayStyle,
  setWorkflowParameter,
  updateWorkflowInput,
  updateWorkflowSettings,
  updateWorkflowStep
} from '../../src/workflows/workflow-edit';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowDocument,
  type WorkflowDisplayStyle,
  type WorkflowInput,
  type WorkflowIssueCode,
  type WorkflowOperation,
  type WorkflowParameterSpec,
  type WorkflowStep
} from '../../src/workflows/workflow-types';
import {makeFixtureDocument, makeFixtureRegistry} from './workflow-fixtures';

const registry = makeFixtureRegistry();

function freezeValue(value: unknown): void {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Object.values(value)) freezeValue(child);
    Object.freeze(value);
  }
}

function edit<Result>(document: WorkflowDocument, action: () => Result): Result {
  const snapshot = structuredClone(document);
  freezeValue(document);
  try {
    return action();
  } finally {
    expect(document).toEqual(snapshot);
  }
}

function expectCode(action: () => unknown, code: WorkflowIssueCode, message?: string): void {
  expect(action).toThrow(WorkflowError);
  try {
    action();
  } catch (error) {
    expect(error).toBeInstanceOf(WorkflowError);
    expect((error as WorkflowError).code).toBe(code);
    if (message) expect((error as WorkflowError).message).toContain(message);
  }
}

function step(id: string, upstream?: string): WorkflowStep {
  return {
    id,
    operationId: 'fixture-scale',
    inputs: upstream ? {points: {stepId: upstream, output: 'scaled'}} : {},
    parameters: {},
    position: {x: 0, y: 0}
  };
}

function registryWithParameters(parameters: readonly WorkflowParameterSpec[]) {
  const operation = {...registry.get('fixture-scale')!, parameters};
  return {
    get: (identifier: string) =>
      identifier === operation.id ? operation : registry.get(identifier),
    list: () => [operation]
  };
}

function withInput(input: WorkflowInput): WorkflowDocument {
  return {...makeFixtureDocument(), inputs: [input]};
}

describe('workflow step edits', () => {
  test('adds known parameters as static bindings without range checking and shares existing steps', () => {
    const document = makeFixtureDocument();
    const result = edit(document, () =>
      addWorkflowStep(document, registry, {
        operationId: 'fixture-scale',
        parameters: {factor: -100},
        label: '  New scale  '
      })
    );
    expect(result.stepId).toBe('fixture-scale');
    expect(result.document.steps[3]).toEqual({
      id: 'fixture-scale',
      operationId: 'fixture-scale',
      label: 'New scale',
      parameters: {factor: {from: 'static', value: -100}},
      inputs: {},
      position: {x: 840, y: 0}
    });
    document.steps.forEach((original, index) =>
      expect(result.document.steps[index]).toBe(original)
    );
    expect(result.document.inputs).toBe(document.inputs);
    const second = edit(result.document, () =>
      addWorkflowStep(result.document, registry, {operationId: 'fixture-scale'})
    );
    expect(second.stepId).toBe('fixture-scale-2');
  });

  test('places the first step at the origin, rounds explicit positions and honours an explicit id', () => {
    const document = {...makeFixtureDocument(), steps: []};
    expect(
      edit(document, () => addWorkflowStep(document, registry, {operationId: 'fixture-source'}))
        .document.steps[0].position
    ).toEqual({x: 0, y: 0});
    const result = edit(document, () =>
      addWorkflowStep(document, registry, {
        operationId: 'fixture-source',
        id: 'chosen',
        position: {x: 1.6, y: -2.2}
      })
    );
    expect(result.stepId).toBe('chosen');
    expect(result.document.steps[0].position).toEqual({x: 2, y: -2});
  });

  test('uses the rightmost step as the placement anchor when no producer is selected', () => {
    const document = {
      ...makeFixtureDocument(),
      steps: [
        {...step('left'), position: {x: -900, y: 40}},
        {...step('right'), position: {x: -200.2, y: 310.7}}
      ]
    };
    const result = edit(document, () =>
      addWorkflowStep(document, registry, {operationId: 'fixture-source'})
    );
    expect(result.document.steps.at(-1)!.position).toEqual({x: 80, y: 311});
  });

  test('connects maybe-compatible outputs and finds a free row beside the producer', () => {
    const document = makeFixtureDocument();
    const result = edit(document, () =>
      addWorkflowStep(document, registry, {
        operationId: 'fixture-scale',
        connectFrom: {stepId: 'source', output: 'data'}
      })
    );
    expect(result.document.steps[3].inputs).toEqual({points: {stepId: 'source', output: 'data'}});
    expect(result.document.steps[3].position).toEqual({x: 280, y: 150});
    const next = edit(result.document, () =>
      addWorkflowStep(result.document, registry, {
        operationId: 'fixture-scale',
        connectFrom: {stepId: 'source', output: 'data'},
        connectTo: 'points'
      })
    );
    expect(next.document.steps[4].position).toEqual({x: 280, y: 300});
    const unconnected = edit(document, () =>
      addWorkflowStep(document, registry, {
        operationId: 'fixture-source',
        connectFrom: {stepId: 'source', output: 'data'}
      })
    );
    expect(unconnected.document.steps[3].inputs).toEqual({});
  });

  test('refuses unknown operations, parameters, invalid or duplicate ids and the step limit', () => {
    const document = makeFixtureDocument();
    edit(document, () => {
      expectCode(
        () => addWorkflowStep(document, registry, {operationId: 'missing'}),
        'unknown-operation'
      );
      expectCode(
        () =>
          addWorkflowStep(document, registry, {
            operationId: 'fixture-scale',
            parameters: {missing: 3}
          }),
        'unknown-parameter',
        'Parameter missing'
      );
      expectCode(
        () => addWorkflowStep(document, registry, {operationId: 'fixture-scale', id: 'scale'}),
        'duplicate-id'
      );
      for (const identifier of ['', 'Bad id', 'a'.repeat(49)])
        expectCode(
          () => addWorkflowStep(document, registry, {operationId: 'fixture-scale', id: identifier}),
          'parameter-out-of-range'
        );
    });
    const full = {
      ...document,
      steps: Array.from({length: WORKFLOW_LIMITS.maximumSteps}, (_, index) => step(`step-${index}`))
    };
    edit(full, () =>
      expectCode(
        () => addWorkflowStep(full, registry, {operationId: 'fixture-source'}),
        'limit-exceeded',
        'maximumSteps (64)'
      )
    );
  });

  test('removes steps and their consumers connections, keeping unrelated parameters and objects', () => {
    const document = makeFixtureDocument();
    const result = edit(document, () =>
      removeWorkflowSteps(document, ['source', 'missing', 'source'])
    );
    expect(result.steps.map(candidate => candidate.id)).toEqual(['scale', 'count']);
    expect(result.steps[0].inputs).toEqual({});
    expect(result.steps[0].parameters).toBe(document.steps[1].parameters);
    expect(result.steps[1]).toBe(document.steps[2]);
    expect(edit(document, () => removeWorkflowSteps(document, ['missing']))).toBe(document);
  });

  test('replaces and disconnects input connections, with exact no-op identity', () => {
    const document = makeFixtureDocument();
    const connection = {
      from: {stepId: 'source', output: 'data'},
      to: {stepId: 'count', input: 'rows'}
    };
    expect(edit(document, () => checkWorkflowConnection(document, registry, connection))).toEqual({
      valid: true
    });
    const result = edit(document, () => connectWorkflowSteps(document, registry, connection));
    expect(result.steps[2].inputs).toEqual({rows: connection.from});
    expect(result.steps[0]).toBe(document.steps[0]);
    expect(result.steps[1]).toBe(document.steps[1]);
    expect(edit(result, () => connectWorkflowSteps(result, registry, connection))).toBe(result);
    const disconnected = edit(result, () => disconnectWorkflowInput(result, 'count', 'rows'));
    expect(disconnected.steps[2].inputs).toEqual({});
    expect(edit(disconnected, () => disconnectWorkflowInput(disconnected, 'count', 'rows'))).toBe(
      disconnected
    );
  });

  test('connection refusals name exact codes and check returns the same readable reason', () => {
    const document = makeFixtureDocument();
    const cases = [
      {
        from: {stepId: 'missing', output: 'data'},
        to: {stepId: 'scale', input: 'points'},
        code: 'unknown-step'
      },
      {
        from: {stepId: 'source', output: 'data'},
        to: {stepId: 'missing', input: 'points'},
        code: 'unknown-step'
      },
      {
        from: {stepId: 'source', output: 'missing'},
        to: {stepId: 'scale', input: 'points'},
        code: 'unknown-output-port'
      },
      {
        from: {stepId: 'source', output: 'data'},
        to: {stepId: 'scale', input: 'missing'},
        code: 'unknown-input-port'
      },
      {
        from: {stepId: 'scale', output: 'scaled'},
        to: {stepId: 'scale', input: 'points'},
        code: 'cycle'
      }
    ] as const;
    edit(document, () => {
      for (const connection of cases) {
        expectCode(() => connectWorkflowSteps(document, registry, connection), connection.code);
        const checked = checkWorkflowConnection(document, registry, connection);
        expect(checked.valid).toBe(false);
        expect(checked.reason).toMatch(/^[A-Z].*\.$/);
      }
    });
    const unknown = {
      ...document,
      steps: document.steps.map(candidate =>
        candidate.id === 'scale' ? {...candidate, operationId: 'missing'} : candidate
      )
    };
    edit(unknown, () =>
      expectCode(() => connectWorkflowSteps(unknown, registry, cases[2]), 'unknown-operation')
    );
  });

  test('refuses a three-step cycle', () => {
    const document = {
      ...makeFixtureDocument(),
      steps: [step('first'), step('second', 'first'), step('third', 'second')]
    };
    edit(document, () =>
      expectCode(
        () =>
          connectWorkflowSteps(document, registry, {
            from: {stepId: 'third', output: 'scaled'},
            to: {stepId: 'first', input: 'points'}
          }),
        'cycle',
        'cannot create a cycle'
      )
    );
  });

  test('refuses incompatible kinds with both kinds in the message and leaves an automatic add unconnected', () => {
    const polygonOperation = {
      ...registry.get('fixture-scale')!,
      inputs: [
        {name: 'polygons', label: 'Polygons', accepts: ['polygon' as const], maximumRows: 10}
      ]
    };
    const polygonRegistry = {
      get: (identifier: string) =>
        identifier === 'fixture-scale' ? polygonOperation : registry.get(identifier),
      list: () => [polygonOperation]
    };
    const document = makeFixtureDocument();
    const connection = {
      from: {stepId: 'scale', output: 'scaled'},
      to: {stepId: 'count', input: 'polygons'}
    };
    const target = {
      ...document,
      steps: [...document.steps, {...step('target'), label: 'Target polygons'}]
    };
    connection.to.stepId = 'target';
    edit(target, () => {
      expectCode(
        () => connectWorkflowSteps(target, polygonRegistry, connection),
        'incompatible-port',
        'Target polygons: Input Polygons accepts polygon; output Scaled has kind point.'
      );
      expect(checkWorkflowConnection(target, polygonRegistry, connection)).toEqual({
        valid: false,
        reason: 'Target polygons: Input Polygons accepts polygon; output Scaled has kind point.'
      });
      const result = addWorkflowStep(target, polygonRegistry, {
        operationId: 'fixture-scale',
        connectFrom: connection.from
      });
      expect(result.document.steps.at(-1)!.inputs).toEqual({});
      expectCode(
        () =>
          addWorkflowStep(target, polygonRegistry, {
            operationId: 'fixture-scale',
            connectFrom: connection.from,
            connectTo: 'polygons'
          }),
        'incompatible-port'
      );
      expectCode(
        () =>
          addWorkflowStep(target, polygonRegistry, {
            operationId: 'fixture-scale',
            connectFrom: {stepId: 'scale', output: 'missing'}
          }),
        'unknown-output-port'
      );
      expectCode(
        () =>
          addWorkflowStep(target, polygonRegistry, {
            operationId: 'fixture-scale',
            connectFrom: {stepId: 'missing', output: 'scaled'}
          }),
        'unknown-step'
      );
      expectCode(
        () =>
          addWorkflowStep(target, polygonRegistry, {
            operationId: 'fixture-scale',
            connectFrom: connection.from,
            connectTo: 'missing'
          }),
        'unknown-input-port'
      );
    });
  });

  test('sets arbitrary parameter names, stores defaults, removes bindings and detects equal list values', () => {
    const document = makeFixtureDocument();
    const result = edit(document, () =>
      setWorkflowParameter(document, 'scale', 'unknown', {from: 'static', value: [1, 2]})
    );
    expect(result.steps[1].parameters['unknown']).toEqual({from: 'static', value: [1, 2]});
    expect(result.steps[0]).toBe(document.steps[0]);
    expect(
      edit(result, () =>
        setWorkflowParameter(result, 'scale', 'unknown', {from: 'static', value: [1, 2]})
      )
    ).toBe(result);
    const removed = edit(result, () => setWorkflowParameter(result, 'scale', 'unknown', undefined));
    expect(removed.steps[1].parameters).toEqual(document.steps[1].parameters);
    expect(edit(removed, () => setWorkflowParameter(removed, 'scale', 'absent', undefined))).toBe(
      removed
    );
    expect(
      edit(document, () =>
        setWorkflowParameter(document, 'scale', 'factor', {from: 'static', value: 2})
      ).steps[1].parameters['factor']
    ).toEqual({from: 'static', value: 2});
  });

  test('stores every parameter name as an own binding for planner validation', () => {
    const document = makeFixtureDocument();
    const result = edit(document, () =>
      setWorkflowParameter(document, 'scale', '__proto__', {from: 'static', value: 7})
    );
    expect(Object.hasOwn(result.steps[1].parameters, '__proto__')).toBe(true);
    expect(result.steps[1].parameters['__proto__']).toEqual({from: 'static', value: 7});
    expect(Object.getPrototypeOf(result.steps[1].parameters)).toBe(Object.prototype);
    const removed = edit(result, () =>
      setWorkflowParameter(result, 'scale', '__proto__', undefined)
    );
    expect(removed.steps[1].parameters).toEqual(document.steps[1].parameters);
  });

  test('automatic connection skips incompatible ports and explicit connections choose the named port', () => {
    const operation = {
      ...registry.get('fixture-scale')!,
      inputs: [
        {name: 'polygons', label: 'Polygons', accepts: ['polygon' as const], maximumRows: 10},
        {name: 'first', label: 'First points', accepts: ['point' as const], maximumRows: 10},
        {name: 'second', label: 'Second points', accepts: ['point' as const], maximumRows: 10}
      ]
    };
    const multipleRegistry = {
      get: (identifier: string) =>
        identifier === 'fixture-scale' ? operation : registry.get(identifier),
      list: () => [operation]
    };
    const document = makeFixtureDocument();
    const from = {stepId: 'scale', output: 'scaled'};
    const automatic = edit(document, () =>
      addWorkflowStep(document, multipleRegistry, {operationId: 'fixture-scale', connectFrom: from})
    );
    expect(automatic.document.steps.at(-1)!.inputs).toEqual({first: from});
    const explicit = edit(document, () =>
      addWorkflowStep(document, multipleRegistry, {
        operationId: 'fixture-scale',
        connectFrom: from,
        connectTo: 'second',
        position: {x: 12.7, y: 33.2}
      })
    );
    expect(explicit.document.steps.at(-1)!.inputs).toEqual({second: from});
    expect(explicit.document.steps.at(-1)!.position).toEqual({x: 13, y: 33});
  });

  test('trims labels, deduplicates display, limits notes and updates flags without touching other steps', () => {
    const document = makeFixtureDocument();
    const result = edit(document, () =>
      updateWorkflowStep(document, 'scale', {
        label: '  Scaled points  ',
        display: ['scaled', 'scaled', 'other'],
        disabled: true,
        engine: 'cpu',
        onFallback: 'stop',
        note: 'n'.repeat(500)
      })
    );
    expect(result.steps[1]).toMatchObject({
      label: 'Scaled points',
      display: ['scaled', 'other'],
      disabled: true,
      engine: 'cpu',
      onFallback: 'stop',
      note: 'n'.repeat(500)
    });
    expect(result.steps[0]).toBe(document.steps[0]);
    expect(result.steps[2]).toBe(document.steps[2]);
    expect(
      edit(result, () =>
        updateWorkflowStep(result, 'scale', {
          label: ' Scaled points ',
          display: ['scaled', 'other']
        })
      )
    ).toBe(result);
    const cleared = edit(result, () =>
      updateWorkflowStep(result, 'scale', {label: '   ', display: [], note: undefined})
    );
    expect(Object.hasOwn(cleared.steps[1], 'label')).toBe(false);
    expect(Object.hasOwn(cleared.steps[1], 'display')).toBe(false);
    expect(Object.hasOwn(cleared.steps[1], 'note')).toBe(false);
    edit(document, () =>
      expectCode(
        () => updateWorkflowStep(document, 'scale', {note: 'n'.repeat(501)}),
        'limit-exceeded',
        'maximumNoteCharacters (500)'
      )
    );
  });

  test('rounds and clamps moves, shares stationary steps and rejects non-finite coordinates', () => {
    const document = makeFixtureDocument();
    const result = edit(document, () =>
      moveWorkflowSteps(document, {scale: {x: 100000.9, y: -100000.9}, source: {x: 0.1, y: 0.2}})
    );
    expect(result.steps[1].position).toEqual({x: 100000, y: -100000});
    expect(result.steps[0]).toBe(document.steps[0]);
    expect(result.steps[2]).toBe(document.steps[2]);
    expect(edit(document, () => moveWorkflowSteps(document, {}))).toBe(document);
    edit(document, () =>
      expectCode(
        () => moveWorkflowSteps(document, {scale: {x: NaN, y: 0}}),
        'parameter-out-of-range',
        'finite'
      )
    );
  });

  test('duplicates internal edges, retains external inputs and all configuration with unique ids', () => {
    const document = makeFixtureDocument();
    const configured = edit(document, () =>
      updateWorkflowStep(document, 'scale', {
        label: 'Scale',
        disabled: true,
        engine: 'cpu',
        onFallback: 'stop',
        note: 'Keep'
      })
    );
    const result = edit(configured, () =>
      duplicateWorkflowSteps(configured, ['scale', 'count', 'scale'])
    );
    expect(result.stepIds).toEqual(['scale-2', 'count-2']);
    configured.steps.forEach((original, index) =>
      expect(result.document.steps[index]).toBe(original)
    );
    expect(result.document.steps[3]).toEqual({
      ...configured.steps[1],
      id: 'scale-2',
      label: 'Scale copy',
      position: {x: 320, y: 40}
    });
    expect(result.document.steps[4]).toEqual({
      ...configured.steps[2],
      id: 'count-2',
      inputs: {rows: {stepId: 'scale-2', output: 'scaled'}},
      position: {x: 600, y: 40}
    });
    expect(edit(document, () => duplicateWorkflowSteps(document, [])).document).toBe(document);
    const full = {
      ...document,
      steps: Array.from({length: 64}, (_, index) => step(`step-${index}`))
    };
    edit(full, () => expectCode(() => duplicateWorkflowSteps(full, ['step-0']), 'limit-exceeded'));
  });

  test('step-specific edits reject unknown steps', () => {
    const document = makeFixtureDocument();
    edit(document, () => {
      for (const action of [
        () => disconnectWorkflowInput(document, 'missing', 'points'),
        () => setWorkflowParameter(document, 'missing', 'factor', undefined),
        () => updateWorkflowStep(document, 'missing', {}),
        () => moveWorkflowSteps(document, {missing: {x: 0, y: 0}}),
        () => duplicateWorkflowSteps(document, ['missing']),
        () => promoteWorkflowParameter(document, registry, 'missing', 'factor'),
        () => bindWorkflowParameter(document, registry, 'missing', 'factor', 'missing'),
        () => demoteWorkflowParameter(document, 'missing', 'factor')
      ])
        expectCode(action, 'unknown-step');
    });
  });
});

describe('workflow input edits', () => {
  test('promotes the current value and metadata, updates, binds another step, demotes and removes without losing values', () => {
    const document = makeFixtureDocument();
    const parameterRegistry = registryWithParameters([
      {
        name: 'factor',
        label: 'Factor',
        kind: 'number',
        minimum: 0,
        maximum: 100,
        unit: 'meters',
        defaultValue: 2
      }
    ]);
    const promoted = edit(document, () =>
      promoteWorkflowParameter(document, parameterRegistry, 'scale', 'factor', {
        label: 'Distance',
        inputId: 'distance'
      })
    );
    expect(promoted.inputId).toBe('distance');
    expect(promoted.document.inputs).toEqual([
      {
        id: 'distance',
        label: 'Distance',
        kind: 'number',
        minimum: 0,
        maximum: 100,
        unit: 'meters',
        value: 3
      }
    ]);
    expect(promoted.document.steps[1].parameters['factor']).toEqual({
      from: 'input',
      inputId: 'distance'
    });
    expect(promoted.document.steps[0]).toBe(document.steps[0]);
    expect(
      edit(promoted.document, () =>
        promoteWorkflowParameter(promoted.document, parameterRegistry, 'scale', 'factor')
      )
    ).toEqual({document: promoted.document, inputId: 'distance'});
    expect(
      promoteWorkflowParameter(promoted.document, parameterRegistry, 'scale', 'factor').document
    ).toBe(promoted.document);
    const changed = edit(promoted.document, () =>
      updateWorkflowInput(promoted.document, 'distance', {value: 10})
    );
    expect(changed.steps).toBe(promoted.document.steps);
    const duplicated = edit(changed, () => duplicateWorkflowSteps(changed, ['scale'])).document;
    const bound = edit(duplicated, () =>
      bindWorkflowParameter(duplicated, parameterRegistry, 'scale-2', 'factor', 'distance')
    );
    expect(bound).toBe(duplicated);
    const demoted = edit(bound, () => demoteWorkflowParameter(bound, 'scale', 'factor'));
    expect(demoted.steps[1].parameters['factor']).toEqual({from: 'static', value: 10});
    expect(demoted.inputs).toBe(bound.inputs);
    const removed = edit(demoted, () => removeWorkflowInput(demoted, 'distance'));
    expect(removed.inputs).toEqual([]);
    expect(removed.steps[3].parameters['factor']).toEqual({from: 'static', value: 10});
    expect(removed.steps[1]).toBe(demoted.steps[1]);
  });

  test('promotes defaults for all six eligible kinds and copies enum options', () => {
    const parameters: WorkflowParameterSpec[] = [
      {name: 'number', label: 'Number', kind: 'number', defaultValue: 2},
      {name: 'integer', label: 'Integer', kind: 'integer', defaultValue: 2},
      {name: 'boolean', label: 'Boolean', kind: 'boolean', defaultValue: false},
      {name: 'enum', label: 'Choice', kind: 'enum', defaultValue: 'a', options: ['a', 'b']},
      {name: 'string', label: 'String', kind: 'string', defaultValue: 'value'},
      {name: 'list', label: 'List', kind: 'number-list', defaultValue: [1, 2]}
    ];
    const parameterRegistry = registryWithParameters(parameters);
    const document = makeFixtureDocument();
    for (const parameter of parameters) {
      const result = edit(document, () =>
        promoteWorkflowParameter(document, parameterRegistry, 'scale', parameter.name)
      );
      expect(result.inputId).toBe(parameter.name);
      expect(result.document.inputs[0]).toEqual({
        id: parameter.name,
        label: parameter.label,
        kind: parameter.kind,
        value: parameter.defaultValue,
        ...(parameter.options ? {options: parameter.options} : {})
      });
    }
  });

  test('refuses unsupported promotion, unknown parameters and operations, duplicate ids and input limits', () => {
    const document = makeFixtureDocument();
    for (const kind of ['text', 'dataset', 'column', 'column-list', 'expression'] as const) {
      const parameterRegistry = registryWithParameters([
        {name: 'value', label: 'Value', kind, defaultValue: ''}
      ]);
      edit(document, () =>
        expectCode(
          () => promoteWorkflowParameter(document, parameterRegistry, 'scale', 'value'),
          'parameter-out-of-range',
          'cannot be promoted'
        )
      );
    }
    edit(document, () =>
      expectCode(
        () => promoteWorkflowParameter(document, registry, 'scale', 'missing'),
        'unknown-parameter'
      )
    );
    const unknown = {...document, steps: [{...step('unknown'), operationId: 'missing'}]};
    edit(unknown, () =>
      expectCode(
        () => promoteWorkflowParameter(unknown, registry, 'unknown', 'factor'),
        'unknown-operation'
      )
    );
    const full = {
      ...document,
      inputs: Array.from(
        {length: 16},
        (_, index): WorkflowInput => ({
          id: `input-${index}`,
          label: 'Input',
          kind: 'number',
          value: 1
        })
      )
    };
    edit(full, () =>
      expectCode(
        () => promoteWorkflowParameter(full, registry, 'scale', 'factor'),
        'limit-exceeded',
        'maximumInputs (16)'
      )
    );
    const existing = withInput({id: 'factor', label: 'Factor', kind: 'number', value: 1});
    edit(existing, () => {
      expectCode(
        () => promoteWorkflowParameter(existing, registry, 'scale', 'factor', {inputId: 'factor'}),
        'duplicate-id'
      );
      expectCode(
        () => promoteWorkflowParameter(existing, registry, 'scale', 'factor', {inputId: 'BAD'}),
        'parameter-out-of-range'
      );
      expect(promoteWorkflowParameter(existing, registry, 'scale', 'factor').inputId).toBe(
        'factor-2'
      );
    });
  });

  test('binds integer inputs to number parameters, refuses other kinds and checks every enum option', () => {
    const integerDocument = withInput({id: 'value', label: 'Value', kind: 'integer', value: 4});
    const bound = edit(integerDocument, () =>
      bindWorkflowParameter(integerDocument, registry, 'scale', 'factor', 'value')
    );
    expect(bound.steps[1].parameters['factor']).toEqual({from: 'input', inputId: 'value'});
    expect(bound.steps[0]).toBe(integerDocument.steps[0]);
    edit(integerDocument, () =>
      expectCode(
        () => bindWorkflowParameter(integerDocument, registry, 'scale', 'fail', 'value'),
        'parameter-out-of-range',
        'Fail requires kind boolean'
      )
    );
    const numberDocument = withInput({id: 'value', label: 'Value', kind: 'number', value: 4});
    edit(numberDocument, () =>
      expectCode(
        () => bindWorkflowParameter(numberDocument, registry, 'source', 'count', 'value'),
        'parameter-out-of-range'
      )
    );
    const parameterRegistry = registryWithParameters([
      {name: 'choice', label: 'Choice', kind: 'enum', defaultValue: 'a', options: ['a', 'b']}
    ]);
    const allowed = withInput({
      id: 'choice',
      label: 'Choice',
      kind: 'enum',
      value: 'a',
      options: ['a']
    });
    expect(
      edit(allowed, () =>
        bindWorkflowParameter(allowed, parameterRegistry, 'scale', 'choice', 'choice')
      ).steps[1].parameters['choice']
    ).toEqual({from: 'input', inputId: 'choice'});
    const refused = withInput({
      id: 'choice',
      label: 'Choice',
      kind: 'enum',
      value: 'a',
      options: ['a', 'c']
    });
    edit(refused, () =>
      expectCode(
        () => bindWorkflowParameter(refused, parameterRegistry, 'scale', 'choice', 'choice'),
        'parameter-out-of-range',
        'options a, b'
      )
    );
  });

  test('unknown input and parameter refusals and unbound demotion no-ops', () => {
    const document = makeFixtureDocument();
    edit(document, () => {
      expect(demoteWorkflowParameter(document, 'scale', 'factor')).toBe(document);
      expect(demoteWorkflowParameter(document, 'scale', 'absent')).toBe(document);
      expectCode(
        () => bindWorkflowParameter(document, registry, 'scale', 'missing', 'missing'),
        'unknown-parameter'
      );
      expectCode(
        () => bindWorkflowParameter(document, registry, 'scale', 'factor', 'missing'),
        'unknown-workflow-input'
      );
      expectCode(
        () => updateWorkflowInput(document, 'missing', {value: 1}),
        'unknown-workflow-input'
      );
      expectCode(() => removeWorkflowInput(document, 'missing'), 'unknown-workflow-input');
    });
    const dangling = edit(document, () =>
      setWorkflowParameter(document, 'scale', 'factor', {from: 'input', inputId: 'missing'})
    );
    edit(dangling, () => {
      expectCode(
        () => demoteWorkflowParameter(dangling, 'scale', 'factor'),
        'unknown-workflow-input'
      );
      expectCode(
        () => promoteWorkflowParameter(dangling, registry, 'scale', 'factor'),
        'unknown-workflow-input'
      );
    });
  });

  test.each([
    {
      kind: 'number' as const,
      value: 2,
      valid: 0,
      invalid: [-1, 11, NaN, Infinity, '2'],
      fragment: 'finite number'
    },
    {
      kind: 'integer' as const,
      value: 2,
      valid: 10,
      invalid: [1.5, -1, 11, '2'],
      fragment: 'whole number'
    },
    {
      kind: 'boolean' as const,
      value: false,
      valid: true,
      invalid: ['false', 0],
      fragment: 'boolean'
    },
    {kind: 'enum' as const, value: 'a', valid: 'b', invalid: ['c', false], fragment: 'one of a, b'},
    {kind: 'string' as const, value: 'old', valid: 'new', invalid: [2, false], fragment: 'string'},
    {
      kind: 'number-list' as const,
      value: [1, 2],
      valid: [0, 10],
      invalid: [[], [NaN], [-1], [11], '1,2'],
      fragment: 'numbers'
    }
  ])('validates $kind workflow input updates without coercing types', ({
    kind,
    value,
    valid,
    invalid
  }) => {
    const input: WorkflowInput = {
      id: 'value',
      label: 'Input value',
      kind,
      value,
      ...(kind === 'enum' ? {options: ['a', 'b']} : {}),
      ...(['number', 'integer', 'number-list'].includes(kind) ? {minimum: 0, maximum: 10} : {})
    };
    const document = withInput(input);
    expect(edit(document, () => updateWorkflowInput(document, 'value', {value}))).toBe(document);
    const updated = edit(document, () => updateWorkflowInput(document, 'value', {value: valid}));
    expect(updated.inputs[0].value).toEqual(valid);
    expect(updated.steps).toBe(document.steps);
    for (const invalidValue of invalid)
      edit(document, () =>
        expectCode(
          () => updateWorkflowInput(document, 'value', {value: invalidValue}),
          'parameter-out-of-range',
          'Input value: Value'
        )
      );
  });

  test('updates metadata, clears optional fields, checks new ranges and preserves other inputs', () => {
    const document = {
      ...withInput({
        id: 'value',
        label: 'Value',
        kind: 'number',
        value: 2,
        minimum: 0,
        maximum: 10,
        unit: 'meters'
      }),
      inputs: [
        {
          id: 'value',
          label: 'Value',
          kind: 'number' as const,
          value: 2,
          minimum: 0,
          maximum: 10,
          unit: 'meters'
        },
        {id: 'other', label: 'Other', kind: 'boolean' as const, value: false}
      ]
    };
    const updated = edit(document, () =>
      updateWorkflowInput(document, 'value', {
        label: 'Renamed',
        unit: undefined,
        minimum: undefined
      })
    );
    expect(updated.inputs[0]).toEqual({
      id: 'value',
      label: 'Renamed',
      kind: 'number',
      value: 2,
      maximum: 10
    });
    expect(updated.inputs[1]).toBe(document.inputs[1]);
    edit(document, () => {
      expectCode(
        () => updateWorkflowInput(document, 'value', {minimum: 11}),
        'parameter-out-of-range',
        'minimum must be at most maximum'
      );
      expectCode(
        () => updateWorkflowInput(document, 'value', {maximum: 1}),
        'parameter-out-of-range',
        'at most 1'
      );
      expectCode(
        () => updateWorkflowInput(document, 'value', {minimum: Infinity}),
        'parameter-out-of-range',
        'minimum must be finite'
      );
    });
  });
});

describe('workflow metadata and layout', () => {
  test('settings and title edits share steps and detect no-ops; titles have exact limits', () => {
    const document = makeFixtureDocument();
    expect(edit(document, () => updateWorkflowSettings(document, {autoRun: false}))).toBe(document);
    const settings = edit(document, () =>
      updateWorkflowSettings(document, {engine: 'cpu', autoRun: true})
    );
    expect(settings.settings).toEqual({engine: 'cpu', autoRun: true});
    expect(settings.steps).toBe(document.steps);
    expect(edit(document, () => renameWorkflow(document, '  Fixture workflow  '))).toBe(document);
    const renamed = edit(document, () => renameWorkflow(document, ' New title ', 'Description'));
    expect(renamed.title).toBe('New title');
    expect(renamed.description).toBe('Description');
    expect(renamed.steps).toBe(document.steps);
    const removed = edit(renamed, () => renameWorkflow(renamed, 'New title'));
    expect(Object.hasOwn(removed, 'description')).toBe(false);
    expect(edit(document, () => renameWorkflow(document, 'x'.repeat(80))).title).toHaveLength(80);
    edit(document, () => {
      expectCode(
        () => renameWorkflow(document, '  '),
        'limit-exceeded',
        '1 to maximumTitleCharacters (80)'
      );
      expectCode(() => renameWorkflow(document, 'x'.repeat(81)), 'limit-exceeded');
    });
  });

  test('layout uses longest paths and a previous-column barycentre, independent of original positions or array order', () => {
    const document = {
      ...makeFixtureDocument(),
      steps: [
        step('b'),
        step('a'),
        step('c', 'b'),
        step('d', 'a'),
        {
          ...step('e', 'c'),
          inputs: {points: {stepId: 'c', output: 'scaled'}, other: {stepId: 'a', output: 'scaled'}}
        }
      ]
    };
    const result = edit(document, () => layoutWorkflow(document));
    expect(
      Object.fromEntries(result.steps.map(candidate => [candidate.id, candidate.position]))
    ).toEqual({
      a: {x: 0, y: 0},
      b: {x: 0, y: 150},
      d: {x: 280, y: 0},
      c: {x: 280, y: 150},
      e: {x: 560, y: 0}
    });
    expect(edit(result, () => layoutWorkflow(result))).toBe(result);
    const reordered = {
      ...document,
      steps: [...document.steps]
        .reverse()
        .map(candidate => ({...candidate, position: {x: 930, y: -13}}))
    };
    const again = edit(reordered, () => layoutWorkflow(reordered));
    const byId = (current: WorkflowDocument) =>
      [...current.steps]
        .sort((left, right) => left.id.localeCompare(right.id))
        .map(candidate => ({id: candidate.id, position: candidate.position}));
    expect(byId(again)).toEqual(byId(result));
    const spaced = edit(document, () =>
      layoutWorkflow(document, {columnSpacing: 100, rowSpacing: 50, origin: {x: 10, y: 20}})
    );
    expect(spaced.steps.find(candidate => candidate.id === 'c')!.position).toEqual({x: 110, y: 70});
    expect(spaced.steps.find(candidate => candidate.id === 'e')!.position).toEqual({x: 210, y: 20});
  });

  test('layout puts cyclic, dangling and blocked descendants in the final column deterministically', () => {
    const document = {
      ...makeFixtureDocument(),
      steps: [
        step('a'),
        step('b', 'a'),
        step('cycle-a', 'cycle-b'),
        step('cycle-b', 'cycle-a'),
        step('dangling', 'missing'),
        step('blocked', 'dangling')
      ]
    };
    const result = edit(document, () => layoutWorkflow(document));
    expect(result.steps.map(candidate => [candidate.id, candidate.position])).toEqual([
      ['a', {x: 0, y: 0}],
      ['b', {x: 280, y: 0}],
      ['cycle-a', {x: 560, y: 150}],
      ['cycle-b', {x: 560, y: 300}],
      ['dangling', {x: 560, y: 450}],
      ['blocked', {x: 560, y: 0}]
    ]);
    expect(edit(result, () => layoutWorkflow(result))).toBe(result);
    const empty = {...makeFixtureDocument(), steps: []};
    expect(edit(empty, () => layoutWorkflow(empty))).toBe(empty);
    const onlyCycle = {...empty, steps: [step('a', 'b'), step('b', 'a')]};
    expect(
      edit(onlyCycle, () => layoutWorkflow(onlyCycle)).steps.map(candidate => candidate.position)
    ).toEqual([
      {x: 0, y: 0},
      {x: 0, y: 150}
    ]);
  });
});

describe('review regressions: owned edit values (F1)', () => {
  test('copies reused bindings so history retains both intended values', () => {
    const document = makeFixtureDocument();
    const history = new WorkflowHistory(document);
    const binding = {from: 'static' as const, value: 4};
    const first = edit(document, () => setWorkflowParameter(document, 'scale', 'factor', binding));
    history.apply(first);
    binding.value = 9;
    expect(first.steps[1].parameters['factor']).toEqual({from: 'static', value: 4});
    const second = edit(first, () => setWorkflowParameter(first, 'scale', 'factor', binding));
    expect(second).not.toBe(first);
    history.apply(second);
    expect(history.undo()).toBe(true);
    expect(history.document).toBe(first);
    expect(history.document.steps[1].parameters['factor']).toEqual({from: 'static', value: 4});
  });

  test('copies static array values and workflow input binding objects', () => {
    const document = makeFixtureDocument();
    const values = [1, 2];
    const binding = {from: 'static' as const, value: values};
    const updated = edit(document, () =>
      setWorkflowParameter(document, 'scale', 'factor', binding)
    );
    values[0] = Infinity;
    binding.value = [99];
    expect(updated.steps[1].parameters['factor']).toEqual({from: 'static', value: [1, 2]});
    const inputBinding = {from: 'input' as const, inputId: 'original'};
    const bound = edit(document, () =>
      setWorkflowParameter(document, 'scale', 'factor', inputBinding)
    );
    inputBinding.inputId = 'changed';
    expect(bound.steps[1].parameters['factor']).toEqual({from: 'input', inputId: 'original'});
  });

  test('copies arrays supplied when adding a step', () => {
    const document = makeFixtureDocument();
    const values = [1, 2];
    const parameterRegistry = registryWithParameters([
      {name: 'values', label: 'Values', kind: 'number-list', defaultValue: []}
    ]);
    const result = edit(document, () =>
      addWorkflowStep(document, parameterRegistry, {
        operationId: 'fixture-scale',
        parameters: {values}
      })
    );
    values[0] = Infinity;
    expect(result.document.steps.at(-1)!.parameters['values']).toEqual({
      from: 'static',
      value: [1, 2]
    });
    expect(() => serializeWorkflowDocument(result.document)).not.toThrow();
  });

  test('copies validated input values and enum option arrays', () => {
    const document = withInput({
      id: 'values',
      label: 'Values',
      kind: 'number-list',
      value: [1],
      minimum: 0,
      maximum: 10
    });
    const values = [2, 3];
    const updated = edit(document, () => updateWorkflowInput(document, 'values', {value: values}));
    values[0] = Infinity;
    expect(updated.inputs[0].value).toEqual([2, 3]);
    expect(() => serializeWorkflowDocument(updated)).not.toThrow();
    const choiceDocument = withInput({
      id: 'choice',
      label: 'Choice',
      kind: 'enum',
      value: 'a',
      options: ['a']
    });
    const options = ['a', 'b'];
    const choice = edit(choiceDocument, () =>
      updateWorkflowInput(choiceDocument, 'choice', {options})
    );
    options[0] = 'removed';
    expect(choice.inputs[0].options).toEqual(['a', 'b']);
    expect(() => serializeWorkflowDocument(choice)).not.toThrow();
  });

  test('copies promoted registry list defaults and enum options', () => {
    const document = makeFixtureDocument();
    const values = [1, 2];
    const options = ['a', 'b'];
    const parameterRegistry = registryWithParameters([
      {name: 'values', label: 'Values', kind: 'number-list', defaultValue: values},
      {name: 'choice', label: 'Choice', kind: 'enum', defaultValue: 'a', options}
    ]);
    const promoted = edit(document, () =>
      promoteWorkflowParameter(document, parameterRegistry, 'scale', 'values')
    );
    const choice = edit(document, () =>
      promoteWorkflowParameter(document, parameterRegistry, 'scale', 'choice')
    );
    values[0] = Infinity;
    options[0] = 'removed';
    expect(promoted.document.inputs[0].value).toEqual([1, 2]);
    expect(choice.document.inputs[0].options).toEqual(['a', 'b']);
    expect(() => serializeWorkflowDocument(promoted.document)).not.toThrow();
    expect(() => serializeWorkflowDocument(choice.document)).not.toThrow();
  });

  test('isolates caller display, connection and position objects while sharing untouched steps', () => {
    const document = makeFixtureDocument();
    const display = ['scaled'];
    const changed = edit(document, () => updateWorkflowStep(document, 'source', {display}));
    display.push('other');
    expect(changed.steps[0].display).toEqual(['scaled']);
    expect(changed.steps[1]).toBe(document.steps[1]);
    const from = {stepId: 'source', output: 'data'};
    const connected = edit(document, () =>
      connectWorkflowSteps(document, registry, {from, to: {stepId: 'count', input: 'rows'}})
    );
    from.output = 'changed';
    expect(connected.steps[2].inputs['rows']).toEqual({stepId: 'source', output: 'data'});
    const position = {x: 12, y: 34};
    const moved = edit(document, () => moveWorkflowSteps(document, {source: position}));
    position.x = 999;
    expect(moved.steps[0].position).toEqual({x: 12, y: 34});
  });
});

describe('review regressions: persistence limits (F2)', () => {
  test('accepts and saves a 500-character description and refuses 501 characters', () => {
    const document = makeFixtureDocument();
    const updated = edit(document, () => renameWorkflow(document, 'Renamed', 'x'.repeat(500)));
    expect(serializeWorkflowDocument(updated)).toMatchObject({description: 'x'.repeat(500)});
    edit(document, () =>
      expectCode(
        () => renameWorkflow(document, 'Renamed', 'x'.repeat(501)),
        'limit-exceeded',
        'maximumNoteCharacters (500)'
      )
    );
  });

  test('accepts and saves exact input text limits and refuses one character more', () => {
    const document = withInput({id: 'text', label: 'Text', kind: 'string', value: ''});
    const updated = edit(document, () =>
      updateWorkflowInput(document, 'text', {
        value: 'x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters)
      })
    );
    expect(() => serializeWorkflowDocument(updated)).not.toThrow();
    edit(document, () =>
      expectCode(
        () =>
          updateWorkflowInput(document, 'text', {
            value: 'x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters + 1)
          }),
        'limit-exceeded',
        'Text:'
      )
    );
  });

  test('enforces text limits for added and edited static bindings and promoted defaults without checking operation ranges', () => {
    const document = makeFixtureDocument();
    const text = 'x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters + 1);
    edit(document, () => {
      expectCode(
        () =>
          addWorkflowStep(document, registry, {
            operationId: 'fixture-scale',
            parameters: {factor: text}
          }),
        'limit-exceeded',
        'maximumTextCharacters'
      );
      expectCode(
        () => setWorkflowParameter(document, 'scale', 'factor', {from: 'static', value: text}),
        'limit-exceeded',
        'maximumTextCharacters'
      );
      const parameterRegistry = registryWithParameters([
        {name: 'text', label: 'Text', kind: 'string', defaultValue: text}
      ]);
      expectCode(
        () => promoteWorkflowParameter(document, parameterRegistry, 'scale', 'text'),
        'limit-exceeded',
        'maximumTextCharacters'
      );
      expect(
        serializeWorkflowDocument(
          setWorkflowParameter(document, 'scale', 'factor', {from: 'static', value: -999})
        )
      ).toBeDefined();
    });
  });

  test('refuses cumulative document growth even when every input string is individually permitted', () => {
    let document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: Array.from({length: 5}, (_, index) => ({
        id: `text-${index}`,
        label: `Text ${index}`,
        kind: 'string',
        value: ''
      }))
    };
    for (let index = 0; index < 3; index++) {
      document = edit(document, () =>
        updateWorkflowInput(document, `text-${index}`, {
          value: 'x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters)
        })
      );
      expect(() => serializeWorkflowDocument(document)).not.toThrow();
    }
    edit(document, () =>
      expectCode(
        () =>
          updateWorkflowInput(document, 'text-3', {
            value: 'x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters)
          }),
        'limit-exceeded',
        'maximumDocumentCharacters (1048576)'
      )
    );
  });

  test('measures exact persistence limits after empty optional fields are omitted', () => {
    const initial = makeFixtureDocument();
    const document: WorkflowDocument = {
      ...initial,
      description: '',
      steps: initial.steps.map(candidate => ({
        ...candidate,
        disabled: false,
        note: '',
        ...(candidate.id === 'source' ? {display: []} : {})
      })),
      inputs: Array.from({length: 4}, (_, index) => ({
        id: `text-${index}`,
        label: `Text ${index}`,
        kind: 'string',
        value: index < 3 ? 'x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters) : ''
      }))
    };
    const available =
      WORKFLOW_LIMITS.maximumDocumentCharacters -
      JSON.stringify(serializeWorkflowDocument(document)).length;
    const updated = edit(document, () =>
      updateWorkflowInput(document, 'text-3', {value: 'x'.repeat(available)})
    );
    expect(JSON.stringify(serializeWorkflowDocument(updated))).toHaveLength(
      WORKFLOW_LIMITS.maximumDocumentCharacters
    );
    edit(updated, () =>
      expectCode(
        () => updateWorkflowInput(updated, 'text-3', {value: 'x'.repeat(available + 1)}),
        'limit-exceeded',
        'maximumDocumentCharacters'
      )
    );
  });

  test('accepts exactly maximumDocumentCharacters and refuses an extra serialized character', () => {
    let document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: Array.from({length: 4}, (_, index) => ({
        id: `text-${index}`,
        label: `Text ${index}`,
        kind: 'string',
        value: index < 3 ? 'x'.repeat(WORKFLOW_LIMITS.maximumTextCharacters) : ''
      }))
    };
    const available =
      WORKFLOW_LIMITS.maximumDocumentCharacters -
      JSON.stringify(serializeWorkflowDocument(document)).length;
    expect(available).toBeGreaterThan(0);
    expect(available).toBeLessThan(WORKFLOW_LIMITS.maximumTextCharacters);
    document = edit(document, () =>
      updateWorkflowInput(document, 'text-3', {value: 'x'.repeat(available)})
    );
    expect(JSON.stringify(serializeWorkflowDocument(document))).toHaveLength(
      WORKFLOW_LIMITS.maximumDocumentCharacters
    );
    edit(document, () =>
      expectCode(
        () => updateWorkflowInput(document, 'text-3', {value: 'x'.repeat(available + 1)}),
        'limit-exceeded',
        'maximumDocumentCharacters'
      )
    );
    edit(document, () =>
      expectCode(
        () => updateWorkflowInput(document, 'text-3', {value: '\\'.repeat(available)}),
        'limit-exceeded',
        'maximumDocumentCharacters'
      )
    );
    const changes: Array<() => unknown> = [
      () => addWorkflowStep(document, registry, {operationId: 'fixture-source'}),
      () => setWorkflowParameter(document, 'source', 'count', {from: 'static', value: 5}),
      () => updateWorkflowStep(document, 'scale', {note: 'Note'}),
      () => duplicateWorkflowSteps(document, ['scale']),
      () => promoteWorkflowParameter(document, registry, 'scale', 'fail'),
      () => renameWorkflow(document, 'A longer workflow title'),
      () => moveWorkflowSteps(document, {source: {x: 10000, y: 0}})
    ];
    for (const change of changes)
      edit(document, () => expectCode(change, 'limit-exceeded', 'maximumDocumentCharacters'));
  });
});

describe('review regressions: canvas boundaries (F3)', () => {
  test.each([
    0, 100000, -100000
  ])('finds a legal free row after coordinate clamping at y=%s', vertical => {
    const document = {
      ...makeFixtureDocument(),
      steps: [
        {...makeFixtureDocument().steps[0], position: {x: 99900, y: vertical}},
        {...step('occupied'), position: {x: 100000, y: vertical}}
      ]
    };
    const result = edit(document, () =>
      addWorkflowStep(document, registry, {
        operationId: 'fixture-scale',
        connectFrom: {stepId: 'source', output: 'data'}
      })
    );
    const position = result.document.steps.at(-1)!.position;
    expect(position).toEqual({x: 100000, y: vertical === 100000 ? 99850 : vertical + 150});
    for (const existing of document.steps)
      expect(
        Math.abs(existing.position.x - position.x) >= 40 ||
          Math.abs(existing.position.y - position.y) >= 40
      ).toBe(true);
    expect(() => serializeWorkflowDocument(result.document)).not.toThrow();
  });
});

describe('workflow composition', () => {
  test('sets, copies and clears display styles while preserving no-op and untouched identities', () => {
    const document = makeFixtureDocument();
    const style = {
      colorColumn: 'value',
      colorScale: 'quantile' as const,
      palette: 'diverging' as const
    };
    const changed = edit(document, () =>
      setWorkflowDisplayStyle(document, 'scale', 'scaled', style)
    );
    style.colorColumn = 'changed';
    expect(changed.steps[1].displayStyle).toEqual({
      scaled: {colorColumn: 'value', colorScale: 'quantile', palette: 'diverging'}
    });
    expect(changed.steps[0]).toBe(document.steps[0]);
    expect(
      edit(changed, () =>
        setWorkflowDisplayStyle(changed, 'scale', 'scaled', {
          colorColumn: 'value',
          colorScale: 'quantile',
          palette: 'diverging'
        })
      )
    ).toBe(changed);
    const another = edit(changed, () =>
      setWorkflowDisplayStyle(changed, 'scale', 'other', {palette: 'categorical'})
    );
    const cleared = edit(another, () => setWorkflowDisplayStyle(another, 'scale', 'scaled', {}));
    expect(cleared.steps[1].displayStyle).toEqual({other: {palette: 'categorical'}});
    const empty = edit(cleared, () =>
      setWorkflowDisplayStyle(cleared, 'scale', 'other', undefined)
    );
    expect(Object.hasOwn(empty.steps[1], 'displayStyle')).toBe(false);
    expect(edit(empty, () => setWorkflowDisplayStyle(empty, 'scale', 'other', {}))).toBe(empty);
    edit(document, () =>
      expectCode(() => setWorkflowDisplayStyle(document, 'missing', 'data', {}), 'unknown-step')
    );
    const duplicated = edit(changed, () => duplicateWorkflowSteps(changed, ['scale']));
    expect(duplicated.document.steps.at(-1)!.displayStyle).toEqual(changed.steps[1].displayStyle);
  });

  test('copies only selected edges and bound inputs, clears display and owns the copied values', () => {
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [
        {id: 'values', label: 'Values', kind: 'number-list', value: [1, 2]},
        {id: 'unused', label: 'Unused', kind: 'boolean', value: false}
      ]
    };
    document.steps[1].parameters = {factor: {from: 'input', inputId: 'values'}};
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value'}};
    const clipboard = edit(document, () =>
      copyWorkflowSteps(document, ['scale', 'count', 'scale'])
    );
    expect(clipboard.steps.map(candidate => candidate.id)).toEqual(['scale', 'count']);
    expect(clipboard.steps[0].inputs).toEqual({});
    expect(clipboard.steps[1].inputs).toEqual({rows: {stepId: 'scale', output: 'scaled'}});
    expect(clipboard.steps[0].display).toBeUndefined();
    expect(clipboard.steps[0].displayStyle).toEqual({scaled: {colorColumn: 'value'}});
    expect(clipboard.inputs.map(input => input.id)).toEqual(['values']);
    clipboard.steps[0].displayStyle!['scaled'].colorColumn = 'changed';
    (clipboard.inputs[0].value as number[])[0] = 99;
    expect(document.steps[1].displayStyle).toEqual({scaled: {colorColumn: 'value'}});
    expect(document.inputs[0].value).toEqual([1, 2]);
    edit(document, () =>
      expectCode(() => copyWorkflowSteps(document, ['missing']), 'unknown-step')
    );
  });

  test('strictly parses clipboard objects and JSON through the document parser, retaining unknown operations', () => {
    const clipboard = copyWorkflowSteps(makeFixtureDocument(), ['scale', 'count']);
    clipboard.steps[0].operationId = 'operation-no-longer-installed';
    clipboard.steps[0].displayStyle = {scaled: {colorColumn: 'value', palette: 'diverging'}};
    expect(parseWorkflowClipboard(JSON.stringify(clipboard))).toEqual(clipboard);
    expect(parseWorkflowClipboard(clipboard)).toEqual(clipboard);
    expectCode(
      () =>
        parseWorkflowClipboard({
          ...clipboard,
          steps: [{...clipboard.steps[0], displayStyle: {scaled: {palette: 'invalid'}}}]
        }),
      'malformed'
    );
    expectCode(() => parseWorkflowClipboard({...clipboard, schemaVersion: 2}), 'malformed');
    expectCode(() => parseWorkflowClipboard({...clipboard, extra: true}), 'malformed');
    expectCode(
      () => parseWorkflowClipboard({...clipboard, steps: [clipboard.steps[0], clipboard.steps[0]]}),
      'duplicate-id'
    );
    expectCode(
      () =>
        parseWorkflowClipboard({
          ...clipboard,
          steps: [{...clipboard.steps[0], position: {x: Infinity, y: 0}}]
        }),
      'malformed'
    );
    expectCode(
      () =>
        parseWorkflowClipboard({
          ...clipboard,
          inputs: [{id: 'invalid', label: 'Invalid', kind: 'integer', value: 1.5}]
        }),
      'malformed'
    );
    expectCode(() => parseWorkflowClipboard('{'), 'malformed');
    expectCode(() => parseWorkflowClipboard(Object.create(clipboard)), 'malformed');
    let readAccessor = false;
    expectCode(
      () =>
        parseWorkflowClipboard({
          schemaVersion: 1,
          inputs: [],
          get steps() {
            readAccessor = true;
            return [];
          }
        }),
      'malformed'
    );
    expect(readAccessor).toBe(false);
  });

  test('pastes beside the originals, remaps internal edges and reuses a matching input with its destination value', () => {
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [
        {
          id: 'factor',
          label: 'Current factor',
          kind: 'number',
          unit: 'meters',
          minimum: 0,
          maximum: 100,
          value: 9
        }
      ]
    };
    document.steps[1].parameters = {factor: {from: 'input', inputId: 'factor'}};
    document.steps[1].displayStyle = {scaled: {colorColumn: 'value'}};
    const clipboard = copyWorkflowSteps(document, ['scale', 'count']);
    clipboard.inputs[0].value = 4;
    clipboard.inputs[0].label = 'Old label';
    const result = edit(document, () => pasteWorkflowSteps(document, registry, clipboard));
    expect(result.stepIds).toEqual(['scale-2', 'count-2']);
    expect(result.document.inputs).toBe(document.inputs);
    expect(result.document.inputs[0].value).toBe(9);
    expect(result.document.steps[3].position).toEqual({x: 320, y: 40});
    expect(result.document.steps[4].position).toEqual({x: 600, y: 40});
    expect(result.document.steps[4].inputs).toEqual({rows: {stepId: 'scale-2', output: 'scaled'}});
    expect(result.document.steps[3].parameters).toEqual({
      factor: {from: 'input', inputId: 'factor'}
    });
    expect(result.document.steps[3].display).toBeUndefined();
    expect(result.document.steps[3].displayStyle).toEqual({scaled: {colorColumn: 'value'}});
    document.steps.forEach((original, index) =>
      expect(result.document.steps[index]).toBe(original)
    );
    clipboard.steps[0].displayStyle!['scaled'].colorColumn = 'changed';
    expect(result.document.steps[3].displayStyle!['scaled'].colorColumn).toBe('value');
  });

  test('pastes another document under free input ids when any constraint differs', () => {
    const source: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [
        {
          id: 'choice',
          label: 'Choice',
          kind: 'enum',
          unit: 'category',
          minimum: 0,
          maximum: 10,
          options: ['a', 'b'],
          value: 'a'
        }
      ]
    };
    source.steps[1].operationId = 'unknown-placeholder';
    source.steps[1].parameters = {choice: {from: 'input', inputId: 'choice'}};
    const clipboard = copyWorkflowSteps(source, ['scale', 'count']);
    const differences: Array<Partial<WorkflowInput>> = [
      {kind: 'string'},
      {unit: 'different'},
      {minimum: 1},
      {maximum: 11},
      {options: ['a', 'c']}
    ];
    for (const difference of differences) {
      const document: WorkflowDocument = {
        ...makeFixtureDocument(),
        steps: [step('destination')],
        inputs: [{...source.inputs[0], ...difference}]
      };
      const result = edit(document, () => pasteWorkflowSteps(document, registry, clipboard));
      expect(result.stepIds).toEqual(['scale', 'count']);
      expect(result.document.steps[1].operationId).toBe('unknown-placeholder');
      expect(result.document.steps[1].position).toEqual({x: 280, y: 0});
      expect(result.document.steps[2].position).toEqual({x: 560, y: 0});
      expect(result.document.inputs[1]).toEqual({...source.inputs[0], id: 'choice-2'});
      expect(result.document.steps[1].parameters).toEqual({
        choice: {from: 'input', inputId: 'choice-2'}
      });
    }
  });

  test('inserts every source input and retains display flags and styles, with explicit or default block placement', () => {
    const source = makeFixtureDocument();
    source.inputs.push({id: 'unbound', label: 'Unbound', kind: 'boolean', value: true});
    source.steps[1].displayStyle = {scaled: {palette: 'diverging'}};
    const document = {
      ...makeFixtureDocument(),
      steps: [{...step('existing'), position: {x: 100, y: 300}}]
    };
    const result = edit(document, () =>
      edit(source, () => insertWorkflowSteps(document, registry, source))
    );
    expect(result.stepIds).toEqual(['source', 'scale', 'count']);
    expect(result.document.steps.slice(1).map(candidate => candidate.position)).toEqual([
      {x: 100, y: 450},
      {x: 380, y: 450},
      {x: 660, y: 450}
    ]);
    expect(result.document.steps[2].display).toEqual(['scaled']);
    expect(result.document.steps[2].displayStyle).toEqual({scaled: {palette: 'diverging'}});
    expect(result.document.inputs).toEqual(source.inputs);
    const positioned = edit(document, () =>
      insertWorkflowSteps(document, registry, source, {x: 10.4, y: -20.6})
    );
    expect(positioned.document.steps.slice(1).map(candidate => candidate.position)).toEqual([
      {x: 10, y: -21},
      {x: 290, y: -21},
      {x: 570, y: -21}
    ]);
  });

  test('preflights both capacity totals atomically and accepts exact limits', () => {
    const source: WorkflowDocument = {
      ...makeFixtureDocument(),
      steps: [step('incoming')],
      inputs: [{id: 'incoming', label: 'Incoming', kind: 'boolean', value: true}]
    };
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      steps: Array.from({length: 64}, (_, index) => step(`existing-${index}`)),
      inputs: Array.from({length: 16}, (_, index) => ({
        id: `input-${index}`,
        label: 'Input',
        kind: 'boolean',
        value: false
      }))
    };
    edit(document, () =>
      edit(source, () => {
        const action = () => insertWorkflowSteps(document, registry, source);
        expectCode(action, 'limit-exceeded', 'maximumSteps (64)');
        expectCode(action, 'limit-exceeded', 'total 65 (64 existing + 1 added)');
        expectCode(action, 'limit-exceeded', 'maximumInputs (16)');
        expectCode(action, 'limit-exceeded', 'total 17 (16 existing + 1 added)');
      })
    );
    const available = {
      ...document,
      steps: document.steps.slice(1),
      inputs: document.inputs.slice(1)
    };
    const result = edit(available, () => insertWorkflowSteps(available, registry, source));
    expect(result.document.steps).toHaveLength(64);
    expect(result.document.inputs).toHaveLength(16);
  });

  test('keeps block spacing at canvas boundaries, handles empty copies and rejects missing source inputs', () => {
    const document = makeFixtureDocument();
    const clipboard = copyWorkflowSteps(document, ['scale', 'count']);
    const placed = edit(document, () =>
      pasteWorkflowSteps(document, registry, clipboard, {x: 100000, y: 100000})
    );
    expect(placed.document.steps.slice(3).map(candidate => candidate.position)).toEqual([
      {x: 99720, y: 100000},
      {x: 100000, y: 100000}
    ]);
    expect(
      edit(document, () =>
        pasteWorkflowSteps(document, registry, {schemaVersion: 1, steps: [], inputs: []})
      ).document
    ).toBe(document);
    expect(
      edit(document, () =>
        insertWorkflowSteps(document, registry, {...document, steps: [], inputs: []})
      ).document
    ).toBe(document);
    clipboard.steps[0].parameters = {factor: {from: 'input', inputId: 'missing'}};
    edit(document, () =>
      expectCode(
        () => pasteWorkflowSteps(document, registry, clipboard),
        'unknown-workflow-input',
        'missing workflow input missing'
      )
    );
  });
});

describe('replaceWorkflowStepOperation', () => {
  function extendFixtureRegistry(...operations: WorkflowOperation[]) {
    return {
      get: (operationId: string) =>
        operations.find(operation => operation.id === operationId) ?? registry.get(operationId),
      list: () => [...operations, ...registry.list()]
    };
  }

  const geojson: WorkflowOperation = {
    ...registry.get('fixture-source')!,
    id: 'source-geojson',
    title: 'Embedded GeoJSON',
    parameters: [
      {name: 'shared', label: 'Shared', kind: 'string', defaultValue: ''},
      {name: 'geojson', label: 'GeoJSON', kind: 'text', defaultValue: ''}
    ]
  };
  const dataset: WorkflowOperation = {
    ...geojson,
    id: 'source-dataset',
    title: 'Dataset',
    parameters: [
      {name: 'shared', label: 'Shared', kind: 'string', defaultValue: ''},
      {name: 'datasetId', label: 'Dataset', kind: 'dataset', defaultValue: ''}
    ]
  };
  const sourceRegistry = extendFixtureRegistry(geojson, dataset);

  test('swaps embedded GeoJSON for a dataset without rewiring consumers or changing step metadata', () => {
    const document = makeFixtureDocument();
    document.steps[0] = {
      ...document.steps[0],
      operationId: 'source-geojson',
      label: 'Source points',
      note: 'Keep me',
      disabled: true,
      engine: 'cpu',
      onFallback: 'stop',
      parameters: {shared: {from: 'static', value: 'kept'}, geojson: {from: 'static', value: '{}'}},
      display: ['data', 'obsolete'],
      displayStyle: {data: {colorColumn: 'value'}, obsolete: {palette: 'diverging'}}
    };
    const result = edit(document, () =>
      replaceWorkflowStepOperation(document, sourceRegistry, 'source', 'source-dataset', {
        datasetId: 'chosen-dataset'
      })
    );
    expect(result.steps[0]).toEqual({
      ...document.steps[0],
      operationId: 'source-dataset',
      parameters: {
        shared: {from: 'static', value: 'kept'},
        datasetId: {from: 'static', value: 'chosen-dataset'}
      },
      display: ['data'],
      displayStyle: {data: {colorColumn: 'value'}}
    });
    expect(result.steps[0].position).toBe(document.steps[0].position);
    expect(result.steps[1]).toBe(document.steps[1]);
    expect(result.steps[1].inputs).toEqual({points: {stepId: 'source', output: 'data'}});
    expect(result.steps[2]).toBe(document.steps[2]);
    expect(result.inputs).toBe(document.inputs);
    expect(() => serializeWorkflowDocument(result)).not.toThrow();
  });

  test('disconnects consumers and clears display entries only when their output disappears', () => {
    const document = makeFixtureDocument();
    document.steps[0] = {
      ...document.steps[0],
      operationId: 'source-geojson',
      display: ['data'],
      displayStyle: {data: {palette: 'diverging'}}
    };
    const result = edit(document, () =>
      replaceWorkflowStepOperation(document, sourceRegistry, 'source', 'fixture-count')
    );
    expect(result.steps[0].operationId).toBe('fixture-count');
    expect(result.steps[0].display).toBeUndefined();
    expect(result.steps[0].displayStyle).toBeUndefined();
    expect(result.steps[1].inputs).toEqual({});
    expect(result.steps[1].parameters).toBe(document.steps[1].parameters);
    expect(result.steps[2]).toBe(document.steps[2]);
    expect(result.steps[2].inputs).toEqual({rows: {stepId: 'scale', output: 'scaled'}});
  });

  test('keeps static parameters only for the same name and kind, checks input bindings, and applies copied overrides last', () => {
    const parameterNames = [
      'same',
      'changed',
      'widened',
      'enumKeep',
      'enumDrop',
      'wrongKind',
      'missingInput',
      'removed',
      'overridden'
    ];
    const previous: WorkflowOperation = {
      ...registry.get('fixture-scale')!,
      id: 'previous',
      parameters: parameterNames.map(name => ({name, label: name, kind: 'number', defaultValue: 0}))
    };
    const next: WorkflowOperation = {
      ...previous,
      id: 'next',
      parameters: [
        {name: 'same', label: 'Same', kind: 'number', defaultValue: 0},
        {name: 'changed', label: 'Changed', kind: 'integer', defaultValue: 0},
        {name: 'widened', label: 'Widened', kind: 'number', defaultValue: 0},
        {
          name: 'enumKeep',
          label: 'Keep choice',
          kind: 'enum',
          options: ['a', 'b'],
          defaultValue: 'a'
        },
        {name: 'enumDrop', label: 'Drop choice', kind: 'enum', options: ['a'], defaultValue: 'a'},
        {name: 'wrongKind', label: 'Wrong kind', kind: 'number', defaultValue: 0},
        {name: 'missingInput', label: 'Missing input', kind: 'number', defaultValue: 0},
        {name: 'overridden', label: 'Overridden', kind: 'number', defaultValue: 0},
        {name: 'values', label: 'Values', kind: 'number-list', defaultValue: [0]}
      ]
    };
    const operationRegistry = extendFixtureRegistry(previous, next);
    const document: WorkflowDocument = {
      ...makeFixtureDocument(),
      inputs: [
        {id: 'integer', label: 'Integer', kind: 'integer', value: 4},
        {id: 'choice', label: 'Choice', kind: 'enum', options: ['a', 'b'], value: 'a'},
        {id: 'boolean', label: 'Boolean', kind: 'boolean', value: false}
      ]
    };
    document.steps[1] = {
      ...document.steps[1],
      operationId: 'previous',
      parameters: {
        same: {from: 'static', value: 7},
        changed: {from: 'static', value: 3},
        widened: {from: 'input', inputId: 'integer'},
        enumKeep: {from: 'input', inputId: 'choice'},
        enumDrop: {from: 'input', inputId: 'choice'},
        wrongKind: {from: 'input', inputId: 'boolean'},
        missingInput: {from: 'input', inputId: 'absent'},
        removed: {from: 'static', value: 9},
        overridden: {from: 'static', value: 1}
      }
    };
    const values = [1, 2];
    const result = edit(document, () =>
      replaceWorkflowStepOperation(document, operationRegistry, 'scale', 'next', {
        overridden: 10,
        values
      })
    );
    values[0] = Infinity;
    expect(result.steps[1].parameters).toEqual({
      same: {from: 'static', value: 7},
      widened: {from: 'input', inputId: 'integer'},
      enumKeep: {from: 'input', inputId: 'choice'},
      overridden: {from: 'static', value: 10},
      values: {from: 'static', value: [1, 2]}
    });
    expect(result.steps[0]).toBe(document.steps[0]);
    expect(result.steps[2]).toBe(document.steps[2]);
  });

  test('keeps maybe-compatible named inputs and drops incompatible, missing and unknown producer ports', () => {
    const document = makeFixtureDocument();
    document.steps.push({
      ...step('target'),
      operationId: 'fixture-scale',
      inputs: {
        points: {stepId: 'source', output: 'data'},
        polygons: {stepId: 'scale', output: 'scaled'},
        removed: {stepId: 'scale', output: 'scaled'},
        unknownOutput: {stepId: 'scale', output: 'missing'},
        dangling: {stepId: 'missing', output: 'data'}
      }
    });
    const next: WorkflowOperation = {
      ...registry.get('fixture-scale')!,
      id: 'different-ports',
      inputs: [
        {name: 'points', label: 'Points', accepts: ['point'], maximumRows: 10},
        {name: 'polygons', label: 'Polygons', accepts: ['polygon'], maximumRows: 10},
        {name: 'unknownOutput', label: 'Unknown output', accepts: ['point'], maximumRows: 10},
        {name: 'dangling', label: 'Dangling', accepts: ['point'], maximumRows: 10}
      ]
    };
    const result = edit(document, () =>
      replaceWorkflowStepOperation(
        document,
        extendFixtureRegistry(next),
        'target',
        'different-ports'
      )
    );
    expect(result.steps[3].inputs).toEqual({points: {stepId: 'source', output: 'data'}});
    document.steps
      .slice(0, 3)
      .forEach((original, index) => expect(result.steps[index]).toBe(original));
  });

  test('returns the same document for a no-op and reports exact refusal codes', () => {
    const document = makeFixtureDocument();
    edit(document, () => {
      expect(replaceWorkflowStepOperation(document, registry, 'scale', 'fixture-scale')).toBe(
        document
      );
      expect(replaceWorkflowStepOperation(document, registry, 'scale', 'fixture-scale', {})).toBe(
        document
      );
      expectCode(
        () => replaceWorkflowStepOperation(document, registry, 'missing', 'fixture-scale'),
        'unknown-step'
      );
      expectCode(
        () => replaceWorkflowStepOperation(document, registry, 'scale', 'missing'),
        'unknown-operation'
      );
      expectCode(
        () =>
          replaceWorkflowStepOperation(document, registry, 'scale', 'fixture-scale', {unknown: 1}),
        'unknown-parameter',
        'Parameter unknown'
      );
    });
    const overridden = edit(document, () =>
      replaceWorkflowStepOperation(document, registry, 'scale', 'fixture-scale', {factor: -9})
    );
    expect(overridden.steps[1].parameters['factor']).toEqual({from: 'static', value: -9});
    expect(overridden.steps[0]).toBe(document.steps[0]);
    expect(
      edit(overridden, () =>
        replaceWorkflowStepOperation(overridden, registry, 'scale', 'fixture-scale', {factor: -9})
      )
    ).toBe(overridden);
  });
});

describe('composition review regressions', () => {
  test('F2 refuses styles that violate the persistence schema before committing them', () => {
    const document = makeFixtureDocument();
    document.steps[1].label = 'Scaled points';
    const invalidStyles: Array<{style: unknown; code: WorkflowIssueCode; message: string}> = [
      {
        style: {colorColumn: 'a'.repeat(65)},
        code: 'limit-exceeded',
        message: 'maximum colorColumn characters'
      },
      {style: {colorColumn: ''}, code: 'malformed', message: 'colorColumn'},
      {style: {colorScale: 'invalid'}, code: 'malformed', message: 'colorScale'},
      {style: {palette: 'invalid'}, code: 'malformed', message: 'palette'},
      {style: {unexpected: true}, code: 'malformed', message: 'unexpected'},
      {style: null, code: 'malformed', message: 'displayStyle'}
    ];
    for (const {style, code, message} of invalidStyles) {
      edit(document, () => {
        expectCode(
          () => setWorkflowDisplayStyle(document, 'scale', 'scaled', style as WorkflowDisplayStyle),
          code,
          message
        );
        expectCode(
          () => setWorkflowDisplayStyle(document, 'scale', 'scaled', style as WorkflowDisplayStyle),
          code,
          'Scaled points:'
        );
      });
    }
    edit(document, () =>
      expectCode(
        () => setWorkflowDisplayStyle(document, 'scale', '', {palette: 'diverging'}),
        'malformed',
        'output name must be non-empty'
      )
    );
  });

  test('F2 normalizes valid styles and accepts an exactly 64-character column while preserving snapshots', () => {
    const document = makeFixtureDocument();
    const style: WorkflowDisplayStyle = {
      colorColumn: 'a'.repeat(64),
      colorScale: 'quantile',
      palette: undefined
    };
    const updated = edit(document, () =>
      setWorkflowDisplayStyle(document, 'scale', 'scaled', style)
    );
    expect(updated.steps[1].displayStyle).toEqual({
      scaled: {colorColumn: 'a'.repeat(64), colorScale: 'quantile'}
    });
    expect(() => serializeWorkflowDocument(updated)).not.toThrow();
    style.colorColumn = 'changed';
    expect(updated.steps[1].displayStyle!['scaled'].colorColumn).toBe('a'.repeat(64));
    expect(
      edit(updated, () =>
        setWorkflowDisplayStyle(updated, 'scale', 'scaled', {
          colorColumn: 'a'.repeat(64),
          colorScale: 'quantile'
        })
      )
    ).toBe(updated);
    const cleared = edit(updated, () =>
      setWorkflowDisplayStyle(updated, 'scale', 'scaled', {palette: undefined})
    );
    expect(cleared.steps[1].displayStyle).toBeUndefined();
    expect(() => serializeWorkflowDocument(cleared)).not.toThrow();
  });

  test('F5 treats matching template step ids as foreign when the workflows differ', () => {
    const source = {...makeFixtureDocument(), id: 'workflow-a'};
    const destination: WorkflowDocument = {
      ...makeFixtureDocument(),
      id: 'workflow-b',
      steps: makeFixtureDocument().steps.map((candidate, index) => ({
        ...candidate,
        position: {x: (index + 1) * 1000, y: 200}
      }))
    };
    const clipboard = edit(source, () => copyWorkflowSteps(source, ['source', 'scale']));
    const result = edit(destination, () => pasteWorkflowSteps(destination, registry, clipboard));
    expect(result.document.steps.slice(3).map(candidate => candidate.position)).toEqual([
      {x: 3280, y: 200},
      {x: 3560, y: 200}
    ]);
    expect(result.document.steps[4].inputs).toEqual({points: {stepId: 'source-2', output: 'data'}});
    destination.steps.forEach((candidate, index) =>
      expect(result.document.steps[index]).toBe(candidate)
    );
  });

  test('F5 preserves and validates explicit clipboard origin and treats legacy clipboards as foreign', () => {
    const document = {...makeFixtureDocument(), id: 'workflow-a'};
    const clipboard = edit(document, () => copyWorkflowSteps(document, ['source', 'scale']));
    expect(clipboard.sourceWorkflowId).toBe('workflow-a');
    const parsed = parseWorkflowClipboard(JSON.stringify(clipboard));
    expect(parsed).toEqual(clipboard);
    const local = edit(document, () => pasteWorkflowSteps(document, registry, parsed));
    expect(local.document.steps[3].position).toEqual({x: 40, y: 40});
    const legacy = parseWorkflowClipboard({
      schemaVersion: 1,
      steps: clipboard.steps,
      inputs: clipboard.inputs
    });
    const foreign = edit(document, () => pasteWorkflowSteps(document, registry, legacy));
    expect(foreign.document.steps[3].position).toEqual({x: 840, y: 0});
    expectCode(
      () => parseWorkflowClipboard({...clipboard, sourceWorkflowId: 'Invalid id'}),
      'malformed'
    );
    expectCode(
      () => parseWorkflowClipboard({...clipboard, sourceWorkflowId: 'a'.repeat(49)}),
      'limit-exceeded'
    );
    expectCode(() => parseWorkflowClipboard({...clipboard, sourceWorkflowId: null}), 'malformed');
  });
});
