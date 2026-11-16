// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  createWorkflowDocument,
  isWorkflowId,
  makeUniqueWorkflowId,
  parseWorkflowCollection,
  parseWorkflowDocument,
  serializeWorkflowCollection,
  serializeWorkflowDocument
} from '../../src/workflows/workflow-document';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowDocument,
  type WorkflowDisplayStyle,
  type WorkflowInput,
  type WorkflowIssueCode
} from '../../src/workflows/workflow-types';
import {makeFixtureDocument} from './workflow-fixtures';

function expectRefusal(action: () => unknown, code: WorkflowIssueCode, path: string): void {
  let failure: unknown;
  try {
    action();
  } catch (error) {
    failure = error;
  }
  expect(failure).toBeInstanceOf(WorkflowError);
  expect((failure as WorkflowError).code).toBe(code);
  expect((failure as WorkflowError).message).toContain(path);
}

function makeInput(overrides: Partial<WorkflowInput> = {}): WorkflowInput {
  return {id: 'factor', label: 'Factor', kind: 'number', value: 2, ...overrides};
}

function replaceStepField(field: string, value: unknown): unknown {
  const document = makeFixtureDocument();
  return {...document, steps: [{...document.steps[0], [field]: value}]};
}

describe('workflow document canonicalisation', () => {
  test('accepts the fixture and deeply copies every object and array', () => {
    const document = makeFixtureDocument();
    document.inputs = [makeInput({kind: 'number-list', value: [1, 2]})];
    document.steps[0].parameters['values'] = {from: 'static', value: [1, 2]};
    const parsed = parseWorkflowDocument(document);
    expect(parsed).toEqual(document);
    const visit = (original: unknown, copy: unknown): void => {
      if (original === null || typeof original !== 'object') return;
      expect(copy).not.toBe(original);
      for (const key of Object.keys(original)) {
        visit((original as Record<string, unknown>)[key], (copy as Record<string, unknown>)[key]);
      }
    };
    visit(document, parsed);
    parsed.steps[1].inputs['points'].output = 'changed';
    expect(document.steps[1].inputs['points'].output).toBe('data');
  });

  test('canonical field order, optional omissions and stable record ordering', () => {
    const document = makeFixtureDocument();
    document.title = '  Workflow  ';
    document.description = '';
    Object.assign(document.steps[0], {label: '', note: '', disabled: false, display: []});
    Object.assign(document.steps[1], {
      label: 'Scale',
      note: 'Note',
      disabled: true,
      engine: 'cpu',
      onFallback: 'stop'
    });
    document.steps[0].parameters = {
      zebra: {from: 'static', value: -0},
      alpha: {from: 'input', inputId: 'missing'}
    };
    document.steps[2].inputs = {
      zebra: {stepId: 'missing', output: 'unknown'},
      alpha: {stepId: 'source', output: 'data'}
    };
    const canonical = parseWorkflowDocument(document);
    expect(canonical.title).toBe('Workflow');
    expect(canonical.description).toBeUndefined();
    expect(Object.keys(canonical)).toEqual([
      'schemaVersion',
      'id',
      'title',
      'inputs',
      'steps',
      'settings'
    ]);
    expect(Object.keys(canonical.steps[0])).toEqual([
      'id',
      'operationId',
      'inputs',
      'parameters',
      'position'
    ]);
    expect(Object.keys(canonical.steps[1])).toEqual([
      'id',
      'operationId',
      'label',
      'inputs',
      'parameters',
      'disabled',
      'display',
      'engine',
      'onFallback',
      'position',
      'note'
    ]);
    expect(Object.keys(canonical.steps[0].parameters)).toEqual(['alpha', 'zebra']);
    expect(Object.keys(canonical.steps[2].inputs)).toEqual(['alpha', 'zebra']);
    expect(canonical.steps[0].parameters['zebra']).toEqual({from: 'static', value: 0});
    const serialized = serializeWorkflowDocument(document);
    expect(parseWorkflowDocument(serialized)).toEqual(canonical);
    expect(JSON.stringify(serializeWorkflowDocument(parseWorkflowDocument(serialized)))).toBe(
      JSON.stringify(serialized)
    );
    document.steps[0].parameters = {
      alpha: {inputId: 'missing', from: 'input'},
      zebra: {value: 0, from: 'static'}
    };
    expect(JSON.stringify(serializeWorkflowDocument(document))).toBe(JSON.stringify(serialized));
  });

  test('preserves optional values and uses declared field order for every object', () => {
    const document = makeFixtureDocument();
    document.description = 'Details';
    document.inputs = [
      makeInput({
        kind: 'enum',
        value: 'a',
        unit: 'meters',
        minimum: 2,
        maximum: 2,
        options: ['a', 'b']
      })
    ];
    document.steps[0].parameters = {choice: {from: 'input', inputId: 'factor'}};
    document.steps[0].engine = 'auto';
    document.steps[0].onFallback = 'warn';
    const canonical = parseWorkflowDocument(document);
    expect(Object.keys(canonical)).toEqual([
      'schemaVersion',
      'id',
      'title',
      'description',
      'inputs',
      'steps',
      'settings'
    ]);
    expect(Object.keys(canonical.inputs[0])).toEqual([
      'id',
      'label',
      'kind',
      'value',
      'unit',
      'minimum',
      'maximum',
      'options'
    ]);
    expect(Object.keys(canonical.steps[0].parameters['choice'])).toEqual(['from', 'inputId']);
    expect(Object.keys(canonical.steps[1].parameters['factor'])).toEqual(['from', 'value']);
    expect(Object.keys(canonical.steps[1].inputs['points'])).toEqual(['stepId', 'output']);
    expect(Object.keys(canonical.steps[0].position)).toEqual(['x', 'y']);
    expect(Object.keys(canonical.settings)).toEqual(['engine', 'autoRun']);
    expect(parseWorkflowDocument(serializeWorkflowDocument(document))).toEqual(canonical);
  });

  test('unknown operations, dangling references, input ids and multi-step cycles remain structural', () => {
    const document = makeFixtureDocument();
    document.steps[0].operationId = 'an unknown future operation';
    document.steps[0].inputs = {future: {stepId: 'count', output: 'future'}};
    document.steps[0].parameters = {future: {from: 'input', inputId: 'missing'}};
    expect(parseWorkflowDocument(document)).toEqual(document);
  });

  test.each([
    ['number', 2],
    ['integer', 2],
    ['boolean', false],
    ['enum', 'a'],
    ['string', 'text'],
    ['number-list', [1, 2]]
  ] as const)('accepts %s inputs', (kind, value) => {
    const document = makeFixtureDocument();
    document.inputs = [makeInput({kind, value, ...(kind === 'enum' ? {options: ['a', 'b']} : {})})];
    expect(parseWorkflowDocument(document).inputs).toEqual(document.inputs);
  });

  test.each(['text', true, 2, [], [1, 2]])('accepts static parameter %j', value => {
    const document = replaceStepField('parameters', {value: {from: 'static', value}});
    expect(parseWorkflowDocument(document).steps[0].parameters['value']).toEqual({
      from: 'static',
      value
    });
  });
});

describe('workflow document refusals', () => {
  const documentCases: Array<[string, unknown, WorkflowIssueCode, string]> = [
    ['null', null, 'malformed', 'document'],
    ['array', [], 'malformed', 'document'],
    ['version', {...makeFixtureDocument(), schemaVersion: 2}, 'malformed', 'schemaVersion'],
    ['missing title', {...makeFixtureDocument(), title: undefined}, 'malformed', 'title'],
    ['empty title', {...makeFixtureDocument(), title: '  '}, 'malformed', 'title'],
    ['invalid id', {...makeFixtureDocument(), id: 'Bad'}, 'malformed', 'id'],
    ['id with newline', {...makeFixtureDocument(), id: 'a\n'}, 'malformed', 'id'],
    ['extra root field', {...makeFixtureDocument(), extra: true}, 'malformed', 'document.extra'],
    ['bad description', {...makeFixtureDocument(), description: 1}, 'malformed', 'description'],
    ['bad inputs', {...makeFixtureDocument(), inputs: {}}, 'malformed', 'inputs'],
    ['bad steps', {...makeFixtureDocument(), steps: {}}, 'malformed', 'steps'],
    ['missing settings', {...makeFixtureDocument(), settings: undefined}, 'malformed', 'settings'],
    [
      'missing autoRun',
      {...makeFixtureDocument(), settings: {engine: 'auto'}},
      'malformed',
      'settings.autoRun'
    ],
    [
      'missing engine',
      {...makeFixtureDocument(), settings: {autoRun: false}},
      'malformed',
      'settings.engine'
    ],
    [
      'bad autoRun',
      {...makeFixtureDocument(), settings: {engine: 'cpu', autoRun: 1}},
      'malformed',
      'settings.autoRun'
    ],
    [
      'extra settings',
      {...makeFixtureDocument(), settings: {engine: 'auto', autoRun: false, extra: 1}},
      'malformed',
      'settings.extra'
    ]
  ];
  test.each(documentCases)('%s', (_name, value, code, path) => {
    expectRefusal(() => parseWorkflowDocument(value), code, path);
  });

  const stepCases: Array<[string, unknown, string]> = [
    ['id', 'Invalid', 'id'],
    ['operationId', '', 'operationId'],
    ['operationId', '  ', 'operationId'],
    ['operationId', 4, 'operationId'],
    ['label', false, 'label'],
    ['note', false, 'note'],
    ['extra', true, 'extra'],
    ['inputs', [], 'inputs'],
    ['inputs', {points: {stepId: 'source', output: 'data'}}, 'inputs.points.stepId'],
    ['inputs', {points: {stepId: 1, output: 'data'}}, 'inputs.points.stepId'],
    ['inputs', {points: {stepId: 'other', output: 1}}, 'inputs.points.output'],
    ['inputs', {points: {stepId: 'other', output: 'data', extra: 1}}, 'inputs.points.extra'],
    ['parameters', [], 'parameters'],
    ['parameters', {distance: {from: 'static', value: 1, form: 2}}, 'parameters.distance.form'],
    ['parameters', {distance: {from: 'other', value: 1}}, 'parameters.distance.from'],
    ['parameters', {distance: {from: 'static', value: null}}, 'parameters.distance.value'],
    ['parameters', {distance: {from: 'static', value: Infinity}}, 'parameters.distance.value'],
    ['parameters', {distance: {from: 'static', value: [1, NaN]}}, 'parameters.distance.value[1]'],
    ['parameters', {distance: {from: 'static', value: [true]}}, 'parameters.distance.value[0]'],
    [
      'parameters',
      {distance: {from: 'static', value: 1, inputId: 'factor'}},
      'parameters.distance.inputId'
    ],
    ['parameters', {distance: {from: 'input', inputId: 2}}, 'parameters.distance.inputId'],
    [
      'parameters',
      {distance: {from: 'input', inputId: 'factor', value: 1}},
      'parameters.distance.value'
    ],
    ['parameters', {distance: {from: 'input'}}, 'parameters.distance.inputId'],
    ['display', ['data', 'data'], 'display'],
    ['display', [''], 'display[0]'],
    ['display', [2], 'display[0]'],
    ['display', {}, 'display'],
    ['engine', 'gpu', 'engine'],
    ['onFallback', 'continue', 'onFallback'],
    ['disabled', 0, 'disabled'],
    ['position', {x: NaN, y: 0}, 'position.x'],
    ['position', {x: 0, y: Infinity}, 'position.y'],
    ['position', {x: 0}, 'position.y'],
    ['position', {x: 0, y: 0, z: 0}, 'position.z']
  ];
  test.each(stepCases)('step %s = %j', (field, value, path) => {
    expectRefusal(
      () => parseWorkflowDocument(replaceStepField(field, value)),
      'malformed',
      `steps[0].${path}`
    );
  });

  test('step errors start with the label or operation id', () => {
    const document = makeFixtureDocument();
    document.steps[1].label = 'Buffer points';
    document.steps[1].position.x = Infinity;
    expect(() => parseWorkflowDocument(document)).toThrow(
      /^Buffer points: .*steps\[1\]\.position.x/
    );
    delete document.steps[1].label;
    expect(() => parseWorkflowDocument(document)).toThrow(/^fixture-scale:/);
  });

  const inputCases: Array<[Record<string, unknown>, string]> = [
    [{id: 'Invalid'}, 'id'],
    [{label: 1}, 'label'],
    [{kind: 'text'}, 'kind'],
    [{kind: 'number', value: '2'}, 'value'],
    [{kind: 'number', value: NaN}, 'value'],
    [{kind: 'integer', value: 1.5}, 'value'],
    [{kind: 'boolean', value: 1}, 'value'],
    [{kind: 'string', value: 1}, 'value'],
    [{kind: 'number-list', value: []}, 'value'],
    [{kind: 'number-list', value: [Infinity]}, 'value[0]'],
    [{kind: 'enum', value: 'a'}, 'options'],
    [{kind: 'enum', value: 'a', options: []}, 'options'],
    [{kind: 'enum', value: 'a', options: ['b']}, 'value'],
    [{kind: 'enum', value: 'a', options: [1]}, 'options[0]'],
    [{options: ['a']}, 'options'],
    [{minimum: 3, maximum: 2}, 'minimum'],
    [{minimum: Infinity}, 'minimum'],
    [{maximum: '4'}, 'maximum'],
    [{unit: 1}, 'unit'],
    [{extra: 1}, 'extra']
  ];
  test.each(inputCases)('input %j', (fields, path) => {
    const document = {...makeFixtureDocument(), inputs: [{...makeInput(), ...fields}]};
    expectRefusal(() => parseWorkflowDocument(document), 'malformed', `inputs[0].${path}`);
  });

  test.each(['steps', 'inputs'] as const)('duplicate %s ids', field => {
    const document = makeFixtureDocument();
    document.inputs = [makeInput(), makeInput()];
    if (field === 'steps') document.steps.push({...document.steps[0]});
    const path = field === 'steps' ? 'steps[3].id' : 'inputs[1].id';
    if (field === 'steps') document.inputs = [];
    expectRefusal(() => parseWorkflowDocument(document), 'duplicate-id', path);
  });
});

describe('untrusted properties', () => {
  test.each([
    '__proto__',
    'constructor',
    'prototype'
  ])('refuses %s throughout the document', key => {
    const malicious = JSON.parse(`{"${key}": {"polluted": true}}`) as Record<string, unknown>;
    const document = makeFixtureDocument();
    const cases = [
      [{...document, ...malicious}, `document.${key}`],
      [replaceStepField(key, malicious[key]), `steps[0].${key}`],
      [replaceStepField('parameters', malicious), `parameters.${key}`],
      [replaceStepField('inputs', malicious), `inputs.${key}`],
      [
        replaceStepField('inputs', {points: {stepId: 'other', output: 'data', ...malicious}}),
        `inputs.points.${key}`
      ],
      [replaceStepField('position', {x: 0, y: 0, ...malicious}), `position.${key}`],
      [
        replaceStepField('parameters', {distance: {from: 'static', value: 1, ...malicious}}),
        `parameters.distance.${key}`
      ],
      [{...document, inputs: [{...makeInput(), ...malicious}]}, `inputs[0].${key}`],
      [{...document, settings: {engine: 'auto', autoRun: false, ...malicious}}, `settings.${key}`]
    ] as const;
    const previous = Object.getOwnPropertyDescriptors(Object.prototype);
    for (const [value, path] of cases)
      expectRefusal(() => parseWorkflowDocument(value), 'malformed', path);
    expect(Object.getOwnPropertyDescriptors(Object.prototype)).toEqual(previous);
    expect(({} as Record<string, unknown>)['polluted']).toBeUndefined();
  });

  test('refuses inherited fields, accessors, symbols, nonenumerable fields and sparse arrays', () => {
    const inherited = Object.assign(Object.create({title: 'Inherited'}), makeFixtureDocument());
    expectRefusal(() => parseWorkflowDocument(inherited), 'malformed', 'document.title');
    let getterCalls = 0;
    const accessor = Object.defineProperty(makeFixtureDocument(), 'title', {
      enumerable: true,
      get: () => {
        getterCalls++;
        return 'Title';
      }
    });
    expectRefusal(() => parseWorkflowDocument(accessor), 'malformed', 'document.title');
    expect(getterCalls).toBe(0);
    const symbol = {...makeFixtureDocument(), [Symbol('extra')]: 1};
    expectRefusal(() => parseWorkflowDocument(symbol), 'malformed', 'Symbol(extra)');
    const hidden = Object.defineProperty(makeFixtureDocument(), 'extra', {value: 1});
    expectRefusal(() => parseWorkflowDocument(hidden), 'malformed', 'document.extra');
    const sparse = new Array(1);
    expectRefusal(
      () => parseWorkflowDocument({...makeFixtureDocument(), steps: sparse}),
      'malformed',
      'steps[0]'
    );
    let arrayGetterCalls = 0;
    const accessorArray = Object.defineProperty(['data'], '0', {
      enumerable: true,
      get: () => {
        arrayGetterCalls++;
        return 'data';
      }
    });
    expectRefusal(
      () => parseWorkflowDocument(replaceStepField('display', accessorArray)),
      'malformed',
      'display[0]'
    );
    expect(arrayGetterCalls).toBe(0);
    const inheritedArray = Object.setPrototypeOf(['data'], {extra: 1});
    expectRefusal(
      () => parseWorkflowDocument(replaceStepField('display', inheritedArray)),
      'malformed',
      'display'
    );
    const numericLookingArray = Object.assign(['data'], {'0\n': 'data'});
    expectRefusal(
      () => parseWorkflowDocument(replaceStepField('display', numericLookingArray)),
      'malformed',
      'display.0\n'
    );
    const extraArray = Object.assign([], {extra: 1});
    expectRefusal(
      () => parseWorkflowDocument({...makeFixtureDocument(), steps: extraArray}),
      'malformed',
      'steps.extra'
    );
  });

  test('supports null-prototype JSON objects without retaining their prototype', () => {
    const document = Object.assign(Object.create(null), makeFixtureDocument());
    expect(parseWorkflowDocument(document)).toEqual(makeFixtureDocument());
    expect(Object.getPrototypeOf(parseWorkflowDocument(document))).toBe(Object.prototype);
  });
});

describe('named limits', () => {
  const limitCases: Array<[string, string, number, (size: number) => unknown]> = [
    [
      'maximumIdCharacters',
      'id',
      WORKFLOW_LIMITS.maximumIdCharacters,
      size => ({...makeFixtureDocument(), id: 'a'.repeat(size)})
    ],
    [
      'maximumTitleCharacters',
      'title',
      WORKFLOW_LIMITS.maximumTitleCharacters,
      size => ({...makeFixtureDocument(), title: ` ${'a'.repeat(size)} `})
    ],
    [
      'maximumNoteCharacters',
      'description',
      WORKFLOW_LIMITS.maximumNoteCharacters,
      size => ({...makeFixtureDocument(), description: 'a'.repeat(size)})
    ],
    [
      'maximumNoteCharacters',
      'steps[0].note',
      WORKFLOW_LIMITS.maximumNoteCharacters,
      size => replaceStepField('note', 'a'.repeat(size))
    ],
    [
      'maximumTextCharacters',
      'parameters.text.value',
      WORKFLOW_LIMITS.maximumTextCharacters,
      size => replaceStepField('parameters', {text: {from: 'static', value: 'a'.repeat(size)}})
    ],
    [
      'maximumCanvasCoordinate',
      'position.x',
      WORKFLOW_LIMITS.maximumCanvasCoordinate,
      size => replaceStepField('position', {x: size, y: -size})
    ],
    [
      'maximum operationId characters',
      'operationId',
      80,
      size => replaceStepField('operationId', 'a'.repeat(size))
    ],
    [
      'maximum parameter number-list length',
      'parameters.list.value',
      WORKFLOW_LIMITS.maximumParameterNumbers,
      size =>
        replaceStepField('parameters', {
          list: {from: 'static', value: Array.from({length: size}, () => 1)}
        })
    ],
    [
      'maximumSteps',
      'steps',
      WORKFLOW_LIMITS.maximumSteps,
      size => ({
        ...makeFixtureDocument(),
        steps: Array.from({length: size}, (_, index) => ({
          ...makeFixtureDocument().steps[0],
          id: `step-${index}`
        }))
      })
    ],
    [
      'maximumInputs',
      'inputs',
      WORKFLOW_LIMITS.maximumInputs,
      size => ({
        ...makeFixtureDocument(),
        inputs: Array.from({length: size}, (_, index) => makeInput({id: `input-${index}`}))
      })
    ]
  ];
  test.each(limitCases)('%s at boundary and one over (%s)', (limit, path, maximum, makeValue) => {
    expect(() => parseWorkflowDocument(makeValue(maximum))).not.toThrow();
    expectRefusal(() => parseWorkflowDocument(makeValue(maximum + 1)), 'limit-exceeded', path);
    expectRefusal(() => parseWorkflowDocument(makeValue(maximum + 1)), 'limit-exceeded', limit);
    expectRefusal(
      () => parseWorkflowDocument(makeValue(maximum + 1)),
      'limit-exceeded',
      String(maximum + 1)
    );
  });

  test('maximumDocumentCharacters uses the canonical serialized length including JSON escaping', () => {
    const document = makeFixtureDocument();
    document.steps[0].parameters = Object.fromEntries(
      Array.from({length: 5}, (_, index) => [`text-${index}`, {from: 'static', value: ''}])
    );
    const overhead = JSON.stringify(serializeWorkflowDocument(document)).length;
    let remaining = WORKFLOW_LIMITS.maximumDocumentCharacters - overhead;
    for (const binding of Object.values(document.steps[0].parameters)) {
      if (binding.from !== 'static') throw new Error('Test needs a static binding');
      const length = Math.min(remaining, WORKFLOW_LIMITS.maximumTextCharacters);
      binding.value = 'a'.repeat(length);
      remaining -= length;
    }
    expect(remaining).toBe(0);
    expect(JSON.stringify(serializeWorkflowDocument(document)).length).toBe(
      WORKFLOW_LIMITS.maximumDocumentCharacters
    );
    const binding = document.steps[0].parameters['text-4'];
    if (binding.from !== 'static') throw new Error('Test needs a static binding');
    binding.value = `${binding.value}a`;
    expectRefusal(
      () => parseWorkflowDocument(document),
      'limit-exceeded',
      'maximumDocumentCharacters'
    );
    expectRefusal(
      () => serializeWorkflowDocument(document),
      'limit-exceeded',
      String(WORKFLOW_LIMITS.maximumDocumentCharacters + 1)
    );
    binding.value = '\\'.repeat(1000);
    expectRefusal(
      () => serializeWorkflowDocument(document),
      'limit-exceeded',
      'maximumDocumentCharacters'
    );
  });
});

describe('workflow ids and creation', () => {
  test.each(['a', '1', 'workflow-2', 'a-', 'a'.repeat(48)])('valid id %s', value =>
    expect(isWorkflowId(value)).toBe(true));
  test.each([
    '',
    '-a',
    'A',
    'a_b',
    'a b',
    'é',
    'a\n',
    'a\r',
    'a\u2028',
    'a'.repeat(49)
  ])('invalid id %s', value => expect(isWorkflowId(value)).toBe(false));

  test('normalises ids, skips occupied suffixes and leaves room for the suffix', () => {
    expect(makeUniqueWorkflowId('  MY cool __ workflow!! ', [])).toBe('my-cool-workflow');
    expect(makeUniqueWorkflowId('!!!', ['workflow', 'workflow-2'])).toBe('workflow-3');
    expect(makeUniqueWorkflowId('a'.repeat(60), ['a'.repeat(48), `${'a'.repeat(46)}-2`])).toBe(
      `${'a'.repeat(46)}-3`
    );
    expect(makeUniqueWorkflowId('Workflow', new Set(['workflow-2']))).toBe('workflow');
    expect(makeUniqueWorkflowId(`${'a'.repeat(47)} b`, [])).toBe('a'.repeat(47));
    expect(makeUniqueWorkflowId(`${'a'.repeat(45)} bbb`, ['a'.repeat(45) + '-bb'])).toBe(
      `${'a'.repeat(45)}-2`
    );
    expect(
      makeUniqueWorkflowId(
        'a'.repeat(60),
        Array.from({length: 9}, (_, index) =>
          index === 0 ? 'a'.repeat(48) : `${'a'.repeat(46)}-${index + 1}`
        )
      )
    ).toBe(`${'a'.repeat(45)}-10`);
  });

  test('creates a validated canonical empty workflow', () => {
    expect(createWorkflowDocument()).toEqual({
      schemaVersion: 1,
      id: 'untitled-workflow',
      title: 'Untitled workflow',
      inputs: [],
      steps: [],
      settings: {engine: 'auto', autoRun: false}
    });
    expect(createWorkflowDocument({title: ' My flow ', description: 'Details'})).toEqual({
      ...createWorkflowDocument(),
      id: 'my-flow',
      title: 'My flow',
      description: 'Details'
    });
    expect(createWorkflowDocument({id: 'chosen', title: 'Title'}).id).toBe('chosen');
    expectRefusal(() => createWorkflowDocument({id: 'Bad'}), 'malformed', 'id');
    expectRefusal(() => createWorkflowDocument({title: ''}), 'malformed', 'title');
  });
});

describe('workflow collection', () => {
  function makeCollection(count: number): {schemaVersion: 1; workflows: WorkflowDocument[]} {
    return {
      schemaVersion: 1,
      workflows: Array.from({length: count}, (_, index) => ({
        ...makeFixtureDocument(),
        id: `workflow-${index}`
      }))
    };
  }

  test('empty collections have no saved member', () => {
    expect(serializeWorkflowCollection(makeCollection(0))).toBeUndefined();
    expect(parseWorkflowCollection(makeCollection(0))).toEqual({schemaVersion: 1, workflows: []});
  });

  test('round-trips and copies the active collection', () => {
    const collection = {...makeCollection(2), activeWorkflowId: 'workflow-1'};
    const serialized = serializeWorkflowCollection(collection);
    expect(parseWorkflowCollection(serialized)).toEqual(collection);
    expect(JSON.stringify(serialized)).toBe(
      JSON.stringify(serializeWorkflowCollection(parseWorkflowCollection(serialized)))
    );
    expect(parseWorkflowCollection(collection).workflows[0]).not.toBe(collection.workflows[0]);
    expect(Object.keys(parseWorkflowCollection(collection))).toEqual([
      'schemaVersion',
      'workflows',
      'activeWorkflowId'
    ]);
  });

  test('maximumWorkflows at and one over', () => {
    expect(
      parseWorkflowCollection(makeCollection(WORKFLOW_LIMITS.maximumWorkflows)).workflows
    ).toHaveLength(WORKFLOW_LIMITS.maximumWorkflows);
    expectRefusal(
      () => parseWorkflowCollection(makeCollection(WORKFLOW_LIMITS.maximumWorkflows + 1)),
      'limit-exceeded',
      'maximumWorkflows'
    );
    expectRefusal(
      () => parseWorkflowCollection(makeCollection(WORKFLOW_LIMITS.maximumWorkflows + 1)),
      'limit-exceeded',
      '17'
    );
  });

  test('rejects duplicate ids, unknown active ids and any malformed workflow with its index', () => {
    const collection = makeCollection(2);
    collection.workflows[1].id = collection.workflows[0].id;
    expectRefusal(() => parseWorkflowCollection(collection), 'duplicate-id', 'workflows[1].id');
    expectRefusal(
      () => parseWorkflowCollection({...makeCollection(1), activeWorkflowId: 'missing'}),
      'malformed',
      'activeWorkflowId'
    );
    expectRefusal(
      () => parseWorkflowCollection({...makeCollection(0), activeWorkflowId: 'missing'}),
      'malformed',
      'activeWorkflowId'
    );
    expectRefusal(
      () => parseWorkflowCollection({...makeCollection(1), activeWorkflowId: 'Bad'}),
      'malformed',
      'activeWorkflowId'
    );
    collection.workflows[1].id = 'unique';
    collection.workflows[1].steps[2].position.x = Infinity;
    expectRefusal(
      () => parseWorkflowCollection(collection),
      'malformed',
      'workflows[1].steps[2].position.x'
    );
    expectRefusal(
      () => serializeWorkflowCollection(collection),
      'malformed',
      'workflows[1].steps[2].position.x'
    );
  });

  test.each([
    [null, 'collection'],
    [{schemaVersion: 2, workflows: []}, 'schemaVersion'],
    [{schemaVersion: 1, workflows: {}}, 'workflows'],
    [{schemaVersion: 1, workflows: [], extra: 1}, 'collection.extra'],
    [JSON.parse('{"schemaVersion":1,"workflows":[],"__proto__":{}}'), '__proto__']
  ])('refuses malformed collection %j', (value, path) => {
    expectRefusal(() => parseWorkflowCollection(value), 'malformed', path as string);
  });
});

describe('review regressions', () => {
  test('F1: serializes an explicitly undefined description as absent', () => {
    const document = {...createWorkflowDocument(), description: undefined};
    const serialized = serializeWorkflowDocument(document);
    expect(serialized).toEqual(createWorkflowDocument());
    expect(JSON.stringify(serialized)).toBe(
      JSON.stringify(serializeWorkflowDocument(createWorkflowDocument()))
    );
    expect(Object.hasOwn(document, 'description')).toBe(true);
    expectRefusal(() => parseWorkflowDocument(document), 'malformed', 'document.description');
  });

  test.each([
    'label',
    'disabled',
    'display',
    'engine',
    'onFallback',
    'note'
  ] as const)('F1: serializes an explicitly undefined step %s as absent', field => {
    const document = makeFixtureDocument();
    document.steps[0][field] = undefined;
    const serialized = serializeWorkflowDocument(document);
    expect(serialized).toEqual(parseWorkflowDocument(makeFixtureDocument()));
    expect(Object.hasOwn(document.steps[0], field)).toBe(true);
    expectRefusal(() => parseWorkflowDocument(document), 'malformed', `steps[0].${field}`);
  });

  test.each([
    'unit',
    'minimum',
    'maximum',
    'options'
  ] as const)('F1: serializes an explicitly undefined input %s as absent', field => {
    const document = makeFixtureDocument();
    document.inputs = [makeInput()];
    document.inputs[0][field] = undefined;
    const expected = makeFixtureDocument();
    expected.inputs = [makeInput()];
    expect(serializeWorkflowDocument(document)).toEqual(parseWorkflowDocument(expected));
    expect(Object.hasOwn(document.inputs[0], field)).toBe(true);
    expectRefusal(() => parseWorkflowDocument(document), 'malformed', `inputs[0].${field}`);
  });

  test('F1: serializes undefined collection and nested document optionals as absent', () => {
    const document = makeFixtureDocument();
    document.description = undefined;
    document.steps[0].label = undefined;
    document.inputs = [
      makeInput({unit: undefined, minimum: undefined, maximum: undefined, options: undefined})
    ];
    const collection = {
      schemaVersion: 1 as const,
      workflows: [document],
      activeWorkflowId: undefined
    };
    const expected = {...makeFixtureDocument(), inputs: [makeInput()]};
    const serialized = serializeWorkflowCollection(collection);
    expect(serialized).toEqual({schemaVersion: 1, workflows: [parseWorkflowDocument(expected)]});
    expect(JSON.stringify(serializeWorkflowCollection(parseWorkflowCollection(serialized)))).toBe(
      JSON.stringify(serialized)
    );
    expect(
      serializeWorkflowCollection({schemaVersion: 1, workflows: [], activeWorkflowId: undefined})
    ).toBeUndefined();
    expectRefusal(
      () =>
        parseWorkflowCollection({
          schemaVersion: 1,
          workflows: [makeFixtureDocument()],
          activeWorkflowId: undefined
        }),
      'malformed',
      'collection.activeWorkflowId'
    );
  });

  test('F1: serialization still refuses unknown, required and conditionally required undefined fields', () => {
    expectRefusal(
      () =>
        serializeWorkflowDocument({...makeFixtureDocument(), extra: undefined} as WorkflowDocument),
      'malformed',
      'document.extra'
    );
    const document = makeFixtureDocument();
    document.inputs = [makeInput({kind: 'enum', value: 'a', options: undefined})];
    expectRefusal(() => serializeWorkflowDocument(document), 'malformed', 'inputs[0].options');
    expectRefusal(
      () =>
        serializeWorkflowDocument({
          ...makeFixtureDocument(),
          title: undefined
        } as unknown as WorkflowDocument),
      'malformed',
      'document.title'
    );
    document.inputs = [];
    document.steps[0].parameters = {
      distance: {
        from: 'static',
        value: undefined
      } as unknown as WorkflowDocument['steps'][number]['parameters'][string]
    };
    expectRefusal(
      () => serializeWorkflowDocument(document),
      'malformed',
      'parameters.distance.value'
    );
  });

  const countCases: Array<[string, string, number, (values: unknown[]) => unknown]> = [
    [
      'steps',
      'maximumSteps',
      WORKFLOW_LIMITS.maximumSteps,
      values => parseWorkflowDocument({...makeFixtureDocument(), steps: values})
    ],
    [
      'inputs',
      'maximumInputs',
      WORKFLOW_LIMITS.maximumInputs,
      values => parseWorkflowDocument({...makeFixtureDocument(), inputs: values})
    ],
    [
      'workflows',
      'maximumWorkflows',
      WORKFLOW_LIMITS.maximumWorkflows,
      values => parseWorkflowCollection({schemaVersion: 1, workflows: values})
    ],
    [
      'parameters.values.value',
      'maximum parameter number-list length',
      WORKFLOW_LIMITS.maximumParameterNumbers,
      values =>
        parseWorkflowDocument(
          replaceStepField('parameters', {values: {from: 'static', value: values}})
        )
    ]
  ];
  describe.each(
    countCases
  )('F2: %s enforces %s before reading array contents', (path, limit, maximum, parseValues) => {
    test.each(['dense', 'sparse'] as const)('%s array one over the limit', kind => {
      const values: unknown[] = new Array(maximum + 1);
      if (kind === 'dense') values.fill(null);
      let keyEnumerations = 0;
      let elementDescriptorReads = 0;
      const observed = new Proxy(values, {
        ownKeys(target) {
          keyEnumerations++;
          return Reflect.ownKeys(target);
        },
        getOwnPropertyDescriptor(target, property) {
          if (property !== 'length') elementDescriptorReads++;
          return Reflect.getOwnPropertyDescriptor(target, property);
        }
      });
      expectRefusal(() => parseValues(observed), 'limit-exceeded', path);
      expectRefusal(
        () => parseValues(observed),
        'limit-exceeded',
        `${limit} is ${maximum}; got ${maximum + 1}`
      );
      expect(keyEnumerations).toBe(0);
      expect(elementDescriptorReads).toBe(0);
    });
  });

  test('F2: a million-element array is rejected before enumerating or inspecting any element', () => {
    let keyEnumerations = 0;
    let elementDescriptorReads = 0;
    const values = new Proxy(new Array(1_000_000).fill(null), {
      ownKeys(target) {
        keyEnumerations++;
        return Reflect.ownKeys(target);
      },
      getOwnPropertyDescriptor(target, property) {
        if (property !== 'length') elementDescriptorReads++;
        return Reflect.getOwnPropertyDescriptor(target, property);
      }
    });
    expectRefusal(
      () => parseWorkflowDocument({...makeFixtureDocument(), steps: values}),
      'limit-exceeded',
      'maximumSteps is 64; got 1000000'
    );
    expect(keyEnumerations).toBe(0);
    expect(elementDescriptorReads).toBe(0);
  });

  test.each([
    42,
    true,
    false,
    0,
    {},
    ['Label'],
    null,
    undefined
  ])('F3: malformed label %j uses the operation-name prefix', label => {
    const action = () => parseWorkflowDocument(replaceStepField('label', label));
    expectRefusal(action, 'malformed', 'steps[0].label');
    expect(action).toThrow(/^fixture-source: document\.steps\[0\]\.label: must be a string\.$/);
  });
});

describe('serializer validation after optional omission', () => {
  test('rejects optional accessors without calling them and keeps the operation-name prefix', () => {
    let getterCalls = 0;
    const document = makeFixtureDocument();
    Object.defineProperty(document.steps[0], 'label', {
      enumerable: true,
      get: () => {
        getterCalls++;
        return undefined;
      }
    });
    expectRefusal(() => serializeWorkflowDocument(document), 'malformed', 'steps[0].label');
    expect(() => serializeWorkflowDocument(document)).toThrow(/^fixture-source:/);
    expect(getterCalls).toBe(0);
  });

  test('rejects unknown collection fields even when undefined', () => {
    const collection = {
      schemaVersion: 1 as const,
      workflows: [makeFixtureDocument()],
      extra: undefined
    };
    expectRefusal(() => serializeWorkflowCollection(collection), 'malformed', 'collection.extra');
  });
});

describe('display styles and saved user templates', () => {
  test('copies and canonicalises styles after display, sorting output names and omitting empty styles', () => {
    const document = makeFixtureDocument();
    document.steps[1].engine = 'cpu';
    document.steps[1].displayStyle = {
      zebra: {palette: 'diverging', colorScale: 'quantile', colorColumn: 'a'.repeat(64)},
      empty: {},
      alpha: {colorScale: 'ordinal', palette: 'categorical'},
      middle: {colorColumn: 'x', colorScale: 'quantize', palette: 'sequential'}
    };
    const canonical = parseWorkflowDocument(document);
    const styles = canonical.steps[1].displayStyle!;
    expect(styles).toEqual({
      alpha: {colorScale: 'ordinal', palette: 'categorical'},
      middle: {colorColumn: 'x', colorScale: 'quantize', palette: 'sequential'},
      zebra: {colorColumn: 'a'.repeat(64), colorScale: 'quantile', palette: 'diverging'}
    });
    expect(Object.keys(styles)).toEqual(['alpha', 'middle', 'zebra']);
    expect(Object.keys(styles['zebra'])).toEqual(['colorColumn', 'colorScale', 'palette']);
    expect(styles).not.toBe(document.steps[1].displayStyle);
    expect(styles['zebra']).not.toBe(document.steps[1].displayStyle['zebra']);
    expect(Object.keys(canonical.steps[1])).toEqual([
      'id',
      'operationId',
      'inputs',
      'parameters',
      'display',
      'displayStyle',
      'engine',
      'position'
    ]);
    const serialized = serializeWorkflowDocument(document);
    expect(parseWorkflowDocument(serialized)).toEqual(canonical);
    expect(JSON.stringify(serializeWorkflowDocument(parseWorkflowDocument(serialized)))).toBe(
      JSON.stringify(serialized)
    );
  });

  test('omits empty styles, style records and undefined style optionals when serializing', () => {
    const document = makeFixtureDocument();
    const emptyStyles: Record<string, WorkflowDisplayStyle>[] = [{}, {scaled: {}}];
    for (const styles of emptyStyles) {
      document.steps[1].displayStyle = styles;
      expect(serializeWorkflowDocument(document)).toEqual(
        parseWorkflowDocument(makeFixtureDocument())
      );
    }
    document.steps[1].displayStyle = {
      scaled: {colorColumn: undefined, colorScale: undefined, palette: undefined}
    };
    expect(serializeWorkflowDocument(document)).toEqual(
      parseWorkflowDocument(makeFixtureDocument())
    );
    document.steps[1].displayStyle = undefined;
    expect(serializeWorkflowDocument(document)).toEqual(
      parseWorkflowDocument(makeFixtureDocument())
    );
  });

  test('refuses malformed style fields, names, special keys and column lengths with exact codes and paths', () => {
    const cases: Array<[unknown, WorkflowIssueCode, string]> = [
      [[], 'malformed', 'displayStyle'],
      [{'': {}}, 'malformed', 'displayStyle.'],
      [{scaled: {extra: true}}, 'malformed', 'displayStyle.scaled.extra'],
      [{scaled: {colorColumn: ''}}, 'malformed', 'displayStyle.scaled.colorColumn'],
      [{scaled: {colorColumn: 1}}, 'malformed', 'displayStyle.scaled.colorColumn'],
      [
        {scaled: {colorColumn: 'a'.repeat(65)}},
        'limit-exceeded',
        'displayStyle.scaled.colorColumn'
      ],
      [{scaled: {colorScale: 'linear'}}, 'malformed', 'displayStyle.scaled.colorScale'],
      [{scaled: {palette: 'rainbow'}}, 'malformed', 'displayStyle.scaled.palette'],
      [JSON.parse('{"__proto__":{}}'), 'malformed', 'displayStyle.__proto__'],
      [{scaled: JSON.parse('{"constructor":{}}')}, 'malformed', 'displayStyle.scaled.constructor']
    ];
    for (const [value, code, path] of cases) {
      expectRefusal(
        () => parseWorkflowDocument(replaceStepField('displayStyle', value)),
        code,
        `steps[0].${path}`
      );
    }
  });

  test('saves template-only collections, copies documents and allows ids shared with workflows', () => {
    const template = makeFixtureDocument();
    const onlyTemplates = {schemaVersion: 1 as const, workflows: [], templates: [template]};
    const serialized = serializeWorkflowCollection(onlyTemplates);
    expect(serialized).toEqual(onlyTemplates);
    const parsed = parseWorkflowCollection(serialized);
    expect(parsed.templates![0]).not.toBe(template);
    expect(parsed.templates![0].steps[0]).not.toBe(template.steps[0]);
    expect(JSON.stringify(serializeWorkflowCollection(parsed))).toBe(JSON.stringify(serialized));
    const withWorkflow = {...onlyTemplates, workflows: [template], activeWorkflowId: template.id};
    expect(parseWorkflowCollection(withWorkflow)).toEqual(withWorkflow);
    expect(Object.keys(serializeWorkflowCollection(withWorkflow)!)).toEqual([
      'schemaVersion',
      'workflows',
      'activeWorkflowId',
      'templates'
    ]);
  });

  test('omits empty and undefined template lists, and serializes undefined optionals inside templates', () => {
    const collection = {schemaVersion: 1 as const, workflows: [makeFixtureDocument()]};
    expect(serializeWorkflowCollection({...collection, templates: []})).toEqual(collection);
    expect(serializeWorkflowCollection({...collection, templates: undefined})).toEqual(collection);
    expect(
      serializeWorkflowCollection({...collection, workflows: [], templates: []})
    ).toBeUndefined();
    const template = {...createWorkflowDocument(), description: undefined};
    expect(
      serializeWorkflowCollection({schemaVersion: 1, workflows: [], templates: [template]})
    ).toEqual({schemaVersion: 1, workflows: [], templates: [createWorkflowDocument()]});
  });

  test('enforces the user-template count at the boundary and before reading oversized arrays', () => {
    const templates = Array.from({length: WORKFLOW_LIMITS.maximumUserTemplates}, (_, index) =>
      createWorkflowDocument({id: `template-${index}`})
    );
    const collection = {schemaVersion: 1 as const, workflows: [], templates};
    expect(parseWorkflowCollection(collection).templates).toHaveLength(
      WORKFLOW_LIMITS.maximumUserTemplates
    );
    let keyEnumerations = 0;
    const oversized = new Proxy([...templates, createWorkflowDocument({id: 'extra'})], {
      ownKeys(target) {
        keyEnumerations++;
        return Reflect.ownKeys(target);
      }
    });
    expectRefusal(
      () => parseWorkflowCollection({...collection, templates: oversized}),
      'limit-exceeded',
      'collection.templates: maximumUserTemplates is 32; got 33'
    );
    expect(keyEnumerations).toBe(0);
  });

  test('rejects duplicate and malformed templates with their index, and requires the active id to name a workflow', () => {
    const template = makeFixtureDocument();
    const collection = {schemaVersion: 1 as const, workflows: [], templates: [template]};
    expectRefusal(
      () => parseWorkflowCollection({...collection, templates: [template, template]}),
      'duplicate-id',
      'collection.templates[1].id'
    );
    const malformed = {...template, steps: [{...template.steps[0], operationId: ''}]};
    expectRefusal(
      () => parseWorkflowCollection({...collection, templates: [template, malformed]}),
      'malformed',
      'collection.templates[1].steps[0].operationId'
    );
    expectRefusal(
      () => serializeWorkflowCollection({...collection, templates: [malformed]}),
      'malformed',
      'collection.templates[0].steps[0].operationId'
    );
    expectRefusal(
      () => parseWorkflowCollection({...collection, activeWorkflowId: template.id}),
      'malformed',
      'collection.activeWorkflowId'
    );
    expectRefusal(
      () => parseWorkflowCollection({...collection, templates: {}}),
      'malformed',
      'collection.templates'
    );
  });
});
