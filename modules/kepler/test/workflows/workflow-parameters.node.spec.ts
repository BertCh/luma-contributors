// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import type {CatalogParameterValue} from '../../src/analysis/catalog-binding-types';
import {
  formatWorkflowParameterValue,
  getWorkflowParameterDefaults,
  getWorkflowParameterProblem,
  isWorkflowParameterVisible,
  normalizeWorkflowParameters,
  parseWorkflowColumnList
} from '../../src/workflows/workflow-parameters';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowIssueCode,
  type WorkflowParameterKind,
  type WorkflowParameterSpec
} from '../../src/workflows/workflow-types';

const CONTEXT = {operationTitle: 'Buffer points'};

function makeSpec(
  kind: WorkflowParameterKind,
  overrides: Partial<WorkflowParameterSpec> = {}
): WorkflowParameterSpec {
  return {name: 'value', label: 'Value', kind, defaultValue: '', ...overrides};
}

function expectParameterError(
  run: () => unknown,
  message: string,
  parameter = 'value',
  code: WorkflowIssueCode = 'parameter-out-of-range'
): void {
  let failure: unknown;
  try {
    run();
  } catch (error) {
    failure = error;
  }
  expect(failure).toBeInstanceOf(WorkflowError);
  expect(failure).toMatchObject({name: 'WorkflowError', code, message, details: {parameter}});
}

type AcceptedCase = {
  name: string;
  spec: WorkflowParameterSpec;
  raw: unknown;
  expected: CatalogParameterValue;
};

const ACCEPTED_CASES: AcceptedCase[] = [
  {name: 'number', spec: makeSpec('number'), raw: 1.5, expected: 1.5},
  {name: 'trimmed decimal', spec: makeSpec('number'), raw: ' \t+1.5\n', expected: 1.5},
  {name: 'leading decimal point', spec: makeSpec('number'), raw: '-.5', expected: -0.5},
  {name: 'trailing decimal point', spec: makeSpec('number'), raw: '12.', expected: 12},
  {name: 'exponent', spec: makeSpec('number'), raw: '1.25e+2', expected: 125},
  {name: 'negative exponent', spec: makeSpec('number'), raw: '1E-2', expected: 0.01},
  {
    name: 'finite limit',
    spec: makeSpec('number'),
    raw: Number.MAX_VALUE,
    expected: Number.MAX_VALUE
  },
  {name: 'integer', spec: makeSpec('integer'), raw: 5, expected: 5},
  {name: 'whole decimal', spec: makeSpec('integer'), raw: ' 2.0 ', expected: 2},
  {name: 'whole exponent', spec: makeSpec('integer'), raw: '1e2', expected: 100},
  {name: 'true', spec: makeSpec('boolean'), raw: true, expected: true},
  {name: 'false', spec: makeSpec('boolean'), raw: false, expected: false},
  {name: 'enum', spec: makeSpec('enum', {options: ['a', 'b']}), raw: 'b', expected: 'b'},
  {name: 'enum empty option', spec: makeSpec('enum', {options: ['']}), raw: '', expected: ''},
  {name: 'list array', spec: makeSpec('number-list'), raw: [1, -2, 3.5], expected: [1, -2, 3.5]},
  {name: 'list text', spec: makeSpec('number-list'), raw: '1, -2, 3.5', expected: [1, -2, 3.5]},
  {name: 'list exponents', spec: makeSpec('number-list'), raw: ' 1e2 , .5 ', expected: [100, 0.5]},
  {name: 'one number list', spec: makeSpec('number-list'), raw: '1', expected: [1]},
  {
    name: 'eight numbers',
    spec: makeSpec('number-list'),
    raw: [1, 2, 3, 4, 5, 6, 7, 8],
    expected: [1, 2, 3, 4, 5, 6, 7, 8]
  },
  {
    name: 'custom list length and range',
    spec: makeSpec('number-list', {maximumLength: 2, minimum: 0, maximum: 100}),
    raw: '0,100',
    expected: [0, 100]
  },
  {
    name: 'column list whitespace',
    spec: makeSpec('column-list'),
    raw: ' first , , second, first ,',
    expected: 'first, second, first'
  },
  {name: 'empty column list', spec: makeSpec('column-list'), raw: ' , ', expected: ''},
  {
    name: 'unparsed expression',
    spec: makeSpec('expression'),
    raw: 'unclosed( + unknown',
    expected: 'unclosed( + unknown'
  },
  {
    name: 'multiline text',
    spec: makeSpec('text'),
    raw: '{\n  "a": 1\n}',
    expected: '{\n  "a": 1\n}'
  }
];

type RefusedCase = {
  name: string;
  spec: WorkflowParameterSpec;
  raw: unknown;
  problem: string;
};

const REFUSED_CASES: RefusedCase[] = [
  ...(['number', 'integer'] as const).flatMap(kind =>
    [
      ['', 'must be a number; got ""'],
      ['  ', 'must be a number; got "  "'],
      ['1e', 'must be a number; got "1e"'],
      ['12abc', 'must be a number; got "12abc"'],
      ['abc', 'must be a number; got "abc"'],
      ['1 2', 'must be a number; got "1 2"'],
      ['0x10', 'must be a number; got "0x10"'],
      ['1e309', 'must be a number; got "1e309"'],
      [NaN, 'must be a number; got NaN'],
      [Infinity, 'must be a number; got Infinity'],
      [-Infinity, 'must be a number; got -Infinity'],
      [null, 'must be a number; got null'],
      [undefined, 'must be a number; got undefined'],
      [true, 'must be a number; got true'],
      [[], 'must be a number; got an array'],
      [{}, 'must be a number; got an object']
    ].map(([raw, problem]) => ({
      name: `${kind}: ${String(raw)}`,
      spec: makeSpec(kind),
      raw,
      problem: problem as string
    }))
  ),
  {
    name: 'fractional integer',
    spec: makeSpec('integer'),
    raw: 1.5,
    problem: 'must be a whole number; got 1.5'
  },
  {
    name: 'fractional integer text',
    spec: makeSpec('integer'),
    raw: '1.5',
    problem: 'must be a whole number; got 1.5'
  },
  {
    name: 'both range limits',
    spec: makeSpec('number', {minimum: 0, maximum: 100}),
    raw: 250,
    problem: 'must be between 0 and 100; got 250'
  },
  {
    name: 'minimum',
    spec: makeSpec('number', {minimum: 0}),
    raw: '-1',
    problem: 'must be at least 0; got -1'
  },
  {
    name: 'maximum',
    spec: makeSpec('integer', {maximum: 10}),
    raw: 11,
    problem: 'must be at most 10; got 11'
  },
  {
    name: 'string boolean',
    spec: makeSpec('boolean'),
    raw: 'false',
    problem: 'must be a boolean; got "false"'
  },
  {name: 'numeric boolean', spec: makeSpec('boolean'), raw: 0, problem: 'must be a boolean; got 0'},
  {
    name: 'enum choice',
    spec: makeSpec('enum', {options: ['a', 'b', 'c']}),
    raw: 'x',
    problem: 'must be one of a, b, c; got "x"'
  },
  {
    name: 'enum string required',
    spec: makeSpec('enum', {options: ['1']}),
    raw: 1,
    problem: 'must be one of 1; got 1'
  },
  {
    name: 'enum untrimmed',
    spec: makeSpec('enum', {options: ['a']}),
    raw: ' a ',
    problem: 'must be one of a; got " a "'
  },
  {
    name: 'enum choices bounded',
    spec: makeSpec('enum', {options: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']}),
    raw: 'x',
    problem: 'must be one of a, b, c, d, e, f, g, h, …; got "x"'
  },
  {
    name: 'empty list',
    spec: makeSpec('number-list'),
    raw: [],
    problem: 'must have 1 to 8 numbers; got 0'
  },
  {
    name: 'empty list text',
    spec: makeSpec('number-list'),
    raw: ' \n',
    problem: 'must have 1 to 8 numbers; got 0'
  },
  {
    name: 'nine entries',
    spec: makeSpec('number-list'),
    raw: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    problem: 'must have 1 to 8 numbers; got 9'
  },
  {
    name: 'custom list limit',
    spec: makeSpec('number-list', {maximumLength: 2}),
    raw: '1,2,3',
    problem: 'must have 1 to 2 numbers; got 3'
  },
  {
    name: 'list range',
    spec: makeSpec('number-list', {minimum: 0, maximum: 100}),
    raw: [1, 500],
    problem: 'entry 2 must be between 0 and 100; got 500'
  },
  {
    name: 'list minimum',
    spec: makeSpec('number-list', {minimum: 0}),
    raw: '1,-1',
    problem: 'entry 2 must be at least 0; got -1'
  },
  {
    name: 'list maximum',
    spec: makeSpec('number-list', {maximum: 10}),
    raw: [11],
    problem: 'entry 1 must be at most 10; got 11'
  },
  {
    name: 'list invalid token',
    spec: makeSpec('number-list'),
    raw: '1,12abc',
    problem: 'entry 2 must be a number; got "12abc"'
  },
  {
    name: 'list incomplete exponent',
    spec: makeSpec('number-list'),
    raw: '1,1e',
    problem: 'entry 2 must be a number; got "1e"'
  },
  {
    name: 'list trailing delimiter',
    spec: makeSpec('number-list'),
    raw: '1,',
    problem: 'entry 2 must be a number; got ""'
  },
  {
    name: 'list missing entry',
    spec: makeSpec('number-list'),
    raw: '1,,2',
    problem: 'entry 2 must be a number; got ""'
  },
  {
    name: 'list nonfinite number',
    spec: makeSpec('number-list'),
    raw: [1, Infinity],
    problem: 'entry 2 must be a number; got Infinity'
  },
  {
    name: 'list NaN',
    spec: makeSpec('number-list'),
    raw: [NaN],
    problem: 'entry 1 must be a number; got NaN'
  },
  {
    name: 'list array numeric string',
    spec: makeSpec('number-list'),
    raw: [1, '2'],
    problem: 'entry 2 must be a number; got "2"'
  },
  {
    name: 'list nested array',
    spec: makeSpec('number-list'),
    raw: [[1]],
    problem: 'entry 1 must be a number; got an array'
  },
  {
    name: 'list sparse array',
    spec: makeSpec('number-list'),
    raw: Array(1),
    problem: 'entry 1 must be a number; got undefined'
  },
  {
    name: 'list scalar',
    spec: makeSpec('number-list'),
    raw: 1,
    problem: 'must be a list of numbers; got 1'
  },
  {
    name: 'truncated quoted value',
    spec: makeSpec('number'),
    raw: 'x'.repeat(100),
    problem: `must be a number; got "${'x'.repeat(39)}…"`
  },
  {
    name: 'escaped quoted value',
    spec: makeSpec('number'),
    raw: '"\n',
    problem: 'must be a number; got "\\"\\n"'
  }
];

describe('workflow parameters', () => {
  test.each(ACCEPTED_CASES)('accepts $name and coerces it', ({spec, raw, expected}) => {
    expect(getWorkflowParameterProblem(spec, raw)).toBeNull();
    expect(normalizeWorkflowParameters([spec], {value: raw}, CONTEXT)).toEqual({value: expected});
  });

  test.each(REFUSED_CASES)('refuses $name with the exact problem', ({spec, raw, problem}) => {
    expect(getWorkflowParameterProblem(spec, raw)).toBe(problem);
    expectParameterError(
      () => normalizeWorkflowParameters([spec], {value: raw}, CONTEXT),
      `Buffer points: Value ${problem}.`
    );
  });

  test.each(['number', 'integer'] as const)('%s includes both range boundaries', kind => {
    const spec = makeSpec(kind, {minimum: 0, maximum: 100});
    for (const raw of [0, 100, '0', '100']) {
      expect(getWorkflowParameterProblem(spec, raw)).toBeNull();
      expect(normalizeWorkflowParameters([spec], {value: raw}, CONTEXT)).toEqual({
        value: Number(raw)
      });
    }
    expect(getWorkflowParameterProblem(spec, -1)).toBe('must be between 0 and 100; got -1');
  });

  test.each([
    ['string', 64],
    ['column', 64],
    ['dataset', 64],
    ['column-list', 256],
    ['expression', 2000],
    ['text', WORKFLOW_LIMITS.maximumTextCharacters]
  ] as const)('%s enforces its default and custom character limits', (kind, maximumLength) => {
    const spec = makeSpec(kind);
    const boundary = 'x'.repeat(maximumLength);
    for (const raw of ['', '  x  ', boundary]) {
      expect(getWorkflowParameterProblem(spec, raw)).toBeNull();
      expect(normalizeWorkflowParameters([spec], {value: raw}, CONTEXT)).toEqual({
        value: kind === 'column-list' ? raw.trim() : raw
      });
    }
    const problem = `must be at most ${maximumLength} characters; got ${maximumLength + 1}`;
    expect(getWorkflowParameterProblem(spec, `${boundary}x`)).toBe(problem);
    expectParameterError(
      () => normalizeWorkflowParameters([spec], {value: `${boundary}x`}, CONTEXT),
      `Buffer points: Value ${problem}.`
    );
    for (const raw of [null, undefined, false, 1, [], {}]) {
      expect(getWorkflowParameterProblem(spec, raw)).toMatch(/^must be a string; got /);
    }
    const customSpec = makeSpec(kind, {maximumLength: 3});
    expect(getWorkflowParameterProblem(customSpec, 'abc')).toBeNull();
    expect(getWorkflowParameterProblem(customSpec, 'abcd')).toBe(
      'must be at most 3 characters; got 4'
    );
    expect(getWorkflowParameterProblem(makeSpec(kind, {maximumLength: 0}), '')).toBeNull();
    expect(getWorkflowParameterProblem(makeSpec(kind, {maximumLength: 0}), 'x')).toBe(
      'must be at most 0 characters; got 1'
    );
  });

  test('text reports lengths without echoing its contents', () => {
    const spec = makeSpec('text');
    const raw = 'x'.repeat(300000);
    expect(getWorkflowParameterProblem(spec, raw)).toBe(
      'must be at most 262144 characters; got 300000'
    );
    expectParameterError(
      () => normalizeWorkflowParameters([spec], {value: raw}, CONTEXT),
      'Buffer points: Value must be at most 262144 characters; got 300000.'
    );
  });

  test('column-list validation bounds the canonical representation', () => {
    const spec = makeSpec('column-list', {name: 'columns', label: 'Columns'});
    const raw = `${'a,'.repeat(127)}a`;
    expect(raw.length).toBe(255);
    expect(getWorkflowParameterProblem(spec, raw)).toBe('must be at most 256 characters; got 382');
    expectParameterError(
      () => normalizeWorkflowParameters([spec], {columns: raw}, CONTEXT),
      'Buffer points: Columns must be at most 256 characters; got 382.',
      'columns'
    );
    expectParameterError(
      () => normalizeWorkflowParameters([{...spec, defaultValue: raw}], {}, CONTEXT),
      'Buffer points: Columns must be at most 256 characters; got 382.',
      'columns'
    );
  });

  test.each([
    [256, `${'a,'.repeat(85)}a`, 256],
    [4, 'a,b', 4],
    [7, ' a,b, ', 4]
  ])('column lists revalidate and normalize idempotently at limit %s', (maximumLength, raw, length) => {
    const spec = makeSpec('column-list', {maximumLength});
    const normalized = normalizeWorkflowParameters([spec], {value: raw}, CONTEXT);
    expect(normalized['value']).toHaveLength(length);
    expect(normalized['value']).toBe(parseWorkflowColumnList(raw).join(', '));
    expect(getWorkflowParameterProblem(spec, normalized['value'])).toBeNull();
    expect(normalizeWorkflowParameters([spec], normalized, CONTEXT)).toEqual(normalized);
  });

  test.each([
    [256, `a${'a,'.repeat(85)}a`, 257],
    [3, 'a,b', 4],
    [3, ' a  ', 4]
  ])('column lists reject raw or canonical lengths over %s', (maximumLength, raw, length) => {
    const spec = makeSpec('column-list', {maximumLength});
    const problem = `must be at most ${maximumLength} characters; got ${length}`;
    expect(getWorkflowParameterProblem(spec, raw)).toBe(problem);
    expectParameterError(
      () => normalizeWorkflowParameters([spec], {value: raw}, CONTEXT),
      `Buffer points: Value ${problem}.`
    );
  });

  test.each([
    [8, 500000],
    [2, 3],
    [8, 8]
  ])('rejects list count before parsing: limit %s, commas %s', (maximumLength, commaCount) => {
    const spec = makeSpec('number-list', {maximumLength});
    const raw = `${'1,'.repeat(commaCount)}1`;
    const numericCheck = vi.spyOn(Number, 'isFinite');
    const split = vi.spyOn(String.prototype, 'split');
    let problem: string | null;
    let numericCheckCount: number;
    let splitCount: number;
    try {
      problem = getWorkflowParameterProblem(spec, raw);
      numericCheckCount = numericCheck.mock.calls.length;
      splitCount = split.mock.calls.length;
    } finally {
      numericCheck.mockRestore();
      split.mockRestore();
    }
    expect(problem).toBe(`must have 1 to ${maximumLength} numbers; got ${commaCount + 1}`);
    expect(numericCheckCount).toBe(0);
    expect(splitCount).toBe(0);
  });

  test('counts empty number-list entries before parsing', () => {
    const spec = makeSpec('number-list', {maximumLength: 2});
    expect(getWorkflowParameterProblem(spec, ',,')).toBe('must have 1 to 2 numbers; got 3');
    expect(getWorkflowParameterProblem(spec, '  ')).toBe('must have 1 to 2 numbers; got 0');
  });

  test('numeric record keys require consumers to iterate specs for spec order', () => {
    const specs = ['10', '2', 'value'].map(name => makeSpec('string', {name, defaultValue: name}));
    const result = normalizeWorkflowParameters(specs, {}, CONTEXT);
    // The Record contract cannot give ordinary objects a custom integer-key enumeration order.
    expect(Object.keys(result)).toEqual(['2', '10', 'value']);
    expect(Object.keys(getWorkflowParameterDefaults(specs))).toEqual(['2', '10', 'value']);
    expect(specs.map(spec => [spec.name, result[spec.name]])).toEqual([
      ['10', '10'],
      ['2', '2'],
      ['value', 'value']
    ]);
    const reloaded = JSON.parse(JSON.stringify(result)) as Record<string, CatalogParameterValue>;
    expect(specs.map(spec => reloaded[spec.name])).toEqual(['10', '2', 'value']);
  });

  test.each([
    [undefined, 'Distance'],
    ['meters', 'Distance (meters)'],
    ['crs', 'Distance (CRS units)'],
    ['', 'Distance']
  ])('formats the parameter label and unit %s', (unit, label) => {
    const spec = makeSpec('number', {label: 'Distance', unit, minimum: 0, maximum: 100000});
    expectParameterError(
      () => normalizeWorkflowParameters([spec], {value: -5}, CONTEXT),
      `Buffer points: ${label} must be between 0 and 100000; got -5.`
    );
  });

  test('fills defaults in spec order and retains explicit falsy values', () => {
    const specs = [
      makeSpec('number', {name: 'distance', defaultValue: 10}),
      makeSpec('boolean', {name: 'enabled', defaultValue: true}),
      makeSpec('string', {name: 'title', defaultValue: 'default'}),
      makeSpec('number-list', {name: 'breaks', defaultValue: [1, 2]})
    ];
    expect(getWorkflowParameterDefaults(specs)).toEqual({
      distance: 10,
      enabled: true,
      title: 'default',
      breaks: [1, 2]
    });
    const result = normalizeWorkflowParameters(
      specs,
      {title: '', enabled: false, distance: 0},
      CONTEXT
    );
    expect(result).toEqual({distance: 0, enabled: false, title: '', breaks: [1, 2]});
    expect(Object.keys(result)).toEqual(['distance', 'enabled', 'title', 'breaks']);
    expect(normalizeWorkflowParameters([], {}, CONTEXT)).toEqual({});
    expect(getWorkflowParameterDefaults([])).toEqual({});
  });

  test('validates and coerces defaults through the same rules', () => {
    expect(
      normalizeWorkflowParameters([makeSpec('number', {defaultValue: '2'})], {}, CONTEXT)
    ).toEqual({value: 2});
    expect(
      normalizeWorkflowParameters([makeSpec('column-list', {defaultValue: ' a , b,'})], {}, CONTEXT)
    ).toEqual({value: 'a, b'});
    expectParameterError(
      () =>
        normalizeWorkflowParameters(
          [makeSpec('number', {defaultValue: -1, minimum: 0})],
          {},
          CONTEXT
        ),
      'Buffer points: Value must be at least 0; got -1.'
    );
    expectParameterError(
      () =>
        normalizeWorkflowParameters(
          [makeSpec('number', {defaultValue: 2})],
          {value: undefined},
          CONTEXT
        ),
      'Buffer points: Value must be a number; got undefined.'
    );
  });

  test('rejects unknown names and records the parameter name', () => {
    expectParameterError(
      () => normalizeWorkflowParameters([makeSpec('number')], {extra: 1}, CONTEXT),
      'Buffer points: Unknown parameter "extra".',
      'extra',
      'unknown-parameter'
    );
    expectParameterError(
      () => normalizeWorkflowParameters([], {toString: 1}, CONTEXT),
      'Buffer points: Unknown parameter "toString".',
      'toString',
      'unknown-parameter'
    );
  });

  test('uses only own raw properties and safely preserves special parameter names', () => {
    const raw = Object.create({value: 99}) as Record<string, unknown>;
    expect(
      normalizeWorkflowParameters([makeSpec('number', {defaultValue: 2})], raw, CONTEXT)
    ).toEqual({value: 2});
    const specs = [makeSpec('string', {name: '__proto__', defaultValue: 'default'})];
    const result = normalizeWorkflowParameters(
      specs,
      JSON.parse('{"__proto__":"custom"}'),
      CONTEXT
    );
    expect(Object.keys(result)).toEqual(['__proto__']);
    expect(result['__proto__']).toBe('custom');
    expect(Object.getPrototypeOf(result)).toBe(Object.prototype);
    expect(getWorkflowParameterDefaults(specs)['__proto__']).toBe('default');
  });

  test('copies every list from raw values and defaults', () => {
    const rawList = Object.freeze([1, 2]);
    const defaultList = Object.freeze([3, 4]);
    const specs = [
      makeSpec('number-list', {name: 'first', defaultValue: defaultList}),
      makeSpec('number-list', {name: 'second', defaultValue: defaultList})
    ];
    const raw = Object.freeze({first: rawList});
    const first = normalizeWorkflowParameters(specs, raw, CONTEXT);
    const second = normalizeWorkflowParameters(specs, raw, CONTEXT);
    const defaults = getWorkflowParameterDefaults(specs);
    expect(first).toEqual({first: [1, 2], second: [3, 4]});
    expect(first['first']).not.toBe(rawList);
    expect(first['second']).not.toBe(defaultList);
    expect(first['first']).not.toBe(second['first']);
    expect(first['second']).not.toBe(second['second']);
    expect(defaults['first']).not.toBe(defaultList);
    expect(defaults['first']).not.toBe(defaults['second']);
    (first['first'] as number[]).push(9);
    (defaults['first'] as number[]).push(10);
    expect(rawList).toEqual([1, 2]);
    expect(defaultList).toEqual([3, 4]);
    expect(second).toEqual({first: [1, 2], second: [3, 4]});
  });

  test.each([
    ['number', 1.5, '1.5'],
    ['number', 1e-7, '1e-7'],
    ['integer', 2, '2'],
    ['boolean', true, 'true'],
    ['boolean', false, 'false'],
    ['number-list', [1, 2, 3], '1, 2, 3'],
    ['number-list', [], ''],
    ['enum', 'a', 'a'],
    ['string', ' x ', ' x '],
    ['column', 'value', 'value'],
    ['dataset', 'points', 'points'],
    ['column-list', 'a, b', 'a, b'],
    ['expression', 'x + 1', 'x + 1'],
    ['text', 'a\nb', 'a\nb']
  ] as const)('formats %s value %s', (kind, value, expected) => {
    expect(formatWorkflowParameterValue(makeSpec(kind), value)).toBe(expected);
  });

  test('visibility checks current values without coercion', () => {
    expect(isWorkflowParameterVisible(makeSpec('string'), {})).toBe(true);
    const spec = makeSpec('string', {
      visibleWhen: {parameter: 'mode', equals: ['a', 'b', 2, false]}
    });
    for (const value of ['a', 'b', 2, false]) {
      expect(isWorkflowParameterVisible(spec, {mode: value})).toBe(true);
    }
    for (const value of ['c', '2', 'false', true, 0]) {
      expect(isWorkflowParameterVisible(spec, {mode: value})).toBe(false);
    }
    expect(isWorkflowParameterVisible(spec, {})).toBe(false);
    expect(
      isWorkflowParameterVisible(
        makeSpec('string', {visibleWhen: {parameter: 'mode', equals: []}}),
        {mode: 'a'}
      )
    ).toBe(false);
  });

  test('list visibility survives defaults, normalization and JSON reloads', () => {
    const listValue = [1, 2];
    const source = makeSpec('number-list', {name: 'breaks', defaultValue: listValue});
    const dependent = makeSpec('string', {
      visibleWhen: {parameter: 'breaks', equals: [listValue]}
    });
    const values = [
      {breaks: listValue},
      getWorkflowParameterDefaults([source]),
      normalizeWorkflowParameters([source], {}, CONTEXT),
      JSON.parse('{"breaks":[1,2]}') as Record<string, CatalogParameterValue>
    ];
    for (const current of values) {
      expect(isWorkflowParameterVisible(dependent, current)).toBe(true);
    }
    expect(values[1]['breaks']).not.toBe(listValue);
    expect(values[2]['breaks']).not.toBe(listValue);
  });

  test.each([
    [[1, 2], true],
    [[2, 1], false],
    [[1], false],
    [[1, 2, 3], false],
    [[1, 3], false],
    ['1, 2', false],
    [1, false],
    [false, false]
  ] as const)('compares list visibility by ordered contents: %s', (value, expected) => {
    const spec = makeSpec('string', {visibleWhen: {parameter: 'breaks', equals: [[1, 2]]}});
    expect(isWorkflowParameterVisible(spec, {breaks: value})).toBe(expected);
  });

  test('visibility compares empty lists and checks each candidate', () => {
    const spec = makeSpec('string', {
      visibleWhen: {parameter: 'breaks', equals: [[3], [], [1, 2]]}
    });
    expect(isWorkflowParameterVisible(spec, {breaks: []})).toBe(true);
    expect(isWorkflowParameterVisible(spec, {breaks: [1, 2]})).toBe(true);
    expect(isWorkflowParameterVisible(spec, {breaks: [3]})).toBe(true);
    expect(isWorkflowParameterVisible(spec, {breaks: [4]})).toBe(false);
    expect(isWorkflowParameterVisible(spec, {})).toBe(false);
    expect(
      isWorkflowParameterVisible(
        {...spec, visibleWhen: {parameter: 'breaks', equals: []}},
        {breaks: []}
      )
    ).toBe(false);
  });

  test.each([
    ['', []],
    [' , , \n ', []],
    ['first', ['first']],
    [' first ,\tsecond\n, , third, ', ['first', 'second', 'third']],
    ['first,first', ['first', 'first']],
    ['space in name, a.b', ['space in name', 'a.b']],
    ['"a,b",c', ['"a', 'b"', 'c']]
  ])('parses column names from %s', (value, expected) => {
    expect(parseWorkflowColumnList(value as string)).toEqual(expected);
    expect(parseWorkflowColumnList(value as string)).not.toBe(
      parseWorkflowColumnList(value as string)
    );
  });
});
