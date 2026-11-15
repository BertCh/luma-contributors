// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  parseWorkflowExpression,
  WORKFLOW_EXPRESSION_FUNCTIONS,
  type WorkflowExpressionValue
} from '../../src/workflows/workflow-expression';
import {WorkflowError} from '../../src/workflows/workflow-types';

type Row = Readonly<Record<string, WorkflowExpressionValue>>;

function evaluate(source: string, row: Row = {}): WorkflowExpressionValue {
  return parseWorkflowExpression(source).evaluate(row);
}

function expectExpressionError(source: string, message: string): void {
  let caught: unknown;
  try {
    parseWorkflowExpression(source);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(WorkflowError);
  expect((caught as WorkflowError).code).toBe('parameter-out-of-range');
  expect((caught as WorkflowError).message).toBe(message);
}

describe('workflow expression precedence and associativity', () => {
  test.each([
    ['1 + 2 * 3', 7],
    ['(1 + 2) * 3', 9],
    ['7 - 3 - 2', 2],
    ['24 / 4 / 2', 3],
    ['17 % 5 * 2', 4],
    ['2 ^ 3 ^ 2', 512],
    ['-2 ^ 2', -4],
    ['(-2) ^ 2', 4],
    ['2 ^ -2', 0.25],
    ['-2 ^ -2', -0.25],
    ['--2', 2],
    ['!2 ^ 0', false],
    ['!0 + 1', null],
    ['2 * 3 ^ 2', 18],
    ['2 + 3 < 6 == true', true],
    ['false == 1 < 2', false],
    ['1 < 2 < 3', false],
    ['1 == 1 == true', true],
    ['1 != 2 && 3 >= 3', true],
    ['false || true && false', false],
    ['true || false ? 3 : 4', 3],
    ['false ? 1 : false ? 2 : 3', 3],
    ['true ? false ? 1 : 2 : 3', 2],
    ['false ? 1 : true ? 2 : 3', 2],
    ['true ? 4 : 1 / 0', 4]
  ])('%s evaluates to %s', (source, expected) => {
    expect(evaluate(source as string)).toBe(expected);
  });
});

describe('workflow expression literals and columns', () => {
  test.each([
    ['12', 12],
    ['.5', 0.5],
    ['5.', 5],
    ['1.25e+2', 125],
    ['2E-3', 0.002],
    ['true', true],
    ['false', false],
    ['null', null],
    ['"hello"', 'hello'],
    ["'hello'", 'hello'],
    [String.raw`'a\\b\'c\"d\n\t'`, 'a\\b\'c"d\n\t'],
    ['""', '']
  ])('reads %s', (source, expected) => {
    expect(evaluate(source as string)).toBe(expected);
  });

  test('keeps distinct columns in first-use order and preserves source', () => {
    const source = ' value + `population 2020` + value + _value2 + `true` ';
    const expression = parseWorkflowExpression(source);
    expect(expression.source).toBe(source);
    expect(expression.columns).toEqual(['value', 'population 2020', '_value2', 'true']);
    expect(expression.evaluate({value: 2, 'population 2020': 3, _value2: 4, true: 5})).toBe(16);
    expect(evaluate('missing')).toBeNull();
    expect(evaluate('`a\\`b`', {'a`b': 7})).toBe(7);
    expect(evaluate('`a.b`', {'a.b': 9})).toBe(9);
    expect(evaluate('`__proto__`', JSON.parse('{"__proto__": 12}') as Row)).toBe(12);
    expect(evaluate('`constructor`', {constructor: 13})).toBe(13);
    expect(evaluate('toString', {toString: 'own value'})).toBe('own value');
  });

  test('reuses one compiled expression across 100000 rows without retaining row values', () => {
    const expression = parseWorkflowExpression('if(value % 2 == 0, max(value, 10), value * 2)');
    const row = {value: 0};
    let total = 0;
    for (let index = 0; index < 100000; index++) {
      row.value = index;
      total += expression.evaluate(row) as number;
    }
    expect(total).toBe(7499950030);
    expect(expression.columns).toEqual(['value']);
    expect(expression.evaluate({})).toBeNull();
  });

  test('never reads prototype members', () => {
    const inherited = Object.create({
      constructor: 1,
      __proto__: null,
      toString: 'inherited',
      value: 8
    }) as Row;
    for (const name of ['constructor', '__proto__', 'toString', 'value']) {
      expect(evaluate(name, inherited)).toBeNull();
      expect(evaluate(name)).toBeNull();
    }
    expect(evaluate('value', Object.assign(Object.create(null), {value: 6}) as Row)).toBe(6);
  });

  test('treats unsupported or throwing row values as null without coercing objects', () => {
    const hostileValues: unknown[] = [undefined, {}, [], Symbol('value'), 2n, () => 1];
    for (const value of hostileValues) {
      const row = {value} as unknown as Row;
      expect(evaluate('value', row)).toBeNull();
      expect(evaluate('number(value)', row)).toBeNull();
      expect(evaluate('concat(value)', row)).toBe('');
    }
    const getterRow = Object.defineProperty({}, 'value', {
      get() {
        throw new Error('hostile');
      }
    }) as Row;
    const proxyRow = new Proxy(
      {},
      {
        getOwnPropertyDescriptor() {
          throw new Error('hostile');
        }
      }
    ) as Row;
    expect(evaluate('value', getterRow)).toBeNull();
    expect(evaluate('value', proxyRow)).toBeNull();
    const revoked = Proxy.revocable({}, {});
    revoked.revoke();
    expect(evaluate('value', revoked.proxy as Row)).toBeNull();
    for (const value of [NaN, Infinity, -Infinity]) {
      expect(evaluate('value + 1', {value})).toBeNull();
      expect(evaluate('number(value)', {value})).toBeNull();
    }
    expect(evaluate('value', null as unknown as Row)).toBeNull();
  });
});

describe('workflow expression functions', () => {
  test.each([
    ['abs(-4)', 4],
    ['sqrt(9)', 3],
    ['exp(0)', 1],
    ['ln(1)', 0],
    ['log10(100)', 2],
    ['floor(1.9)', 1],
    ['ceil(1.1)', 2],
    ['sign(-7)', -1],
    ['round(1.6)', 2],
    ['round(1.234, 2)', 1.23],
    ['round(1234, -2)', 1200],
    ['pow(2, 3)', 8],
    ['min(null, 3, -2, 4)', -2],
    ['max(null, 3, -2, 4)', 4],
    ['min(null, null)', null],
    ['max(null)', null],
    ['clamp(12, 0, 10)', 10],
    ['clamp(-2, 0, 10)', 0],
    ['clamp(5, 0, 10)', 5],
    ['if(true, 2, 3)', 2],
    ['if(false, 2, 3)', 3],
    ['isnull(null)', true],
    ['isnull(0)', false],
    ['coalesce(null, null, 0, 2)', 0],
    ['coalesce(null, false, 1)', false],
    ['coalesce(null, "", 1)', ''],
    ['coalesce(null, null)', null],
    ['coalesce()', null],
    ['number("12.5")', 12.5],
    ['number("x")', null],
    ['number(true)', 1],
    ['number(false)', 0],
    ['number(12)', 12],
    ['number(null)', null],
    ['number("Infinity")', null],
    ['number("1e999")', null],
    ['string(null)', null],
    ['string(12.5)', '12.5'],
    ['string(false)', 'false'],
    ['lower("AbC")', 'abc'],
    ['upper("AbC")', 'ABC'],
    ['length("abc")', 3],
    ['length("😀")', 2],
    ['concat("a", 12, true, null, "b")', 'a12trueb'],
    ['concat()', ''],
    ['contains("abc", "b")', true],
    ['contains("abc", "B")', false],
    ['startswith("abc", "ab")', true],
    ['startswith("abc", "b")', false],
    ['contains("abc", "")', true],
    ['startswith("abc", "")', true]
  ])('%s evaluates to %s', (source, expected) => {
    expect(evaluate(source as string)).toBe(expected);
  });

  test('exports every function name sorted, once', () => {
    const expected = [
      'abs',
      'ceil',
      'clamp',
      'coalesce',
      'concat',
      'contains',
      'exp',
      'floor',
      'if',
      'isnull',
      'length',
      'ln',
      'log10',
      'lower',
      'max',
      'min',
      'number',
      'pow',
      'round',
      'sign',
      'sqrt',
      'startswith',
      'string',
      'upper'
    ];
    expect(WORKFLOW_EXPRESSION_FUNCTIONS).toEqual(expected);
  });

  test('every fixed arity is checked at parse time, including unselected branches', () => {
    const unaryNames = [
      'abs',
      'sqrt',
      'exp',
      'ln',
      'log10',
      'floor',
      'ceil',
      'sign',
      'isnull',
      'number',
      'string',
      'lower',
      'upper',
      'length'
    ];
    for (const name of unaryNames) {
      expectExpressionError(
        `${name}()`,
        `Expression: ${name} takes 1 argument; got 0 at character 1`
      );
      expectExpressionError(
        `${name}(1, 2)`,
        `Expression: ${name} takes 1 argument; got 2 at character 1`
      );
    }
    for (const name of ['pow', 'contains', 'startswith']) {
      expectExpressionError(
        `${name}(1)`,
        `Expression: ${name} takes 2 arguments; got 1 at character 1`
      );
      expectExpressionError(
        `${name}(1, 2, 3)`,
        `Expression: ${name} takes 2 arguments; got 3 at character 1`
      );
    }
    for (const name of ['if', 'clamp']) {
      expectExpressionError(
        `${name}(1, 2)`,
        `Expression: ${name} takes 3 arguments; got 2 at character 1`
      );
      expectExpressionError(
        `${name}(1, 2, 3, 4)`,
        `Expression: ${name} takes 3 arguments; got 4 at character 1`
      );
    }
    for (const name of ['min', 'max']) {
      expectExpressionError(
        `${name}()`,
        `Expression: ${name} takes 1 or more arguments; got 0 at character 1`
      );
    }
    expectExpressionError(
      'round()',
      'Expression: round takes 1 or 2 arguments; got 0 at character 1'
    );
    expectExpressionError(
      'round(1, 2, 3)',
      'Expression: round takes 1 or 2 arguments; got 3 at character 1'
    );
    expectExpressionError(
      'true ? 1 : sqrt()',
      'Expression: sqrt takes 1 argument; got 0 at character 12'
    );
  });
});

describe('workflow expression null, type and truthiness semantics', () => {
  test('arithmetic and numeric functions accept numbers only', () => {
    for (const value of ['null', 'true', '"2"']) {
      for (const operator of ['+', '-', '*', '/', '%', '^']) {
        expect(evaluate(`${value} ${operator} 2`)).toBeNull();
        expect(evaluate(`2 ${operator} ${value}`)).toBeNull();
      }
      expect(evaluate(`-${value}`)).toBeNull();
      for (const name of ['abs', 'sqrt', 'exp', 'ln', 'log10', 'floor', 'ceil', 'sign', 'round']) {
        expect(evaluate(`${name}(${value})`)).toBeNull();
      }
      expect(evaluate(`pow(2, ${value})`)).toBeNull();
      expect(evaluate(`round(2, ${value})`)).toBeNull();
      for (const argumentsText of [`${value}, 0, 2`, `1, ${value}, 2`, `1, 0, ${value}`]) {
        expect(evaluate(`clamp(${argumentsText})`)).toBeNull();
      }
      if (value !== 'null') {
        expect(evaluate(`min(1, ${value})`)).toBeNull();
        expect(evaluate(`max(${value}, 1)`)).toBeNull();
      }
    }
  });

  test.each([
    '1 / 0',
    '0 / 0',
    '1 % 0',
    '1 / -0',
    '1e308 * 2',
    '(-1)^.5',
    'sqrt(-1)',
    'ln(0)',
    'exp(1000)',
    'pow(10, 1000)'
  ])('returns null for non-finite arithmetic: %s', source => {
    expect(evaluate(source)).toBeNull();
  });

  test('equality is strict and relational comparison only accepts matching number or string types', () => {
    expect(evaluate('null == null')).toBe(true);
    expect(evaluate('null != null')).toBe(false);
    expect(evaluate('"2" == 2')).toBe(false);
    expect(evaluate('true != 1')).toBe(true);
    expect(evaluate('"same" == "same"')).toBe(true);
    for (const operator of ['<', '<=', '>', '>=']) {
      for (const [left, right] of [
        ['null', 'null'],
        ['true', 'true'],
        ['2', '"2"'],
        ['"2"', '2']
      ]) {
        expect(evaluate(`${left} ${operator} ${right}`)).toBe(false);
      }
    }
    expect(evaluate('2 < 10')).toBe(true);
    expect(evaluate('2 <= 2')).toBe(true);
    expect(evaluate('3 > 2')).toBe(true);
    expect(evaluate('3 >= 3')).toBe(true);
    expect(evaluate('"10" < "2"')).toBe(true);
    expect(evaluate('"Z" < "a"')).toBe(true);
    expect(evaluate('"a" <= "a"')).toBe(true);
    expect(evaluate('"b" > "a"')).toBe(true);
    expect(evaluate('"b" >= "b"')).toBe(true);
  });

  test('all condition forms use truthiness and boolean operators return booleans', () => {
    for (const value of ['null', 'false', '0', '""']) {
      expect(evaluate(`!${value}`)).toBe(true);
      expect(evaluate(`${value} && 42`)).toBe(false);
      expect(evaluate(`${value} || 42`)).toBe(true);
      expect(evaluate(`${value} ? 1 : 2`)).toBe(2);
      expect(evaluate(`if(${value}, 1, 2)`)).toBe(2);
    }
    for (const value of ['true', '-1', '"false"']) {
      expect(evaluate(`!(${value})`)).toBe(false);
      expect(evaluate(`${value} && 42`)).toBe(true);
      expect(evaluate(`${value} || 0`)).toBe(true);
      expect(evaluate(`${value} ? 1 : 2`)).toBe(1);
      expect(evaluate(`if(${value}, 1, 2)`)).toBe(1);
    }
  });

  test('short-circuits boolean, conditional, if and coalesce branches', () => {
    let reads = 0;
    const row = Object.defineProperty({}, 'unread', {
      get() {
        reads++;
        return 42;
      }
    }) as Row;
    const sources = [
      'false && unread',
      'true || unread',
      'true ? 1 : unread',
      'false ? unread : 1',
      'if(true, 1, unread)',
      'if(false, unread, 1)',
      'coalesce(0, unread)'
    ];
    for (const source of sources) parseWorkflowExpression(source).evaluate(row);
    expect(reads).toBe(0);
    expect(evaluate('false || unread', row)).toBe(true);
    expect(evaluate('true && unread', row)).toBe(true);
    expect(reads).toBe(2);
  });

  test('string functions refuse non-string inputs', () => {
    for (const value of ['null', 'true', '12']) {
      for (const name of ['length', 'lower', 'upper']) {
        expect(evaluate(`${name}(${value})`)).toBeNull();
      }
      for (const name of ['contains', 'startswith']) {
        expect(evaluate(`${name}(${value}, "x")`)).toBe(false);
        expect(evaluate(`${name}("x", ${value})`)).toBe(false);
      }
    }
  });
});

describe('workflow expression syntax errors and limits', () => {
  test.each([
    ['', 'expected a literal, column, function or "("', 1],
    ['   ', 'expected a literal, column, function or "("', 4],
    ['1 +', 'expected a literal, column, function or "("', 4],
    ['(1 + 2', 'expected ")"', 7],
    ['min(1, 2', 'expected ")"', 9],
    ['min(1,)', 'expected a literal, column, function or "("', 7],
    ['true ? 1', 'expected ":"', 9],
    ['1 2', 'expected end of expression', 3],
    ['1)', 'expected end of expression', 2],
    ['sqr(2)', 'unknown function "sqr"', 1],
    ['1 + Sqrt(2)', 'unknown function "Sqrt"', 5],
    ['constructor(2)', 'unknown function "constructor"', 1],
    ['1 === 2', 'expected a token; got "="', 5],
    ['value.member', 'expected a token; got "."', 6],
    ['value[0]', 'expected a token; got "["', 6],
    ['"unterminated', 'expected """', 14],
    ['`unterminated', 'expected "`"', 14],
    [String.raw`'bad\r'`, 'expected a valid escape', 5],
    ['1e+', 'expected exponent digits', 4],
    ['1e999', 'expected a finite decimal number', 1]
  ])('reports the exact position for %s', (source, message, position) => {
    expectExpressionError(String(source), `Expression: ${message} at character ${position}`);
  });

  test('accepts each named limit and refuses one above it', () => {
    expect(evaluate(`${' '.repeat(1999)}1`)).toBe(1);
    expectExpressionError(
      `${' '.repeat(2000)}1`,
      'Expression: source length must not exceed 2000 characters at character 2001'
    );
    expect(evaluate(`${'('.repeat(64)}1${')'.repeat(64)}`)).toBe(1);
    expectExpressionError(
      `${'('.repeat(65)}1${')'.repeat(65)}`,
      'Expression: nesting depth must not exceed 64 at character 66'
    );
    expect(evaluate(`${'!'.repeat(64)}0`)).toBe(false);
    expectExpressionError(
      `${'!'.repeat(65)}0`,
      'Expression: nesting depth must not exceed 64 at character 66'
    );
    expect(evaluate(Array.from({length: 65}, () => '1').join('^'))).toBe(1);
    expectExpressionError(
      Array.from({length: 66}, () => '1').join('^'),
      'Expression: nesting depth must not exceed 64 at character 131'
    );
    expect(evaluate(`${'abs('.repeat(64)}1${')'.repeat(64)}`)).toBe(1);
    expectExpressionError(
      `${'abs('.repeat(65)}1${')'.repeat(65)}`,
      'Expression: nesting depth must not exceed 64 at character 261'
    );
    const maximumTokens = `-${Array.from({length: 250}, () => '1').join('+')}`;
    expect(evaluate(maximumTokens)).toBe(248);
    expectExpressionError(
      `${maximumTokens}+1`,
      'Expression: token count must not exceed 500 at character 501'
    );
  });
});

type GeneratedExpression =
  | {kind: 'literal'; value: WorkflowExpressionValue}
  | {kind: 'column'; name: string}
  | {kind: 'unary'; operator: '-' | '!'; operand: GeneratedExpression}
  | {kind: 'binary'; operator: string; left: GeneratedExpression; right: GeneratedExpression}
  | {
      kind: 'conditional';
      condition: GeneratedExpression;
      consequent: GeneratedExpression;
      alternate: GeneratedExpression;
    }
  | {kind: 'call'; name: 'abs' | 'coalesce' | 'concat'; argumentsList: GeneratedExpression[]};

function renderGeneratedExpression(expression: GeneratedExpression): string {
  switch (expression.kind) {
    case 'literal':
      return JSON.stringify(expression.value);
    case 'column':
      return `\`${expression.name}\``;
    case 'unary':
      return `(${expression.operator}${renderGeneratedExpression(expression.operand)})`;
    case 'binary':
      return `(${renderGeneratedExpression(expression.left)} ${expression.operator} ${renderGeneratedExpression(expression.right)})`;
    case 'conditional':
      return `(${renderGeneratedExpression(expression.condition)} ? ${renderGeneratedExpression(expression.consequent)} : ${renderGeneratedExpression(expression.alternate)})`;
    case 'call':
      return `${expression.name}(${expression.argumentsList.map(renderGeneratedExpression).join(', ')})`;
  }
}

// This interpreter consumes the generated tree directly; it never uses the production parser or closures.
function interpretGeneratedExpression(
  expression: GeneratedExpression,
  row: Row
): WorkflowExpressionValue {
  switch (expression.kind) {
    case 'literal':
      return expression.value;
    case 'column':
      return Object.hasOwn(row, expression.name) ? row[expression.name] : null;
    case 'unary': {
      const value = interpretGeneratedExpression(expression.operand, row);
      return expression.operator === '!' ? !value : typeof value === 'number' ? -value : null;
    }
    case 'conditional':
      return interpretGeneratedExpression(
        interpretGeneratedExpression(expression.condition, row)
          ? expression.consequent
          : expression.alternate,
        row
      );
    case 'call': {
      const values = expression.argumentsList.map(argument =>
        interpretGeneratedExpression(argument, row)
      );
      if (expression.name === 'coalesce') return values.find(value => value !== null) ?? null;
      if (expression.name === 'concat') return values.filter(value => value !== null).join('');
      return typeof values[0] === 'number' ? Math.abs(values[0]) : null;
    }
    case 'binary': {
      const left = interpretGeneratedExpression(expression.left, row);
      if (expression.operator === '&&')
        return Boolean(left) && Boolean(interpretGeneratedExpression(expression.right, row));
      if (expression.operator === '||')
        return Boolean(left) || Boolean(interpretGeneratedExpression(expression.right, row));
      const right = interpretGeneratedExpression(expression.right, row);
      if (expression.operator === '==') return left === right;
      if (expression.operator === '!=') return left !== right;
      if (['<', '<=', '>', '>='].includes(expression.operator)) {
        if (
          typeof left !== typeof right ||
          (typeof left !== 'string' && typeof left !== 'number') ||
          (typeof right !== 'string' && typeof right !== 'number')
        )
          return false;
        switch (expression.operator) {
          case '<':
            return left < right;
          case '<=':
            return left <= right;
          case '>':
            return left > right;
          default:
            return left >= right;
        }
      }
      if (typeof left !== 'number' || typeof right !== 'number') return null;
      let result: number;
      switch (expression.operator) {
        case '+':
          result = left + right;
          break;
        case '-':
          result = left - right;
          break;
        case '*':
          result = left * right;
          break;
        case '/':
          result = left / right;
          break;
        case '%':
          result = left % right;
          break;
        default:
          result = left ** right;
      }
      return Number.isFinite(result) ? result : null;
    }
  }
}

test('200 deterministic grammar-generated expressions match an independent value oracle', () => {
  let state = 871;
  function nextInteger(maximum: number): number {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state % maximum;
  }
  const leaves: GeneratedExpression[] = [
    ...[null, true, false, 0, 2, 3.5, 'text', ''].map(value => ({kind: 'literal' as const, value})),
    {kind: 'column', name: 'value'},
    {kind: 'column', name: 'population 2020'},
    {kind: 'column', name: 'missing'}
  ];
  const operators = ['+', '-', '*', '/', '%', '^', '==', '!=', '<', '<=', '>', '>=', '&&', '||'];
  function createExpression(depth: number): GeneratedExpression {
    if (depth === 0) return leaves[nextInteger(leaves.length)];
    switch (nextInteger(5)) {
      case 0:
        return {
          kind: 'binary',
          left: createExpression(depth - 1),
          operator: operators[nextInteger(operators.length)],
          right: createExpression(depth - 1)
        };
      case 1:
        return {
          kind: 'unary',
          operator: nextInteger(2) ? '-' : '!',
          operand: createExpression(depth - 1)
        };
      case 2:
        return {
          kind: 'conditional',
          condition: createExpression(depth - 1),
          consequent: createExpression(depth - 1),
          alternate: createExpression(depth - 1)
        };
      case 3:
        return {
          kind: 'call',
          name: nextInteger(2) ? 'coalesce' : 'concat',
          argumentsList: [createExpression(depth - 1), createExpression(depth - 1)]
        };
      default:
        return {kind: 'call', name: 'abs', argumentsList: [createExpression(depth - 1)]};
    }
  }
  for (let index = 0; index < 200; index++) {
    const tree = createExpression(3);
    const source = renderGeneratedExpression(tree);
    const expression = parseWorkflowExpression(source);
    const rows: Row[] = [
      {value: index - 100, 'population 2020': index % 5},
      {value: 'text', 'population 2020': null},
      {}
    ];
    for (const row of rows) {
      expect(expression.evaluate(row), source).toBe(interpretGeneratedExpression(tree, row));
    }
  }
});

test('F1 preserves exact large integers when rounding to decimal places', () => {
  expect(evaluate('round(100000000000000000000, 3)')).toBe(100000000000000000000);
});

test('F2 returns null instead of throwing when concatenation exceeds engine string limits', () => {
  const value = 'x'.repeat(8 * 1024 * 1024);
  const source = `concat(${Array.from({length: 65}, () => 'value').join(',')})`;
  const expression = parseWorkflowExpression(source);
  expect(expression.evaluate({value})).toBeNull();
});

test('rounding integers to nonnegative decimal places preserves their exact value', () => {
  const expression = parseWorkflowExpression('round(value, digits)');
  for (const value of [
    0,
    -0,
    1,
    -1,
    1234,
    -1234,
    Number.MAX_SAFE_INTEGER,
    -Number.MAX_SAFE_INTEGER,
    1e20,
    -1e20,
    1e100,
    -1e100,
    Number.MAX_VALUE,
    -Number.MAX_VALUE
  ]) {
    for (const digits of [0, 1, 3, 12, 100, 308, 400]) {
      expect(expression.evaluate({value, digits})).toBe(value);
    }
  }
});

test.each([
  [1.23, 2, 1.23],
  [-1.23, 2, -1.23],
  [0.30000000000000004, 17, 0.30000000000000004],
  [1.234, 400, 1.234],
  [1.234e-308, 309, 1.2e-308],
  [-1.234e-308, 309, -1.2e-308],
  [Number.MIN_VALUE, 323, 0],
  [Number.MIN_VALUE, 324, Number.MIN_VALUE],
  [23000000000000004, -1, 23000000000000000],
  [-23000000000000004, -1, -23000000000000000],
  [125, -1, 130],
  [-125, -1, -120],
  [-1, -1, -0],
  [1234, -400, 0],
  [-1234, -400, -0]
])('round(%s, %s) preserves precision and tie direction', (value, digits, expected) => {
  expect(evaluate('round(value, digits)', {value, digits})).toBe(expected);
});

test('generated strings respect the construction limit and remain usable below it', () => {
  const maximumCharacters = 0x0ffffff0;
  const value = 'x'.repeat(maximumCharacters);
  expect(evaluate('length(concat(value))', {value})).toBe(maximumCharacters);
  expect(evaluate('concat(value, "x")', {value})).toBeNull();
  expect(evaluate('concat(value, true)', {value})).toBeNull();
  expect(evaluate('coalesce(concat(value, "x"), "fallback")', {value})).toBe('fallback');
  expect(evaluate('concat("a", value, "b")', {value: 'x'.repeat(1024 * 1024)})).toBe(
    `a${'x'.repeat(1024 * 1024)}b`
  );
});

test('string capacity failures do not leak from case conversion', () => {
  // Force the host failure without allocating and flattening hundreds of megabytes in the test.
  const lowerCase = String.prototype.toLowerCase;
  const upperCase = String.prototype.toUpperCase;
  try {
    String.prototype.toLowerCase = () => {
      throw new RangeError('Invalid string length');
    };
    String.prototype.toUpperCase = () => {
      throw new RangeError('Invalid string length');
    };
    expect(evaluate('lower(value)', {value: 'text'})).toBeNull();
    expect(evaluate('upper(value)', {value: 'text'})).toBeNull();
    expect(evaluate('coalesce(upper(value), "fallback")', {value: 'text'})).toBe('fallback');
  } finally {
    String.prototype.toLowerCase = lowerCase;
    String.prototype.toUpperCase = upperCase;
  }
});
