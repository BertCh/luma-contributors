// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WorkflowError} from './workflow-types';

export type WorkflowExpressionValue = string | number | boolean | null;

export type WorkflowExpression = {
  source: string;
  /** Column names the expression reads, in first-use order. */
  columns: readonly string[];
  evaluate(row: Readonly<Record<string, WorkflowExpressionValue>>): WorkflowExpressionValue;
};

/** Function names the expression language accepts, for the form's help text. */
export const WORKFLOW_EXPRESSION_FUNCTIONS: readonly string[] = Object.freeze([
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
]);

const MAXIMUM_SOURCE_CHARACTERS = 2000;
const MAXIMUM_NESTING_DEPTH = 64;
const MAXIMUM_TOKENS = 500;
// Bound generated strings consistently across hosts, independently of the input row's size.
const MAXIMUM_GENERATED_STRING_CHARACTERS = 0x0ffffff0;

type Row = Readonly<Record<string, WorkflowExpressionValue>>;
type Evaluate = (row: Row) => WorkflowExpressionValue;
type Token = {
  kind: 'number' | 'string' | 'identifier' | 'column' | 'operator' | 'end';
  text: string;
  position: number;
};

function failExpression(expected: string, position: number): never {
  throw new WorkflowError(
    'parameter-out-of-range',
    `Expression: ${expected} at character ${position + 1}`
  );
}

function tokenizeExpression(source: string): Token[] {
  const tokens: Token[] = [];
  let position = 0;
  while (position < source.length) {
    if (/\s/.test(source[position])) {
      position++;
      continue;
    }
    if (tokens.length === MAXIMUM_TOKENS) {
      failExpression(`token count must not exceed ${MAXIMUM_TOKENS}`, position);
    }
    const start = position;
    const character = source[position];
    if (character === '"' || character === "'" || character === '`') {
      position++;
      let text = '';
      while (position < source.length && source[position] !== character) {
        if (source[position] === '\\') {
          const escapePosition = position++;
          const escaped = source[position++];
          switch (escaped) {
            case '\\':
            case "'":
            case '"':
              text += escaped;
              break;
            case '`':
              if (character !== '`') failExpression('expected a valid escape', escapePosition);
              text += escaped;
              break;
            case 'n':
              text += '\n';
              break;
            case 't':
              text += '\t';
              break;
            default:
              failExpression('expected a valid escape', escapePosition);
          }
        } else {
          text += source[position++];
        }
      }
      if (source[position] !== character) failExpression(`expected "${character}"`, position);
      position++;
      tokens.push({kind: character === '`' ? 'column' : 'string', text, position: start});
      continue;
    }
    const numberMatch = /^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/.exec(
      source.slice(position)
    );
    if (numberMatch) {
      position += numberMatch[0].length;
      if (!Number.isFinite(Number(numberMatch[0]))) {
        failExpression('expected a finite decimal number', start);
      }
      if (source[position] === 'e' || source[position] === 'E') {
        const digitPosition =
          position + (source[position + 1] === '+' || source[position + 1] === '-' ? 2 : 1);
        failExpression('expected exponent digits', digitPosition);
      }
      tokens.push({kind: 'number', text: numberMatch[0], position: start});
      continue;
    }
    const identifierMatch = /^[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(position));
    if (identifierMatch) {
      position += identifierMatch[0].length;
      tokens.push({kind: 'identifier', text: identifierMatch[0], position: start});
      continue;
    }
    const pair = source.slice(position, position + 2);
    if (['||', '&&', '==', '!=', '<=', '>='].includes(pair)) {
      position += 2;
      tokens.push({kind: 'operator', text: pair, position: start});
    } else if ('?:<>+-*/%!^(),'.includes(character)) {
      position++;
      tokens.push({kind: 'operator', text: character, position: start});
    } else {
      failExpression(`expected a token; got "${character}"`, position);
    }
  }
  tokens.push({kind: 'end', text: '', position: source.length});
  return tokens;
}

function finiteNumber(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

function roundNumber(value: number, digits: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(digits)) return NaN;
  if (digits === 0) return Math.round(value);
  if (digits > 0) {
    // Scaling an integer can discard significant bits even when the requested rounding is a no-op.
    if (Number.isInteger(digits) && Number.isInteger(value)) return value;
    const quantum = 10 ** -digits;
    // A quantum below the representable spacing cannot change the stored value.
    if (quantum === 0 || quantum < (Math.abs(value) * Number.EPSILON) / 2) return value;
    const scale = 10 ** digits;
    if (!Number.isFinite(scale)) {
      // Tiny finite values can still need rounding when the scale itself overflows.
      const remainingScale = 10 ** (digits - 308);
      return Math.round(value * 1e308 * remainingScale) / remainingScale / 1e308;
    }
    const scaled = value * scale;
    return Number.isInteger(scaled) ? value : Math.round(scaled) / scale;
  }
  // Remainders avoid losing the rounding decision when a large quotient rounds to a half-integer.
  const quantum = 10 ** -digits;
  if (!Number.isFinite(quantum)) return value < 0 ? -0 : 0;
  const remainder = value % quantum;
  if (remainder === 0) return value;
  let result = value - remainder;
  if (remainder >= quantum / 2) result += quantum;
  else if (remainder < -quantum / 2) result -= quantum;
  return result === 0 && value < 0 ? -0 : result;
}

function compileNumericUnary(operand: Evaluate, operation: (value: number) => number): Evaluate {
  return row => {
    const value = operand(row);
    return typeof value === 'number' ? finiteNumber(operation(value)) : null;
  };
}

function compileNumericBinary(
  left: Evaluate,
  right: Evaluate,
  operation: (left: number, right: number) => number
): Evaluate {
  return row => {
    const leftValue = left(row);
    const rightValue = right(row);
    return typeof leftValue === 'number' && typeof rightValue === 'number'
      ? finiteNumber(operation(leftValue, rightValue))
      : null;
  };
}

function compileBinary(operator: string, left: Evaluate, right: Evaluate): Evaluate {
  switch (operator) {
    case '||':
      return row => Boolean(left(row)) || Boolean(right(row));
    case '&&':
      return row => Boolean(left(row)) && Boolean(right(row));
    case '==':
      return row => left(row) === right(row);
    case '!=':
      return row => left(row) !== right(row);
    case '+':
      return compileNumericBinary(left, right, (first, second) => first + second);
    case '-':
      return compileNumericBinary(left, right, (first, second) => first - second);
    case '*':
      return compileNumericBinary(left, right, (first, second) => first * second);
    case '/':
      return compileNumericBinary(left, right, (first, second) =>
        second === 0 ? NaN : first / second
      );
    case '%':
      return compileNumericBinary(left, right, (first, second) =>
        second === 0 ? NaN : first % second
      );
    case '^':
      return compileNumericBinary(left, right, Math.pow);
    default:
      return row => {
        const first = left(row);
        const second = right(row);
        if (
          !(
            (typeof first === 'number' && typeof second === 'number') ||
            (typeof first === 'string' && typeof second === 'string')
          )
        ) {
          return false;
        }
        switch (operator) {
          case '<':
            return first < second;
          case '<=':
            return first <= second;
          case '>':
            return first > second;
          default:
            return first >= second;
        }
      };
  }
}

function compileFunction(name: string, argumentsList: readonly Evaluate[]): Evaluate {
  const [first, second, third] = argumentsList;
  switch (name) {
    case 'abs':
      return compileNumericUnary(first, Math.abs);
    case 'sqrt':
      return compileNumericUnary(first, Math.sqrt);
    case 'exp':
      return compileNumericUnary(first, Math.exp);
    case 'ln':
      return compileNumericUnary(first, Math.log);
    case 'log10':
      return compileNumericUnary(first, Math.log10);
    case 'floor':
      return compileNumericUnary(first, Math.floor);
    case 'ceil':
      return compileNumericUnary(first, Math.ceil);
    case 'sign':
      return compileNumericUnary(first, Math.sign);
    case 'pow':
      return compileNumericBinary(first, second, Math.pow);
    case 'round':
      if (!second) return compileNumericUnary(first, Math.round);
      return compileNumericBinary(first, second, roundNumber);
    case 'clamp':
      return row => {
        const value = first(row);
        const low = second(row);
        const high = third(row);
        return typeof value === 'number' && typeof low === 'number' && typeof high === 'number'
          ? finiteNumber(Math.min(Math.max(value, low), high))
          : null;
      };
    case 'min':
    case 'max':
      return row => {
        let result: number | null = null;
        for (let index = 0; index < argumentsList.length; index++) {
          const value = argumentsList[index](row);
          if (value === null) continue;
          if (typeof value !== 'number') return null;
          result =
            result === null
              ? value
              : name === 'min'
                ? Math.min(result, value)
                : Math.max(result, value);
        }
        return result === null ? null : finiteNumber(result);
      };
    case 'if':
      return row => (first(row) ? second(row) : third(row));
    case 'isnull':
      return row => first(row) === null;
    case 'coalesce':
      return row => {
        for (let index = 0; index < argumentsList.length; index++) {
          const value = argumentsList[index](row);
          if (value !== null) return value;
        }
        return null;
      };
    case 'number':
      return row => {
        const value = first(row);
        return value === null ? null : finiteNumber(Number(value));
      };
    case 'string':
      return row => {
        const value = first(row);
        return value === null ? null : String(value);
      };
    case 'lower':
    case 'upper':
    case 'length':
      return row => {
        const value = first(row);
        if (typeof value !== 'string') return null;
        if (name === 'length') return value.length;
        if (value.length > MAXIMUM_GENERATED_STRING_CHARACTERS) return null;
        try {
          const result = name === 'lower' ? value.toLowerCase() : value.toUpperCase();
          return result.length <= MAXIMUM_GENERATED_STRING_CHARACTERS ? result : null;
        } catch {
          // Unicode case conversion can expand a valid input beyond the host's string capacity.
          return null;
        }
      };
    case 'concat':
      return row => {
        let result = '';
        for (let index = 0; index < argumentsList.length; index++) {
          const value = argumentsList[index](row);
          if (value === null) continue;
          const text = String(value);
          if (text.length > MAXIMUM_GENERATED_STRING_CHARACTERS - result.length) return null;
          try {
            result += text;
          } catch {
            // A host may impose a smaller string capacity than the shared construction limit.
            return null;
          }
        }
        return result;
      };
    case 'contains':
    case 'startswith':
      return row => {
        const text = first(row);
        const part = second(row);
        return typeof text === 'string' && typeof part === 'string'
          ? name === 'contains'
            ? text.includes(part)
            : text.startsWith(part)
          : false;
      };
    default:
      // The parser checks the function name before compiling its arguments.
      throw new WorkflowError('parameter-out-of-range', `Expression: unknown function "${name}"`);
  }
}

function getPrecedence(operator: string): number {
  switch (operator) {
    case '||':
      return 1;
    case '&&':
      return 2;
    case '==':
    case '!=':
      return 3;
    case '<':
    case '<=':
    case '>':
    case '>=':
      return 4;
    case '+':
    case '-':
      return 5;
    case '*':
    case '/':
    case '%':
      return 6;
    default:
      return 0;
  }
}

class ExpressionParser {
  readonly columns: string[] = [];
  private readonly columnNames = new Set<string>();
  private position = 0;

  constructor(private readonly tokens: readonly Token[]) {}

  private get token(): Token {
    return this.tokens[this.position];
  }

  private consume(text: string): boolean {
    if (this.token.kind !== 'operator' || this.token.text !== text) return false;
    this.position++;
    return true;
  }

  private expect(text: string): void {
    if (!this.consume(text)) failExpression(`expected "${text}"`, this.token.position);
  }

  private checkDepth(depth: number): void {
    if (depth > MAXIMUM_NESTING_DEPTH) {
      failExpression(`nesting depth must not exceed ${MAXIMUM_NESTING_DEPTH}`, this.token.position);
    }
  }

  parse(): Evaluate {
    const evaluate = this.parseConditional(0);
    if (this.token.kind !== 'end')
      failExpression('expected end of expression', this.token.position);
    return evaluate;
  }

  private parseConditional(depth: number): Evaluate {
    this.checkDepth(depth);
    const condition = this.parseBinary(1, depth);
    if (!this.consume('?')) return condition;
    const consequent = this.parseConditional(depth + 1);
    this.expect(':');
    const alternate = this.parseConditional(depth + 1);
    return row => (condition(row) ? consequent(row) : alternate(row));
  }

  private parseBinary(minimumPrecedence: number, depth: number): Evaluate {
    let left = this.parseUnary(depth);
    while (this.token.kind === 'operator') {
      const operator = this.token.text;
      const precedence = getPrecedence(operator);
      if (precedence < minimumPrecedence) break;
      this.position++;
      const right = this.parseBinary(precedence + 1, depth);
      left = compileBinary(operator, left, right);
    }
    return left;
  }

  private parseUnary(depth: number): Evaluate {
    this.checkDepth(depth);
    if (this.consume('-')) return compileNumericUnary(this.parseUnary(depth + 1), value => -value);
    if (this.consume('!')) {
      const operand = this.parseUnary(depth + 1);
      return row => !operand(row);
    }
    const left = this.parsePrimary(depth);
    // The exponent may itself start with a unary operator: 2^-2 = 0.25, -2^2 = -4.
    return this.consume('^') ? compileBinary('^', left, this.parseUnary(depth + 1)) : left;
  }

  private parsePrimary(depth: number): Evaluate {
    const token = this.token;
    if (this.consume('(')) {
      const evaluate = this.parseConditional(depth + 1);
      this.expect(')');
      return evaluate;
    }
    if (token.kind === 'number' || token.kind === 'string') {
      this.position++;
      const value = token.kind === 'number' ? Number(token.text) : token.text;
      return () => value;
    }
    if (token.kind !== 'identifier' && token.kind !== 'column') {
      failExpression('expected a literal, column, function or "("', token.position);
    }
    this.position++;
    if (token.kind === 'identifier') {
      if (this.consume('(')) return this.parseFunction(token, depth);
      if (token.text === 'true') return () => true;
      if (token.text === 'false') return () => false;
      if (token.text === 'null') return () => null;
    }
    const name = token.text;
    if (!this.columnNames.has(name)) {
      this.columnNames.add(name);
      this.columns.push(name);
    }
    return row => {
      // Rows may have hostile getters or proxies; neither inherited nor non-scalar values escape.
      try {
        if (!Object.prototype.hasOwnProperty.call(row, name)) return null;
        const value: unknown = row[name];
        return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
          ? value
          : null;
      } catch {
        return null;
      }
    };
  }

  private parseFunction(token: Token, depth: number): Evaluate {
    this.checkDepth(depth + 1);
    const name = token.text;
    if (!WORKFLOW_EXPRESSION_FUNCTIONS.includes(name)) {
      failExpression(`unknown function "${name}"`, token.position);
    }
    const argumentsList: Evaluate[] = [];
    if (!this.consume(')')) {
      do {
        argumentsList.push(this.parseConditional(depth + 1));
      } while (this.consume(','));
      this.expect(')');
    }
    const minimum = name === 'coalesce' || name === 'concat' ? 0 : 1;
    const arity =
      name === 'if' || name === 'clamp'
        ? 3
        : name === 'pow' || name === 'contains' || name === 'startswith'
          ? 2
          : 1;
    const variadic = name === 'min' || name === 'max' || name === 'coalesce' || name === 'concat';
    const valid = variadic
      ? argumentsList.length >= minimum
      : name === 'round'
        ? argumentsList.length === 1 || argumentsList.length === 2
        : argumentsList.length === arity;
    if (!valid) {
      const expected = variadic
        ? `${minimum} or more`
        : name === 'round'
          ? '1 or 2'
          : String(arity);
      failExpression(
        `${name} takes ${expected} ${expected === '1' ? 'argument' : 'arguments'}; got ${argumentsList.length}`,
        token.position
      );
    }
    return compileFunction(name, argumentsList);
  }
}

/**
 * Parses an arithmetic or boolean expression over columns. No `eval` and no `Function`: a
 * hand-written parser builds a closure tree. Throws `WorkflowError('parameter-out-of-range')`
 * with the character position on a syntax error or an unknown function.
 */
export function parseWorkflowExpression(source: string): WorkflowExpression {
  if (source.length > MAXIMUM_SOURCE_CHARACTERS) {
    failExpression(
      `source length must not exceed ${MAXIMUM_SOURCE_CHARACTERS} characters`,
      MAXIMUM_SOURCE_CHARACTERS
    );
  }
  const parser = new ExpressionParser(tokenizeExpression(source));
  const evaluate = parser.parse();
  return {source, columns: Object.freeze(parser.columns), evaluate};
}
