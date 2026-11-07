// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ResidentSqlToken} from './resident-sql-strings';

/**
 * Pure SQL expression pieces for the resident GPU SQL explorer: a small expression parser, a
 * per-dictionary-label evaluator (string functions run once per label on the CPU, then a GPU gather
 * by code applies them to every row), a numeric expression compiler (WGSL plus CPU mirror) for
 * CAST of expressions, DATE/TIMESTAMP parsing and DuckDB-style number formatting.
 */
export type ResidentSqlNode =
  | {readonly kind: 'column'; readonly name: string}
  | {readonly kind: 'string'; readonly value: string}
  | {readonly kind: 'number'; readonly value: number; readonly isInteger: boolean}
  | {readonly kind: 'call'; readonly name: string; readonly args: readonly ResidentSqlNode[]}
  | {readonly kind: 'cast'; readonly operand: ResidentSqlNode; readonly typeName: string}
  | {
      readonly kind: 'binary';
      readonly operator: string;
      readonly left: ResidentSqlNode;
      readonly right: ResidentSqlNode;
    }
  | {readonly kind: 'negate'; readonly operand: ResidentSqlNode};

export const RESIDENT_SQL_STRING_FUNCTIONS: ReadonlySet<string> = new Set([
  'UPPER',
  'LOWER',
  'TRIM',
  'LTRIM',
  'RTRIM',
  'LENGTH',
  'LEN',
  'CHAR_LENGTH',
  'CHARACTER_LENGTH',
  'SUBSTR',
  'SUBSTRING',
  'REPLACE',
  'CONCAT'
]);
const NUMERIC_FUNCTIONS: ReadonlySet<string> = new Set([
  'ABS',
  'FLOOR',
  'CEIL',
  'CEILING',
  'ROUND',
  'SQRT'
]);

/** Pairs of parentheses: index of the `)` matching the `(` at `open`, or -1. */
export function findResidentSqlClosingParenthesis(
  tokens: readonly ResidentSqlToken[],
  open: number
): number {
  let level = 0;
  for (let index = open; index < tokens.length; index++) {
    if (tokens[index].text === '(') level++;
    else if (tokens[index].text === ')' && --level === 0) return index;
  }
  return -1;
}

/**
 * Parses tokens `[start, end)` as one expression: `||` over `+ -` over `* /` over unary minus over
 * literals, columns (resolved by `resolveColumn`), function calls and `CAST(x AS type)`.
 */
export function parseResidentSqlExpression(
  tokens: readonly ResidentSqlToken[],
  start: number,
  end: number,
  resolveColumn: (token: ResidentSqlToken) => string
): ResidentSqlNode {
  let position = start;
  const text = () => (position < end ? tokens[position].text : '');
  const upper = () =>
    position < end && tokens[position].kind === 'word' ? text().toUpperCase() : '';
  const expect = (symbol: string) => {
    if (text() !== symbol) throw new Error(`Expression expected "${symbol}" near "${text()}"`);
    position++;
  };
  const parseConcat = (): ResidentSqlNode => {
    let node = parseAdditive();
    while (text() === '||') {
      position++;
      node = {kind: 'binary', operator: '||', left: node, right: parseAdditive()};
    }
    return node;
  };
  const parseAdditive = (): ResidentSqlNode => {
    let node = parseMultiplicative();
    while (text() === '+' || text() === '-') {
      const operator = text();
      position++;
      node = {kind: 'binary', operator, left: node, right: parseMultiplicative()};
    }
    return node;
  };
  const parseMultiplicative = (): ResidentSqlNode => {
    let node = parseUnary();
    while (text() === '*' || text() === '/') {
      const operator = text();
      position++;
      node = {kind: 'binary', operator, left: node, right: parseUnary()};
    }
    return node;
  };
  const parseUnary = (): ResidentSqlNode => {
    if (text() === '-') {
      position++;
      return {kind: 'negate', operand: parseUnary()};
    }
    return parsePrimary();
  };
  const parsePrimary = (): ResidentSqlNode => {
    if (position >= end) throw new Error('Expression ended unexpectedly');
    const token = tokens[position];
    if (token.kind === 'string') {
      position++;
      return {kind: 'string', value: token.value};
    }
    if (token.kind === 'number') {
      position++;
      const value = Number(token.text);
      return {kind: 'number', value, isInteger: /^\d+$/.test(token.text)};
    }
    if (token.text === '(') {
      position++;
      const node = parseConcat();
      expect(')');
      return node;
    }
    if (token.kind !== 'word') throw new Error(`Expression does not support "${token.text}"`);
    if (upper() === 'CAST' && tokens[position + 1]?.text === '(') {
      position += 2;
      const operand = parseConcat();
      if (upper() !== 'AS') throw new Error('CAST requires AS type');
      position++;
      let typeName = upper();
      if (!typeName) throw new Error('CAST requires a type name');
      position++;
      if (typeName === 'DOUBLE' && upper() === 'PRECISION') {
        typeName = 'DOUBLE PRECISION';
        position++;
      }
      if (text() === '(') {
        while (position < end && text() !== ')') position++;
        position++;
      }
      expect(')');
      return {kind: 'cast', operand, typeName};
    }
    if (tokens[position + 1]?.text === '(') {
      const name = upper();
      if (!RESIDENT_SQL_STRING_FUNCTIONS.has(name) && !NUMERIC_FUNCTIONS.has(name))
        throw new Error(`Function ${name} is not supported on the GPU`);
      position += 2;
      const args: ResidentSqlNode[] = [];
      if (text() !== ')') {
        args.push(parseConcat());
        while (text() === ',') {
          position++;
          args.push(parseConcat());
        }
      }
      expect(')');
      return {kind: 'call', name, args};
    }
    position++;
    return {kind: 'column', name: resolveColumn(token)};
  };
  const node = parseConcat();
  if (position !== end) throw new Error(`Expression has unexpected "${text()}"`);
  return node;
}

/** Distinct column names an expression reads. */
export function collectResidentSqlColumns(
  node: ResidentSqlNode,
  found = new Set<string>()
): Set<string> {
  switch (node.kind) {
    case 'column':
      found.add(node.name);
      break;
    case 'call':
      for (const argument of node.args) collectResidentSqlColumns(argument, found);
      break;
    case 'cast':
    case 'negate':
      collectResidentSqlColumns(node.operand, found);
      break;
    case 'binary':
      collectResidentSqlColumns(node.left, found);
      collectResidentSqlColumns(node.right, found);
      break;
    default:
      break;
  }
  return found;
}

/** True when the expression has string literals, `||` or string functions (not pure numerics). */
export function isResidentSqlStringExpression(node: ResidentSqlNode): boolean {
  switch (node.kind) {
    case 'string':
      return true;
    case 'call':
      return (
        RESIDENT_SQL_STRING_FUNCTIONS.has(node.name) ||
        node.args.some(isResidentSqlStringExpression)
      );
    case 'cast':
    case 'negate':
      return isResidentSqlStringExpression(node.operand);
    case 'binary':
      return (
        node.operator === '||' ||
        isResidentSqlStringExpression(node.left) ||
        isResidentSqlStringExpression(node.right)
      );
    default:
      return false;
  }
}

export type ResidentSqlTypeClass =
  | 'varchar'
  | 'sint32'
  | 'uint32'
  | 'float32'
  | 'bool'
  | 'date'
  | 'timestamp';
const TYPE_CLASSES: Readonly<Record<string, ResidentSqlTypeClass>> = {
  INT: 'sint32',
  INTEGER: 'sint32',
  INT4: 'sint32',
  SIGNED: 'sint32',
  SMALLINT: 'sint32',
  INT2: 'sint32',
  TINYINT: 'sint32',
  INT1: 'sint32',
  BIGINT: 'sint32',
  INT8: 'sint32',
  UINTEGER: 'uint32',
  UINT32: 'uint32',
  UBIGINT: 'uint32',
  USMALLINT: 'uint32',
  UTINYINT: 'uint32',
  FLOAT: 'float32',
  FLOAT4: 'float32',
  REAL: 'float32',
  DOUBLE: 'float32',
  FLOAT8: 'float32',
  'DOUBLE PRECISION': 'float32',
  DECIMAL: 'float32',
  NUMERIC: 'float32',
  BOOLEAN: 'bool',
  BOOL: 'bool',
  LOGICAL: 'bool',
  VARCHAR: 'varchar',
  TEXT: 'varchar',
  STRING: 'varchar',
  CHAR: 'varchar',
  BPCHAR: 'varchar',
  DATE: 'date',
  TIMESTAMP: 'timestamp',
  DATETIME: 'timestamp',
  TIMESTAMP_S: 'timestamp'
};

/** Maps a SQL type name to its GPU class, or throws the refusal reason. */
export function classifyResidentSqlTypeName(typeName: string): ResidentSqlTypeClass {
  const typeClass = TYPE_CLASSES[typeName.toUpperCase()];
  if (!typeClass) throw new Error(`CAST to ${typeName} is not supported on the GPU`);
  return typeClass;
}

/** Column metadata that tells Kepler a sint32 column holds epoch seconds (time format `X`). */
export const RESIDENT_SQL_TIME_METADATA: Readonly<Record<string, string>> = {
  'luma.gl.time.format': 'X',
  'luma.gl.time.unit': 'second'
};

export function getResidentSqlTypeMetadata(
  typeClass: ResidentSqlTypeClass
): Readonly<Record<string, string>> | undefined {
  return typeClass === 'date' || typeClass === 'timestamp'
    ? {...RESIDENT_SQL_TIME_METADATA, 'luma.gl.sql.cast': typeClass.toUpperCase()}
    : undefined;
}

const DATE_PATTERN =
  /^\s*(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?(?:Z|[+-]00(?::?00)?)?\s*$/;

/**
 * Epoch seconds of an ISO date or timestamp label (UTC). DATE truncates to midnight. Fractions of a
 * second, invalid calendar dates and instants outside the int32 second range are refused.
 */
export function parseResidentSqlEpochSeconds(
  label: string,
  typeClass: 'date' | 'timestamp'
): number {
  const match = DATE_PATTERN.exec(label);
  if (!match)
    throw new Error(`CAST of label "${label}" to ${typeClass.toUpperCase()} would fail in SQL`);
  const [, year, month, day, hour = '0', minute = '0', second = '0', fraction = ''] = match;
  if (typeClass === 'timestamp' && /[1-9]/.test(fraction))
    throw new Error(`Label "${label}" has fractional seconds; GPU timestamps carry whole seconds`);
  const milliseconds = Date.UTC(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second)
  );
  const check = new Date(milliseconds);
  if (
    check.getUTCFullYear() !== Number(year) ||
    check.getUTCMonth() !== Number(month) - 1 ||
    check.getUTCDate() !== Number(day) ||
    Number(hour) > 23 ||
    Number(minute) > 59 ||
    Number(second) > 59
  )
    throw new Error(`CAST of label "${label}" to ${typeClass.toUpperCase()} would fail in SQL`);
  let seconds = milliseconds / 1000;
  if (typeClass === 'date') seconds = Math.floor(seconds / 86400) * 86400;
  if (seconds < -2147483648 || seconds > 2147483647)
    throw new Error(
      `Label "${label}" is outside the 32-bit epoch-second range of GPU time columns`
    );
  return seconds;
}

/** DuckDB-style text of a float32 value: shortest round-trip digits, `.0` on whole numbers. */
export function formatResidentSqlFloat(value: number): string {
  if (Number.isNaN(value)) return 'nan';
  if (!Number.isFinite(value)) return value > 0 ? 'inf' : '-inf';
  if (value === 0) return Object.is(value, -0) ? '-0.0' : '0.0';
  let text = '';
  for (let digits = 1; digits <= 9; digits++) {
    text = value.toPrecision(digits);
    if (Math.fround(Number(text)) === value) break;
  }
  let number = Number(text);
  if (Math.abs(number) >= 1e15 || Math.abs(number) < 1e-4) {
    const [mantissa, exponent] = number.toExponential().split('e');
    return `${mantissa}e${exponent.startsWith('-') ? exponent : exponent.replace('+', '+')}`;
  }
  text = String(number);
  number = 0;
  return text.includes('.') ? text : `${text}.0`;
}

// ---- per-label evaluation -----------------------------------------------------------------

export type ResidentSqlValueType = 'string' | 'int' | 'float' | 'date' | 'timestamp';

/** Static result type of an expression whose columns are all dictionary (string) columns. */
export function inferResidentSqlType(node: ResidentSqlNode): ResidentSqlValueType {
  switch (node.kind) {
    case 'column':
    case 'string':
      return 'string';
    case 'number':
      return node.isInteger && Math.abs(node.value) < 2 ** 31 ? 'int' : 'float';
    case 'negate':
      return inferResidentSqlType(node.operand);
    case 'cast': {
      const typeClass = classifyResidentSqlTypeName(node.typeName);
      return typeClass === 'varchar'
        ? 'string'
        : typeClass === 'sint32' || typeClass === 'uint32'
          ? 'int'
          : typeClass === 'float32'
            ? 'float'
            : typeClass === 'bool'
              ? (() => {
                  throw new Error(
                    'CAST to BOOLEAN of a string expression is not supported on the GPU'
                  );
                })()
              : typeClass;
    }
    case 'call':
      switch (node.name) {
        case 'LENGTH':
        case 'LEN':
        case 'CHAR_LENGTH':
        case 'CHARACTER_LENGTH':
          return 'int';
        case 'ABS':
          return inferResidentSqlType(node.args[0]);
        case 'FLOOR':
        case 'CEIL':
        case 'CEILING':
        case 'ROUND':
        case 'SQRT':
          return 'float';
        default:
          return 'string';
      }
    default:
      if (node.operator === '||') return 'string';
      if (node.operator === '/') return 'float';
      return inferResidentSqlType(node.left) === 'int' && inferResidentSqlType(node.right) === 'int'
        ? 'int'
        : 'float';
  }
}

function roundHalfAwayFromZero(value: number): number {
  return Math.sign(value) * Math.floor(Math.abs(value) + 0.5) + 0;
}
const characterCount = (text: string) => Array.from(text).length;
const spaceTrim = (text: string, characters: string, left: boolean, right: boolean) => {
  const set = new Set(Array.from(characters));
  const points = Array.from(text);
  let from = 0;
  let to = points.length;
  while (left && from < to && set.has(points[from])) from++;
  while (right && to > from && set.has(points[to - 1])) to--;
  return points.slice(from, to).join('');
};

/** Evaluates an expression for one assignment of dictionary labels (column name to label). */
export function evaluateResidentSqlLabelNode(
  node: ResidentSqlNode,
  labels: Readonly<Record<string, string>>
): string | number {
  const evaluate = (child: ResidentSqlNode) => evaluateResidentSqlLabelNode(child, labels);
  const asText = (child: ResidentSqlNode) => {
    const value = evaluate(child);
    return typeof value === 'string'
      ? value
      : inferResidentSqlType(child) === 'float'
        ? formatResidentSqlFloat(Math.fround(value))
        : String(value);
  };
  const asNumber = (child: ResidentSqlNode) => {
    const value = evaluate(child);
    if (typeof value !== 'number')
      throw new Error('Arithmetic over a string expression needs CAST');
    return value;
  };
  switch (node.kind) {
    case 'column':
      return labels[node.name];
    case 'string':
      return node.value;
    case 'number':
      return node.value;
    case 'negate':
      return -asNumber(node.operand);
    case 'binary': {
      if (node.operator === '||') return asText(node.left) + asText(node.right);
      const left = asNumber(node.left);
      const right = asNumber(node.right);
      switch (node.operator) {
        case '+':
          return left + right;
        case '-':
          return left - right;
        case '*':
          return left * right;
        default:
          if (right === 0) throw new Error('Division by zero over a dictionary label');
          return left / right;
      }
    }
    case 'cast': {
      const typeClass = classifyResidentSqlTypeName(node.typeName);
      const value = evaluate(node.operand);
      if (typeClass === 'varchar') return asText(node.operand);
      if (typeClass === 'date' || typeClass === 'timestamp') {
        if (typeof value === 'string') return parseResidentSqlEpochSeconds(value, typeClass);
        return typeClass === 'date'
          ? Math.floor(value / 86400) * 86400
          : roundHalfAwayFromZero(value);
      }
      if (typeClass === 'bool')
        throw new Error('CAST to BOOLEAN of a string expression is not supported on the GPU');
      let number: number;
      if (typeof value === 'string') {
        number = value.trim() === '' ? NaN : Number(value);
        if (!Number.isFinite(number))
          throw new Error(
            `CAST of string label "${value}" to a number would fail in SQL; refusing instead of nulling it`
          );
        if (typeClass !== 'float32' && !Number.isInteger(number))
          throw new Error(`CAST of string label "${value}" does not fit ${typeClass} exactly`);
      } else number = value;
      if (typeClass === 'float32') return Math.fround(number);
      number = roundHalfAwayFromZero(number);
      if (
        typeClass === 'sint32'
          ? number < -2147483648 || number > 2147483647
          : number < 0 || number > 4294967295
      )
        throw new Error(`Cast value ${number} does not fit ${typeClass}`);
      return number;
    }
    default:
      break;
  }
  const args = node.args;
  switch (node.name) {
    case 'UPPER':
      return asText(args[0]).toUpperCase();
    case 'LOWER':
      return asText(args[0]).toLowerCase();
    case 'TRIM':
    case 'LTRIM':
    case 'RTRIM':
      return spaceTrim(
        asText(args[0]),
        args[1] ? asText(args[1]) : ' ',
        node.name !== 'RTRIM',
        node.name !== 'LTRIM'
      );
    case 'LENGTH':
    case 'LEN':
    case 'CHAR_LENGTH':
    case 'CHARACTER_LENGTH':
      return characterCount(asText(args[0]));
    case 'SUBSTR':
    case 'SUBSTRING': {
      const start = args[1] ? asNumber(args[1]) : NaN;
      const length = args[2] ? asNumber(args[2]) : Infinity;
      if (
        !Number.isInteger(start) ||
        start < 1 ||
        !(length >= 0) ||
        (args[2] && !Number.isInteger(length))
      )
        throw new Error(
          'SUBSTR on the GPU needs a positive integer start and a nonnegative integer length'
        );
      return Array.from(asText(args[0]))
        .slice(start - 1, Number.isFinite(length) ? start - 1 + length : undefined)
        .join('');
    }
    case 'REPLACE': {
      if (args.length !== 3) throw new Error('REPLACE needs three arguments');
      const needle = asText(args[1]);
      return needle === '' ? asText(args[0]) : asText(args[0]).split(needle).join(asText(args[2]));
    }
    case 'CONCAT':
      return args.map(asText).join('');
    case 'ABS':
      return Math.abs(asNumber(args[0]));
    case 'FLOOR':
      return Math.floor(asNumber(args[0]));
    case 'CEIL':
    case 'CEILING':
      return Math.ceil(asNumber(args[0]));
    case 'ROUND':
      if (args.length !== 1) throw new Error('ROUND with digits is not supported on the GPU');
      return roundHalfAwayFromZero(asNumber(args[0]));
    case 'SQRT':
      return Math.fround(Math.sqrt(asNumber(args[0])));
    default:
      throw new Error(`Function ${node.name} is not supported on the GPU`);
  }
}

export type ResidentSqlLabelLookup = {
  readonly format: 'uint32' | 'sint32' | 'float32';
  readonly words: Uint32Array;
  /** Present for string results: the new sorted, deduplicated dictionary the words index. */
  readonly dictionary?: readonly string[];
  readonly metadata?: Readonly<Record<string, string>>;
};

function compareCodePoints(left: string, right: string): number {
  const leftPoints = Array.from(left);
  const rightPoints = Array.from(right);
  for (let index = 0; index < Math.min(leftPoints.length, rightPoints.length); index++) {
    const difference = leftPoints[index].codePointAt(0)! - rightPoints[index].codePointAt(0)!;
    if (difference) return difference;
  }
  return leftPoints.length - rightPoints.length;
}

/**
 * Evaluates `node` for every combination of dictionary labels (`dictionaries` in the order of
 * `columns`, row-major with the last column fastest) and packs the results as a lookup table.
 */
export function makeResidentSqlLabelLookup(
  node: ResidentSqlNode,
  columns: readonly string[],
  dictionaries: readonly (readonly string[])[]
): ResidentSqlLabelLookup {
  const type = inferResidentSqlType(node);
  const sizes = dictionaries.map(dictionary => dictionary.length);
  const total = sizes.reduce((product, size) => product * size, 1);
  const values: (string | number)[] = [];
  const indices = columns.map(() => 0);
  for (let entry = 0; entry < total; entry++) {
    let remainder = entry;
    for (let column = columns.length - 1; column >= 0; column--) {
      indices[column] = remainder % sizes[column];
      remainder = Math.floor(remainder / sizes[column]);
    }
    const labels: Record<string, string> = Object.create(null);
    columns.forEach((name, column) => {
      labels[name] = dictionaries[column][indices[column]];
    });
    values.push(evaluateResidentSqlLabelNode(node, labels));
  }
  const words = new Uint32Array(total);
  if (type === 'string') {
    const unique = [...new Set(values as string[])].sort(compareCodePoints);
    const codeByLabel = new Map(unique.map((label, code) => [label, code]));
    values.forEach((value, entry) => {
      words[entry] = codeByLabel.get(value as string)!;
    });
    return {format: 'uint32', words, dictionary: unique};
  }
  const scratch = new DataView(new ArrayBuffer(4));
  values.forEach((value, entry) => {
    const number = value as number;
    if (type === 'float') scratch.setFloat32(0, number, true);
    else {
      if (number < -2147483648 || number > 2147483647)
        throw new Error(`Expression value ${number} does not fit a 32-bit integer`);
      scratch.setInt32(0, number, true);
    }
    words[entry] = scratch.getUint32(0, true);
  });
  const metadata =
    type === 'date' || type === 'timestamp' ? getResidentSqlTypeMetadata(type) : undefined;
  return {
    format: type === 'float' ? 'float32' : 'sint32',
    words,
    ...(metadata ? {metadata} : {})
  };
}

// ---- numeric expressions (CAST of expressions) --------------------------------------------

export type ResidentSqlNumericFormat = 'float32' | 'sint32' | 'uint32';
export type ResidentSqlNumericPlan = {
  readonly columns: readonly {readonly name: string; readonly format: ResidentSqlNumericFormat}[];
  /** Computation type; the result is cast from it to the CAST target. */
  readonly resultFormat: 'float32' | 'sint32';
  readonly node: ResidentSqlNode;
};

function inferNumericType(
  node: ResidentSqlNode,
  formatOf: (name: string) => ResidentSqlNumericFormat
): 'int' | 'float' {
  switch (node.kind) {
    case 'column':
      return formatOf(node.name) === 'float32' ? 'float' : 'int';
    case 'number':
      return node.isInteger && Math.abs(node.value) < 2 ** 31 ? 'int' : 'float';
    case 'negate':
      return inferNumericType(node.operand, formatOf);
    case 'binary':
      if (node.operator === '/' || node.operator === '||') return 'float';
      return inferNumericType(node.left, formatOf) === 'int' &&
        inferNumericType(node.right, formatOf) === 'int'
        ? 'int'
        : 'float';
    case 'call':
      if (node.name === 'ABS') return inferNumericType(node.args[0], formatOf);
      if (NUMERIC_FUNCTIONS.has(node.name)) return 'float';
      throw new Error(`Function ${node.name} cannot be computed on numeric columns on the GPU`);
    default:
      throw new Error(
        'CAST of an expression supports numeric columns, numbers, + - * /, ABS, FLOOR, CEIL, ROUND and SQRT'
      );
  }
}

/** Plans a numeric expression over resident numeric columns; throws the refusal reason. */
export function planResidentSqlNumericExpression(
  node: ResidentSqlNode,
  formatOf: (name: string) => ResidentSqlNumericFormat
): ResidentSqlNumericPlan {
  const names = [...collectResidentSqlColumns(node)];
  return {
    columns: names.map(name => ({name, format: formatOf(name)})),
    resultFormat: inferNumericType(node, formatOf) === 'int' ? 'sint32' : 'float32',
    node
  };
}

/**
 * WGSL statements computing `word` (raw 32-bit result) and `valid` from `source0..` bindings and
 * optional `validityIn0..` bindings. Division by zero, square root of a negative and unsigned
 * column values above int32 clear `valid` (DuckDB errors or returns NULL there).
 */
export function getResidentSqlNumericExpressionBody(
  plan: ResidentSqlNumericPlan,
  validityFlags: readonly boolean[]
): string {
  const lines: string[] = ['  var valid: u32 = 1u;'];
  plan.columns.forEach((column, columnIndex) => {
    if (validityFlags[columnIndex])
      lines.push(`  if (validityIn${columnIndex}[index] == 0u) { valid = 0u; }`);
  });
  let counter = 0;
  const emit = (node: ResidentSqlNode): {name: string; type: 'int' | 'float'} => {
    const name = `t${counter++}`;
    const formatOf = (columnName: string) =>
      plan.columns.find(column => column.name === columnName)!.format;
    switch (node.kind) {
      case 'column': {
        const columnIndex = plan.columns.findIndex(column => column.name === node.name);
        const format = plan.columns[columnIndex].format;
        if (format === 'float32')
          lines.push(`  let ${name} = bitcast<f32>(source${columnIndex}[index]);`);
        else if (format === 'sint32')
          lines.push(`  let ${name} = bitcast<i32>(source${columnIndex}[index]);`);
        else
          lines.push(
            `  if (source${columnIndex}[index] > 2147483647u) { valid = 0u; }`,
            `  let ${name} = bitcast<i32>(source${columnIndex}[index]);`
          );
        return {name, type: format === 'float32' ? 'float' : 'int'};
      }
      case 'number':
        if (node.isInteger && Math.abs(node.value) < 2 ** 31) {
          lines.push(`  let ${name} = ${node.value}i;`);
          return {name, type: 'int'};
        }
        lines.push(
          `  let ${name} = f32(${Number.isInteger(node.value) ? `${node.value}.0` : String(node.value)});`
        );
        return {name, type: 'float'};
      case 'negate': {
        const operand = emit(node.operand);
        lines.push(`  let ${name} = -${operand.name};`);
        return {name, type: operand.type};
      }
      case 'binary': {
        const left = emit(node.left);
        const right = emit(node.right);
        const type = inferNumericType(node, formatOf);
        const convert = (operand: {name: string; type: 'int' | 'float'}) =>
          type === 'float' && operand.type === 'int' ? `f32(${operand.name})` : operand.name;
        if (node.operator === '/') {
          lines.push(
            `  var ${name}: f32 = 0.0;`,
            `  if (${convert(right)} == 0.0) { valid = 0u; } else { ${name} = ${convert(left)} / ${convert(right)}; }`
          );
        } else lines.push(`  let ${name} = ${convert(left)} ${node.operator} ${convert(right)};`);
        return {name, type};
      }
      default: {
        const argument = emit(node.kind === 'call' ? node.args[0] : node);
        const value = argument.type === 'int' ? `f32(${argument.name})` : argument.name;
        const functionName = (node as {name: string}).name;
        if (functionName === 'ABS') {
          lines.push(`  let ${name} = abs(${argument.name});`);
          return {name, type: argument.type};
        }
        if (functionName === 'FLOOR') lines.push(`  let ${name} = floor(${value});`);
        else if (functionName === 'CEIL' || functionName === 'CEILING')
          lines.push(`  let ${name} = ceil(${value});`);
        else if (functionName === 'ROUND')
          lines.push(`  let ${name} = sign(${value}) * floor(abs(${value}) + 0.5);`);
        else {
          lines.push(
            `  if (${value} < 0.0) { valid = 0u; }`,
            `  let ${name} = sqrt(max(${value}, 0.0));`
          );
        }
        return {name, type: 'float'};
      }
    }
  };
  const result = emit(plan.node);
  lines.push(
    plan.resultFormat === 'float32'
      ? `  let word = bitcast<u32>(${result.type === 'int' ? `f32(${result.name})` : result.name});`
      : `  let word = bitcast<u32>(${result.name});`
  );
  return lines.join('\n');
}

/** CPU mirror of the numeric expression kernel over raw column words. */
export function mirrorResidentSqlNumericExpression(
  plan: ResidentSqlNumericPlan,
  words: Readonly<Record<string, number>>,
  validity: Readonly<Record<string, boolean>> = {}
): {word: number; valid: boolean} {
  let valid = true;
  const scratch = new DataView(new ArrayBuffer(4));
  for (const column of plan.columns) if (validity[column.name] === false) valid = false;
  const evaluate = (node: ResidentSqlNode): {value: number; type: 'int' | 'float'} => {
    const formatOf = (columnName: string) =>
      plan.columns.find(column => column.name === columnName)!.format;
    switch (node.kind) {
      case 'column': {
        const word = words[node.name] >>> 0;
        const format = formatOf(node.name);
        scratch.setUint32(0, word, true);
        if (format === 'float32') return {value: scratch.getFloat32(0, true), type: 'float'};
        if (format === 'uint32' && word > 2147483647) valid = false;
        return {value: scratch.getInt32(0, true), type: 'int'};
      }
      case 'number':
        return node.isInteger && Math.abs(node.value) < 2 ** 31
          ? {value: node.value, type: 'int'}
          : {value: Math.fround(node.value), type: 'float'};
      case 'negate': {
        const operand = evaluate(node.operand);
        return {
          value: operand.type === 'int' ? -operand.value | 0 : -operand.value,
          type: operand.type
        };
      }
      case 'binary': {
        const left = evaluate(node.left);
        const right = evaluate(node.right);
        const type = inferNumericType(node, formatOf);
        if (node.operator === '/') {
          if (right.value === 0) {
            valid = false;
            return {value: 0, type};
          }
          return {value: Math.fround(Math.fround(left.value) / Math.fround(right.value)), type};
        }
        if (type === 'int') {
          const value =
            node.operator === '+'
              ? (left.value + right.value) | 0
              : node.operator === '-'
                ? (left.value - right.value) | 0
                : Math.imul(left.value, right.value);
          return {value, type};
        }
        const a = Math.fround(left.value);
        const b = Math.fround(right.value);
        return {
          value: Math.fround(node.operator === '+' ? a + b : node.operator === '-' ? a - b : a * b),
          type
        };
      }
      default: {
        const call = node as Extract<ResidentSqlNode, {kind: 'call'}>;
        const argument = evaluate(call.args[0]);
        if (call.name === 'ABS')
          return {
            value:
              argument.type === 'int' ? Math.abs(argument.value) | 0 : Math.abs(argument.value),
            type: argument.type
          };
        const value = Math.fround(argument.value);
        if (call.name === 'FLOOR') return {value: Math.floor(value), type: 'float'};
        if (call.name === 'CEIL' || call.name === 'CEILING')
          return {value: Math.ceil(value), type: 'float'};
        if (call.name === 'ROUND')
          return {value: Math.sign(value) * Math.floor(Math.abs(value) + 0.5) + 0, type: 'float'};
        if (value < 0) valid = false;
        return {value: Math.fround(Math.sqrt(Math.max(value, 0))), type: 'float'};
      }
    }
  };
  const result = evaluate(plan.node);
  if (plan.resultFormat === 'float32') scratch.setFloat32(0, result.value, true);
  else scratch.setInt32(0, result.value, true);
  return {word: scratch.getUint32(0, true), valid};
}
