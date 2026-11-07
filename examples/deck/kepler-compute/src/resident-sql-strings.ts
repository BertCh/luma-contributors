// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure SQL text planning for string and CAST support in the resident GPU SQL explorer.
 *
 * GPUSQL has no string literals, LIKE/IN, or CAST. Instead of widening the shipped graph module,
 * this file rewrites those constructs into plain numeric GPUSQL over derived columns:
 * - string predicates (`=`, `<>`, `<`, `IN`, `LIKE`, `ILIKE`) are evaluated against the
 *   dictionary labels once on the CPU, giving a per-label lookup table; a GPU gather by dictionary
 *   code then yields a 0/1 mask column and the predicate becomes `mask = 1`;
 * - `col AS alias` over a dictionary column becomes a zero-copy alias column;
 * - `ORDER BY` a string column sorts by a per-label rank column (hidden) when the dictionary is not
 *   already in code point order;
 * - numeric `CAST` becomes a derived GPU column with explicit range and null rules.
 * Everything unsupported throws an error whose message is the DuckDB-fallback reason.
 */

import {
  classifyResidentSqlTypeName,
  collectResidentSqlColumns,
  findResidentSqlClosingParenthesis,
  getResidentSqlTypeMetadata,
  isResidentSqlStringExpression,
  makeResidentSqlLabelLookup,
  parseResidentSqlEpochSeconds,
  parseResidentSqlExpression,
  planResidentSqlNumericExpression,
  RESIDENT_SQL_STRING_FUNCTIONS,
  type ResidentSqlNode,
  type ResidentSqlNumericPlan
} from './resident-sql-expressions';

/** Prefix for derived columns that must never reach a published result. */
export const RESIDENT_SQL_HIDDEN_PREFIX = '__luma_sql_';

export type ResidentSqlScalarFormat = 'float32' | 'sint32' | 'uint32';
export type ResidentSqlColumnInfo = {
  readonly format: ResidentSqlScalarFormat;
  /** Present for string columns stored as dictionary codes. */
  readonly dictionary?: readonly string[];
};
export type ResidentSqlTableInfo = {
  readonly name: string;
  readonly columns: Readonly<Record<string, ResidentSqlColumnInfo>>;
};
export type ResidentSqlCastTarget = ResidentSqlScalarFormat | 'bool' | 'date' | 'timestamp';

export type ResidentSqlDerivedColumn =
  /** `words[code]` gathered per row; mask, rank and string-to-number casts. */
  | {
      readonly kind: 'lookup';
      readonly name: string;
      readonly hidden: boolean;
      readonly sourceColumn: string;
      readonly format: ResidentSqlScalarFormat;
      readonly words: Uint32Array;
      /** Present when the lookup yields codes of a new dictionary (string function results). */
      readonly dictionary?: readonly string[];
      readonly metadata?: Readonly<Record<string, string>>;
    }
  /** Zero-copy second name for a column, keeping its dictionary. */
  | {
      readonly kind: 'alias';
      readonly name: string;
      readonly hidden: false;
      readonly sourceColumn: string;
    }
  /** Numeric conversion kernel. */
  | {
      readonly kind: 'cast';
      readonly name: string;
      readonly hidden: boolean;
      readonly sourceColumn: string;
      readonly sourceFormat: ResidentSqlScalarFormat;
      readonly target: ResidentSqlCastTarget;
    }
  /** Numeric value to dictionary code by binary search over sorted order-preserving keys. */
  | {
      readonly kind: 'valueLookup';
      readonly name: string;
      readonly hidden: boolean;
      readonly sourceColumn: string;
      readonly sourceFormat: ResidentSqlScalarFormat;
      /** Sorted order-preserving keys of the distinct values (see makeResidentSqlOrderedKey). */
      readonly keys: Uint32Array;
      /** Dictionary code of each key. */
      readonly codes: Uint32Array;
      readonly dictionary: readonly string[];
    }
  /** Two dictionary columns combined: `words[first * secondSize + second]`. */
  | {
      readonly kind: 'pairLookup';
      readonly name: string;
      readonly hidden: boolean;
      readonly sourceColumn: string;
      readonly secondColumn: string;
      readonly firstSize: number;
      readonly secondSize: number;
      /** Third and further columns (mixed radix, last fastest) and their dictionary sizes. */
      readonly extraColumns?: readonly string[];
      readonly extraSizes?: readonly number[];
      readonly format: ResidentSqlScalarFormat;
      readonly words: Uint32Array;
      readonly dictionary?: readonly string[];
    }
  /** CAST of a numeric expression over resident columns, computed by a generated kernel. */
  | {
      readonly kind: 'expression';
      readonly name: string;
      readonly hidden: boolean;
      readonly plan: ResidentSqlNumericPlan;
      readonly target: ResidentSqlCastTarget;
    };

export type ResidentSqlRewriteOptions = {
  /** Distinct-value labels of a numeric column for CAST(... AS VARCHAR); absent while pending. */
  readonly resolveValueDictionary?: (
    columnName: string,
    tableName: string
  ) =>
    | {
        readonly labels: readonly string[];
        readonly keys: Uint32Array;
        readonly codes: Uint32Array;
      }
    | undefined;
};

export type ResidentSqlRewrite = {
  readonly sql: string;
  readonly tableName: string;
  readonly derived: readonly ResidentSqlDerivedColumn[];
  /** Grouped queries: the COUNT(*) column that is zero for empty dictionary slots. */
  readonly groupPresenceColumn?: string;
  /** Visible derived columns that carry field metadata (for example epoch-second time columns). */
  readonly columnMetadata: Readonly<Record<string, Readonly<Record<string, string>>>>;
  /** Numeric columns whose CAST to VARCHAR still needs a GPU distinct pass (placeholder plan). */
  readonly pendingValueColumns: readonly string[];
};

export type ResidentSqlToken = {
  readonly kind: 'word' | 'string' | 'number' | 'symbol' | 'parameter';
  readonly text: string;
  /** Decoded value of a string literal. */
  readonly value: string;
  readonly start: number;
  readonly end: number;
};
type Token = ResidentSqlToken;
type Edit = {readonly start: number; readonly end: number; readonly text: string};

const TOKEN_PATTERN =
  /\s+|(?:--[^\n\r]*)|('(?:[^']|'')*')|(::)|(?::([A-Za-z_][A-Za-z0-9_]*))|((?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*))?)|(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|(<=|>=|<>|!=|\|\||[(),;.*+\-/=<>])/gy;

/** Splits SQL into tokens with source offsets; whitespace and `--` comments are dropped. */
export function tokenizeResidentSql(sql: string): Token[] {
  const tokens: Token[] = [];
  let cursor = 0;
  while (cursor < sql.length) {
    TOKEN_PATTERN.lastIndex = cursor;
    const match = TOKEN_PATTERN.exec(sql);
    if (!match) throw new Error(`SQL contains an unsupported token at character ${cursor}`);
    const start = cursor;
    cursor = TOKEN_PATTERN.lastIndex;
    const text = match[0];
    if (match[1])
      tokens.push({
        kind: 'string',
        text,
        value: text.slice(1, -1).replaceAll("''", "'"),
        start,
        end: cursor
      });
    else if (match[2]) tokens.push({kind: 'symbol', text, value: text, start, end: cursor});
    else if (match[3]) tokens.push({kind: 'parameter', text, value: text, start, end: cursor});
    else if (match[4]) tokens.push({kind: 'word', text, value: text, start, end: cursor});
    else if (match[5]) tokens.push({kind: 'number', text, value: text, start, end: cursor});
    else if (match[6]) tokens.push({kind: 'symbol', text, value: text, start, end: cursor});
  }
  return tokens;
}

/** Applies non-overlapping edits to SQL text. */
export function applyResidentSqlEdits(sql: string, edits: readonly Edit[]): string {
  let result = '';
  let cursor = 0;
  for (const edit of [...edits].sort((a, b) => a.start - b.start || a.end - b.end)) {
    result += sql.slice(cursor, edit.start) + edit.text;
    cursor = edit.end;
  }
  return result + sql.slice(cursor);
}

/** Decodes `"quoted ""name"""` into `quoted "name"`; bare names pass through. */
export function decodeResidentSqlIdentifier(text: string): string {
  return text.startsWith('"') && text.endsWith('"')
    ? text.slice(1, -1).replaceAll('""', '"')
    : text;
}

/** Quotes a column name only when it is not a bare identifier. */
export function quoteResidentSqlIdentifier(name: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) ? name : `"${name.replaceAll('"', '""')}"`;
}

/** Compares strings by Unicode code point (the byte order of UTF-8), like DuckDB's default. */
export function compareResidentSqlStrings(left: string, right: string): number {
  let leftIndex = 0;
  let rightIndex = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    const leftPoint = left.codePointAt(leftIndex)!;
    const rightPoint = right.codePointAt(rightIndex)!;
    if (leftPoint !== rightPoint) return leftPoint < rightPoint ? -1 : 1;
    leftIndex += leftPoint > 0xffff ? 2 : 1;
    rightIndex += rightPoint > 0xffff ? 2 : 1;
  }
  return left.length - leftIndex === right.length - rightIndex
    ? 0
    : left.length - leftIndex < right.length - rightIndex
      ? -1
      : 1;
}

export type ResidentSqlLikeKind = 'equals' | 'prefix' | 'suffix' | 'contains' | 'general';

/** Classifies a LIKE pattern; only `%` at the ends and no `_`/escapeCharacter make the fast kinds. */
export function classifyResidentSqlLikePattern(
  pattern: string,
  escapeCharacter?: string
): {kind: ResidentSqlLikeKind; literal: string} {
  if (escapeCharacter !== undefined && pattern.includes(escapeCharacter))
    return {kind: 'general', literal: ''};
  if (pattern.includes('_')) return {kind: 'general', literal: ''};
  const leading = pattern.startsWith('%');
  const trailing = pattern.length > 1 && pattern.endsWith('%');
  const body = pattern.slice(leading ? 1 : 0, trailing ? -1 : undefined);
  if (body.includes('%')) return {kind: 'general', literal: ''};
  if (leading && trailing) return {kind: 'contains', literal: body};
  if (leading) return {kind: 'suffix', literal: body};
  if (trailing) return {kind: 'prefix', literal: body};
  return {kind: 'equals', literal: body};
}

/** Compiles a LIKE/ILIKE pattern into a label predicate (`%` any run, `_` one code point). */
export function compileResidentSqlLike(
  pattern: string,
  options: {caseInsensitive?: boolean; escapeCharacter?: string} = {}
): (label: string) => boolean {
  const fold = options.caseInsensitive ? (text: string) => text.toLowerCase() : (t: string) => t;
  const foldedPattern = fold(pattern);
  const escapeCharacter =
    options.escapeCharacter === undefined ? undefined : fold(options.escapeCharacter);
  const {kind, literal} = classifyResidentSqlLikePattern(foldedPattern, escapeCharacter);
  switch (kind) {
    case 'equals':
      return label => fold(label) === literal;
    case 'prefix':
      return label => fold(label).startsWith(literal);
    case 'suffix':
      return label => fold(label).endsWith(literal);
    case 'contains':
      return label => fold(label).includes(literal);
    default:
      break;
  }
  const expression = new RegExp(`${buildLikeSource(foldedPattern, escapeCharacter)}$`, 'u');
  return label => expression.test(fold(label));
}

function buildLikeSource(pattern: string, escapeCharacter: string | undefined): string {
  const characters = Array.from(pattern);
  let source = '^';
  for (let index = 0; index < characters.length; index++) {
    const character = characters[index];
    if (
      escapeCharacter !== undefined &&
      character === escapeCharacter &&
      index + 1 < characters.length
    ) {
      source += escapeRegex(characters[++index]);
    } else {
      source +=
        character === '%' ? '[\\s\\S]*' : character === '_' ? '[\\s\\S]' : escapeRegex(character);
    }
  }
  return source;
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

type StringComparison = '=' | '<>' | '<' | '<=' | '>' | '>=';

function compareLabel(comparison: StringComparison, label: string, literal: string): boolean {
  const order = compareResidentSqlStrings(label, literal);
  switch (comparison) {
    case '=':
      return order === 0;
    case '<>':
      return order !== 0;
    case '<':
      return order < 0;
    case '<=':
      return order <= 0;
    case '>':
      return order > 0;
    default:
      return order >= 0;
  }
}

/** One 0/1 word per dictionary label for `label <comparison> literal`. */
export function makeResidentSqlComparisonLookup(
  dictionary: readonly string[],
  comparison: StringComparison,
  literal: string
): Uint32Array {
  return Uint32Array.from(dictionary, label => (compareLabel(comparison, label, literal) ? 1 : 0));
}

/** One 0/1 word per label for `label [NOT] IN (literals)`. */
export function makeResidentSqlInLookup(
  dictionary: readonly string[],
  literals: readonly string[],
  negated: boolean
): Uint32Array {
  const set = new Set(literals);
  return Uint32Array.from(dictionary, label => (set.has(label) !== negated ? 1 : 0));
}

/** One 0/1 word per label for `label [NOT] LIKE|ILIKE pattern`. */
export function makeResidentSqlLikeLookup(
  dictionary: readonly string[],
  pattern: string,
  options: {caseInsensitive: boolean; negated: boolean; escapeCharacter?: string}
): Uint32Array {
  const matches = compileResidentSqlLike(pattern, options);
  return Uint32Array.from(dictionary, label => (matches(label) !== options.negated ? 1 : 0));
}

/** Rank of each label in code point order; equal labels share the rank of their first entry. */
export function makeResidentSqlRankLookup(dictionary: readonly string[]): Uint32Array {
  const order = dictionary.map((_, code) => code);
  order.sort((a, b) => compareResidentSqlStrings(dictionary[a], dictionary[b]) || a - b);
  const ranks = new Uint32Array(dictionary.length);
  let rank = 0;
  for (let position = 0; position < order.length; position++) {
    if (
      position > 0 &&
      compareResidentSqlStrings(dictionary[order[position - 1]], dictionary[order[position]]) !== 0
    )
      rank = position;
    ranks[order[position]] = rank;
  }
  return ranks;
}

/** True when sorting by dictionary code already sorts by label. */
export function isResidentSqlDictionarySorted(dictionary: readonly string[]): boolean {
  for (let code = 1; code < dictionary.length; code++)
    if (compareResidentSqlStrings(dictionary[code - 1], dictionary[code]) > 0) return false;
  return true;
}

const CAST_TYPES: Readonly<Record<string, ResidentSqlCastTarget | 'string'>> = {
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
  DATE: 'date',
  TIMESTAMP: 'timestamp',
  DATETIME: 'timestamp',
  TIMESTAMP_S: 'timestamp',
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
  VARCHAR: 'string',
  TEXT: 'string',
  STRING: 'string',
  CHAR: 'string',
  BPCHAR: 'string'
};

/** Result value of a numeric CAST with DuckDB rounding; invalid means null (DuckDB would error). */
export function castResidentSqlValue(
  value: number,
  source: ResidentSqlScalarFormat,
  target: ResidentSqlCastTarget
): {value: number; valid: boolean} {
  if (target === 'bool') return {value: value !== 0 ? 1 : 0, valid: true};
  if (target === 'date' || target === 'timestamp') {
    const seconds = castResidentSqlValue(value, source, 'sint32');
    return target === 'date' && seconds.valid
      ? {value: Math.floor(seconds.value / 86400) * 86400, valid: true}
      : seconds;
  }
  if (target === 'float32') return {value: Math.fround(value), valid: true};
  if (Number.isNaN(value)) return {value: 0, valid: false};
  // Round half away from zero, like DuckDB's float to integer cast.
  const rounded =
    source === 'float32' ? Math.sign(value) * Math.floor(Math.abs(value) + 0.5) + 0 : value;
  if (target === 'sint32')
    return rounded >= -2147483648 && rounded < 2147483648
      ? {value: rounded, valid: true}
      : {value: 0, valid: false};
  return rounded >= 0 && rounded < 4294967296
    ? {value: rounded, valid: true}
    : {value: 0, valid: false};
}

/** Maps a SQL type name to a GPU cast target, or throws the refusal reason. */
export function resolveResidentSqlCastType(typeName: string): ResidentSqlCastTarget | 'string' {
  const target = CAST_TYPES[typeName.toUpperCase()];
  if (!target) throw new Error(`CAST to ${typeName} is not supported on the GPU`);
  return target;
}

/** Per-label numeric value of a string column cast, or the reason a label cannot convert. */
export function makeResidentSqlStringCastLookup(
  dictionary: readonly string[],
  target: ResidentSqlScalarFormat
): Uint32Array {
  const words = new Uint32Array(dictionary.length);
  const scratch = new DataView(new ArrayBuffer(4));
  for (const [code, label] of dictionary.entries()) {
    const value = label.trim() === '' ? NaN : Number(label);
    if (!Number.isFinite(value))
      throw new Error(
        `CAST of string label "${label}" to a number would fail in SQL; refusing instead of nulling it`
      );
    if (target === 'float32') scratch.setFloat32(0, value, true);
    else {
      const cast = castResidentSqlValue(value, 'sint32', target);
      if (!cast.valid || !Number.isInteger(value))
        throw new Error(`CAST of string label "${label}" does not fit ${target} exactly`);
      if (target === 'sint32') scratch.setInt32(0, cast.value, true);
      else scratch.setUint32(0, cast.value, true);
    }
    words[code] = scratch.getUint32(0, true);
  }
  return words;
}

type StatementLayout = {
  tokens: Token[];
  depth: number[];
  selectEnd: number;
  tableToken: Token;
  tableInfo: ResidentSqlTableInfo;
  qualifiers: Set<string>;
  clauseStart: Record<'where' | 'group' | 'order' | 'limit', number>;
};

const NON_ALIAS_WORDS =
  /^(WHERE|GROUP|ORDER|LIMIT|JOIN|INNER|LEFT|RIGHT|FULL|OUTER|SEMI|ANTI|ON|USING)$/i;

function layoutStatement(
  sql: string,
  findTable: (name: string) => ResidentSqlTableInfo | undefined
): StatementLayout | null {
  const tokens = tokenizeResidentSql(sql);
  const depth: number[] = [];
  let level = 0;
  for (const token of tokens) {
    if (token.text === ')') level--;
    depth.push(level);
    if (token.text === '(') level++;
  }
  const isWord = (index: number, word: string) =>
    tokens[index]?.kind === 'word' && tokens[index].text.toUpperCase() === word;
  if (!isWord(0, 'SELECT')) return null;
  let selectEnd = -1;
  const clauseStart = {where: -1, group: -1, order: -1, limit: -1};
  for (let index = 1; index < tokens.length; index++) {
    const token = tokens[index];
    if (token.kind !== 'word') continue;
    const word = token.text.toUpperCase();
    // Subqueries and joins have their own planners.
    if (word === 'SELECT' || word === 'JOIN' || word === 'UNION') return null;
    if (depth[index] !== 0) continue;
    if (word === 'FROM' && selectEnd < 0) selectEnd = index;
    else if (word === 'WHERE' && clauseStart.where < 0) clauseStart.where = index;
    else if (word === 'GROUP' && clauseStart.group < 0) clauseStart.group = index;
    else if (word === 'ORDER' && clauseStart.order < 0) clauseStart.order = index;
    else if (word === 'LIMIT' && clauseStart.limit < 0) clauseStart.limit = index;
  }
  if (selectEnd < 0 || tokens[selectEnd + 1]?.kind !== 'word') return null;
  const tableToken = tokens[selectEnd + 1];
  const tableInfo = findTable(decodeResidentSqlIdentifier(tableToken.text).toLowerCase());
  if (!tableInfo) return null;
  const qualifiers = new Set([decodeResidentSqlIdentifier(tableToken.text).toLowerCase()]);
  let aliasIndex = selectEnd + 2;
  if (isWord(aliasIndex, 'AS')) aliasIndex++;
  const aliasToken = tokens[aliasIndex];
  if (aliasToken?.kind === 'word' && !NON_ALIAS_WORDS.test(aliasToken.text))
    qualifiers.add(decodeResidentSqlIdentifier(aliasToken.text).toLowerCase());
  return {tokens, depth, selectEnd, tableToken, tableInfo, qualifiers, clauseStart};
}

/** Resolves a (possibly qualified) word token to a table column name. */
function resolveColumn(layout: StatementLayout, token: Token | undefined): string | undefined {
  if (token?.kind !== 'word') return undefined;
  const parts = token.text.match(/"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*/g) ?? [];
  if (parts.length === 0 || parts.length > 2) return undefined;
  if (
    parts.length === 2 &&
    !layout.qualifiers.has(decodeResidentSqlIdentifier(parts[0]).toLowerCase())
  )
    return undefined;
  const requested = decodeResidentSqlIdentifier(parts[parts.length - 1]).toLowerCase();
  const matches = Object.keys(layout.tableInfo.columns).filter(
    name => name.toLowerCase() === requested
  );
  return matches.length === 1 ? matches[0] : undefined;
}

type SelectItem = {start: number; end: number};

function splitSelectItems(layout: StatementLayout): SelectItem[] {
  const items: SelectItem[] = [];
  let start = 1;
  for (let index = 1; index <= layout.selectEnd; index++) {
    if (
      index === layout.selectEnd ||
      (layout.tokens[index].text === ',' && layout.depth[index] === 0)
    ) {
      items.push({start, end: index});
      start = index + 1;
    }
  }
  return items;
}

function toWord(tokens: Token[], index: number): string {
  return tokens[index]?.kind === 'word' ? tokens[index].text.toUpperCase() : '';
}

/** Words that may precede `(` without being a function name. */
const NON_FUNCTION_WORDS =
  /^(AND|OR|NOT|IN|ON|AS|BY|FROM|WHERE|SELECT|WHEN|THEN|ELSE|END|LIKE|ILIKE|ESCAPE|LIMIT|BETWEEN|IS|HAVING|USING)$/i;

type ExpressionRoot = {
  start: number;
  end: number;
  /** Prebuilt node for `column::type`. */
  prebuilt?: ResidentSqlNode;
};

function findMatchingOpen(layout: StatementLayout, close: number): number {
  for (let index = close - 1; index >= 0; index--)
    if (layout.tokens[index].text === '(' && layout.depth[index] === layout.depth[close])
      return index;
  return -1;
}

/** Finds CAST, `::`, string function and `||` expressions as minimal token ranges. */
function findExpressionRoots(layout: StatementLayout): ExpressionRoot[] {
  const {tokens} = layout;
  const covered = new Array<boolean>(tokens.length).fill(false);
  const roots: ExpressionRoot[] = [];
  const skipIndex = layout.selectEnd + 1;
  const primaryEnd = (index: number): number => {
    const token = tokens[index];
    if (!token) return -1;
    if (token.kind === 'string' || token.kind === 'number') return index + 1;
    if (token.text === '(') {
      const close = findResidentSqlClosingParenthesis(tokens, index);
      return close < 0 ? -1 : close + 1;
    }
    if (token.kind !== 'word') return -1;
    if (tokens[index + 1]?.text === '(') {
      const close = findResidentSqlClosingParenthesis(tokens, index + 1);
      return close < 0 ? -1 : close + 1;
    }
    return index + 1;
  };
  const primaryStart = (end: number): number => {
    const token = tokens[end - 1];
    if (!token) return -1;
    if (token.text === ')') {
      const open = findMatchingOpen(layout, end - 1);
      if (open < 0) return -1;
      const previous = tokens[open - 1];
      return previous?.kind === 'word' && !NON_FUNCTION_WORDS.test(previous.text) ? open - 1 : open;
    }
    return token.kind === 'string' || token.kind === 'number' || token.kind === 'word'
      ? end - 1
      : -1;
  };
  for (let index = 1; index < tokens.length; index++) {
    if (tokens[index].text !== '||' || covered[index]) continue;
    let start = primaryStart(index);
    let end = primaryEnd(index + 1);
    while (start > 0 && tokens[start - 1]?.text === '||' && start >= 0)
      start = primaryStart(start - 1);
    while (end >= 0 && tokens[end]?.text === '||') end = primaryEnd(end + 1);
    if (start < 0 || end < 0)
      throw new Error('|| on the GPU needs column, literal or function operands');
    for (let cursor = start; cursor < end; cursor++) covered[cursor] = true;
    roots.push({start, end});
  }
  for (let index = 1; index < tokens.length; index++) {
    if (covered[index] || index === skipIndex) continue;
    const word = toWord(tokens, index);
    if (
      (word === 'CAST' || RESIDENT_SQL_STRING_FUNCTIONS.has(word)) &&
      tokens[index + 1]?.text === '('
    ) {
      const close = findResidentSqlClosingParenthesis(tokens, index + 1);
      if (close < 0) throw new Error('Unbalanced parentheses');
      for (let cursor = index; cursor <= close; cursor++) covered[cursor] = true;
      roots.push({start: index, end: close + 1});
    } else if (tokens[index].text === '::') {
      const operand = tokens[index - 1];
      if (operand?.kind !== 'word' || (index > 1 && tokens[index - 2].text === '.'))
        throw new Error('CAST on the GPU needs a single column operand');
      let end = index + 1;
      let typeName = toWord(tokens, end);
      if (!typeName) throw new Error('CAST requires a type name');
      end++;
      if (typeName === 'DOUBLE' && toWord(tokens, end) === 'PRECISION') {
        typeName = 'DOUBLE PRECISION';
        end++;
      }
      if (tokens[end]?.text === '(') {
        while (end < tokens.length && tokens[end].text !== ')') end++;
        end++;
      }
      roots.push({
        start: index - 1,
        end,
        prebuilt: {kind: 'cast', operand: {kind: 'column', name: operand.text}, typeName}
      });
      for (let cursor = index - 1; cursor < end; cursor++) covered[cursor] = true;
    }
  }
  // A `||` inside a function call (UPPER(a || b)) is part of the outer expression: keep the outer.
  return roots
    .filter(
      root =>
        !roots.some(
          other =>
            other !== root &&
            other.start <= root.start &&
            other.end >= root.end &&
            (other.start < root.start || other.end > root.end)
        )
    )
    .sort((a, b) => a.start - b.start);
}

function planNumericCast(
  info: ResidentSqlColumnInfo,
  columnName: string,
  target: ResidentSqlCastTarget
): {kind: 'identity'} | {kind: 'cast'} | {kind: 'lookup'; words: Uint32Array} {
  if (info.dictionary) {
    if (target === 'bool')
      throw new Error(
        `CAST of string column "${columnName}" to BOOLEAN is not supported on the GPU`
      );
    return {
      kind: 'lookup',
      words: makeResidentSqlStringCastLookup(
        info.dictionary,
        target === 'date' || target === 'timestamp' ? 'sint32' : target
      )
    };
  }
  return info.format === target ? {kind: 'identity'} : {kind: 'cast'};
}

/** Largest dictionary-pair product a two-column string expression may expand to. */
export const MAXIMUM_RESIDENT_SQL_PAIR_LABELS = 65_536;
/** Most dictionary columns one string expression may combine on the GPU. */
export const MAXIMUM_RESIDENT_SQL_EXPRESSION_COLUMNS = 8;

/** A planned expression: no GPU work (identity), or a derived column to build once named. */
type ExpressionPlan =
  | {kind: 'identity'; text: string}
  | {
      kind: 'derive';
      info: ResidentSqlColumnInfo;
      baseName: string;
      /** DATE or TIMESTAMP result: a compared string literal becomes epoch seconds. */
      timeType?: 'date' | 'timestamp';
      build: (name: string, hidden: boolean) => ResidentSqlDerivedColumn;
    };

function planExpressionRoot(
  layout: StatementLayout,
  root: ExpressionRoot,
  options: ResidentSqlRewriteOptions,
  pendingValueColumns: string[]
): ExpressionPlan {
  const {tokens} = layout;
  const resolve = (token: Token) => {
    const name = resolveColumn(layout, token);
    if (!name) throw new Error(`${token.text} is not a column of the table`);
    return name;
  };
  const node =
    root.prebuilt && root.prebuilt.kind === 'cast'
      ? ({
          ...root.prebuilt,
          operand: {kind: 'column', name: resolve(tokens[root.start])}
        } as ResidentSqlNode)
      : parseResidentSqlExpression(tokens, root.start, root.end, resolve);
  const columns = [...collectResidentSqlColumns(node)];
  const infoOf = (name: string) => layout.tableInfo.columns[name];
  const dictionaryColumns = columns.filter(name => infoOf(name).dictionary);
  const columnSafe = (name: string) => name.replace(/[^A-Za-z0-9_]/g, '_');
  // CAST(column AS type): the single-column forms keep their compact plans.
  if (node.kind === 'cast' && node.operand.kind === 'column') {
    const columnName = node.operand.name;
    const info = infoOf(columnName);
    const typeClass = classifyResidentSqlTypeName(node.typeName);
    const operandText = tokens[root.prebuilt ? root.start : root.start + 2].text;
    if (typeClass === 'varchar') {
      if (info.dictionary) return {kind: 'identity', text: operandText};
      // Numeric to VARCHAR: distinct values come from a GPU pass, resolved at execution.
      const resolved = options.resolveValueDictionary?.(columnName, layout.tableInfo.name);
      if (!resolved && !pendingValueColumns.includes(columnName))
        pendingValueColumns.push(columnName);
      const dictionary = resolved?.labels ?? [''];
      return {
        kind: 'derive',
        info: {format: 'uint32', dictionary},
        baseName: `cast_${columnSafe(columnName)}`,
        build: (name, hidden) => ({
          kind: 'valueLookup',
          name,
          hidden,
          sourceColumn: columnName,
          sourceFormat: info.format,
          keys: resolved?.keys ?? new Uint32Array(0),
          codes: resolved?.codes ?? new Uint32Array(0),
          dictionary
        })
      };
    }
    if (typeClass === 'date' || typeClass === 'timestamp') {
      const metadata = getResidentSqlTypeMetadata(typeClass)!;
      if (info.dictionary) {
        const lookup = makeResidentSqlLabelLookup(node, [columnName], [info.dictionary]);
        return {
          kind: 'derive',
          info: {format: 'sint32'},
          baseName: `cast_${columnSafe(columnName)}`,
          timeType: typeClass,
          build: (name, hidden) => ({
            kind: 'lookup',
            name,
            hidden,
            sourceColumn: columnName,
            format: 'sint32',
            words: lookup.words,
            metadata
          })
        };
      }
      return {
        kind: 'derive',
        info: {format: 'sint32'},
        baseName: `cast_${columnSafe(columnName)}`,
        timeType: typeClass,
        build: (name, hidden) => ({
          kind: 'cast',
          name,
          hidden,
          sourceColumn: columnName,
          sourceFormat: info.format,
          target: typeClass
        })
      };
    }
    const target = typeClass as ResidentSqlCastTarget;
    const plan = planNumericCast(info, columnName, target);
    if (plan.kind === 'identity') return {kind: 'identity', text: operandText};
    const format: ResidentSqlScalarFormat =
      target === 'bool' ? 'uint32' : (target as ResidentSqlScalarFormat);
    return {
      kind: 'derive',
      info: {format},
      baseName: `cast_${columnSafe(columnName)}`,
      build: (name, hidden) =>
        plan.kind === 'lookup'
          ? {kind: 'lookup', name, hidden, sourceColumn: columnName, format, words: plan.words}
          : {
              kind: 'cast',
              name,
              hidden,
              sourceColumn: columnName,
              sourceFormat: info.format,
              target
            }
    };
  }
  const baseName =
    node.kind === 'call'
      ? `${node.name.toLowerCase()}_${columnSafe(columns[0] ?? 'value')}`
      : node.kind === 'cast'
        ? `cast_${columnSafe(columns[0] ?? 'value')}`
        : `concat_${columnSafe(columns[0] ?? 'value')}`;
  if (dictionaryColumns.length || isResidentSqlStringExpression(node)) {
    if (columns.length !== dictionaryColumns.length)
      throw new Error('A string expression cannot mix dictionary and numeric columns on the GPU');
    if (!columns.length)
      throw new Error('A constant string expression has no column to evaluate per row');
    if (columns.length > MAXIMUM_RESIDENT_SQL_EXPRESSION_COLUMNS)
      throw new Error(
        `String expressions over more than ${MAXIMUM_RESIDENT_SQL_EXPRESSION_COLUMNS} columns run on DuckDB`
      );
    const dictionaries = columns.map(name => infoOf(name).dictionary!);
    const product = dictionaries.reduce((total, dictionary) => total * dictionary.length, 1);
    if (product > MAXIMUM_RESIDENT_SQL_PAIR_LABELS)
      throw new Error(
        `Concatenating or combining ${columns.join(' and ')} would need ${product} labels (limit ${MAXIMUM_RESIDENT_SQL_PAIR_LABELS}); unbounded, runs on DuckDB`
      );
    const lookup = makeResidentSqlLabelLookup(node, columns, dictionaries);
    return {
      kind: 'derive',
      info: {format: lookup.format, ...(lookup.dictionary ? {dictionary: lookup.dictionary} : {})},
      baseName,
      build: (name, hidden) =>
        columns.length === 1
          ? {
              kind: 'lookup',
              name,
              hidden,
              sourceColumn: columns[0],
              format: lookup.format,
              words: lookup.words,
              ...(lookup.dictionary ? {dictionary: lookup.dictionary} : {}),
              ...(lookup.metadata ? {metadata: lookup.metadata} : {})
            }
          : {
              kind: 'pairLookup',
              name,
              hidden,
              sourceColumn: columns[0],
              secondColumn: columns[1],
              secondSize: dictionaries[1].length,
              firstSize: dictionaries[0].length,
              ...(columns.length > 2
                ? {
                    extraColumns: columns.slice(2),
                    extraSizes: dictionaries.slice(2).map(dictionary => dictionary.length)
                  }
                : {}),
              format: lookup.format,
              words: lookup.words,
              ...(lookup.dictionary ? {dictionary: lookup.dictionary} : {})
            }
    };
  }
  // CAST of a numeric expression.
  if (node.kind !== 'cast')
    throw new Error('Only CAST, string functions and || are planned as GPU expressions');
  const typeClass = classifyResidentSqlTypeName(node.typeName);
  if (typeClass === 'varchar')
    throw new Error('CAST of a numeric expression to VARCHAR is not supported on the GPU');
  if (typeClass === 'bool')
    throw new Error('CAST of an expression to BOOLEAN is not supported on the GPU');
  const plan = planResidentSqlNumericExpression(node.operand, name => infoOf(name).format);
  const target = typeClass as ResidentSqlCastTarget;
  return {
    kind: 'derive',
    info: {
      format:
        target === 'date' || target === 'timestamp' ? 'sint32' : (target as ResidentSqlScalarFormat)
    },
    baseName,
    ...(target === 'date' || target === 'timestamp' ? {timeType: target} : {}),
    build: (name, hidden) => ({kind: 'expression', name, hidden, plan, target})
  };
}

/**
 * Rewrites CAST, string functions and `||`. Each distinct expression becomes one derived column;
 * a top-level select item `expr [AS alias]` is replaced as a whole by the visible column name.
 */
function rewriteExpressions(
  layout: StatementLayout,
  derived: ResidentSqlDerivedColumn[],
  extra: Record<string, ResidentSqlColumnInfo>,
  options: ResidentSqlRewriteOptions,
  pendingValueColumns: string[]
): Edit[] {
  const roots = findExpressionRoots(layout);
  if (!roots.length) return [];
  const {tokens} = layout;
  const edits: Edit[] = [];
  const items = splitSelectItems(layout);
  const usedNames = new Set(
    [...Object.keys(layout.tableInfo.columns), ...derived.map(spec => spec.name)].map(name =>
      name.toLowerCase()
    )
  );
  type Group = {
    plan: ExpressionPlan;
    ownerStart: number;
    visibleName?: string;
    name?: string;
    built: boolean;
  };
  const groups = new Map<string, Group>();
  const placed = roots.map(root => {
    const key = tokens
      .slice(root.start, root.end)
      .map(token => token.text.toLowerCase())
      .join(' ');
    const item = items.find(candidate => candidate.start === root.start);
    const trailing = item ? tokens.slice(root.end, item.end) : [];
    const pure =
      item &&
      (trailing.length === 0 ||
        (trailing.length === 2 &&
          toWord(tokens, root.end) === 'AS' &&
          trailing[1].kind === 'word'));
    let group = groups.get(key);
    if (!group) {
      group = {
        plan: planExpressionRoot(layout, root, options, pendingValueColumns),
        ownerStart: -1,
        built: false
      };
      groups.set(key, group);
    }
    return {
      root,
      key,
      item,
      pure: Boolean(pure),
      aliasToken: trailing.length ? trailing[1] : undefined,
      group
    };
  });
  for (const entry of placed) {
    if (entry.group.plan.kind === 'identity') {
      edits.push({
        start: tokens[entry.root.start].start,
        end: tokens[entry.root.end - 1].end,
        text: entry.group.plan.text
      });
      continue;
    }
    const {group} = entry;
    const plan = group.plan as Extract<ExpressionPlan, {kind: 'derive'}>;
    if (entry.pure && group.ownerStart < 0) {
      group.ownerStart = entry.root.start;
      const aliasName = entry.aliasToken
        ? decodeResidentSqlIdentifier(entry.aliasToken.text)
        : undefined;
      if (aliasName && usedNames.has(aliasName.toLowerCase()))
        throw new Error(
          `Output name "${aliasName}" collides with a source column; use a different AS alias`
        );
      let name = aliasName ?? plan.baseName;
      for (let counter = 2; !aliasName && usedNames.has(name.toLowerCase()); counter++)
        name = `${plan.baseName}_${counter}`;
      usedNames.add(name.toLowerCase());
      group.visibleName = name;
    }
  }
  for (const entry of placed) {
    const {group} = entry;
    if (group.plan.kind === 'identity') continue;
    const plan = group.plan;
    if (!group.built) {
      group.built = true;
      group.name = group.visibleName ?? `${RESIDENT_SQL_HIDDEN_PREFIX}expr_${derived.length}`;
      derived.push(plan.build(group.name, group.visibleName === undefined));
      extra[group.name] = plan.info;
    }
    if (entry.pure && entry.item) {
      const isOwner = group.ownerStart === entry.root.start;
      let name = group.name!;
      let text = entry.aliasToken ? entry.aliasToken.text : name;
      if (!isOwner) {
        // A second visible copy of the same expression is a zero-copy alias.
        const aliasName = entry.aliasToken
          ? decodeResidentSqlIdentifier(entry.aliasToken.text)
          : `${group.name}_${derived.length}`;
        if (usedNames.has(aliasName.toLowerCase()))
          throw new Error(`Output name "${aliasName}" collides with another column`);
        usedNames.add(aliasName.toLowerCase());
        derived.push({kind: 'alias', name: aliasName, hidden: false, sourceColumn: group.name!});
        extra[aliasName] = plan.info;
        name = aliasName;
        text = entry.aliasToken ? entry.aliasToken.text : aliasName;
      }
      edits.push({
        start: tokens[entry.item.start].start,
        end: tokens[entry.item.end - 1].end,
        text
      });
    } else {
      edits.push({
        start: tokens[entry.root.start].start,
        end: tokens[entry.root.end - 1].end,
        text: group.name!
      });
    }
  }
  // A DATE/TIMESTAMP column compared with a string literal compares epoch seconds.
  const comparisonTexts = new Set(['=', '<>', '!=', '<', '<=', '>', '>=']);
  for (const entry of placed) {
    const plan = entry.group.plan;
    if (plan.kind !== 'derive' || !plan.timeType) continue;
    // Look through redundant parentheses such as the ones BETWEEN adds around an operand.
    let operandStart = entry.root.start;
    let operandEnd = entry.root.end;
    while (tokens[operandStart - 1]?.text === '(' && tokens[operandEnd]?.text === ')') {
      operandStart--;
      operandEnd++;
    }
    const literalIndex = comparisonTexts.has(tokens[operandEnd]?.text ?? '')
      ? operandEnd + 1
      : comparisonTexts.has(tokens[operandStart - 1]?.text ?? '') &&
          tokens[operandStart - 2]?.kind === 'string'
        ? operandStart - 2
        : -1;
    const literal = tokens[literalIndex];
    if (literal?.kind !== 'string') continue;
    edits.push({
      start: literal.start,
      end: literal.end,
      text: String(parseResidentSqlEpochSeconds(literal.value, plan.timeType))
    });
  }
  return edits;
}

const COMPARISONS: Readonly<Record<string, StringComparison>> = {
  '=': '=',
  '<>': '<>',
  '!=': '<>',
  '<': '<',
  '<=': '<=',
  '>': '>',
  '>=': '>='
};
const FLIPPED: Readonly<Record<StringComparison, StringComparison>> = {
  '=': '=',
  '<>': '<>',
  '<': '>',
  '<=': '>=',
  '>': '<',
  '>=': '<='
};

/** Rewrites string predicates, dictionary aliases and string ORDER BY (second stage). */
function rewriteStrings(layout: StatementLayout, derived: ResidentSqlDerivedColumn[]): Edit[] {
  const {tokens} = layout;
  const edits: Edit[] = [];
  const consumed = new Set<number>();
  const makeMaskName = () => `${RESIDENT_SQL_HIDDEN_PREFIX}mask_${derived.length}`;
  const dictionaryOf = (token: Token | undefined) => {
    const name = resolveColumn(layout, token);
    if (!name) return undefined;
    const dictionary = layout.tableInfo.columns[name].dictionary;
    return dictionary ? {name, dictionary} : undefined;
  };
  const addMask = (
    start: number,
    end: number,
    column: {name: string; dictionary: readonly string[]},
    words: Uint32Array
  ) => {
    const name = makeMaskName();
    derived.push({
      kind: 'lookup',
      name,
      hidden: true,
      sourceColumn: column.name,
      format: 'uint32',
      words
    });
    edits.push({start: tokens[start].start, end: tokens[end - 1].end, text: `(${name} = 1)`});
    for (let index = start; index < end; index++) consumed.add(index);
  };
  const stringAt = (index: number): string | undefined =>
    tokens[index]?.kind === 'string' ? tokens[index].value : undefined;
  for (let index = 1; index < tokens.length; index++) {
    if (consumed.has(index) || index === layout.selectEnd + 1) continue;
    const token = tokens[index];
    const column = dictionaryOf(token);
    if (column) {
      const next = tokens[index + 1];
      const comparison = next ? COMPARISONS[next.text] : undefined;
      const otherColumn = comparison ? dictionaryOf(tokens[index + 2]) : undefined;
      if (comparison && otherColumn) {
        // Two string columns: compare their ranks in the union of both dictionaries.
        const union = [...new Set([...column.dictionary, ...otherColumn.dictionary])].sort(
          compareResidentSqlStrings
        );
        const rankByLabel = new Map(union.map((label, rank) => [label, rank]));
        const addRank = (side: {name: string; dictionary: readonly string[]}) => {
          const name = makeMaskName();
          derived.push({
            kind: 'lookup',
            name,
            hidden: true,
            sourceColumn: side.name,
            format: 'uint32',
            words: Uint32Array.from(side.dictionary, label => rankByLabel.get(label)!)
          });
          return name;
        };
        const leftName = addRank(column);
        const rightName = column.name === otherColumn.name ? leftName : addRank(otherColumn);
        edits.push({
          start: tokens[index].start,
          end: tokens[index + 2].end,
          text: `(${leftName} ${next.text} ${rightName})`
        });
        for (let consume = index; consume < index + 3; consume++) consumed.add(consume);
        continue;
      }
      if (comparison && stringAt(index + 2) !== undefined) {
        addMask(
          index,
          index + 3,
          column,
          makeResidentSqlComparisonLookup(column.dictionary, comparison, stringAt(index + 2)!)
        );
        continue;
      }
      let cursor = index + 1;
      const negated = toWord(tokens, cursor) === 'NOT';
      if (negated) cursor++;
      const operator = toWord(tokens, cursor);
      if (operator === 'IN' && tokens[cursor + 1]?.text === '(') {
        const literals: string[] = [];
        let position = cursor + 2;
        while (stringAt(position) !== undefined) {
          literals.push(stringAt(position)!);
          position++;
          if (tokens[position]?.text === ',') position++;
          else break;
        }
        if (!literals.length || tokens[position]?.text !== ')')
          throw new Error('IN on the GPU supports a list of string literals only');
        addMask(
          index,
          position + 1,
          column,
          makeResidentSqlInLookup(column.dictionary, literals, negated)
        );
        continue;
      }
      if ((operator === 'LIKE' || operator === 'ILIKE') && stringAt(cursor + 1) !== undefined) {
        let end = cursor + 2;
        let escapeCharacter: string | undefined;
        if (toWord(tokens, end) === 'ESCAPE') {
          escapeCharacter = stringAt(end + 1);
          if (escapeCharacter === undefined || Array.from(escapeCharacter).length !== 1)
            throw new Error('LIKE ESCAPE requires one character');
          end += 2;
        }
        addMask(
          index,
          end,
          column,
          makeResidentSqlLikeLookup(column.dictionary, stringAt(cursor + 1)!, {
            caseInsensitive: operator === 'ILIKE',
            negated,
            ...(escapeCharacter !== undefined ? {escapeCharacter} : {})
          })
        );
        continue;
      }
    } else if (token.kind === 'string') {
      const comparison = tokens[index + 1] ? COMPARISONS[tokens[index + 1].text] : undefined;
      const right = dictionaryOf(tokens[index + 2]);
      if (comparison && right) {
        addMask(
          index,
          index + 3,
          right,
          makeResidentSqlComparisonLookup(right.dictionary, FLIPPED[comparison], token.value)
        );
        continue;
      }
    }
  }
  for (const [index, token] of tokens.entries()) {
    if (token.kind === 'string' && !consumed.has(index))
      throw new Error(
        'String literal outside a supported predicate (column =, <>, <, IN, LIKE, ILIKE over a string column)'
      );
  }
  return edits;
}

const BETWEEN_LEFT_STOP = /^(AND|OR|NOT|WHERE|SELECT|WHEN|THEN|ELSE|ON|HAVING|CASE|BY|AS|FROM)$/i;
const BETWEEN_RIGHT_STOP =
  /^(AND|OR|WHERE|GROUP|ORDER|LIMIT|HAVING|THEN|ELSE|END|WHEN|FROM|AS|ASC|DESC|JOIN|INNER|LEFT|RIGHT|FULL|SEMI|ANTI)$/i;
const BETWEEN_COMPARISONS = new Set(['=', '<>', '!=', '<', '<=', '>', '>=']);

/**
 * Rewrites `x [NOT] BETWEEN lo AND hi` into `(x >= lo AND x <= hi)` (NOT: `(x < lo OR x > hi)`).
 * The operand text is repeated, so it must be side-effect free (SQL expressions are).
 */
export function rewriteResidentSqlBetween(sql: string): string {
  let working = sql;
  for (let pass = 0; pass < 64; pass++) {
    const tokens = tokenizeResidentSql(working);
    const depth: number[] = [];
    let level = 0;
    for (const token of tokens) {
      if (token.text === ')') level--;
      depth.push(level);
      if (token.text === '(') level++;
    }
    const between = tokens.findIndex(
      token => token.kind === 'word' && token.text.toUpperCase() === 'BETWEEN'
    );
    if (between < 0) return working;
    const negated = toWord(tokens, between - 1) === 'NOT';
    const operandEnd = negated ? between - 1 : between;
    let operandStart = operandEnd;
    while (operandStart > 0) {
      const token = tokens[operandStart - 1];
      if (depth[operandStart - 1] < depth[between]) break;
      if (depth[operandStart - 1] === depth[between]) {
        if (token.kind === 'word' && BETWEEN_LEFT_STOP.test(token.text)) break;
        if (token.text === ',' || BETWEEN_COMPARISONS.has(token.text)) break;
      }
      operandStart--;
    }
    let andIndex = -1;
    for (let index = between + 1; index < tokens.length; index++) {
      if (depth[index] < depth[between]) break;
      if (depth[index] === depth[between] && toWord(tokens, index) === 'AND') {
        andIndex = index;
        break;
      }
    }
    if (andIndex < 0 || operandStart === operandEnd || andIndex === between + 1)
      throw new Error('BETWEEN needs `operand BETWEEN low AND high`');
    let highEnd = andIndex + 1;
    while (highEnd < tokens.length) {
      const token = tokens[highEnd];
      if (depth[highEnd] < depth[between]) break;
      if (depth[highEnd] === depth[between]) {
        if (token.kind === 'word' && BETWEEN_RIGHT_STOP.test(token.text)) break;
        if (token.text === ',' || token.text === ';' || BETWEEN_COMPARISONS.has(token.text)) break;
      }
      highEnd++;
    }
    if (highEnd === andIndex + 1) throw new Error('BETWEEN is missing its upper bound');
    // Single-token operands stay bare so string predicates (`col >= 'x'`) still match.
    const wrap = (from: number, to: number) => {
      const text = working.slice(tokens[from].start, tokens[to - 1].end);
      return to - from === 1 ? text : `(${text})`;
    };
    const operand = wrap(operandStart, operandEnd);
    const low = wrap(between + 1, andIndex);
    const high = wrap(andIndex + 1, highEnd);
    const text = negated
      ? `(${operand} < ${low} OR ${operand} > ${high})`
      : `(${operand} >= ${low} AND ${operand} <= ${high})`;
    working =
      working.slice(0, tokens[operandStart].start) + text + working.slice(tokens[highEnd - 1].end);
  }
  throw new Error('BETWEEN nesting is too deep');
}

/**
 * SQL promotes integers to floating point in mixed arithmetic; the dataframe expression compiler
 * requires one scalar format per expression. A chain `a * b + c` of plain columns and numbers that
 * mixes a float32 column with integer columns gets `CAST(integer AS DOUBLE)` on the integer ones.
 */
function promoteMixedArithmetic(layout: StatementLayout): Edit[] {
  const {tokens} = layout;
  const edits: Edit[] = [];
  const isOperator = (index: number) => /^[-+*/]$/.test(tokens[index]?.text ?? '');
  const isOperand = (index: number) =>
    tokens[index]?.kind === 'number' ||
    (tokens[index]?.kind === 'word' && resolveColumn(layout, tokens[index]) !== undefined);
  let index = 1;
  while (index < tokens.length) {
    if (!isOperand(index) || tokens[index - 1]?.text === '.') {
      index++;
      continue;
    }
    const chain = [index];
    while (isOperator(chain[chain.length - 1] + 1) && isOperand(chain[chain.length - 1] + 2))
      chain.push(chain[chain.length - 1] + 2);
    index = chain[chain.length - 1] + 1;
    if (chain.length < 2) continue;
    // CAST and string function arguments are typed by the expression compiler itself.
    let enclosing = 0;
    let inFunctionCall = false;
    for (let back = chain[0] - 1; back >= 0 && !inFunctionCall; back--) {
      if (tokens[back].text === ')') enclosing++;
      else if (tokens[back].text === '(') {
        if (enclosing) enclosing--;
        else
          inFunctionCall =
            tokens[back - 1]?.kind === 'word' && !NON_FUNCTION_WORDS.test(tokens[back - 1].text);
      }
    }
    if (inFunctionCall) continue;
    const columns = chain
      .filter(position => tokens[position].kind === 'word')
      .map(position => ({
        token: tokens[position],
        info: layout.tableInfo.columns[resolveColumn(layout, tokens[position])!]
      }));
    if (columns.some(column => column.info.dictionary)) continue;
    const hasFloat = columns.some(column => column.info.format === 'float32');
    if (!hasFloat) continue;
    for (const column of columns)
      if (column.info.format !== 'float32')
        edits.push({
          start: column.token.start,
          end: column.token.end,
          text: `CAST(${column.token.text} AS DOUBLE)`
        });
  }
  return edits;
}

/**
 * Plans string, alias, ORDER BY, CAST and string-function support. Returns null when the
 * statement needs no rewrite or is not a single-table SELECT over a known table (GPUSQL then
 * plans it as written).
 */
export function rewriteResidentSql(
  sql: string,
  findBaseTable: (name: string) => ResidentSqlTableInfo | undefined,
  options: ResidentSqlRewriteOptions = {}
): ResidentSqlRewrite | null {
  const derived: ResidentSqlDerivedColumn[] = [];
  // Derived columns are visible to later stages as extra table columns.
  const extra: Record<string, ResidentSqlColumnInfo> = Object.create(null);
  const findTable = (name: string): ResidentSqlTableInfo | undefined => {
    const base = findBaseTable(name);
    return base ? {name: base.name, columns: {...base.columns, ...extra}} : undefined;
  };
  const betweenText = rewriteResidentSqlBetween(sql);
  let working = betweenText;
  let layout = layoutStatement(working, findTable);
  if (!layout) return null;
  const promotionEdits = promoteMixedArithmetic(layout);
  if (promotionEdits.length) {
    working = applyResidentSqlEdits(working, promotionEdits);
    layout = layoutStatement(working, findTable)!;
  }
  const tableName = layout.tableInfo.name;
  const baseColumns = layout.tableInfo.columns;
  const pendingValueColumns: string[] = [];
  // Stage 1: CAST, string functions and ||.
  const castEdits = rewriteExpressions(layout, derived, extra, options, pendingValueColumns);
  if (castEdits.length) {
    working = applyResidentSqlEdits(working, castEdits);
    layout = layoutStatement(working, findTable)!;
  }
  const edits = rewriteStrings(layout, derived);
  // Dictionary aliases in the select list.
  const items = splitSelectItems(layout);
  const sourceNames = Object.keys(layout.tableInfo.columns).map(name => name.toLowerCase());
  const aliasSource = new Map<string, string>();
  for (const item of items) {
    const itemTokens = layout.tokens.slice(item.start, item.end);
    if (itemTokens.length === 3 && toWord(layout.tokens, item.start + 1) === 'AS') {
      const column = resolveColumn(layout, itemTokens[0]);
      if (!column || !layout.tableInfo.columns[column].dictionary) continue;
      const alias = decodeResidentSqlIdentifier(itemTokens[2].text);
      if (alias.toLowerCase() === column.toLowerCase()) continue;
      if (sourceNames.includes(alias.toLowerCase()))
        throw new Error(`Alias "${alias}" collides with a source column`);
      derived.push({kind: 'alias', name: alias, hidden: false, sourceColumn: column});
      aliasSource.set(alias.toLowerCase(), column);
      edits.push({start: itemTokens[0].start, end: itemTokens[2].end, text: itemTokens[2].text});
    }
  }
  // ORDER BY a string column sorts by label rank when codes are not already label ordered.
  const appended: string[] = [];
  if (layout.clauseStart.order >= 0) {
    const orderToken = layout.tokens[layout.clauseStart.order + 2];
    const orderName =
      orderToken?.kind === 'word' ? decodeResidentSqlIdentifier(orderToken.text) : '';
    const column = aliasSource.get(orderName.toLowerCase()) ?? resolveColumn(layout, orderToken);
    const info = column ? layout.tableInfo.columns[column] : undefined;
    if (column && info?.dictionary && !isResidentSqlDictionarySorted(info.dictionary)) {
      const name = `${RESIDENT_SQL_HIDDEN_PREFIX}rank_${derived.length}`;
      derived.push({
        kind: 'lookup',
        name,
        hidden: true,
        sourceColumn: column,
        format: 'uint32',
        words: makeResidentSqlRankLookup(info.dictionary)
      });
      edits.push({start: orderToken.start, end: orderToken.end, text: name});
      appended.push(name);
    }
  }
  // GROUP BY emits one dense row per dictionary slot; a COUNT(*) column tells which are occupied.
  let groupPresenceColumn: string | undefined;
  if (layout.clauseStart.group >= 0) {
    for (const item of items) {
      const itemTokens = layout.tokens.slice(item.start, item.end);
      if (
        toWord(layout.tokens, item.start) === 'COUNT' &&
        itemTokens[1]?.text === '(' &&
        itemTokens[2]?.text === '*' &&
        itemTokens[3]?.text === ')'
      ) {
        groupPresenceColumn =
          toWord(layout.tokens, item.start + 4) === 'AS' && itemTokens[5]
            ? decodeResidentSqlIdentifier(itemTokens[5].text)
            : 'count_all';
        break;
      }
    }
    if (!groupPresenceColumn) {
      groupPresenceColumn = `${RESIDENT_SQL_HIDDEN_PREFIX}group_count`;
      appended.push(`COUNT(*) AS ${groupPresenceColumn}`);
    }
  }
  const hasExtras = derived.length > 0;
  if (
    !hasExtras &&
    !castEdits.length &&
    !appended.length &&
    !groupPresenceColumn &&
    working === sql
  )
    return null;
  // Star must not leak derived columns into the output.
  if (hasExtras && items.length === 1 && layout.tokens[items[0].start].text === '*') {
    const names = Object.keys(baseColumns).map(quoteResidentSqlIdentifier);
    edits.push({
      start: layout.tokens[items[0].start].start,
      end: layout.tokens[items[0].start].end,
      text: names.join(', ')
    });
  }
  if (appended.length) {
    const lastToken = layout.tokens[layout.selectEnd - 1];
    edits.push({start: lastToken.end, end: lastToken.end, text: `, ${appended.join(', ')}`});
  }
  const columnMetadata: Record<string, Readonly<Record<string, string>>> = Object.create(null);
  for (const spec of derived) {
    if (spec.hidden) continue;
    const metadata =
      spec.kind === 'lookup' || spec.kind === 'pairLookup'
        ? 'metadata' in spec
          ? spec.metadata
          : undefined
        : spec.kind === 'cast' && (spec.target === 'date' || spec.target === 'timestamp')
          ? getResidentSqlTypeMetadata(spec.target)
          : spec.kind === 'expression' && (spec.target === 'date' || spec.target === 'timestamp')
            ? getResidentSqlTypeMetadata(spec.target)
            : undefined;
    if (metadata) columnMetadata[spec.name] = metadata;
  }
  return {
    sql: applyResidentSqlEdits(working, edits),
    tableName,
    derived,
    columnMetadata,
    pendingValueColumns,
    ...(groupPresenceColumn ? {groupPresenceColumn} : {})
  };
}
