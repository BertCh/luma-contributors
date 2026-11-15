// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import * as arrow from 'apache-arrow';
import {aggregateInteger64OnGpu} from '@luma-contributors/analysis/dataframe-analysis';
import {
  makeInteger64RangeSelectionExpression,
  type Integer64Bound,
  type Integer64Range
} from '@luma-contributors/analysis/dataframe-analysis';
import type {WideInteger64Summary} from '@luma-contributors/analysis/dataframe-analysis';
import {makeWideNumericWordVectors} from './wide-numeric-gpu-summary';

export type WideSqlAggregateFunction = 'count' | 'sum' | 'min' | 'max';
export type WideSqlAggregateItem = {
  functionName: WideSqlAggregateFunction;
  /** Source column, or null for COUNT(*). */
  column: string | null;
  outputName: string;
};
export type WideSqlAggregatePlan = {
  tableName: string;
  items: readonly WideSqlAggregateItem[];
  /** Exact range on the single filtered column; every item references that column (or COUNT(*)). */
  filter?: {column: string; range: Integer64Range};
};

const IDENTIFIER = '(?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*)';
const STATEMENT_PATTERN = new RegExp(
  `^\\s*SELECT\\s+([\\s\\S]+?)\\s+FROM\\s+(${IDENTIFIER})(?:\\s+WHERE\\s+([\\s\\S]+?))?\\s*;?\\s*$`,
  'i'
);
const ITEM_PATTERN = new RegExp(
  `^(COUNT|SUM|MIN|MAX)\\s*\\(\\s*(\\*|${IDENTIFIER})\\s*\\)(?:\\s+AS\\s+(${IDENTIFIER}))?$`,
  'i'
);
const PREDICATE_PATTERN = new RegExp(`^(${IDENTIFIER})\\s*(>=|<=|=|>|<)\\s*(-?\\d+)$`);

function unquoteIdentifier(text: string): string {
  return text.startsWith('"') ? text.slice(1, -1).replace(/""/g, '"') : text;
}

/** Splits at top-level commas (outside parentheses and quotes). */
function splitSelectItems(text: string): string[] {
  const items: string[] = [];
  let depth = 0;
  let quote = '';
  let start = 0;
  for (let index = 0; index < text.length; index++) {
    const character = text[index];
    if (quote) {
      if (character === quote) quote = '';
    } else if (character === '"' || character === "'") quote = character;
    else if (character === '(') depth++;
    else if (character === ')') depth--;
    else if (character === ',' && depth === 0) {
      items.push(text.slice(start, index).trim());
      start = index + 1;
    }
  }
  items.push(text.slice(start).trim());
  return items;
}

function tightenRange(range: Integer64Range, operator: string, value: bigint): Integer64Range {
  const lower = (bound: Integer64Bound): Integer64Bound | undefined =>
    !range.lower ||
    bound.value > range.lower.value ||
    (bound.value === range.lower.value && !bound.inclusive)
      ? bound
      : range.lower;
  const upper = (bound: Integer64Bound): Integer64Bound | undefined =>
    !range.upper ||
    bound.value < range.upper.value ||
    (bound.value === range.upper.value && !bound.inclusive)
      ? bound
      : range.upper;
  switch (operator) {
    case '>':
      return {...range, lower: lower({value, inclusive: false})};
    case '>=':
      return {...range, lower: lower({value, inclusive: true})};
    case '<':
      return {...range, upper: upper({value, inclusive: false})};
    case '<=':
      return {...range, upper: upper({value, inclusive: true})};
    default:
      return {lower: lower({value, inclusive: true}), upper: upper({value, inclusive: true})};
  }
}

/**
 * Parses `SELECT COUNT|SUM|MIN|MAX(col | *) [AS alias], ... FROM table [WHERE col <op> integer
 * [AND ...]]` (BETWEEN is rewritten before this). Anything else returns null and keeps its normal
 * route. Names are matched case-insensitively against the Int64 and Uint64 columns of the table.
 */
export function parseWideSqlAggregate(
  sql: string,
  getWideColumns: (tableName: string) => readonly string[] | undefined
): WideSqlAggregatePlan | null {
  const statement = STATEMENT_PATTERN.exec(sql);
  if (!statement) return null;
  const tableName = unquoteIdentifier(statement[2]);
  const wideColumns = getWideColumns(tableName);
  if (!wideColumns?.length) return null;
  const resolveColumn = (text: string): string | undefined => {
    const name = unquoteIdentifier(text);
    return (
      wideColumns.find(candidate => candidate === name) ??
      wideColumns.find(candidate => candidate.toLowerCase() === name.toLowerCase())
    );
  };
  const items: WideSqlAggregateItem[] = [];
  for (const text of splitSelectItems(statement[1])) {
    const match = ITEM_PATTERN.exec(text);
    if (!match) return null;
    const functionName = match[1].toLowerCase() as WideSqlAggregateFunction;
    const isStar = match[2] === '*';
    if (isStar && functionName !== 'count') return null;
    const column = isStar ? null : (resolveColumn(match[2]) ?? '');
    if (column === '') return null;
    items.push({
      functionName,
      column,
      outputName: match[3]
        ? unquoteIdentifier(match[3])
        : column === null
          ? 'count_star()'
          : `${functionName}(${column})`
    });
  }
  if (!items.some(item => item.column !== null)) return null;
  if (!statement[3]) return {tableName, items};
  let range: Integer64Range = {};
  let filterColumn: string | undefined;
  for (const text of statement[3].split(/\s+AND\s+/i)) {
    const match = PREDICATE_PATTERN.exec(text.trim());
    if (!match) return null;
    const column = resolveColumn(match[1]);
    if (!column || (filterColumn && filterColumn !== column)) return null;
    filterColumn = column;
    range = tightenRange(range, match[2], BigInt(match[3]));
  }
  if (items.some(item => item.column !== null && item.column !== filterColumn)) return null;
  return {tableName, items, filter: {column: filterColumn!, range}};
}

/** Int64 and Uint64 column names of a table, for planning. */
export function listWideIntegerColumns(table: arrow.Table): string[] {
  return table.schema.fields
    .filter(field => arrow.DataType.isInt(field.type) && field.type.bitWidth === 64)
    .map(field => field.name);
}

/** Int32 when every value fits, otherwise exact decimal text (resident columns cannot hold Int64). */
function makeExactResultColumn(values: readonly (bigint | null)[]): arrow.Vector {
  const fits = values.every(
    value => value === null || (value >= -(2n ** 31n) && value < 2n ** 31n)
  );
  return fits
    ? arrow.vectorFromArray(
        values.map(value => (value === null ? null : Number(value))),
        new arrow.Int32()
      )
    : arrow.vectorFromArray(
        values.map(value => (value === null ? null : value.toString())),
        new arrow.Utf8()
      );
}

/** Exact values of a plan, one BigInt (or null) per item; also used by the node spec's reference path. */
export function selectWideSqlAggregateValues(
  plan: WideSqlAggregatePlan,
  summaries: ReadonlyMap<string, WideInteger64Summary>
): (bigint | null)[] {
  return plan.items.map(item => {
    const summary = item.column === null ? [...summaries.values()][0] : summaries.get(item.column)!;
    if (item.functionName === 'count')
      return BigInt(summary.count + (item.column === null && !plan.filter ? summary.nullCount : 0));
    if (!summary.count) return null;
    return item.functionName === 'sum'
      ? summary.sum
      : item.functionName === 'min'
        ? summary.minimum
        : summary.maximum;
  });
}

export function makeWideSqlAggregateTable(
  plan: WideSqlAggregatePlan,
  summaries: ReadonlyMap<string, WideInteger64Summary>
): arrow.Table {
  const values = selectWideSqlAggregateValues(plan, summaries);
  const columns: Record<string, arrow.Vector> = {};
  plan.items.forEach((item, index) => {
    columns[item.outputName] = makeExactResultColumn([values[index]]);
  });
  return new arrow.Table(columns);
}

/** Scans every referenced Int64 or Uint64 column once on the GPU and returns the one-row exact result. */
export async function executeWideSqlAggregate(
  device: Device,
  plan: WideSqlAggregatePlan,
  table: arrow.Table
): Promise<arrow.Table> {
  const summaries = new Map<string, WideInteger64Summary>();
  const columns = new Set(
    plan.items.map(item => item.column).filter((name): name is string => name !== null)
  );
  if (plan.filter) columns.add(plan.filter.column);
  for (const name of columns) {
    const column = table.getChild(name)!;
    const signed = (column.type as arrow.Int).isSigned;
    const vectors = makeWideNumericWordVectors(device, column);
    try {
      summaries.set(
        name,
        await aggregateInteger64OnGpu({
          device,
          words: vectors.words,
          signed,
          validity: vectors.validity,
          selectionExpression:
            plan.filter?.column === name
              ? makeInteger64RangeSelectionExpression(plan.filter.range, signed)
              : undefined
        })
      );
    } finally {
      vectors.words.destroy();
      vectors.validity?.destroy();
      vectors.unusedValidity?.destroy();
    }
  }
  return makeWideSqlAggregateTable(plan, summaries);
}
