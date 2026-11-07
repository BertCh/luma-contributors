// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  decodeResidentSqlIdentifier,
  quoteResidentSqlIdentifier,
  tokenizeResidentSql
} from './resident-sql-strings';

/**
 * The dataframe aggregate path has no ORDER BY or LIMIT. A grouped or global aggregate followed
 * by `ORDER BY <output column> [ASC|DESC] [NULLS FIRST|LAST]` and/or `LIMIT n` therefore runs as
 * two stages: the aggregate, then a plain `SELECT * FROM <aggregate result> ORDER BY .. LIMIT ..`
 * over its (small, resident) result. This file is pure text analysis.
 */
export type ResidentSqlAggregateTail = {
  /** The statement without its ORDER BY / LIMIT. */
  readonly aggregateSql: string;
  readonly orderBy?: {
    readonly column: string;
    readonly direction: 'ASC' | 'DESC';
    readonly nulls?: 'FIRST' | 'LAST';
  };
  readonly limit?: number;
};

const AGGREGATE_FUNCTIONS = /^(COUNT|SUM|MIN|MAX|AVG|MEAN)$/i;

/** Splits an aggregate statement from its ORDER BY / LIMIT, or returns null when not applicable. */
export function splitResidentSqlAggregateTail(sql: string): ResidentSqlAggregateTail | null {
  const tokens = tokenizeResidentSql(sql);
  const wordAt = (index: number) =>
    tokens[index]?.kind === 'word' ? tokens[index].text.toUpperCase() : '';
  if (wordAt(0) !== 'SELECT') return null;
  let depth = 0;
  let fromIndex = -1;
  let tailIndex = -1;
  for (let index = 1; index < tokens.length; index++) {
    const text = tokens[index].text;
    if (text === '(') depth++;
    else if (text === ')') depth--;
    else if (depth === 0) {
      const word = wordAt(index);
      if (word === 'SELECT') return null;
      if (word === 'FROM' && fromIndex < 0) fromIndex = index;
      if ((word === 'ORDER' && wordAt(index + 1) === 'BY') || word === 'LIMIT') {
        tailIndex = index;
        break;
      }
    }
  }
  if (fromIndex < 0 || tailIndex < 0) return null;
  // Select items: output names and whether any item calls an aggregate.
  const names = new Map<string, string>();
  let hasAggregate = false;
  depth = 0;
  let itemStart = 1;
  for (let index = 1; index <= fromIndex; index++) {
    const text = tokens[index].text;
    if (text === '(') depth++;
    else if (text === ')') depth--;
    if (index === fromIndex || (text === ',' && depth === 0)) {
      const item = tokens.slice(itemStart, index);
      itemStart = index + 1;
      if (
        item.some(
          (token, offset) =>
            token.kind === 'word' &&
            AGGREGATE_FUNCTIONS.test(token.text) &&
            item[offset + 1]?.text === '('
        )
      )
        hasAggregate = true;
      const last = item[item.length - 1];
      if (item.length >= 3 && item[item.length - 2].text.toUpperCase() === 'AS')
        names.set(
          decodeResidentSqlIdentifier(last.text).toLowerCase(),
          decodeResidentSqlIdentifier(last.text)
        );
      else if (item.length === 1 && last.kind === 'word') {
        const name = decodeResidentSqlIdentifier(last.text.split('.').pop() ?? last.text);
        names.set(name.toLowerCase(), name);
      }
    }
  }
  if (!hasAggregate) return null;
  let cursor = tailIndex;
  let orderBy: ResidentSqlAggregateTail['orderBy'];
  if (wordAt(cursor) === 'ORDER') {
    const token = tokens[cursor + 2];
    if (token?.kind !== 'word') return null;
    const column = names.get(
      decodeResidentSqlIdentifier(token.text.split('.').pop() ?? token.text).toLowerCase()
    );
    if (!column) return null;
    cursor += 3;
    let direction: 'ASC' | 'DESC' = 'ASC';
    if (wordAt(cursor) === 'ASC' || wordAt(cursor) === 'DESC') {
      direction = wordAt(cursor) as 'ASC' | 'DESC';
      cursor++;
    }
    let nulls: 'FIRST' | 'LAST' | undefined;
    if (
      wordAt(cursor) === 'NULLS' &&
      (wordAt(cursor + 1) === 'FIRST' || wordAt(cursor + 1) === 'LAST')
    ) {
      nulls = wordAt(cursor + 1) as 'FIRST' | 'LAST';
      cursor += 2;
    }
    orderBy = {column, direction, ...(nulls ? {nulls} : {})};
  }
  let limit: number | undefined;
  if (wordAt(cursor) === 'LIMIT') {
    if (tokens[cursor + 1]?.kind !== 'number' || !/^\d+$/.test(tokens[cursor + 1].text))
      return null;
    limit = Number(tokens[cursor + 1].text);
    cursor += 2;
  }
  if (tokens[cursor]?.text === ';') cursor++;
  if (cursor !== tokens.length) return null;
  return {
    aggregateSql: sql.slice(0, tokens[tailIndex].start).trimEnd(),
    ...(orderBy ? {orderBy} : {}),
    ...(limit !== undefined ? {limit} : {})
  };
}

/** The second-stage statement over the registered aggregate result. */
export function makeResidentSqlAggregateTailSql(
  tail: ResidentSqlAggregateTail,
  tableName: string
): string {
  const order = tail.orderBy
    ? ` ORDER BY ${quoteResidentSqlIdentifier(tail.orderBy.column)} ${tail.orderBy.direction}${
        tail.orderBy.nulls ? ` NULLS ${tail.orderBy.nulls}` : ''
      }`
    : '';
  return `SELECT * FROM ${tableName}${order}${tail.limit !== undefined ? ` LIMIT ${tail.limit}` : ''}`;
}
