// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {CompiledGPUDataFrameAggregation, CompiledGPUDataFrameGlobalSort, CompiledGPUDataFrameGroupedAggregation, CompiledGPUDataFrameQuery} from '@luma.gl/graph/gpu-dataframe';

/** Scalar storage formats a resident Kepler dataset can adopt. */
export const RESIDENT_SQL_FORMATS: ReadonlySet<string> = new Set(['float32', 'sint32', 'uint32']);

/** How a compiled GPUSQL plan's output reaches a resident dataset. */
export type ResidentSqlMaterializationPlan =
  /** Per-source-batch compacted row ids: one gather per batch, chunk boundaries preserved. */
  | {readonly kind: 'batches'}
  /** Globally ordered row ids addressing the whole source: one output chunk. */
  | {readonly kind: 'global-sort'}
  /** Dense GPU result columns of known length (global or grouped aggregates): copied on the GPU. */
  | {readonly kind: 'dense'; readonly rowCount: number}
  /** Not expressible as resident GPU buffers: bounded Arrow readback with a stated reason. */
  | {readonly kind: 'readback'; readonly reason: string};

type PlannedTable = {
  readonly table: {
    readonly numRows: number;
    readonly schema: {readonly fields: readonly {readonly name: string}[]};
    readonly gpuVectors: Readonly<Record<string, GPUVector | undefined>>;
  };
  readonly validity: Readonly<Record<string, GPUVector<'uint32'> | undefined>>;
};

/**
 * Chooses the materialization of a compiled plan. Aggregates are checked before the generic query
 * branch because they extend it but their rows are not source rows.
 */
export function planResidentSqlMaterialization(
  compiled: PlannedTable
): ResidentSqlMaterializationPlan {
  for (const field of compiled.table.schema.fields) {
    const vector = compiled.table.gpuVectors[field.name];
    if (!vector || !RESIDENT_SQL_FORMATS.has(vector.format ?? ''))
      return {kind: 'readback', reason: `column "${field.name}" has no resident storage format`};
    const columnValidity = compiled.validity[field.name];
    if (columnValidity && columnValidity.data.length !== vector.data.length)
      return {kind: 'readback', reason: `column "${field.name}" validity is batched differently`};
  }
  if (compiled instanceof CompiledGPUDataFrameGlobalSort) return {kind: 'global-sort'};
  if (
    compiled instanceof CompiledGPUDataFrameAggregation ||
    compiled instanceof CompiledGPUDataFrameGroupedAggregation
  )
    return {kind: 'dense', rowCount: compiled.table.numRows};
  if (compiled instanceof CompiledGPUDataFrameQuery) return {kind: 'batches'};
  return {kind: 'readback', reason: 'result has no GPU row identity'};
}

export type ResidentSqlBatchGather = {
  /** Source chunk the batch gathers from, or null when indices address the whole column. */
  readonly sourceChunkIndex: number | null;
  readonly rowCount: number;
};

/**
 * Turns per-batch selected counts into gathers. Empty batches are dropped (a resident chunk needs
 * at least one row); every non-empty batch keeps its own output chunk.
 */
export function planResidentSqlBatchGathers(counts: readonly number[]): {
  readonly gathers: readonly ResidentSqlBatchGather[];
  readonly rowCount: number;
} {
  const gathers: ResidentSqlBatchGather[] = [];
  let rowCount = 0;
  counts.forEach((count, sourceChunkIndex) => {
    if (!count) return;
    gathers.push({sourceChunkIndex, rowCount: count});
    rowCount += count;
  });
  return {gathers, rowCount};
}

const SIMPLE_IDENTIFIER = '(?:"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*)';
const CTE_BODY_PATTERN = new RegExp(
  `^\\s*SELECT\\s+(\\*|${SIMPLE_IDENTIFIER}(?:\\s*,\\s*${SIMPLE_IDENTIFIER})*)\\s+FROM\\s+(${SIMPLE_IDENTIFIER})(?:\\s+WHERE\\s+([\\s\\S]+?))?\\s*;?\\s*$`,
  'i'
);
const CLAUSE_KEYWORDS = /\b(GROUP|ORDER|LIMIT)\b|;/i;

/** Finds the index of the parenthesis closing the one opened at `start`, skipping quoted text. */
function findClosingParenthesis(sql: string, start: number): number {
  let depth = 0;
  let quote = '';
  for (let index = start; index < sql.length; index++) {
    const character = sql[index];
    if (quote) {
      if (character === quote) quote = '';
    } else if (character === "'" || character === '"') quote = character;
    else if (character === '(') depth++;
    else if (character === ')' && --depth === 0) return index;
  }
  return -1;
}

type SimpleCte = {name: string; star: boolean; base: string; predicate?: string};

/** Splices one simple CTE into a SELECT, or returns null when it cannot be merged exactly. */
function mergeSimpleCte(target: string, cte: SimpleCte): string | null {
  const escapedName = cte.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (new RegExp(`${escapedName}\\s*\\.`, 'i').test(target)) return null;
  if (cte.star === false && /^\s*SELECT\s+\*/i.test(target)) return null;
  const fromPattern = new RegExp(`\\bFROM\\s+${escapedName}(?![A-Za-z0-9_"])`, 'i');
  const joinPattern = new RegExp(`\\bJOIN\\s+${escapedName}(?![A-Za-z0-9_"])`, 'i');
  let merged = target;
  if (joinPattern.test(merged)) {
    // A filtered CTE on the join side would change which right rows match.
    if (cte.predicate) return null;
    merged = merged.replace(joinPattern, `JOIN ${cte.base}`);
  }
  if (fromPattern.test(merged)) {
    merged = merged.replace(fromPattern, `FROM ${cte.base}`);
    if (cte.predicate) {
      const whereMatch = /\bWHERE\b/i.exec(merged);
      if (whereMatch) {
        const tail = merged.slice(whereMatch.index + 5);
        const endMatch = CLAUSE_KEYWORDS.exec(tail);
        const end = endMatch ? endMatch.index : tail.length;
        merged = `${merged.slice(0, whereMatch.index)}WHERE (${cte.predicate}) AND (${tail.slice(0, end).trim()}) ${tail.slice(end)}`;
      } else {
        const endMatch = CLAUSE_KEYWORDS.exec(merged);
        const end = endMatch ? endMatch.index : merged.length;
        merged = `${merged.slice(0, end).trimEnd()} WHERE ${cte.predicate} ${merged.slice(end)}`;
      }
    }
  }
  return merged;
}

/**
 * Inlines non-recursive CTEs of the form `name AS (SELECT * | columns FROM table [WHERE p])` into
 * the statements that use them, so GPU planning sees a plain SELECT. CTEs with expressions,
 * aliases, aggregates, joins, ordering or limits, recursive CTEs, and qualified references to the
 * CTE name are left alone, so the statement keeps its original text and routes as before.
 */
export function inlineResidentSqlCtes(sql: string): string {
  const withMatch = /^\s*WITH\s+/i.exec(sql);
  if (!withMatch || /^\s*WITH\s+RECURSIVE\b/i.test(sql)) return sql;
  let position = withMatch[0].length;
  const definitions: {name: string; body: string}[] = [];
  while (true) {
    const header = new RegExp(`^(${SIMPLE_IDENTIFIER})\\s+AS\\s*\\(`, 'i').exec(
      sql.slice(position)
    );
    if (!header) return sql;
    const open = position + header[0].length - 1;
    const close = findClosingParenthesis(sql, open);
    if (close < 0) return sql;
    definitions.push({name: header[1], body: sql.slice(open + 1, close)});
    position = close + 1;
    const separator = /^\s*,\s*/.exec(sql.slice(position));
    if (!separator) break;
    position += separator[0].length;
  }
  let main = sql.slice(position);
  if (!/^\s*SELECT\b/i.test(main)) return sql;
  const bodies = definitions.map(definition => definition.body);
  for (let index = 0; index < definitions.length; index++) {
    const match = CTE_BODY_PATTERN.exec(bodies[index]);
    if (!match) return sql;
    const cte: SimpleCte = {
      name: definitions[index].name,
      star: match[1] === '*',
      base: match[2],
      ...(match[3] ? {predicate: match[3].trim()} : {})
    };
    for (let later = index + 1; later < bodies.length; later++) {
      const merged = mergeSimpleCte(bodies[later], cte);
      if (merged === null) return sql;
      bodies[later] = merged;
    }
    const mergedMain = mergeSimpleCte(main, cte);
    if (mergedMain === null) return sql;
    main = mergedMain;
  }
  return main.trim();
}
