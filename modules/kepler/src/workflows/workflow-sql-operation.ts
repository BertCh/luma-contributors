// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {normalizeWorkflowParameters} from './workflow-parameters';
import {
  WorkflowError,
  type WorkflowData,
  type WorkflowOperation,
  type WorkflowParameterSpec,
  type WorkflowPortKind
} from './workflow-types';

const MAXIMUM_QUERY_LENGTH = 4000;
const MAXIMUM_ROWS = 100000;
const MAXIMUM_VERTICES = 2000000;
const FORBIDDEN_FUNCTIONS = [
  'glob',
  'getenv',
  'read_text',
  'read_blob',
  'sniff_csv',
  'query',
  'query_table',
  'current_setting'
] as const;
const FORBIDDEN_KEYWORDS = [
  'INSERT',
  'UPDATE',
  'DELETE',
  'DROP',
  'CREATE',
  'ALTER',
  'TRUNCATE',
  'MERGE',
  'GRANT',
  'REVOKE',
  'CALL',
  'SET',
  'RESET',
  'VACUUM',
  'CHECKPOINT',
  'DETACH',
  'USE',
  'COPY',
  'ATTACH',
  'INSTALL',
  'LOAD',
  'PRAGMA',
  'EXPORT',
  'IMPORT'
] as const;

type MaskedSqlText = {
  text: string;
  problem: string | null;
  lastQuotedEnd: number;
  quotedTokens: Array<{start: number; marker: "'" | '"'}>;
};

/** Preserve token boundaries while removing text that cannot be SQL syntax. */
function maskSqlText(query: string, preserveIdentifiers = false): MaskedSqlText {
  const characters = query.split('');
  const quotedTokens: MaskedSqlText['quotedTokens'] = [];
  let index = 0;
  let lastQuotedEnd = 0;
  while (index < query.length) {
    const character = query[index];
    if (character === "'" || character === '"') {
      const start = index++;
      let closed = false;
      while (index < query.length) {
        if (query[index++] !== character) continue;
        if (query[index] === character) {
          index++;
        } else {
          closed = true;
          break;
        }
      }
      if (!closed) {
        return {
          text: '',
          lastQuotedEnd,
          quotedTokens,
          problem:
            character === "'"
              ? 'must close every string literal'
              : 'must close every quoted identifier'
        };
      }
      lastQuotedEnd = index;
      quotedTokens.push({start, marker: character});
      if (character === "'" || !preserveIdentifiers) characters.fill(' ', start, index);
    } else if (character === '-' && query[index + 1] === '-') {
      const start = index;
      while (index < query.length && query[index] !== '\n' && query[index] !== '\r') index++;
      characters.fill(' ', start, index);
    } else if (character === '/' && query[index + 1] === '*') {
      const start = index;
      index += 2;
      let depth = 1;
      while (index < query.length && depth > 0) {
        if (query[index] === '/' && query[index + 1] === '*') {
          depth++;
          index += 2;
        } else if (query[index] === '*' && query[index + 1] === '/') {
          depth--;
          index += 2;
        } else {
          index++;
        }
      }
      if (depth > 0) {
        return {text: '', problem: 'must close every block comment', lastQuotedEnd, quotedTokens};
      }
      characters.fill(' ', start, index);
    } else {
      index++;
    }
  }
  return {text: characters.join(''), problem: null, lastQuotedEnd, quotedTokens};
}

function hasQuotedTableName(masked: MaskedSqlText): boolean {
  const characters = masked.text.split('');
  for (const {start, marker} of masked.quotedTokens) characters[start] = marker;
  const scopes = [{isQuery: false, inFromList: false, expectsTable: false}];
  for (const match of characters.join('').matchAll(/\b[a-z_][a-z0-9_]*\b|['"(),;]/gi)) {
    const token = match[0].toUpperCase();
    const scope = scopes[scopes.length - 1];
    if (token === '(') {
      // A table grouping inherits FROM; function arguments start their own expression scope.
      scopes.push({
        isQuery: false,
        inFromList: scope.expectsTable,
        expectsTable: scope.expectsTable
      });
      scope.expectsTable = false;
    } else if (token === ')') {
      if (scopes.length > 1) scopes.pop();
    } else if (token === 'SELECT') {
      scope.isQuery = true;
      scope.inFromList = false;
      scope.expectsTable = false;
    } else if (token === 'FROM' && scope.isQuery) {
      scope.inFromList = true;
      scope.expectsTable = true;
    } else if ((token === 'JOIN' || token === ',') && scope.inFromList) {
      scope.expectsTable = true;
    } else if (
      [
        'WHERE',
        'GROUP',
        'HAVING',
        'ORDER',
        'LIMIT',
        'OFFSET',
        'UNION',
        'EXCEPT',
        'INTERSECT',
        'QUALIFY',
        'WINDOW',
        ';'
      ].includes(token)
    ) {
      scope.inFromList = false;
      scope.expectsTable = false;
    } else if (token !== 'LATERAL' || !scope.expectsTable) {
      if (token === "'" && scope.expectsTable) return true;
      scope.expectsTable = false;
    }
  }
  return false;
}

function wrapSqlFailure(cause: unknown): WorkflowError {
  try {
    if (cause instanceof WorkflowError) return cause;
    const message = cause instanceof Error ? cause.message : String(cause);
    return new WorkflowError('operation-failed', `SQL: ${message}`, {cause});
  } catch {
    // A host can reject with a value whose message or string conversion throws.
    return new WorkflowError('operation-failed', 'SQL: Unknown error', {cause});
  }
}

export const WORKFLOW_SQL_OPERATION_ID = 'table-sql';

/**
 * Why a query is not an acceptable workflow SQL statement, or null. Accepts exactly one `SELECT`
 * or `WITH ... SELECT` statement; refuses a second statement, comments that hide one, data
 * modification and definition keywords, file and network readers (`read_csv`, `read_parquet`,
 * `httpfs`, `COPY`, `ATTACH`, `INSTALL`, `LOAD`, `PRAGMA`, `EXPORT`, `IMPORT`) and queries over
 * 4,000 characters.
 */
export function getWorkflowSqlProblem(query: string): string | null {
  if (query.length > MAXIMUM_QUERY_LENGTH) {
    return `must be at most ${MAXIMUM_QUERY_LENGTH} characters; got ${query.length}`;
  }
  const masked = maskSqlText(query);
  const {text, problem, lastQuotedEnd} = masked;
  if (problem) return problem;
  if (!/^\s*(SELECT|WITH)\b/i.test(text) || !/\bSELECT\b/i.test(text)) {
    return 'must be a single SELECT statement';
  }
  const semicolon = text.indexOf(';');
  if (semicolon !== -1 && (text.slice(semicolon + 1).trim() !== '' || lastQuotedEnd > semicolon)) {
    return 'must be a single SELECT statement';
  }
  if (hasQuotedTableName(masked)) {
    return 'may not read files or URLs (a quoted file name after FROM)';
  }
  for (const match of text.matchAll(/\b[a-z_][a-z0-9_]*\b/gi)) {
    const name = match[0].toLowerCase();
    if (
      name === 'httpfs' ||
      name === 'read_csv' ||
      name === 'read_parquet' ||
      ((FORBIDDEN_FUNCTIONS.some(functionName => functionName === name) ||
        name.startsWith('read_') ||
        name.endsWith('_scan')) &&
        /^\s*\(/.test(text.slice(match.index + name.length)))
    ) {
      return `may not read files or URLs (${name})`;
    }
    if (FORBIDDEN_KEYWORDS.some(keyword => keyword === name.toUpperCase())) {
      return `may not use ${name.toUpperCase()}`;
    }
  }
  return null;
}

/** Table names (`input1`, `input2`, `input3`) a query refers to, in order of first use. */
export function getWorkflowSqlTableNames(query: string): string[] {
  const {text} = maskSqlText(query, true);
  return [...new Set([...text.matchAll(/\binput[123]\b/gi)].map(match => match[0].toLowerCase()))];
}

/** `table-sql`: one read-only SQL statement over up to three inputs, run by the host. */
export function getWorkflowSqlOperations(): readonly WorkflowOperation[] {
  const parameters: readonly WorkflowParameterSpec[] = [
    {
      name: 'query',
      label: 'Query',
      kind: 'text',
      maximumLength: MAXIMUM_QUERY_LENGTH,
      defaultValue: 'SELECT * FROM input1'
    },
    {
      name: 'maximumRows',
      label: 'Maximum rows',
      kind: 'integer',
      minimum: 1,
      maximum: MAXIMUM_ROWS,
      defaultValue: MAXIMUM_ROWS
    }
  ];
  const accepts: readonly WorkflowPortKind[] = ['point', 'line', 'polygon', 'features', 'table'];
  return [
    {
      id: WORKFLOW_SQL_OPERATION_ID,
      title: 'SQL',
      category: 'table',
      summary: 'Query up to three upstream tables with one read-only SQL statement.',
      origin: 'builtin',
      version: '1',
      engine: 'host',
      inputs: ['input1', 'input2', 'input3'].map((name, index) => ({
        name,
        label: name,
        accepts: [...accepts],
        maximumRows: MAXIMUM_ROWS,
        maximumVertices: MAXIMUM_VERTICES,
        ...(index > 0 ? {optional: true} : {})
      })),
      parameters: parameters.map(parameter => ({...parameter})),
      outputs: [
        {
          name: 'result',
          label: 'Result',
          kind: 'features',
          identityColumns: [],
          maximumRows: MAXIMUM_ROWS
        }
      ],
      capabilityIds: [],
      contributorNames: [],
      declaredBehavior: [
        'Runs one read-only SELECT or WITH statement over input1, input2 and input3.',
        'Geometry is available as GeoJSON text in its input column.'
      ],
      limitation:
        'Requires a host SQL engine; file and URL readers are refused. At most 100000 rows.',
      normalizeParameters: raw => {
        const normalized = normalizeWorkflowParameters(parameters, raw, {operationTitle: 'SQL'});
        const problem = getWorkflowSqlProblem(normalized['query'] as string);
        if (problem) {
          throw new WorkflowError('parameter-out-of-range', `SQL: Query ${problem}.`, {
            parameter: 'query'
          });
        }
        return normalized;
      },
      run: async (context, inputs, normalized) => {
        if (!context.host.runSql) {
          throw new WorkflowError(
            'host-capability-missing',
            'SQL: this application has no SQL engine.'
          );
        }
        const query = normalized['query'] as string;
        const maximumRows = normalized['maximumRows'] as number;
        const tables: Record<string, WorkflowData> = {};
        for (const name of getWorkflowSqlTableNames(query)) {
          if (!Object.hasOwn(inputs, name)) {
            throw new WorkflowError('input-required', `SQL: ${name} must be connected.`, {
              port: name
            });
          }
          tables[name] = inputs[name];
        }
        let data: WorkflowData;
        try {
          data = await context.host.runSql({query, tables, maximumRows}, context.signal);
        } catch (cause) {
          throw wrapSqlFailure(cause);
        }
        if (data.table.rowCount > maximumRows) {
          throw new WorkflowError(
            'capacity-exceeded',
            `SQL: Result must have at most ${maximumRows} rows (Maximum rows); got ${data.table.rowCount}.`,
            {port: 'result'}
          );
        }
        return {
          outputs: {result: data},
          diagnostics: [],
          summary: {rows: data.table.rowCount, columns: Object.keys(data.table.columns).length},
          engine: 'host',
          dispatchedContributors: []
        };
      }
    }
  ];
}
