// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createResidentJoinPlan,
  type ResidentJoinDuplicatePolicy,
  type ResidentJoinPlan,
  type ResidentJoinSourceMetadata,
  type ResidentJoinType
} from './resident-join-plan';
import {decodeResidentSqlIdentifier, tokenizeResidentSql} from './resident-sql-strings';

/**
 * SQL joins that need right-side columns or duplicate right keys run on the resident join executor
 * (GPU key canonicalization, duplicate-aware range probe, output-parallel fill, GPU gathers of left
 * and right columns). This file parses the statement and plans the executor's inputs; it never
 * touches the GPU.
 */
export type ResidentSqlJoinPolicy = {
  /** Which right rows join a left row with a repeated key: every match, the first, or the last. */
  readonly duplicateRightMatches: ResidentJoinDuplicatePolicy;
  /** Appended to a right column whose output name is already taken (then a counter). */
  readonly rightColumnSuffix: string;
  /** Omit the right key column when it repeats the left key's name (always so under USING). */
  readonly dropRightKey: boolean;
};

export const DEFAULT_RESIDENT_SQL_JOIN_POLICY: ResidentSqlJoinPolicy = {
  duplicateRightMatches: 'all',
  rightColumnSuffix: '_right',
  dropRightKey: false
};

/** Reads `-- luma: duplicateRightMatches=first, rightColumnSuffix=_r, dropRightKey=true`. */
export function readResidentSqlJoinPolicyDirective(sql: string): Partial<ResidentSqlJoinPolicy> {
  const policy: {
    duplicateRightMatches?: ResidentJoinDuplicatePolicy;
    rightColumnSuffix?: string;
    dropRightKey?: boolean;
  } = {};
  for (const line of sql.matchAll(/^\s*--\s*luma:\s*(.+)$/gim)) {
    for (const part of line[1].split(',')) {
      const [key, value = 'true'] = part
        .trim()
        .split('=')
        .map(text => text.trim());
      if (key === 'duplicateRightMatches') {
        if (value !== 'all' && value !== 'first' && value !== 'last')
          throw new Error(`duplicateRightMatches must be all, first or last (found ${value})`);
        policy.duplicateRightMatches = value;
      } else if (key === 'rightColumnSuffix') policy.rightColumnSuffix = value;
      else if (key === 'dropRightKey') policy.dropRightKey = value !== 'false';
      else if (key) throw new Error(`Unknown SQL join directive "${key}"`);
    }
  }
  return policy;
}

type ColumnReference = {qualifier?: string; name: string};
type JoinSelectItem =
  | {star: true; qualifier?: string}
  | {star?: false; reference: ColumnReference; alias?: string};
export type ResidentSqlJoinStatement = {
  readonly items: readonly JoinSelectItem[];
  readonly left: {table: string; alias?: string};
  readonly right: {table: string; alias?: string};
  readonly type: 'INNER' | 'LEFT' | 'RIGHT' | 'FULL';
  readonly on: readonly [ColumnReference, ColumnReference];
  readonly using: boolean;
};

function readReference(text: string): ColumnReference {
  const parts = (text.match(/"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*/g) ?? []).map(
    decodeResidentSqlIdentifier
  );
  if (parts.length === 1) return {name: parts[0]};
  return {qualifier: parts[0].toLowerCase(), name: parts[1]};
}

/**
 * Parses `SELECT items FROM a [alias] [INNER|LEFT|RIGHT|FULL [OUTER]] JOIN b [alias] ON x = y |
 * USING (k)`. Returns null when the text has no JOIN, or a reason when the resident join
 * executor cannot run it (such statements go to the GPUSQL planner or DuckDB).
 */
export function parseResidentSqlJoin(
  sql: string
): ResidentSqlJoinStatement | {readonly reason: string} | null {
  const tokens = tokenizeResidentSql(sql);
  const word = (index: number) =>
    tokens[index]?.kind === 'word' ? tokens[index].text.toUpperCase() : '';
  if (word(0) !== 'SELECT' || !tokens.some(token => token.text.toUpperCase() === 'JOIN'))
    return null;
  const fromIndex = tokens.findIndex(token => token.text.toUpperCase() === 'FROM');
  if (fromIndex < 0) return {reason: 'Join statement has no FROM'};
  const items: JoinSelectItem[] = [];
  let cursor = 1;
  while (cursor < fromIndex) {
    if (tokens[cursor].text === '*') {
      items.push({star: true});
      cursor++;
    } else if (
      tokens[cursor].kind === 'word' &&
      tokens[cursor + 1]?.text === '.' &&
      tokens[cursor + 2]?.text === '*'
    ) {
      items.push({
        star: true,
        qualifier: decodeResidentSqlIdentifier(tokens[cursor].text).toLowerCase()
      });
      cursor += 3;
    } else if (tokens[cursor].kind === 'word' && tokens[cursor + 1]?.text !== '(') {
      const reference = readReference(tokens[cursor].text);
      cursor++;
      let alias: string | undefined;
      if (word(cursor) === 'AS' && tokens[cursor + 1]?.kind === 'word') {
        alias = decodeResidentSqlIdentifier(tokens[cursor + 1].text);
        cursor += 2;
      }
      items.push({reference, ...(alias ? {alias} : {})});
    } else {
      return {reason: 'Join select items must be columns or * (no expressions or aggregates)'};
    }
    if (cursor < fromIndex) {
      if (tokens[cursor].text !== ',') return {reason: 'Join select list could not be parsed'};
      cursor++;
    }
  }
  if (!items.length) return {reason: 'Join statement selects nothing'};
  const readTable = (index: number) => {
    if (tokens[index]?.kind !== 'word') return null;
    let next = index + 1;
    if (word(next) === 'AS') next++;
    const aliasToken = tokens[next];
    const hasAlias =
      aliasToken?.kind === 'word' &&
      !/^(INNER|LEFT|RIGHT|FULL|OUTER|SEMI|ANTI|CROSS|JOIN|ON|USING)$/i.test(aliasToken.text);
    return {
      table: decodeResidentSqlIdentifier(tokens[index].text),
      ...(hasAlias ? {alias: decodeResidentSqlIdentifier(aliasToken.text)} : {}),
      next: hasAlias ? next + 1 : next
    };
  };
  const left = readTable(fromIndex + 1);
  if (!left) return {reason: 'Join statement has no left table'};
  cursor = left.next;
  let type: ResidentSqlJoinStatement['type'] = 'INNER';
  if (word(cursor) === 'INNER') cursor++;
  else if (word(cursor) === 'LEFT' || word(cursor) === 'RIGHT' || word(cursor) === 'FULL') {
    type = word(cursor) as 'LEFT' | 'RIGHT' | 'FULL';
    cursor++;
    if (word(cursor) === 'OUTER') cursor++;
  } else if (word(cursor) === 'SEMI' || word(cursor) === 'ANTI' || word(cursor) === 'CROSS') {
    return {reason: `${word(cursor)} JOIN has no right columns; the GPUSQL planner runs it`};
  }
  if (word(cursor) !== 'JOIN') return {reason: 'Join statement has an unsupported join clause'};
  const right = readTable(cursor + 1);
  if (!right) return {reason: 'Join statement has no right table'};
  cursor = right.next;
  let on: [ColumnReference, ColumnReference];
  let using = false;
  if (word(cursor) === 'USING') {
    if (
      tokens[cursor + 1]?.text !== '(' ||
      tokens[cursor + 2]?.kind !== 'word' ||
      tokens[cursor + 3]?.text !== ')'
    )
      return {reason: 'USING supports exactly one column'};
    const key = readReference(tokens[cursor + 2].text);
    on = [key, key];
    using = true;
    cursor += 4;
  } else if (word(cursor) === 'ON') {
    if (
      tokens[cursor + 1]?.kind !== 'word' ||
      tokens[cursor + 2]?.text !== '=' ||
      tokens[cursor + 3]?.kind !== 'word'
    )
      return {reason: 'ON supports a single column equality'};
    on = [readReference(tokens[cursor + 1].text), readReference(tokens[cursor + 3].text)];
    cursor += 4;
  } else {
    return {reason: 'Join needs ON column = column or USING (column)'};
  }
  if (tokens[cursor]?.text === ';') cursor++;
  if (cursor < tokens.length)
    return {
      reason: `${tokens[cursor].text.toUpperCase()} after a join is not supported by the resident join executor (no WHERE, GROUP BY, ORDER BY or LIMIT on joined rows)`
    };
  return {
    items,
    left: {table: left.table, ...(left.alias ? {alias: left.alias} : {})},
    right: {table: right.table, ...(right.alias ? {alias: right.alias} : {})},
    type,
    on,
    using
  };
}

export type ResidentSqlJoinSource = {
  readonly metadata: ResidentJoinSourceMetadata;
  /** Column names in table order, including columns that are not resident. */
  readonly columnNames: readonly string[];
};

export type ResidentSqlJoinPlan = {
  readonly plan: ResidentJoinPlan;
  /** SQL output names in select order with the side each came from. */
  readonly outputs: readonly {name: string; side: 'left' | 'right'; sourceColumn: string}[];
  /** True when RIGHT JOIN was run as a LEFT JOIN with the tables swapped. */
  readonly swapped: boolean;
};

/**
 * Plans the resident join executor inputs for a parsed statement. Select order and SQL output
 * names replace the executor's default column order and `<label>_` prefixes. Throws the refusal
 * reason for unsupported key, column or size combinations.
 */
export function planResidentSqlJoin(props: {
  statement: ResidentSqlJoinStatement;
  findSource: (tableName: string) => ResidentSqlJoinSource | undefined;
  policy: ResidentSqlJoinPolicy;
  resultId: string;
}): ResidentSqlJoinPlan {
  const {statement, policy} = props;
  const sqlLeft = props.findSource(statement.left.table.toLowerCase());
  const sqlRight = props.findSource(statement.right.table.toLowerCase());
  if (!sqlLeft) throw new Error(`Table "${statement.left.table}" is not registered`);
  if (!sqlRight) throw new Error(`Table "${statement.right.table}" is not registered`);
  const qualifiersOf = (table: {table: string; alias?: string}) =>
    new Set([table.table.toLowerCase(), ...(table.alias ? [table.alias.toLowerCase()] : [])]);
  const sides = {
    left: {qualifiers: qualifiersOf(statement.left), source: sqlLeft},
    right: {qualifiers: qualifiersOf(statement.right), source: sqlRight}
  };
  const findColumn = (source: ResidentSqlJoinSource, name: string) => {
    const matches = source.columnNames.filter(
      candidate => candidate.toLowerCase() === name.toLowerCase()
    );
    return matches.length === 1 ? matches[0] : undefined;
  };
  const locate = (reference: ColumnReference): {side: 'left' | 'right'; column: string} => {
    if (reference.qualifier) {
      for (const side of ['left', 'right'] as const) {
        if (!sides[side].qualifiers.has(reference.qualifier)) continue;
        const column = findColumn(sides[side].source, reference.name);
        if (!column)
          throw new Error(`Column "${reference.name}" does not exist in ${reference.qualifier}`);
        return {side, column};
      }
      throw new Error(`Unknown table qualifier "${reference.qualifier}"`);
    }
    const inLeft = findColumn(sqlLeft, reference.name);
    const inRight = findColumn(sqlRight, reference.name);
    if (inLeft && inRight) throw new Error(`Column "${reference.name}" is ambiguous; qualify it`);
    if (inLeft) return {side: 'left', column: inLeft};
    if (inRight) return {side: 'right', column: inRight};
    throw new Error(`Column "${reference.name}" does not exist`);
  };
  let keyA: {side: 'left' | 'right'; column: string};
  let keyB: {side: 'left' | 'right'; column: string};
  if (statement.using) {
    const leftColumn = findColumn(sqlLeft, statement.on[0].name);
    const rightColumn = findColumn(sqlRight, statement.on[0].name);
    if (!leftColumn || !rightColumn)
      throw new Error(`USING column "${statement.on[0].name}" must exist in both tables`);
    keyA = {side: 'left', column: leftColumn};
    keyB = {side: 'right', column: rightColumn};
  } else {
    keyA = locate(statement.on[0]);
    keyB = locate(statement.on[1]);
  }
  if (keyA.side === keyB.side)
    throw new Error('Join ON must compare a left column with a right column');
  if (keyA.side === 'right') [keyA, keyB] = [keyB, keyA];

  // RIGHT JOIN is a LEFT JOIN with the tables swapped; select order restores the column order.
  const swapped = statement.type === 'RIGHT';
  const planSide = (side: 'left' | 'right'): 'left' | 'right' =>
    swapped ? (side === 'left' ? 'right' : 'left') : side;
  const type: ResidentJoinType = statement.type === 'RIGHT' ? 'LEFT' : statement.type;

  type Output = {name: string; side: 'left' | 'right'; sourceColumn: string; explicit: boolean};
  const selected: Output[] = [];
  const pushColumn = (side: 'left' | 'right', sourceColumn: string, alias?: string) => {
    if (selected.some(entry => entry.side === side && entry.sourceColumn === sourceColumn))
      throw new Error(
        `Column "${sourceColumn}" is selected twice; the resident join emits each column once`
      );
    selected.push({name: alias ?? sourceColumn, side, sourceColumn, explicit: alias !== undefined});
  };
  const policyDropsKey = (side: 'left' | 'right', column: string) =>
    side === 'right' &&
    column === keyB.column &&
    keyB.column.toLowerCase() === keyA.column.toLowerCase() &&
    (statement.using || policy.dropRightKey);
  for (const item of statement.items) {
    if (item.star) {
      for (const side of ['left', 'right'] as const) {
        if (item.qualifier && !sides[side].qualifiers.has(item.qualifier)) continue;
        for (const column of sides[side].source.columnNames) {
          if (!item.qualifier && policyDropsKey(side, column)) continue;
          pushColumn(side, column);
        }
      }
    } else {
      const located = locate(item.reference);
      pushColumn(located.side, located.column, item.alias);
    }
  }
  if (!selected.length) throw new Error('Join requires at least one output column');
  // Unique output names: only unaliased right columns may be suffixed.
  const used = new Set<string>();
  for (const entry of selected.filter(candidate => candidate.explicit)) {
    if (used.has(entry.name)) throw new Error(`Duplicate output name "${entry.name}"`);
    used.add(entry.name);
  }
  for (const entry of selected.filter(candidate => !candidate.explicit)) {
    let name = entry.name;
    if (used.has(name)) {
      name = `${entry.name}${policy.rightColumnSuffix}`;
      for (let counter = 2; used.has(name); counter++)
        name = `${entry.name}${policy.rightColumnSuffix}${counter}`;
    }
    used.add(name);
    entry.name = name;
  }

  const planLeft = swapped ? sqlRight : sqlLeft;
  const planRight = swapped ? sqlLeft : sqlRight;
  const leftColumns = selected
    .filter(entry => planSide(entry.side) === 'left')
    .map(entry => entry.sourceColumn);
  const rightColumns = selected
    .filter(entry => planSide(entry.side) === 'right')
    .map(entry => entry.sourceColumn);
  const plan = createResidentJoinPlan(
    {
      id: props.resultId,
      leftDataId: planLeft.metadata.datasetId,
      rightDataId: planRight.metadata.datasetId,
      leftField: swapped ? keyB.column : keyA.column,
      rightField: swapped ? keyA.column : keyB.column,
      type,
      leftColumns,
      rightColumns,
      duplicateRightMatches: policy.duplicateRightMatches,
      rightColumnSuffix: policy.rightColumnSuffix,
      allowSameDataset: true,
      resultId: props.resultId,
      resultLabel: 'SQL join result'
    },
    planLeft.metadata,
    planRight.metadata
  );
  // Reorder to select order and apply SQL output names.
  const ordered = selected.map(entry => {
    const column = plan.columns.find(
      candidate =>
        candidate.side === planSide(entry.side) && candidate.sourceColumn === entry.sourceColumn
    );
    if (!column) throw new Error(`Column "${entry.sourceColumn}" was dropped by the join planner`);
    return {column, entry};
  });
  const renamed = ordered.map(({column, entry}) => ({...column, outputColumn: entry.name}));
  const fieldByOutput = new Map(plan.provenance.fields.map(field => [field.name, field]));
  const fields = ordered.map(({column, entry}) => ({
    ...(fieldByOutput.get(column.outputColumn) ?? {
      type: column.type,
      analyzerType: column.analyzerType
    }),
    name: entry.name
  }));
  return {
    plan: {
      ...plan,
      columns: renamed,
      provenance: {
        ...plan.provenance,
        columns: renamed.map(column => ({
          side: column.side,
          column: column.sourceColumn,
          name: column.outputColumn
        })),
        fields
      }
    },
    outputs: selected.map(({name, side, sourceColumn}) => ({name, side, sourceColumn})),
    swapped
  };
}

type JoinSnapshotLike = {
  readonly id: string;
  readonly revision: number;
  readonly rowCount: number;
  readonly value: {
    readonly source: {readonly schema: {readonly fields: readonly {readonly name: string}[]}};
    readonly columns: Readonly<Record<string, {readonly format?: string} | undefined>>;
    readonly prepared: {
      readonly dictionaries: Readonly<
        Record<string, {readonly values: readonly string[]; readonly ordered: boolean}>
      >;
    };
    readonly analytics: {
      readonly nullCounts: Readonly<Record<string, readonly number[] | undefined>>;
    };
  };
};

/** Builds the join planner's source metadata from a resident dataset; types follow storage. */
export function makeResidentSqlJoinSource(snapshot: JoinSnapshotLike): ResidentSqlJoinSource {
  const {value} = snapshot;
  const columnNames = value.source.schema.fields.map(field => field.name);
  const columns: Record<string, {format?: string}> = Object.create(null);
  const nullCounts: Record<string, readonly number[]> = Object.create(null);
  const fields = columnNames.map(name => {
    const format = value.columns[name]?.format;
    if (format) columns[name] = {format};
    const counts = value.analytics.nullCounts[name];
    if (counts) nullCounts[name] = counts;
    const type = Object.hasOwn(value.prepared.dictionaries, name)
      ? 'string'
      : format === 'float32'
        ? 'real'
        : format
          ? 'integer'
          : 'unsupported';
    return {
      name,
      type,
      analyzerType: type === 'string' ? 'STRING' : type === 'real' ? 'FLOAT' : 'INT'
    };
  });
  return {
    columnNames,
    metadata: {
      datasetId: snapshot.id,
      label: snapshot.id,
      revision: snapshot.revision,
      rowCount: snapshot.rowCount,
      fields,
      columns,
      dictionaries: value.prepared.dictionaries,
      nullCounts
    }
  };
}
