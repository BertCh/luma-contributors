// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  makeResidentJoinCompositeKeyPlan,
  RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN,
  MAXIMUM_RESIDENT_JOIN_COMPOSITE_KEYS,
  type ResidentJoinCompositeKeyPlan
} from './resident-join-composite-key';
import {
  planResidentSqlJoin,
  type ResidentSqlJoinPlan,
  type ResidentSqlJoinPolicy,
  type ResidentSqlJoinSource,
  type ResidentSqlJoinStatement
} from './resident-sql-join';
import {
  applyResidentSqlEdits,
  decodeResidentSqlIdentifier,
  quoteResidentSqlIdentifier,
  tokenizeResidentSql
} from './resident-sql-strings';

/**
 * Composed joins: `SELECT exprs FROM a [INNER|LEFT|RIGHT|FULL|SEMI|ANTI] JOIN b ON k1 = k1 [AND ..]
 * [WHERE ..] [GROUP BY ..] [ORDER BY ..] [LIMIT n]`. The join itself runs on the resident join
 * executor into an intermediate dataset holding every column the statement references; the rest
 * of the statement is rewritten to read that intermediate and runs through the dataframe path.
 * This file is pure (parse, plan, text rewrite); the runtime runs the two stages.
 */
export const RESIDENT_SQL_COMPOSED_TABLE_NAME = 'luma_sql_joined';

export type ResidentSqlComposeJoinType = 'INNER' | 'LEFT' | 'RIGHT' | 'FULL' | 'SEMI' | 'ANTI';
type ComposeReference = {qualifier?: string; name: string};
type ComposeTable = {table: string; alias?: string};

export type ResidentSqlComposeStatement = {
  /** The statement text the token offsets refer to (CTEs already inlined). */
  readonly sql: string;
  readonly left: ComposeTable;
  readonly right: ComposeTable;
  readonly type: ResidentSqlComposeJoinType;
  /** Key equalities in written order; USING pairs repeat one name on both sides. */
  readonly keys: readonly [ComposeReference, ComposeReference][];
  readonly using: boolean;
  /** Token index range of `FROM a JOIN b ON ...` and the select list. */
  readonly fromStart: number;
  readonly joinEnd: number;
};

export type ResidentSqlComposedPlan = {
  readonly joinPlan: ResidentSqlJoinPlan;
  /** Present for multi-column ON/USING keys. */
  readonly compositeKey?: ResidentJoinCompositeKeyPlan;
  readonly tableName: string;
  /**
   * `SELECT .. FROM <tableName> ..`: the statement the dataframe path runs over the join. For an
   * INNER join on numeric multi-column keys this carries the hash verification residual.
   */
  readonly outerSql: string;
  /** Same statement without the hash verification residual, for exactly ranked composite keys. */
  readonly exactOuterSql: string;
  readonly joinType: ResidentSqlComposeJoinType;
};

const CORE_KEYWORDS = new Set(
  (
    'SELECT FROM WHERE GROUP BY ORDER HAVING LIMIT OFFSET AS AND OR NOT IN IS LIKE ILIKE BETWEEN ' +
    'ASC DESC DISTINCT CASE WHEN THEN ELSE END CAST ESCAPE NULLS ALL ON JOIN USING INNER LEFT RIGHT ' +
    'FULL OUTER SEMI ANTI CROSS EXISTS UNION QUALIFY'
  ).split(' ')
);
const SOFT_KEYWORDS = new Set('NULL TRUE FALSE FIRST LAST DATE TIMESTAMP INTERVAL'.split(' '));
const CLAUSE_KEYWORDS = new Set([
  'WHERE',
  'GROUP',
  'HAVING',
  'ORDER',
  'LIMIT',
  'OFFSET',
  'QUALIFY'
]);

function readReference(text: string): ComposeReference {
  const parts = (text.match(/"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*/g) ?? []).map(
    decodeResidentSqlIdentifier
  );
  return parts.length === 1
    ? {name: parts[0]}
    : {qualifier: parts[0].toLowerCase(), name: parts[1]};
}

/**
 * Parses the join head of a statement with a trailing clause, multi-column keys or SEMI/ANTI.
 * Returns null when the text is not a join, or a reason when its shape is unsupported.
 */
export function parseResidentSqlCompose(
  sql: string
): ResidentSqlComposeStatement | {readonly reason: string} | null {
  const tokens = tokenizeResidentSql(sql);
  const upper = (index: number) =>
    tokens[index]?.kind === 'word' ? tokens[index].text.toUpperCase() : '';
  if (upper(0) !== 'SELECT' || !tokens.some(token => token.text.toUpperCase() === 'JOIN'))
    return null;
  if (tokens.filter(token => token.text.toUpperCase() === 'SELECT').length > 1)
    return {reason: 'Subqueries in joined statements are not supported on the GPU'};
  const fromIndex = tokens.findIndex(token => token.text.toUpperCase() === 'FROM');
  if (fromIndex < 0) return {reason: 'Join statement has no FROM'};
  const readTable = (index: number) => {
    if (tokens[index]?.kind !== 'word') return null;
    let next = index + 1;
    if (upper(next) === 'AS') next++;
    const aliasToken = tokens[next];
    const hasAlias =
      aliasToken?.kind === 'word' &&
      !/^(INNER|LEFT|RIGHT|FULL|OUTER|SEMI|ANTI|JOIN|ON|USING|WHERE)$/i.test(aliasToken.text);
    return {
      table: decodeResidentSqlIdentifier(tokens[index].text),
      ...(hasAlias ? {alias: decodeResidentSqlIdentifier(aliasToken.text)} : {}),
      next: hasAlias ? next + 1 : next
    };
  };
  const left = readTable(fromIndex + 1);
  if (!left) return {reason: 'Join statement has no left table'};
  let cursor = left.next;
  let type: ResidentSqlComposeJoinType = 'INNER';
  const typeWord = upper(cursor);
  if (typeWord === 'INNER') cursor++;
  else if (typeWord === 'LEFT' || typeWord === 'RIGHT' || typeWord === 'FULL') {
    type = typeWord;
    cursor++;
    if (upper(cursor) === 'OUTER') cursor++;
    else if (upper(cursor) === 'SEMI' || upper(cursor) === 'ANTI') {
      type = upper(cursor) as 'SEMI' | 'ANTI';
      cursor++;
    }
  } else if (typeWord === 'SEMI' || typeWord === 'ANTI') {
    type = typeWord;
    cursor++;
  } else if (typeWord === 'CROSS') return {reason: 'CROSS JOIN is not supported on the GPU'};
  if (upper(cursor) !== 'JOIN') return {reason: 'Join statement has an unsupported join clause'};
  const right = readTable(cursor + 1);
  if (!right) return {reason: 'Join statement has no right table'};
  cursor = right.next;
  const keys: [ComposeReference, ComposeReference][] = [];
  let using = false;
  if (upper(cursor) === 'USING') {
    using = true;
    if (tokens[cursor + 1]?.text !== '(') return {reason: 'USING needs a column list'};
    cursor += 2;
    while (tokens[cursor]?.kind === 'word') {
      const key = readReference(tokens[cursor].text);
      keys.push([key, key]);
      cursor++;
      if (tokens[cursor]?.text === ',') cursor++;
      else break;
    }
    if (tokens[cursor]?.text !== ')' || !keys.length)
      return {reason: 'USING list could not be parsed'};
    cursor++;
  } else if (upper(cursor) === 'ON') {
    cursor++;
    for (;;) {
      if (
        tokens[cursor]?.kind !== 'word' ||
        tokens[cursor + 1]?.text !== '=' ||
        tokens[cursor + 2]?.kind !== 'word'
      )
        return {reason: 'ON supports column = column equalities joined by AND'};
      keys.push([readReference(tokens[cursor].text), readReference(tokens[cursor + 2].text)]);
      cursor += 3;
      if (upper(cursor) === 'AND') cursor++;
      else break;
    }
  } else {
    return {reason: 'Join needs ON column = column or USING (columns)'};
  }
  if (keys.length > MAXIMUM_RESIDENT_JOIN_COMPOSITE_KEYS)
    return {
      reason: `Join supports at most ${MAXIMUM_RESIDENT_JOIN_COMPOSITE_KEYS} key columns on the GPU`
    };
  if (/^(JOIN|INNER|LEFT|RIGHT|FULL|CROSS|SEMI|ANTI|OR)$/.test(upper(cursor)))
    return {reason: `${upper(cursor)} after the join condition is not supported on the GPU`};
  if (tokens.slice(cursor).some(token => token.text.toUpperCase() === 'JOIN'))
    return {reason: 'Joins of more than two tables are not supported on the GPU'};
  return {
    sql,
    left: {table: left.table, ...(left.alias ? {alias: left.alias} : {})},
    right: {table: right.table, ...(right.alias ? {alias: right.alias} : {})},
    type,
    keys,
    using,
    fromStart: fromIndex,
    joinEnd: cursor
  };
}

/** Plans the join stage and rewrites the rest of the statement over the intermediate table. */
export function planResidentSqlCompose(props: {
  statement: ResidentSqlComposeStatement;
  findSource: (tableName: string) => ResidentSqlJoinSource | undefined;
  policy: ResidentSqlJoinPolicy;
  resultId: string;
  tableName?: string;
}): ResidentSqlComposedPlan {
  const {statement} = props;
  const {sql, type} = statement;
  const tableName = props.tableName ?? RESIDENT_SQL_COMPOSED_TABLE_NAME;
  const sqlLeft = props.findSource(statement.left.table.toLowerCase());
  const sqlRight = props.findSource(statement.right.table.toLowerCase());
  if (!sqlLeft) throw new Error(`Table "${statement.left.table}" is not registered`);
  if (!sqlRight) throw new Error(`Table "${statement.right.table}" is not registered`);
  const semiLike = type === 'SEMI' || type === 'ANTI';
  const qualifiersOf = (table: ComposeTable) =>
    new Set([table.table.toLowerCase(), ...(table.alias ? [table.alias.toLowerCase()] : [])]);
  const leftQualifiers = qualifiersOf(statement.left);
  const rightQualifiers = qualifiersOf(statement.right);
  const findColumn = (source: ResidentSqlJoinSource, name: string) => {
    const matches = source.columnNames.filter(
      candidate => candidate.toLowerCase() === name.toLowerCase()
    );
    return matches.length === 1 ? matches[0] : undefined;
  };
  type Located = {side: 'left' | 'right'; column: string};
  const locate = (reference: ComposeReference): Located => {
    if (reference.qualifier) {
      const side = leftQualifiers.has(reference.qualifier)
        ? 'left'
        : rightQualifiers.has(reference.qualifier)
          ? 'right'
          : undefined;
      if (!side) throw new Error(`Unknown table qualifier "${reference.qualifier}"`);
      if (type === 'SEMI' && side === 'right')
        // DuckDB's binder rejects this too: a semi join returns left rows only.
        throw new Error(`SEMI JOIN exposes only left columns; "${reference.name}" is on the right`);
      const column = findColumn(side === 'left' ? sqlLeft : sqlRight, reference.name);
      if (!column)
        throw new Error(`Column "${reference.name}" does not exist in ${reference.qualifier}`);
      return {side, column};
    }
    const inLeft = findColumn(sqlLeft, reference.name);
    // Unqualified names prefer the left side for ANTI (DuckDB exposes only the left columns).
    const inRight =
      type === 'SEMI' || (type === 'ANTI' && inLeft)
        ? undefined
        : findColumn(sqlRight, reference.name);
    if (inLeft && inRight) throw new Error(`Column "${reference.name}" is ambiguous; qualify it`);
    if (inLeft) return {side: 'left', column: inLeft};
    if (inRight) return {side: 'right', column: inRight};
    throw new Error(`Column "${reference.name}" does not exist`);
  };

  // Key pairs oriented as (SQL-left column, SQL-right column).
  const pairs = statement.keys.map(([first, second]) => {
    let a: Located;
    let b: Located;
    if (statement.using) {
      const leftColumn = findColumn(sqlLeft, first.name);
      const rightColumn = findColumn(sqlRight, first.name);
      if (!leftColumn || !rightColumn)
        throw new Error(`USING column "${first.name}" must exist in both tables`);
      a = {side: 'left', column: leftColumn};
      b = {side: 'right', column: rightColumn};
    } else {
      const locateKey = (reference: ComposeReference): Located => {
        const side = reference.qualifier
          ? leftQualifiers.has(reference.qualifier)
            ? 'left'
            : 'right'
          : findColumn(sqlLeft, reference.name)
            ? 'left'
            : 'right';
        const column = findColumn(side === 'left' ? sqlLeft : sqlRight, reference.name);
        if (!column) throw new Error(`Column "${reference.name}" does not exist`);
        return {side, column};
      };
      a = locateKey(first);
      b = locateKey(second);
      if (a.side === b.side)
        throw new Error('Join ON must compare a left column with a right column');
      if (a.side === 'right') [a, b] = [b, a];
    }
    return {leftColumn: a.column, rightColumn: b.column};
  });
  const keyedByRightName =
    statement.using || props.policy.dropRightKey
      ? new Set(
          pairs
            .filter(pair => pair.leftColumn.toLowerCase() === pair.rightColumn.toLowerCase())
            .map(pair => pair.rightColumn)
        )
      : new Set<string>();

  // Walk the select list and trailing clauses: collect references, stars and aliases.
  const tokens = tokenizeResidentSql(sql);
  const references = new Map<string, Located>();
  const addReference = (located: Located) => {
    const key = `${located.side}:${located.column}`;
    if (!references.has(key)) references.set(key, located);
    return key;
  };
  type Pending = {start: number; end: number; key?: string; star?: Located[]; append?: string};
  const pending: Pending[] = [];
  const aliases = new Set<string>();
  for (let index = 1; index < statement.fromStart; index++)
    if (
      tokens[index - 1]?.text.toUpperCase() === 'AS' &&
      tokens[index].kind === 'word' &&
      tokens[index + 1]?.text !== '('
    )
      aliases.add(tokens[index].text.toLowerCase());
  const visit = (from: number, to: number, inSelect: boolean) => {
    let depth = 0;
    let clause = inSelect ? 'SELECT' : '';
    for (let index = from; index < to; index++) {
      const token = tokens[index];
      const previous = tokens[index - 1]?.text.toUpperCase();
      if (token.text === '(') depth++;
      else if (token.text === ')') depth--;
      if (token.kind !== 'word' && token.text !== '*') continue;
      const word = token.text.toUpperCase();
      if (depth === 0 && CLAUSE_KEYWORDS.has(word)) {
        clause = word;
        continue;
      }
      if (token.text === '*') {
        const itemStart = previous === 'SELECT' || previous === ',' || previous === 'DISTINCT';
        if (!inSelect || depth !== 0 || !itemStart) continue;
        pending.push({start: token.start, end: token.end, star: expandStar(undefined)});
        continue;
      }
      if (
        tokens[index + 1]?.text === '.' &&
        tokens[index + 2]?.text === '*' &&
        inSelect &&
        depth === 0
      ) {
        pending.push({
          start: token.start,
          end: tokens[index + 2].end,
          star: expandStar(decodeResidentSqlIdentifier(token.text).toLowerCase())
        });
        index += 2;
        continue;
      }
      if (tokens[index + 1]?.text === '(' || previous === 'AS' || previous === '::') continue;
      const bare = !token.text.includes('.') || token.text.startsWith('"');
      if (bare && CORE_KEYWORDS.has(word)) continue;
      const reference = readReference(token.text);
      if (!reference.qualifier) {
        // A select alias is visible to GROUP BY / HAVING / ORDER BY, and shadows nothing in WHERE.
        if (clause !== 'WHERE' && clause !== 'SELECT' && aliases.has(reference.name.toLowerCase()))
          continue;
        if (SOFT_KEYWORDS.has(word)) {
          const located = tryLocate(reference);
          if (!located) continue;
        }
      }
      pending.push({start: token.start, end: token.end, key: addReference(locate(reference))});
    }
  };
  const tryLocate = (reference: ComposeReference): Located | undefined => {
    try {
      return locate(reference);
    } catch {
      return undefined;
    }
  };
  function expandStar(qualifier: string | undefined): Located[] {
    const columns: Located[] = [];
    // ANTI JOIN right columns are NULL for every kept row; only an explicit `b.*` selects them.
    for (const side of type === 'SEMI' || (type === 'ANTI' && !qualifier)
      ? (['left'] as const)
      : (['left', 'right'] as const)) {
      if (qualifier && !(side === 'left' ? leftQualifiers : rightQualifiers).has(qualifier))
        continue;
      for (const column of (side === 'left' ? sqlLeft! : sqlRight!).columnNames) {
        if (side === 'right' && !qualifier && keyedByRightName.has(column)) continue;
        columns.push({side, column});
      }
    }
    if (!columns.length) throw new Error(`Unknown table qualifier "${qualifier}"`);
    for (const located of columns) addReference(located);
    return columns;
  }
  visit(1, statement.fromStart, true);
  visit(statement.joinEnd, tokens.length, false);

  // Hidden references: the ANTI probe column and the keys of hash-verified composites.
  const needsComposite = pairs.length > 1;
  const swapped = type === 'RIGHT';
  const compositeKey = needsComposite
    ? makeResidentJoinCompositeKeyPlan({
        pairs: pairs.map(pair =>
          swapped
            ? {leftColumn: pair.rightColumn, rightColumn: pair.leftColumn}
            : {leftColumn: pair.leftColumn, rightColumn: pair.rightColumn}
        ),
        left: (swapped ? sqlRight : sqlLeft).metadata,
        right: (swapped ? sqlLeft : sqlRight).metadata
      })
    : undefined;
  if (type === 'ANTI') addReference({side: 'right', column: pairs[0].rightColumn});
  // A statement like COUNT(*) references no column; the join still needs one to gather.
  if (!references.size) addReference({side: 'left', column: pairs[0].leftColumn});
  // SEMI/ANTI never expose right columns, but the planner still gathers the probe key.
  const leftQualifier = (statement.left.alias ?? statement.left.table).toLowerCase();
  const rightQualifier = (statement.right.alias ?? statement.right.table).toLowerCase();
  const plannerStatement: ResidentSqlJoinStatement = {
    // Left columns first so they keep their names; colliding right columns take the suffix.
    items: [...references.values()]
      .sort((a, b) => Number(a.side === 'right') - Number(b.side === 'right'))
      .map(located => ({
        reference: {
          qualifier: located.side === 'left' ? leftQualifier : rightQualifier,
          name: located.column
        }
      })),
    left: statement.left,
    right: statement.right,
    type: type === 'SEMI' ? 'INNER' : type === 'ANTI' ? 'LEFT' : type,
    on: [
      {qualifier: leftQualifier, name: pairs[0].leftColumn},
      {qualifier: rightQualifier, name: pairs[0].rightColumn}
    ],
    using: false
  };
  if (!plannerStatement.items.length)
    throw new Error('Join requires at least one referenced column');
  const joinPlan = planResidentSqlJoin({
    statement: plannerStatement,
    findSource: props.findSource,
    policy: {
      ...props.policy,
      ...(semiLike ? {duplicateRightMatches: 'first' as const} : {})
    },
    resultId: props.resultId
  });
  const nameOf = (side: 'left' | 'right', column: string) =>
    joinPlan.outputs.find(output => output.side === side && output.sourceColumn === column)!.name;

  // Output names for plain `SELECT b.x` items keep the SQL column name when no one else uses it.
  const edits: {start: number; end: number; text: string}[] = [];
  const outputNames = new Set<string>();
  for (let index = 1; index < statement.fromStart; index++)
    if (tokens[index - 1]?.text.toUpperCase() === 'AS' && tokens[index].kind === 'word')
      outputNames.add(tokens[index].text.toLowerCase());
  const itemOf = (entry: Pending) => {
    const tokenIndex = tokens.findIndex(token => token.start === entry.start);
    const nextText = tokens[tokenIndex + 1]?.text.toUpperCase();
    return (
      tokenIndex > 0 &&
      tokenIndex < statement.fromStart &&
      (tokens[tokenIndex - 1].text === ',' ||
        tokens[tokenIndex - 1].text.toUpperCase() === 'SELECT') &&
      (nextText === ',' || tokenIndex + 1 === statement.fromStart)
    );
  };
  const splitKey = (key: string) => {
    const [side, ...columnParts] = key.split(':');
    return {side: side as 'left' | 'right', column: columnParts.join(':')};
  };
  for (const entry of pending)
    if (entry.key && itemOf(entry)) {
      const {side, column} = splitKey(entry.key);
      if (nameOf(side, column) === column) outputNames.add(column.toLowerCase());
    }
  for (const entry of pending) {
    if (entry.star) {
      edits.push({
        start: entry.start,
        end: entry.end,
        text: entry.star
          .map(located => quoteResidentSqlIdentifier(nameOf(located.side, located.column)))
          .join(', ')
      });
      continue;
    }
    const {side, column} = splitKey(entry.key!);
    const name = nameOf(side, column);
    let text = quoteResidentSqlIdentifier(name);
    if (name !== column && itemOf(entry) && !outputNames.has(column.toLowerCase())) {
      outputNames.add(column.toLowerCase());
      text += ` AS ${quoteResidentSqlIdentifier(column)}`;
    }
    edits.push({start: entry.start, end: entry.end, text});
  }

  // Residual predicates joined into the WHERE clause. Numeric multi-column keys of an INNER join
  // may have to run hashed (when exact ranking does not fit); that variant verifies the key parts.
  const makeOuterSql = (verifyHashedKeys: boolean): string => {
    const residual: string[] = [];
    if (type === 'ANTI')
      residual.push(`${quoteResidentSqlIdentifier(nameOf('right', pairs[0].rightColumn))} IS NULL`);
    if (verifyHashedKeys) residual.push(`${RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN} = 1`);
    const finalEdits = [...edits];
    let fromText = `FROM ${tableName}`;
    const whereIndex = findClause(tokens, statement.joinEnd, 'WHERE');
    if (residual.length && whereIndex < 0) fromText += ` WHERE ${residual.join(' AND ')}`;
    finalEdits.push({
      start: tokens[statement.fromStart].start,
      end: tokens[statement.joinEnd - 1].end,
      text: fromText
    });
    if (residual.length && whereIndex >= 0) {
      let whereEnd = whereIndex + 1;
      let depth = 0;
      for (; whereEnd < tokens.length; whereEnd++) {
        const token = tokens[whereEnd];
        if (token.text === '(') depth++;
        else if (token.text === ')') depth--;
        else if (
          depth === 0 &&
          (token.text === ';' ||
            (token.kind === 'word' && CLAUSE_KEYWORDS.has(token.text.toUpperCase())))
        )
          break;
      }
      finalEdits.push({
        start: tokens[whereIndex].end,
        end: tokens[whereIndex].end,
        text: ` ${residual.join(' AND ')} AND (`
      });
      finalEdits.push({
        start: tokens[whereEnd - 1].end,
        end: tokens[whereEnd - 1].end,
        text: ')'
      });
    }
    return applyResidentSqlEdits(sql, finalEdits);
  };
  const hashedVariant = Boolean(compositeKey && !compositeKey.exact && type === 'INNER');
  return {
    joinPlan,
    ...(compositeKey ? {compositeKey} : {}),
    tableName,
    outerSql: makeOuterSql(hashedVariant),
    exactOuterSql: makeOuterSql(false),
    joinType: type
  };
}

function findClause(
  tokens: readonly {kind: string; text: string}[],
  from: number,
  clause: string
): number {
  let depth = 0;
  for (let index = from; index < tokens.length; index++) {
    if (tokens[index].text === '(') depth++;
    else if (tokens[index].text === ')') depth--;
    else if (
      depth === 0 &&
      tokens[index].kind === 'word' &&
      tokens[index].text.toUpperCase() === clause
    )
      return index;
  }
  return -1;
}
