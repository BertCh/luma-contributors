// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SqlExecutionRefusal} from './sql-backend-router';
import {
  executeResidentSqlWideAggregate,
  planResidentSqlWideAggregate,
  type ResidentSqlWideAggregate
} from './resident-sql-wide-aggregate';
import {
  estimateTimeColumnByteLength,
  isResidentSqlTimeColumn,
  readResidentSqlTimeColumn
} from './resident-sql-time-columns';
import {makeArrowTableFromGPUAnalyticsTable} from '@luma.gl/graph/arrow';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  CompiledGPUDataFrameGlobalSort,
  CompiledGPUDataFrameGroupedAggregation,
  CompiledGPUDataFrameJoin,
  GPUDataFrame
} from '@luma.gl/graph/gpu-dataframe';
import {GPUCommandGraph, GPUGather, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUSQLContext, type GPUSQLQuery} from '@luma.gl/graph/gpu-sql';
import {
  estimateKeplerResidentRowCacheByteLength,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetSnapshot,
  type KeplerResidentColumnSchema
} from '@luma-contributors/kepler';
import {
  inlineResidentSqlCtes,
  planResidentSqlBatchGathers,
  planResidentSqlMaterialization,
  RESIDENT_SQL_FORMATS,
  type ResidentSqlBatchGather
} from './resident-sql-plan';
import type {SqlPlanner} from './sql-backend-router';
import {
  findResidentSqlDistinctKeys,
  MAXIMUM_RESIDENT_SQL_DISTINCT_VALUES,
  makeResidentSqlValueDictionary,
  type ResidentSqlValueDictionary
} from './resident-sql-distinct';
import {executeResidentJoin} from './resident-join-runtime';
import {
  assembleSqlFrame,
  buildResidentSqlDerivedFrame,
  estimateResidentSqlDerivedByteLength,
  type ResidentSqlDerivedFrame
} from './resident-sql-derived';
import {
  DEFAULT_RESIDENT_SQL_JOIN_POLICY,
  makeResidentSqlJoinSource,
  parseResidentSqlJoin,
  planResidentSqlJoin,
  readResidentSqlJoinPolicyDirective,
  type ResidentSqlJoinPlan,
  type ResidentSqlJoinPolicy,
  type ResidentSqlJoinSource
} from './resident-sql-join';
import {RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN} from './resident-join-composite-key';
import {rankResidentJoinCompositeKeyPlan} from './resident-join-composite-rank';
import {
  parseResidentSqlCompose,
  planResidentSqlCompose,
  type ResidentSqlComposedPlan,
  type ResidentSqlComposeStatement
} from './resident-sql-compose';
import {
  makeResidentSqlAggregateTailSql,
  splitResidentSqlAggregateTail,
  type ResidentSqlAggregateTail
} from './resident-sql-aggregate-tail';
import {
  decodeResidentSqlIdentifier,
  RESIDENT_SQL_HIDDEN_PREFIX,
  rewriteResidentSql,
  rewriteResidentSqlBetween,
  tokenizeResidentSql,
  type ResidentSqlTableInfo
} from './resident-sql-strings';

/** Join executor errors that depend on the data and are answered correctly by DuckDB. */
const RESIDENT_SQL_JOIN_REFUSAL_PATTERN =
  /^Join (produced no rows|Float32 keys must|output exceeds|key has an invalid)/;

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;

/** Names a resident dataset can be queried by: its id, plus a bare identifier form of the id. */
export function getResidentSqlTableNames(datasetId: string): readonly string[] {
  const bare = datasetId.replace(/[^A-Za-z0-9_]/g, '_');
  return bare === datasetId ? [datasetId] : [datasetId, bare];
}

/** Statement planning over registered frames; the one method the runtime needs from GPUSQLContext. */
export type ResidentSqlContext = Pick<GPUSQLContext, 'query'>;

const GPU_SQL_IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;

/**
 * Gives every dataset id that is not a bare identifier its own unique internal table name, so ids
 * that sanitize alike (for example "a-b" and "a b") never share a table. Keys are lowercased ids.
 */
export function assignResidentSqlTableNames(names: readonly string[]): Map<string, string> {
  const usedNames = new Set(
    names.filter(name => GPU_SQL_IDENTIFIER.test(name)).map(name => name.toLowerCase())
  );
  const quotedToBare = new Map<string, string>();
  for (const name of names) {
    if (GPU_SQL_IDENTIFIER.test(name)) continue;
    const sanitized = name.replace(/[^A-Za-z0-9_]/g, '_');
    const base = GPU_SQL_IDENTIFIER.test(sanitized) ? sanitized : `table_${sanitized}`;
    let bare = base;
    for (let suffix = 2; usedNames.has(bare.toLowerCase()); suffix++) bare = `${base}_${suffix}`;
    usedNames.add(bare.toLowerCase());
    quotedToBare.set(name.toLowerCase(), bare);
  }
  return quotedToBare;
}

/**
 * Published GPUSQLContext accepts only bare identifier table names, while resident datasets are
 * named by their ids ("resident-observations"). Tables are registered under their bare identifier
 * form, and double-quoted references to the original id in the statement text are rewritten to it.
 */
export function createResidentSqlContext(
  tables: Readonly<Record<string, GPUDataFrame>>
): ResidentSqlContext {
  const registered: Record<string, GPUDataFrame> = Object.create(null);
  const quotedToBare = assignResidentSqlTableNames(Object.keys(tables));
  for (const [name, frame] of Object.entries(tables))
    registered[GPU_SQL_IDENTIFIER.test(name) ? name : quotedToBare.get(name.toLowerCase())!] =
      frame;
  const context = new GPUSQLContext(registered);
  if (!quotedToBare.size) return context;
  return {
    query: (sql, options) =>
      context.query(rewriteQuotedResidentTableNames(sql, quotedToBare), options)
  };
}

/**
 * Rewrites double-quoted dataset ids to their internal names where they name a table: after FROM or
 * JOIN, or as the qualifier of a column reference. Column identifiers, output aliases and string
 * literals keep their text.
 */
export function rewriteQuotedResidentTableNames(
  sql: string,
  quotedToBare: ReadonlyMap<string, string>
): string {
  let tokens: ReturnType<typeof tokenizeResidentSql>;
  try {
    tokens = tokenizeResidentSql(sql);
  } catch {
    return sql;
  }
  const edits: {start: number; end: number; text: string}[] = [];
  const rewritePart = (part: string, start: number) => {
    if (!part.startsWith('"')) return;
    const bare = quotedToBare.get(decodeResidentSqlIdentifier(part).toLowerCase());
    if (bare) edits.push({start, end: start + part.length, text: bare});
  };
  tokens.forEach((token, index) => {
    if (token.kind !== 'word') return;
    const parts = token.text.match(/"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_]*/g) ?? [];
    if (!parts.length) return;
    const previous = tokens[index - 1];
    const followsTableKeyword = previous?.kind === 'word' && /^(from|join)$/i.test(previous.text);
    const [first, second] = parts;
    if (second !== undefined) {
      rewritePart(first!, token.start);
      if (followsTableKeyword) rewritePart(second, token.end - second.length);
    } else if (followsTableKeyword) {
      rewritePart(first!, token.start);
    }
  });
  let result = '';
  let cursor = 0;
  for (const edit of edits.sort((a, b) => a.start - b.start)) {
    result += sql.slice(cursor, edit.start) + edit.text;
    cursor = edit.end;
  }
  return result + sql.slice(cursor);
}

export type ResidentSqlRegistration = {
  /** Tables the planner can see right now, with the reason a dataset was left out. */
  readonly tables: readonly {name: string; datasetId: string; columnCount: number}[];
  readonly skipped: readonly {datasetId: string; reason: string}[];
};

export type ResidentSqlExecution = {
  readonly dataset: KeplerDatasetSnapshot<KeplerArrowDataset>;
  readonly rowCount: number;
  /** resident: produced on the GPU and adopted; arrow: bounded readback and upload. */
  readonly materialization: 'resident' | 'arrow';
  /** Why the result was read back instead of staying GPU-resident. */
  readonly fallbackReason?: string;
  /** Join results are registered before publication; commit once the host attached them. */
  readonly commit?: () => void;
  /** Rolls back a join result the host failed to attach. */
  readonly discard?: () => Promise<void>;
};

/** A statement planned for the GPU: a GPUSQL dataframe plan or a resident join plan. */
export type ResidentSqlQuery =
  | {
      readonly kind: 'dataframe';
      readonly sql: string;
      readonly query: GPUSQLQuery;
      /** Derived string/CAST columns the plan reads; freed after execution. */
      readonly derived: ResidentSqlDerivedFrame | null;
      readonly release: () => void;
      /** Grouped queries: COUNT(*) column used to drop empty dictionary slots. */
      readonly groupPresenceColumn?: string;
      /** Field metadata of visible derived columns (epoch-second time columns). */
      readonly columnMetadata?: Readonly<Record<string, Readonly<Record<string, string>>>>;
      /** ORDER BY / LIMIT of an aggregate: a second stage over the aggregate's resident result. */
      readonly tail?: ResidentSqlAggregateTail;
      /** Group keys given a dense stand-in dictionary; their codes are the values, so no labels publish. */
      readonly syntheticDictionaryColumns?: ReadonlySet<string>;
      /**
       * CAST(numeric AS VARCHAR) needs a GPU distinct pass before its labels exist. The plan above
       * validates with a placeholder; execution awaits this to get the resolved plan.
       */
      readonly refine?: () => Promise<Extract<ResidentSqlQuery, {kind: 'dataframe'}>>;
    }
  | {
      readonly kind: 'join';
      readonly sql: string;
      readonly joinPlan: ResidentSqlJoinPlan;
    }
  | {
      /** A join plus WHERE/GROUP BY/ORDER BY/LIMIT/expressions: join stage, then the outer query. */
      readonly kind: 'composed';
      readonly sql: string;
      readonly composed: ResidentSqlComposedPlan;
      /** Intermediate columns that can hold NULL; the rest drop their validity vectors. */
      readonly nullableColumns: ReadonlySet<string>;
    }
  | {
      /** Exact Int64/Uint64 COUNT/SUM/MIN/MAX on the wide-word GPU kernels. */
      readonly kind: 'wide-aggregate';
      readonly sql: string;
      readonly wide: ResidentSqlWideAggregate;
    };

/** Tables one statement is planned against: the registry, or a composed join's intermediate. */
export type ResidentSqlScope = {
  readonly context: ResidentSqlContext;
  readonly tables: Readonly<Record<string, GPUDataFrame>>;
  /** Lower-case table name to its frame, row count and column info. */
  findEntry(
    name: string
  ):
    | {readonly frame: GPUDataFrame; readonly rowCount: number; readonly info: ResidentSqlTableInfo}
    | undefined;
};

type Registered = {
  readonly frame: GPUDataFrame;
  readonly datasetId: string;
  readonly revision: number;
  readonly rowCount: number;
  readonly info: ResidentSqlTableInfo;
  readonly joinSource: ResidentSqlJoinSource;
};

/**
 * Registers resident Kepler datasets with the GPU SQL planner and executes planned statements
 * on the GPU. Filters, joins (per source batch, chunk boundaries preserved), global sorts and
 * dense aggregates stay GPU-resident (`adoptResidentDataset`, only tiny row counts are read);
 * results with non-resident column formats fall back to a bounded Arrow readback with a reason.
 */
export class ResidentSqlGpuRuntime {
  readonly session: KeplerComputeSession;
  private context: ResidentSqlContext | null = null;
  private registered: Registered[] = [];
  private signature = '';
  private registration: ResidentSqlRegistration = {tables: [], skipped: []};
  private resultSequence = 0;
  private tables: Record<string, GPUDataFrame> = Object.create(null);
  private registeredByName = new Map<string, Registered>();
  private pendingRelease: (() => void) | null = null;
  /** Distinct-value dictionaries of numeric columns for CAST to VARCHAR, per resident frame. */
  private valueDictionaries = new WeakMap<GPUDataFrame, Map<string, ResidentSqlValueDictionary>>();
  /** Dense group counts (largest value + 1) of uint32 GROUP BY keys without a dictionary. */
  private denseGroupCounts = new WeakMap<GPUDataFrame, Map<string, number>>();
  private joinPolicy: ResidentSqlJoinPolicy = DEFAULT_RESIDENT_SQL_JOIN_POLICY;
  /** Stable planner whose context is refreshed against the registry on every statement. */
  readonly planner: SqlPlanner<ResidentSqlQuery> = {
    query: sql => this.planStatement(sql)
  };

  constructor(session: KeplerComputeSession) {
    this.session = session;
  }

  /** Default duplicate-key and naming policy for SQL joins; `-- luma:` directives override it. */
  setJoinPolicy(policy: Partial<ResidentSqlJoinPolicy>): void {
    this.joinPolicy = {...DEFAULT_RESIDENT_SQL_JOIN_POLICY, ...policy};
  }

  /** Plans one statement: resident join, string/CAST rewrite plus GPUSQL, or plain GPUSQL. */
  private planStatement(sql: string): ResidentSqlQuery {
    this.refresh();
    if (!this.context) throw new Error('No resident dataset has columns GPU SQL can read');
    // A planned but never executed statement must not hold derived scratch.
    this.pendingRelease?.();
    this.pendingRelease = null;
    const text = rewriteResidentSqlBetween(inlineResidentSqlCtes(sql));
    const wide = planResidentSqlWideAggregate(
      this.session,
      text,
      name => this.registeredByName.get(name)?.datasetId
    );
    if (wide) return {kind: 'wide-aggregate', sql, wide};
    const join = parseResidentSqlJoin(text);
    if (join && !('reason' in join)) {
      const joinPlan = planResidentSqlJoin({
        statement: join,
        findSource: name => this.registeredByName.get(name)?.joinSource,
        policy: {...this.joinPolicy, ...readResidentSqlJoinPolicyDirective(sql)},
        resultId: `sql-gpu-result-${++this.resultSequence}`
      });
      return {kind: 'join', sql, joinPlan};
    }
    const composeStatement = parseResidentSqlCompose(text);
    if (composeStatement && !('reason' in composeStatement))
      return this.planComposed(sql, composeStatement);
    const reason = composeStatement
      ? composeStatement.reason
      : join && 'reason' in join
        ? join.reason
        : undefined;
    try {
      return this.planDataframe(sql, text);
    } catch (error) {
      if (reason && error instanceof Error) throw new Error(`${error.message} (${reason})`);
      throw error;
    }
  }

  /** Plans the join stage and checks the outer statement's structure without touching the GPU. */
  private planComposed(sql: string, statement: ResidentSqlComposeStatement): ResidentSqlQuery {
    const composed = planResidentSqlCompose({
      statement,
      findSource: name => this.registeredByName.get(name)?.joinSource,
      policy: {...this.joinPolicy, ...readResidentSqlJoinPolicyDirective(sql)},
      resultId: `sql-gpu-join-${++this.resultSequence}`
    });
    // Column formats and dictionaries of the intermediate follow its source columns.
    const {plan} = composed.joinPlan;
    const sources = {
      left: this.registered.find(entry => entry.datasetId === plan.left.datasetId),
      right: this.registered.find(entry => entry.datasetId === plan.right.datasetId)
    };
    const columns: Record<string, ResidentSqlTableInfo['columns'][string]> = Object.create(null);
    const vectors: Record<string, GPUVector> = Object.create(null);
    const validity: Record<string, GPUVector<'uint32'>> = Object.create(null);
    const dictionaries: Record<string, readonly unknown[]> = Object.create(null);
    // Planning needs only names, formats and nullability; one-row views keep row counts equal.
    const nullableColumns = new Set<string>();
    const sideMayBeNull = (side: 'left' | 'right') =>
      side === 'left'
        ? composed.joinType === 'RIGHT' || composed.joinType === 'FULL'
        : composed.joinType === 'LEFT' ||
          composed.joinType === 'FULL' ||
          composed.joinType === 'ANTI';
    const asPlanningVector = (vector: GPUVector, name: string, format = vector.format) =>
      new GPUVector({
        name,
        type: 'data',
        format,
        ownsData: false,
        data: [
          new GPUData({
            format,
            buffer: vector.data[0].buffer,
            byteOffset: vector.data[0].byteOffset,
            length: 1,
            ownsBuffer: false
          })
        ]
      } as never) as GPUVector;
    for (const output of composed.joinPlan.outputs) {
      const source =
        sources[(output.side === 'left') !== composed.joinPlan.swapped ? 'left' : 'right']!;
      columns[output.name] = source.info.columns[output.sourceColumn];
      const sourceVector = source.frame.table.gpuVectors[output.sourceColumn] as GPUVector;
      vectors[output.name] = asPlanningVector(sourceVector, output.name);
      const sourceValidity = source.frame.validity[output.sourceColumn] as GPUVector<'uint32'>;
      if (sourceValidity || sideMayBeNull(output.side)) {
        nullableColumns.add(output.name);
        validity[output.name] = asPlanningVector(
          sourceValidity ?? sourceVector,
          `${output.name}-validity`,
          'uint32'
        ) as GPUVector<'uint32'>;
      }
      const dictionary = source.frame.dictionaries[output.sourceColumn];
      if (dictionary) dictionaries[output.name] = dictionary as readonly unknown[];
    }
    if (composed.compositeKey && composed.outerSql !== composed.exactOuterSql) {
      const [firstName] = Object.keys(vectors);
      columns[RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN] = {format: 'uint32'};
      vectors[RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN] = asPlanningVector(
        vectors[firstName],
        RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN,
        'uint32'
      );
    }
    const info: ResidentSqlTableInfo = {name: composed.tableName, columns};
    // An aggregate's ORDER BY / LIMIT run as a second stage; validate the aggregate itself here.
    const aggregateText =
      splitResidentSqlAggregateTail(composed.outerSql)?.aggregateSql ?? composed.outerSql;
    const rewrite = rewriteResidentSql(aggregateText, name =>
      name.toLowerCase() === composed.tableName ? info : undefined
    );
    // Derived columns are built at execution; their statements are parsed then.
    if (!rewrite || !rewrite.derived.length) {
      const frame = assembleSqlFrame(vectors, validity, dictionaries);
      new GPUSQLContext({[composed.tableName]: frame}).query(rewrite ? rewrite.sql : aggregateText);
    }
    return {kind: 'composed', sql, composed, nullableColumns};
  }

  /** Runs the join into an intermediate dataset, then the outer query over it; frees both. */
  private async executeComposed(
    planned: Extract<ResidentSqlQuery, {kind: 'composed'}>
  ): Promise<ResidentSqlExecution> {
    const {session} = this;
    const {composed} = planned;
    const revisions = this.registered
      .map(entry => `${entry.datasetId}@${entry.revision}`)
      .join('|');
    const registered = this.registered.slice();
    // Numeric multi-column keys become exact ranks when they fit; otherwise an INNER join runs
    // hashed and verified, and every other join type is answered by DuckDB.
    let compositeKey = composed.compositeKey;
    let outerSql = composed.exactOuterSql;
    if (compositeKey && !compositeKey.exact) {
      const ranked = await rankResidentJoinCompositeKeyPlan({
        device: session.device,
        plan: compositeKey,
        getColumn: (side, column) => {
          const source = session.registry.getDataset<KeplerArrowDataset>(
            side === 'left'
              ? composed.joinPlan.plan.left.datasetId
              : composed.joinPlan.plan.right.datasetId
          )!;
          return {
            vector: source.value.columns[column] as GPUVector,
            validity: source.value.analytics.validity[column] as GPUVector<'uint32'> | undefined
          };
        }
      });
      if (ranked) compositeKey = ranked;
      else {
        const sameFormats = compositeKey.components.every(
          component => component.leftFormat === component.rightFormat
        );
        if (
          composed.joinType !== 'INNER' ||
          composed.joinPlan.plan.duplicateRightMatches !== 'all' ||
          !sameFormats
        )
          throw new SqlExecutionRefusal(
            'Numeric multi-column join keys have too many distinct values to rank exactly on the GPU'
          );
        outerSql = composed.outerSql;
      }
    }
    const joined = await executeResidentJoin({
      session,
      plan: composed.joinPlan.plan,
      ...(compositeKey ? {compositeKey} : {}),
      isCurrent: () =>
        registered.every(
          entry => session.registry.getDataset(entry.datasetId)?.revision === entry.revision
        ) && revisions === this.signature
    });
    try {
      const joinedValue = joined.dataset.value;
      // Columns that cannot be NULL skip their validity vector, which saves shader bindings.
      const frame = makeSqlFrame({
        columns: joinedValue.columns,
        prepared: joinedValue.prepared,
        analytics: {
          validity: Object.fromEntries(
            Object.entries(joinedValue.analytics.validity).filter(([name]) =>
              planned.nullableColumns.has(name)
            )
          )
        }
      } as unknown as KeplerArrowDataset);
      if (typeof frame === 'string') throw new Error(frame);
      const entry = {
        frame,
        rowCount: joined.rowCount,
        info: makeSqlTableInfo(composed.tableName, frame)
      };
      const tables = {[composed.tableName]: frame};
      const outer = this.planDataframe(outerSql, outerSql, {
        context: createResidentSqlContext(tables),
        tables,
        findEntry: name => (name === composed.tableName ? entry : undefined)
      });
      if (outer.kind !== 'dataframe')
        throw new Error('Composed join outer query is not a dataframe');
      let outerCurrent = outer;
      try {
        while (outerCurrent.refine) outerCurrent = await outerCurrent.refine();
        return await this.executeStages(outerCurrent, [
          {datasetId: joined.dataset.id, revision: joined.dataset.revision}
        ]);
      } finally {
        outerCurrent.release();
        if (this.pendingRelease === outerCurrent.release) this.pendingRelease = null;
      }
    } finally {
      await joined.destroy();
    }
  }

  /** The registered datasets as one planning scope (composed joins plan over a temporary scope). */
  private getSqlScope(): ResidentSqlScope {
    return {
      context: this.context!,
      tables: this.tables,
      findEntry: name => this.registeredByName.get(name)
    };
  }

  private planDataframe(
    sql: string,
    text: string,
    scope: ResidentSqlScope = this.getSqlScope()
  ): ResidentSqlQuery {
    const {context} = scope;
    const tail = splitResidentSqlAggregateTail(text);
    if (tail && (tail.orderBy || tail.limit !== undefined)) {
      const withTail = (
        planned: Extract<ResidentSqlQuery, {kind: 'dataframe'}>
      ): Extract<ResidentSqlQuery, {kind: 'dataframe'}> => {
        const {refine} = planned;
        return {
          ...planned,
          tail,
          ...(refine
            ? {
                refine: async () => withTail(await refine())
              }
            : {})
        };
      };
      const aggregate = this.planDataframe(sql, tail.aggregateSql, scope);
      return aggregate.kind === 'dataframe' ? withTail(aggregate) : aggregate;
    }
    const denseKey = this.findDenseGroupKey(text, scope);
    if (denseKey) return this.planDenseGroupKey(sql, text, scope, denseKey);
    const rewrite = rewriteResidentSql(text, name => scope.findEntry(name)?.info, {
      resolveValueDictionary: (columnName, tableName) => {
        const frame = scope.findEntry(tableName.toLowerCase())?.frame;
        return frame ? this.valueDictionaries.get(frame)?.get(columnName) : undefined;
      }
    });
    if (!rewrite)
      return {kind: 'dataframe', sql, query: context.query(text), derived: null, release: () => {}};
    const presence = {
      ...(rewrite.groupPresenceColumn ? {groupPresenceColumn: rewrite.groupPresenceColumn} : {}),
      ...(Object.keys(rewrite.columnMetadata).length
        ? {columnMetadata: rewrite.columnMetadata}
        : {})
    };
    const pendingColumns = rewrite.pendingValueColumns;
    const refine = (planned: () => void) =>
      pendingColumns.length
        ? {
            refine: async () => {
              planned();
              const entryToResolve = scope.findEntry(rewrite.tableName.toLowerCase())!;
              let cache = this.valueDictionaries.get(entryToResolve.frame);
              if (!cache) {
                cache = new Map();
                this.valueDictionaries.set(entryToResolve.frame, cache);
              }
              for (const columnName of pendingColumns) {
                const vector = entryToResolve.frame.table.gpuVectors[columnName] as GPUVector;
                const format = vector.format as 'float32' | 'sint32' | 'uint32';
                const keys = await findResidentSqlDistinctKeys({
                  device: this.session.device,
                  vector,
                  validity: entryToResolve.frame.validity[columnName] as
                    | GPUVector<'uint32'>
                    | undefined,
                  format
                });
                cache.set(columnName, makeResidentSqlValueDictionary(format, keys));
              }
              const resolved = this.planDataframe(sql, text, scope);
              if (resolved.kind !== 'dataframe') throw new Error('Refined SQL is not a dataframe');
              return resolved;
            }
          }
        : {};
    if (!rewrite.derived.length)
      return {
        kind: 'dataframe',
        sql,
        query: context.query(rewrite.sql),
        derived: null,
        release: () => {},
        ...presence,
        ...refine(() => {})
      };
    const entry = scope.findEntry(rewrite.tableName.toLowerCase())!;
    const reservation = this.session.registry.reserveMemory(
      estimateResidentSqlDerivedByteLength(rewrite.derived, entry.rowCount),
      'gpu',
      'scratch'
    );
    let derived: ResidentSqlDerivedFrame | null = null;
    try {
      derived = buildResidentSqlDerivedFrame({
        device: this.session.device,
        base: entry.frame,
        specs: rewrite.derived
      });
      const frames = {...scope.tables};
      for (const [name, frame] of Object.entries(scope.tables))
        if (frame === entry.frame) frames[name] = derived.frame;
      const query = createResidentSqlContext(frames).query(rewrite.sql);
      const built = derived;
      let released = false;
      const release = () => {
        if (released) return;
        released = true;
        built.destroy();
        reservation.release();
      };
      this.pendingRelease = release;
      return {
        kind: 'dataframe',
        sql,
        query,
        derived: built,
        release,
        ...presence,
        ...refine(release)
      };
    } catch (error) {
      derived?.destroy();
      reservation.release();
      throw error;
    }
  }

  getRegistration(): ResidentSqlRegistration {
    this.refresh();
    return this.registration;
  }

  /** Rebuilds frames only when the set of dataset revisions changed. */
  private refresh(): void {
    const footprints = this.session.registry.getDatasetFootprints();
    const snapshots = footprints
      .map(({id}) => this.session.registry.getDataset<KeplerArrowDataset>(id))
      .filter((snapshot): snapshot is KeplerDatasetSnapshot<KeplerArrowDataset> =>
        Boolean(snapshot)
      );
    const signature = snapshots.map(snapshot => `${snapshot.id}@${snapshot.revision}`).join('|');
    if (signature === this.signature) return;
    this.signature = signature;
    this.registered = [];
    this.registeredByName = new Map();
    const tables: Record<string, GPUDataFrame> = Object.create(null);
    const registeredTables: {name: string; datasetId: string; columnCount: number}[] = [];
    const skipped: {datasetId: string; reason: string}[] = [];
    for (const snapshot of snapshots) {
      const built = makeSqlFrame(snapshot.value);
      if (typeof built === 'string') {
        skipped.push({datasetId: snapshot.id, reason: built});
        continue;
      }
      const registeredEntry: Registered = {
        frame: built,
        datasetId: snapshot.id,
        revision: snapshot.revision,
        rowCount: snapshot.rowCount,
        info: makeSqlTableInfo(snapshot.id, built),
        joinSource: makeResidentSqlJoinSource(snapshot)
      };
      this.registered.push(registeredEntry);
      for (const name of getResidentSqlTableNames(snapshot.id)) {
        if (tables[name.toLowerCase()]) continue;
        tables[name.toLowerCase()] = built;
        this.registeredByName.set(name.toLowerCase(), registeredEntry);
        registeredTables.push({name, datasetId: snapshot.id, columnCount: built.numCols});
      }
    }
    this.registration = {tables: registeredTables, skipped};
    this.tables = tables;
    this.context = registeredTables.length ? createResidentSqlContext(tables) : null;
  }

  /** Runs a planned statement and publishes its result as a resident dataset. */
  async execute(planned: ResidentSqlQuery): Promise<ResidentSqlExecution> {
    try {
      return await this.executePlanned(planned);
    } catch (error) {
      // Data-dependent join limits mean the GPU cannot give the exact SQL result; DuckDB can.
      if (error instanceof Error && RESIDENT_SQL_JOIN_REFUSAL_PATTERN.test(error.message))
        throw new SqlExecutionRefusal(error.message.split('.')[0]);
      throw error;
    }
  }

  private async executePlanned(planned: ResidentSqlQuery): Promise<ResidentSqlExecution> {
    if (planned.kind === 'join') return this.executeJoin(planned);
    if (planned.kind === 'composed') return this.executeComposed(planned);
    if (planned.kind === 'wide-aggregate')
      return executeResidentSqlWideAggregate(
        this.session,
        planned.wide,
        `sql-gpu-result-${++this.resultSequence}`
      );
    let current = planned;
    try {
      while (current.refine) current = await current.refine();
      return await this.executeStages(current);
    } finally {
      current.release();
      if (this.pendingRelease === current.release) this.pendingRelease = null;
    }
  }

  private async executeJoin(
    planned: Extract<ResidentSqlQuery, {kind: 'join'}>
  ): Promise<ResidentSqlExecution> {
    const {session} = this;
    const {plan} = planned.joinPlan;
    const revisions = this.registered
      .map(entry => `${entry.datasetId}@${entry.revision}`)
      .join('|');
    const result = await executeResidentJoin({
      session,
      plan,
      isCurrent: () =>
        this.registered.every(
          entry => session.registry.getDataset(entry.datasetId)?.revision === entry.revision
        ) && revisions === this.signature
    });
    return {
      dataset: result.dataset,
      rowCount: result.rowCount,
      materialization: result.dataset.value.resident ? 'resident' : 'arrow',
      ...(result.dataset.value.resident
        ? {}
        : {fallbackReason: 'join result is small enough to be validated on the CPU and adopted'}),
      commit: () => result.transfer(),
      discard: () => result.destroy()
    };
  }

  /** A single GROUP BY key that is a uint32 column without a dictionary (dense grouping needs a count). */
  private findDenseGroupKey(
    text: string,
    scope: ResidentSqlScope
  ): {tableName: string; column: string} | null {
    const tokens = tokenizeResidentSql(text);
    const wordAt = (index: number) =>
      tokens[index]?.kind === 'word' ? tokens[index].text.toUpperCase() : '';
    const groupIndex = tokens.findIndex(
      (token, index) =>
        token.kind === 'word' && wordAt(index) === 'GROUP' && wordAt(index + 1) === 'BY'
    );
    const fromIndex = tokens.findIndex((_, index) => wordAt(index) === 'FROM');
    if (groupIndex < 0 || fromIndex < 0 || tokens[groupIndex + 2]?.kind !== 'word') return null;
    if (tokens[groupIndex + 3]?.text === ',') return null;
    const entry = scope.findEntry(
      decodeResidentSqlIdentifier(tokens[fromIndex + 1].text).toLowerCase()
    );
    if (!entry) return null;
    const requested = decodeResidentSqlIdentifier(
      tokens[groupIndex + 2].text.split('.').pop() ?? tokens[groupIndex + 2].text
    ).toLowerCase();
    const column = Object.keys(entry.info.columns).find(name => name.toLowerCase() === requested);
    const info = column ? entry.info.columns[column] : undefined;
    return column && info?.format === 'uint32' && !info.dictionary
      ? {tableName: decodeResidentSqlIdentifier(tokens[fromIndex + 1].text).toLowerCase(), column}
      : null;
  }

  /**
   * Dense grouping needs the number of group slots. For a uint32 key without a dictionary that is
   * the largest value plus one, found with the GPU distinct pass at execution (`refine`); planning
   * validates with a one-slot stand-in. Values above the distinct-value bound go to DuckDB.
   */
  private planDenseGroupKey(
    sql: string,
    text: string,
    scope: ResidentSqlScope,
    key: {tableName: string; column: string}
  ): ResidentSqlQuery {
    const entry = scope.findEntry(key.tableName)!;
    const known = this.denseGroupCounts.get(entry.frame)?.get(key.column);
    const frame = entry.frame;
    const vectors: Record<string, GPUVector> = Object.create(null);
    for (const [name, vector] of Object.entries(frame.table.gpuVectors))
      if (vector) vectors[name] = vector as GPUVector;
    const validity = frame.validity as Record<string, GPUVector<'uint32'>>;
    const dictionaries = {
      ...(frame.dictionaries as Record<string, readonly unknown[]>),
      [key.column]: new Array<string>(known ?? 1).fill('')
    };
    const swapped = assembleSqlFrame(vectors, validity, dictionaries);
    const tables = {...scope.tables};
    for (const [name, candidate] of Object.entries(scope.tables))
      if (candidate === frame) tables[name] = swapped;
    const swappedInfo = {
      ...entry.info,
      columns: {
        ...entry.info.columns,
        [key.column]: {
          ...entry.info.columns[key.column],
          dictionary: dictionaries[key.column] as string[]
        }
      }
    };
    const planned = this.planDataframe(sql, text, {
      context: createResidentSqlContext(tables),
      tables,
      findEntry: name =>
        name === key.tableName
          ? {...entry, frame: swapped, info: swappedInfo}
          : scope.findEntry(name)
    });
    if (planned.kind !== 'dataframe') return planned;
    const synthetic = new Set([key.column]);
    if (known !== undefined) return {...planned, syntheticDictionaryColumns: synthetic};
    return {
      ...planned,
      syntheticDictionaryColumns: synthetic,
      refine: async () => {
        planned.release();
        if (this.pendingRelease === planned.release) this.pendingRelease = null;
        const keys = await findResidentSqlDistinctKeys({
          device: this.session.device,
          vector: frame.table.gpuVectors[key.column] as GPUVector,
          validity: frame.validity[key.column] as GPUVector<'uint32'> | undefined,
          format: 'uint32'
        });
        const groupCount = keys.reduce((largest, value) => Math.max(largest, value), 0) + 1;
        // Beyond the distinct bound a dense slot per value is not worth it; DuckDB answers.
        if (groupCount > MAXIMUM_RESIDENT_SQL_DISTINCT_VALUES)
          throw new SqlExecutionRefusal('GROUP BY key values are too large for dense GPU grouping');
        const counts = this.denseGroupCounts.get(frame) ?? new Map<string, number>();
        this.denseGroupCounts.set(frame, counts);
        counts.set(key.column, groupCount);
        const resolved = this.planDataframe(sql, text, scope);
        if (resolved.kind !== 'dataframe') throw new Error('Refined SQL is not a dataframe');
        return resolved;
      }
    };
  }

  /** Runs a planned dataframe, then its ORDER BY / LIMIT second stage when it has one. */
  private async executeStages(
    planned: Extract<ResidentSqlQuery, {kind: 'dataframe'}>,
    sources?: readonly {datasetId: string; revision: number}[]
  ): Promise<ResidentSqlExecution> {
    const first = await this.executeDataframe(
      planned.query,
      planned.groupPresenceColumn,
      sources,
      planned.columnMetadata,
      planned.syntheticDictionaryColumns
    );
    if (!planned.tail) return first;
    const {session} = this;
    const tableName = 'luma_sql_aggregated';
    try {
      const frame = makeSqlFrame(first.dataset.value);
      if (typeof frame === 'string') throw new SqlExecutionRefusal(frame);
      const tables = {[tableName]: frame};
      const entry = {
        frame,
        rowCount: first.rowCount,
        info: makeSqlTableInfo(tableName, frame)
      };
      const tailSql = makeResidentSqlAggregateTailSql(planned.tail, tableName);
      let outer: ResidentSqlQuery;
      try {
        outer = this.planDataframe(tailSql, tailSql, {
          context: createResidentSqlContext(tables),
          tables,
          findEntry: name => (name === tableName ? entry : undefined)
        });
      } catch (error) {
        throw new SqlExecutionRefusal(error instanceof Error ? error.message : String(error));
      }
      if (outer.kind !== 'dataframe') throw new Error('Aggregate ordering is not a dataframe');
      try {
        return await this.executeDataframe(
          outer.query,
          outer.groupPresenceColumn,
          [{datasetId: first.dataset.id, revision: first.dataset.revision}],
          planned.columnMetadata
        );
      } finally {
        outer.release();
        if (this.pendingRelease === outer.release) this.pendingRelease = null;
      }
    } finally {
      // The unordered aggregate is scratch: the ordered result was gathered into new buffers.
      if (session.registry.getDataset(first.dataset.id)?.revision === first.dataset.revision) {
        const fence = session.device.createFence();
        try {
          await session.removeDataset(
            first.dataset.id,
            Promise.race([fence.signaled, session.device.lost])
          );
        } finally {
          fence.destroy();
        }
      }
    }
  }

  private async executeDataframe(
    query: GPUSQLQuery,
    groupPresenceColumn?: string,
    sources?: readonly {datasetId: string; revision: number}[],
    columnMetadata?: Readonly<Record<string, Readonly<Record<string, string>>>>,
    syntheticDictionaryColumns?: ReadonlySet<string>
  ): Promise<ResidentSqlExecution> {
    const {session} = this;
    const device = session.device;
    const registered = sources ?? this.registered.slice();
    const request = session.registry.beginRequest(
      `resident-sql:${++this.resultSequence}`,
      registered.map(entry => entry.datasetId)
    );
    const isCurrent = () =>
      !request.signal.aborted &&
      !session.deviceLost &&
      registered.every(
        entry => session.registry.getDataset(entry.datasetId)?.revision === entry.revision
      );
    const graph = new GPUCommandGraph<Record<string, number | boolean | null>>(device, {
      id: `resident-sql-${this.resultSequence}`
    });
    const compiled = query.compile(graph);
    const retained = new Set<Buffer>();
    const scratch: Buffer[] = [];
    let gpuReservation: {release(): void; isActive: boolean} | undefined;
    let cpuReservation: {release(): void; isActive: boolean} | undefined;
    let adopted = false;
    try {
      const encoder = device.createCommandEncoder({id: 'resident-sql-encode'});
      compiled.encode(encoder);
      device.submit(encoder.finish());
      const resultId = `sql-gpu-result-${this.resultSequence}`;
      const readCount = async (data: GPUData): Promise<number> => {
        const bytes = await getBuffer(data).readAsync(data.byteOffset, 4);
        return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
      };
      const readbackArrow = async (reason: string): Promise<ResidentSqlExecution> => {
        // Explicit, bounded Arrow readback; the reason is surfaced to the host.
        const table = await makeArrowTableFromGPUAnalyticsTable({
          table: compiled.table,
          validity: compiled.validity,
          dictionaries: compiled.dictionaries as Record<string, readonly string[]>
        });
        if (!isCurrent()) throw new Error('SQL sources changed during execution');
        if (!table.numRows)
          throw new SqlExecutionRefusal('SQL produced no rows; resident results need one row');
        const dataset = session.importArrowDataset({id: resultId, table});
        adopted = true;
        return {dataset, rowCount: table.numRows, materialization: 'arrow', fallbackReason: reason};
      };
      const plan = planResidentSqlMaterialization(compiled);
      if (plan.kind === 'readback') return await readbackArrow(plan.reason);
      if (compiled instanceof CompiledGPUDataFrameJoin) {
        // Join output capacity exceeded or right keys not unique: results would be incomplete.
        const flags = await Promise.all([
          ...compiled.overflows.data.map(readCount),
          readCount(compiled.contractViolation.data[0])
        ]);
        if (flags.some(Boolean))
          throw new SqlExecutionRefusal('GPU join exceeded its bounds or right keys repeat');
      }
      let gathers: readonly ResidentSqlBatchGather[];
      let rowCount: number;
      // GROUP BY returns every dictionary slot; keep only slots that have rows (SQL semantics).
      let denseCompaction: {buffer: Buffer; liveCount: number} | null = null;
      if (
        plan.kind === 'dense' &&
        groupPresenceColumn &&
        compiled instanceof CompiledGPUDataFrameGroupedAggregation
      ) {
        const presence = compiled.table.gpuVectors[groupPresenceColumn];
        if (presence) {
          const data = presence.data[0];
          const bytes = await getBuffer(data).readAsync(data.byteOffset, plan.rowCount * 4);
          const counts = new Uint32Array(bytes.buffer, bytes.byteOffset, plan.rowCount);
          const live = Array.from(counts.keys()).filter(slot => counts[slot] !== 0);
          if (live.length !== plan.rowCount) {
            const buffer = device.createBuffer({
              id: 'sql-live-groups',
              data: live.length ? Uint32Array.from(live) : new Uint32Array(1),
              usage: STORAGE_USAGE
            });
            scratch.push(buffer);
            retained.add(buffer);
            denseCompaction = {buffer, liveCount: live.length};
          }
        }
      }
      if (denseCompaction) {
        gathers = denseCompaction.liveCount
          ? [{sourceChunkIndex: null, rowCount: denseCompaction.liveCount}]
          : [];
        rowCount = denseCompaction.liveCount;
      } else if (plan.kind === 'dense') {
        gathers = [{sourceChunkIndex: 0, rowCount: plan.rowCount}];
        rowCount = plan.rowCount;
      } else if (plan.kind === 'global-sort') {
        // Only for CompiledGPUDataFrameGlobalSort; indices address the whole source.
        const count = await readCount(
          (compiled as CompiledGPUDataFrameGlobalSort).globalSelectedCount.data[0]
        );
        gathers = count ? [{sourceChunkIndex: null, rowCount: count}] : [];
        rowCount = count;
      } else {
        const counts = await Promise.all(compiled.selectedCounts.data.map(readCount));
        ({gathers, rowCount} = planResidentSqlBatchGathers(counts));
      }
      if (!isCurrent()) throw new Error('SQL sources changed during execution');
      if (!rowCount)
        throw new SqlExecutionRefusal('SQL produced no rows; resident results need one row');
      // Derived helper columns (masks, ranks, hidden casts) never reach the published result.
      const names = compiled.table.schema.fields
        .map(field => field.name)
        .filter(name => !name.startsWith(RESIDENT_SQL_HIDDEN_PREFIX));
      if (!names.length) throw new Error('SQL selected no output columns');
      const schema: KeplerResidentColumnSchema[] = [];
      for (const name of names) {
        const vector = compiled.table.gpuVectors[name];
        const dictionary = syntheticDictionaryColumns?.has(name)
          ? undefined
          : compiled.dictionaries[name];
        const labels = Array.isArray(dictionary)
          ? dictionary
          : (dictionary as {values?: readonly unknown[]} | undefined)?.values;
        schema.push({
          name,
          format: vector!.format as KeplerResidentColumnSchema['format'],
          ...(labels ? {dictionary: labels.map(String)} : {}),
          ...(columnMetadata?.[name] ? {metadata: columnMetadata[name]} : {})
        });
      }
      const timeNames = schema
        .filter(entry => entry.format === 'sint32' && isResidentSqlTimeColumn(entry.metadata))
        .map(entry => entry.name);
      const timeBytes = timeNames.length * estimateTimeColumnByteLength(rowCount);
      cpuReservation = session.registry.reserveMemory(
        estimateKeplerResidentRowCacheByteLength(
          schema,
          gathers.map(gather => gather.rowCount)
        ) + timeBytes,
        'cpu',
        'readback'
      );
      const outputBytes = rowCount * 4 * names.length * 2 + 65536 * gathers.length + timeBytes;
      gpuReservation = session.registry.reserveMemory(outputBytes, 'gpu', 'scratch');
      const gatherGraph = new GPUCommandGraph<undefined>(device, {
        id: `resident-sql-gather-${this.resultSequence}`
      });
      const bufferHandles = new Map<Buffer, ReturnType<typeof gatherGraph.importBuffer>>();
      const importShared = (buffer: Buffer, label: string) => {
        let handle = bufferHandles.get(buffer);
        if (!handle) {
          handle = gatherGraph.importBuffer(
            {id: label, usage: buffer.usage, byteLength: buffer.byteLength},
            buffer
          );
          bufferHandles.set(buffer, handle);
        }
        return handle;
      };
      const rowIdentity = (index: number): GPUData => {
        const gather = gathers[index];
        if (denseCompaction)
          return new GPUData({
            format: 'uint32',
            buffer: denseCompaction.buffer,
            length: denseCompaction.liveCount,
            ownsBuffer: false
          });
        if (plan.kind === 'global-sort')
          return (compiled as CompiledGPUDataFrameGlobalSort).globalRowIndices.data[0];
        return compiled.rowIndices.data[gather.sourceChunkIndex!];
      };
      const useCopy = plan.kind === 'dense' && !denseCompaction;
      const copyEncoder = device.createCommandEncoder({id: 'resident-sql-copy'});
      let nodeIndex = 0;
      const gatherInto = <Format extends 'float32' | 'sint32' | 'uint32'>(
        label: string,
        source: GPUVector<Format>,
        format: Format
      ): GPUVector<Format> => {
        const importedSource = useCopy
          ? null
          : gatherGraph.importGPUVector(`sql-source-${nodeIndex}`, source);
        const chunks = gathers.map((gather, gatherIndex) => {
          const output = device.createBuffer({
            id: `sql-output-${label}-${gatherIndex}`,
            byteLength: Math.max(4, gather.rowCount * 4),
            usage: STORAGE_USAGE
          });
          scratch.push(output);
          retained.add(output);
          if (useCopy) {
            const sourceData = source.data[0];
            copyEncoder.copyBufferToBuffer({
              sourceBuffer: getBuffer(sourceData),
              sourceOffset: sourceData.byteOffset,
              destinationBuffer: output,
              size: gather.rowCount * 4
            });
          } else {
            const indexData = rowIdentity(gatherIndex);
            const indexView: GraphDataView<'uint32'> = gatherGraph.createDataView(
              importShared(getBuffer(indexData), `sql-row-indices-${gatherIndex}`),
              {format: 'uint32', length: gather.rowCount, byteOffset: indexData.byteOffset}
            );
            gatherGraph.add(
              new GPUGather({
                id: `sql-gather-${nodeIndex++}`,
                // Row indices are global source-row ids, so every batch gathers from the whole vector.
                source: importedSource!,
                indices: indexView,
                output: gatherGraph.createDataView(
                  importShared(output, `sql-out-${label}-${gatherIndex}`),
                  {format, length: gather.rowCount}
                )
              })
            );
          }
          return new GPUData({
            format,
            buffer: output,
            length: gather.rowCount,
            ownsBuffer: false
          });
        });
        nodeIndex++;
        return new GPUVector({name: label, format, type: 'data', ownsData: true, data: chunks});
      };
      const columns: Record<string, GPUVector<'float32' | 'sint32' | 'uint32'>> = Object.create(
        null
      );
      const validity: Record<string, GPUVector<'uint32'>> = Object.create(null);
      for (const entry of schema) {
        const source = compiled.table.gpuVectors[entry.name] as GPUVector<
          'float32' | 'sint32' | 'uint32'
        >;
        columns[entry.name] = gatherInto(entry.name, source, entry.format);
        const sourceValidity = compiled.validity[entry.name];
        if (sourceValidity)
          validity[entry.name] = gatherInto(`${entry.name}-validity`, sourceValidity, 'uint32');
      }
      if (!useCopy) {
        const gathered = gatherGraph.compile();
        gathered.encode(copyEncoder, {parameters: undefined});
        device.submit(copyEncoder.finish());
        const fence = device.createFence();
        await Promise.race([fence.signaled, device.lost]);
        fence.destroy();
        gathered.destroy();
      } else {
        device.submit(copyEncoder.finish());
        const fence = device.createFence();
        await Promise.race([fence.signaled, device.lost]);
        fence.destroy();
      }
      if (!isCurrent()) throw new Error('SQL sources changed during execution');
      // Time columns also get exact time words (bounded 8-byte-per-row readback) so the host can
      // type them as Kepler timestamp fields for the time filter and charts.
      const timeColumns: Record<
        string,
        Awaited<ReturnType<typeof readResidentSqlTimeColumn>>
      > = Object.create(null);
      for (const name of timeNames)
        timeColumns[name] = await readResidentSqlTimeColumn(
          name,
          columns[name] as GPUVector<'sint32'>,
          validity[name]
        );
      if (!isCurrent()) throw new Error('SQL sources changed during execution');
      const dataset = session.adoptResidentDataset({
        id: resultId,
        schema,
        columns,
        validity,
        ...(timeNames.length ? {timeColumns} : {}),
        schemaMetadata: {'luma.gl.sql': query.sql},
        request,
        cpuReservation: cpuReservation as never,
        gpuReservation: gpuReservation as never,
        destroy: () => {
          for (const buffer of retained) buffer.destroy();
        }
      });
      if (!dataset) throw new Error('SQL result was superseded before publication');
      adopted = true;
      return {dataset, rowCount, materialization: 'resident'};
    } finally {
      compiled.destroy();
      if (!adopted) {
        for (const buffer of scratch) buffer.destroy();
        gpuReservation?.release();
        cpuReservation?.release();
      }
      await session.registry.finishRequest(request);
    }
  }

  destroy(): void {
    this.pendingRelease?.();
    this.pendingRelease = null;
    this.registered = [];
    this.context = null;
    this.signature = '';
  }
}

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

/** Wraps a resident dataset's prepared scalar columns as a borrowed GPU dataframe, or a reason. */
function makeSqlFrame(source: KeplerArrowDataset): GPUDataFrame | string {
  const vectors: Record<string, GPUVector> = Object.create(null);
  const validity: Record<string, GPUVector<'uint32'>> = Object.create(null);
  const dictionaries: Record<string, readonly unknown[]> = Object.create(null);
  for (const [name, vector] of Object.entries(source.columns)) {
    if (!vector || !RESIDENT_SQL_FORMATS.has(vector.format ?? '')) continue;
    vectors[name] = new GPUVector({
      name,
      type: 'data',
      format: vector.format,
      data: vector.data,
      ownsData: false
    });
    const columnValidity = source.analytics.validity[name];
    if (columnValidity) validity[name] = columnValidity;
    const dictionary = source.prepared.dictionaries[name];
    if (dictionary) dictionaries[name] = dictionary.values;
  }
  if (!Object.keys(vectors).length) return 'No scalar resident columns';
  return assembleSqlFrame(vectors, validity, dictionaries);
}

/** Column formats and dictionary labels of a frame, for string and CAST planning. */
function makeSqlTableInfo(name: string, frame: GPUDataFrame): ResidentSqlTableInfo {
  const columns: Record<string, ResidentSqlTableInfo['columns'][string]> = Object.create(null);
  for (const columnName of frame.columnNames) {
    const entry = frame.dictionaries[columnName] as
      | readonly unknown[]
      | {readonly values?: readonly unknown[]}
      | undefined;
    const dictionary = Array.isArray(entry)
      ? entry
      : (entry as {values?: readonly unknown[]} | undefined)?.values;
    columns[columnName] = {
      format: frame.table.gpuVectors[columnName]!.format as 'float32' | 'sint32' | 'uint32',
      ...(dictionary ? {dictionary: dictionary.map(String)} : {})
    };
  }
  return {name, columns};
}

/** Executor for the native SQL explorer: runs on the GPU, then hands the result to the host. */
export function makeNativeSqlGpuExecutor(
  runtime: ResidentSqlGpuRuntime,
  publish: (execution: ResidentSqlExecution) => void | Promise<void>
) {
  let lastDescription = '';
  return {
    planner: runtime.planner,
    describeTables: () => runtime.getRegistration(),
    describeLastResult: () => lastDescription,
    async execute(query: ResidentSqlQuery): Promise<number> {
      const execution = await runtime.execute(query);
      lastDescription =
        execution.materialization === 'resident'
          ? 'GPU-resident'
          : `bounded readback: ${execution.fallbackReason ?? 'unspecified'}`;
      try {
        await publish(execution);
        execution.commit?.();
      } catch (error) {
        await execution.discard?.();
        throw error;
      }
      return execution.rowCount;
    }
  };
}

const SQL_RUNTIMES = new WeakMap<KeplerComputeSession, ResidentSqlGpuRuntime>();

/** One GPU SQL runtime per compute session, so frames survive host re-renders. */
export function getResidentSqlGpuRuntime(session: KeplerComputeSession): ResidentSqlGpuRuntime {
  let runtime = SQL_RUNTIMES.get(session);
  if (!runtime) {
    runtime = new ResidentSqlGpuRuntime(session);
    SQL_RUNTIMES.set(session, runtime);
  }
  return runtime;
}
