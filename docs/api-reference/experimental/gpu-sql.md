import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU SQL

<ExperimentalDocsTabs active="gpu-sql" />

## Overview

`@luma-contributors/graph/gpu-sql` provides a bounded SQL frontend for
[`GPUDataFrame`](/docs/api-reference/experimental/gpu-dataframe). `GPUSQLContext` registers
existing GPU dataframes by name and lowers supported statements into the same immutable GPU
dataframe planner. Planning does not upload, submit, allocate query output, or read GPU buffers.

:::caution Experimental
`@luma-contributors/graph/gpu-sql` is marked `@experimental`. The supported SQL subset and the plan API may
change between releases.
:::

The optional loaders.gl integration lives in the
[GPU data analysis example](https://github.com/visgl/luma.gl/blob/master/examples/experimental/gpu-data-analysis/src/loaders-sql-query.ts).
Its `planGPUDataFrameQuery()` helper consumes the loaders.gl plan directly. Copy the adapter into
your application and install `@loaders.gl/core` and `@loaders.gl/sql` when needed.
The `@luma-contributors/graph/gpu-sql` module exports the native frontend without a loaders.gl dependency.

## When to use it

Use GPU SQL when an application already has GPU dataframe inputs and a constrained SQL surface is
more convenient than manually composing expressions, filters, aggregations, ordering, and joins.
Use GPU Dataframe directly when an application needs behavior outside the supported grammar.

Use the loaders.gl adapter when a query originates in a loader, catalog, or application component
that already emits `TableQueryOptions` and `SQLPredicate` values:

```ts
import {parseSQLPredicate} from '@loaders.gl/sql';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {planGPUDataFrameQuery} from './loaders-sql-query';

const predicate = parseSQLPredicate('fare >= :minimum AND category IN (0, 1)', {
  preserveParameters: true
});
const query = planGPUDataFrameQuery(frame, {
  predicate,
  columns: ['fare', 'category'],
  parameters: {minimum: 8}
});
const compiled = query.compile(new GPUCommandGraph(device));

const commandEncoder = device.createCommandEncoder();
compiled.encode(commandEncoder, {minimum: 10});
device.submit(commandEncoder.finish());
```

The compiled result stays GPU-resident as `table`, `validity`, `dictionaries`, `rowIndices`, and
`selectedCounts`. Rendering or another command graph can consume those resources directly. Arrow
readback remains an explicit adapter operation rather than the query's required result type.

## `GPUSQLQuery`

Plan a bounded SELECT statement into a dataframe query and compile it onto a caller-owned command graph.

::lens{scenario=gpu-sql.lu-sql-query rung=result caption="774 of 791 tracts pass population > 1,000; the other 17 stay out of the compact ids."}

`GPUSQLContext.query()` parses and validates one statement on the CPU and returns an immutable
`GPUSQLQuery`. It keeps the parsed `statement` (clauses and expression trees, names as written) next
to the plan; `describeStatement()` turns that AST into an `intent` graph with one node per clause
and expression node and dataflow edges in SQL evaluation order. Neither touches the GPU.

```ts
const query = new GPUSQLContext({tracts}).query(
  'SELECT population, noVehicle / households AS noVehicleRate FROM tracts WHERE population > :minimumPopulation',
  {parameters: {minimumPopulation: 1000}}
);

query.statement;
query.describeStatement();
const compiled = query.compile(graph);
```

#### How it computes

{/* lens:GPUSQLQuery */}

The graph rung links the dataframe plan back to the statement nodes through `derivedFrom`.

::lens{scenario=gpu-sql.lu-sql-query rung=datum bind=page datum=0 caption="Row 0 passes: 4,284 residents; 885 of 2,322 households have no vehicle (rate 0.381)."}
::lens{scenario=gpu-sql.lu-sql-query rung=formula bind=page datum=0}
::lens{scenario=gpu-sql.lu-sql-query rung=kernel caption="Look at the expression{N} lines: the SQL trees compile to the same statements as a dataframe query."}
::lens{scenario=gpu-sql.lu-sql-query rung=graph caption="SQL statement, dataframe expression tree, WGSL statements and command nodes, linked by derivedFrom."}

## Execution boundary

The execution boundary is explicit: Arrow input → GPU dataframe execution → GPU consumers.
Use `makeGPUAnalyticsTableFromArrowTable()` to upload input, construct a `GPUDataFrame`, compile
the query into a caller-owned `GPUCommandGraph`, then encode and submit the graph. Downstream
rendering and compute can use the compiled GPU resources directly. Call
`makeArrowTableFromGPUAnalyticsTable()` only when the application explicitly needs CPU readback.

## Supported grammar

- `SELECT *`, named columns, numeric expressions, and explicit `AS` aliases.
- One registered `FROM` table with optional aliases.
- Double-quoted identifiers, including spaces, reserved words, literal dots and doubled embedded
  quotes. Qualified components resolve against registered tables/aliases; lookup is case-insensitive,
  including quoted names.
- `WHERE` comparisons, `AND`, `OR`, `NOT`, parentheses, `IS NULL`, and `IS NOT NULL`.
- Numeric `CASE WHEN ... THEN ... ELSE ... END`, simple `CASE value WHEN ...`, and
  `COALESCE(...)`; boolean results can be used in predicates.
- Named `:parameter` values with planning defaults and encoding-time updates.
- One `ORDER BY` column with direction, null ordering, and optional bounded `LIMIT`.
- `COUNT(*)`, nullable `COUNT(column)`, `SUM`, `MIN`, `MAX`, and `AVG`, with an optional single-column `GROUP BY`.
  Dictionary-backed keys infer their group count. A plain `uint32` key needs a dense group count
  (`maximum key + 1`) because planning never reads GPU data:
  `context.query(sql, {groupCounts: {identifier: 8}})`. Without it planning throws.
- One `INNER`, `LEFT`, `SEMI`, or `ANTI JOIN` over a single equality key, expressed with `ON` or
  single-key `USING (key)`. USING requires explicit left projections that retain the join key;
  wildcard and right-column projections are rejected.

Unsupported statements, unknown columns, unregistered tables, unsupported strings, ambiguous
expressions, multiple sort/group keys, and `LIMIT` without ordering fail during CPU-only planning.
Applications still own command submission, synchronization, result readback, and cleanup.

Joins retain the dataframe contract: uint32 keys, unique right keys, reserved-key restrictions
and bounded index capacity. Consume the compiled join's `contractViolation` diagnostic (see
[GPU Dataframe](/docs/api-reference/experimental/gpu-dataframe)) before trusting results; duplicate right keys do not produce a many-to-many SQL join.

This subset accepts selected Kepler-generated numeric projections and bounded assistant merge
queries. `CAST(... AS DOUBLE)`, `ST_AsWKB`, arbitrary string functions,
right-column projections, `UNION ALL`, CTEs and subqueries remain
unsupported. Native DuckDB 1.4.1 reference fixtures verify the new quoted-name and USING cases,
not general compatibility with a Kepler DuckDB WASM version.

`CASE` selects the first valid true condition; a null condition does not match. Simple CASE
uses ordinary nullable equality, so `WHEN NULL` never matches. Omitting `ELSE` yields null.
`COALESCE` selects the first non-null operand. Branches must have compatible scalar formats;
these operations do not introduce implicit casts or text expressions. Each CASE/COALESCE accepts
at most 64 branches/arguments, subject to the existing 64-level nesting, 4096-token and expanded
expression-node limits. Scalar branches are evaluated as pure WGSL expressions before value and
validity selection; this is not a procedural short-circuit or exception-handling construct.

`COUNT(column)` counts selected entries with valid sidecars, including valid NaN/infinity values.
It returns a non-null `uint32`, including zero for empty/all-null input. Grouped counts retain
the existing dense-domain and null-key rules. Direct DuckDB 1.4.1/PyArrow 22.0.0
[fixtures](https://github.com/visgl/luma.gl/blob/master/modules/graph/test/oracles/sql-null-expressions.py)
cover these null-expression and count semantics.

```sql
SELECT COALESCE(fare, 0) AS filled,
       CASE WHEN fare >= 10 THEN fare ELSE NULL END AS highFare
FROM trips
```

The loaders.gl adapter currently supports predicate pushdown and output projection. It accepts
finite numeric, Boolean, and null predicate values, including reusable named parameters. Portable
source-order `limit`, streaming, in-flight cancellation, string/binary/date values, and relational
extensions are reported or rejected as unsupported instead of silently falling back to CPU work.

## Related modules

- [GPU Dataframe](/docs/api-reference/experimental/gpu-dataframe)
- [GPU Core](/docs/api-reference/experimental/gpu-core)
- [Arrow](/docs/api-reference/arrow)
