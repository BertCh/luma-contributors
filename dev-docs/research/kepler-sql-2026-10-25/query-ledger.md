# KG06: bounded Kepler SQL compatibility

Status: partial. This ledger scopes consumer evidence, implemented slices and remaining work;
it does not claim general DuckDB or OpenAssistant query parity.

## Pinned consumer evidence

Reviewed source checkouts, not running upstream applications:

- Kepler `4b5d4b4d9d4f73fbacaf973c292a871ddf657302`, package
  `@kepler.gl/duckdb` 3.3.0-alpha.18:
  [table utilities](https://github.com/keplergl/kepler.gl/blob/4b5d4b4d9d4f73fbacaf973c292a871ddf657302/src/duckdb/src/table/duckdb-table-utils.ts),
  `quoteTableName`, `quoteColumnName`, `castDuckDBTypesForKepler` (lines 96–154).
  The generator always quotes columns and tables, doubles embedded quotes, optionally emits
  `CAST(column AS DOUBLE)` for large integer/decimal types and `ST_AsWKB(column)` for geometry.
- OpenAssistant `a1160abe86912bc26ad5989569f3e5a55b9c6e57`, package
  `@openassistant/duckdb` 1.0.0-alpha.0:
  [local query](https://github.com/geodaai/openassistant/blob/a1160abe86912bc26ad5989569f3e5a55b9c6e57/packages/tools/duckdb/src/query-tool.ts),
  [merge prompt](https://github.com/geodaai/openassistant/blob/a1160abe86912bc26ad5989569f3e5a55b9c6e57/packages/tools/duckdb/src/merge-tool.ts).
  `localQuery` accepts model-authored SQL and calls `conn.query(sql)`; there is no finite emitted
  query corpus to certify. The merge prompt explicitly demonstrates
  `SELECT A.id, A.name, A.INCOME, B.POP, B.INCOME AS B_INCOME FROM A JOIN B USING (id)`
  and `SELECT id, name, income FROM A UNION ALL SELECT id, name, income FROM B`.
  Its manifest requests DuckDB WASM `^1.29.0` and Arrow `^17.0.0`; the direct reference below
  deliberately pins native DuckDB 1.4.1, rather than asserting identical host WASM versions.

## Supported and unsupported ledger / scoped subitems

| Subitem | Consumer requirement | Result |
| --- | --- | --- |
| KG06.1 | Kepler generated `SELECT "column" FROM "table"` | Implemented quoted identifiers, including spaces, reserved words, escaped double quotes and literal dots. Mixed quoted/unquoted qualified components resolve against explicit registered tables; aliases and ORDER BY aliases decode once. Existing numeric projections, filters, derived arithmetic, null propagation and sorting retain their GPU lowering. |
| KG06.2 | Merge `JOIN B USING (id)` | Implemented one unqualified shared key, lowered to existing inner/left/semi/anti GPU joins. Explicit left projections only; key must remain selected. Multi-key USING and SELECT * USING fail eagerly. Right columns fail resolution. This supports a bounded left-projection subset, not the complete demonstrated merge prompt. |
| KG06.3 | Kepler large integer/decimal `CAST(... AS DOUBLE)` | Unsupported and rejected; GPU dataframe arithmetic is not DuckDB f64. No unsafe identity cast is introduced. Requires an audited width/precision implementation. |
| KG06.4 | Kepler `ST_AsWKB` | Unsupported SQL function. Geometry serialization remains an explicit adapter concern. |
| KG06.5 | Assistant string/category expressions | String literals/functions and arbitrary text comparisons remain unsupported. Dictionary metadata can survive projection/output; GPU dictionary codes are not SQL text semantics. No finite list of assistant-emitted string expressions was established. |
| KG06.6 | Assistant null/derived expressions | Existing numeric arithmetic, boolean predicates, IS NULL/IS NOT NULL are supported. October 26: nullable COUNT(column), bounded numeric/boolean CASE and COALESCE are implemented over the shared expression/validity compiler; arbitrary functions remain rejected. No consumer generator here proves a finite required function list. |
| KG06.7 | Merge right projection / UNION ALL / composition | Right projection, UNION ALL, CTEs/subqueries, multiple joins remain unsupported. They need explicit schema and ownership-preserving composition, not concatenation or CPU fallback. |

Join preconditions are inherited from the existing GPU dataframe join: uint32 keys,
unique right keys, reserved-key restrictions and bounded index capacity. Duplicate right keys
are **not** many-to-many SQL joins: the GPU `contractViolation` word reports invalid input;
callers must consume that explicit diagnostic before trusting results. No planning-time GPU
readback is inserted to establish uniqueness. Existing tests cover these join diagnostics.
Qualified identifiers reference one registered table/alias, not database/schema resolution.
Table registration accepts arbitrary nonempty names; source names never become shader source.
Identifier lookup remains case-insensitive (including quoted identifiers), matching DuckDB.

## Validation and reproducible reference

`reference.py` requires `duckdb==1.4.1` and `pyarrow==22.0.0`. It registers Arrow float32/uint32
arrays with the same names and values used in SQL GPU tests. `reference-results.json` records
its actual output. The GPU tests compare the selected values, escaped output names and anti-join
row identities against these expectations; Arrow upload/readback remains explicit in
`@luma.gl/arrow`. This is a direct value comparison for these cases, not a live dependency
or a claim that every existing SQL feature matches DuckDB. No performance claim is made.

Planning tests also assert no buffer creation or submit, malformed quote rejection, unsupported
casts, unknown columns, multi-key/qualified USING keys, wildcard and right projection rejection.
Validation command results are recorded by the implementing agent after final edits.

Actual validation (2026-10-25):

- `nvm use`: Node 22.23.4.
- `yarn lint fix`: passed after final source edits (8062 files checked).
- `yarn test-node --maxWorkers=2 modules/graph/test/gpu-sql/lu-sql.node.spec.ts modules/graph/test/gpu-dataframe/describe/lu-sql-statement.node.spec.ts`: 10 tests passed.
- `yarn test-headless --no-coverage --maxWorkers=1 modules/graph/test/gpu-sql/lu-sql.spec.ts`: 7 tests passed on WebGPU, including the added quoted-name case, inner/anti USING and duplicate-key diagnostic.
- Native DuckDB/PyArrow `reference.py`: passed, producing checked-in `reference-results.json`.
- Full build/test/website gates are integration responsibilities; these focused results do not substitute for them.

## October 26 scoped follow-up

The user narrowed this follow-up to straightforward SQL additions; full SQL support, precision/text
extensions and relational composition remain future work. KG06.6 adds searched/simple CASE,
COALESCE, global/grouped nullable counts, typed dataframe `choose`/`coalesce`, and column-count
reductions. Compatible 32-bit scalar branches only, at most 64 branches/arguments and bounded
expanded AST size. No CPU evaluation or implicit packing/readback.

Direct reference: `modules/graph/test/oracles/sql-null-expressions.py`, DuckDB 1.4.1 and PyArrow
22.0.0, with checked-in fixture consumed by WebGPU tests. Cases cover null conditions/branches,
first-match selection, omitted ELSE, NaN/infinity counts, empty/all-null selection and grouping;
GPU-only cases add empty batches, uint32 maximum, parameter replay and inspection lineage.
Final verification on October 26 (no performance claim):

| Check | Result |
| --- | --- |
| `nvm use` | Passed: Node 22.23.4. |
| `yarn install` | Passed with existing peer-dependency warnings. |
| `yarn lint fix` | Passed after source/test edits. |
| `yarn build` | Passed after the final conditional-format validator fix, including all modules and CommonJS identity checks. |
| Focused node tests | 53 dataframe/SQL files, 417 tests passed on the final source. |
| Focused WebGPU tests | 53 dataframe/SQL files, 820 tests passed on the final source, including all 11 SQL cases. |
| Direct DuckDB oracle | Executed DuckDB 1.4.1/PyArrow 22.0.0 generator; committed fixture is consumed by GPU tests. |
| `yarn test` | Attempted; its node phase hit an unrelated `ENOSPC` write failure. The screenshot file passed all 7 tests after temporary-file cleanup. A later full node run passed 12,610 tests before the final validator fix. |
| Final `yarn test-node` | 12,610 tests passed, 439 skipped; one existing network kernel-audit case exceeded its 60-second timeout under concurrent build load. The identical case passed in an isolated one-worker retry (15.39 seconds for the run). The full command therefore remains recorded as exit 1, with the failed case resolved separately. |
| `yarn test-headless` | All eight shards attempted. Shards 1, 3 and 8 passed; shard 4 exposed the conditional-format bug, subsequently fixed and covered by the final 820-test run. Shards 2, 5, 6 and 7 lost browser connections before completing. All four interrupted files passed in isolation (7 tests), but their unexecuted shard tails were not rerun: this is **not a completed green full-suite gate**. |
| `yarn website:build` | Passed after the final source fix; 1210 raw documentation pages validated. Initial low-disk attempt exited 1. |
| `(cd website && yarn build)` | Passed after the final source fix with the same documentation validation. Initial low-disk attempt exited 1. |

The interrupted browser files were network-isochrones cell minimum, batch hash join, Parquet
byte-array dictionary decoding and the straight-line catchments recipe. No assertions, timeouts
or tolerances were weakened. The full-suite infrastructure interruptions remain separate from
the passing verification of the modified dataframe/SQL surface.
