# Stream `kepler`: KE-TESTS

## 1. Done

### Measurement: import-closure classification of the 271 excluded specs

Method: a TypeScript-AST walk (scripts in the scratchpad, `kepler/closure.mjs` and `kepler/luma-names.mjs`) from each spec through relative imports into `examples/deck/kepler-compute/src` and `modules/kepler/test/utils`. It stops at `modules/*/src` (workspace code). Every bare package import is recorded and checked against the root and `modules/kepler` `package.json` (declared) and `node_modules` (installed). Value imports and type-only imports are tracked separately. A second pass checks every named `@luma.gl/*` import in the closure against the published 10.0.0-alpha.10 `.d.ts` exports. On average a spec's runtime closure covers 12 example files.

| Class | Specs |
|---|---|
| Runtime closure reaches an undeclared package | 98 |
| - of which reach `@deck.gl/*` or `@deck.gl-community/gpu-layers` (35 reach gpu-layers) | 49 |
| - of which reach other undeclared packages only (`@kepler.gl/charts|actions|utils|components|cloud-providers`, `@math.gl/web-mercator`, `d3-time`, `d3-dsv`, `@flowmap.gl/data`, `react`, `a5-js`, `s2-geometry`, `proj4`, `parquet-wasm`, `@loaders.gl/scene`) | 49 |
| - of which reach at least one package that is not installed at all | 55 |
| Runtime closure clean, type-only imports reach undeclared packages | 74 |
| Runtime and type closure clean | 99 |
| Runtime-clean (173), but the closure names `@luma.gl/*` exports that published alpha.10 lacks | 59 |

The main hub files that pull in deck.gl are `analysis-cartogram-projection.ts` (`@deck.gl/core`, 26 specs), `cartogram-mercator-layer.ts` (gpu-layers, 25), `interpolation-runtime.ts` (`@deck.gl/layers`, 25) and `aggregation-runtime.ts` (`@math.gl/web-mercator` plus deck types, 13). These are reached through `analysis-runtime.ts` (2,618 lines) and other large runtimes, so moving "only the code the specs need" would mean copying thousands of lines. I chose not to do that.

### Route chosen: run the specs in place against the example sources, without copying anything

- All 173 runtime-clean specs were tried in place. 140 now run and pass.
- No example code was moved or copied under `modules/kepler/test`, and nothing is duplicated. Where a spec failed, its imports needed deck.gl, undeclared packages or unpublished luma.gl APIs. Moving code would not have fixed that.

### Files changed

- `test/kepler-compute-example-specs.json`: it was a flat array. It is now `{description, excludedFromRuns: {spec: reason}, excludedFromTypecheck: {spec: reason}}`, with a reason for every spec. It has 131 run exclusions and 58 typecheck-only exclusions.
- `vitest.config.ts` (kepler block, lines 7-14): reads `Object.keys(json.excludedFromRuns)`. The comment is updated.
- `scripts/typecheck-tests.mjs` (lines 12-21): excludes `excludedFromRuns` plus `excludedFromTypecheck`. The other exclusions (the example glob, `host-wkb-selection.spec.ts`, the two trip fixtures) are unchanged.
- `it.skip`/`test.skip` with a stated reason in four specs that otherwise pass:
  - `modules/kepler/test/binding3.node.spec.ts`: 'recognizes allocation failures only'. Needs `OutOfMemoryError`, which `@luma.gl/core` alpha.10 does not export.
  - `modules/kepler/test/binding5.node.spec.ts`: 'turns a reported device out-of-memory into a typed allocation error'. Same cause.
  - `modules/kepler/test/label-streamed.node.spec.ts`: 'bounds are measured per window...' and 'only visible rows get glyph instances...'. Need `walkTextGlyphs`, which is not in `@luma.gl/engine/text` alpha.10.
  - `modules/kepler/test/resident-result-readback.node.spec.ts`: 'builds one Arrow batch per GPU chunk...' and 'refuses invalid validity...'. Need `prepareArrowTableForGPUAnalytics`, which is not in `@luma.gl/graph/arrow` alpha.10.
- No change to `modules/kepler/src`: no re-enabled spec exposed a kepler source defect.

### Counts, before and after

| | Before | After |
|---|---|---|
| Kepler spec files excluded from runs via the JSON | 271 | 131 |
| Kepler spec files running (of 368; `host-wkb-selection.spec.ts` is still excluded separately) | 96 | 236 (+140: 125 node, 15 headless) |
| Tests in the re-enabled specs: passing / failing / skipped | none run | 858 / 0 / 9 (6 skips are mine, listed above; 3 are a pre-existing `describe.skipIf(!keplerNextFrame)` in `time-contract-playback.node.spec.ts`) |
| Re-enabled specs also in the test typecheck | none | 82 of the 140 (58 run but are not typechecked) |

I did not run the 96 kepler specs that were already running, so I have no test count for them.

### Remaining run exclusions by reason (131)

- 49: need deck.gl packages (`@deck.gl/core`, `@deck.gl/layers`, `@deck.gl-community/gpu-layers`, ...).
- 50: need other undeclared packages (list above). This includes `native-topology-filter.node.spec.ts`, which `require.resolve`s `@kepler.gl/table`/`utils` (installed only as transitive dependencies) and `@kepler.gl/components` (not installed).
- 7: fail to load because they load undeclared `@kepler.gl/table`/`utils`/`charts` files directly. `@babel/runtime` and `@kepler.gl/charts` are not installed. These are `resident-chart-plan`, `resident-hour-periods`, `resident-column-values`, `resident-numeric-plot-edges`, `resident-group-by-plan`, `resident-join-plan` and `resident-spatial-join-plan` (all `.node`).
- 9: fail to load in the browser. The example value-imports names that published luma.gl lacks: `@luma.gl/engine/arrow` geometry and trip APIs, `prepareArrowTableForGPUAnalytics` from `@luma.gl/graph/arrow`, and `alignGraphVectorViews`.
- 16: fail at run time on unpublished luma.gl APIs:
  - 9 chart runtimes read `stats.physicalTransientResourceBytes` / `compilationStats`.
  - `point-altitude-binding` calls `graph.planResources()`.
  - 4 trip and polygon node specs need `prepareTabularTrips`, `prepareGeoJSONTrips` and `prepareGeometrySource`.
  - `interpolation-geotiff` imports `three`, which is not installed.
  - `duckdb-host` is described under Noticed.

The reason for each spec is in the JSON.

### Typecheck-only exclusions (58; these specs run and pass)

- 21: type-only imports reach undeclared packages.
- 8: the closure names luma.gl exports that are not published.
- 29: both.

## 2. Verified

- Node runs: `npx vitest run --project node <25 files>` ×5 over the final 125 node specs. Result: 125 files passed, 821 tests passed, 0 failed, 9 skipped.
- Headless runs: `npx vitest run --project headless <5 files>` ×3 over the final 15 browser specs. Result: 15 files passed, 37 tests passed, 0 failed. Every file recorded at least one test.
- Earlier exploratory runs over all 173 runtime-clean candidates (node 121/138 files passed; headless 15/35) produced the failure reasons above. The first headless pass hit Vite's dependency-reload problem, which gave 0-test files, so I reran it in batches of 6.
- `npx vitest run --project node` over the 4 edited specs gave 22 passed and 6 skipped.
- `NODE_OPTIONS=--max-old-space-size=8192 node scripts/typecheck-tests.mjs`, run once: exit 2, **35 errors, all in another agent's file** `modules/analysis/src/gpu-spatial-analysis/spatial-clustering/gpu-optics.ts`. They are syntax errors (`TS1002 Unterminated string literal`, line 832 onward; that file has uncommitted changes). There are **0 errors in my files or the example**. However, tsc does not run semantic checking when there are syntax errors, so **this run does not confirm that the 82 re-included specs typecheck**. That has to be rechecked once `gpu-optics.ts` parses.
- `npx biome check --write` on the edited files: no fixes needed.

## 3. Not done

- Moving example code under `modules/kepler/test`: not done, deliberately. Every failing closure needs deck.gl, undeclared packages or unpublished luma.gl APIs, and moving the code would not remove any of them. The deck hubs sit in very large runtimes, so moving them would duplicate thousands of lines.
- Subagents: not used. The fixes amounted to six skips plus the JSON and config changes.
- `duckdb-host.spec.ts`: not diagnosed (see Noticed).

## 4. Wave 2 requests

- Confirm the test typecheck with the 82 re-included specs once `gpu-optics.ts` is fixed. If it fails, move the failing specs into `excludedFromTypecheck`.
- KE-EX owner: `examples/deck/kepler-compute` imports `prepareArrowTableForGPUAnalytics`, `PreparedArrowAnalyticsTable`, `PreparedArrowAnalyticsTimeColumn` and `PrepareArrowTableForGPUAnalyticsOptions` from `@luma.gl/graph/arrow`. These now live in `modules/analysis/src/analysis-core/graph-support/arrow-analytics-normalization.ts` but are not exported from the analysis barrel. Exporting them and repointing the example's imports would clear the luma.gl-name blocker for about 35 specs: 4 headless specs would load, and many more could join the typecheck. The analysis barrel owner would need to approve the export.
- Docs I do not own still say "271 of 367/368 excluded": `dev-docs/upstream-workarounds.md:54`, `CHANGELOG.md:23` and `dev-docs/roadmap.md:44,198`. Update them to 131 excluded from runs and 58 more excluded from the typecheck. Also note that the JSON format is now an object.

## 5. Roadmap

- KE-TESTS: replace the row with: "131 of 368 kepler spec files are still excluded from runs (`test/kepler-compute-example-specs.json`, `excludedFromRuns`, one reason per spec): 49 need deck.gl, 57 need other undeclared or uninstalled packages, 25 need luma.gl APIs that alpha.10 does not publish (graph stats/planResources, engine/arrow geometry and trips, OutOfMemoryError, walkTextGlyphs, graph/arrow analytics preparation). 58 more run but are left out of the test typecheck (`excludedFromTypecheck`). Six skipped cases cite unpublished luma.gl APIs. The rest follows KE-EX."
- KE-EX: unchanged. Add that 140 specs now depend on the example sources in place, so deleting the example would remove their coverage.

## 6. Noticed

- `modules/kepler/test/duckdb-host.spec.ts` (headless) fails inside `prepareArrowTableForGPUAnalytics` (`analysis-core/graph-support/arrow-analytics-normalization.ts`, the `new Table(schema, batches)` near line 199) with "Table and inner RecordBatch schemas must be equivalent" when importing the DuckDB SQL result. There is only one apache-arrow copy (17.0.0). One possible cause is that a rebuilt child chunk length differs from `batch.numRows`, so Arrow's `ensureSameLengthData` makes a field nullable and the schemas then differ. Not diagnosed; possible analysis defect.
- The test typecheck is currently blocked by syntax errors in `gpu-optics.ts`, an in-flight edit by another agent.

## Follow-up (after the coordinator's combined typecheck: 195 errors in total, 140 of them kepler)

Result: **0 kepler errors.** The final `NODE_OPTIONS=--max-old-space-size=8192 node scripts/typecheck-tests.mjs` run exits 2 with 66 errors. All 66 are in `modules/analysis/src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy.ts` (2) and `modules/analysis/test/**` (64), which belong to other agents. None are TS1xxx syntax errors, so semantic checking ran over all files. **Re-enabled specs included in the typecheck: 81 of the 140 that run** (the 59 others are in `excludedFromTypecheck`, each with a reason).

Spec fixes (strictness or API drift in the spec; no assertion weakened, behaviour unchanged):
- Index-signature access (`noPropertyAccessFromIndexSignature`, TS4111) changed to bracket access in:
  - `webcodecs-muxers.node.spec.ts` (10)
  - `host-table-bridge.node.spec.ts` (9)
  - `resident-sql-expressions.node.spec.ts` (2)
  - `resident-sql-strings.node.spec.ts` (1)
  - `resident-synchronous-guards.node.spec.ts` (3)
  - `raster-tiles.node.spec.ts` (1)
- `host-table-bridge.node.spec.ts:906`: reads `lumaComputePlot` through a typed destructure. `updateFilterPlot` adds that field, but `KeplerComputeFilter` does not declare it.
- `resident-sql-expressions.node.spec.ts`: casts use `readonly string[]` to match `ResidentSqlDerivedColumn.dictionary`.
- `acceptance-report.node.spec.ts`:
  - The local `createReport` takes `limits: Record<string, number>`.
  - `markdownOutput` is read through a cast, because the `.mjs` adds it after building the options.
  - `!` added on two `results.find(...)` calls.
- `analysis2-raster-terrain.node.spec.ts`, `globe-split-matrix.node.spec.ts`:
  - Removed the unused `@ts-expect-error` comments: with `allowJs`, the `.mjs` helpers are typed by inference.
  - `isValid` is read through a `Record<string, unknown>` report.
  - The region destructure no longer uses a mismatched cast.
- `duckdb-runtime.node.spec.ts`: `Promise.withResolvers` (ES2024; the lib is ES2022) replaced by a local `createDeferredTable()`.
- `flow-field-gpu-reference.node.spec.ts`: `ArrayLike.map` replaced by `Array.from(…, mapper)`.
- `session/source-only.spec.ts`: added the now-required `name: 'mask'` to `new GPUVector({type: 'data', …})`.
- `standalone-viewer-layers.node.spec.ts`: the filter fixtures gain the now-required `layerIds: []` and `enabled: true`. `selectFilteredRows` does not read either field.

Typecheck exclusion added (the spec keeps running):
- `modules/kepler/test/analysis4-policy-panels.node.spec.ts`. It imports the example's `analysis4-join-policy-panel.tsx` and `analysis4-suitability-policy-panel.tsx`. Those files account for the 79 example errors (TS7026/TS2874). They are written for the example's `jsx: react-jsx` setting and React types, while the root tsconfig uses `jsx: react` and declares no React. The other importer of the suitability panel, `suitability-category-scores.node.spec.ts`, was already excluded from runs.

Verified after the follow-up:
- `npx vitest run --project node` on the 12 touched node specs: 12 files passed, 99 tests passed.
- `npx vitest run --project headless modules/kepler/test/session/source-only.spec.ts`: 1 file passed, 2 tests passed.
- `npx biome check --write` on the touched files. The four remaining `noExplicitAny` warnings were already there before my edits (`acceptance-report.node.spec.ts:178-180`, `globe-split-matrix.node.spec.ts:14`).

## Batch 2 (export and repoint arrow analytics, diagnose duckdb-host)

### Steps 1 and 2: stopped, as the task requires when the code is a copy of luma.gl code

- **Correction to my wave-1 report.** The symbols are already exported. `modules/analysis/src/analysis-core/index.ts:6` does `export * from './graph-support/index'`, and `graph-support/index.ts` exports `prepareArrowTableForGPUAnalytics`, `PrepareArrowTableForGPUAnalyticsOptions`, `PreparedArrowAnalyticsTable`, `PreparedArrowAnalyticsTimeColumn` and the `ArrowAnalytics*` types. `modules/kepler/src/session/compute-session.ts:13` and `arrow-stream-publisher.ts:8` already import them from there. My earlier grep searched the barrel for the symbol name and missed the `export *`. No export change was needed.
- **Provenance.** The evidence says this is a copy of luma.gl code, so per the instruction I stopped and did not repoint the example:
  - It was first added at `modules/graph/src/arrow/gpu/arrow-analytics-normalization.ts` in `fd19008c`, inside the forked luma.gl graph package.
  - It was quarantined under `src/_pending` as a "non-analysis luma extension" in `58dd86d8`.
  - It was then "Rebuilt unchanged on public APIs" (`dev-docs/upstream-first-pending.md:8`).
  - `dev-docs/upstream-workarounds.md:49` lists it as a workaround whose upstream fix is a "Public `prepareArrowTableForGPUAnalytics` in `@luma.gl/graph/arrow`".
  - The example imports it from `@luma.gl/graph/arrow`, which is where the luma.gl branch had it.
  - This conflicts with AGENTS.md: "Never fork, re-implement or re-export luma.gl APIs".
  - Repointing the example would add consumers of that copy. It already has two in `modules/kepler/src`, which is an existing departure that someone should decide on.
- **The unlock is smaller than I estimated in wave 1.** I estimated "about 35 specs". A dry run over the closure data shows that repointing would at most:
  - let 4 headless specs load (`host-group-by`, `resident-sql-compose-gpu`, `resident-sql-expressions-gpu`, `resident-sql-strings-gpu`), with run-time results unknown;
  - move 3 specs back into the typecheck.
  
  The other specs whose only luma.gl problem is these names are still blocked by undeclared type packages, or by the compiled-graph stats and `planResources` APIs (8 specs).
- No edits were made to `examples/deck/kepler-compute/**` or `modules/analysis/src/**`.

### Step 3: `duckdb-host.spec.ts`, diagnosed. It is not a defect in the function, not a fixture problem and not a version mismatch

Evidence:
- There is one apache-arrow for the workspace and duckdb-wasm: `node_modules/apache-arrow` 17.0.0, and duckdb-wasm depends on `^17.0.0`. The 21.2.0 copies are nested under `@kepler.gl/*`, `@deck.gl/core` and `@loaders.gl/*`, and are not on this path.
- A temporary headless probe, since deleted, ran the real DuckDB worker SQL and then `prepareArrowTableForGPUAnalytics(result)`. The result table, fields and children are all `instanceof` the workspace Arrow classes; it has 1 batch of 2 rows; every field is nullable with `nullCount` 0. The call succeeds.
- A temporary copy of the spec, since deleted, passed alone and passed twice with its original batch neighbours (3 files, 5 tests).
- No file on the code path has changed since the base commit. That covers `graph-support/**`, `kepler/src/session/**` and `examples/.../duckdb-*.ts`.
- The failing run's log (`runs2/browser.part.aa.log`) shows `[vite] dependency optimized: @duckdb/duckdb-wasm` followed by `optimized dependencies changed. reloading`, at the moment the spec ran.
- Arrow's `compareSchemas` and `compareFields` check `other instanceof schema.constructor` (`apache-arrow/visitor/typecomparator.mjs:20-34`). With two module instances of the re-bundled Arrow chunk loaded after the reload, that check fails and produces "Table and inner RecordBatch schemas must be equivalent".
- Conclusion: the failure is a Vite dependency-optimizer artifact on a cold cache.
- Fix, in a file I do not own (`vitest.config.ts` outside the kepler block): add `@duckdb/duckdb-wasm` to the headless project's `optimizeDeps.include`.
- Until then the spec stays excluded from runs. A cold-cache CI run would flake. The JSON reason now says this.

### Step 4: re-classification

- The example's imports are unchanged, so no spec changes class.
- The JSON is regenerated with corrected reasons. The four "now lives unexported" reasons now say "analysis-core exports a copy of it, but repointing the example is on hold because that copy is luma.gl branch code". The `duckdb-host` reason is the diagnosis above.

### Counts, before and after Batch 2 (unchanged)

| | Before | After |
|---|---|---|
| Excluded from runs | 131 | 131 |
| Kepler spec files running | 236 (140 re-enabled) | 236 (140 re-enabled) |
| Tests in the re-enabled specs: passing / skipped | 858 / 9 | 858 / 9 |
| Re-enabled specs in the test typecheck | 81 | 81 |

### Verified

- `npx vitest run --project headless` on the temporary probe and spec copy: the probe printed its output; the spec copy passed alone (1 test) and twice with neighbours (3 files, 5 tests, each time). Both files are deleted.
- I did not run `scripts/typecheck-tests.mjs`. No TypeScript changed in this batch: only JSON reason strings changed, and the typecheck reads only the keys.

### Requests

- **Decision needed:** may the example and `modules/kepler/src` keep consuming the analysis copy of luma.gl's `prepareArrowTableForGPUAnalytics`? Alternatives are to take it upstream, or to classify it as analysis-owned. Only with a yes is repointing the example worth doing, and it would gain 4 runnable specs at most plus 3 typechecked ones.
- **`vitest.config.ts`** (shared): add `@duckdb/duckdb-wasm` to the headless `optimizeDeps.include`. After that, re-enable `duckdb-host.spec.ts`.
