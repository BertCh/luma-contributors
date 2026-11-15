<!-- SPDX-License-Identifier: MIT -->
<!-- SPDX-FileCopyrightText: Copyright (c) vis.gl contributors -->

# Kepler source parity audit, 2026-11-14

Wave 3, `parity-sources`, first round. This is a static audit and Node/build verification, **not a browser acceptance result**. The 17 Playwright runs await the driver. No source-path port regression was established in this round; the concrete fixes below repair pre-existing script assumptions. No runtime feature or tolerance was removed or relaxed.

## Reference and scope

Compared current sources with `e0ed33791` using read-only `git show` in `~/Documents/GitHub/vis.gl-build/luma.gl`. Read the KE-PORT roadmap, example README, all three evaluation digests, and `.typecheck/review/contributors-reuse/report.md`.

The old and current `native-auto-prepare.ts`, `native-dataset-source.ts`, `native-hosts.ts`, `source-type-routes.ts`, `remote-source-loader.ts`, `native-source-access.ts`, `local-cloud-provider.ts` and `native-time-converter.ts` are identical. The source/publication/preview modules' port changes are imports of contributor normalization/session APIs and the renamed geometry/trip preparation helpers. Those imports resolve. No new ingestion implementation was needed; the only new helper is for acceptance scripts.

## Per-script audit and changes

All script names below are under `examples/deck/kepler-compute/scripts/`. Each changed script uses `source-check-utils.mjs` to preserve unrelated URL parameters while setting the startup population its assertions require, or to resolve its fixture page when a driver supplies the app's base URL. Previously an explicit URL bypassed the script's default query/path. The retry scripts now merge `retry` into an existing `automatic` policy instead of silently leaving it disabled.

| Script | Cause/evidence and fix | Runtime dependencies checked |
| --- | --- | --- |
| `native-csv.mjs` | Harness URL assumption; set 100,000 startup rows. Existing ingestion/render/policy assertions retained. | Native Load Files, `getState`, `renderedLayers`, auto-prepare report/notice. |
| `native-arrow-stream.mjs` | Harness URL assumption; set 256 rows. | Stream file/URL controls, pause/next/continue/EOF/partial/cancel labels; publications, `sourceBatches`, `sourceBufferIds`, `setPublishedRange`. All present. |
| `native-source.mjs` | Base URL incorrectly navigated to app. Resolve `native-source.html`. Its fixture omitted time-parser registration in the old example too; load `native-utils-host` before timestamp materialization. | `fixtures/native-source-check.mjs`, native KeplerTable/import/append, `makeArrowSourceFromNativeDataset`, `window.nativeSourceResult`. |
| `native-saved-map.mjs` | Config was sampled before asynchronous preparation finished. Raw rendering counted the startup layer and could pass a failed upload. Wait for each Prepared outcome and count/render uploaded layer IDs only; set 100,000 startup rows. | All 6 raw and 7 saved fixtures exist; native config/layer/filter/tooltip hooks exist. Saved layer assertions remain intact. |
| `source-export.mjs` | Base URL incorrectly navigated to app. Resolve `source-export.html`. | `serializeSelectedSource`, `window.sourceExportChecks`; all script assertions also executed through Vite SSR. |
| `source-only.mjs` | Stale assertion expected Uint64 sorting disabled. Old `original-table-sort.ts:isOriginalSortType` and `resident-table-provider.ts` already enable exact bounded sorting with `readRows`. Require enabled and verify exact sorted values/row IDs; set 256 startup rows. | H3 publication, picking, original/prepared table, native count chart, export/republication APIs/selectors present. |
| `source-preview.mjs` | Base URL incorrectly navigated to app. Resolve `source-preview.html`. | Preview field/precision controls, explicit publish, `window.sourceUploadCalls` present. |
| `source-types.mjs` | Harness URL assumption; set 100,000 rows. | Native typed Arrow ingestion, converted Int64/retained nested field routes, auto-prepare report present. |
| `original-table.mjs` | Harness URL assumption; set 256 rows. Correct success log that contradicted its enabled-sort assertions. | Original/derived value display and sorting, table controls, analysis region, publication/storage hooks present. |
| `remote-source.mjs` | Harness URL assumption; set 100,000 rows. | URL/status/error/cancel controls, CSV/GeoJSON decoders, mocked fixture routes present. |
| `csv-resume.mjs` | `norange` omitted Accept-Ranges, but both old/current cross-origin importer optimistically probe hidden headers. Expose explicit `Accept-Ranges: none`; retain no-Range assertion. Also assert original If-Range validator in range mode. Set 256 rows and enable retry. | CSV parser and byte-resume API present; Node interruption/oracle tests pass. |
| `stream-resume.mjs` | Same missing-vs-explicit-none fixture mismatch and URL/retry assumptions as CSV resume. Same fixes, including If-Range assertion. | Stream controller, EOF/publication, preserved `[2,0,3]` batches, resume status text present. |
| `cross-origin-resume.mjs` | Harness URL/retry assumption; set 256 rows and merge retry policy. Hidden-header and rejected-preflight expectations remain unchanged. | Optimistic range, Last-Modified validator, hidden Content-Range length check and restart fallback present. |
| `restore-rollback.mjs` | Harness URL assumption; set 1,024 rows. | Portable bundle controls, `injectRestoreFailure`, persistent notice, publications/nativeLayers/ready hooks present. |
| `publications.mjs` | Harness URL assumption; set 4,096 rows. | Explicit precision/roles/publication, point/geometry layer IDs, table/filter/removal controls present. |
| `published-datasets.mjs` | Base URL incorrectly navigated to app. Resolve `published-dataset.html`. | Fixture now imports `@luma-contributors/kepler`; published runtime, preview, GPU sorting/submission APIs resolve in fixture build. |
| `native-cloud-provider.mjs` | Harness URL assumption; set 4,096 rows. | `__lumaLocalCloudProvider`, upload/list/download/delete, selection save/replay hooks present. Existing explicit public-publication refusal retained. |

Additional files changed:

- `scripts/fixtures/native-source-check.mjs`: isolated host registration described above.
- `scripts/source-check-utils.mjs`: URL setup and asynchronous preparation outcome waiter. Comments explain the old harness failures.
- `scripts/kepler-demo-files.mjs`: use real CSV records for counts (quoted text/geometry can contain commas/newlines); wait for Prepared, exclude startup layers, require uploaded visible layers to finish loading. The old `layers >= 0` assertion could never fail.
- `modules/kepler/test/cross-origin-recovery-sharing5.node.spec.ts`: verify exact resumed bytes and no Range requests for explicit `none`, with optimistic cross-origin probing enabled.

## Upload/source coverage: old versus port

No removed source entry path was found. Both hosts call native `loadFiles` and install the same automatic preparer/table factory. Source-only and explicit preparation continue to use contributor normalization and session ownership.

| Entry path | Old and current behavior | Qualification still needed |
| --- | --- | --- |
| Native CSV/TSV/DSV and JSON/GeoJSON | Kepler native processor/analyzer, then automatic resident preparation under the same dataset ID. | Browser upload, selection, rendering and saved config retention. |
| Native Arrow/Feather and Parquet | Kepler's lazy loader registry includes `GeoArrowLoader` and `ParquetArrowLoader`; both dependencies resolve in this installation. | Native binary uploads and Parquet WASM execution are not proved by static resolution. |
| Native additional formats | Registry also retains NDJSON/JSONL, KML/GPX/TCX, shapefiles/ZIP, Excel and FlatGeobuf. The corresponding installed loaders resolve. | These are outside the 17 scripts; installed packages are not proof that every format works in the WebGPU host. |
| Prepare data local files and URL | Custom loader accepts CSV/TSV, GeoJSON/JSON record arrays, Arrow IPC and Parquet/GeoParquet. Parquet uses installed `parquet-wasm`; WKB geometry metadata feeds the public loaders.gl codecs. | Remote and browser WASM execution. Saved-map JSON belongs in native Load Files, not this decoder. |
| Sequential Arrow stream | Dedicated IPC-stream controller with explicit EOF/partial completion, bounded admission, retained batches and ownership transfer. Feather/file input uses the file importer. | Driver `native-arrow-stream` and both `stream-resume` modes. |
| Saved maps and portable workspaces | Native Kepler JSON loads through native files; portable Arrow bundle export/replay and transaction rollback remain separate, supported workflows. | Full browser replay and injected mid-restore rollback. |
| Cloud/publication | Only the `This browser` provider is registered; IndexedDB with memory fallback, private portable bundles and local mock account. Public URLs/storage/auth are not configured. | Driver save/list/download/replay. Remote providers/public hosting were absent in the old example too. |

Type policies are unchanged: exact decimal categories for wide integers/decimals by default; declared Float64-to-Float32 conversion; supported timestamps retain time words; unsupported nested/binary types stay retained/excluded with reasons; WKT/WKB geometry derives an explicit geometry column. That is not support for arbitrary nested Arrow fields as GPU scalar analysis inputs.

## Demo datasets

`kepler-demo-files.mjs --download-only` found all six in the existing gitignored cache; no download or dataset commit was needed. Independent CSV/JSON inspection and decoding through the current `makeSourceCandidateFromBytes` produced:

| Dataset | Rows | Columns | Bytes |
| --- | ---: | ---: | ---: |
| earthquakes | 54,936 | 12 | 4,639,421 |
| nyctrips | 97,986 | 12 | 12,087,115 |
| sftrees | 65,499 | 12 | 8,619,511 |
| movement-pittsburgh | 792 | 13 | 1,514,863 |
| county-unemployment | 3,220 | 14 | 2,665,541 |
| buildings-australia | 597 | 22 | 559,161 |

These fit the existing automatic materialization cap (1,000,000 rows/16,000,000 cells). Decoding is **not** native upload/analysis/rendering proof. The raw generic URL loader does not suggest pickup/dropoff coordinate pairs for nyctrips or geometry-text columns for movement-pittsburgh; native Load Files uses Kepler layer inference and geometry-column variants. Driver should run all six with the tightened demo script.

## Executed checks

- Selected Node batch 1: 10 files, 49 tests passed: `csv-resume`, `resumable-remote-stream`, `cross-origin-recovery-sharing5`, `restore-transaction`, `host-arrow-stream`, `host-stream-provenance`, `source-only-preview`, `session/source-export`, `host-h3-publication`, `polygon-publication-source` (`*.node.spec.ts`).
- Selected Node batch 2: 10 files, 96 tests passed: `local-cloud-service`, `automatic-recovery-policy`, `resident-original-table`, `original-table-sort`, `resident-materialization`, `recipes/sandbox-state`, `recipes/sandbox-host`, `recipes/native-config-sources`, `session/arrow-stream`, `h3-preview` (`*.node.spec.ts`).
- After adding the explicit-none case, `yarn vitest run --project node modules/kepler/test/cross-origin-recovery-sharing5.node.spec.ts`: 4 tests passed. This reruns 3 tests from batch 1. A final combined run of all 20 files passed all 146 tests.
- Final combined Node command: `yarn vitest run --project node modules/kepler/test/{csv-resume,resumable-remote-stream,cross-origin-recovery-sharing5,restore-transaction,host-arrow-stream,host-stream-provenance,source-only-preview,host-h3-publication,polygon-publication-source,local-cloud-service,automatic-recovery-policy,resident-original-table,original-table-sort,resident-materialization,h3-preview}.node.spec.ts modules/kepler/test/recipes/{sandbox-state,sandbox-host,native-config-sources}.node.spec.ts modules/kepler/test/session/{source-export,arrow-stream}.node.spec.ts`: 20 files / 146 tests passed.
- `NODE_OPTIONS=--max-old-space-size=8192 yarn tsc -p .typecheck/wave3-sources.json --noEmit`: passed for 14 source entry files, the changed Node spec and the example's two ambient declaration files. The first config omitted the ambient files and reported missing `d3-dsv` types; fixed the scoped config, with no source/type suppression.
- Vite production multi-entry build of **all 8** `scripts/fixtures/*.html` and their transitive modules: passed, including after fixing native-source registration. Output is outside the repo at `/tmp/kepler-parity-sources-fixtures`. This proves module resolution/linking, not DOM/GPU runtime behavior.
- Vite middleware/SSR (no browser): decoded all six cached demos; executed `source-export-check.mjs`, `sandbox-bundle-check.mjs` and `analysis-export-check.mjs` to completion. Reused every assertion from `source-export.mjs` against its SSR result: passed. Three URL-helper cases passed; a separate preparation-waiter check verified extension-bearing labels and immediate failure reporting. Bundle and analysis fixture results were inspected, not their Playwright scripts run.
- `node .../scripts/kepler-fixtures/validate-saved-maps.mjs`: exit 0, 9 raw fixtures and 7 saved maps validated with native schemas/processors. It emitted a DNS warning fetching the optional hosted icon catalog; the acceptance script already mocks that catalog locally.
- `node .../scripts/kepler-demo-files.mjs --download-only`: all six cached, none skipped.
- `node scripts/check-upstream-first.mjs`: no violations. `node scripts/check-contributor-contract.mjs`: no new findings.
- `node --check` on all changed `.mjs` files and `git diff --check`: passed.
- Formatting: the root Biome includes `.js/.ts` but excludes `.mjs`. The required `yarn biome check --write <files>` therefore initially processed zero scripts. Re-ran with `.typecheck/wave3-biome.json`, extending the same formatter settings with `.mjs` included and formatter-only checking; all 20 script files formatted. The changed TypeScript spec passed the normal Biome check.

## Gaps and next driver run

No new published-luma source parity gap was established by the available checks. Full Kepler upload, analysis and UI parity is **not yet demonstrated**. Chromium/dev-server listening is unavailable in this sandbox; no Playwright/GPU run was attempted. Run the 17 scripts (CSV/Arrow resume in range and norange modes; cross-origin resume in hidden and preflight modes), plus all six demos. Saved-map all-layer-types still asserts every saved layer; nothing was disabled. The scope does not qualify every other native upload format or arbitrary Kepler analysis/layer behavior.

Existing limits are visible in both references: browser-local cloud only, public publishing refused (`local-cloud-provider.ts:hasSharingUrl/uploadMap`); nested GPU scalar exclusions (`source-type-routes.ts`); explicit materialization/memory/stream admission limits. They are not regressions introduced by the port, and were not relaxed here.

Full repository build/test/headless gates and full example tsc were not run; only the scoped checks above. No git state-changing command was used.

## Round 2: driver failures traced and fixed

The driver reports 15/17 browser scripts passing after round 1. Remaining failures were `source-only` and two cases in `native-saved-map`. The following replaces any inference that the old script assertions themselves prove old behavior. No new browser run was possible here.

### Original H3 sort: retain enabled sorting, with explicit old-source evidence

There is no old header-disabling branch for the actual published provider at `e0ed33791`:

- `src/map-container.tsx:355,430` selects `publications.get(datasetId)?.consumer.table` in the old host as in the port.
- Old `src/published-dataset-runtime.ts:92` supplies `readRows: (sourceRows, columns) => this.inspector.readRows(sourceRows, columns)` unconditionally.
- Old `src/resident-table-provider.ts:158-159` computes `gpuSortable || (Boolean(provider.readRows) && isOriginalSortType(field.type))`.
- Old `src/original-table-sort.ts:14-24` accepts both integers (including Uint64) and Utf8. Old `src/resident-data-table.tsx:530` sets `disabled={!field.sortable}`.

Those availability expressions are unchanged in the current example (session package imports were renamed). Thus the old script's disabled assertion contradicts its own commit's provider, not a deck/luma-dependent enablement change. I cannot claim that assertion passed in the old browser. Round 1's explanation did not establish that distinction clearly enough. The script now cites these exact branches and retains the enabled/exact-order assertion.

Added a Node case in `resident-original-table.node.spec.ts` proving that an original Uint64 field is disabled without `readRows`, enabled with it, and sorts exact keys without prepared scalar fields. In tracing correctness, found and fixed a separate pre-existing bug: `original-table-rich-sort.ts` treated a selected population with no scalar identity column as all rows. It now requests bounded identity-only pages (`columns: []`), so source-only sorting preserves the selected population. The test selects only rows 1 and 3 and verifies exact order [3,1], excluding unselected rows 0 and 2.

### Uint64 Arrow export: fix the product before serialization

The first lossy-schema step is `remote-source-loader.ts:makeSourceCandidateFromTable`, not the Arrow writer or a BigUint64 vector fallback. It calls `routeArrowTable` before H3 roles are chosen and retains only its routed table. `source-type-routes.ts` converts Int64/Uint64 scalars to decimal text in place (`decimal-text` -> `getConvertedType` -> Utf8). Publication stores that table as `dataset.value.source`; the export writer correctly preserves the schema it receives, which was already Utf8.

The same routing/discard chain exists in old `remote-source-loader.ts:356-365`, `source-type-routes.ts:268-275,424-433`, and the old workspace file input. So this is a real product bug against the intended exact-source contract that the old code also contains; there is no evidence here that an Arrow/deck/luma version bump introduced it. The old Uint64 export assertion is retained unchanged.

Fix: candidates retain the decoded `originalTable`. When a user assigns an H3/DGGS source role, `restoreArrowSourceRoleColumns` substitutes that column's original field and batch data, sharing buffers and preserving nulls, schema metadata and batch seams. The workspace uses this exact table for role validation, preview and publication. Other wide scalar columns keep the existing decimal-category policy. The helper belongs beside the existing host type-routing adapter; no new generic luma API or analysis kernel was introduced.

Files: `src/remote-source-loader.ts`, `src/source-type-routes.ts`, `src/source-workspace-panel.tsx`. New `source-role-retention.node.spec.ts` exercises IPC decode -> routing -> role assignment -> real zero-scalar session import -> export -> IPC read -> reload. It verifies Uint64, exact null/zero/large IDs, metadata, `[3,3]` batches, zero scalar upload and unaffected wide scalar routing.

### Raw arcs CSV: stale rendering expectation, not lost layers

The fixture has `lat0,lng0,lat1,lng1,count,names,source_name,target_name`, 400 rows. Installed native `findPointFieldPairs` matches separator-delimited `lat/lng`, `lat/lon`, `lat/long` or `latitude/longitude`. A digit is not a separator. Calling the unmodified installed matcher body on these headers returns no pairs. Old `native-auto-prepare.ts:297-302` uses this same native matcher when there are no native layers; old `auto-prepare-plan.ts:inferLayerRolesFromFields` adds only native geometry/H3/recognized point roles. No explicit role is supplied by this raw file, so it is correctly table-only.

Old `scripts/native-saved-map.mjs:154-160` counted every layer, including the initial resident point layer, so `minimumLayers: 1` passed without any uploaded layer. Round 1's upload-specific check exposed that false positive; waiting for preparation did not delete a layer.

Adaptation in `native-saved-map.mjs`: this one raw fixture explicitly expects no uploaded layers and asserts one 400-row resident publication with all 400 rows selected. The comment explains why. `arcs-flows-map` still requires all saved arc/line/flow layers to render with explicit roles. No renderer or feature is disabled. This is the only relaxation of a rendering expectation, justified by the old native inference behavior rather than by the new failure.

### `anchorCommon` compile failure: Kepler globe text glyph/background helper

Exact emitter: installed `@kepler.gl/deckgl-layers/dist/esm/globe/globe-backface-cull.js:31`, `BACKFACE_CULL_INJECT`. `EnhancedMultiIconLayer` and `EnhancedTextBackgroundLayer` call it; native `@kepler.gl/layers/dist/esm/base-layer.js:1571-1577` chooses these for globe-mode label sublayers. The saved all-layer fixture has aggregate count labels on hexagon/grid/cluster. This is shared Kepler label code, not a flow-field/trip simulation shader. Its legacy `vs:DECKGL_FILTER_GL_POSITION` map injects the GLSL snippet into a WGSL hook too.

The old example pins the same Kepler 3.3.0-alpha.18; its tracked compatibility transform has only the filter patch, and its deck-core yarn patch contains no `BACKFACE_CULL_INJECT`/`anchorCommon` WGSL adapter. The dependency source is not tracked at that git commit, so I cannot substantiate an old WGSL counterpart from `git show`; I did not invent one.

Fix: `patchKeplerGlobeBackfaceCullSource` in `kepler-compute-transform.ts` replaces the pinned helper's legacy injection with a **public luma ShaderPlugin**, providing separate GLSL and WGSL strings. Original GLSL is retained verbatim. WGSL uses `let`, `vec4<f32>` and pointer assignment `*position`, preserving the globe guard, zero-radius/camera guard and original `radius / cameraDistance + 0.02` horizon rule. Existing hooks/plugins remain present. Wired into both the normal Vite transform and dependency optimizer (`vite.config.ts`), including viewer builds through the base config. No edits to node_modules, no luma source copies and no culling removal.

New `kepler-globe-label-cull.node.spec.ts` transforms and evaluates the actual installed helper, checks preservation/idempotence, resolves the plugin through public `resolveShaderPlugins` in both languages, and parses the resolved WGSL hook with `WgslReflect`. Actual WebGPU shader compilation remains the driver's check. Restart/reload Vite so the changed optimizer applies before rerunning `all-layer-types-map`.

### Round-2 verification and resource use

- `yarn vitest run --project node --maxWorkers=1 modules/kepler/test/source-role-retention.node.spec.ts modules/kepler/test/resident-original-table.node.spec.ts modules/kepler/test/kepler-globe-label-cull.node.spec.ts`: 3 files / 14 tests passed.
- `yarn vitest run --project node --maxWorkers=1 modules/kepler/test/host-h3-publication.node.spec.ts modules/kepler/test/source-only-preview.node.spec.ts modules/kepler/test/original-table-sort.node.spec.ts modules/kepler/test/session/source-export.node.spec.ts`: 4 files / 13 tests passed.
- `NODE_OPTIONS=--max-old-space-size=4096 yarn tsc -p .typecheck/wave3-round2.json --noEmit`: passed (changed source graph, Vite config/transform, three specs, ambient declarations).
- `node scripts/check-upstream-first.mjs`: no violations. `node scripts/check-contributor-contract.mjs`: no new findings.
- Scoped `yarn biome check --write` passed for changed TypeScript; formatter-only Biome config formatted the changed `.mjs` files. Both scripts pass `node --check`; `git diff --check` passed.
- A direct CommonJS import of native table/processors failed before the probe with installed deck versions 9.4.1 vs 10.0.0-alpha.1 (the browser has Vite aliases). For the native inference evidence, evaluated only the installed, unmodified `findPointFieldPairs` function body with its actual `TRIP_POINT_FIELDS`/separator constant and Babel iteration helper: 400 fixture rows, zero inferred pairs. No runtime packages or assertions were changed to work around the Node bootstrap failure.
- No full build, new dependency install, node_modules copy, browser launch or large output/cache generation. Used one test worker and capped tsc at 4 GiB. Browser reruns of the two failing scripts remain pending; the driver's 15 passes are reported evidence, not reruns performed here.

## Independent review follow-up: globe label injection order

Driver result before this follow-up: all 17 browser scripts passed after round 2; `native-csv` needed one retry after a cold Vite start. Independent review then found a P2 ordering defect in the culling replacement.

Cause: public luma assembly collects plugin injections before legacy `shaders.inject` contributions. The culling plugin's default order 0 therefore preceded an inherited position assignment at order 0 or above. That later assignment could undo the clip-space cull, contrary to Kepler's original append-after-existing behavior. Preserving the hook's presence alone, as the previous test did, did not establish correct execution order.

Fix in `kepler-compute-transform.ts`: move the inherited position-hook contribution into the appended plugin, preserving its string/object source and explicit order; keep all other legacy hooks and existing plugins untouched. Append the language-specific culling injection at positive-infinite order. Moving the inherited hook is necessary as well as setting the order: otherwise a legacy object-form hook at positive infinity would still be collected after the plugin and win the tie. With the inherited hook in the appended plugin, stable ties put all existing contributions before culling, including existing plugin/inherited hooks at infinity. The input descriptor is not mutated. The culling formula, margin and GLSL/WGSL source are unchanged.

Extended `kepler-globe-label-cull.node.spec.ts` uses public `resolveShaderPlugins`, `GLSLShaderAssembler` and `WGSLShaderAssembler` to assemble final shaders containing inherited position writes. Cases cover string hooks, object-form order 41 and object-form order infinity, each with another existing plugin at infinity, in both languages. Assertions verify the inherited/plugin ordering is retained and both execute before the culling assignment; the assembled WGSL is parsed too. All six new cases failed against the prior helper and passed after the fix (7 total tests, including the existing preservation/language test).

Verification: scoped `yarn vitest run --project node --maxWorkers=1 modules/kepler/test/kepler-globe-label-cull.node.spec.ts` passed 7/7; scoped tsc using `.typecheck/wave3-round2.json` with a 4 GiB heap passed; upstream-first and contributor-contract checks passed with no new findings; scoped Biome and `git diff --check` passed. Manifest entries were refreshed. No browser run, dependency installation or large build/cache was added for this follow-up.
