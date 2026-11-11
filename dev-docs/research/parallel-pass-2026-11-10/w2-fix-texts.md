# w2-fix-texts report

## 1. Done
1. Catalog (website/src/catalog/entries):
   - geometry-construction-overlay.ts: line segmentize, smooth, chunk status now "count is lines; vertexCount (optional) is vertices; overflow flag" (source: line-segmentize-types.ts GPULineGeometryOutput status, gpu-line-segmentize.ts output doc). Polygon buffer and polygon overlay: outputs.status type `GPUPolygonOverlayOutput['status']`, count is polygons, requiredRingCount sizes ring capacity; overlay geometry description says no source ids (gpu-polygon-overlay.ts:127-150). Linear referencing: `{candidateOverflow, requiredCandidateCount?}` (gpu-linear-referencing.ts:96-118).
   - geometry-processing.ts rectangle clip: status `{count, overflow, vertexCount?, requiredCount?, requiredVertexCount?}`, count is lines (rings for polygons) (gpu-rectangle-clip.ts:129-132).
   - terrain-relief-light.ts GPUTerrainCastShadow rowDirection default now "from grid.affine[4]", description says a contradicting value throws (gpu-terrain-cast-shadow.ts:185-188).
   - dataframe-pointcloud-records.ts GPUDatumGridTransform tolerance default 1e-5 -> 5e-9 (gpu-datum-grid-transform.ts:46, DEFAULT_INVERSE_TOLERANCE_DEGREES).
   - Sweep: 532 entries, 3549 params, 275 params with a documented default checked against the TSDoc or constructor `??` of the class (script: scratchpad/sweep.ts, output scratchpad/sweep-out.txt); the "no default in doc" ones read in source by hand. No other default mismatch found. Outputs: check-catalog already errors on non-fields and omitted required fields; I grepped all entries for lineCount, candidateCount, requiredCount, trackValid, GPUKMeans, validatePackedView: the remaining hits are other classes where the name is current (point-in-polygon and nearest-join `candidateCount`, map-matching prop `candidateCount`, line-simplification `lineCounts`).
   - check-catalog: `0 errors, 2 warnings` (the warnings are pre-existing, space-time-kernel-density outputs).
2. space-time-variogram.ts run: `count: 7` replaced by TICK_SEGMENT_COUNT derived from the same step constants as the loops (5 time + 4 distance = 9). Screenshots read: 5 top ticks and 4 left ticks in dark and light.
3. Lens texts (modules/lenses/src): marked-point formula now N = n_a n_b for a != b, n(n-1) for a = b, ascii and doc comment say both cases and that a single-point same-type set has no K (gpu-marked-point-statistics.ts:90-95, 262-268); symbol bindings unchanged (`wgsl:result`). network-pack-structure.ts degree = live slots, self-loop once (gpu-network-analytics-columns.ts:72-75, 299-326). Conditional lens ascii: sentinel never applies to per-frame constants (gpu-raster-conditional.ts:53-54, 176-181). Empirical Bayes ascii: zero-weight slots are not neighbours, self slot skipped (gpu-spatial-empirical-bayes-rates.ts:74-80, 109-113). Variogram doc comment in pair-statistics.ts:18 gains `+0.045/N^2` (gpu-variogram.ts:497); the analysis-side comment already had it.
4. modules/analysis/src comments: geodesic-wgsl.ts example imports the radius from `@luma-contributors/analysis/analysis-core`; ports.ts GPUPackedGeometryPort comment names the two readers of input sourceIds; spatial-analysis-operators.ts contract for GPUPolygonOverlay and GPUBufferSurface is now `'GPUPolygonOverlayOutput'` (a string in the ontology data, not a comment; only that token is validated by the ontology spec); gpu-composite-score.ts doc says closed form for two indicators, cyclic Jacobi for more; gpu-shape-generator.ts comment describes geodesicDestination's half-angle form.
5. dev-docs/upstream-workarounds.md row now says `GPUSpatialKMeans` in `gpu-spatial-kmeans.ts` (and drops "Renaming ... is roadmap 12.4"). docs/api/computation-record.md: paragraph on device-specific timings (gpuBegin, gpuEnd, cpuEncodeMs, encoding.timestamped, timebase).
6. docs/api grep (lineCount, requiredCount, geometry.sourceIds, candidateCount, trackValid, GPUKMeans, validatePackedView): everything current (overlay and line pages already carry the new text; `requiredCount` hits belong to other classes; `GPUKMeans` in raster-embeddings.md:93 is the published class). No edit needed.

## 2. Verified
- `node --experimental-strip-types scripts/check-catalog.ts --quiet` (website): 0 errors, 2 warnings.
- `npx tsc -p website/tsconfig.json --noEmit`: no output. `npx tsc -p tsconfig.check-src.json --noEmit`: no errors in files I touched.
- `npx vitest run --project node modules/lenses/test`: 95 files passed, 1196 tests passed (376 skipped).
- `UPDATE_LENS_TWINS=1 ... lens-twins.node.spec.ts`: 7 passed; no file under docs/ or website/src changed afterwards (twins derive from records, which still hold the old pack text until the next capture).
- doc-parameters, doc-snippets, ontology specs: 6 files, 34 tests passed.
- `node scripts/verify-runs.mjs space-time-variogram --themes dark,light --out .reports/w2-fix-texts`: 2/2 PASS (server was not answering on 3301; the script started its own).
- biome check on all touched src files: clean (the run file was formatted with --write).
- Not run: yarn build, website-build, lens recapture.

## 3. Not done
- Lens records and website lens/generated copies still contain the old pack text (no regeneration was allowed); they refresh on the next capture, then twins must be regenerated.

## 4. Website and examples work list
None beyond what I edited.

## 5. Roadmap
None touched.

## 6. Noticed
- docs/api/computation-record.md line 145 still says the metrics helper converts to "nanosecond metrics"; the sidecar can also be `timebase: 'ms'`.
- GPUPolygonOverlayOutput['status'] in the catalog type string is not a name the barrel exports separately; the checker accepts it.
