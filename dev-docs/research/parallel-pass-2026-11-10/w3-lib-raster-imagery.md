# w3-lib-raster-imagery

## 1. Done
Rename table (old -> new; no alias kept). Paths under `modules/analysis/src/raster-analysis/`.

| Class | Old | New |
| --- | --- | --- |
| GPURasterRegionGraph (and `addRasterRegionGraphTiles`, `GPURasterRegionGraphTilesProps`) | `output.status` (one uint32 view) | `output.status.iterationLimitReached` (one uint32 view, same word, same meaning; validated with `validateGPUStatusPort`) |
| GPURasterTexture, GPURasterConservativeRemapping, GPURasterTemporal, GPURasterSpectralUnmixing, GPURasterSeasonalBreak (+ `seasonal-break-extended.ts`) | `validity` (input) | `mask` |
| GPURasterRegistration (masked-linear mode) | `referenceValidity`, `movingValidity` | `referenceMask`, `movingMask` |
| 17 time-series classes: BestAvailablePixel, BFASTMonitor, ClimateRunAccumulator, COLDMonitor, EOFProjection, GeometricMedianComposite, HANTS, HarmonicRegression, LaggedCorrelation, MedoidComposite, RollingMean, SnapshotCovariance, StandardizedIndexCalibration, StandardizedIndex, StreamingAccumulator, TemporalResample, WhittakerSmoother | `validity` (input) | `mask` |
| Embeddings: KMeans, Covariance, LinearProbe, NearestPixels, Projection, Similarity, KNearestClassifier, ZoneMean, PyramidLevel (input only) | `validity` (input) | `mask` |
| GPURasterEmbeddingChange | `beforeValidity`, `afterValidity` | `beforeMask`, `afterMask` |
| GPUChangeDetection | `validity` (per-observation input; `mask` stays the per-cell filter) | `observationMask` |

Unchanged on purpose: `output.validity` of GPURasterEmbeddingDequantize and GPURasterEmbeddingPyramidLevel (output flag, already the rule). Internal helper names (`createRasterEmbeddingMatrix` argument, luma.gl `validity` chunk field, internal WGSL bindings `referenceValidity`/`movingValidity` in masked registration, which would collide with local WGSL variables) are not API.
COLD monitor: only the prop name and its WGSL binding changed; its logic, fixtures and the stability test of the other session are untouched. Results are identical everywhere (renames only).

Consumers updated: 8 kernel-audit case files (`cases-coverage-raster.ts`, `cases-rts-01-03-embedding-fdr.ts`, `cases-rts-time-series.ts`, `cases-rts04-rts07-monitoring-eof.ts`, `cases-rts05-compositing.ts`), 15 specs under `test/gpu-raster/{remote-sensing,raster-time-series,raster-embeddings,change-detection}`, lenses `remote-sensing-runner.ts` (region graph output, temporal `mask`, registration `movingMask`), docs `remote-sensing.md`, `raster-time-series.md`, `raster-embeddings.md`, `raster-change-monitoring.md`, `operations-analysis.md` (line 1338). New spec: `remote-sensing.node.spec.ts` "STD-04 and STD-07 ..." (status port shape rejection with `^region-graph iterationLimitReached must contain one uint32 row`, texture `^texture values and mask must hold`).

### Per-class state
| Class | Status ports | No-data props | Remains |
| --- | --- | --- | --- |
| RegionGraph | `iterationLimitReached` | labels (not a filter) | - |
| Registration | per-result code in `output.status` (0 ok, 1 flat/nonfinite, 2 ambiguous peak, 3 inadequate support), documented in class TSDoc | `referenceMask`, `movingMask` | codes are a per-result status word, not scalar counts; no named port added |
| Texture | none (cannot fail) | `mask` | takes bare views, not published bands |
| ConservativeRemapping | per-cell code in `output.status` (documented) | `mask` | same |
| Temporal | per-cell code (1 support, 2 rank), documented | `mask` | same |
| SpectralUnmixing | per-cell code (incl. 4 = iteration limit), `iterationCounts` per cell | `mask` | same |
| SeasonalBreak | per-cell code | `mask` | same |
| 17 time-series classes | per-cell `status` words, untouched | `mask` | outputs `status` words unchanged; bands not published bands |
| 9 embedding classes + Change | none | `mask` / `beforeMask`,`afterMask` | - |
| ChangeDetection | none | `mask` (cells) + `observationMask` (cell x slice) | - |

## 2. Verified
- `npx vitest run --project node modules/analysis/test/gpu-raster modules/analysis/test/analysis-core/kernel-audit modules/lenses/test`: 4338 passed, 376 skipped, 1 failed (`lens-twins.node.spec.ts` documentation twins, timeout under load); rerun alone: 7/7. Kernel-audit directory again after last edit: 50 files, 2701/2701.
- `npx vitest run --project headless` on `test/gpu-raster/{raster-time-series,change-detection,remote-sensing,raster-embeddings}`: 21 files, 275/275 (includes `rts04-cold-pyxccd.spec.ts` and `rts04-change-monitoring.spec.ts`, fixtures untouched; both were 38/38 before any edit and are in the 275 after).
- `node scripts/typecheck-status.mjs 'raster-analysis|remote-sensing|lenses|kepler|embedding'`: 0 errors. Scratch tsconfig over 1249 files (all spec/audit files of these dirs, `modules/lenses/{src,test}`): 0 errors.
- `node scripts/check-upstream-first.mjs`: no violations. `node scripts/check-contributor-contract.mjs`: 1 new finding, `raster-cell-area-scale.ts` root-file (not mine).
- `npx biome check --write` on my source and spec dirs.
- Not run: the vocabulary spec by name (kernel-audit and gpu-raster node projects, which hold it, pass), parity specs, website, examples.

## 3. Not done
- STD-04 per-cell and per-result codes (registration, remapping, temporal, unmixing, seasonal break, the 17 time-series words) not converted to scalar named ports: they are documented per-row status words (contract rule 9); a scalar count would be a new reduction node and a new output, not a rename. Unmixing/seasonal "iteration limit" stays a per-cell code 4/8.
- STD-07 "through the published band validity" for classes taking bare value views: not done (needs `GPURasterBand` inputs, a larger breaking change).
- Operator catalog entries: they carry no prop names, no edit needed.

## 4. Work list for others
- Regenerate lens records (LN-REC2): `docs/lens-records/gpu-raster.raster-{texture,registration,region-graph,conservative-remapping,temporal,spectral-unmixing,seasonal-break}.json` and `twins/*~graph.json` (WGSL binding `validity` -> `mask`, region-graph `status` output now `iterationLimitReached` binding). Not touched.
- Website/examples: `git grep -nw "validity"` in `website/src/gpu/runs/entries` and `examples/` for the classes above and replace per the rename table: `validity:` -> `mask:` (Texture, Remapping, Temporal, Unmixing, SeasonalBreak, the 17 time-series, embedding classes except Dequantize/PyramidLevel outputs), `referenceValidity/movingValidity` -> `referenceMask/movingMask`, `beforeValidity/afterValidity` -> `beforeMask/afterMask`, ChangeDetection `validity:` -> `observationMask:`, RegionGraph `status: view` -> `status: {iterationLimitReached: view}`. I did not search them.
- `modules/lenses/src/gpu-raster/computation-pack/operators/algebra.ts` names `validity`; not for these classes.
- CPU twin session (RegionGraph, MedoidComposite, GeometricMedianComposite): use `output.status.iterationLimitReached` and `mask`.

## 5. Roadmap
- STD-04: remaining adds "region graph done 2026-11-11 (`output.status.iterationLimitReached`)"; registration, temporal, unmixing, seasonal break, remapping and time-series per-cell codes stay documented per-row words.
- STD-07: done for texture, registration, remapping, temporal, unmixing, seasonal break, 17 time-series classes, 9 embedding classes, embedding change and change detection (`observationMask`). Remaining: published band inputs for these classes; arithmetic, cell statistics, overlay; `outputValidity`, `outputMask`, `seedMask`; the remaining embedding/time-series classes I did not list (none known).

## 6. Noticed
- `docs/api/gpu-raster/operations-analysis.md` line 78 still says the validity output is `validity`: consistent now. Line 1338 edited.
- The first run of `lens-twins.node.spec.ts` inside the full node project fails by timeout under load, passes alone.
