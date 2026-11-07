# Upstream-first plan

Goal: this repository contains only geospatial and analysis code. Everything else comes from
published `@luma.gl/*` packages through their public entry points. We accept more verbose code and
upstream design decisions over carrying parallel, forked or divergent code.

## Rules

1. Import luma.gl only through published public entry points. Do not use `_internal`, deep `dist`
   paths, copied luma.gl source or a forked graph layer.
2. When luma.gl lacks something, try these in order:
   1. Use the public API, even if it is more verbose.
   2. Adapt to luma.gl's design.
   3. Drop the feature.
   4. Write a small helper named for the analysis domain, not a luma.gl replacement.
3. Improvements to luma.gl go to luma.gl as ordinary pull requests. This repository never waits on
   them and never carries them. When a release ships them, delete the local workaround.
4. A check fails the build on any non-public `@luma.gl` import and on any file under
   `modules/graph/src` that is not in an analysis-owned directory.

## Measured gap

The experiment is in `scripts/split/unfork-experiment.mjs`. It deletes all 422 forked luma.gl
files, points every import at npm `@luma.gl/graph@10.0.0-alpha.10`, and typechecks the sources.
That gives 704 errors with these root causes:

| Bucket | Errors | Cause | Resolution |
| --- | --- | --- | --- |
| A | 401 | Analysis symbols that were exported only through luma.gl barrels (`gpu-dataframe/index.ts`, `gpu-raster/index.ts`), such as `GPUClassBreaks`, `GPURipley` and `makeGPURasterBand` | Mechanical: move them into analysis-owned directories with their own entry points |
| C | 116 | Computation-record hooks on `GPUCommandGraph` (`withRecordScope`, `recording`, `describe`, node `annotation`) | Design decision 1 |
| B | 63 uses, 35 symbols | Non-public luma.gl helpers (see `unfork-missing-symbols.json`): `getSpatialOperationNodes`, `GPUSegmentedReduction`, bounded-dispatch helpers, raster validation utils, command-graph compiler internals | Per symbol: public alternative, inline in the analysis domain, or drop |
| G | 23 | Primitive API changes: `GraphDataView` vs `GPUVector`, `GPUGridIndex` `boundsBuffer`, `GPUCompaction` `strategy`, point-in-polygon `queryPositions` | Call the published signatures |
| D | 16 | Crossfilter extensions: external masks, `invalidate`, `setExternalActive` | kepler builds selection masks on the public crossfilter API or in its own kernel |
| E | 8 | Raster metadata: `verticalDatum`, `makeGPURasterBand` | Keep vertical-datum metadata beside the band in terrain code |
| H, Z | 75 | Cascades: implicit `any`, `unknown` | Disappear once A to G are fixed |

The three core shims also go:
- Replace `waitForSubmittedWork` with `device.createFence()`, which is public.
- Call `setBindings` without cache keys, accepting the loss of bind-group reuse.
- Detect out-of-memory errors by `error.name` or `GPUOutOfMemoryError`, inline in kepler.

## Keep (analysis code)

- `gpu-spatial-analysis`, `gpu-network` and `gpu-terrain`.
- `analysis-core` domain contracts.
- The analytic subdirectories of `gpu-dataframe`:
  - classification
  - profile
  - pair statistics
  - temporal reduction
  - time windows
  - calendar buckets
  - composite indicators
  - group statistics
  - key join
  - standardization
- The analytic subdirectories of `gpu-raster`:
  - algebra
  - cost distance
  - distance field
  - isolines
  - patches
  - sampling
  - zonal statistics
  - remote sensing
  - flow textures
  - polygon rasterization
- The `gpu-pointcloud` analytics: ground, SMRF, ICP, features and change.
- The `gpu-project` datum grid transform.

## Drop (luma.gl extensions that are not analysis)

Consumers move to the published equivalents.

- **`gpu-core` additions:**
  - Onesweep sort and look-back scan (use `GPUSort` and `GPUScan` defaults).
  - `gpu-kernel-node` and `utils/wgsl-kernel-nodes` (write nodes with the published
    `GPUCommandGraph` node API).
  - `gpu-double-single`, the describe and record helpers, and the benchmarks.
- **`gpu-graph-algorithms` additions:** strongly connected components and the iteration gate.
  Network analysis uses the published algorithms, or an SCC written in `gpu-network` if it is
  essential.
- **Other non-analysis additions:**
  - `gpu-simple-graph` kernel dispatch
  - `gpu-physics` rigid-body dispatch
  - `gpu-tables` residency arena and tile LOD selection
  - the `arrow/gpu` analytics adapters and batch stream

  Each is reviewed for whether an analysis consumer needs it. If one does, it is rewritten on
  public APIs inside that consumer.
- **`utils/` infrastructure:** `gpu-contributor-utils` and `gpu-contributor-types`. The WGSL math
  helpers (exact predicates, Philox, fixed-point accumulation) stay only where analysis kernels use
  them, under an analysis directory.

## Decisions needed

1. **Computation records and lenses.** This is the largest coupling: 116 errors, plus most of the
   lens errors. Options:
   - Spike a recorder built from public `GPUCommandGraph` data (plan steps, encoding stats, timings).
   - Drop recording and keep lenses as static documentation.
   - Take lenses out of this library's scope.
2. **Kepler adapters.** Keep them as an integration package on public APIs, or treat them as out of
   scope like the parked `kepler-compute` example.
3. **Regressions on published primitives.** Published graph lacks the branch's fixes, so we accept
   that until luma.gl releases them:
   - bounded dispatch above about 4.2M rows
   - compensated sums
   - block compaction

   The optional upstream track sends those fixes to luma.gl as pull requests and does not maintain
   them here.

## Phases

1. **Guard rails.** Add the public-import and allowed-directory check. Rename
   `@luma-contributors/graph` to an analysis-specific name, since it no longer carries a graph.
2. **Re-home bucket A.** Move the analysis files out of luma-owned directories into analysis-owned
   ones (`dataframe-analysis/`, `raster-analysis/`, `pointcloud-analysis/`, ...) and give each an
   entry point. Mechanical, about 240 files.
3. **Delete the fork.** Delete the 422 forked files and the compat shim, and point imports at npm.
   Expect the measured 704 errors.
4. **Fix B, D, E and G.** Fix the remaining errors symbol by symbol, following the rules above.
   Record each workaround with the upstream release that would remove it.
5. **Drop the non-analysis extensions** and move their consumers to the published primitives.
6. **Resolve decisions 1 and 2.**
7. **Tests.** Delete tests of luma.gl code. Keep analysis, oracle and parity tests. Investigate any
   numerical drift from the published primitives, and accept it or fix it in the analysis code.
8. **Upgrade cadence.** Bump `@luma.gl/*` with each alpha. Delete workarounds the release makes
   unnecessary.
