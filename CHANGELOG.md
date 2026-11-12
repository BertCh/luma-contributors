# Changelog

## Unreleased (0.1.0-alpha.0)

First standalone version, split on 2026-11-07 from the luma.gl `map-contributors` branch.

- New packages: `@luma-contributors/analysis` (published), `@luma-contributors/lenses` and
  `@luma-contributors/kepler` (private until dataset licensing is resolved).
- Runs on published `@luma.gl/*@10.0.0-alpha.13` through public entry points only. No luma.gl code is
  copied or forked. See `dev-docs/upstream-first-plan.md`.
- Import paths moved from `@luma.gl/graph/*` analysis subpaths to `@luma-contributors/analysis/*`.
  `gpu-raster` analysis became `raster-analysis`, `gpu-dataframe` analysis became
  `dataframe-analysis`, `gpu-pointcloud` analysis became `pointcloud-analysis` and `geospatial` became
  `geospatial-analysis`. Base primitives such as `GPUSort`, `GPUScan`, `GPUGridIndex` and
  `GPUCrossfilter` now come from `@luma.gl/graph`.
- Capabilities dropped or regressed by depending on published luma.gl are listed in
  `dev-docs/upstream-workarounds.md`.
- Restored sparse resistance connectivity (`addSparseResistanceConnectivityRecipe`, EC03) on public
  luma.gl APIs: up to 1,048,576 vertices and 2,097,152 edges, Jacobi-preconditioned,
  Circuitscape-validated; adds node current maps.
- `yarn typecheck-tests` (`scripts/typecheck-tests.mjs`) typechecks tests and source with the same
  exclusions as vitest: it generates `.typecheck/tsconfig.tests.json` without the 271 kepler specs listed
  in `test/kepler-compute-example-specs.json`, `examples/deck/kepler-compute`,
  `host-wkb-selection.spec.ts` and two trip fixtures. It reports no errors.
- CI (`.github/workflows/test.yml`) also runs `yarn check-upstream-first`, `yarn typecheck-tests`, the
  showcase and explorer typechecks, and a `website` job (typecheck, catalog check, build).
- Kernel-audit cases for the 12 point-cloud and datum-grid contributors
  (`kernel-audit/cases-pointcloud-and-datum.ts`); only the four chunked spatial-weights contributors are
  still skipped.
- `GPUNetworkCentrality` exposes `estimatedCommandNodeCount` and its constructor rejects configurations
  above 32,768 command nodes instead of building a very large unrolled graph.
- Internal: the masked weak-components kernel moved from `gpu-network` to
  `analysis-core/kernels/masked-weak-components.ts`, shared by network statistics, network analytics
  columns and sparse resistance connectivity. No public API change.
- Math correctness review (2026-11-10 and 2026-11-11): fixes in the regression family, coverage
  validity, ring assembly, polygon triangulation, the Diggle test, the COLD monitor and others, each
  pinned against a reference library. Every number a caller can see change is listed in
  `dev-docs/research/math-correctness-review-2026-11-10/behaviour-changes.md`; the limits left are in
  `dev-docs/roadmap.md` section 12.7.
- Breaking, from the contract alignment (`dev-docs/roadmap.md` sections 11.6 and 11.8): the deprecated
  aliases are removed; `parameters` is required on walking cost and fractional coverage; there is one
  variable-length geometry output layout; the raster family follows the `mask` rule (about 40 classes
  renamed props, results unchanged); point-cloud rasters default to `'south'` rows; the k-means
  contributor is `GPUSpatialKMeans`.
- CPU twins and mixed CPU/GPU graphs: 125 contributors have an operation form
  (`GPU<Name>Operation`) for a published luma.gl `GPUGraph`, a float64 CPU twin
  (`CPU<Name>Operation`) over the same named values, and a `compute<Name>OnCPU` function usable
  without a graph. `GPUContributorOperation` (`analysis-core`) lowers a contributor into a `GPUGraph`.
  Rules are in section 14 of `dev-docs/contributor-contract.md`.
- luma.gl usage review (2026-11-11 and 2026-11-12, `dev-docs/roadmap.md` section 16): 21 hand-written
  gather kernels, a flag total, a prefix sum and a high/low split now call the published
  `GPUUint32Gather`, `GPUGather`, `GPUFlagOffsets`, `GPUScan` and `fp64ify`, with equal results.
  `createWGSLKernelNode` takes `modules` and `defines` and assembles them with the published
  `WGSLShaderAssembler` (fp64 modules get the integer variant). 22 raster and terrain contributors
  declare the published `GPURasterHaloStage` contract.
- Changed values, from that review: `requiredHalo` is 2 for `GPUIsobands` when edges are produced
  (0 for classes only), at least 2 for `GPUTerrainDerivatives`, `GPUReliefShading` and the 3x3
  methods of `GPUTerrainCurvature` in the one-sided border mode, and the scaled support for
  `GPUTextureShading` with decimated levels. The former values did not cover what the kernels read.
  `GPUCubeLayoutTranspose` copies raw words, so NaN payloads survive.
- Website (Gridlight): 532 catalog entries, each with a live picture, and thirteen how-it-works
  pages. It can be built for a sub-path (`SITE_BASE_PATH`) and `yarn website-deploy` publishes it to
  GitHub Pages.
- Documentation moved from `docs/api-reference/experimental` to `docs/api`. Release notes from the
  luma.gl era are in `dev-docs/archive/luma-era-release-notes.md`.
