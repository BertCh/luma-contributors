# Changelog

## Unreleased (0.1.0-alpha.0)

First standalone version, split on 2026-11-07 from the luma.gl `map-contributors` branch.

- New packages: `@luma-contributors/analysis` (published), `@luma-contributors/lenses` and
  `@luma-contributors/kepler` (private until dataset licensing is resolved).
- Runs on published `@luma.gl/*@10.0.0-alpha.10` through public entry points only. No luma.gl code is
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
- Documentation moved from `docs/api-reference/experimental` to `docs/api`. Release notes from the
  luma.gl era are in `dev-docs/archive/luma-era-release-notes.md`.
