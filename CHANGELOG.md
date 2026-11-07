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
- Documentation moved from `docs/api-reference/experimental` to `docs/api`. Release notes from the
  luma.gl era are in `dev-docs/archive/luma-era-release-notes.md`.
