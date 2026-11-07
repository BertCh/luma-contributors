# Evidence

Small measurement artifacts (about 1.2 MB in total) copied from the luma.gl `map-contributors` branch (`dev-docs/perf/`) because LIB documents cite them. Each was at most 1 MB. They are immutable captures; do not edit. The rest of `dev-docs/perf/` (5.2 GB of PCG, shared-SpMV, allocator and residual solver work) stays in luma.gl.

Another session owns the citing files below; the citing link must be updated from `dev-docs/perf/<name>` to `dev-docs/research/evidence/<name>` (paths relative to the citing file).

| File | Size | Cited by |
| --- | ---: | --- |
| geospatial-coverage-tranche-one-2026-10-23.json | 26 KB | modules/analysis/test/gpu-raster/remote-sensing/raster-performance-README.md |
| raster-coverage-2026-10-24.json | 109 KB | modules/analysis/test/gpu-raster/remote-sensing/raster-performance-README.md |
| interpolation-coverage-2026-10-23.json | 47 KB | modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/interpolation-performance-README.md |
| interpolation-advanced-2026-10-23.json | 78 KB XX-validation-2026-10-23.json | 42 KB | docs/api-reference/experimental/gpu-spatial-analysis/spatial-validation-advanced.md |
| pointcloud-coverage-2026-10-23.json | 5 KB | docs/api-reference/experimental/pointcloud-analysis.md |
| pointcloud-advanced-2026-10-23.json | 6 KB | docs/api-reference/experimental/pointcloud-analysis.md |
| urban-ecology-coverage-2026-10-23.json | 6 KB | examples/deck/spatial-analysis-showcase/dist/assets/landscape-ecology-*.js (build output only; no source citation found) |
| urban-ecology-complete-2026-10-24.json | 9 KB | same as above |
| point-process-complete-2026-10-23.json | 6 KB | examples/deck/spatial-analysis-showcase/dist/assets/point-process-*.js (build output only; no source citation found) |
| kepler-resident-2026-10-29/100k-smoke.json | 18 KB | examples/deck/kepler-compute/scripts/resident-performance.README.md (line 69) |
| kepler-resident-2026-10-29/1m-smoke.json | 22 KB | examples/deck/kepler-compute/scripts/resident-performance.README.md (line 87) |
| kepler-resident-2026-10-29/1m-diagnostics.json | 42 KB | examples/deck/kepler-compute/scripts/resident-performance.README.md (line 114) |
| kepler-resident-2026-10-29/1m-diagnostics.cpuprofile | 764 KB | examples/deck/kepler-compute/scripts/resident-performance.README.md (line 114) |
| kepler-resident-2026-10-29/1m-coalesced-create-or-update.json | 23 KB | examples/deck/kepler-compute/scripts/resident-performance.README.md (line 135) |
| kepler-resident-2026-10-29/1m-coalesced-value-only.json | 23 KB | examples/deck/kepler-compute/scripts/resident-performance.README.md (line 135) |

Not copied: `dev-docs/perf/variogram-baseline-session-one` referenced by interpolation-performance-README.md line 9 is a `--output` directory for a command, not an existing artifact.
