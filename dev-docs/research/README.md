# Research

Research reports and measurement evidence carried over from the luma.gl `map-contributors` branch on 2026-11-07. Paths inside these documents (`modules/graph/src/...`, `@luma.gl/graph/...`, `../../perf/...`) refer to that branch unless a directory README says otherwise. Superseded planning documents are in [../archive](../archive/README.md); current plans are in [../roadmap.md](../roadmap.md).

| Directory | What | Status | Size |
| --- | --- | --- | --- |
| [kepler-sandbox-2026-10-28](./kepler-sandbox-2026-10-28/README.md) | Independent review of the Kepler sandbox: capability matrix, verification log, source manifest | current | 156 KB |
| [gpu-geospatial-implementations-2026-10-23](./gpu-geospatial-implementations-2026-10-23/README.md) | Primary-source survey of reference implementations by domain (geometry, network/movement, raster, statistics, terrain) with licensing findings | current (basis of the upstream-first policy) | 116 KB |
| [kepler-sql-2026-10-25](./kepler-sql-2026-10-25/query-ledger.md) | SQL query ledger, DuckDB reference script and results for the GPU SQL subset used by Kepler | historical | 20 KB |
| [gpu-geospatial-performance-2026-10-23](./gpu-geospatial-performance-2026-10-23/README.md) | Performance research: sources, algorithm practices, decision brief, evidence audit, next decisions, plus [integration-notes.md](./gpu-geospatial-performance-2026-10-23/integration-notes.md) condensing the analysis integration reports | historical (analysis-level subset) | 224 KB |
| [showcase-overhaul-2026-10-25](./showcase-overhaul-2026-10-25/README.md) | Narrative research, acceptance ledgers and reference oracles for the spatial-analysis showcase; read by `examples/deck/spatial-analysis-showcase/scripts/check-*.mjs` | historical, oracle JSON still read by scripts | 9.3 MB |
| [evidence](./evidence/README.md) | Small measurement JSON cited by LIB READMEs | historical | 1.2 MB |

## What was left behind

- **luma.gl solver work.** `dev-docs/perf/` (5.2 GB, about 12,500 files) and about 80 reports in `gpu-geospatial-performance-2026-10-23/` cover PCG, shared SpMV, allocator, Jacobi, residual and neighborhood-cooperative experiments. This is luma.gl graph infrastructure, not analysis code, and stays in luma.gl (at git ref `6ca1f10bc` and on disk in the branch worktree). Only the analysis-level documents and the small JSON cited by LIB READMEs were copied.
- **Large blobs.** Showcase JSON snapshots of 1 MB or more (about 106 MB), source-audit tarballs, Python/Node reference scripts and bytecode are listed with sha256 in [showcase-overhaul-2026-10-25/MANIFEST.md](./showcase-overhaul-2026-10-25/MANIFEST.md). They remain on disk in the branch worktree and should be published as a release asset (for example `evidence-2026-10`) rather than committed. Rule: nothing over 1 MB, no `.pyc`, `.gz` or `.tar` in git.
- **Not research.** Luma-owned roadmaps, benchmarks, RFCs and the whole-tree GPU review stay in luma.gl.
