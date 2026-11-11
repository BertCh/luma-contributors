# Parallel completion pass (2026-11-10 and 2026-11-11)

Reports written by the agents of [roadmap](../../roadmap.md) section 11, one per stream, copied here unedited because the roadmap cites their inventories and wave 2 lists. Each report has the same headings: done, verified (with the exact commands and counts), not done, wave 2 requests, roadmap text, noticed. Paths that point at a session scratch folder no longer exist; everything the roadmap cites is in this folder.

The reports state what each agent ran itself. The gates run on the combined tree afterwards are recorded in roadmap section 11.5 (wave 1) and section 11.7 (wave 2).

## Wave 1: additive changes

| Report | Stream | Roadmap items |
| --- | --- | --- |
| [core-wgsl.md](./core-wgsl.md) | Shared WGSL modules in `analysis-core`, with the adoption map for the rest of the library | STD-10 |
| [trajectories.md](./trajectories.md) | Trajectory and track-similarity directories | STD-03, STD-04, STD-06, STD-07 |
| [interpolation-clustering.md](./interpolation-clustering.md) | Local kriging and OPTICS through a neighbour grid; spatial context on the interpolators | GPU-05, PP04, STD-06 |
| [weights-regression-pairs.md](./weights-regression-pairs.md) | Fuzzy contiguity, island attachment, cross-variogram, marked statistics | GPU-05 |
| [geometry-output.md](./geometry-output.md) | Inventory of variable-length geometry layouts, the target layout, additive fields | STD-08 |
| [joins-zonal.md](./joins-zonal.md) | Join overflow flags and their consumers; region statistics status | GPU-04, STD-04 |
| [raster.md](./raster.md) | Per-class inventory of raster georeferencing, no-data and status; walking cost and fractional coverage parameters | STD-04, STD-06, STD-07 |
| [terrain.md](./terrain.md) | Terrain status ports; per-class table | STD-03, STD-04 |
| [pointcloud.md](./pointcloud.md) | Why the published grid index cannot replace the private one; row direction; limits | GPU-05, STD-07, AUD-02 |
| [network-urban.md](./network-urban.md) | Network snapping cap, visibility graph and viewpoint classifier through the segment BVH | GPU-05, STD-04 |
| [cells-ecology.md](./cells-ecology.md) | The `atan2(y, -0.0)` defect with the list of every other call site; the raster resistance recipe | FX-17, EC03-R |
| [records.md](./records.md) | Recording through `createWGSLKernelNode`; what a record holds before and after | LN-REC |
| [kepler.md](./kepler.md) | Classification of the 271 parked kepler specs and the 140 re-enabled | KE-TESTS |
| [sweep.md](./sweep.md) | Scattered asserts, registry gaps, stale documentation | STD-03, DOC-01, UF-06, UF-09 |
| [fixups.md](./fixups.md) | Status fields moved under `status`, spec typing fixes after the first combined gates | |
| [bump.md](./bump.md), `bump.patch` | Trial of `@luma.gl` alpha.12 in a worktree | UF-11 |
| [web-g1.md](./web-g1.md), [web-g2.md](./web-g2.md), [web-g3.md](./web-g3.md) | Inert sliders and weak gestures in the twelve existing run groups (g1 and g3 were saved by the lead from the agents' final messages) | WEB-PIC-13, WEB-PIC-14, WEB-PIC-17 |
| [web-text.md](./web-text.md), `text-wiring.diff` | The glyph-atlas text layer and the diff that wires it into the run framework | WEB-INT-16 |

## Wave 1: reviews and fixes

| Report | What |
| --- | --- |
| [review-indexes.md](./review-indexes.md), [review-joins-geometry-raster.md](./review-joins-geometry-raster.md), [review-records-tests-contract.md](./review-records-tests-contract.md), [review-website.md](./review-website.md) | Four independent reviews of the wave 1 diff, each with its findings and the evidence for them |
| [fix-indexes.md](./fix-indexes.md), [fix-core.md](./fix-core.md), [fix-contract.md](./fix-contract.md), [fix-atan2.md](./fix-atan2.md), [fix-website.md](./fix-website.md) | What was changed for each review finding, and what was left |
| [label-runs.md](./label-runs.md) | The 18 website runs that gained text labels |

## Wave 2: breaking changes (2026-11-11)

Same headings as wave 1, with "Website and examples work list" in place of "Wave 2 requests".

| Report | Stream | Roadmap items |
| --- | --- | --- |
| [w2-raster.md](./w2-raster.md) | Raster aliases removed; `parameters` required on walking cost and fractional coverage | STD-04, STD-06 |
| [w2-trajectory-pointcloud.md](./w2-trajectory-pointcloud.md) | Trajectory aliases removed; point-cloud `rowDirection` default `'south'` | STD-06, STD-07 |
| [w2-geometry.md](./w2-geometry.md), [w2-ring-assembly.md](./w2-ring-assembly.md) | The one variable-length geometry layout (`count`, `vertexCount`, `ringCount`, source identifiers) and the padding-row fixes | STD-08, FX-18 |
| [w2-kmeans.md](./w2-kmeans.md) | `GPUSpatialKMeans` | DUP-02 |
| [w2-adoption.md](./w2-adoption.md) | Adoption of the shared WGSL modules across the library | STD-10 |
| [w2-core-surface.md](./w2-core-surface.md) | `validateSinglePackedView`; contract names removed from the domain barrels | STD-09 |
| [w2-lenses-records.md](./w2-lenses-records.md) | Lens records regenerated through the kernel recorder | LN-REC |
| [w2-website.md](./w2-website.md), [w2-resistance-run.md](./w2-resistance-run.md) | Website consumers of the renames; the raster resistance recipe run | EC03-R |
| [w2-examples.md](./w2-examples.md), [w2-snippets.md](./w2-snippets.md) | Example consumers of the renames; the snippet import check and the 144 import names it found wrong | UF-12 |
| [review-kernels.md](./review-kernels.md), [review-api.md](./review-api.md), [review-consumers.md](./review-consumers.md) | Three independent reviews of the wave 2 diff | |
| [w2-fix-kernels.md](./w2-fix-kernels.md), [w2-fix-texts.md](./w2-fix-texts.md) | What was changed for each wave 2 review finding | |

## Wave 3: remaining pictures, indexes, padding rows and raster names (2026-11-11)

Same headings as wave 2. Paths in these reports that point at `website/.reports/` name screenshot folders that are not kept.

| Report | Stream | Roadmap items |
| --- | --- | --- |
| [w3-web-regression.md](./w3-web-regression.md), [w3-web-statistics.md](./w3-web-statistics.md), [w3-web-geometry-construction.md](./w3-web-geometry-construction.md), [w3-web-processing-trajectories.md](./w3-web-processing-trajectories.md), [w3-web-network-terrain.md](./w3-web-network-terrain.md), [w3-web-pointcloud-raster.md](./w3-web-pointcloud-raster.md) | The weakest card pictures of the independent re-score and the remaining labels, by run group; each lists the posters to regenerate | WEB-PIC-12, WEB-INT-16 |
| [w3-lib-padding.md](./w3-lib-padding.md) | Padding rows: line clip's polygon side, Delaunay, concave hull, the join index builders, with a poisoned-padding spec per site | FX-18 |
| [w3-lib-indexes.md](./w3-lib-indexes.md) | `GPUNeighborSearch` cross-join queries outside its bounds; the island-attachment grid path | GPU-05 |
| [w3-lib-raster-imagery.md](./w3-lib-raster-imagery.md), [w3-lib-raster-algebra.md](./w3-lib-raster-algebra.md) | Status ports, the `mask` and `validity` names and `cellSizeMode` in the raster family, with a rename table and a per-class state table each | STD-04, STD-06, STD-07 |

