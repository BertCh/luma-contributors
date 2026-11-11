# Stream pointcloud report

## 1. Done
- GPU-05 (decision: not migrated). Published `GPUGridIndex` (`@luma.gl/graph/gpu-core`, `gpu-grid-index.d.ts`) cannot serve the five callers (neighborhoods, change, ICP, surfaces, SMRF). The missing pieces are written as a doc comment on `createPointCloudGridIndex` in `modules/analysis/src/pointcloud-analysis/pointcloud-grid-index.ts` (upstream request): (1) takes `bounds` + `gridSize`, not origin + cellSize, so float32 cell edges differ and the raster kernels' `cellIndex == index` check could not hold; (2) accepts points on the upper bound, callers drop them; (3) in-cell order unspecified (atomic scatter), callers need ascending order (existing spec pins it; the GWR family already adds a rank sort for the same reason); (4) float32x3 with 2-D raster mode ignoring Z; (5)/(6) caller-sized objectIds/count/overflow rows vs plain offsets and sorted IDs views. Private index left in place.
- STD-03: `gpu-point-buffer.ts` and `gpu-datum-grid-transform.ts` no longer import `assert`; all checks use named-limit helpers with the contributor ID prefix (`validateCapacity`, `validateViewLength`, `validateFiniteNumber`, `validateMaximumCount`, `validateRequirement`). Rejection cases in new `modules/analysis/test/geospatial-analysis/geospatial-validation.node.spec.ts` (verticesPerRing, offsets rows, parameters rows, radius, grid width/sampleOffset/origin/spacing, heightUnitMeters, tolerance, maximumIterations, vertical references, positionsLow pairing).
- AUD-02: added kernel-audit cases and pinned rejections (`cases-pointcloud-and-datum.ts`, `rejections-pointcloud.ts`) for: IDW power one above 8, change cells-per-core (101 per axis passes, 102 refused, limit 1048576), neighborhoods grid cell count above 4294967294. Existing cases already pinned neighbor capacity (neighborhoods, features, statistical filter), grid axis, ground radii and count, SMRF window and cut, ICP iterations, unindexed work (change, neighborhoods, ICP), IDW reach, sampling covariance. Gap left: no companion at exactly 4294967294 cells (its offsets exceed a storage binding, so it cannot be built).
- STD-07 row order: `PointCloudRasterGrid.rowDirection?: 'north' | 'south'`, default `'north'` (today). `'south'` puts row 0 at the maximum y (raster/terrain family convention). Implemented in `getPointCloudCellSource` and the raster mode of `createPointCloudGridIndex` (new optional 5th parameter) in `gpu-pointcloud-surfaces.ts`, `pointcloud-grid-index.ts`, `gpu-pointcloud-smrf.ts`; validated in `validatePointCloudGrid`. Documented in the type and in `docs/api/pointcloud-analysis.md`. Spec: `modules/analysis/test/gpu-pointcloud/gpu-pointcloud-row-direction.spec.ts` (Surfaces and Height north vs south mirror, corner and edge points).
- STD-07 naming: `GPUDatumGridTransform` input `validity` -> `mask` (deprecated alias `validity`, `mask` wins; resolved `contributor.mask`); `GPUPointCloudStatisticalFilter` output `mask` -> `validity` (deprecated alias `mask`; resolved `contributor.validity`). WGSL binding names unchanged. `docs/api/geospatial.md` and `pointcloud-analysis.md` updated. Ground `groundMask`, ICP `inlierMask` left (classification/inlier masks, not row filters).

## 2. Verified
- `npx vitest run --project node` geospatial-validation, kernel-audit-pointcloud-and-datum, gpu-datum-grid-transform.node, gpu-pointcloud-analysis.node, gpu-pointcloud-limits.node: 5 files, 60 tests passed (after fixing one test of mine that reused a graph ID); rerun of validation + audit: 49 passed.
- `npx vitest run --project headless` row-direction, pointcloud-grid-index, gpu-pointcloud-analysis, gpu-pointcloud-smrf: 4 files, 27 passed. Also gpu-point-buffer, gpu-pointcloud-advanced, gpu-pointcloud-change-grid, global-coordinates: 4 files, 51 passed.
- `node scripts/typecheck-status.mjs` on my paths: 0 matching errors, but the cycle may predate my last edits. `check-upstream-first`: no violations. `yarn check-contributor-contract --all`: no findings for pointcloud or geospatial paths. `biome check --write` on my files.
- Not run: SMRF with `rowDirection: 'south'` (SMRF and Ground only share the cell source and index; no spec for south there); the whole kernel-audit across families.

## 3. Not done
- GPU-05 migration (see above). The 5 callers still use the private index.

## 4. Wave 2 requests
- Flip the default `rowDirection` to `'south'` (breaking; point-cloud rasters then match the raster family). Consumers to adjust: website `dataframe-pointcloud-records`/lidar runs and `examples/deck/spatial-analysis-showcase/src/scenes/terrain/lidar-ground.*` (grep `GPUPointCloudSurfaces|Height|Ground|SMRF`), any lens that reads point-cloud rasters.
- Remove deprecated aliases: `GPUDatumGridTransform` `validity` -> `mask` (consumer: `website/src/gpu/runs/entries/dataframe-pointcloud-records/datum-grid-transform.ts:54`); `GPUPointCloudStatisticalFilter` output `mask` -> `validity` (consumer: `website/src/gpu/runs/entries/dataframe-pointcloud-records/pointcloud-statistical-filter.ts:138`, plus tests `gpu-pointcloud-analysis*.spec.ts`, `gpu-pointcloud-advanced.spec.ts`). Nothing in lenses, kepler or examples uses the renamed props. Datum specs (`global-coordinates*.spec.ts`, `gpu-datum-grid-transform.node.spec.ts`) still pass `validity`.
- Upstream luma.gl request: the GPUGridIndex items listed in the file comment (cell size, half-open domain, stable in-cell order).

## 5. Roadmap
- GPU-05: remains: upstream request (cell size, half-open domain, ascending in-cell order); migrate the 5 callers once published.
- STD-07: remains: flip the point-cloud default row direction to south (option exists); remove `validity`/`mask` aliases above; the other ~60 names and four raster no-data conventions.
- STD-03: bare asserts in geospatial-analysis done; delete from this directory's list.
- AUD-02: point-cloud limits all pinned; no companion for the 4294967294-cell total.

## 6. Noticed
- `gpu-pointcloud-neighborhoods` `validPoint`/`PointCloudGrid.dimensions` axis check message reads "must be an integer in [1, 16777216]" while the total check says "at most", fine; nothing wrong.
- Roadmap claims about point-cloud rasters feeding `GPURasterMetadata` are not implemented (point-cloud contributors still emit no metadata).
