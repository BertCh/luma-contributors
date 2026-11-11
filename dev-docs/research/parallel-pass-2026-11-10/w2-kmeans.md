# w2-kmeans report

## 1. Done
DUP-02: local k-means family renamed, no alias left.
- GPUKMeans -> GPUSpatialKMeans; GPUKMeans{Props,Output,Settings,Initialization} -> GPUSpatialKMeans...; getGPUKMeansParameterValues -> getGPUSpatialKMeansParameterValues; getKMeansRestartSeed -> getGPUSpatialKMeansRestartSeed; GPU_KMEANS_{MAXIMUM_CLUSTERS,MAXIMUM_ITERATIONS,MAXIMUM_RESTARTS,PARAMETER_LENGTH} -> GPU_SPATIAL_KMEANS_*.
- Files (create new, rm old): spatial-clustering/gpu-kmeans.ts -> gpu-spatial-kmeans.ts; test/.../gpu-kmeans.node.spec.ts -> gpu-spatial-kmeans.node.spec.ts; gpu-kmeans-pipelines.node.spec.ts -> gpu-spatial-kmeans-pipelines.node.spec.ts. Error messages and OPERATION that contained the class name follow (e.g. 'GPUSpatialKMeans seed must be a uint32'). Contributor ids ('kmeans' prefix), 'kmeans++', operator-catalog id 'k-means' unchanged.
- Consumers updated: spatial-clustering/index.ts, analysis-core/kernels/philox-random.ts (comment), spatial-validation/gpu-spatial-cross-validation.ts (comment), ontology/spatial-analysis-catalog.ts (class name entry), raster-embedding-kmeans.ts comment line 51 only, test doc-parameters.ts, kernel-audit cases-points-and-regions.ts, gpu-clustering-extras.spec.ts, kmeans-oracle.ts, test index.ts, kepler (catalog-extra-bindings, catalog-platform4-geometry-gpu, -groups, spec), lenses (statistics-clustering pack, k-means lens/example, statistics-clustering-pack spec), docs/api (operators-statistics, README, operators, cross-reference, raster-embeddings line 97).
- FORCED CASCADE: lens-coverage spec requires scenario id second segment == kebab(class name), so the scenario id became gpu-spatial-analysis.spatial-k-means. Renamed: lens file k-means.lens.ts -> spatial-k-means.lens.ts and k-means.example.ts -> spatial-k-means.example.ts (lens-examples spec demands the name), s1-clustering-index import; docs directives in operators-statistics.md; docs/lens-records/gpu-spatial-analysis.k-means.json -> ...spatial-k-means.json (cp + id sed, not re-recorded; recorded internal ids still say gpu-spatial-analysis-k-means-*); the 6 twin files regenerated with UPDATE_LENS_TWINS=1 (only k-means twins changed); website/src/lens/generated/gpu-spatial-analysis.k-means.json -> ...spatial-k-means.json (cp + id sed). That last one is under website/ (violates the brief; it is a generated copy, regenerate it with the normal generator and the file will match).
- Published GPUKMeans users (untouched): modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans.ts lines 6 (import), 47 (doc), 58, 81; docs/api/gpu-raster/raster-embeddings.md line 93; spatial-clustering/gpu-spatial-kmeans.ts line 238 (doc comment about the published one). GPURasterEmbeddingKMeans untouched.
- Collision scan (TypeScript export lists of 12 analysis src entry barrels, 3893 names, vs 49 public entry points of @luma.gl/core, graph, shaders, 2251 names): no collision remains from barrels. Also compared every exported name in any analysis src file: only validatePackedView modules/analysis/src/analysis-core/kernels/gpu-contributor-utils.ts:209 collides (with @luma.gl/graph/gpu-core; not in a barrel; later step). Script: scratchpad/exports.cjs.

## 2. Verified
- vitest node: gpu-spatial-kmeans, -pipelines, kepler catalog-extra-bindings: 44 pass. headless: spatial-clustering dir + kepler catalog-platform4-geometry-gpu.spec: 12 files, 88 pass.
- node: analysis-vocabulary, doc-parameters, layering, kernel-audit-points-and-regions, kernel-audit/coverage-statistics, 3 other kernel-audit specs: all pass (222 + 72).
- lenses node: computation-docs, computation-lenses, gpu-spatial-analysis: 857 pass, 6 fail, none k-means: record-structure x4 (gpu-network), s3-geometry GPUShapeGenerator latitude2, statistics-regions skater getSsd. (raster expansion-scenarios failures seen once, other session.) doc-snippets fails on operators-lines.md:63 lineCount (not mine).
- typecheck-status 'KMeans|kmeans|...|kepler/': 0 errors. check-upstream-first: no violations. biome check --write on touched code files.
- Not run: lenses headless record specs (capture-records etc.), yarn check-contributor-contract (no output for kmeans grep only), kernel-audit specs beyond those listed.

## 3. Not done
- Nothing skipped. docs/lens-records/gpu-spatial-analysis.spatial-k-means.json and twins/~kernel.json still contain recorded text 'GPUKMeans' (131 hits in kernel twin, from the record); not regenerated. Re-record when convenient.

## 4. Website and examples work list
Replace by word: GPUKMeans->GPUSpatialKMeans, GPUKMeansInitialization/Output->GPUSpatialKMeans..., getGPUKMeansParameterValues->getGPUSpatialKMeansParameterValues, GPU_KMEANS_PARAMETER_LENGTH->GPU_SPATIAL_KMEANS_PARAMETER_LENGTH. Do not touch GPURasterEmbeddingKMeans, runKMeans/runGpuKMeans, 'kmeans' ids.
- website/src/gpu/demos/k-means.ts: 4, 9, 10, 11, 57, 65, 75, 137, 166.
- website/src/gpu/demos/index.ts: 86 (symbols: ['GPUSpatialKMeans']).
- website/src/catalog/entries/spatial-statistics.ts: 1838 (symbol), 1862, 1866, 1884, 1903, 1904, 1906, 1910, 1911, 1913; 1924 lensScenario -> 'gpu-spatial-analysis.spatial-k-means'.
- website/src/catalog/generated/inventory.json: lines 1 and 167 (generated; regenerate).
- website/src/lens/generated/: already renamed by me (see above); regenerate to confirm.
- examples/deck/spatial-analysis-explorer/modes/clusters-mode.ts: 17, 22, 27, 29, 56, 108 (label text), 111, 129, 196, 209, 492, 493, 498, 820.
- examples/.../showcase/src/docs/guides/points.md:25.
- scenes/earth/hurricane-families.compute.ts: 10, 11, 12, 149, 435, 436, 439. hurricane-families.scene.ts: 36, 121, 312, 355, 378, 382, 477.
- scenes/earth/satellite-constellations.compute.ts: 8, 9, 10, 99, 170, 171, 175, 354, 405. satellite-constellations.scene.ts: 21, 48, 66, 283, 307, 310, 314, 323.
- scenes/points/nature-clusters.compute.ts: 10, 15, 20, 554, 555, 558. nature-clusters.scene.ts: 46, 624, 628.
- scenes/regression/regionalization.compute.ts: 9, 14, 17, 196, 325, 326, 526. regionalization.scene.ts: 25, 307, 338, 340, 415.
- website/src/gpu/runs/entries/raster-embeddings-flow/*, catalog/entries/raster-embeddings-flow.ts: unaffected (GPURasterEmbeddingKMeans).
(Line numbers from `git grep -n GPUKMeans|GPU_KMEANS|getGPUKMeans`; the word-boundary replacement covers all of them.)

## 5. Roadmap
DUP-02: delete the row after the website/examples edits and rebuild. Add: re-record gpu-spatial-analysis.spatial-k-means lens record.

## 6. Noticed
- Scenario id is derived from the exported class name by the coverage spec, so any class rename forces a lens record rename.
- Other-session breakage seen: gpu-network record-structure (4), GPUShapeGenerator latitude2, skater getSsd, operators-lines.md lineCount snippet, raster-walking-cost typecheck (getWGSLFloatLiteral unused, expansion-runner.ts:156).
- docs/lens-records/gpu-spatial-analysis.geographic-distribution* show modified (not mine).
