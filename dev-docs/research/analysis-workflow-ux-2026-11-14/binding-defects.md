# Catalog binding defects found during the AX pass (2026-11-14)

Found by the AX implementors while describing each catalog binding, and confirmed by independent
reviewers with probes (node runs of the CPU engines in a staging copy of the working tree on
`fbb78ec5` plus uncommitted changes). None is fixed by the AX pass: the bindings belong to
`modules/kepler/src/analysis`, which another session was porting at the time. Where the AX layer
works around a defect, the last column says how. Line numbers are from the staging copy and may
have moved.

"Confirmed" means a reviewer reproduced it in a probe. "Reported" means an implementor observed it
and no reviewer re-ran it.

## Status on 2026-11-15

The session that owns the bindings reports every row below fixed, with binding ids unchanged, plus
eight further problems found by its own independent review (kriging scale invariance, local Geary
order dependence, an explicit Knox threshold of 0, the S2 clamp, density-clustering bucket
deduplication and boundary tolerance, multipart centroid precision). Its gates ran green in the
working tree (`yarn test-node`: 16,958 passed, 0 failed). The AX pass did not re-run the probes
below against the fixed bindings; it re-ran the layer's descriptor and presentation specs
(11 files, 1,273 tests). The tables are kept as the record of what was found.

Defaults and behaviour that changed with the fixes: local Moran defaults to Benjamini-Hochberg and
999 permutations; the hot spot lattice defaults to Benjamini-Hochberg (measured by that session:
2 of 220 pattern-free runs flagged, 0.91 percent, against 39 of 220 before); kriging fits its
variogram; GWR derives its bandwidth and throws when more than 10 percent of local fits are
singular; Knox derives thresholds when none is given; Moran's I and Geary's C both use the
randomisation variance; `similar-locations` needs `referenceIndex`; `empirical-bayes-rates` rejects
cases above population; `areal-interpolation` needs `variableType`; the network edge cap is 50,000;
`network-structure`, `network-line-graph` and `network-snap-points` lost their cost and one-way
parameters; `network-k-function` rejects property costs; a point in overlapping polygons counts in
every containing polygon and the summary carries `multipleMatchCount`; a Gi* island is `not-tested`.

Not fixed: `recipe-zonal-distribution` still gives a point in two overlapping polygons to the first
(`computeDistributionCpu` uses `assignRecipePoints`). It was not in the tables below; its descriptor
keeps the note and the disjoint-coverage warning.

AX workarounds, the last column of the tables:

- Removed with the fixes: the blocking missing-cost requirement (now a warning), the weight
  requirements on network accessibility, the length-mode preset on network K, the inert street
  roles, the island exclusion in the Gi* report, the Benjamini-Hochberg preset and the false-positive
  note on the hot spot lattice, the disjoint-coverage warnings on the two points-in-polygons
  bindings, and the notes describing each defect. The count sentence now says "point memberships"
  when `multipleMatchCount` is above 0.
- Left in place because they are correct and change no result that was verified: canonical row
  ordering and predictor standardisation in the fused bindings, the profile's own anchors, the
  local-frame hull area in the fused clusters binding, and the 1 percent margin on derived radii and
  distance bands. Roadmap AX-87.

## Wrong results

| Binding or function | Defect | Evidence | Status | AX workaround |
| --- | --- | --- | --- | --- |
| Network cost reading, `catalog-network.ts` about line 178 | A cost of `null` or `''` is read with `Number(...)` and becomes 0, so the arc is free; no diagnostic. An absent key removes the arc instead | Shortest path in property mode: `null` cost gives `total_cost = 0` | Confirmed | The cost role's missing-value requirement blocks |
| `hotspot-gi-star`, `catalog-platform3-b-statistics.ts` about lines 560 to 575 | A feature with no neighbour is classified from its own value: an island with value 60 got `hot-99`, z 6.88. Local Moran marks such rows not tested | Two reviewers, 3 islands in 147 rows: one hot, one cold, one not significant | Confirmed | The report excludes rows with a neighbour count of 0 from the counts |
| `fitOrdinaryLeastSquares`, `catalog-platform4-regression.ts` about line 140 | The pivot test is `1e-12 * scale` on the raw centred Gram matrix, so it depends on units: income (sd 15,000) with a share (sd 0.01), correlation 0.10, is rejected as collinear. NumPy fits it | Reviewer probe | Confirmed | The fused regression binding standardises predictors before the fit. `regression-ols` is still affected |
| `getFeatureAnchor`, `catalog-platform3-b-common.ts` | Holes are ignored (square with a hole: 5 where Shapely gives 5.571); multipolygon parts of opposite ring orientation cancel (anchor outside both parts); below an absolute area of 1e-12 it falls back to the vertex mean, which is wrong for small polygons in degrees | Compared with Shapely | Confirmed | The dataset profile computes its own anchors |
| `network-accessibility` | A missing or non-numeric weight counts as reachable with weight 0; a negative weight subtracts. No diagnostic | Weights 5, missing, `"abc"`, -4: `reachable_count = 4`, `accessibility = 1` | Confirmed | Requirements and a note |
| `recipe-points-in-polygons-choropleth`, `raster-join-points-in-polygons` (CPU) | A point inside two overlapping polygons is given to the first; the later polygon silently gets 0 | `value = [2, 0]` | Confirmed | A disjoint-coverage warning |
| Geometry bindings run on projected data with the default `EPSG:4326` | `geometry-measures` area 3.7e13; `geometry-shape-descriptors` all null; `geometry-label-point` clearance null; `voronoi-diagram` 0 rows; `nearest-feature-join` null distance; `network-shortest-path` null cost | Planar metres at (390000, 5820000) | Confirmed | The defaults module sets the coordinate system from the profile; a declared-coordinates warning |
| Density clustering at exactly the search distance | Points exactly 1.0 apart with a search distance of 1.0 come out as noise: the local-frame shift makes the distance 1.0000000000000002. scikit-learn finds two clusters. The binding documents "distance at most the search distance" | Reviewer probe | Confirmed | The derived radius carries a 1 percent margin and is rounded up |
| `cell-compare` with `sum` or `mean` and no value column | Gives zeros with no error | Implementor run | Reported | The value role is required for those statistics |
| `cell-aggregation` with a value column | Rows without a value are dropped from `count` too | count 2 of 4 points | Confirmed | A note |

### Added by the second review round

| Binding or function | Defect | Evidence | Status | AX workaround |
| --- | --- | --- | --- | --- |
| `network-k-function`, `catalog-platform3-a-network.ts` about line 956 | K is computed from physical edge length and then compared with `2 * radius`, where the radius is in property-cost units. Rescaling costs and radii together flips the apparent clustering: K 666.7 against 1,200 expected at cost 1,000 and radius 600; against 12 expected at cost 10 and radius 6 | Reviewer probe on one 1,000-unit line with three points | Confirmed | The descriptor presets length mode and says property costs are unsupported; the binding still accepts them |
| `measureRingSignedArea`, `catalog-geojson.ts` about lines 144 to 146 | Uses the raw longitude difference, so a ring across the antimeridian measures 1,799 times too large; the planar shoelace sum loses all precision at large coordinate offsets (area 38 becomes 0 at an offset of 1e9) | Compared with pyproj and Shapely | Confirmed | The fused clusters binding measures hulls in local-frame coordinates |
| Neighbour selection in `catalog-platform3-b-statistics.ts` about line 156 and kriging in `catalog-platform3-b-grids.ts` about line 559 | Equal-distance ties are broken by input index, so reversing the rows changes k-nearest weights and local kriging neighbourhoods: a residual Moran pseudo p-value moved from 0.121 to 0.162 and a kriging prediction from -0.621 to -0.697 on 25 grid locations | Reviewer probe | Confirmed | The fused bindings order rows canonically before calling these functions. The catalog's own bindings are still order-dependent |
| `recipe-hot-spot-lattice` | With Benjamini-Hochberg on, pattern-free points still produced at least one flagged cell at the 90 percent class in 39 of 220 runs (17.7 percent; 95 percent interval 12.9 to 23.4): the count test's p-values are approximate | 220 seeded null datasets | Confirmed | A note on the descriptor states the measurement; the default cell follows point density |

## Declared limits that the code does not honour

| Binding | Declared | Actual | Status |
| --- | --- | --- | --- |
| `network-accessibility`, `network-subgraph-filter`, `network-coarsening`, `network-flow-aggregation`, `network-alternative-routes`, `network-map-matching`, `network-k-function` | 20,000 lines, 200,000 vertices; declared behaviour mentions 50,000 edges | `MAXIMUM_ROUTING_EDGES = 20000` (`catalog-platform3-a-network.ts:44`): a grid of 210 lines with 21,840 edges throws "Edge capacity exceeded". `splitAtSharedVertices` is forced on, so that parameter does nothing | Confirmed |
| `network-centrality` | 5,000 rows, 50,000 vertices | 1,500 nodes (`catalog-platform4-geometry-gpu.ts:42`, `catalog-extra-bindings.ts:107`) | Confirmed |
| `density-clustering` (CPU engine) | 20,000 points | "Work capacity exceeded" at 15,000 | Confirmed |
| `polygon-triangulation` | 50,000 vertices | A part above 1,500 vertices returns 0 rows with `part-too-large` | Confirmed |
| `grid-generator`, random points, `line-chunk`, `geometry-line-segmentize` | Input limits only | Output caps (20,000 cells, 20,000 points, 50,000 chunks, 500,000 vertices) throw after the run | Reported |
| `curated`-independent: `kriging-grid` work cap | Named cell cap | 1,501 points by about 40,000 cells throws a generic `KeplerAllocationError: Work capacity exceeded` without the binding, the limit or the remedy (`catalog-platform3-b-grids.ts:138`) | Confirmed |

## Parameters that do nothing, or defaults that mislead

| Binding | Defect | Status |
| --- | --- | --- |
| `network-structure`, `network-line-graph`, `network-snap-points` | The one-way flag and the cost columns change nothing | Confirmed |
| All network bindings | A cost column is read only when `costMode` is `property`; with a column named and the mode left at `length` it is ignored silently | Confirmed (by design; undocumented to the user) |
| `raster-join-points-in-polygons` | The CPU engine ignores `cellSize` (exact join); only the GPU engine rasterises | Confirmed |
| `recipe-hot-spot-lattice` | `multipleTesting` defaults to `none`: uniform random points gave 9.6 percent of cells flagged; with Benjamini-Hochberg none | Confirmed |
| `local-moran-statistics` | 99 permutations by default and no correction parameter; differs from the host's Local Moran panel (binary weights, corrections) without saying so | Confirmed |
| `kriging-grid` | Defaults nugget 0, sill 1, no fitting: white noise of variance 100 gives a confident surface with a "variance" of 0.002 to 0.5 | Confirmed |
| `regression-gwr` | The constant bandwidth 1,000 made 83 of 100 local fits singular on points over 20 km; collinear predictors run with every fit singular and no error | Confirmed |
| `knox-test` | Thresholds 1,000 and 1 are constants in the data's units: clustered events with time in epoch milliseconds give p = 1 | Confirmed |
| `areal-interpolation` | `variableType` defaults to extensive whatever the column measures: a rate is summed | Confirmed |
| `hierarchical-density-clustering` | A single flat cut of the hierarchy, not HDBSCAN's stability selection; the name overclaims | Reported |
| `global-spatial-statistics` | Moran's I uses the randomisation variance and Geary's C the normality variance | Reported |
| `similar-locations` | The reference silently defaults to row 0 | Confirmed |
| `empirical-bayes-rates` | Cases above population are accepted (rate 5.0) | Confirmed |
| `buffer-selection` | The output spec declares geometry `none` but the table carries a geometry column | Reported |
| `s2-cap-covering` | `maximumCells` accepts up to 6,144 but is capped at `6 * 4^level` | Reported |
| `geometry-polygonize` | `snapTolerance` has unit `crs` but the CPU run applies no metric frame | Reported |
| `geometry-minimum-clearance` | Reported to treat an open line as a ring. Not reproduced on the CPU engine: 39 cases match Shapely to 1e-6. If real, it is in the GPU path | Refuted on CPU |
