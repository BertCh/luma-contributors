# C. Analysis vocabulary and compositions

**Verified** means inspected source; **inferred** means interpretation of that source. This is a working-tree snapshot, not a runtime audit.

**Verified:** concurrent changes include the quantile-LISA recipe becoming `GPUQuantileLISA`; I treated that as work in progress. Evidence: read-only `git status --short`, `modules/analysis/src/gpu-spatial-analysis/recipes/index.ts:123`, `modules/analysis/src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa.ts:109`.

## Vocabulary and counts

**Verified:** the ontology declares 15 logical data kinds, independent of storage. Capability records contain domain, stage, summary, kinds, coordinate models, bounds, operator names, related capabilities, limitations and evidence. Operator records add package, tags, dimensionality, topology, semantics, determinism, status, readback, tier and numeric limits. Constructors and named ports are absent. `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:6`, `:210`, `:314`.

**Verified:** source evaluation returned **534 records: 529 operations/recipes and five helpers**. The older spatial array contains 261 records. Counts below exclude helpers; domain means the primary capability’s domain. Kinds count record memberships, so columns do not sum to 529. `modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog.ts:58`, `:101`.

Count command actually run:

```sh
node --input-type=module -e 'import {build} from "esbuild";let r=await build({entryPoints:["modules/analysis/src/gpu-spatial-analysis/ontology/index.ts"],bundle:true,write:false,format:"esm"}),m=await import("data:text/javascript;base64,"+Buffer.from(r.outputFiles[0].text).toString("base64")),o=m.queryGPUAnalysisOperators(),c=m.getGPUAnalysisCapabilityCatalog();for(let k of ["domain","inputs","outputs"])console.log(k,o.reduce((a,x)=>{for(let v of k==="domain"?[c.find(y=>y.id===x.primaryCapability).domain]:x[k])a[v]=(a[v]||0)+1;return a},{}));console.log("total",o.length);'
```

| Domain — verified | Count | Kind — verified | Input | Output |
|---|---:|---|---:|---:|
| dataframe-analysis | 32 | attributes | 182 | 282 |
| density-and-interpolation | 25 | cells | 13 | 5 |
| discrete-global-grids | 25 | lines | 33 | 53 |
| geometry | 56 | network | 52 | 28 |
| movement | 25 | pairs | 1 | 2 |
| network-analysis | 44 | parameters | 186 | 0 |
| point-cloud-analysis | 10 | points | 170 | 19 |
| proximity | 10 | polygons | 57 | 39 |
| raster-analysis | 83 | regions | 23 | 10 |
| regional-analysis | 34 | render-geometry | 1 | 8 |
| spatial-models | 36 | spatial-weights | 47 | 25 |
| spatial-statistics | 64 | summary | 4 | 212 |
| terrain-analysis | 45 | surface | 154 | 146 |
| topology | 13 | time-series | 44 | 9 |
| workflow | 27 | trajectories | 26 | 1 |
| execution | 0 | | | |

**Verified:** a TypeScript-AST source-barrel walk, executed through `node --input-type=module -e`, found complete contributor-class coverage below. Operation wrappers were counted separately and excluded from the contributor denominator.

| Entry point | Catalogued/exported contributor classes | Additional GPU operation forms |
|---|---:|---:|
| raster-analysis | 83/83 | 13 |
| gpu-terrain | 46/46 | 11 |
| gpu-network | 40/40 | 17 |
| pointcloud-analysis | 10/10 | 0 |
| dataframe-analysis | 32/32 | 7 |

The repository also declares an export-coverage test; I read it but did not run it. `modules/analysis/test/gpu-spatial-analysis/ontology/analysis-operator-catalog.node.spec.ts:41`.

**Verified:** completeness of names exceeds completeness of descriptions: **151/529 records have empty input and output arrays**. Older records default to empty arrays; newer entry-point records inherit whole-family kinds. These are discovery descriptors, not precise port contracts. `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-operators.ts:860`, `modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog.ts:32`.

## Queries and executable composition

- **Verified:** `queryGPUAnalysisOperators({input: 'points'})`, `{output: 'surface'}`, or both filter by array membership. Further filters cover package, tier, capability, dimensionality, coordinate model, semantics, evidence and text. `modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog.ts:101`.
- **Verified:** capability queries additionally expose domain/stage; “connections” resolve manually listed related capabilities. Neither API searches paths. `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-catalog.ts:1088`, `:1116`.
- **Inferred:** bounded A→B path search is possible as a discovery heuristic. It cannot certify execution: arrays omit required/optional distinctions, companion inputs, port identities, layouts and compatibility rules; empty records disappear entirely.
- **Verified:** `GPUContributorOperation` declares named graph values and output formats/lengths. Lowering resolves packed views and constructs the contributor; it does not attach ontology kinds. `modules/analysis/src/analysis-core/operations/gpu-contributor-operation.ts:37`, `:117`, `:166`.
- **Verified:** CPU twins can reuse the same props/ports. **Inferred:** no declarative twin registry exists in the analysis source searched; pairing is expressed through implementations and exports. `modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/cpu-vecchia-log-likelihood-operation.ts:55`, `modules/analysis/src/analysis-core/operations/index.ts:5`.

## Composite inventory

**Verified:** `rg -n 'export function add\w+Recipe' modules/analysis/src | wc -l` returned **41**; restricting to `gpu-spatial-analysis/recipes` returned **24**.

Names below omit `add`/`Recipe`. Ordered analytical stages are **verified**; logical-kind mappings and question labels are **inferred** from props/results, rather than copied from the incomplete catalog. Parameter buffers and status words are omitted.

Kinds: **A** attributes, **P** points, **L** lines, **G** polygons, **T** trajectories, **H** time-series, **C** cells, **W** spatial-weights, **R** regions, **S** surface, **N** network, **Q** summary.

| Recipe and source citation | Ordered stages — verified | Kinds — inferred | Question — inferred |
|---|---|---|---|
| HotSpotAnalysis — `modules/analysis/src/gpu-spatial-analysis/recipes/hot-spot-analysis-recipe.ts:527` | Optional polygon preparation→study cover→cell aggregation/join→density→weights→Gi*→optional permutation→optional classification/color | P,G or S→C,A,Q | Where are hot/cold spots? |
| RateClusterMap — `modules/analysis/src/gpu-spatial-analysis/recipes/rate-cluster-map-recipe.ts:270` | EB rates→contiguity→row standardization→local Moran→optional permutation→colors | G,A→W,A | Where do rates cluster? |
| PointsInPolygonsChoropleth — `modules/analysis/src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe.ts:192` | Optional preparation→zonal statistics, or join/group→breaks→colors | P,G,A→A | What occurs per area? |
| SpaceTimeHotSpots — `modules/analysis/src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe.ts:343` | Calendar→period index→atomic/grouped cube counts→emerging hotspots→optional color | P/C,H,W→H,A | How do hotspots evolve? |
| ClusterAndOutline — `modules/analysis/src/gpu-spatial-analysis/recipes/cluster-and-outline-recipe.ts:154` | DBSCAN→group geometry→convex/alpha hull→measures | P→R,G,A | What footprints contain clusters? |
| SpatialRegression — `modules/analysis/src/gpu-spatial-analysis/recipes/spatial-regression-recipe.ts:371` | OLS→diagnostics→specification decision→residual Moran→optional GWR | A,W,(P)→A,Q | How does X relate to Y? |
| StraightLineCatchments — `modules/analysis/src/gpu-spatial-analysis/recipes/straight-line-catchments-recipe.ts:125` | Distance/allocation field→raster zonal statistics | P,S→S,Q | What surrounds each facility? |
| ChangeOfSupport — `modules/analysis/src/gpu-spatial-analysis/recipes/change-of-support-recipe.ts:216` | Optional preparation→two rasterizations→areal weights→optional spatial lags | G,A→W,A | Transfer values between boundaries |
| FleetDwell — `modules/analysis/src/gpu-spatial-analysis/recipes/fleet-dwell-recipe.ts:270` | Stop metrics→polygon join→mask→zone grouping | T,G→P,A,Q | Where do vehicles stop? |
| FleetDwellZoneEvents — `modules/analysis/src/gpu-spatial-analysis/recipes/fleet-dwell-recipe.ts:531` | Zone events→membership adapter→zone grouping | T,G→A,Q | Where is time spent? |
| PeriodComparison — `modules/analysis/src/gpu-spatial-analysis/recipes/period-comparison-recipe.ts:152` | Two aggregations→outer comparison→breaks→colors | P/C,A→C,A | What changed between periods? |
| ExcessRisk — `modules/analysis/src/gpu-spatial-analysis/recipes/rate-mapping-recipes.ts:74` | EB reductions/excess risk→invalid tally | A→A | Where exceeds expected counts? |
| SpatialRate — `modules/analysis/src/gpu-spatial-analysis/recipes/rate-mapping-recipes.ts:126` | Neighborhood pooling→island policy/scaling | A,W→A | What are neighborhood rates? |
| ZonalDistribution — `modules/analysis/src/gpu-spatial-analysis/recipes/zonal-distribution-recipe.ts:65` | Optional point join→group distributions | P,G,A or R,A→A,Q | How do zones’ distributions differ? |
| RasterZonalDistribution — `modules/analysis/src/gpu-spatial-analysis/recipes/raster-summary-recipes.ts:54` | Band validity/calibration→zonal distributions | S,R→A,Q | What is each zone’s distribution? |
| ExposureByZone — `modules/analysis/src/gpu-spatial-analysis/recipes/raster-summary-recipes.ts:85` | Prepare bands→exposure weights→zonal sums→shares | S,R→A,Q | Who is exposed? |
| RegionalRasterChange — `modules/analysis/src/gpu-spatial-analysis/recipes/raster-summary-recipes.ts:228` | Difference→area masks→zonal summaries→coverage | S,H,R→S,A,Q | Where changed, on observed support? |
| InterpolationDiagnostics — `modules/analysis/src/gpu-spatial-analysis/recipes/interpolation-diagnostics-recipe.ts:34` | Kriging/LOO→residuals→error summaries | P,A→S,A,Q | How reliable are predictions? |
| DrainageNetwork — `modules/analysis/src/gpu-spatial-analysis/recipes/drainage-network-recipe.ts:28` | D8 flow→order→watersheds→distance→stream network→group summaries | S→S,R,N,Q | Where does water drain? |
| LocalVariability — `modules/analysis/src/gpu-spatial-analysis/recipes/local-variability-recipe.ts:56` | LOSH→local Moran→status tallies | A,W→A,Q | Where is local dispersion? |
| CellFlowTopology — `modules/analysis/src/gpu-spatial-analysis/recipes/cell-flow-topology-recipe.ts:59` | H3 edges→pair validation→cell totals→boundaries→edge flows | C,A→N,L,A | Which neighboring cells exchange flow? |
| SurfacePreparation — `modules/analysis/src/gpu-spatial-analysis/recipes/surface-preparation-recipe.ts:33` | Nodata fill→provenance→optional rank filter→unresolved tally | S→S,A,Q | Which gaps remain? |
| DasymetricRedistribution — `modules/analysis/src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe.ts:92` | Fractional coverage→eligibility→tuple weights→areal interpolation→lag→mass ledger | G/S,A,R→A,Q | Redistribute totals onto eligible support |
| ContiguityOrderCorrelogram — `modules/analysis/src/gpu-spatial-analysis/recipes/contiguity-order-correlogram-recipe.ts:111` | Per order: ring weights→standardize→global Moran→collect | W,A→Q | At which adjacency order is association? |
| BuildingTessellation — `modules/analysis/src/gpu-spatial-analysis/urban-morphology/building-tessellation.ts:83` | Boundary sampling→clipped Voronoi fragments→dissolve→optional enclosure overlay | L,G,A→G | What territory belongs to buildings? |
| StreetProfiles — `modules/analysis/src/gpu-spatial-analysis/urban-morphology/street-profiles.ts:156` | Segment checks→side rays→quality→canyon→frontage→optional facades | P,L,A→A | How enclosed is this street? |
| BuildingAdjacency — `modules/analysis/src/gpu-spatial-analysis/urban-morphology/building-adjacency.ts:56` | Shared-path evidence→lengths→pairs→neighborhood→components→alignment→optional structures | L,A,(G)→A,R | Which buildings share walls? |
| Isovists — `modules/analysis/src/gpu-spatial-analysis/urban-morphology/isovists.ts:82` | Obstacle index/classification→angles→ray casts→validity→measures | P,L,(G)→G,A | What is visible here? |
| VisibilityGraph — `modules/analysis/src/gpu-spatial-analysis/urban-morphology/visibility-graph.ts:82` | Visibility→compact pairs→edges→hop analysis→choice→normalization | P,L,(G)→N,A | Which viewpoints connect visually? |
| LandscapeMetrics — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/landscape-metrics.ts:75` | Optional reconciliation→patch→class adjacency→class→landscape summaries | S,R,A→A,Q | How fragmented is the landscape? |
| PatchProximity — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/patch-proximity.ts:59` | Optional reconciliation→core distances→horizontal/vertical transforms→patch distances→summary | S,R,A→S,A,Q | How isolated are patches? |
| ResistanceConnectivity — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/resistance-connectivity.ts:61` | Component/grounded CG solve→currents; deprecated | N,A→A,Q | What resistance separates endpoints? |
| SparseResistanceConnectivity — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance-connectivity.ts:127` | Sparse system/connectivity→PCG→true residual→vertex/edge currents | N,A→A,Q | Where does current concentrate? |
| RasterResistanceConnectivity — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/raster-resistance-connectivity.ts:270` | Resolve band→conductance edges→invalid tally→sparse solve→status | S→S,A,Q | Where are resistance corridors? |
| SegregationDecomposition — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/segregation-decomposition.ts:62` | Four dissimilarities→Shapley terms→optional randomization | A→Q | Why did segregation change? |
| SegregationCounterfactual — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual.ts:65` | Two quantile mappings→decomposition | A→A,Q | Composition or spatial redistribution? |
| PopulationSupportAlignment — `modules/analysis/src/gpu-spatial-analysis/landscape-ecology/segregation-counterfactual.ts:178` | ID join→missing/duplicate/excluded tallies→certificate | A→A,Q | Are populations on common support? |
| NetworkDisruption — `modules/analysis/src/gpu-network/recipes/network-scenarios-recipe.ts:45` | Baseline routing→closures→scenario routing→two accessibility scores→differences | N,A→A,Q | What does disruption disconnect? |
| FacilityScenario — `modules/analysis/src/gpu-network/recipes/network-scenarios-recipe.ts:279` | Supply mask→accessibility→nearest facility→demand summary | A→A,Q | Whom does this facility set cover? |
| DriveTimeCatchment — `modules/analysis/src/gpu-network/recipes/drive-time-catchment-recipe.ts:474` | Facility snapping/adapters→service areas→optional isochrones/join→demand snapping→bands→grouping | N,P,A→A,(G),Q | Who lies within travel-time bands? |
| TrackQuality — `modules/analysis/src/gpu-network/recipes/track-quality-recipe.ts:64` | Metrics→quality flags/counts→optional similarity→optional map matching | T,(L,N)→A,Q | Which observations need scrutiny? |

## Preconditions and input meaning

| Finding — verified | Source |
|---|---|
| KDE requires a common planar frame/unit; longitude/latitude throws. The catalog’s coordinate model does not encode this full requirement. | `modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts:220` |
| Rates require positive populations; invalid rows are excluded. Exposure distinguishes counts from densities and requires matching area units. | `modules/analysis/src/gpu-spatial-analysis/recipes/rate-mapping-recipes.ts:38`; `raster-summary-recipes.ts:69` |
| Uploaded weights need ordered, unique neighbors and finite nonnegative weights. A GPU validator exists; connectivity is not among its checks. | `modules/analysis/src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-validation.ts:36` |
| Connectedness is operation-specific: AZP requires connected initial regions; disconnected regions prevent searching. | `modules/analysis/src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions.ts:132` |
| Diagnostics require more than `predictorCount + 2` fitted rows. OLS excludes nonfinite rows; there is no universal “no nulls” rule. | `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics.ts:187`; `gpu-ordinary-least-squares.ts:125` |
| Dasymetric coverage uniqueness and disjoint partition cores remain caller preconditions. | `modules/analysis/src/gpu-spatial-analysis/recipes/dasymetric-redistribution-recipe.ts:78` |

**Verified:** there is no generic measurement-level field. Local roles do exist: Poisson counts versus logistic binary response, exposure offsets, categories, extensive values, and temporal unit/epoch/encoding/axis. Catalog `role` means operator/helper. `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts:72`, `modules/analysis/src/analysis-core/contracts/temporal-context.ts:42`, `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:338`.

## Data-driven parameters

| Behavior — verified | Source |
|---|---|
| KDE optionally derives Scott/Silverman bandwidth; otherwise bandwidth is supplied. Adaptive KDE derives local factors but still requires pilot bandwidth. | `modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts:260`; `gpu-adaptive-bandwidth-kernel-density.ts:157` |
| GWR selects by AICc; generalized GWR offers LOO-deviance candidate selection; MGWR offers per-component LOO/AICc search. Caller candidates/bounds remain necessary. | `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression.ts:291`; `gpu-generalized-geographically-weighted-regression.ts:128`; `gpu-multiscale-geographically-weighted-regression.ts:70` |
| Correlogram optionally returns first-peak and maximum-z band indices, rather than universally choosing an “optimal distance.” | `modules/analysis/src/dataframe-analysis/pair-statistics/gpu-spatial-correlogram.ts:121` |
| Nested interpolation selects covariance/RBF candidates using inner held-out MSE. Ordinary interpolation diagnostics evaluates supplied kriging parameters. | `modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation.ts:55`; `modules/analysis/src/gpu-spatial-analysis/recipes/interpolation-diagnostics-recipe.ts:27` |
| Spatial cross-validation constructs folds; cross-validated OLS evaluates fits without bandwidth/variogram selection. Variogram-model fitting separately estimates nugget/sill/range. | `modules/analysis/src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation.ts:39`; `gpu-spatial-cross-validated-ols.ts:46`; `modules/analysis/src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit.ts:40` |

## Opportunities

All **inferred**, directly from the findings:

- Fill empty kinds and replace family-wide kinds with named, required/optional port contracts before treating paths as executable. Ontology: `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:314`.
- Declare checkable requirements, distinguishing construction checks, GPU validation and caller certificates. Contract: `dev-docs/contributor-contract.md:60`.
- Add question intents and input roles: “clusters” needs to distinguish point grouping, value association and rate clustering; “near” needs metric/units; “X relates to Y” needs response/predictor roles. Existing discovery fields: `modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:210`.
- Attach recipe questions, selectable parameters and result interpretation to curated records; current recipe results describe execution/status rather than discovery. `modules/analysis/src/gpu-spatial-analysis/recipes/recipe-utils.ts:50`.

## What I checked

- **Verified:** earlier workflow study; coverage matrix; contributor contract; API root/domain indexes and recipe documentation. Matrix statuses describe bounded subsets, not complete upstream equivalence: `dev-docs/coverage-matrix.md:5`; API organization: `docs/api/README.md:14`.
- **Verified:** ontology/catalog source imports; all 41 recipe declarations, stage calls and returns; recipe helpers; operation bridge and representative CPU twin; targeted validation/selection implementations cited above.
- **Verified commands:** `rg`, `nl`, `sed`, `cat`, `wc -l`; in-memory esbuild catalog evaluation; TypeScript-AST export/recipe walks; read-only `git status --short`.

## Not checked

- **Verified scope limits:** no files edited; no state-changing Git commands; no repository `dist` read.
- No GPU execution, tests, builds, benchmarks, UI interaction, full CPU-twin parity audit, or exhaustive validation of every contributor.