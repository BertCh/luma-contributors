# D. Existing curated content

**Verified** means source was read; **inferred** means a conclusion from that source. This review follows the earlier study without repeating its executor design.

**Verified:** the checkout includes concurrent lens edits, including network registrations, raster scenarios and a new dataframe scenario. Counts below describe the working tree, not a release. These edits are not reported as defects (`modules/lenses/src/computation-lenses/scenarios/gpu-network/register-expansion.ts:7`; read-only status command below).

## Scenarios and packs

- **Verified:** a scenario descriptor has `id`, `entry`, exported `element`, title, audience, available/default rungs, optional example path and dataset `{id, licence, source}`, plus lazy `load()`. Its runner takes a device and optional cancellation signal and returns a computation record with sidecars (`modules/lenses/src/computation-lenses/types.ts:18`).
- **Verified:** a *computation pack* is explanatory metadata: operator descriptors, formulas, analysis/capture stages, variable kinds, neighbourhoods, metadata, views and guards. It is distinct from a runnable scenario (`modules/lenses/src/computation-views/registry/view-registration.ts:54`, `:222`).
- **Verified:** registration imports cover five domains. The source-registration census found **417 unique scenarios**; it evaluated registrations in memory without invoking `load()` (`modules/lenses/src/computation-lenses/scenarios/index.ts:5`).

| Domain | Verified count | Verified primary dataset families and recorded licence classes |
|---|---:|---|
| Spatial analysis | 254 | `s1-chicago-tracts`; `s1-dixie-lattice`; North Side `s2-` nature, tracts, areas, roads, parks and facilities; Midwest counties; Helene rain; `s3-` Chicago parks, areas, roads/area-roads and places; NY harbour AIS/zones; OpenFlights; combinations; repository-authored synthetic families. Terms vary below. `modules/lenses/src/computation-lenses/data/README.md:3` |
| Network | 41 | Loop roads/polylines: ODbL; simulated traces: ODbL-derived; crashes and taxi OD flows: Chicago terms; synthetic graph, routing, optimisation and trajectory fixtures: MIT. `modules/lenses/src/computation-lenses/data/README.md:30`; `modules/lenses/src/computation-lenses/scenarios/gpu-network/register-expansion.ts:15` |
| Raster | 51 | Dixie DEM: public-domain USGS with terrain-provider attribution; bands: Copernicus; landcover: CC BY 4.0; perimeter: public-domain NIFC; GFS wind: public-domain NOAA; synthetic expansion, coverage and remote-sensing families: MIT. `modules/lenses/src/computation-lenses/scenarios/gpu-raster/raster-lens-datasets.ts:8` |
| Terrain | 46 | Dixie DEM/Terrarium: same USGS/provider terms; bounded-grid and terrain-processing synthetics: MIT. `modules/lenses/src/computation-lenses/scenarios/gpu-raster/raster-lens-datasets.ts:43`; `modules/lenses/src/computation-lenses/scenarios/gpu-terrain/terrain-processing.scenarios.ts:57` |
| Dataframe | 25 | `f-chicago-tracts` and `f-chicago-nature`: mixed licences including non-commercial observations; `f-ghcn-stations`: public domain; pair-statistics and variogram/coregionalisation synthetics: MIT. `modules/lenses/src/computation-lenses/data/README.md:27`; `modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/expansion.lens.ts:7` |

**Verified:** spatial dataset terms divide as follows:

- Nature and North Side tract extracts include CC BY-NC observations/counts; counties and rainfall are public domain; community areas use Chicago terms; facilities combine Chicago and Overture terms (`modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/s2-datasets.ts:6`).
- Parks, roads and OpenFlights use ODbL; places combine CDLA-Permissive, Apache and CC0; AIS/zones use CC0/public-domain NOAA data (`modules/lenses/src/computation-lenses/data/README.md:10`).
- The current clean `s1-chicago-tracts` README excludes nature columns and records federal public-domain inputs, but the top-level notice still lists it among nature derivatives. This is a **record conflict**, not a resolved licensing finding (`modules/lenses/src/computation-lenses/data/README.md:3`; `THIRD-PARTY-DATA.md:60`).

**Verified:** another host can register, look up and invoke scenarios through exported APIs; registration itself does not execute them (`modules/lenses/src/computation-lenses/registry.ts:18`; `modules/lenses/src/computation-lenses/run-lens-scenario.ts:15`; `modules/lenses/package.json:22`). **Inferred:** these are loadable, preconfigured analysis examples, rather than general workflow templates with declared replaceable ports.

**Verified:** an exception supports caller-mapped numeric columns for **20 dataframe scenarios** using three embedded dataset IDs. Runs serialize through module-scoped input; missing roles produce named errors (`modules/lenses/src/computation-lenses/user-input.ts:8`, `:68`, `:84`). Bundled scenarios otherwise require a bundler-compatible JSON-loading environment (`modules/lenses/src/computation-lenses/data/README.md:38`).

## Discovery vocabulary

**Verified:** source contains **532 catalog operations**, **12 data types**, **12 use cases** and **10 domains**. The following counts match the website’s **input-or-output** facet rule (`website/src/catalog/index.ts:82`; `website/src/catalog/taxonomy.ts:9`). Mapping between vocabularies is **inferred**, not an implemented translation.

| Website type — verified matches | Ontology correspondence — inferred | Kepler binding geometry correspondence — inferred |
|---|---|---|
| points — 152 | points | point |
| lines — 71 | lines | line |
| polygons — 111 | polygons; sometimes regions | polygon |
| raster — 158 | surface | None |
| terrain — 53 | surface | None |
| network — 50 | network | No topology kind; lines can represent edges |
| table — 298 | attributes, summary, pairs | `none` output; no table input kind |
| time-series — 60 | time-series | None |
| trajectories — 29 | trajectories | No timestamped-track kind |
| point-cloud — 11 | No dedicated kind; points is approximate | No dedicated 3D kind |
| cells — 30 | cells | No cell-key kind |
| spatial-weights — 78 | spatial-weights | No weights kind |

**Verified:** ontology has **15 kinds**, additionally including `parameters` and `render-geometry`; bindings accept only `point|line|polygon` and output those or `none`. These vocabularies therefore carry different information (`modules/analysis/src/gpu-spatial-analysis/ontology/spatial-analysis-ontology.ts:6`; `modules/kepler/src/analysis/catalog-binding-types.ts:47`).

**Verified:** search requires every whitespace-separated word to occur somewhere in title, symbol, summary or ID. Ranking favours title, then symbol. Filters combine one selection per facet; counts ignore that facet’s own selection (`website/src/catalog/index.ts:89`, `:112`, `:136`).

**Verified:** “Before” and “Next” score overlapping output/input types within a shared domain, discount generic tables, and show up to three suggestions. **Inferred:** these are useful discovery hints, not compatibility checks: they do not check complete inputs, units or capacities (`website/src/catalog/index.ts:173`; `website/src/components/catalog/function-page.tsx:59`).

## A visitor’s route today

**Verified**, from route/component source:

1. Home asks “What do you have?” and links type cards to facet pages. Points, polygons, raster and network have direct cards; “trips” requires recognising **Trajectories**, described as tracks with timestamps (`website/src/routes/index.tsx:23`, `:125`; `website/src/catalog/taxonomy.ts:23`).
2. The facet page shows matching operations and “Combine with other filters” (`website/src/components/catalog/facet-page.tsx:11`).
3. Explore adds text search, use case and domain; selections persist in URL parameters. Results initially show 36 cards (`website/src/routes/explore/index.tsx:12`).
4. A function page offers its picture, controls, explanation, typed parameters, returns, example, related operations and sometimes a recorded run (`website/src/components/catalog/function-page.tsx:98`, `:160`).

**Inferred:** visitors manually identify their type and question, inspect candidates, then use the API example in a host. This reviewed path does not inspect an uploaded dataset or infer a complete analysis from its schema. The development harness mounts one operation ID and changes parameters/pointer state (`website/src/gpu/runs/harness.ts:102`, `:182`).

## Documented chains and strongest existing candidates

**Verified:** the workflow page documents five recipes plus a hand-wired Local Moran chain. Its recipe records describe chained executions, but step-card pictures load independent operation posters, **not chain intermediates**. Drainage’s record uses a 3×3 synthetic grid (`website/src/components/meta/pages/workflows/index.tsx:181`; `website/src/components/meta/pages/workflows/chain-figure.tsx:13`; `website/src/components/meta/pages/workflows/workflows.ts:465`).

Selection as a strong curated candidate is **inferred**; inputs and steps below are **verified**. Shipping assessments are **inferred from repository licence records**, for templates **including these fixtures**. Code-only templates with user-supplied data are a separate case.

| Candidate | Inputs and existing steps — verified | Fixture shipping — inferred | Evidence |
|---|---|---|---|
| Points to hot spots | Points + study polygon → cover → cell aggregation/zero-fill → centres/density → topology → Gi*; optional permutation/classification/colour | Nature fixture: non-commercial restriction | `website/src/components/meta/pages/workflows/workflows.ts:132` |
| Rate clusters | Polygons + counts/population → empirical Bayes → contiguity → row weights → Local Moran → permutation → colour | North Side tracts: mixed/non-commercial | `website/src/components/meta/pages/workflows/workflows.ts:227` |
| Emerging hot spots | Timed points + lattice → calendar/period indices → count cube → emerging classes; optional colour | Nature fixture: non-commercial | `website/src/components/meta/pages/workflows/workflows.ts:288` |
| Drive-time catchment | Network + facilities/demand → snap → service areas → optional isochrones/join → demand travel times → band statistics | ODbL obligations unresolved here | `website/src/components/meta/pages/workflows/workflows.ts:343` |
| Drainage | Elevation raster → flow → stream order → watersheds → downstream distance → stream lines → catchment summary | Synthetic fixture: MIT | `website/src/components/meta/pages/workflows/workflows.ts:411` |
| Local Moran | Tract polygons + poverty → queen weights → row standardisation → Local Moran with correction/gating | Clean extract marked public domain; notice conflict | `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/local-moran.example.ts:55` |
| Cluster and outline | Nature points → DBSCAN, 250m/six points → cluster hulls/measures | Non-commercial | `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/cluster-and-outline-recipe.example.ts:33` |
| Fleet dwell | AIS trajectories + anchorages → stops → polygon join → slot mask → dwell statistics | CC0/public domain | `modules/analysis/src/gpu-spatial-analysis/recipes/fleet-dwell-recipe.ts:256` |
| Points-in-polygons map | Points + tracts → zonal aggregation or join/group → breaks → colour | Nature/tract fixture: non-commercial | `modules/analysis/src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe.ts:180` |
| Period comparison | Spring/autumn points → two cell tables → comparison → classes/colour | Nature fixture: non-commercial | `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/period-comparison-recipe.example.ts:21` |
| Surface preparation | Raster → fill missing cells → optional rank filter → provenance/unresolved counts | Synthetic lens fixture: MIT | `modules/analysis/src/gpu-spatial-analysis/recipes/surface-preparation-recipe.ts:32`; `modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/expansion.lens.ts:93` |
| Network disruption | Network + closure/cost scenario → baseline/scenario cost matrices → accessibility → differences/unreachable flags | Synthetic diamond: MIT | `modules/analysis/src/gpu-network/recipes/network-scenarios-recipe.ts:145`; `modules/lenses/src/computation-lenses/scenarios/gpu-network/register-expansion.ts:67` |

**Verified:** website runs sometimes provide different, more permissive fixtures: earthquake points replace nature observations; website recipe notes distinguish observed inputs from constructed zones, populations and hazards (`website/src/gpu/data/recipe-sources.md:7`). Website pictures therefore do not establish the lens fixture’s licensing or provenance.

## Reusable explanation and run inspection

- **Verified:** all **532** entries have explanations, parameter docs, output docs, examples and reference links; **409** name a lens scenario. There is no separate result-interpretation field in the schema (`website/src/catalog/schema.ts:9`). Existing text includes HH/LL/HL/LH meanings, undefined z-scores and noise sentinels (`website/src/catalog/entries/spatial-statistics.ts:71`, `:94`).
- **Verified:** the **13** how-it-works topics explain buffers/offsets, capacity, precision, graph hazards, scheduling, kernels, parallel primitives, iteration, contracts, workflows, records and package ownership. They are system explainers rather than one tutorial per operation (`website/src/components/meta/topics.ts:38`). Anatomy walks kernel density from buffers to readback (`website/src/components/meta/pages/anatomy/index.tsx:30`).
- **Verified:** packs add formulas, assumptions, alternatives and guards; Local Moran includes row-standardisation, inference and correction guidance (`modules/lenses/src/gpu-spatial-analysis/computation-pack/operators/statistics.ts:16`).
- **Verified:** record tables expose graphs, reads/writes, resources, captures, decisions, choices and metrics. Datum receipts order captured values by stage/node; provenance distinguishes readback, replay, derived and metadata (`modules/lenses/src/computation-views/views/table-view.ts:120`; `modules/lenses/src/computation-views/views/datum-receipt.ts:56`; `modules/lenses/src/computation-views/types.ts:151`).
- **Inferred:** these views could explain a multi-step run when stages, captures and key spaces are recorded. The workflow record itself stores only ordered graph invocations, labels, reasons and iterations—not dataset dependency edges (`modules/analysis/src/computation-record/computation-record-types.ts:504`).

## What I checked

- **Verified:** read the earlier study first; `SITE-PLAN.md`, `BRAND.md`, `THIRD-PARTY-DATA.md`; cited lens contracts, registration/data files, runners/examples, view models, catalog/routes, workflow components, topic content and harness.
- **Verified:** read `website/scripts/extract-inventory.ts`: it resolves source exports/types, joins capability metadata and recorded scenarios, then writes inventory. I did **not** execute it (`website/scripts/extract-inventory.ts:62`, `:92`, `:347`).
- **Verified:** count commands: read-only `node -e` censuses using `typescript.transpileModule` and `vm.runInNewContext`; scenario imports were intercepted, registrations collected, and lazy runners never called. Catalog entries were evaluated in memory. A separate `python3 -c` regex census reproduced all 532 data-flow entries and facet counts.
- **Verified:** shell cross-checks included `rg '^    symbol:' website/src/catalog/entries --glob '*.ts' | wc -l` and equivalent searches for `explanation`, `lensScenario`, `reference`; `rg --files`, `nl -ba`, `sed`; and `GIT_OPTIONAL_LOCKS=0 git status --short -- …` scoped to reviewed paths.

## Not checked

- Browser/WebGPU execution, screenshots, every scenario runner, full generated records, build/test results, package publication, or external licence texts.
- Concurrent Kepler implementation beyond binding types; generated inventory freshness against every current export.

## Opportunities

All **inferred** from the findings above:

1. Reuse explanations, output meanings, formulas and guards for contextual help.
2. Distinguish “accepts this type” from “produces this type” in discovery.
3. Present existing neighbour suggestions as hints with explicit compatibility limits.
4. Curate initial templates from existing chains while identifying fixture terms and resolving the clean-tract notice conflict.