# E. Analysis UI patterns review

**Verified** means source was read; **inferred** means a conclusion from that source. This is a read-only snapshot, following the earlier workflow study, without repeating its proposed document or executor.

## Choosing an analysis

| Status | Example | Inventory and navigation |
|---|---|---|
| **Verified** | Explorer | **66 modes in 10 categories**, selected through category buttons and mode tabs. Categories: points/density **6**; joins/geometry **12**; weights/statistics **8**; regression/models **2**; cells **2**; networks **8**; trajectories/time **10**; change of support **4**; raster/terrain **13**; recipes **1**. Uncategorised registrations remain reachable under “Other.” `examples/deck/spatial-analysis-explorer/modes/index.ts:79`; `examples/deck/spatial-analysis-explorer/modes/mode-categories.ts:22`; `examples/deck/spatial-analysis-explorer/app-ui.ts:159` |
| **Verified** | Explorer recipes | The Recipes mode contains **14 choices**, including alternative rate, choropleth and fleet-dwell variants. These are additional configurations within one mode. `examples/deck/spatial-analysis-explorer/modes/recipes-scenes.ts:34` |
| **Verified** | Showcase | **173 stories across 17 chapters**. The atlas searches titles, questions, contributors and dataset metadata; filters select chapters, charts, timelines or comparisons. Story pages have **Story / All controls / Details** tabs. `examples/deck/spatial-analysis-showcase/src/scenes/registry.ts:15`; `examples/deck/spatial-analysis-showcase/src/shell/home-view.ts:171`; `examples/deck/spatial-analysis-showcase/src/shell/story-view.ts:68` |
| **Verified** | Showcase curation | **Seven curated routes**, each linking three studies around a question. These navigate between stories; the route renderer does not connect their computation outputs. `examples/deck/spatial-analysis-showcase/src/shell/atlas-routes.ts:11`; `examples/deck/spatial-analysis-showcase/src/shell/home-view.ts:61` |

**Verified chapter counts:** cells 5, tables/time series 7, earth 14, flows 14, geometry 17, hydrology 7, interpolation 7, joins 6, movement 14, networks 14, points 10, raster 16, regression 9, statistics 12, terrain 11, space-time 6, weights 4. Chapter labels are defined at `examples/deck/spatial-analysis-showcase/src/scenes/chapters.ts:14`.

**Verified count commands run:**

```sh
sed -n '/^export const SPATIAL_ANALYSIS_MODES:/,$p' examples/deck/spatial-analysis-explorer/modes/index.ts | rg -c '^  \w+Mode,?$'

rg -c '^    id:' examples/deck/spatial-analysis-explorer/modes/mode-categories.ts examples/deck/spatial-analysis-explorer/modes/recipes-scenes.ts examples/deck/spatial-analysis-showcase/src/scenes/chapters.ts

python3 -c 'from pathlib import Path; from collections import Counter; files=list(Path("examples/deck/spatial-analysis-showcase/src/scenes").glob("*/*.scene.ts")); print(len(files)); print(dict(sorted(Counter(p.parent.name for p in files).items())))'

sed -n '/^export const ATLAS_ROUTES =/,/] as const/p' examples/deck/spatial-analysis-showcase/src/shell/atlas-routes.ts | rg -c '^    title:'
```

An additional Python source scan counted each scene’s declared `chapter` and produced the same grouping.

## Parameters and live execution

| Status | Pattern | Observed behaviour |
|---|---|---|
| **Verified** | Explorer controls | Imperative builders provide sliders, selects, toggles, buttons, notes, legends and readouts. Sliders call their handler on browser `input`, allowing continuous adjustment. `examples/deck/spatial-analysis-explorer/app-ui.ts:20`; `examples/deck/spatial-analysis-explorer/app-ui.ts:280` |
| **Verified** | Explorer density | Sigma and smoothing write kernel buffers/select existing graphs; time-window choices rewrite a mask; source-point visibility changes layers. Switching square/hexagonal binning builds replacement graphs. Viewport bounds are written during frame encoding. `examples/deck/spatial-analysis-explorer/modes/density-mode.ts:293`; `examples/deck/spatial-analysis-explorer/modes/density-mode.ts:408` |
| **Verified** | Explorer map gestures | Lasso supports drawn polygons, dragged rectangles and movable circles. Shape parameters are uploaded each frame. Shape/path variants are compiled beforehand; point population, index and capacity controls request rebuilds. `examples/deck/spatial-analysis-explorer/modes/lasso-mode.ts:613`; `examples/deck/spatial-analysis-explorer/modes/lasso-mode.ts:692`; `examples/deck/spatial-analysis-explorer/modes/lasso-mode.ts:1030`; `examples/deck/spatial-analysis-explorer/modes/lasso-mode.ts:1115` |
| **Verified** | Showcase controls | Declarative options include sliders, ranges, segmented choices, chips, toggles and multi-value presets. `apply: 'param'` and `apply: 'compile'` distinguish update costs; the panel displays rebuild badges and folds expert controls. `examples/deck/spatial-analysis-showcase/src/scenes/scene.ts:68`; `examples/deck/spatial-analysis-showcase/src/scenes/scene.ts:148`; `examples/deck/spatial-analysis-showcase/src/shell/options-panel.ts:99` |
| **Verified** | Showcase density | Binning, resolution, statistic and accumulation select a cached graph set or compile a missing one. Sigma/smoothing write kernels; hours/category/day choices rewrite masks; display options request layers. Computation runs when dirty, with settled-field readback after 500 ms. `examples/deck/spatial-analysis-showcase/src/scenes/points/nature-density.compute.ts:714`; `examples/deck/spatial-analysis-showcase/src/scenes/points/nature-density.compute.ts:821`; `examples/deck/spatial-analysis-showcase/src/scenes/points/nature-density.compute.ts:933` |
| **Verified** | Sweeps and spatial presets | Showcase slider sweeps write through the option store every animation frame, with pause/step controls. Reduced-motion mode advances once per press. Viewshed has uncertainty presets and draggable observer/target markers. `examples/deck/spatial-analysis-showcase/src/shell/options-panel.ts:912`; `examples/deck/spatial-analysis-showcase/src/scenes/terrain/viewshed.scene.ts:183`; `examples/deck/spatial-analysis-showcase/src/scenes/terrain/viewshed.compute.ts:2080` |

**Verified scheduling mechanics:**

- The shared effect encodes computation into Deck’s frame encoder before layers draw. It supplies viewport, elapsed time and frame index. Both examples use this implementation. `examples/deck/spatial-analysis-shared/analysis-effect.ts:83`; `examples/deck/spatial-analysis-showcase/src/engine/analysis-effect.ts:6`.
- `scheduleReadback` defers starting the read until the next task, allowing Deck to submit the encoded copies first. Explorer’s local file re-exports it. `examples/deck/spatial-analysis-shared/readback-scheduling.ts:5`; `examples/deck/spatial-analysis-explorer/readback-scheduling.ts:6`.
- `SummaryReader` allows one read in flight, retains a stale request, and retries through subsequent frame flushes. Showcase wraps it with scene-labelled logging. `examples/deck/spatial-analysis-shared/summary-reader.ts:60`; `examples/deck/spatial-analysis-showcase/src/engine/summary-reader.ts:17`.
- `GraphSwap` serialises builds, keeps only the latest queued request, discards superseded results and commits synchronously. Existing graphs remain active during compilation. Explorer and showcase re-export this shared implementation. `examples/deck/spatial-analysis-shared/graph-swap.ts:47`; `examples/deck/spatial-analysis-explorer/graph-swap.ts:6`; `examples/deck/spatial-analysis-showcase/src/engine/graph-swap.ts:6`.
- “Live” does not always mean recomputing continuously: recipe `DirtyEncoder` runs after a parameter change, then only flushes pending summaries. `examples/deck/spatial-analysis-explorer/modes/recipes-kit.ts:454`.

## Explaining results and exposing chains

| Status | Evidence |
|---|---|
| **Verified** | Explorer shows contributor names, a mode description, legends, numerical summaries and provenance. Density additionally reports dense/separable timings, speedup and maximum field difference. `examples/deck/spatial-analysis-explorer/app-ui.ts:214`; `examples/deck/spatial-analysis-explorer/modes/density-mode.ts:369` |
| **Verified** | Lasso compares GPU selected counts with a CPU recount, displaying “match,” “recounting,” or “differs or in flight.” Large-population recounts wait until the shape settles; picking-texture selections explicitly lack this comparison. `examples/deck/spatial-analysis-explorer/modes/lasso-mode.ts:938` |
| **Verified** | Showcase step cards combine headlines, explanatory text, evidence, limits, charts, selected controls and numerical tiles. Technical readouts remain under “Under the hood” unless explicitly included. `examples/deck/spatial-analysis-showcase/src/shell/story-view.ts:2124`; `examples/deck/spatial-analysis-showcase/src/shell/story-view.ts:2209` |
| **Verified** | Burn severity places stored CPU-reference values beside GPU means. Changed formulas produce a message that the reference requires defaults; the comparison is conditional. `examples/deck/spatial-analysis-showcase/src/scenes/raster/burn-severity.compute.ts:814` |
| **Verified** | Explorer’s cluster-and-outline recipe puts DBSCAN, group geometry, convex hulls, geometry measures and a display adapter into one graph. Points and hulls are visible; counts, areas, perimeters and overflow become summaries. Other scratch products stay internal. `examples/deck/spatial-analysis-explorer/modes/recipes-scenes-spatial.ts:88`; `examples/deck/spatial-analysis-explorer/modes/recipes-scenes-spatial.ts:158`; `examples/deck/spatial-analysis-explorer/modes/recipes-scenes-spatial.ts:235` |
| **Verified** | Showcase burn severity chains band arithmetic → change calculation → cloud screening → burned mask/reclassification in one main graph. Statistics, stretching and patch processing use additional graphs. Display choices expose selected products rather than every buffer. `examples/deck/spatial-analysis-showcase/src/scenes/raster/burn-severity.compute.ts:156`; `examples/deck/spatial-analysis-showcase/src/scenes/raster/burn-severity.compute.ts:284`; `examples/deck/spatial-analysis-showcase/src/scenes/raster/burn-severity.compute.ts:389` |
| **Verified** | Showcase pipeline chips can reveal intermediate products by writing options. Nature clustering exposes records, core flags and hulls; union/label stages remain explanatory chips. This is selective inspection, not arbitrary execution to a node. `examples/deck/spatial-analysis-showcase/src/shell/story-card-parts.ts:101`; `examples/deck/spatial-analysis-showcase/src/scenes/points/nature-clusters.scene.ts:504` |

## Dataset binding and analysis inference

- **Verified:** Explorer modes obtain named fixtures through six fixed catalog getters, with deterministic synthetic fallback. Some modes offer fixture-derived populations, such as trip endpoints versus POIs; this is not an arbitrary dataset selector. `examples/deck/spatial-analysis-explorer/spatial-analysis-data.ts:250`; `examples/deck/spatial-analysis-explorer/spatial-analysis-data.ts:303`; `examples/deck/spatial-analysis-explorer/modes/clusters-mode.ts:102`.
- **Verified:** Showcase scenes declare dataset IDs and descriptive roles. Activation loads those IDs and rejects access to undeclared datasets. The Data page finds stories already associated with a source. `examples/deck/spatial-analysis-showcase/src/scenes/scene.ts:650`; `examples/deck/spatial-analysis-showcase/src/shell/story-view.ts:1808`; `examples/deck/spatial-analysis-showcase/src/shell/story-view.ts:1828`; `examples/deck/spatial-analysis-showcase/src/shell/data-view.ts:17`.
- **Inferred:** Neither reviewed selection path infers suitable analyses from a newly supplied schema. Discovery uses authored metadata and fixed dataset associations. Those same files support this conclusion.
- **Inferred:** Running an existing mode on user data requires a dataset adapter, field/geometry mapping, appropriate spatial context, capacities and replacement geographic defaults. Density’s nature adapter, for example, expects named timestamp/category/research-grade/species columns and assumes a 2023 calendar. Explorer’s frame helper recognises only its two fixture origins. `examples/deck/spatial-analysis-showcase/src/scenes/points/b1-nature-data.ts:43`; `examples/deck/spatial-analysis-explorer/spatial-analysis-data.ts:78`.
- **Verified:** The “your own data” guide describes developer integration through arrays, buffers and a new scene; it supplies no upload interaction. Its projection example differs from the explicit analysis-frame guidance, so it is not sufficient evidence for the current coordinate contract. `examples/deck/spatial-analysis-showcase/src/docs/guides/your-own-data.md:7`; `examples/deck/spatial-analysis-showcase/src/engine/analysis-frame.ts:15`.

## Comparison with the current Kepler host

Absence claims below are **inferred**, limited to the inspected host source and searches.

| Status | Pattern without an equivalent found | Evidence on both sides |
|---|---|---|
| **Inferred** | Guided investigation with step-specific controls, evidence and limits | Showcase `examples/deck/spatial-analysis-showcase/src/shell/story-view.ts:2124`; Kepler uses operation/configuration forms at `examples/deck/kepler-compute/src/analysis-panel.tsx:169` and `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:347`. |
| **Inferred** | Clickable intermediate-stage strip | Showcase `examples/deck/spatial-analysis-showcase/src/shell/story-card-parts.ts:101`; Kepler lists contributors/contracts and output tables at `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:367` and `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:503`. |
| **Inferred** | General parameter sweeps plus rebuild-cost badges | Showcase `examples/deck/spatial-analysis-showcase/src/shell/options-panel.ts:184` and `examples/deck/spatial-analysis-showcase/src/shell/options-panel.ts:912`; Kepler catalog parameters update drafts before Run, while DBSCAN uses a submitted configuration form. `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:388`; `examples/deck/kepler-compute/src/analysis-dbscan-panel.tsx:54`. |

**Verified existing equivalents:** Kepler already has lens scenarios with user-table mappings, sample descriptions, CPU-reference selection, split/swipe views and resident legends. These are not missing-example features. `examples/deck/kepler-compute/src/analysis-catalog-panel.tsx:305`; `examples/deck/kepler-compute/src/analysis-catalog-lens-mapping.tsx:56`; `examples/deck/kepler-compute/src/sample-gallery-panel.tsx:45`; `examples/deck/kepler-compute/src/split-view.ts:46`; `examples/deck/kepler-compute/src/resident-channel-legend.tsx:35`.

## Small screens

- **Verified:** Both `mobile-support.ts` files declare reduced, large-data, WebGPU-only support. These declarations alone do not demonstrate reduced workloads for every analysis. `examples/deck/spatial-analysis-explorer/mobile-support.ts:7`; `examples/deck/spatial-analysis-showcase/mobile-support.ts:7`.
- **Verified:** Below 600 px, Explorer replaces category buttons with a select, hides introductory text and limits the scrolling panel to 58% height. `examples/deck/spatial-analysis-explorer/app-ui.ts:136`.
- **Verified:** Below 900 px, Showcase uses a draggable peek/half/full bottom sheet. Map furniture clears its height; companion charts move into the sheet. Touch controls have 44 px targets. `examples/deck/spatial-analysis-showcase/src/shell/story-sheet.ts:29`; `examples/deck/spatial-analysis-showcase/src/shell/story-view.css:747`; `examples/deck/spatial-analysis-showcase/src/shell/story-view.css:859`.
- **Inferred:** A workflow UI must accommodate map visibility, controls and evidence at different times; the existing sheet already embodies that constraint.

## Opportunities

All are **inferred** from the cited findings:

1. Reuse guided steps for curated analyses.
2. Expose selected intermediate products through pipeline chips.
3. Carry parameter update-cost metadata into analysis forms.
4. Offer sweeps for suitable sensitivity parameters.
5. Separate reusable dataset-role mappings from fixture-specific story assumptions.
6. Preserve sheet-based controls and evidence on small screens.

## What I checked

- **Verified:** Read `AGENTS.md`, the workflow study, and targeted source ranges in every file cited above.
- **Verified:** Additional reads/searches covered Explorer `app.ts`, `spatial-analysis-mode.ts`, recipe scaffolding; Showcase `main.ts`, `data/catalog.ts`, `scenes/options-state.ts`, `engine/deck-host.ts`, terrain-conditioning/source-finder sources, and `styles.css`; shared scheduling/resource files; Kepler catalog, sample, analysis, split-view and legend sources.
- **Verified:** Ran `cat`, `nl -ba`, `sed`, `head`, `tail`, `rg`, `rg --files`, `wc`, Python source-count commands, and scoped `git status --short`. A heredoc count attempt failed because the read-only environment rejected its temporary file; subsequent `python3 -c` counts succeeded.
- **Verified:** Scoped status showed locally modified example package manifests. Their dependency changes were treated as ongoing work, not defects. `examples/deck/spatial-analysis-explorer/package.json:14`; `examples/deck/spatial-analysis-showcase/package.json:19`.

## Not checked

- Browser execution, screenshots, actual mobile devices or accessibility testing.
- GPU correctness, measured latency, compilation costs or memory use.
- Every parameter and graph across all modes/stories.
- Exhaustive Kepler feature parity, persistence or workflow execution.
- `dist`, dependency installation, builds, linting or tests. No files or Git state were changed.