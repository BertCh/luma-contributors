> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/showcase-sprint-design-2026-10-24.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Sprint finished; historical record only.

# Spatial statistics showcase sprint: editorial and design brief

Date: 2026-10-24
Scope: design/review only. This document does not change showcase runtime or claim that the analyses listed as underway have passed numerical validation.

## Status vocabulary

- **Implemented foundation** means the shared showcase shell, chart vocabulary, cartographic conventions, and existing stories observed in the repository.
- **In progress** means an implementation is underway in this sprint according to the sprint brief. It is not a claim of completion or scientific validation.
- **Planned** means editorial, interaction, or acceptance work proposed here for those in-progress scenes.
- **Hypothetical next wave** means a research and story concept only. It is not an implementation commitment or a claim that the data and methods are ready.

## Editorial thesis

Present spatial analysis as a sequence of decisions, not a catalogue of algorithms. Each story should start with a place-based question, reveal the evidence and assumptions that answer it, and let the reader change one consequential choice at a time. Keep an explicit path from observation to derived statistic to uncertainty or limitation. A spatial pattern is not automatically a mechanism or a causal explanation.

Use one shared editorial rhythm:

1. **Question and evidence:** name the place, population or feature set, time period, unit, source, and what is directly observed.
2. **Construct:** show how the statistic or model turns those inputs into a map. Give the denominator, neighborhood, distance model, class scheme, or raster support in the same view as the result.
3. **Sensitivity:** let one interpretable parameter or alternative specification change the result. Preserve a stable reference where possible.
4. **Check:** compare against held-out observations, a baseline, an alternate scenario, or an explicit null. State what that check does and does not establish.
5. **Limit and next question:** identify missing variables, boundary effects, data coverage, and the inference the analysis cannot support; then link to a related story that supplies another piece of evidence.

Use the shell's `story` steps, readouts, chart insets, map callouts, and `about` fields rather than building parallel narrative UI. The established grammar already supports chapter discovery, a persistent map, camera transitions, controls scoped to each step, all-controls and details tabs, chart-to-option links, hover/tooltips, legends, compare swipes, time bars, reduced motion, responsive sheets, and shareable state. Keep compute modules lazy-loaded and story metadata small.

## Implemented foundation

The existing gallery has chapters for points and density; joins and overlay; geometry; spatial weights; rates, classes and segregation; regression and regionalization; interpolation; cells; networks; flows; movement; space-time; terrain; hydrology; rasters; and dataframes. Stories already demonstrate strong precedent for explicit caveats and linked analysis:

- Rainfall interpolation distinguishes measured gauges from estimated cells, provides no-data where support is insufficient, exposes variogram and method choices, and offers random gauge holdout. The current baseline says elevation and anisotropy are omitted; the spatial-block validation described below is in progress and must not be treated as implemented until verified.
- Existing segregation, rate smoothing, spatial autocorrelation, LISA/cluster, movement and network stories establish ways to combine maps with distributions, scatter plots, transition matrices, timelines, permutation nulls, and operational constraints.
- Chart kinds include lines and uncertainty envelopes, bars and histograms, scatter plots with quadrants/brushes, timelines, matrices, Lorenz curves, forest/dumbbell/slope plots, rose and stacked charts, multiples and diagrams. Shared charts can show values in a table and link a plotted marker back to a scene option.
- The cartography guide establishes semantic palettes, ground-aware light/dark colors, explicit class breaks, no-data and significance states, unit-bearing legends, data-boundary fades, stable category colors, and label/outline cues in addition to color.
- Scene guides define lightweight metadata and data-driven `compute`, `legend`, `readouts`, `story`, and `about` responsibilities. Keep those existing conventions as the implementation boundary.

The above is a repository snapshot, not an assertion that every listed scene currently passes every desired criterion below.

## In-progress scenes and planned editorial shape

All eight analyses below are **in progress**, as reported by the sprint brief. The story titles and interactions are **planned** editorial suggestions. Before publication, verify each implementation against the scientific and acceptance criteria in this document.

| In-progress scene | Editorial question and preferred visual form | Core interaction | Evidence and caveat to keep visible |
| --- | --- | --- | --- |
| Rainfall blocked validation | “Does the surface predict gauges in places with fewer nearby observations?” Pair held-out gauge marks on the map with a residual or observed-versus-predicted scatter, and a small score comparison for IDW and kriging. | Switch random versus spatial blocks; show the held-out fold and block scale; link residual selection to the map. | Show sample count, RMSE/MAE/bias, seed/fold, and support. Spatial blocks are a stress test for spatial extrapolation, not proof of transfer to a new storm; neighboring train blocks can still leak spatial dependence. Refit the variogram and all tuning choices from training data only. One fold is not a cross-validation average. |
| Chicago facility placement | “Which candidate sites improve access under the chosen demand and travel assumptions?” Use a Chicago map with demand and selected facilities, plus a ranked marginal-gain chart and a coverage-by-distance curve. | Change facility type, candidate budget, and distance/travel threshold; select a candidate to see which demand it newly covers; compare before/after access. | Identify how demand weights, candidate locations, distance metric, and existing facilities were chosen. Separate nearest-facility assignment from p-median optimization and distinguish modelled access from service quality. Treat unreachable demand explicitly; do not let a finite sentinel enter an objective sum. Report unserved demand and ties. |
| Dixie habitat connectivity | “How does a resistance surface change the least-cost connections across a burn scar?” Show burn/landcover context beside paths or corridors and resistance, with a side profile or cost-distance distribution. | Switch labeled landcover-to-resistance scenarios; adjust corridor threshold; compare the same endpoints and show the cost scale. | This is a modeled connectivity scenario, not observed animal movement. State that classes and resistance values are assumptions, identify how burn severity/land cover is derived, and explain whether paths are symmetric and whether edge/no-data cells block travel. Use more than one plausible resistance assignment or show a sensitivity comparison. |
| Chicago street enclosure from real footprints | “Where do streets form enclosed blocks, and which geometric details change that answer?” Show actual building footprints and street/block geometry at city and neighborhood scales, with a linked distribution of enclosure measures. | Select a neighborhood or percentile from the chart to highlight geometry; compare a simplified/thresholded view to full footprints only if the statistic remains comparable. | Define “enclosure” quantitatively and identify the footprint and street sources/date. Address gaps, courtyards, bridges/alleys, overlaps, and the effect of the analysis scale. Do not present block geometry as a direct proxy for safety or access without external evidence. |
| Watershed flow distance and reaches | “How far does water travel through this terrain before it reaches a channel or outlet?” Pair a flow-distance surface and extracted reaches with an elevation profile or distance histogram. | Compare routing methods and a stated stream threshold; click a cell/reach to inspect its downstream route and cumulative distance. | Label DEM resolution, vertical/horizontal units, routing, pit/flat handling, outlet convention, and threshold. D8/MFD changes are model choices. Distinguish along-flow distance from straight-line distance; these outputs do not model rainfall, discharge, or flood timing. Any iteration cap must expose incomplete status. |
| Nature density clusters and outliers | “Which observation concentrations persist when the neighborhood scale changes?” Map cluster/noise classes and a density curve or ranked cluster-size chart; keep the raw point pattern available. | Link radius/epsilon and minimum-neighbor controls to the chart; select a cluster to highlight its members and show size/composition. | Nature observations are shaped by volunteer coverage, access, species detectability, and time. State the point support and suppression rule, show noise and zero-density distinctly, and avoid calling a dense cluster a biodiversity hotspot without effort correction or independent ecological evidence. |
| Election rank dynamics | “How did county-level relative position and vote share change across elections?” Use a small-multiple map or selected-year map plus a slope/dumbbell or rank-transition matrix; show absolute share alongside rank. | Scrub election year; select a county to compare its time series and rank; switch measure between Democratic share, Republican share, margin, and turnout proxy with distinct labels. | County boundaries and data coverage differ over time; handle Alaska/Virginia county-equivalents and missing counties explicitly. Rank is relative and discards magnitude. Identify the denominator, election years, party definitions, and turnout proxy; do not imply a county change caused a national swing. |
| LiDAR ground / height | “What is ground, and how does removing it change the observed surface?” Use a point-cloud height coloring with a vertical profile/cross-section and a histogram of classified height. | Toggle raw elevation, ground estimate, and height-above-ground; brush a height range or move a transect and inspect the same points. | Name source, point density/classification, units and vertical datum; label ground as estimated if inferred. Distinguish height-above-ground from absolute elevation. Show missing/low-confidence ground and avoid interpreting sparse or clipped tiles as absence. |

### Cross-story journeys

These are **planned editorial links**; implement as ordinary story links or a lightweight related-stories panel, not as a runtime dependency between compute graphs.

| Journey | Sequence | What the reader learns |
| --- | --- | --- |
| From samples to a defensible surface | Rainfall interpolation → rainfall blocked validation → spatial regression / future held-out raster-change story | A smooth map is a model; validation design changes the question; spatial prediction and explanatory association are different tasks. |
| From observation to urban access | Chicago nature density/outliers → Chicago facility placement → street enclosure → network/equity follow-up | Point concentration, facility allocation, built form, and travel access measure separate parts of an urban system. Each has a different denominator and bias. |
| From terrain to ecological scenario | LiDAR ground/height → watershed flow distance/reaches → Dixie habitat connectivity → raster change | Terrain and raster preprocessing determine downstream results; a least-cost path is conditional on the supplied surface and assumptions. |
| From change to relative position | Election rank dynamics → distribution-dynamics / spatial autocorrelation stories | Absolute magnitude, relative rank, and spatial clustering answer different questions; changes in one do not establish causes for another. |

Keep cross-story links descriptive (“next question”) and preserve the reader's place when returning. Do not imply that two stories share a matched spatial scale or population when they do not.

## Planned visualization and interaction rules

Choose the smallest chart that answers the step's question. Prefer a paired map and chart over multiple decorative readouts. Each chart must have a unit-labeled axis, a useful accessible description, a table of values where feasible, and a visible reference when comparing scenarios. Keep the map's title, legend title, control label, and readout language consistent.

- **Distributions:** histogram or ranked bars for rates, error, cluster sizes, height, and reach lengths. Mark the selected feature or threshold; do not let bin widths imply precision beyond the sample.
- **Relationships and model checks:** scatter for observed versus predicted or residual versus fitted; include a 1:1 or zero line, and preserve point selection linkage. A fit line is descriptive unless the design justifies inference.
- **Time:** timeline for elections or raster snapshots; label the actual dates and missing periods. Avoid interpolated animation that makes unobserved intermediate values look measured.
- **Alternatives and transitions:** slope/dumbbell for two scenarios; matrix for movement between classes or ranks. Show absolute counts or denominators and preserve category identity across years.
- **Sensitivity:** link chart markers to a single live control where an interpretable parameter sweep is meaningful. Keep defaults grounded in the scene's stated method and annotate when a parameter changes scientific assumptions rather than only visual display.
- **Map interaction:** picking a unit should update a chart or readout, and chart selection should highlight the same unit. Hover should distinguish observation, estimate, missing value, and model output. Keep essential meaning in keyboard-accessible controls and tables; do not make color or hover the only source.
- **Comparison:** use the existing swipe only when both sides share projection, extent, support, class breaks, and timing. Otherwise use small multiples or a linked chart and state why direct swipe comparison is unsafe.
- **Performance:** use the existing asynchronous scene initialization and frame-updated parameters. Keep expensive rebuild controls explicit, report the selected method/status, and prevent a stale result from appearing current while a graph rebuilds.

## Scientific risk register

1. **Validation leakage and tuning:** Holdout data must not influence variogram fits, standardization, feature selection, raster thresholds, or parameter choice. A spatial partition reduces one form of leakage but does not establish independence. Report the split geometry, scale, seed and repeats.
2. **Support and denominator changes:** Rates require exposure; point density requires a defined window/area and observation effort; access needs demand weights; election share needs a vote denominator. Show the denominator and missingness beside the map. Do not compare different supports as if they were the same units.
3. **Relative versus absolute metrics:** ranks, standardized scores, cluster labels, and Jenks classes can change because peers or breakpoints change. Keep underlying magnitudes available; use stable colors for identities, not ordinal positions.
4. **Model scenario versus observation:** habitat resistance, facility allocation, watershed routing, estimated ground, and fitted surfaces depend on parameters or derived inputs. Mark those as modeled. Never phrase a scenario output as observed behavior or causal impact.
5. **Boundary and selection effects:** study extent, tile edges, coastlines, city limits, truncated networks, and missing raster cells can create artifacts. Fade or mark unsupported edges and list excluded features/counts.
6. **Geometric validity and units:** projected distances and areas need an appropriate CRS, correct scale, and clearly stated units. Explain whether distances are straight-line, geodesic, raster-cell, or network distance. A visually plausible geometry is not evidence of correct measurement.
7. **Iteration and capacity status:** capped iterative procedures must not silently display partial output as complete. Expose convergence/overflow and suppress or mark partial products.
8. **Ecological and political interpretation:** citizen-science effort, demographic aggregation, and vote returns can be misread as individual-level behavior or causes. Keep the unit of analysis in the title/caveat and avoid ecological fallacy claims.

## Hypothetical next wave: gaps in spatial-regression storytelling, raster change, network equity, and movement

These are **hypothetical** concepts for a later research/design wave. Existing APIs, datasets, and other gallery scenes may cover parts of the methods, but none of the story concepts below is claimed complete or committed.

### Spatial regression: association, residual structure, and local instability

Use a county-scale or urban-area dataset with a documented response, covariates, period, and adequate sample size. A story could compare a global linear model, a spatial-error/lag alternative only if assumptions and estimator diagnostics are supported, and geographically weighted regression as a local association surface. Start with a coefficient plot and observed-versus-fitted/residual chart, then map residuals and coefficient uncertainty. Add residual autocorrelation or held-out spatial-block error as a model check.

Acceptance must distinguish coefficient association from causal effect; report collinearity, mask/finite-row policy, model convergence, spatial weights, bandwidth/kernel, multiple-testing or local-inference limits, and failed fits. Do not map only coefficient signs. For local models show standard errors or another reliability measure and avoid presenting unstable coefficients in low-information neighborhoods as real local effects.

### Equity: accessibility distributions and group comparisons

Build on a network accessibility story with explicitly chosen origins, destinations, impedance, service capacity, and population groups. Pair the map with a distribution (median and tails) and a group comparison (dumbbell/forest) plus a Lorenz or concentration curve only when its ordering and population weights are defensible. A scenario could add/remove a facility or change a travel threshold while holding demand and network fixed.

Acceptance must document missing network edges, barriers, time-of-day assumptions, facility capacity, group boundaries and denominator. Report both absolute access and distributional difference; do not equate geographic proximity with realized service or equity. Suppress small groups only under a published rule and show what was suppressed.

### Raster change: observed snapshots, compositing, and uncertainty

Use a date-stamped, cloud-screened raster pair or short time series (for example NDVI over the Dixie burn scar) with the same grid, mask, and processing. Map per-pixel difference and a small-multiple time series, pair with a sampled-pixel chart, and show the valid-observation mask. Allow the reader to compare a selected place's trajectory against a reference zone.

Acceptance must expose pixel size, CRS, date windows, compositing method, cloud/no-data mask, scale factor, and direction of the difference. Separate change in the measured index from land-cover change or ecological recovery. Register the grids and ensure classifications use shared breaks; report the number of valid pixels per period and do not silently fill gaps.

### Networks and movement: reach, exposure, and observed paths

Use one well-attributed network or trajectory set. Candidate narratives include network versus straight-line access, route detours around barriers, or observed movement corridors compared with a modelled resistance surface. Use a map with selectable origins/paths, a distance or travel-time distribution, and a timeline only when timestamps are meaningful. Keep observed trips/tracks visually distinct from inferred routes and model corridors.

Acceptance must state network completeness, edge costs, mode, time dependence, snapping policy, trajectory sampling interval, GPS error, and selection coverage. Show unreachable nodes/segments and truncated tracks. Do not infer population-wide mobility from a convenience sample, and do not describe a least-cost route as a route someone traveled.

## Acceptance criteria for each in-progress story

### Editorial and evidence

- The title is a question or finding in ordinary language; summary identifies algorithm, spatial/time support, dataset scope, and the largest limitation.
- The opening step says what is observed and what is derived. Every map has an explicit unit-bearing legend, a no-data meaning, and an accessible text alternative.
- Every method step names the consequential assumptions (distance/weights, denominator, thresholds, model scenario, or resolution) beside the result.
- A later step checks the result with an appropriate reference, alternative, null, or held-out evidence. It explains the scope of the check and the minimum detectable probability or error floor where applicable.
- The final step states at least one inference the scene cannot support and points to a related question without implying shared support or causality.

### Cartography and charts

- Follow `CARTOGRAPHY-GUIDE.md`: use ground-aware colors, stable class tables where comparisons require them, correct sequential/diverging/categorical semantics, no-data distinct from zero, and a second cue for color-ambiguous categories.
- Breaks, units, data extent, class method, and any cap/clip are discoverable. A meaningful zero or null class is not hidden in “no data.”
- Charts use an appropriate kind, label axes and units, include reference lines or envelopes needed to interpret them, expose values in a table, and describe the chart for assistive technology.
- At least one high-value map-to-chart or chart-to-map interaction is provided. Interaction has a non-pointer route and does not change the underlying analysis silently.
- Direct side-by-side or swipe comparisons preserve the same spatial support and visual scale, or the UI explains why they do not.

### Scientific and implementation evidence

- A deterministic CPU/reference fixture or independent invariant verifies the statistic's important semantics, including no-data, zero, tie, boundary, and capacity/iteration behavior relevant to the scene.
- The story identifies source versions/date ranges, attribution/license, projection/units, exclusions, row/pixel counts, and any preprocessing that affects results.
- Parameter changes described as visual update do not rebuild analysis; controls that rebuild are labeled. Rebuild failure/pending states cannot be mistaken for current results.
- For iterative analyses, status is explicit. For validation, tuning uses training data only and repeat/split uncertainty is described. For model outputs, scenario assumptions are enumerable and sensitivity is shown when dominant.
- Review smoke pass on a desktop and narrow viewport; verify keyboard focus, reduced motion, labels/legends at both grounds, and that the default scene state tells a coherent story without touching controls.

## Verification notes

This is a design document only. No runtime code, tests, builds, or formatting were changed or run for this review. Before claiming any in-progress scene is complete, follow the repository's requested `yarn lint fix`, `yarn build`, `yarn test`, and relevant website/example checks from the root `AGENTS.md`; design acceptance does not substitute for numerical validation or those project gates.

## Focused implementation review notes

These are **static code-review observations** from the sprint tree, recorded separately from the design recommendations above. They do not assert browser or GPU verification. Authors reported focused scene/type/story checks for some fixes; the parent task owns final integration and browser checks.

- **Rainfall blocked validation:** the scene uses the same azimuthal-equidistant coordinates for 250 km block IDs and interpolation distances, with deterministic assignment of whole blocks to one of five folds. The selected fold's training mask is used by the variogram and interpolators, and asynchronous validation readback is guarded by split version. The story correctly calls this a one-fold spatial stress test and warns that adjacent blocks can still be close. No remaining mismatch was found in the reviewed path.
- **Chicago facility placement:** static review found that queued revision IDs could outnumber readback copies after repeated capacity changes during an in-flight read, causing the newest output to be discarded while map buffers had advanced. The author replaced the queue with coalesced in-flight/latest revision tracking. The mean-distance comparison was also revised to use the same population support for both solutions. These changes were confirmed by source inspection only.
- **Dixie habitat connectivity:** review found a unit mismatch where `edgeDepth: cellSize` passed 20 into an integer-cell erosion mode, plus a difference between the selectable square erosion map and the recipe's Manhattan core source. The author changed the fixed cost-source depth to one cell, added an overlay for the exact GPU source mask, and labelled the square-inset map as a separate sensitivity view. Review also found a one-based component-label palette offset; legend swatches were rotated to match the category shader. These corrections were confirmed by source inspection only.
- **Watershed reaches:** endpoint markers were initially attached to the stream-order branch rather than the new network product. The author moved them to the network branch; source inspection confirmed the correction.
- **Election ranks:** copy initially described supported FIPS rows as contiguous states. The author changed it to “supported states and DC” and made county-derived aggregation explicit; source inspection confirmed the correction.
- **Chicago street enclosure:** invalid observer moves preserve the last valid map marker and report the rejected request in status. The profile readout and two-sided first-hit overlay are generated from the GPU profile output. No remaining mismatch was found in the reviewed path.
- **Nature outliers and facility/rainfall story contracts:** focused static reviews found no remaining concrete story/control mismatch at the time reviewed. No browser or GPU validation is claimed here.
- **King County LiDAR ground / height:** the source path confirms that the first surface pass takes the per-cell minimum only from returns already labelled LAS class 2, and sets supported-ground coverage only for those cells. The PMF uses that surface and coverage to test candidate points; cells without an initial class-2 return cannot become supported ground in this implementation. A second surface pass uses the PMF-updated classifications, and the height contributor reports explicit outside/invalid, unsupported-terrain, and non-finite statuses while preserving valid negative heights. The scene overlays nonzero height statuses in gold with wording that includes unsupported, outside-grid, and invalid values. Feature status is computed from counts and covariance directly: over-capacity counts set truncation, fewer than three or degenerate support sets insufficiency, and the eigensolve failure sets its own flag. Review found and corrected two copy mismatches: rejected ground points also include cells without seed support, and the code-panel voxel dimensions now match the compute grid. This is code-path review only; no GPU or browser result is claimed.
