# Card picture re-score (2026-11-10)

Roadmap item WEB-PIC-12 ([../../roadmap.md](../../roadmap.md#76-open-picture-items-web-pic)). This is an independent score of all 428 operation runs against the 1 to 5 picture-quality scale of the [2026-11-09 audit](../card-interaction-audit-2026-11-09/README.md). Until now each group had been reviewed only by the agent that built it.

> **State described.** The scores describe the pictures from the final verification of **2026-11-09** (`website/.reports/final/`). While this review was written, three agents were fixing weak gestures and inert sliders (WEB-PIC-13, WEB-PIC-14) in the same checkout. Some runs below may already have changed. Check against a fresh `verify-runs.mjs` contact sheet before acting on a single entry.

## Files

| File | What |
| --- | --- |
| `README.md` | Method, summary by group, the 40 worst pictures, suspected operation defects, limits |
| [scores.json](./scores.json) | One record per run: `id`, `group`, `rest` (1 to 5), `gesture` (1 to 5), `note` |

## Scale

The audit's scale is used unchanged. 1: the picture does not explain the operation (the subject is missing, input and output cannot be told apart, or the output is empty). 2: the operation can be guessed only from the caption or readout, because the subject is faint, tiny, cluttered, mis-encoded or the same as a sibling's. 3: legible and honest, with one clear weakness. 4: explains the operation at a glance, with a minor weakness. 5: nothing to improve at card size. No run scored 5.

Each run gets two scores:

- **rest**: the picture at rest (the poster frame). Does it explain the operation, is the subject framed, can input and output be told apart. This is the score that compares with the audit's single `pictureQuality`.
- **gesture**: does the hand do what a reader expects for this operation (roadmap 7.3), and does the picture change meaningfully across the pointer positions (7.4). A gesture that only moves a cursor glyph or outline, with the result unchanged, gets 2. A gesture with two or fewer distinct pictures, or whose readout never changes, gets 1 or 2.

Every 1 or 2 has a note saying what is wrong and what would fix it. Notes on 3 and 4 name the main weakness or strength.

## Method

1. Read the audit README, the group audit files (their `pictureQuality` per run is the earlier score) and roadmap 7.3 and 7.4.
2. For each of the twelve groups, looked at the rest sheet `website/.reports/final/<group>.png` (dark theme, one cell per run) and then every contact sheet `<group>-pointer-<n>.png` (63 sheets). Each contact-sheet row shows three of the five verifier positions with the readout under each frame, plus the verifier's header (gesture kind and hint, distinct pictures out of five, largest pixel change along x and y).
3. Used `<group>.json` for the per-run numbers: distinct pictures, per-position pixel change, non-background share and slider changes (an inert slider is one whose change is under 0.2%).
4. Opened run source (`website/src/gpu/runs/entries/<group>/<id>.ts`) for five runs where a readout looked like a defect, to separate a defect in the operation from a picture or gesture choice (`parameter-buffer`, `pointcloud-height`, `seasonal-break-detection`, `weighted-overlay`, `cell-grid-path`). Three of the five turned out to be picture issues and are scored as such.
5. Scored hard: a correct but faint, cluttered or duplicated picture gets 2. "The readout says it" does not lift a picture above 2.

No run was skipped. Every run was seen on its rest sheet and its contact-sheet row; no poster `.webp` was opened, because every contact-sheet cell was large enough to judge (see Limits).

## Summary

Totals over 428 runs:

| Score | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Rest | 6 | 143 | 239 | 40 | 0 |
| Gesture | 5 | 115 | 244 | 64 | 0 |

- **Rest at 1 or 2: 149**, against **177** in the audit before the rework. Of the audit's 177, 96 are still at 1 or 2 and 81 now score 3 or more. 53 pictures that the audit scored 3 or 4 score 2 here, mostly for faint subjects, duplicated siblings and empty frames that the audit did not penalise.
- **Gesture at 1 or 2: 120.** The verifier's "at least three distinct pictures" (424 of 428 pass) does not mean the gesture is meaningful: many probes change only an outline or a spotlight, and many readouts stay the same at every position.
- **Rest or gesture at 1 or 2: 202.** Both at 1 or 2: 67.

By group (counts for scores 1 to 5; "≤2" is the number at 1 or 2):

| Group | Runs | Rest 1/2/3/4/5 | Gesture 1/2/3/4/5 | Rest ≤2 | Gesture ≤2 | Either ≤2 | Audit ≤2 (before) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cells-index-join | 31 | 0/2/20/9/0 | 0/5/16/10/0 | 2 | 5 | 6 | 7 |
| dataframe-pointcloud-records | 44 | 2/18/22/2/0 | 1/12/26/5/0 | 20 | 13 | 24 | 21 |
| density-interpolation-regions | 38 | 0/7/26/5/0 | 0/6/25/7/0 | 7 | 6 | 9 | 14 |
| geometry-construction-overlay | 36 | 0/11/24/1/0 | 2/17/14/3/0 | 11 | 19 | 22 | 5 |
| geometry-processing | 36 | 0/10/24/2/0 | 0/15/16/5/0 | 10 | 15 | 19 | 11 |
| network-routing-access | 39 | 0/13/23/3/0 | 0/12/21/6/0 | 13 | 12 | 18 | 17 |
| raster-bands-indices | 43 | 2/13/25/3/0 | 0/7/31/5/0 | 15 | 7 | 18 | 31 |
| recipes-trajectories | 42 | 0/16/25/1/0 | 0/11/29/2/0 | 16 | 11 | 21 | 17 |
| regression-weights | 38 | 1/28/9/0/0 | 1/14/23/0/0 | 29 | 15 | 33 | 24 |
| spatial-statistics | 27 | 1/17/9/0/0 | 1/11/15/0/0 | 18 | 12 | 22 | 17 |
| terrain-hydrology-flow | 23 | 0/4/12/7/0 | 0/2/13/8/0 | 4 | 2 | 5 | 7 |
| terrain-relief-light | 31 | 0/4/20/7/0 | 0/3/15/13/0 | 4 | 3 | 5 | 6 |
| **Total** | **428** | **6/143/239/40/0** | **5/115/244/64/0** | **149** | **120** | **202** | **177** |

What the groups have in common:

- **Strong:** the two terrain groups, `cells-index-join` and most raster pictures. Place, aim, level and reach gestures on a full-frame raster work well (`viewshed`, `slope-aspect-hillshade`, `isolines`, `network-isochrones`, `walking-cost`, `terrain-watersheds`).
- **Weakest: `regression-weights` (29 of 38 rest at 2 or less) and `spatial-statistics` (18 of 27).** Eleven regression runs share one picture (a dim residual map and unlabelled coefficient bars) and look alike. The weights runs draw faint graphs on black. The statistics runs are mostly dim point clouds where significance is not encoded at rest. These pictures need text (WEB-INT-16) or a chart form that explains itself without labels (Moran scatter, permutation histogram with the observed value marked).
- **Geometry gestures are weak (34 of 72 at 2 or less).** Dragging one vertex of many shapes, or reaching near a line, often changes under 1% of pixels, and many readouts do not change (`line-noding`, `polygonize`, `segment-intersection`, `geometry-orientation`). The vertex-level output (inserted vertices, nodes, clearance, start vertex) is drawn too small to see at 320 px.
- **"Probe lands on nothing."** At least 14 probes read "(no readout)", "no building hovered", "0 neighbors" or an off-data value at one or more test positions (the GWR and spatial-regression probes, `knox-test`, `mantel-test`, `lattice-weights`, `coverage-dissolve`, `street-profiles-recipe`, `building-adjacency-recipe`). Snapping the probe to the nearest element would fix all of them.
- **Siblings still alike (WEB-PIC-15):** `point-to-cell` and `cell-geometry`; `polygon-triangulation` and `constrained-delaunay-triangulation`; `relief-shading` and `slope-aspect-hillshade`; `isolines` and `terrain-contours`; `enhanced-vegetation-index` and `soil-adjusted-vegetation-index`; `variogram` and `cross-variogram`; the three Ripley runs; the three panel regressions; the encounter and zone-event trios; `fractional-zonal-statistics` and `raster-zonal-statistics`; `group-statistics` and `zonal-statistics`.

## The 40 worst pictures

Ranked by rest plus gesture, then by the lower of the two, then by the share of the frame that is drawn (emptier first). Below the first 11, every run scored 2 for both, so the order inside that tier is only by how empty the frame is.

| # | Run | Group | Rest | Gesture | What is wrong, and the fix |
| --- | --- | --- | --- | --- | --- |
| 1 | `pair-gather` | regression-weights | 1 | 2 | 1% non-background: scattered faint segments and dots that do not show two related point sets (WEB-PIC-09). Fix: two real related point sets and bold gathered pairs. |
| 2 | `parameter-buffer` | geometry-construction-overlay | 2 | 1 | Shows a line-simplified shoreline whose tolerance lives in a GPUParameterBuffer; nothing in the picture says the value is written per frame without recompiling. The reach saturates at the 0.1 maximum at every test position ("tolerance 0.100 · kept 9" throughout) because the pointer is never within 0.1 of the shore. Fix: map the reach logarithmically from the shore so the full range is reachable, and show a frame counter against a "compiled once" badge. |
| 3 | `nested-interpolation-validation` | spatial-statistics | 2 | 1 | Faint dots in two panels with no fold boundaries; the readout is "outer fold 1/3 · epsilon 2 · RMSE 0.13" at every position and no station lights. Fix: colour stations by fold and light the hovered station's fold and error. |
| 4 | `raster-registration` | raster-bands-indices | 1 | 2 | An empty grid with a dot; neither raster is shown (5% non-background). Fix: overlay the two rasters with the true and recovered shift as arrows. |
| 5 | `spatial-pearson` | spatial-statistics | 1 | 2 | A lone bar with a whisker under two point maps that look the same; the curtain changes 2.6% and the readout is constant. Fix: a scatter of x against its spatial lag of y with the coefficient line. |
| 6 | `pointcloud-voxel-merge` | dataframe-pointcloud-records | 1 | 2 | Sparse dots on black; tiles, voxels and the merge are invisible and the seam drag changes 3%. Fix: colour points by source tile, draw voxel cells and highlight shared voxels on the seam. |
| 7 | `pointcloud-statistical-filter` | dataframe-pointcloud-records | 2 | 1 | Rejected points are a few red dots among orange; the level ticks move but the picture has 2 distinct states (threshold settles, x 0%). Fix: per-frame threshold on the computed distances and rejected points drawn large and bright. |
| 8 | `pointcloud-icp` | dataframe-pointcloud-records | 1 | 2 | Uniform speckle where source and target clouds cannot be told apart; only a small RMSE curve changes. Fix: two contrasting colours, fewer points, correspondence lines and the aligned result. |
| 9 | `multiscale-gwr` | regression-weights | 2 | 1 | Two flat coefficient maps (one all pink, one all dark); the pad gives 2 distinct pictures and interceptBandwidth is inert. Fix: a scene with spatially varying coefficients and an intercept surface. |
| 10 | `spectral-unmixing` | raster-bands-indices | 1 | 2 | A dim purple field with a spotlight; the water, soil and vegetation fractions are not mapped and the bars are unlabelled grey. Fix: an RGB composite of the three fractions with a three-colour key. |
| 11 | `group-concave-hull` | geometry-construction-overlay | 3 | 1 | Group hulls read; carrying one point changes nothing visible (2 distinct, 0.2%). Fix: carry a whole group or drive the circumradius by reach. |
| 12 | `line-segmentize` | geometry-construction-overlay | 2 | 2 | Inserted vertices are tiny dots that cannot be seen at card size; the picture changes 0.2%. Fix: draw original vertices large and inserted ones in a second colour. |
| 13 | `line-loop-removal` | geometry-processing | 2 | 2 | Faint input with small loops under a white output; the readout stays "240 -> 129" at every position. Fix: draw removed loops in red and let the drag create a loop. |
| 14 | `outline-geometry` | geometry-construction-overlay | 2 | 2 | Blue discs with no visible input points; the reach returns the same 0.100 at two of three positions. Fix: draw the input points and the outline edge distinctly and widen the range. |
| 15 | `multiple-testing-adjustment` | spatial-statistics | 2 | 2 | Bare p-value curve with dots pinned at the top; every method gives "0 significant". Fix: data with some significant tests and a labelled alpha line. |
| 16 | `normalize-geometry` | geometry-processing | 2 | 2 | Tiny glyphs of mixed types; hover draws a dashed link from input to canonical (1.4%). Fix: two aligned columns, input and canonical, with the hovered pair enlarged. |
| 17 | `point-pattern-envelope` | density-interpolation-regions | 2 | 2 | Bare curves with no shaded envelope and no points map; only the rule moves (0.9%). Fix: shade the envelope band, mark where observed leaves it, show the pattern. |
| 18 | `spatial-interaction` | spatial-statistics | 2 | 2 | Sparse abstract scatter (2% non-background); the readout gives fitted equal to observed at every tract. Fix: draw flows on the map with width by fitted value and the residual. |
| 19 | `spatial-balanced-sample` | density-interpolation-regions | 2 | 2 | Sample points and population are both small yellow dots; moving the split changes nothing visible ("west 60 east 100" at every split). Fix: colour the two strata and sampled points distinctly, and recount per stratum as the split moves. |
| 20 | `clock-encounters` | recipes-trajectories | 2 | 2 | Faint tracks with tiny encounter marks and a histogram; the pad changes 4.3% and the scene matches its two encounter siblings. Fix: draw met pairs as bright links at the current clock step and give the trio distinct scenes (WEB-PIC-15). |
| 21 | `normalize-polygons` | geometry-processing | 2 | 2 | Start-vertex dots are barely visible; the aim moves the input start and the readout says "canonical start fixed" every time. Fix: draw the input and canonical start vertices large in two colours. |
| 22 | `trajectory-encounters` | recipes-trajectories | 2 | 2 | Faint tracks; the reach gives 0.120 at two positions and changes 3.6%; sibling of clock-encounters. Fix: draw met pairs as links and widen the distance range. |
| 23 | `spatial-weights-algebra` | regression-weights | 2 | 2 | Faint graph; the readout stays "kNN 3, band 1 -> 1" at every position. Fix: as chunked-weights-algebra. |
| 24 | `linear-referencing` | geometry-construction-overlay | 2 | 2 | Faint roads with tiny measure ticks; moving the query point changes 0.4% (3 distinct). Fix: draw the projection from the query point to the road and the measure along it. |
| 25 | `coverage-validity` | geometry-construction-overlay | 2 | 2 | Grey coverage with a few orange invalid edges; gap stays 0.012 and the picture changes 0.3%. Fix: mark the gaps and overlaps as filled red slivers and let the reach set gapWidth across them. |
| 26 | `neighborhood-summary` | regression-weights | 2 | 2 | Two near-identical dot maps; hover changes under 2% (x 0%). Fix: one map with the hovered point's neighbourhood mean drawn against its own value. |
| 27 | `trajectory-closest-approach` | recipes-trajectories | 2 | 2 | Faint tracks and bars; hovering lights one pair, readout stays near "400 s within 0.05" (2%). Fix: draw the closest-approach segment between the pair and the time on a strip. |
| 28 | `geometry-orientation` | geometry-processing | 2 | 2 | Thin rings with tiny direction arrows; the curtain shows "14 rings reversed" at every position. Fix: draw large winding arrows and colour clockwise against counter-clockwise. |
| 29 | `segment-ring-assembly` | geometry-construction-overlay | 2 | 2 | Assembled rings and input segments are both thin blue lines; hovering lights one joint (1.8%). Fix: colour each assembled ring and show the dangling input segments distinctly. |
| 30 | `track-quality` | network-routing-access | 2 | 2 | Spaghetti of fixes where flagged fixes are not distinct; "flagged 70" at every position. Fix: draw flagged jumps in red and let the drag create a speed outlier. |
| 31 | `chunked-weights-transpose` | regression-weights | 2 | 2 | Faint graph; the out and in arrows of the hovered point bunch into a tiny cluster. Fix: longer arrows in two colours and a dimmer background graph. |
| 32 | `building-adjacency-recipe` | geometry-construction-overlay | 2 | 2 | Building outlines only, in a strip at the top; no adjacency or terrace link is drawn, and two of three positions read "no building hovered". Fix: draw shared walls as links, fit the block to the frame and snap the probe to the nearest building. |
| 33 | `knox-test` | spatial-statistics | 2 | 2 | A narrow band of events at the top and a histogram; two of three positions hover off the events ("0 of 1 within"). Fix: fit events to the frame and snap to the nearest event. |
| 34 | `lon-lat-polygon-preparation` | geometry-processing | 2 | 2 | Abstract date-line diagram; the dragged polygon reads "whole" at every position and never shows a split. Fix: rest across the antimeridian and draw the split pieces. |
| 35 | `network-kernel-density` | network-routing-access | 2 | 2 | Dim purple edges where density is hard to read; dragging an event changes 1.4%. Fix: brighter ramp, thicker edges, and the event's kernel footprint lit along the network. |
| 36 | `mantel-test` | spatial-statistics | 2 | 2 | Same scene as knox-test; two of three positions hover off the events (1 and 0 neighbours). Fix: as knox-test, and plot space against time distances. |
| 37 | `spatial-cross-validated-ols` | spatial-statistics | 2 | 2 | Points and unlabelled fold bars; probing gives fold 1/4 at two positions (3 distinct). Fix: draw test, buffer and training points in three clear colours for the hovered fold. |
| 38 | `catchment-accessibility` | network-routing-access | 2 | 2 | Demand dots and one dashed arc; the catchment itself is not filled and dragging changes 1.7%. Fix: colour network edges or demand by accessibility within the catchment. |
| 39 | `global-spatial-statistics` | spatial-statistics | 2 | 2 | Points with three unlabelled bars; hover draws neighbour links while Moran's I stays the same (0.6%). Fix: a Moran scatter plot with the hovered point and its lag. |
| 40 | `network-snapping` | network-routing-access | 2 | 2 | Snap links are invisible at card size among many points; the reach gives 0.050, 0.044, 0.050 (80% snapped). Fix: fewer points, bold snap links, and a wider distance range. |

The other 28 runs scored 2 for both: `adjacency-matrix`, `hot-spot-analysis-recipe`, `map-matching`, `color-scale`, `pointcloud-neighborhoods`, `minimum-clearance`, `spatial-weights-traces`, `transition-diagnostics`, `pointcloud-height`, `polygon-triangulation`, `cell-geometry`, `rank-markov`, `snapshot-tap`, `computation-record`, `seemingly-unrelated-regression`, `terrain-critical-points`, `spatial-regression-diagnostics`, `spatial-error-maximum-likelihood`, `gwr-nonstationarity-test`, `probit-regression`, `computation-metrics`, `segregation-counterfactual-recipe`, `terrain-dem-coregistration`, `glcm-texture`, `dasymetric-redistribution-recipe`, `network-cost-matrix`, `extrema-pyramid`, `spike-repair`.

## Suspected operation defects

These look like problems in the operation or its inputs, not in the picture. None was confirmed by running anything; each needs a check against the operation's tests or source. Items already in WEB-PIC-17 are marked.

| Run | What was seen | Why it may be a defect |
| --- | --- | --- |
| `contiguity-weights`, `fuzzy-contiguity` | Orange weight links cross the whole Chicago map between tracts that do not touch; `fuzzy-contiguity` has even more | Already in WEB-PIC-17: the default `snapTolerance` of 0.03 world units looks too generous |
| `cell-grid-path` | Readout "paths up to 18 cells (max 14)" and "15 cells (max 14)" | A path longer than `maximumPathLength`. Either the operation does not enforce the limit, the readout counts end cells differently, or the readout shows the new value against outputs of the previous (settled) build. The run rebuilds on `maximumPathLength` |
| `network-point-routing` | At one position (source road 1 at 0.45) the readout says "max cost 0.01" and the whole network draws white (unreached) | The source may snap to a disconnected road, or the cost field may not be written for that source |
| `terrain-dem-coregistration` | A blue disc appears on the corrected side of the curtain at one position and not at the others; the readout ("fitted shift 0.049 · true bias 0.050") never changes | Either a residual left after correction drawn only at some curtain positions, or a stale texture |
| `multiscale-gwr` | Both coefficient maps are flat (one uniform pink, one uniform dark) at every pad position; `interceptBandwidth` is inert | Could be the scenario (no spatial variation in the generated coefficients) rather than the operation; already listed in WEB-PIC-13 and WEB-PIC-14 |
| `nested-interpolation-validation` | Readout "outer fold 1/3 · epsilon 2 · RMSE 0.13" at every probe position and no station is lit | The probe does not seem to find the hovered station (the run's `resolve` reads a hovered fold) |
| `spatial-weights-traces` | Readout "tr(W^q) 0.00 · sum 0" for q = 11 and 16 | For a row-standardized W, tr(W^q) is at least about 1 for even q (eigenvalue 1). If the readout is divided by n, 0.00 is a rounding issue; "sum 0" still needs a look |
| `zonal-distribution-recipe` | Grey zones read "q = no points" | Already in WEB-PIC-17 (quad zones without points give a NaN quantile); shown as grey, not as holes |
| `visibility-graph-recipe` | Not visible at card size | Already in WEB-PIC-17 (`normalizedChoice` NaN for some nodes); not seen here |

Checked in the source and **not** defects (scored as picture or gesture issues): `parameter-buffer` (the reach saturates at the 0.1 maximum because the test positions are far from the shore), `pointcloud-height` ("0 m" because the probe lands on ground points), `seasonal-break-detection` (every cell breaks at month 20 because the modeled series share one break), `weighted-overlay` (the green layer carries the picture when NIR and SWIR weights are 0). `spatial-interaction` shows fitted equal to observed at every tract, which is expected for a production-constrained model at the origin; it is listed as a picture issue. `multiple-testing-adjustment` gives "0 significant" for every method, which is a data choice.

## Limits of this review

- **Three of five positions.** Each contact-sheet row shows three of the five verifier positions. The two others were judged only from the verifier's numbers (distinct count, pixel change).
- **Dark theme only.** The rest sheets and contact sheets are dark. The light posters (`website/public/runs/<id>-light.webp`) were not opened, so light-theme contrast is not scored. No poster was opened at all: every contact-sheet cell was large enough to judge, and the rest sheet is the poster frame.
- **Not run.** No scene was run. Glide-in, latency, keyboard and touch (roadmap 7.4) are not scored. "Changes meaningfully" was judged from still frames, not from motion.
- **One reviewer.** These are one reviewer's judgments at card size; the scale is coarse and 2 against 3 is often close. Notes say why, so a disagreement can be argued per run.
- **Rest compared with the audit's single score.** The audit gave one `pictureQuality` per run, mostly about the picture at rest; it is compared here with the rest score only. The gesture score has no earlier counterpart.
- **State.** The pictures are from 2026-11-09. Other agents were changing gestures and sliders while this was written (see the note at the top).
- **Out of scope.** The ten flagship demos and the eight new groups another session is building.
