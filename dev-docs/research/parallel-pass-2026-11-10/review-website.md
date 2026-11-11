# Review: run changes since 74dddcf5 (12 groups) and the text layer

Reviewer: independent, read-only. Date 2026-11-10.

Method: read every diff in the twelve groups and all of `runs/text/` plus `text-wiring.diff`. Compared
the before posters (`.reports/final/`, 2026-11-09) with the agents' after screenshots (`.reports/w1-*`,
2026-11-10) side by side, at rest and at pointer position 1. Checked operation sources where a prop's
meaning was in doubt. Re-ran `verify-runs.mjs` on five runs (constrained-delaunay-triangulation,
spherical-polygon-contains, pointcloud-ground, pair-gather, terrain-critical-points) into
`website/.reports/review-w1/`. All five pass, with `restDifference 0`. Every changed slider range
still contains its default.

Severity: **H** misrepresents the operation, or is a licence or correctness defect a reader would
hit. **M** makes the picture wrong or weaker. **L** is cleanup.

---

## 1. Findings, most serious first

### H1. `dataframe-pointcloud-records/pointcloud-smrf.ts`: the constructed tilt ruins the poster and shows SMRF failing at its defaults
- **What is wrong.** A constant tilt of `0.17 * x` is added to every return. The left half draws the
  inpainted ground surface by height, so it is now a smooth left-to-right colour gradient. The real
  relief and the hillshade detail that were visible before are gone; compare
  `final/.../pointcloud-smrf-dark.png` with `w1-sm/...`. The default `slope` is 0.15, below the added
  tilt of 0.17. At rest the right half therefore scatters blue (removed) ground points over the whole
  uphill half. The poster now shows SMRF misclassifying ground at the run's own defaults.
- **Why it matters.** The card's first impression is "SMRF removes hillside ground", and the left
  panel no longer shows a surface. The caption mentions the tilt but not that the defaults cannot cope
  with it.
- **What to do: revert the tilt in this run.** If a slope scenario is kept, colour the surface by
  height above the tilt plane and set the default `slope` at or above the tilt. The real fix is the
  WEB-PIC-09 tile with intermediate relief.

### H2. `terrain-relief-light/spike-repair.ts`: undisclosed smoothing of the real DEM, and a false caption
- **What is wrong.** `smoothElevation` runs six 3x3 box passes over the measured tile, which wipes out
  its ridges and valleys (`w1-terrain-relief-light/.../spike-repair-dark.png` against
  `final/`). The caption does not mention the smoothing. It says "Seams are a step plus up to 30 m of
  **real ground change**", but the ±30 m is `random()` noise added to the block offset. Nothing about
  it is real.
- **Why it matters.** The operation is only demonstrated on terrain that has been made easy for it.
  If it mis-detects seams on real relief, that is a property of the operation to show or report, not
  to hide. The caption states something false.
- **What to do: revert the smoothing.** Keep the per-block seam deviation and caption it truthfully,
  for example "Scenario: damaged blocks are offset by one step ±30 m". If real relief then breaks the
  repair, record it as an operation finding (WEB-PIC-17) and caption it.

### H3. `runs/text/font-atlas.ts`: licence header misstates a derived font
- **What is wrong.** The file is a rendered signed-distance atlas of KaTeX_SansSerif and KaTeX_Main
  glyphs, which are SIL OFL 1.1 font software. Its header is `SPDX-License-Identifier: MIT` /
  `Copyright (c) vis.gl contributors`. AGENTS.md: "do not infer MIT licensing or ownership for
  generated, vendored or third-party files". KaTeX is not in `THIRD-PARTY-DATA.md`, and the generator
  reads `node_modules/katex`, which is not a declared dependency (it is only "installed for the docs").
- **What to do, before wiring.** Give the generated file `OFL-1.1` with the KaTeX font copyright
  holders, or a dual expression that names the OFL for the glyph data. Add the KaTeX fonts entry to
  `THIRD-PARTY-DATA.md`, as roadmap WEB-INT-16 already asks. Declare `katex` as a devDependency of the
  workspace that runs the generator.

### H4. `pointcloud-ground.parameters.ts` and `pointcloud-ground.ts`: the operation silently receives a different `maximumThreshold` than the slider shows
- **What is wrong.** The `maximumThreshold` range moved to 0.02–0.3 (default 0.3). `initialThreshold`
  still goes to 0.8, and the gesture's `THRESHOLDS` include 0.6. `build` passes
  `Math.max(values.maximumThreshold, values.initialThreshold)`. Whenever `initialThreshold > 0.3`
  (two of the five gesture levels, and a third of the slider), the slider reads 0.3 while the
  operation gets 0.6. Both `slope` and `maximumThreshold` are then inert. Measured in this review:
  across the gesture the ground count moves only from 7667 to 7711 of 9000. The sliders still change
  1.1–1.5% of the picture, so the tilt barely made them act.
- **What to do.** Keep the ranges consistent. Cap the gesture's threshold levels at the
  `maximumThreshold` range, or clamp `initialThreshold` down rather than raising the maximum, and put
  the effective value in the readout. Given how little the tilt buys here, revert it as in H1 and leave
  WEB-PIC-09 open.

### H5. `geometry-construction-overlay/coverage-validity.ts`: `gapWidth` is a width, but the pointer now sets it to a radius
- **What is wrong.** `gapWidth` is now the pointer's distance to the nearest edge, so it equals the
  radius of the ring that is drawn. Before, it was twice the distance: the diameter of a disc touching
  the edge. A gap is flagged when it is narrower than `gapWidth`, but the disc the reader sees is
  `2 * gapWidth` across.
- **Why it matters.** The glyph misstates the prop's unit by a factor of two. The pixel change rose
  (0.6% to 3.0%) because the mapping doubled the sensitivity, not because the picture explains more.
- **What to do.** Revert to the diameter mapping (`gapWidth = 2 * distance`, ring radius = distance).
  If more sensitivity is wanted, keep the seven graded defects; they are disclosed through `Data: gap`
  and fine.

### H6. `terrain-relief-light/terrain-critical-points.ts`: the rest poster is now nearly black
- **What is wrong.** `drawClassDimmer` now paints every cell outside the picked class in the
  background colour at alpha 0.85. At rest only one class (one colour of cells) remains, and the
  elevation context and the other two classes disappear. See
  `review-w1/terrain-relief-light/terrain-critical-points-dark.png`.
- **Why it matters.** The poster no longer shows what the operation computes, a classification of
  every cell. The 7.6% gesture number comes from blacking out the picture.
- **What to do.** Keep the saddle penalty and the 5-cell search, which are good. Lower the dimmer to
  about 0.5–0.6 so the elevation and the other classes stay readable, or dim only when the pointer
  has moved off rest.

### M1. `regression-weights/multiscale-gwr.ts` and its parameters: the "constant intercept" map is not constant at rest
- At the default `interceptBandwidth` of 0.45 the left map shows a west-to-east drift that has leaked
  in from the drifting slope (`w1-regression-weights/.../multiscale-gwr-dark.png`). The caption says
  "constant intercept". MGWR's lesson is that a constant coefficient is best fitted at a large
  bandwidth, so the poster should show that.
- Set the default `interceptBandwidth` to 1 (as `secondBandwidth` already is) and let the pad's y axis
  pull it down. The new minimum of 0.08 with a bisquare kernel is about one tract spacing on the
  narrower 0.32 map, so local three-coefficient fits may be ill-conditioned. Read `status.codes` and
  raise the minimum if rows fail. The three maps are unlabelled; they are a WEB-INT-16 candidate.
- Keep the third map.

### M2. `dataframe-pointcloud-records/variogram-model-fit.ts`: the readout counts pairs on the CPU although the GPU output has the count
- `found.total` comes from an O(n²) CPU scan (245k pairs per bin change) inside `resolve`. The run
  already allocates `pairCounts` from `GPUVariogram`.
- Read the count with `peek(outputs.pairCounts)[bin]`, which means adding it to `outputs`. Keep the
  CPU scan only to pick the 40 drawn pairs, and say "sample of N pairs" in the readout.
- Drawing the hovered bin's pairs on the map is a good, honest change. Keep it.

### M3. `cells-index-join/nearest-feature-join.ts` and `geometry-construction-overlay/linear-referencing.ts`: the radius sits exactly on the nearest feature
- `nearest-feature-join` rounds the distance to 0.01, sometimes downward, so the pointer's own query
  can lose its nearest station and its link flickers at the boundary. `linear-referencing` writes the
  raw distance, which is not snapped to the slider's 0.005 step, so the slider shows off-step values.
  Its `rest` search lands only near 0.08.
- Snap upward to the parameter step (`snapToParameter`, ceiling), so the nearest feature is always
  inside the ring. Both gestures drive the real `radius` prop. Keep them.

### M4. `dataframe-pointcloud-records/pointcloud-statistical-filter.ts`: the pointer settles on 15 compile-time combinations
- `settle: {standardDeviations, capacity}` gives 5 × 3 = 15 builds against `buildCache: 6`. Feel rule
  9 asks for at most five settled values, so revisits rebuild. The x axis also moves the gesture
  (1.3% to 1.4%) by almost nothing.
- Drop `capacity` from the gesture and leave it to the slider, or keep only two capacities with
  `buildCache` at 10 or more.

### M5. `cells-index-join/point-in-polygon-join.ts`: the probe lattice is honest but lost a readout
- The 45-probe lattice goes through the real join and shows zone boundaries well. Keep it.
- The readout no longer reports `featureCounts`, the operation's per-zone count, and
  `featureCounts` now includes up to 45 lattice rows. Restore "zone N · M points (excluding probes)".
- The comment above `interaction` still says "one more point in the join". Fix it.

### M6. `dataframe-pointcloud-records/datum-grid-transform.ts`: the exhaustion marker tests the wrong status
- `select(0, 1, status != 0)` marks every failure (no coverage, nodata, invalid) as "the iteration
  limit stopped it". Test `GPU_DATUM_GRID_TRANSFORM_STATUS.inverseExhausted`.
- The defaults (inverse direction, limit 3, tolerance 3e-9° ≈ 0.3 mm) are defensible. At limit 3 no
  marker is visible at rest, so the poster does not show the new feature; that is acceptable.
  Accept the defaults.

### M7. Defaults chosen to move pixels
- `spatial-statistics/spatial-pearson.parameters.ts`: `Data: noise` went from 0.25 to 0.5. The two
  swiped fields differ more and the correlation bar is weaker. This is the "default chosen only to
  move pixels" case. **Revert** to 0.25.
- `cells-index-join/point-to-cell.parameters.ts` and `cell-geometry.parameters.ts`: `resolution` went
  from 5 to 4. Fewer, larger cells read better at card size and show the decoded boundary.
  **Accept**, but note it was chosen for the metric.
- `geometry-processing/label-point.parameters.ts`: `refinementRounds` went from 3 to 1. The
  operation's own default is 6. It is acceptable as the start of a `time` gesture, but the caption
  ("at the widest part") is then only approximately true at rest. **Accept**, and add "after one
  refinement round" to the caption.
- `geometry-processing/offset-curve-components`: `distance` 0.09 lets round, mitre and bevel differ.
  **Accept.**
- `regression-weights/contiguity-weights` and `fuzzy-contiguity`: the new `gap` and tolerance
  defaults are justified by measured distances. **Accept.** WEB-PIC-17 says the GPU links were never
  counted. Compare the link count with the rook contiguity that `createTracts` returns before calling
  it resolved; the after poster of `fuzzy-contiguity` still shows long links.
- `regression-weights/island-attachment`: the band radius moved from 0.06 to 0.04, a `Data:` knob,
  and `reciprocal` still changes only 0.5%. **Accept**, and keep it listed as inert.
- `terrain-relief-light/point-horizon-visibility`: `maximumDistance` moved from 3600 to 20000 m, a
  fixed real prop, so the targets are within range. **Accept.**

### M8. Smaller correctness and honesty notes
- `recipes-trajectories/quantile-lisa.ts`: significance is now a GPU kernel over the operation's
  pseudo p-values, which is good. It imports `addKernel` from `../geometry-processing/shared/...`, a
  cross-group dependency; move it to a shared runs module.
- `raster-bands-indices/raster-sampling.ts`: the holes grew (threshold 0.68 to 0.6) and the points
  doubled. The holes are constructed no-data, but the caption does not say so. Add "no-data holes cut
  into the raster (scenario)".
- `network-routing-access/map-matching.ts`: the drag now pulls three fixes on each side. That is an
  honest pointer-owned input, but the hint still reads "drag a GPS fix"; say "drag a stretch of track".
- `terrain-hydrology-flow/terrain-depression-hierarchy.ts`: the new `default-curve` kernel rewrites
  `curve` on every run, so after a slider change the hovered curve flashes to the rest curve until
  the readback re-resolves. Skip the kernel when the pointer owns the chart, or write the rest curve
  only when there is no focus.
- `geometry-construction-overlay/shared/reach.ts`, `concave-hull`, `group-concave-hull` and
  `spatial-statistics/global-spatial-statistics`: the distance from an anchor is bucketed into five
  values of the compile-time prop, so the ring's radius is not the reach distance. This is honest
  (a real prop, settled, `buildCache: 5`), but it is a radial scrub rather than the README's "ring of
  that distance". **Accept**; label the ring with its value once text is wired. The `getReachRest`
  doc says "one unit to the right"; fix it.
- `constrained-delaunay-triangulation`: `level` over `maximumFlips` with a "k of 6 polygons converged"
  readout read back from the operation is a good change. Re-verified: rest reproduces the defaults.
  **Keep.**
- `regression-weights/probit-regression`, `spatial-cross-validated-ols`, `optics-clustering` and
  `terrain-relief-light/peak-snap` draw structure the operation computed (the iteration bands; the
  buffered-out stations and the fold's coefficients read from GPU `training`/`testing`/`coefficients`;
  the cluster members and their reachability valley; every survey point's search ring and move).
  **Keep.**
- `recipes-trajectories/drainage-network-recipe`: removing `Data: relief` is correct (uniform scaling
  cannot change D8 routing). **Keep.**
- `regression-weights/neighborhood-summary`: mirroring the links onto the second map is explanatory.
  **Keep.** It is still under 3%.
- The `raster-bands-indices` comment updates in `fill-holes`, `walking-cost` and
  `morphological-reconstruction` match the operations' current publishing contract. **Keep.**

### M9. Process note: the reported gains overstate the work
`web-g2.md` lists `facility-scenario`, `huff-trade-areas`, `catchment-accessibility`,
`network-accessibility`, `network-disruption`, `network-snapping`, `network-flow` and
`network-facility-optimization` under "Gestures before → after". Their diffs are size changes only,
and most stay under 3%. Only `alternative-routes` is labelled "thicker". The roadmap's WEB-PIC-14
counts should treat these as size-only.

---

## 2. Verdicts on the "thicker strokes or larger dots only" changes

Judged from before/after posters and pointer frames. The question was whether the change makes the
operation's output easier to read at card size, not whether it raised the pixel-change number.

| Run | Change | Verdict | Reason |
| --- | --- | --- | --- |
| line-chunk | edges 2.2→3.6, dots 3.2→6 | **Accept** | the chunk break points become visible; they are the output |
| line-segmentize | output dots 3.6→6, input 6→8 | **Accept** | the new vertex density is the point of the operation and now reads |
| line-smooth | 1.8→3.2 | **Accept** | the smoothed curve was faint at card size; the input polyline stays visible |
| line-locate | tick length 0.05→0.07, stroke 3→4.5 | **Revert** | ticks merge into a band and the tangent direction they encode is harder to read |
| building-tessellation-recipe | 2.6→3.8 | **Revert** | no visible clarity gain (2.2→2.6%), only poster churn |
| group-convex-hull | 2.6→4 | **Accept** | the hull outlines read better; the gesture is still 1.6% |
| group-statistics | member dots 3.6→6 | **Revert** (or about 4.5) | the members of large groups merge into a blob, hiding the individual rows |
| region-mask | 3→5 | **Accept** | the kept points inside the star read clearly |
| pick-region-mask | halo 7→11 | **Revert** | the halos fuse into white blobs that bury the selection ring and the picked colours |
| point-pattern-envelope | chart lines 3.4 | **Accept** | the envelope curves were hairlines at card size |
| spherical-convex-contains | dots 3→4.6 (+ size 60→80 km) | **Accept** | the containment colours read; the larger polygon shows more inside/outside |
| spherical-polygon-contains | dots 3.6→5 (+ hole 40→65 km) | **Accept** | the hole is now legible as an exclusion |
| alternative-routes | routes 3.2/2.6/2.0 → 4.6/3.7/2.8 | **Accept** | the three ranked routes separate visibly |
| network-flow | edges 2.2→3.4 | **Revert** | the zero-flow network gets heavier and the flow path loses contrast; size by flow instead |
| facility-scenario | dots 5→8 | **Accept** | coverage classes read at card size |
| huff-trade-areas | dots 5→8 | **Accept** | the trade-area colours read |
| catchment-accessibility | dots 6→8 | **Accept** (neutral) | slightly more legible, no harm |
| network-accessibility | roads 2.4→3.6 | **Accept** (neutral) | the gradient reads a little better |
| network-disruption | removed edges 1.8→3 | **Accept** | the disrupted edges are the subject |
| network-facility-optimization | dots 6→8 | **Accept** (neutral) | no harm |
| network-snapping | 3.5/7/5 → 4.5/9/7 | **Revert** | clutter; the snapped-versus-input distinction does not improve; still 1.6% |
| track-quality | dots 6→9 | **Revert** | dots crowd the dense centre and hide the tracks; 0.4→0.5% is no gain |
| geodesic-pairs | arcs 1.1→2, alpha 0.85 | **Accept** | the great-circle arcs were too faint |
| raster-sampling | dots 3→4.5 (+ more points and holes) | **Accept** | the sampled values and the no-data points read; add the caption disclosure (M8) |
| pair-gather | 3/1/6 → 4/1.6/8 | **Revert** | the picture is unchanged in substance; re-verified: gesture 1.9–2.1% and `how` 0.46%, so neither problem is fixed |
| profile-peaks | 9→14 | **Revert** (or about 11) | the 14 px discs cover the prominence ticks and the peak shapes they mark |
| spatial-pearson | dots 4.5→5.5 | **Accept** | harmless; revert its noise default (M7) |

---

## 3. Constructed or altered inputs: disclosure check

| Run | Constructed input | Disclosed? | Misrepresents? | Action |
| --- | --- | --- | --- | --- |
| pointcloud-ground | constant tilt 0.17 | yes, in the caption | mildly: the sliders barely act anyway (H4) | revert or fix the ranges |
| pointcloud-smrf | constant tilt 0.17 | the tilt yes; the failing defaults no | **yes**: the poster shows ground removed at defaults, and the surface panel loses its relief | revert (H1) |
| spike-repair | 6-pass smoothing; ±30 m per seam | smoothing **no**; seam noise called "real ground change" | **yes** | revert the smoothing, fix the caption (H2) |
| datum-grid-transform | tolerance 3e-9°, inverse default, limit 1–4 | the caption explains the marker | no | fix the status test (M6) |
| contiguity-weights / fuzzy-contiguity | gap and tolerance defaults | comment, with measurements | no | count the GPU links (M7) |
| multiscale-gwr | bandwidth ranges, third map | caption | the intercept map contradicts the caption at rest | default intercept bandwidth 1 (M1) |
| point-horizon-visibility | maximumDistance 20 km | no (fixed prop) | no | accept |
| coverage-validity | seven graded defects | via `Data: gap` | no; the gesture's unit is the problem (H5) | keep the defects |
| raster-sampling | larger holes | **no** | no | caption (M8) |

---

## 4. Text layer (`runs/text/`) and its wiring

What works:
- The atlas is reference-counted per device (`WeakMap<Device, {buffer, users}>`) and released in
  `destroy`. `DemoLayer.destroy` destroys only its own model and look buffer, never the bindings, so
  the shared atlas survives the other layers.
- The instance buffer grows by destroy, recreate and `setBindings`.
- Layout reruns only when the key changes.
- Halos are drawn before all ink, so neighbouring glyphs are never covered.
- Rule pixel snapping and the rotation are consistent between the CPU and the shader.
- In the wiring, the empty `[]` signature for `text` keeps the layer object across text changes.
  `clip` works through the framework's generic scissor path, and `context` is passed through.

Defects and risks:
1. **Licence (H3).** Must be fixed before wiring.
2. **Bundle cost (M).** `define-run.ts` and `harness.ts` import from `./text`, which statically pulls
   in `font-atlas.ts` (116 KB of source, about 80 KB of base64). Every run chunk and the harness pay
   for it even when they draw no text. Move `CARD_DISPLAY_WIDTH` and the `TextRunLayer` type into a
   tiny module, and load `TextLayer` and the atlas lazily, for example on the first `text` layer
   through a dynamic import awaited in `build` or `afterRun`, or register them once per page.
3. **Coupling to `kit.ts` (M).** The shader reuses kit's `Look` fields under other meanings: `radius`
   is the halo width, `hillshade` the halo strength, `elevation` the field spread, `grid` the atlas
   size (through `writeGrid`, which writes `values[12]` and `values[13]`). The README says `kit.ts` is
   not to be changed by runs. Any reorder there silently breaks text. Give the text layer its own
   small uniform, or document the contract in `kit.ts` beside `PALETTE_STRUCT`.
4. **Sampling at card size (L).** The atlas has 24 texels per em and a spread of 5. At 10 px text and
   a device pixel ratio of 1, `pixelsPerTexel` is about 0.42, so each fragment spans about 2.4 texels
   with one bilinear tap. Thin strokes can alias, and the halo is clamped to `spread * pixelsPerTexel
   - 1 ≈ 1.1 px`, under the intended 1.4. Check the DPR 1 preview. If the stems shimmer, use 32–48
   texels per em or a 2-tap supersample when `pixelsPerTexel < 0.6`.
5. **Theme key (L).** `theme.categories` is not part of the layout key, so swatch colours would go
   stale if only the categories changed between themes. Add it.
6. **`labels.ts` (L).** `createBarValueLabels` assumes bars rise from `range[0]`. For diverging charts
   (negative bars), the `bottom` anchor puts the label on top of the bar; anchor `top` when the value
   is below the baseline. `createLegend` sizes swatches with a cap height of 0.72, but
   `FONT_ATLAS.capHeight` is 0.694; use the atlas value.
7. **Known before wiring (from WEB-INT-16, confirmed).** The verifier counts a text-only layer as
   drawn output; exclude `text` from the "output drawn" check. Hero posters use
   `CARD_DISPLAY_WIDTH`, so their text is about three times too large; pass the hero width when
   capturing hero posters, or render separate hero posters.
8. **Reproducibility (L).** `generate-font-atlas.mjs` rasterizes through Playwright's Chromium, so
   the atlas bytes depend on the browser build. Pin or record the Chromium version in the header.

---

## 5. Not verified
- Whole-group verification was not run; WEB-PIC-14 notes that no group had one.
- `optics-clustering` and `peak-snap` were not re-run after their last edits. Their screenshots
  postdate some edits; see `web-g3.md`.
- The light theme was not checked for any changed run.
