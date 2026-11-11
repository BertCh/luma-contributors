# Review B: wave-3 picture rework (8 groups, 45 runs)

Reviewer: independent, read-only. Date 2026-11-11, from about 09:35.

Method. I took the runs from the three reports and from the files changed after 09:00 today in the eight
folders. I read every diff against `74dddcf5` for the runs, the new shared helpers (`recipes-trajectories/lib/pair-links.ts`,
`network-routing-access/shared/display.ts` and `band-text.ts`, the `scale` option in `geometry-processing/shared/shape-segments.ts`)
and the changed `.parameters.ts` files. I checked operation semantics against `modules/analysis/dist/**/*.d.ts`.
I ran `node scripts/verify-runs.mjs <ids> --themes dark,light --out .reports/w3-review-b` in six batches of 5 to 9 ids on the
running 3301 server. All 45 runs passed (90 of 90 renders). Warnings: track-quality `maximumGap` inert;
raster-registration `upsampleFactor` inert; computation-metrics, computation-record and snapshot-tap
"differs after parameters return"; computation-metrics "differs after the pointer returns". For every run I
looked at the full-size dark and light rest screenshots and all three pointer screenshots (composited per run
with PIL), and read the five readouts per run in the JSON. The report folder was deleted afterwards.

Scores use the re-score scale (rest, gesture; 1 to 5). "Before" is `scores.json` (pictures of 2026-11-09).

## 1. Scores

| Run | Before | Now | Verdict |
| --- | --- | --- | --- |
| geometry-processing/canonical-geometry-equality | 2/3 | 3/3 | Equal and unequal leaves are legible. Tolerance circles and lines are not explained |
| geometry-processing/geometry-orientation | 2/2 | 3/2 | Colour by reversal works. The legend sits on a ring, arrows on holes are bigger than the holes, and the readout is constant |
| geometry-processing/geometry-validity | 2/3 | 3/3 | The five-class legend reads. One hole is clipped at the right edge |
| geometry-processing/line-loop-removal | 2/2 | 4/2 | Removed loops are clear in red. The drag changes 0.3 to 0.7% and the readout gives 129 or 127 |
| geometry-processing/lon-lat-polygon-preparation | 2/2 | 3/2 | The date line cuts through the legend text. One swatch stands for five piece colours. Every probe still reads "whole" |
| geometry-processing/minimum-clearance | 2/2 | 3/2 | Values and witness ends are honest. The drag changes under 0.7% |
| geometry-processing/normalize-geometry | 2/2 | 4/3 | Good slot layout. The caption is stale |
| geometry-processing/normalize-polygons | 2/2 | 3/3 | The fixed canonical start against the moving input start is the right demonstration |
| recipes-trajectories/clock-encounters | 2/2 | 2/3 | Links are honest, but the scene is small and sparse, the bars have no labels, and the scene is still the same as its two siblings |
| recipes-trajectories/trajectory-encounters | 2/2 | 3/3 | Red first-bucket links are legible, and the reach changes the pair count (39, 15, 23) |
| recipes-trajectories/trajectory-closest-approach | 2/2 | 3/2 | Links are honest but collapse into double dots. The readout is about "400 s" at four of five positions |
| recipes-trajectories/dasymetric-redistribution-recipe | 2/2 | 3/3 | Source totals are legible. The rest glyph hides one total |
| recipes-trajectories/hot-spot-analysis-recipe | 2/2 | 3/2 | The legend says "z up to +4" while the readout gives 4.6. The window changes 1.35% at every position |
| recipes-trajectories/segregation-counterfactual-recipe | 2/2 | 3/3 | Bars have names and values. The map is dim |
| recipes-trajectories/population-support-alignment-recipe | 3/3 | 3/3 | Labels only. Fine |
| network-routing-access/track-quality | 2/2 | 3/3 | The drag now creates a bold speed leg. The glitches are simulated, which the caption does not say, and the tracks leave the frame |
| network-routing-access/network-kernel-density | 2/2 | 3/3 | The halo is computed from the input distance matrix, not the KDE output. There is no legend |
| network-routing-access/catchment-accessibility | 2/2 | 3/3 | The lit catchment is the cost-matrix row (operation input). It moves with the facility. There is no legend |
| network-routing-access/network-snapping | 2/2 | 3/2 | Links are visible. The gesture changes about 1% |
| network-routing-access/adjacency-matrix | 2/2 | 3/2 | The new "rows: home, columns: work" note is false (the graph is symmetrized). The "band 3" label is under the rest crosshair |
| network-routing-access/adjacency-matrix-order | 2/3 | 3/3 | Has the same false note |
| network-routing-access/map-matching | 2/2 | 3/2 | **"Matched route" is straight chords between matched positions, and they cut across blocks** |
| network-routing-access/network-k-function | 3/3 | 3/3 | Labels only. The Euclidean circle for a network distance remains |
| cells-index-join/cell-geometry | 2/2 | 3/3 | Decoded boundary, corners and centre are legible. No Honshu or points are visible, and there is no legend |
| cells-index-join/cell-grid-path | 4/4 | 4/4 | The readout now agrees with the class documentation |
| terrain-hydrology-flow/terrain-dem-coregistration | 2/2 | 3/2 | **A constructed "glacier" on the Grand Canyon DEM, not disclosed**. The readout is identical at all five positions |
| terrain-relief-light/terrain-critical-points | 2/2 | 3/3 | Legend counts come from the GPU and match the readouts. The swatches are bright where the cells are drawn dimmed |
| terrain-relief-light/weiss-landforms | 3/3 | 4/3 | The ten-class key helps. Swatches are full-strength over alpha-blended cells |
| terrain-relief-light/viewshed | 3/4 | 4/4 | The legend is fine |
| terrain-relief-light/point-horizon-visibility | 3/3 | 3/3 | The legend is fine |
| terrain-relief-light/point-horizon-profile | 3/4 | 3/4 | The "0° horizon" label sits on the bars, and the title sits on the dashed ring |
| dataframe-pointcloud-records/calendar-buckets | 3/3 | 3/3 | Axis labels read in both themes |
| dataframe-pointcloud-records/computation-metrics | 2/2 | 3/2 | The ms tick labels are drawn on the teal bar and are clipped in the light theme. Dashed outlines clutter the rest picture. There are 3 distinct pictures and x changes nothing |
| dataframe-pointcloud-records/computation-record | 2/2 | 3/3 | Counts read. The lifetime title overlaps the first rows |
| dataframe-pointcloud-records/pointcloud-height | 2/2 | 3/2 | The probe reads "height 0.00 (0 m)" at **all five** positions. "bright" is wrong in the light theme |
| dataframe-pointcloud-records/pointcloud-icp | 1/2 | 3/3 | The three clouds can be told apart. At rest ICP visibly does not converge. The caption gives wrong colours |
| dataframe-pointcloud-records/pointcloud-neighborhoods | 2/2 | 2/2 | The neighbourhood is still about 30 px. The count colouring has no key |
| dataframe-pointcloud-records/pointcloud-statistical-filter | 2/1 | 3/3 | Rejections read. A hidden CPU re-implementation of the threshold draws "estimate" points |
| dataframe-pointcloud-records/pointcloud-voxel-merge | 1/2 | 3/3 | Tiles, seam and shared voxels read. The caption is stale and the merged centroids are not in the legend |
| dataframe-pointcloud-records/snapshot-tap | 2/2 | 3/3 | Titles help, and the residual changes with the tap |
| raster-bands-indices/change-detection | 3/4 | 3/4 | The legend is fine |
| raster-bands-indices/extrema-pyramid | 2/2 | 3/3 | The minimum and maximum titles make the curtain readable |
| raster-bands-indices/glcm-texture | 2/2 | 3/3 | The texture is shown. Its main gradient is the added noise ramp, disclosed in the picture but not on the function page |
| raster-bands-indices/raster-registration | 1/2 | 3/3 | Honest. The "true (hollow)" mark is a filled disc. Keeping `upsampleFactor` is honest (see 2.12) |
| raster-bands-indices/spectral-unmixing | 1/2 | 2/3 | The three panels are nearly flat (water 0.52 to 0.57 everywhere in a burned forest). "bright = larger fraction" is false in the light theme |

Summary: 42 of 45 runs are at or above their earlier rest score, and 35 are higher. Rest at 2 or less: 3 runs
(clock-encounters, pointcloud-neighborhoods, spectral-unmixing). Gesture at 2 or less: 15 runs.

## 2. Findings, most severe first

### Blockers (misrepresent the operation)

**2.1 map-matching draws chords between matched positions as "matched route"**
- Where: `network-routing-access/map-matching.ts:108` (comment "the route the matcher found"), `:213`, `:229`.
- Evidence: in the rest screenshots (both themes), the orange "matched route" segments cut straight across
  city blocks: the long diagonals bottom right, and the leg from about (205,430) to (290,415). They come
  from `addConsecutiveSegments(snappedPositions)`, which joins consecutive matched positions with straight
  lines. A matched route follows road edges. A reader sees the matcher producing off-road routes, which is
  not what the operation outputs.
- Smallest fix: label the entry "matched positions, in order", and draw the chords thin or dashed, or drop
  them. If the operation reports matched edge ids, light those edges instead.

**2.2 terrain-dem-coregistration: a constructed glacier on the Grand Canyon, presented as a glacier**
- Where: `terrain-hydrology-flow/terrain-dem-coregistration.ts:33` (the Gaussian "glacier" moved to x 0.5)
  and `:216` (new label "glacier (unstable)"). No entry in `website/build/demo-data-sources.ts` SCENARIOS.
- Evidence: the DEM is `sampleElevation`, the "Grand Canyon elevation" source (USGS 3DEP). The card's
  dataset label is built from imports (`entry-card.tsx:20`), so the card reads "Grand Canyon elevation".
  The function page shows no scenario note. The bias and the thinning disc are both synthetic. Today's
  change makes them the centre of the poster and names them "glacier". The move itself only changes where
  the synthetic disc sits, and the fit is unaffected (readout "fitted shift 0.049 · true bias 0.050",
  identical at all five positions). But the scene is now titled and labelled as glacier loss on a canyon DEM.
- Smallest fix: add a SCENARIOS entry, for example "The second epoch is the measured DEM plus a constant
  bias and a modeled thinning patch on unstable ground". Rename the label "modeled loss (unstable)" and
  the titles "raw: bias + modeled loss".

**2.3 adjacency-matrix and adjacency-matrix-order: the new note "rows: home tract · columns: work tract" is false**
- Where: `network-routing-access/shared/band-text.ts:60` and `:76`. The graph comes from `shared/graphs.ts:77-79`.
- Evidence: both runs use `createBandedGraph`, which adds every flow in both directions
  (`rows[home].add(work); rows[work].add(home)`). The matrix is therefore symmetric, and rows are not homes.
  The same function also drops 5 of every 6 cross-band flows (`index % 6 !== 0 return`). That is constructed
  input, not new today, and disclosed neither in the caption ("Chicago home-to-work links by tract")
  nor in SCENARIOS.
- Smallest fix: change the note to "tracts linked by commuting, either direction". Add a SCENARIOS line
  for the thinned cross-band flows, or stop thinning.

### Major

**2.4 pointcloud-statistical-filter keeps a CPU re-implementation of the filter on the picture**
- Where: `dataframe-pointcloud-records/pointcloud-statistical-filter.ts`, in `resolve` (the "Immediate
  feedback" block: mean, population deviation and threshold computed in JS, with candidates drawn as amber
  dots and "about N rejected").
- Evidence: the brief forbids drawing CPU-computed values in the operation's role. The block is labelled
  "estimate" and shown only before a settle, so it does not appear in the posters. It duplicates the
  operation's rule (for example, `counts == 0` is counted as rejected) and can disagree with it.
  The header comment says "ringed in orange", but the code draws amber dots. "small grey dots" in the
  header are drawn light blue.
- Smallest fix: delete the preview and show only the pending level tick. Settle already reruns the operation.

**2.5 pointcloud-height: the probe lands on ground at every test position**
- Where: `dataframe-pointcloud-records/pointcloud-height.ts` (probe), `:204-205` ("bright: ...").
- Evidence: the readouts at the five positions are all "height 0.00 (0 m) · bilinear". The report says
  "the probe already rested on a tree". That holds only for the rest position. In the light theme, tall
  points are dark purple, but the legend says "bright: up to 66 m".
- Smallest fix: snap the probe to the tallest point within a few pixels, or report the canopy height of the
  hovered cell. Use "top of ramp" wording that does not depend on the theme.

**2.6 spectral-unmixing: flat panels, and a theme-false key**
- Where: `raster-bands-indices/spectral-unmixing.ts:29-33` (hand-set endmembers), `:211`, `:219`.
- Evidence: the probes read water 0.52 to 0.57, soil 0 to 0.12 and vegetation 0.36 to 0.48 at all five
  positions. All three panels are nearly uniform. The scene is the Dixie Fire Sentinel-2 extract, so the
  picture tells a reader that half of a burned forest watershed is water. This is the operation's real
  output for endmembers that do not fit the scene, not a GPU defect, but the new three-panel picture makes
  it the headline. In the light theme the ramp runs dark for larger values, so "bright = larger fraction"
  and "bright = poor fit" are false.
- Smallest fix: name the dark endmember "dark (water or shade)", or take endmembers from scene pixels.
  Make the key theme-neutral, for example "0 to 1, light to dark" chosen per theme, or a ramp bar with end
  labels.

**2.7 pointcloud-icp: the rest picture shows ICP not converging, and the caption describes other colours**
- Where: `dataframe-pointcloud-records/pointcloud-icp.ts:54` (caption) and `pointcloud-icp.parameters.ts` (defaults
  rotation 8°, 12 iterations).
- Evidence: in the rest screenshots the orange "after" dots mostly sit off the grey targets, and the RMSE
  curve falls only from about 0.04 to about 0.03. At the pad corners (±11°) the readouts give "RMSE 0.044 -> 0.023";
  at rotation 0 they give "0.023 -> 0.000". The caption still says "misaligned copy (faint orange) after ICP (orange)",
  but the picture draws blue before and orange after.
- Smallest fix: fix the caption. Check whether non-convergence at 8° in 12 iterations is expected for this
  cloud or is an operation defect (see "could not check"). If it is expected, say so in the caption rather than
  changing the defaults to hide it.

**2.8 Light-theme and collision defects that make text unreadable at card size**
- `geometry-processing/lon-lat-polygon-preparation.ts:191`: the bottom-right legend sits on the dashed
  date line, which runs through "drawn" and "piece" in both themes.
- `geometry-processing/geometry-orientation.ts:153-154`: the bottom-left legend covers the bottom-left ring.
  `ARROW_SCALE = 1.7` (`:21`) makes the arrows on holes larger than the holes, so the hole arrows
  overlap each other and the outer arrows.
- `network-routing-access/adjacency-matrix.ts`: at rest the dashed crosshair runs through the "band 3" label,
  which reads "nd 3" in the poster.
- `dataframe-pointcloud-records/computation-metrics.ts`: the "0.050 ms / 0.025 / 0.000" ticks are drawn on
  the teal bar, and the first digit is clipped in the light theme (").050"). Dashed outlines on every pass-0
  bar clutter the rest picture.
- `dataframe-pointcloud-records/computation-record.ts`: the "resource lifetimes ..." title overlaps the first lifetime rows.
- `terrain-relief-light/point-horizon-profile.ts:152`: "0° horizon" is drawn on the bars, and the chart title on the dashed ring.
- `recipes-trajectories/dasymetric-redistribution-recipe.ts`: the rest ring glyph hides the bottom-left
  zone total ("-249").

**2.9 Gestures that still change nothing meaningful**
- pointcloud-height (2.5).
- terrain-dem-coregistration: the readout is identical at all five positions, and the curtain only moves the split.
- geometry-orientation: "14 rings reversed" at all five positions (the report says so).
- lon-lat-polygon-preparation: "whole" at all five positions (the report says so).
- line-loop-removal: 0.26 to 0.71% change, 129 or 127 vertices.
- minimum-clearance: 0.3 to 0.6% change. The witness line does not move to the squeezed vertex in the frames.
- hot-spot-analysis-recipe: 1.35% at every position (only the window outline moves).
- network-snapping: about 1% change.
- map-matching: 0.6 to 1.3% change.
- trajectory-closest-approach: readout about "400 s within 0.05" at four of five positions.
- computation-metrics: x changes nothing, and there are 3 distinct pictures.
- pointcloud-neighborhoods: only a 30 px ring moves.

**2.10 clock-encounters and its siblings are still one scene**
- Evidence: the three rest screenshots share the same fleet, frame and bar chart. They differ only in the
  link colour (white, red, orange) and the dot style. The fleet fills the top-right 55% of the frame. In
  clock-encounters only a few dozen instant dots are visible, and the bar charts have no axis or label in
  any of the three runs. The report's "WEB-PIC-15: the encounter trio is distinct now" is not supported.
- Smallest fix: name the bar charts ("buckets together, per pair" and "time within distance, per pair").
  Frame the fleet to fill the frame. Give one sibling a different window of the AIS data.

### Minor

**2.11 Stale or inaccurate captions and legends.**
- `normalize-geometry.ts:81` says "colored by canonical rank", but colour is now the leaf kind.
- `pointcloud-voxel-merge.ts:54` says "white rings", but shared voxels are now outlined boxes, in ink
  (dark in the light theme). The merged centroids, the operation's output, have no legend entry.
- `hot-spot-analysis-recipe.ts:112` says "(z up to +4)". That is the ramp clamp, and the readout reaches z 4.6.
- `track-quality.ts:20`: the caption does not say that the speed spikes are injected. The SCENARIOS note
  exists, but only on the function page. The report says "flagged readout now 12-ish", but the readouts give 26 to 28.
- `terrain-critical-points.ts`: the legend swatches are full-strength while the non-picked classes are
  drawn dimmed (saddles olive, peaks dark red at rest).
- `raster-registration.ts:272`: the truth mark is a filled disc (the default dot), but the label says "hollow".

**2.12 raster-registration `upsampleFactor` (reported inert): keep it.**
The `.d.ts` documents precision as 1/upsampleFactor. At the default factor of 10 the readouts are visibly
quantized ("true [0.27, 0.32] | got [0.30, 0.30]"). The verifier moves the factor to 1 and to 32 and
measures at most 0.12%, because the whole picture of the change is one 11 px dot moving. That is under the
0.2% threshold even for a large move. The slider acts. The report's explanation ("differs only in the third
decimal") is wrong. The "true" shift is the slider value used to synthesize the reference (SCENARIOS: "shifted
by a known offset"), so it is honest ground truth.

**2.13 Network halo and catchment light more than the threshold, and the halo is not the KDE output.**
`addZoneSegments` (`network-routing-access/shared/display.ts:399`) keeps an edge when either end is inside,
so up to one full edge beyond the bandwidth or costLimit is lit. In `network-kernel-density.ts:84` the
comment promises a falloff ("1 at the event falling to 0"), but threshold 0 makes the halo binary. The halo is
read from `addCostMatrix`'s distances, which are the operation's input, not its intensity output. That is
acceptable as context, but it needs a legend entry ("roads within the bandwidth of the dragged event"). The
catchment lights the facility's row of the same input matrix, which matches how the weights are defined.
Neither run has a legend for the lit roads or for the demand or intensity ramps.

**2.14 Voxel-merge counts and outlines are computed in `resolve` from `peek` of the previous run**
(`pointcloud-voxel-merge.ts:168-210`). They lag a seam drag by one run, and `shared < SHARED_LIMIT` (`:180`)
caps the legend count at 200 without saying so. That cap is unreachable with the current voxelSize range by
my estimate (about 25 rows times a few z levels), so it is latent. Track-quality's "flagged" count has the same
one-run lag, which existed before.

**2.15 Cross-group coupling.** `recipes-trajectories/lib/pair-links.ts:10` imports `addKernel` from
`geometry-processing/shared/shape-segments`. The same stream owns both today, but a later edit there would
break the three encounter runs.

## 3. Checked and found correct

- **Pair links.** `addBucketLinks` indexes `samples[track * bucketCount + bucket]` with `ids` (the lower
  track) and `partners`. This matches the `GPUTrajectoryEncounters` docs (row `track * bucketCount + bucket`,
  `ids` = lower track index). `addClosestApproachLinks` interpolates both tracks at `timestamps[first fix
  of lower track] + closestTimes`. This matches "seconds relative to the first sample of the lower track".
  Rows at or after `count` are parked. The links show exactly the pairs and moments the operations report,
  read on the GPU. The approach kernel binds 8 storage buffers, which is at the limit.
- **cell-grid-path.** The readout agrees with `gpu-cell-grid-path.d.ts`: `counts` is the true length,
  `counts > maximumPathLength` is overflow, and cells keep the first `maximumPathLength`. The new range of 3 to 24
  is within the maximum of 256. At the default of 20, one of four paths is still cut off at one position
  (readout "longest path 23 cells · 1 of 4 drawn cut off"), which is reported truthfully.
- **terrain-critical-points.** The legend counts (peak 35, pit 53, saddle 252) come from `afterRun` reading the
  operation's `counts` and match the probe readouts.
- **minimum-clearance.** The printed values are the operation's `clearances`, read back. The witness lines lie on short
  edges, which is the vertex-to-vertex case of minimum clearance, and their lengths match the printed values
  at screenshot scale.
- **ICP and voxel merge.** The "after" cloud is `addTransformedFootprint(source, transform)` on the GPU with
  the operation's `transform`. The shared voxels come from the two downsample tiles' `counts` (the merge's
  inputs). 0 shared voxels with the seam at 0.88 = 11 × 0.08 (a voxel edge) is correct.
- **Spectral unmixing.** The three panels come from `abundances` through one `GPURasterSampling` pass.
- **`addRingArrows` `scale`.** The default 1 emits the same literals as before (0.0240 and so on). The other caller,
  `normalize-polygons`, renders unchanged.
- **Parameter changes.** The new ranges of track-quality, network-snapping (runtime view, exact scan, no
  capacity), cell-geometry (resolution 3 to 6), cell-grid-path and spectral-unmixing (`shown` 0 to 1) contain their
  defaults and cannot make an operation throw, as far as the `.d.ts` limits go. None threw in the runs.
- **Readbacks.** Every new `afterRun` copies the bytes before reinterpreting them, and `draw` guards `extra` before use.
- **The "Modeled scenario" marking.** The card's meta label is derived from imported datasets
  (`entry-card.tsx:20`, `build/demo-data-plugin.ts`). "Modeled scenario" appears only when no dataset is
  imported. Perturbations of real data are disclosed only by `SCENARIOS` in `build/demo-data-sources.ts`, shown
  on the function page. That disclosure exists for track-quality, raster-registration, pointcloud-icp and
  pointcloud-statistical-filter. It is missing for terrain-dem-coregistration (2.2), glcm-texture (the noise
  ramp, disclosed only in-picture), and the adjacency runs (2.3).

## 4. Could not check

- Card-size (320 px) and phone-width views. I judged from 960 px screenshots and 640 px thumbnails.
- Motion, glide and latency. I judged from still frames only.
- Whether ICP's slow convergence at 8° (2.7) is an operation defect or a property of this terrain cloud. That needs a
  spec run at 2°, 4° and 8° against a CPU reference, and I was not allowed to run tests.
- Whether `GPUTrajectoryEncounters` can emit more than one row per pair (one per run of buckets). The `.d.ts`
  wording is ambiguous. The link code is correct either way.
- The map-matching outputs. I did not check whether the operation reports matched edge ids that could be lit instead of chords.
- The earlier-today (00:5x) changes in these folders by another session (`_lib/helpers.ts` `rowDirection`,
  `walking-cost.ts`, `support/zones.ts`, `pointcloud-smrf.ts`, `rectangle-clip.ts` and others). They are out of scope and not reviewed.
- Catalog text in `website/src/catalog/entries/*.ts` against the new pictures. No implementor edited it, and I did
  not cross-read it.
