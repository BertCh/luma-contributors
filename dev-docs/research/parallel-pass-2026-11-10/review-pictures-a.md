# Review A: wave-3 pictures in regression-weights, spatial-statistics, density-interpolation-regions, geometry-construction-overlay

Reviewer: independent, read-only. Date 2026-11-11.

Method. Listed the run files changed today by mtime (09:16 to 09:28; the 00:53 edits to
`geometry-construction-overlay/shared/lines.ts`, `shared/polygons.ts` and `shared-paths.ts` are API renames
by another stream and not reviewed here). Read each run's diff against `74dddcf5`; that diff also contains
the 2026-11-10 waves, so I judged today's claims from the reports and the current source. I ran
`node scripts/verify-runs.mjs <ids> --themes dark,light` on all 38 changed runs in five batches
(14+14+12+16+20 = 76 of 76 PASS), plus `concave-hull` and `line-clip-by-polygon` (the other callers of
`shared/reach.ts`; 2 of 2 PASS). I opened the dark and light rest screenshot of every run and pointer frames
for about fifteen, and read the verifier's per-position readouts and pixel changes. Read the class contracts in
`modules/analysis/dist` for traces, balanced sampling and weights validation. All report folders deleted.

Scores use the re-score scale (1 to 5, rest and gesture). "Before" is `scores.json` (pictures of 2026-11-09).

## 1. Table

| Run | Before rest/gesture | Now rest/gesture | Verdict |
| --- | --- | --- | --- |
| pair-gather | 1/2 | 2/2 | Bolder, but still sparse (2.8% of pixels drawn); the legend leaves out the most visible marks (grey unmatched discs, value-coloured dots) |
| multiscale-gwr | 2/1 | 2/3 | The intercept now varies and the pad acts; the x2 map is invisible in both themes, there is no colour key, and the caption is false |
| spatial-weights-algebra | 2/2 | 3/2 | Readable, legend matches; readout still mostly "band 1 → 1" |
| neighborhood-summary | 2/2 | 2/2 | Titles only; still two look-alike maps, and low values vanish in both themes |
| chunked-weights-transpose | 2/2 | 2/2 | Hover arrows are still a tiny cluster; links almost invisible in light |
| gwr-nonstationarity-test | 2/2 | 3/3 | Labelled well; the "dot: shuffled" dots are white on cream in light |
| probit-regression | 2/2 | 3/2 | Titles help; truth dots invisible in light; the rule cuts the x2 bar in two |
| seemingly-unrelated-regression | 2/2 | 3/2 | Labelled; truth dots invisible in light |
| spatial-error-maximum-likelihood | 2/2 | 2/2 | The new titles stay put while the curtain moves, so they name the wrong side |
| spatial-weights-traces | 2/2 | 2/2 | The claim checks out against the contract; the picture is still a spike and a flat line |
| instrumental-variables-covariance | 2/3 | 3/3 | Labelled; the caption correction is right |
| spatial-lag-impacts | 3/3 | 4/3 | Clean labelled panels |
| restricted-regime-regression | 3/3 | 3/3 | Legend and titles fine; the Chow F bar has no scale |
| spatial-weights-validation | 2/3 | 3/3 | Legend matches the flag bits; it covers some nodes |
| global-spatial-statistics | 2/2 | 2/3 | Reach changes the radius and the bars; the bars grow from −8, so the new ticks misread |
| multiple-testing-adjustment | 2/2 | 2/3 | Methods now differ; at rest the legend is dropped and the significant tests sit hidden at the origin |
| nested-interpolation-validation | 2/1 | 3/3 | Folds coloured, probe varies, labels fine |
| spatial-cross-validated-ols | 2/2 | 2/3 | The new "lit fold" bars draw zeros in every slot and hide the real coefficients |
| spatial-interaction | 2/2 | 2/2 | A hairball in a small corner; the readout is fitted = observed everywhere |
| spatial-pearson | 1/2 | 2/2 | The bar now spans −1 to r under −1/0/1 ticks; the curtain readout is constant |
| lisa-markov | 3/4 | 3/4 | Labels only, clean |
| point-pattern-envelope | 2/2 | 3/2 | Band and inset help; band and dots sit off the curves by half a step; the curve still clips at 0.3 |
| rank-markov | 2/2 | 3/2 | Titles help; the gesture is unchanged |
| rose-statistic | 3/3 | 3/3 | N/E/S/W fine; input legs faint in light |
| spatial-balanced-sample | 2/2 | 3/2 | Strata are readable at rest; 4 of 5 pointer positions produce an invalid allocation that is never reported |
| spatial-markov | 3/3 | 3/3 | Titles fine |
| transition-diagnostics | 2/2 | 3/2 | Labelled; the x title "class" sits at the top; the sparse band is faint in light |
| transition-matrix | 3/3 | 3/3 | Ticks fine; the "sparse" chart is empty at defaults |
| building-adjacency-recipe | 2/2 | 3/3 | Probe snaps and terrace counts change; the shared-wall orange is also a terrace colour |
| coverage-validity | 2/2 | 3/2 | The fill is honest (input only); the gesture saturates at 0.060 at 3 of 5 positions |
| group-concave-hull | 3/1 | 3/3 | Legend added; the teal noise points are not in it |
| line-segmentize | 2/2 | 4/3 | Clear: original discs against written vertices |
| linear-referencing | 2/2 | 3/3 | Legend; the "road" swatch is ink but roads draw grey; measure-0 foot points vanish in light |
| outline-geometry | 2/2 | 2/3 | Distance now varies (0.010 to 0.098); at rest only seven small discs (2.2% drawn) |
| parameter-buffer | 2/1 | 3/2 | Counters are the real calls but module-global; the pointer reaches only 0.002 to 0.021 |
| segment-ring-assembly | 2/2 | 3/2 | Input and output now distinct; "one colour each" shows one ring; "loose" is the wrong word |
| polygon-triangulation | 2/2 | 3/2 | Hole rings distinct; the drag changes 0.3% at 2 of 5 positions |
| constrained-delaunay-triangulation | 3/2 | 3/3 | Constraint legend is good; still looks like its sibling |

## 2. Findings, by severity

### Blockers (the picture misrepresents the operation)

**B1. `spatial-statistics/spatial-cross-validated-ols.ts:202`: the "lit fold" bars draw a zero in every other slot, so the chart no longer shows the coefficients.**
`const lit = new Float32Array(coefficients.length)` is filled only for the hovered fold and drawn as a second
`bars` chart in ink, range [-1, 1]. The bars shader (`layers.ts:156`) grows every bar from the range
minimum, so each zero becomes a half-height ink bar from −1 to 0. Evidence: in both rest screenshots,
every one of the 16 slots has an ink bar covering −1..0, which hides every negative coefficient behind it.
Only the bars above 0 stay visible. Fix: `new Float32Array(coefficients.length).fill(Number.NaN)` (the shader
gives NaN zero height).

**B2. `regression-weights/multiscale-gwr.ts:52`: the caption says the opposite of the scenario.**
The caption says "constant intercept, drifting slope and constant slope", but `getScaledCoefficients` (line 43) makes the intercept rise
from 0.5 to 2.5 south to north, and the file header says so. Fix: "Three coefficient surfaces fitted at
their own scales: an intercept rising south to north, a slope drifting west to east, a constant slope."

**B3. `regression-weights/spatial-error-maximum-likelihood.ts:96-97`: the new titles do not follow the curtain.**
"OLS residuals" and "filtered residuals" are fixed at 0.25 and 0.75 of the map. The curtain moves.
Evidence: at pointer position 1 the curtain is at about x 480 px, and "filtered residuals" (x of about 420) sits over
the OLS half. Fix: anchor each title to the middle of its side of `interaction.curtain`, or put both at the
curtain, left- and right-anchored.

**B4. `spatial-statistics/spatial-pearson.ts:105` and `global-spatial-statistics.ts:167`: signed values are drawn as bars from the range minimum under new −1/0/1 and −8/0/8 ticks.**
The bars shader always grows from the bottom of the range. With the axes added today, spatial Pearson
r = 0.68 reads as a bar from −1 to 0.68 that crosses the zero line. Moran's z = 5.2 reads as a bar from −8, and
Geary's bar looks negative across most of its length. Evidence: both rest screenshots, and the shader source
(`height = normalizeValue(value)`, with no baseline). Fix: use `drawSignedBars` from
`regression-weights/shared/regression.ts:264`, as `spatial-lag-impacts` does, or a dot on a zero line.
Until then, do not draw a 0 tick on these charts.

### Major (the picture or gesture still fails its purpose, or text is unreadable)

**M1. `multiscale-gwr`: the x2 map is empty in both themes, there is no colour key, and the minimum bandwidth is unchecked.**
x2's true coefficient is −1.5 on the shared range [-2, 3], at the dark end of the ramp. In dark the map is
background-coloured with outlines only. In light it is near-white. That was the "one all dark" complaint of
the re-score, and it is still there. There is no ramp legend, so "intercept" against "x1" cannot be read as
values. The new minimum bandwidth of 0.08 is the value the 2026-11-10 review (M1) warned may be
ill-conditioned for a bisquare kernel with three coefficients. `status.codes` is allocated but never read.
At the rest x1 bandwidth of 0.18, the NE tracts reach the top of the ramp. Fix: a diverging ramp centred on 0
with a three-tick key, or per-map ranges with a key each. Read `codes` and raise the minimum if any row fails.
On "constructed input": the simulated response with a known truth predates today. It is disclosed on the
function page, because `createResponse` carries the MODEL_INPUTS note "Responses are simulated from tract
covariates and specified coefficients". The card shows "Chicago tracts", because the "Modeled scenario"
label in `entry-card.tsx:20` appears only when a run has no data source. That is acceptable, but the card
caption must then be true (B2).

**M2. `density-interpolation-regions/spatial-balanced-sample.ts:87-89`: most pointer positions request more units than the stratum has, and the run hides the operation's status.**
The frame is 900 rain stations that occupy only about x 0.44 to 1.33 of the world, but the split is clamped
to 0.25 to 0.75 of the world width. The verifier readouts give west 15 units with 60 asked at positions 0 and 3,
and east 0 units with 100 asked at positions 1 and 4. Only the middle position is a valid design. The class
validates `frameCount >= sampleCount` per stratum (`gpu-spatial-balanced-sample.ts:391`) and writes
status 1, "invalid frame/allocation", with zero selected. So "west: 0 sampled of 15" and "east: 0 sampled of 0" are the documented
contract, not a defect, but the labels present a failed design as a result. Fix: clamp the split to the frame's
x range so each side keeps at least `max(westCount, eastCount)` units (sort the x values once in
`createInputs`). Read `status` back in `afterRun` and label an invalid stratum
"60 asked, only 15 units: invalid allocation".

**M3. `spatial-statistics/multiple-testing-adjustment.ts:184-197`: at rest the legend is dropped and the significant tests cannot be seen.**
At alpha 0.05 the "alpha 0.05" label and the "significant after adjustment" swatch fall on the same spot,
and the swatch label is dropped. It is absent from both rest screenshots and present at alpha 0.11. The
orange significant tests sit at raw p of about 0.001, under the blue dots at the origin, so the poster shows no
significant test even though the readout reports 14. At position 1, dots at adjusted p = 1 run through the
"1.0 adjusted p" title. The 199 to 999 change is a legitimate operation setting (999 is the usual default), not
altered input. But with a two-sided permutation p-value floor of about 0.002 and 120 tests, Bonferroni and Holm can
never be significant inside the alpha range (0.2 / 120 ≈ 0.0017). Their "0 significant" is a resolution
floor, not conservatism, and the picture does not say so. Fix: give the swatch label priority 5 and put it at
the top left; add an inset or log axis for raw p < 0.05; draw the floor `1/(P+1)` as a labelled rule.

**M4. `density-interpolation-regions/point-pattern-envelope.ts:148, 170`: the band, the orange dots and the x ticks are misregistered with the curves, and the curve still clips.**
The line chart places sample i at `(i + 0.5) / count` (`layers.ts:214`). The band, the "outside" dots and the axis
ticks use `i / (count − 1)`. Evidence: the band extends past both curve ends, and the first and last orange
dots sit half a step outside the curve (rest screenshot, x 66 and x 912 against the curve's 88 to 890). The
report says the observed curve "reaches 0.23". In the screenshot it runs flat at the 0.3 maximum for the
last three radii, so it is still clipped. The vertical rule at rest also runs through the "0.10" tick label.
Fix: map radius index i to `(i + 0.5) / RADII.length` everywhere, and raise `RANGE[1]` to the observed
maximum, read back.

**M5. White reference dots are invisible in the light theme, and today's titles point readers at them.**
The colour `[1, 1, 1, 0.95]` is used in `gwr-nonstationarity-test.ts:196` ("spatial SD (dot: shuffled)") and in `shared/regression.ts:350` (the truth
dots of `probit-regression`, "coefficients (dot: latent truth)", and of `seemingly-unrelated-regression`).
Evidence: in the light screenshots, every dot that is not on top of a bar disappears into the cream background. Fix: `[...theme.ink, 0.95]`.

**M6. `regression-weights/chunked-weights-transpose.ts:106`: the requested fix is not done, and light got worse.**
At rest the hovered row's arrows are still a cluster about 30 px across. The links went from alpha 0.3 to 0.26 of a
fixed pale blue, and in light they are nearly invisible. The re-score asked for longer arrows in two colours.
Fix: use a theme colour for the links (`theme.muted`), and hover a row whose sources are far apart, or draw
in and out on a side panel.

**M7. `geometry-construction-overlay/parameter-buffer.ts:31, 53`: the counters are real but global, and the gesture covers a fifth of the range.**
`buildCount` and `writeCount` count real calls of `build` and of `update`, which calls `parameterBuffer.write`.
Both live at module scope, so `buildCount` grows with every new scene instance: re-entering the card after
the 200 ms grace, or the function-page hero. A reader who hovers the card three times sees "graph built 3×"
next to the caption "compiled once". The verifier's five positions give tolerance 0.002 to 0.021. The default
0.04 and the maximum 0.1 are reachable only outside the lake (REACH_SPAN 0.3 is larger than the lake's inner
reach), so the inner 90% does not span the range (roadmap 7.4.6). The picture changes by about 2%. Fix: keep the
counts in the `build` closure (in `outputs`) and label "this graph: built once, buffer written N×". Set
REACH_SPAN to the lake's inner reach (about 0.12), or map from the canvas edge. `getLogarithmicReach` in
`shared/reach.ts` already does what lines 35-41 duplicate.

**M8. `regression-weights/pair-gather.ts`: the legend does not match what is drawn.**
The most visible marks are large grey discs (queries with no match, NaN on the gathered ramp) and the large
gathered dots coloured by value, which are near the background at low values in both themes. Neither is in the legend. The
"left rows (queries)" blue dots are 2.5 px. Fix: add "no match" and "gathered value" entries with a
two-stop ramp key, and draw queries at radius 4 or more.

**M9. `spatial-statistics/spatial-cross-validated-ols.ts:265-290`: "trained on" and negative residuals are both blue.**
The held-out stations use the diverging ramp, so negative residuals are bright blue, next to "trained on" in
`theme.categories[0]`. The legend shows a single salmon swatch for residuals. Evidence: both rest screenshots.
The blue held-out cluster cannot be told from the training points. Fix: draw the training stations in
`theme.muted` and give the residual a two-colour key.

**M10. `geometry-construction-overlay/coverage-validity.ts:164`: the gesture still saturates, and the glyph misstates gapWidth when clamped.**
The readout is gapWidth 0.060 (the maximum) at 3 of 5 positions, and the picture changes 0.6 to 2.4%. When clamped,
the ring's diameter (2 × distance) is larger than the gapWidth the operation gets. The legend panel also covers part of the
top-left cells. The overlap "covered twice" is not visible at alpha 0.2. Fix: map
`gapWidth = 2 × distance` with the ring radius set to `gapWidth / 2` after clamping. Raise `max` to about 0.1, or use the log
mapping from `shared/reach.ts`. Move the legend to bottom-left, outside the coverage.

**M11. `spatial-weights-traces`: the claim is right; the picture still is not.**
The claim checks out. `uploadSpatialWeights` uploads rook contiguity with weights 1 (`inputs.ts:215`). The class contract says
`powerTraces[q−1] = tr((W/s)^q)` with `s = ||W||_inf` (`gpu-spatial-weights-traces.d.ts:7, 98-104`). The
readouts are internally consistent with a spectral radius of W/s of about 0.45: the trace ratio q16/q11 is
3.9e-6 / 2.4e-4, about 0.44 per step, and the sum ratio 3.2e-4 / 0.014 is about 0.47 per step. So the near-zero
traces are correct; I did not recompute them independently. The picture remains a spike at q = 2 and a flat line on
a linear axis. The left map ("tracts a q-step walk reaches") is a CPU walk over the input graph, not
operation output; it predates today and is labelled as such. The vertical rule at rest crosses the title
"power sums of W/s". s is never shown. Fix: put the `infinityNorm` output in the title ("W/s, s = 12"), plot
log10 of the trace (computed in a kernel), and move the title above the rule's range.

**M12. Text unreadable in light.**
The "sparse row" band in `transition-diagnostics` is pale yellow on cream. In `rose-statistic` the input legs
are pale blue on cream. In `neighborhood-summary`, `gwr-nonstationarity-test` (map) and `linear-referencing` the
low end of the sequential ramp is cream on cream. The last is a framework ramp issue that the reports also
note; it belongs in a framework request (a ramp whose low end clears the background in light).

### Minor

- `geometry-construction-overlay/outline-geometry.ts:120-121`: the comment still says "the edge of the outline passes under the cursor", which is false with the log mapping.
- `segment-ring-assembly.ts`: "loose input segment" describes cancelled interior edges. They are not dangling. With one ring at the default, "one colour each" is never visible. The swatch is `theme.muted` but the lines draw `ink` at 0.55.
- `building-adjacency-recipe.ts`: the shared walls are orange, and orange is also a terrace category (two terraces outlined in orange).
- `group-concave-hull.ts`: the noise points (8%, category 3, teal) are not in the legend.
- `linear-referencing.ts`: the "road" swatch is ink, but the roads draw grey (`context`).
- `line-segmentize.ts`: "inserted vertex" orange is also drawn on every original vertex. Say "written vertex (lone dot: inserted)".
- `gwr-nonstationarity-test`: the "0.05" reference label sits on the x2 bar. `probit-regression`: the vertical rule splits the x2 coefficient bar so that it reads as two bars. `transition-diagnostics`: the x title "class" is drawn at the top right of the third chart, and the rest glyph covers the "60" stop.
- `spatial-balanced-sample.ts:115-128`: the "east: …" title overlaps the top-right dots.
- `spatial-weights-validation`: the legend covers nodes at the top left.
- `spatial-pearson`: the new input scatter shows a near-linear y-against-x relation (aspatial r of about 0.95) beside spatial r = 0.68, with no explanation. The curtain readout is constant at all 5 positions.
- `spatial-interaction`: the map occupies a small part of the left half; the hovered readout "observed 921 · fitted 921" is always equal (a property of the production constraint) and so says nothing.
- `multiscale-gwr`: with a large intercept bandwidth, the north-south intercept trend leaks into the x1 map (backfitting). This is the estimator's behaviour, but a reader will read it as x1 varying north-south.

## 3. What I could not check

- Card size (320 px) and phone width: the verifier renders at 960×540 only, and I did not judge label legibility at card scale.
- Glide, latency, keyboard and the function-page hero (relevant to the parameter-buffer counters).
- Whether `GPUMultiscaleGeographicallyWeightedRegression` reports failing rows at bandwidth 0.08 to 0.18 (the `codes` buffer is not drawn, and I could not read it without editing).
- An independent CPU recomputation of the traces; the check above is the contract plus internal consistency.
- Two of the five pointer positions per run were judged from the verifier's numbers only, for runs whose pointer frames I did not open.
- Today's catalog text (none was changed after 09:00, by mtime) and any website file outside the four groups (none was modified after 09:00, by mtime).
- Today's diff in isolation: the uncommitted diff mixes the 2026-11-10 waves with today's work. Attribution to today relies on mtimes and the reports.
