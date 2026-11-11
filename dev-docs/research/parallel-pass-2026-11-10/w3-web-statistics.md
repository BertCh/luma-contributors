# w3-web-statistics

Screenshots: `website/.reports/w3-web-statistics/<group>/<id>-<dark|light>.png` (+ `-pointer-N.png`); envelope rerun in `.reports/w3-web-statistics-b/`.
Files under `website/src/gpu/runs/entries/spatial-statistics/` and `.../density-interpolation-regions/`. No catalog text changed (no entry text described a changed picture).

## 1. Done
- `nested-interpolation-validation`: already coloured by fold with the hovered fold lit (changed 2026-11-10; probe works, readout varies per fold). Added text layer: titles "stations by outer fold" / "held-out fold: observed vs predicted", axis titles, "on the line = predicted well". Before: unlabelled dots; after: labelled two panels.
- `spatial-pearson`: added a scatter of the two input variables (y against x, context layer) bottom-left, bar moved right, labelled `spatial Pearson r` with -1/0/1 ticks, axes x/y. Not the lag scatter asked for (the lag would be computed on the CPU; the operation outputs only the statistic). Readout stays constant under the curtain (the statistic does not depend on it).
- `multiple-testing-adjustment`: permutations 199 -> 999 (min p 0.005 made every method read 0 significant); now BH gives 14 significant at alpha 0.06 on the Chicago tracts. Axes (raw p, adjusted p, 0/0.5/1), live `alpha 0.05` label on the dashed line, swatch legend "significant after adjustment". Before: bare curve, "0 significant" everywhere; after: labelled, some methods significant.
- `point-pattern-envelope`: shaded envelope band (300 segments filled between the GPU lower and upper curves, from peek), orange dots where the observed curve leaves it, inset of the observed epicentres (60 real earthquake points, uniform scale), legend, axes r and K(r) - CSR, y range 0.3 (observed reaches 0.23, was clipped at 0.1).
- `spatial-interaction`: map now draws every fitted flow as a line in three widths by fitted value (classes written from peek of the operation's predictions), hover still highlights the tract's flows and marks its pairs; axis titles and caption line. Residual not drawn (see Not done).
- `spatial-balanced-sample`: frame coloured by stratum, selected units large in ink colour, labels "west: N sampled of M" / "east: ..." from a GPU readback of the chosen rows (`afterRun`) and the input frame counts. Fixed a stale readout (split read before update). The readout says units and asked counts.
- `knox-test`, `mantel-test`: untouched (chart labels from yesterday kept; verified passing).
- `spatial-cross-validated-ols`: hovered fold's training stations drawn in blue (CPU copy of the operation's training mask), buffered orange (already there), held-out in residual colours; legend and coefficient axis (-1/0/1).
- `global-spatial-statistics`: bars named (Moran I, Geary C, Getis-Ord G), y ticks -8/0/8 and title `signed significance`; chart raised so names clear the hover hint. The pointer now changes the radius and the bars (earlier change), so the "stays the same" note no longer holds.
- `rank-markov`: map title `rank in period N`, matrix title, mobility bars named by step (1-2 ... 5-6).
- `transition-diagnostics`: titles west/east counts, steady-state titles with class ticks, labelled threshold stops with `minimumTransitions`, legend swatch for sparse rows.
- Labels only: `lisa-markov` (from/to, n.s./HH/LH/LL/HL ticks, quadrant legend), `spatial-markov` (low/lag 2/.../high lag matrix titles), `transition-matrix` (class ticks, from/to, bar titles), `rose-statistic` (N/E/S/W).

## 2. Verified
- `node scripts/verify-runs.mjs <16 ids> --themes dark,light --out .reports/w3-web-statistics`: 32 / 32 passed (16 per group). `point-pattern-envelope` rerun after the last edit: 2 / 2.
- Read dark or light full screenshots for every run above; pointer sheet for `spatial-balanced-sample`. Light and dark both read for envelope, global-spatial, balanced-sample, lisa-markov, transition-matrix, rank-markov.
- `npx tsc -p website/tsconfig.json --noEmit`: no errors in my folders (filtered). `check-catalog.ts`: 0 errors, 2 warnings (not mine). `npx biome check --write` on touched files: clean.
- Not run: phone-width pictures, card-size live-site check, a pointer-sheet read for every run.

## 3. Not done
- `spatial-pearson` lag scatter, `global-spatial-statistics` Moran scatter: need the spatial lag, only computable on the CPU.
- `rank-markov` rank-change colouring, `spatial-interaction` residual: not operation outputs; would be CPU-derived.
- `nested-interpolation-validation` fold boundary lines: not drawn.
- `knox-test`/`mantel-test` fit-to-frame and snap-to-nearest: not touched (told to keep).
- Gesture of `spatial-pearson` (curtain) cannot change the statistic.

## 4. Work list for others
Regenerate posters (rest picture changed, both themes): nested-interpolation-validation, spatial-pearson, multiple-testing-adjustment, point-pattern-envelope, spatial-interaction, spatial-balanced-sample, spatial-cross-validated-ols, global-spatial-statistics, rank-markov, transition-diagnostics, lisa-markov, spatial-markov, transition-matrix, rose-statistic. No framework request.

## 5. Roadmap
- WEB-PIC-12 (statistics group): remaining: lag scatters (Pearson, Moran), rank-change colouring, interaction residual, fold boundaries, knox/mantel snap-to-event; all need a library output or a framework allowance for CPU-derived overlays.
- WEB-INT-16: ranked runs in my folders now all have a text layer (`lisa-markov`, `spatial-markov`, `transition-matrix`, `rank-markov`, `multiple-testing-adjustment`, `rose-statistic`); `spatial-markov` has no class ticks.

## 6. Noticed
- `spatial-balanced-sample`: with the split at 0.89/1.33 world units a side has 0 or 15 units while 60 or 100 are asked; the picture at split 1.33 shows "east: 0 sampled of 0", so the operation tolerates an allocation larger than the stratum (no error visible). Worth checking its status output.
- `spatial-markov`, `transition-matrix`: the transition-matrix "sparse" bar chart is near-empty at defaults.
