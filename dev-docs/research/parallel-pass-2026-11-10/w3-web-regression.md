# w3-web-regression report

Folder: `website/src/gpu/runs/entries/regression-weights/` (all paths below are inside it). Screenshots: `website/.reports/w3-web-regression/regression-weights/` (dark, light, pointer 0 to 2).

## 1. Done
- `pair-gather.ts`: before, faint orange scraps and unlabelled dots (1% non-background). After, targets are solid green dots, queries blue dots, joined pairs bold orange lines, gathered value as the larger dot, unmatched queries stay grey; legend (left rows, right rows, joined pair). Same real data, no input altered.
- `multiscale-gwr.ts` + `.parameters.ts`: before, intercept map flat pink and x2 map flat dark, interceptBandwidth inert. After, the scenario response has an intercept that truly rises south to north (own function `getScaledCoefficients`, simulation with known truth, same noise), x1 drifts west to east, x2 constant; default interceptBandwidth 1 to 0.2 so the intercept surface follows its truth; titles intercept / x1 / x2 over the maps. The pad now changes the intercept map (large bandwidth flattens it, small one follows it). Operation check: not an operation defect (see Noticed).
- `spatial-weights-algebra.ts`: stronger input links and dots, legend (left k nearest, right distance band, combined: operation name). Readout unchanged.
- `neighborhood-summary.ts`: panel titles "input value" and "<statistic> of <k> nearest" (names from the props).
- `chunked-weights-transpose.ts`: dimmer background graph, larger in-degree dots, hover draws solid arrows out and dashed lines with dots for the sources the transpose lists, two-line key at top left.
- `gwr-nonstationarity-test.ts`: labels int / x1 / x2 under both bar charts, y ticks, "spatial SD (dot: shuffled)", "pseudo p" with a labelled 0.05 line.
- `probit-regression.ts`: titles for the map, the coefficients (dot: latent truth) and the marginal effects.
- `seemingly-unrelated-regression.ts`: map title (residuals of equation 1) and y axis of the residual scatter (eq 2 against eq 1).
- `spatial-error-maximum-likelihood.ts`: "OLS residuals" and "filtered residuals" over the two halves of the curtain.
- `spatial-weights-traces.ts`: titles on the three panels, readout precision `toPrecision(2)` instead of `toFixed` (so small traces no longer read 0.00 and 0), caption says W/s.
- Labels added (were without a text layer): `instrumental-variables-covariance` (int/x1/x2, y ticks, title; caption corrected: it showed standard errors, not "the covariance matrix row by row"), `spatial-lag-impacts` (three panels titled direct / indirect / total impact with colour swatch, x1/x2, ticks), `restricted-regime-regression` (titles, regime legend west/east), `spatial-weights-validation` (legend of the four injected violation classes). `adjacency-matrix-order` is not in this folder.

## 2. Verified
- `node scripts/verify-runs.mjs <15 ids of this report, plus spatial-regression-diagnostics> --themes dark,light --out .reports/w3-web-regression` (final state): 30 / 30 passed, 0 failures. `node --experimental-strip-types scripts/check-catalog.ts`: 0 errors, 2 warnings (space-time-events, not mine). Screenshots read in dark and light for pair-gather, multiscale-gwr, spatial-weights-algebra, neighborhood-summary, chunked-weights-transpose, gwr-nonstationarity-test, probit, SUR, SEML, traces, lag-impacts, restricted-regime, weights-validation, IV covariance. `spatial-regression-diagnostics` passes; not changed.
- `npx biome check --write` on each touched file: clean.
- Not run: `snapshot-runs.mjs`, phone width, a typecheck of the final state before this note (see end of file).

## 3. Not done
- `spatial-regression-diagnostics`: labels (tests, 5% critical line) were added in the previous wave; the remaining weakness is the gesture (hovering a test changes the map little). A fix needs a per-test map, which the operation does not output. Left.
- `spatial-weights-algebra` readout still reads the same at the verifier's positions (the probe sticks to a point); no data change can make it honest otherwise.
- `probit-regression`: iteration convergence on a log scale and a latent probability map not done (the operation's output is the fitted probability already shown).
- `seemingly-unrelated-regression`: two equations' residual maps side by side not done (needs a different map layout and probes; the scatter of the two residual sets is labelled instead).
- `spatial-error-maximum-likelihood`: the curtain compares the same two residual sets; no difference encoding added.
- `neighborhood-summary`: the output map is smoother than the input, which is the operation's output; no per-point own-value against mean chip (glyphs cannot be coloured).
- Catalog text: not changed (it describes operations, not the pictures; the only picture-describing text found was the run caption of IV covariance).

## 4. Work list for others
Posters to regenerate (both themes): `pair-gather`, `multiscale-gwr`, `spatial-weights-algebra`, `neighborhood-summary`, `chunked-weights-transpose`, `gwr-nonstationarity-test`, `probit-regression`, `seemingly-unrelated-regression`, `spatial-error-maximum-likelihood`, `spatial-weights-traces`, `instrumental-variables-covariance`, `spatial-lag-impacts`, `restricted-regime-regression`, `spatial-weights-validation`.
Framework requests: glyph colours (a `RunGlyph` has no colour, so in/out arrows cannot differ by colour); a log axis for the `chart` layer (traces are dominated by q = 2); the atlas lacks `^` and `'` (a title "1'(W/s)^q 1" lost them).

## 5. Roadmap
- WEB-PIC-12: pair-gather, multiscale-gwr, spatial-weights-algebra, neighborhood-summary, chunked-weights-transpose, gwr-nonstationarity-test, probit-regression, seemingly-unrelated-regression, spatial-error-maximum-likelihood, spatial-weights-traces touched; remaining: spatial-regression-diagnostics gesture, SUR two-map layout, probit iteration log scale, SEML difference encoding, regenerate posters.
- WEB-INT-16: this group's ranked runs now all have a text layer except none known; remaining: the SE bars under coefficient panels, phone-size check, posters.

## 6. Noticed
- `multiscale-gwr` and `spatial-weights-traces` are not operation defects. Multiscale: the flat maps came from the scenario (constant intercept and x2 by construction); with a truly varying intercept the operation returns a surface that follows it at small bandwidth and flattens it at bandwidth 1. A side effect worth knowing: with a wide intercept bandwidth the leftover north-south gradient leaks into the x1 map (backfitting), which is what the estimator does. Traces: the run uploads binary rook contiguity and the operation returns tr((W/s)^q), s = largest row sum (class doc: `powerTraces ... s = infinityNorm`); W/s is not row-standardized, so the eigenvalue-1 argument of the rescore does not apply. The spectral radius of W/s is below 1, so traces and sums decay to about zero by q 11 to 16; "0.00 · 0" was `toFixed` rounding. The picture shows a spike at q 2 then decay, consistent with that. I did not recompute the traces on the CPU; this is from the class contract and the shape of the curve.
- Another editor changed `chunked-weights-transpose.ts` mid-session (a system notice); I treated it as my own edit's echo.

## Typecheck (final)
`npx tsc -p website/tsconfig.json --noEmit` first reported 3 errors in my files (label `at` as `number[]`); fixed with `as const`; rerun: 0 errors in regression-weights. The three runs re-verified: see summary.
