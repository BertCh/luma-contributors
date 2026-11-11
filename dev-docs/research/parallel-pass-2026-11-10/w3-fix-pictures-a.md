# w3-fix-pictures-a

## Done (finding: what, how checked; all by verify-runs in dark and light and looking at screenshots)
- B1 spatial-cross-validated-ols: lit-fold buffer filled with NaN; both bar sets now `drawSignedBars` (from the zero line). Negative coefficients visible again.
- M9 same run: trained stations in theme.muted; legend has two residual swatches (above and below observed) from theme.diverging.
- B2 multiscale-gwr: caption corrected. M1: maps on a diverging scale [-3, 3] centred on 0 (x2 = -1.5 is now clearly blue in both themes), three-swatch key bottom left. Minimum bandwidth: status codes now read with peek and shown in the readout ("N rows failed"); probed corners with all three bandwidths at 0.08: 0 rows failed and coefficients finite, so the range stays.
- B3 spatial-error-maximum-likelihood: titles anchored to the middle of each side of the moving curtain, placed above the map (looked at pointer frames).
- B4 spatial-pearson, global-spatial-statistics: signed bars from the zero line (`drawSignedBars`).
- M5 white dots: `drawCoefficientPanel` takes `ink`; all 6 callers pass theme.ink (probit, SUR, OLS, table-model, panel, regime); gwr-nonstationarity-test shuffled dots use theme.ink. Visible in light.
- M6 chunked-weights-transpose: links in theme.muted at 0.45 (visible in light). Hover arrows still short: they are k-nearest-neighbour links, local by construction (not changed).
- M8 pair-gather: queries radius 4; legend adds high/low gathered value and grey no-match disc.
- M4 point-pattern-envelope: band, outside dots, curtain, index and x ticks all use (i + 0.5) / count with the axis range padded by half a step; RANGE[1] raised 0.3 to 0.4 (observed reaches 0.324 at r = 0.20, read back).
- M3 multiple-testing-adjustment: alpha label moved to mid plot, swatch label priority 5 (now present at rest), significant dots radius 9, and for Bonferroni and Holm a label "smallest possible adjusted p: 0.12 (999 shuffles)" (floor 1/(P+1) times tests). No log axis or inset (not cheap).
- M2 spatial-balanced-sample: split constrained to the valid range (west keeps at least westCount units, east at least eastCount; pointer spans that whole range); the status buffer is read in afterRun and a rejected stratum would be titled "rejected, N asked of M". Verifier: all 5 positions give valid designs.
- M7 parameter-buffer: counters live in the scene inputs (no module state); label "this graph: built N x"; REACH_SPAN 0.12 with the shared getLogarithmicReach; pointer now gives tolerance 0.003 to 0.100 (was 0.002 to 0.021).
- M10 coverage-validity: gapWidth max 0.06 to 0.1 (parameters file), ring radius = gapWidth / 2, legend to bottom left. Saturates at 2 of 5 positions instead of 3 (readouts 0.100, 0.086, 0.100, 0.010, 0.008).
- M12 transition-diagnostics sparse band and swatch: theme.categories[1]; rose-statistic input legs mid blue. Ramp low end (neighborhood-summary, linear-referencing, gwr map) is a framework issue, untouched.
- Minors: outline-geometry comment; segment-ring-assembly "input segment" with ink swatch; group-concave-hull noise-point legend entry (label 3); line-segmentize "written vertex (lone dot: inserted)".
- Rest picture changed: spatial-cross-validated-ols, multiscale-gwr, spatial-error-maximum-likelihood, spatial-pearson, global-spatial-statistics, probit-regression, seemingly-unrelated-regression, gwr-nonstationarity-test, ordinary-least-squares and the table-model/panel runs (dots now ink), spatial-regime-regression, chunked-weights-transpose, pair-gather, point-pattern-envelope, multiple-testing-adjustment, spatial-balanced-sample, parameter-buffer, coverage-validity, transition-diagnostics, rose-statistic, group-concave-hull, segment-ring-assembly, line-segmentize.

## Verified
- `node scripts/verify-runs.mjs <ids> --themes dark,light --out .reports/w3-fix-pictures-a`: every batch all PASS (final batches 16/16, 8/8, 10/10, 6/6; last single reruns 1/1, 3/3). Folder deleted.
- `npx tsc -p website/tsconfig.json --noEmit`: no errors in my four folders after two fixes. `check-catalog.ts`: 0 errors, 2 warnings (space-time-events, not mine). `npx biome check --write` on my files.
- Not run: snapshots, card-size or phone legibility.

## Not done
- M11 spatial-weights-traces: log plot needs a kernel and the title/rule collision; left (claim itself was confirmed correct by the review).
- M6 longer hover arrows (geometry of kNN); building-adjacency-recipe orange clash; neighborhood-summary look-alike maps; spatial-weights-validation legend over nodes; probit rule splitting the x2 bar; transition-diagnostics "class" title; spatial-balanced-sample east title over dots; spatial-pearson input-scatter explanation; linear-referencing "road" swatch (code draws ink 0.5, swatch ink is right); chunked/low-priority pictures of other runs in the table.

## Work list for others
- Regenerate posters for the entry ids listed above. Framework request: a sequential ramp whose low end clears the background in light.

## Roadmap
- Remaining: WEB-PIC rows for spatial-weights-traces (log axis), chunked-weights-transpose (longer arrows), coverage-validity and parameter-buffer still saturate at far positions.

## Noticed
- The `bars` chart layer grows from the range minimum, so any signed bars under an axis need `drawSignedBars`; other runs in other groups may repeat the mistake.
