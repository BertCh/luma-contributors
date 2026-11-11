# label-runs report

All verification screenshots: `/Users/robertchristie/Documents/GitHub/luma-contributors/website/.reports/w2-labels/<group>/<id>-<dark|light>.png` (plus `-pointer-0..2.png`). Live site (port 3000) shots at card and hero size: `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/label-runs-shots/` (`live-cardcanvas-<id>-<theme>.png` is the 265 CSS px card canvas at 2x, `live-fn-<id>-<theme>.png` the function page hero). Script: `.../scratchpad/live.cjs`.

## 1. Done

Per run (files under `website/src/gpu/runs/entries/`):

1. `regression-weights/spatial-regression-diagnostics.ts`: y ticks 0/4/8/12, dashed rule labelled `5% critical`, the six test names (LM lag, LM error, robust lag, robust error, SARMA, Moran z) written down from the top of the plot (rotated; the tall free space above the short bars; at card size all six fit, they did not when written up from the baseline because the rule label displaced them). Left out: value labels (readout carries them), a y title (a statistic has no unit).
2. `network-routing-access/network-statistics.ts`: degree class under each bar (`0`..`11+`; lower bound of the class, log2 mode gives `0 1 2 4 8 ...`), node count above each non-empty bar (from the CPU copy already in `inputs.view`), title `node degree` or `node degree (log2 classes)`. Left out: y axis (it collided with the bar values on the card), labels on empty bars.
3. `regression-weights/spatial-weights-summary.ts`: each of the six bars named up the bar with its real count from the operation (`slots 1898`, `asymmetric 858`, `isolates 1`, `min degree 0`, `max degree 19`, `empty rows 1`); counts read back with `afterRun`. Left out: y axis (bars are log(1 + count) with an automatic range, so ticks would be invented).
4. `dataframe-pointcloud-records/class-breaks.ts`: curve y ticks at 1/10/100/1k/10k in energy with title `energy (fJ)`; each GPU break labelled with its energy at the right end of its rule (thinned when crowded); bars: class numbers, `rows` axis, count above each bar (`afterRun` readback of the GPU class sizes), title `class`. Left out: x axis of the curve (rank), count y title beyond `rows`.
5. `dataframe-pointcloud-records/color-scale.ts`: swatches numbered 1..paletteCount with title `palette class` (the strip is the legend), bars use the same numbers and a `rows` axis. Left out: domain stops. The breaks do not line up with the swatches for the threshold and quantile scales, so ticking them would mislead; and per-bar counts (11 px per bar on a card).
6. Shared coefficient panel (`regression-weights/shared/regression.ts`: `createCoefficientText`, `names` on `CoefficientPanel`, `drawCoefficientPanel(panel, {world})`; callers pass names): y ticks -2/0/2 with gridlines under the bars (zero line is one of them), coefficient names under the bars. `ordinary-least-squares`, `probit-regression`: `int x1 x2`. `spatial-error-gm`, `spatial-error-maximum-likelihood`: `int x1 x2 λ`. `spatial-lag-maximum-likelihood`, `spatial-two-stage-least-squares`, `spatial-panel-regression`: `int x1 x2 ρ`. `seemingly-unrelated-regression`: `1:int 1:x1 1:x2 2:int 2:x1 2:x2` (thins to every second on a card). Left out: the standard-error bars under the panel (SUR, probit marginal effects, and the SE row of the table models), a legend for "dots = true value" (caption says it), a y title (it covered the bar tops on the card).
7. to 9. `spatial-statistics/knox-test.ts`, `mantel-test.ts`, `global-permutation-test.ts` (via `shared/permutation-chart.ts`, new `createReferenceHistogramText`): title `shuffled Knox count` / `shuffled Mantel r` / `shuffled Moran I` (or Geary C, Getis-Ord G by the slider), round tick values of the statistic along the baseline (inside the plot, because the strip has no room below; the tick under the dashed mean line is skipped), and `observed <value> · p <pseudo p>` at the solid line. The bin range, observed value and p come from the same GPU readback the glyphs use (stored per histogram buffer).
10. `network-routing-access/network-cost-matrix.ts`: corner legend (`cost 0`, `cost 1.6+` with the ramp ends, `beyond costLimit` muted), `origin` and `destination` titles in the left-bottom and top-left corners. Left out: row and column ticks (14 by 14 cells fill the whole canvas, no margin; the node ids are in the readout).

Other files: `regression-weights/shared/labels.ts` (new, `createBarNames`, up or down the bars), `scripts/verify-runs.mjs` (below), `src/gpu/runs/README.md` (new "Text" section).

Verifier change: after the inputs-only render, `verify-runs.mjs` reads `state.layerKinds` (layers of the latest render) and fails with "no output layer besides text" unless the full render has at least one more non-text layer than the inputs-only render. The existing `outputShare` check is untouched. Unlabelled `spatial-lag`: old report and new run are identical in nonBackground, outputShare, restoreDifference, slider changes and pointer numbers. Labelled `spatial-regression-diagnostics`: the picture legitimately gained label pixels, so nonBackground 0.2788 to 0.2857, outputShare 0.2703 to 0.2772, slider changes within 0.003; failures and warnings unchanged. I did not make a throwaway text-only run to see the new failure fire; the check is a plain count.

## 2. Verified

- `node scripts/verify-runs.mjs --themes dark,light --out .reports/w2-labels <ids>`: all 17 runs pass in both themes (20+6+4+4 pass lines; 0 failures, 0 warnings, no rest-difference warning, pointer rest difference 0 for all). Runs were verified per id in the verifier (it runs each separately), in four invocations by group.
- Screenshots read, dark and light: `spatial-regression-diagnostics`, `spatial-weights-summary`, `network-statistics`, `class-breaks`, `color-scale`, `network-cost-matrix`, `knox-test`, `mantel-test`, `global-permutation-test`, `spatial-lag-maximum-likelihood`, `seemingly-unrelated-regression`, `probit-regression`, `spatial-panel-regression`, `spatial-two-stage-least-squares`, `ordinary-least-squares`. Not individually read in both themes: `spatial-error-gm`, `spatial-error-maximum-likelihood` (same helper and same layout as the lag model, which I read; they pass the verifier in both themes). Some were read only in one theme after a later small change (e.g. coefficient y title removal re-read in light only on 2SLS).
- Live site (port 3000), card size (265 x 165 CSS, 2x) and function page (887 x 499): `spatial-regression-diagnostics` (card light/dark, hero), `ordinary-least-squares` (card, hero), `class-breaks`, `spatial-weights-summary`, `network-statistics`, `knox-test` (cards). Label size reads right on both: about 9-10 px on a card, about 14 px on the hero.
- Gestures with labels: pointer sheets read for `spatial-regression-diagnostics`, `knox-test`, `network-cost-matrix`; the dashed gesture boxes pass beside or over the labels without hiding them (halo keeps them readable).
- `npx tsc -p website/tsconfig.json --noEmit`: 0 errors at the end. `npx biome check --write` on all touched files: clean.
- Not run: phone width (111 px) pictures; `snapshot-runs.mjs`, `yarn website-build` (not allowed).

## 3. Not done

- Standard-error bars under the coefficient panels, probit marginal-effect bars, SUR residual scatter: unlabelled.
- `spatial-regression-diagnostics`/`weights-summary` value labels (done for weights-summary inside the names; not for diagnostics).
- Phone-size check.

## 4. Defects and cautions (text layer and wiring)

1. The hover hint chip on a card (`WebGPU · <hint>`, DOM overlay) covers roughly the bottom 30 CSS px of the canvas until the pointer moves, and hides x labels placed there (`class-breaks`, `network-statistics`, `knox-test` baseline labels, `color-scale`). Labels near the bottom are therefore only seen after the first move. Not a text layer bug; the chip fades after the first move. If it matters, the chip placement or a bottom inset in the helpers is a framework decision.
2. `createChartAxes` with a category x axis puts the title at the top right of the plot, far from the category row; with a short plot it looks detached (`class` in `class-breaks`). Cosmetic.
3. `χ²` in a label is laid out as `χ^2` (the atlas has no superscript two), so I wrote `5% critical` instead.
4. Wiring: `draw` receives no `peek`, so a label that needs a GPU number needs `afterRun` (used three times) or a module-level store written from `resolve` (used for the permutation histograms). Works, but a run author has to know it; the README "Text" section says so.
5. Text layers laid out in `draw` run every render; I kept them to cheap CPU work.

## 5. Rest pictures that changed (regenerate posters; both themes)

The ten requested runs plus the rest of the shared helper's users:
`spatial-regression-diagnostics`, `network-statistics`, `spatial-weights-summary`, `class-breaks`, `color-scale`, `ordinary-least-squares`, `spatial-error-gm`, `spatial-error-maximum-likelihood`, `spatial-lag-maximum-likelihood`, `spatial-two-stage-least-squares`, `spatial-panel-regression`, `seemingly-unrelated-regression`, `probit-regression`, `knox-test`, `mantel-test`, `global-permutation-test`, `network-cost-matrix`.
Also changed by the shared helper, not requested and only run through the verifier once: `spatial-panel-error-ml` (4 coefficients, now gets `int x1 x2 λ` and ticks). `spatial-panel-lag-ml` (5 coefficients) and `spatial-regime-regression`, `restricted-regime-regression` are unchanged (no names given, so no text).
Only text was added; no layer, range or default changed, so posters differ only by labels.
