# Statistics wave19 visual review

CPU-only inspection of actual screenshots from the root's desktop/mobile sweep. No browser,
GPU job or runtime source edit was used for this review. Served bundle SHA256:
`f4742c261276e85b022651df7f3c72f419a7f6d68300c30803839629c67b3675`.

All paths below are in `/tmp/luma-showcase-overhaul-20261025/final-wave19-all-stories/`.
These screenshot observations supplement, not replace, the numerical acceptance suites.

| Exact screenshot filename | Observation |
| --- | --- |
| `nature-density-mobile.png` | Density surface, locator, scale and observer-effort caveat render at 390 px. Drawer and navigation fit; source attribution is ellipsized. Drawer is scrolled into detail content, so the frame does not establish top-of-story readability. |
| `chicago-population-support-05.png` | Tree-cover summary is selected, map/legend use a 0–1 fraction, and exposure chart title refers to residents in low tree-cover cells. Modeled allocation versus observed counts is explicit. No stale tract-count chart visible. |
| `chicago-population-support-mobile.png` | Conservation residual evidence is readable and selected tree-cover layer persists. Confirmed issues: title cartouche largely clipped under top navigation; orange custom 500 m scale label clips at right; document width is 392 px in a 390 px viewport. Source attribution is heavily ellipsized. |
| `meuse-neighborhood-statistics-06.png` | Measured sites follow the river geography. Selected local Lee L, zinc legend and readouts agree; Gamma is explicitly descriptive without an analytic p-value. |
| `meuse-spatial-regimes-04.png` | Selected varying-intercept-and-river-distance profile maps fitted log zinc, matching its legend. All four profile statuses report 153/153 fits. No mismatched chart visible. |
| `meuse-spatial-regimes-mobile.png` | Residual-style diverging map and explanatory limitation fit the phone drawer. Title is above the visible map, consistent with the shared mobile containing-block issue. |
| `rainfall-interpolation-06.png` | Ordinary kriging with maximum focal statistic and radius 3 is visibly selected. Legend explicitly says window maximum and uses millimetres; no prediction-versus-variance mismatch. Gauge outlines remain visible. |
| `meuse-validation-03.png` | Log prediction selection matches zinc log-concentration legend. Surface interpretation correctly says log-metal prediction, 3,103 valid grid cells, zero insufficient-support and zero singular/unreliable solves. Prior unconditional variance wording is absent. |

Supplementary inspection of `point-patterns-07.png` found a clear question-led title,
rectangular study-window boundary, city-exclusion hatching and corresponding legend. Its
panel is scrolled to controls, so this screenshot does not expose the envelope chart.

## Vulnerability-index timeout diagnosis

Inspected `vulnerability-index-02.png`, `vulnerability-index-03.png`, and
`vulnerability-index-failure.png`. Step 2 has a complete blue Limited English quintile map
and both histograms despite the selected-site color heuristic returning zero. Step 3 has
the class-shift map, matching 172 changed/1 changed by two classes readouts, legend and weight
chart. The failure frame has the completed geometric-mean map, robust hatching, 43% first
component variance, 114 robust tracts and loading bars.

The wait is a harness false positive: `story-view.ts` assigns `data-readout-value` to chart
containers, and the PCA chart title is “Loading on the first principal component”. The
busy-text regular expression matches the noun “Loading”. Root accepted the harness-only
selector correction to exclude `.readout-chart`; no runtime computation change is justified.

## Mobile corrections proposed and authorized

The custom scale tick's mark is correct. Its label is centered at the mark even at the right
endpoint, so half the label overhangs the right-aligned bar. Root authorized clamping only
the label center to its measured text bounds within the bar, retaining the true tick mark;
a label wider than the bar is omitted while its mark and endpoint numeral remain.
The separate title-cartouche problem is owned by the atlas/root layout review: the mobile
absolute map lacks a positioned story-layout ancestor and begins under the sticky header.
Neither correction is accepted by these frozen wave19 screenshots; root will validate them
in a later bundle.

## Authorized scale-label correction: source and DOM verification

The bounded `map-furniture.ts` change clamps only the parameter-label center using the
existing rendered-font measurement helper. True tick positions are unchanged. A label
wider than the bar is omitted. The existing fixture gained two actual-renderer cases; no
separate mirrored implementation was introduced. Scoped Biome, syntax and showcase typecheck
passed in the Luna implementation review.

Executed the authorized DOM-only fixture (no scene or GPU workload):

```sh
node examples/deck/spatial-analysis-showcase/scripts/check-map-legend-spacing.mjs /tmp/luma-showcase-overhaul-20261025/wave20-map-legend-spacing
```

Result: 26 existing scale layouts, two endpoint cases, and nested SVG text bounds passed.
For the 60 px case the bar and label both end at pixel 388; the 12 px case retains the tick
and endpoint numeral while omitting its overwide accent label. Viewed
`/tmp/luma-showcase-overhaul-20261025/wave20-map-legend-spacing/legend-spacing.png`.
Raw report is copied to `statistics-wave20-scale-label-dom.json`. Source is frozen.
Actual mobile scene acceptance remains pending the parent's rebuilt-bundle tour.
