# Independent saved-map review — wave12b

Reviewed 20 saved desktop maps without opening a browser or using the GPU. Source remained
frozen. Evidence directory: `/tmp/luma-showcase-overhaul-20261025/wave12b-all-stories/`.
This review excludes known BIXI/county/footpoint/spherical camera and mobile-legend corrections
already integrated into wave13, and root's landforms agreement-legend correction.

## Remaining findings

1. **Meuse prediction readout describes variance.** `meuse-validation-03.png` shows Map quantity
   “Log prediction” and a log-concentration map/legend, but the primary Surface interpretation
   states “Conditional kriging variance (log units²).” The current compute reader chooses its
   diagnostic text from estimator only, independent of the displayed quantity. This can make the
   reader mistake a prediction surface for uncertainty. Make that readout quantity-aware or label
   it explicitly as the separately available diagnostic rather than the displayed surface.
2. **Scale numerals collide.** `ocean-seasonality-04.png` merges midpoint 500 and endpoint 1,000 km;
   `meuse-metal-model-form-04.png` merges 250 and 500 m; `cell-neighborhoods-05.png` merges 10 and
   20 km. `createScaleRow` always includes three absolute-positioned labels; narrow bars do not
   reserve the text extents. Omit the midpoint when it cannot clear both endpoints, retaining the
   numerical endpoints. This is separate from the previously removed orange analytical ticks.
3. **Largest proportional-size label is clipped.** `county-dorling-04.png` clips the top of
   “1000k” in its nested-symbol legend. `renderNestedSizeLegend` puts the largest top at y=1 and
   its label baseline at y=5; the font is 11px. Add a top inset to the SVG geometry/viewBox rather
   than changing the underlying radius scale.
4. **Cluster legend implies absent categories.** `observation-hierarchy-03.png` reports four
   clusters with sizes 22/16/16/12, but the legend names Cluster 1 through Cluster 8. The source
   returns a static eight-entry categorical legend. Show currently observed labels, or explicitly
   label this as a repeating colour key instead of asserting that eight clusters exist.
5. **Slowdown is tagged as closure.** `rail-resilience-04.png` correctly shows Severe slowdown
   and a 60-minute added cost, while its map badge remains “Hypothetical closure.” The title chip
   is static in current metadata. “Hypothetical disruption” covers both modeled modes accurately.

`cell-neighborhoods-05.png` also captures “Rebuilding path graph…” beside existing complete
identity marks. This may be the all-story harness capturing an asynchronous rebuild; the image
alone does not establish a persistent readout defect. Do not treat it as a numerical failure.

## Map sample

| Saved image | Result of this presentation review |
| --- | --- |
| ocean-seasonality-04 | Residual quantity and diverging legend agree; scale-label collision above. |
| spectral-evidence-05 | Mixture weight legend matches modeled coefficient; missing support stays blank and model caveat is visible. |
| spectral-texture-03 | Gradient magnitude, transect and legend agree; outside-boundary assumptions visible. |
| pixel-boundaries-04 | NIR bins match map and query outlines; missing support and non-classification caveats explicit. |
| grid-alignment-05 | Reflectance surface and support counts coherent; no misleading blank-result state. |
| terrain-conditioning-05 | Complete-profile network distinct from grayscale elevation background; status legend coherent. |
| point-cloud-change-04 | Sparse core status points match interval legend; iteration budget exhaustion visible. |
| survey-frame-sensitivity-03 | Corrected difference, missing-cell hatching and hypothetical control mask distinguished. |
| rail-resilience-04 | Scheduled-time legend coherent; closure badge incorrect for displayed slowdown. |
| harbor-continuous-encounters-03 | Within-distance intervals visible against source paths; clipped support limits explicit. |
| harbor-track-similarity-03 | No match is truthfully represented by query-only highlight and zero-survivor readout. |
| aviation-network-lab-04 | Broker ranking labels/colour agree; circles separately encode scheduled legs with note. |
| meuse-validation-03 | Map/legend agree on concentration; interpretation readout incorrectly says variance. |
| meuse-metal-model-form-04 | Uniformly negative coefficients are honestly subtle within a fixed diverging range; scale numerals collide. |
| county-dorling-04 | Displaced centers and geographic anchors distinguished; largest legend label clipped. |
| earthquake-query-lab-06 | Query membership and ranked stars distinguished; dark chart labels now readable. |
| observation-hierarchy-03 | Four visible cluster classes and noise agree with summary; legend incorrectly names eight. |
| cell-neighborhoods-05 | Shared canonical vertices align with first adjacent pair; scale numerals collide; possible rebuild capture noted. |
| footprint-shape-lab-06 | Raw offset curve distinct from source boundary; omitted closing edge is explained. |
| fire-scar-topology-05 | Accepted inside/outside classes agree with legend; missing support and internal-boundary limits visible. |

## Authorized shared fixes and independent verification

Root authorized findings 2 and 3 for this lane. Scale numeral placement now measures the actual
connected text with an invisible span inheriting the scale's typography and tabular-number
settings. The midpoint is omitted unless it clears both endpoints by four pixels. If even zero
and the unit-bearing endpoint cannot fit on one line, they use two font-relative lines; the bar
itself keeps its measured geographic length. The render cache includes connection state and
font settings, so connecting an initially detached scale or changing font size recalculates the
layout. The nested size SVG now reserves ten pixels above its largest circle/leader; all radii
and relative circle positions remain unchanged.

`scripts/check-map-legend-spacing.mjs` bundles the actual MapFurniture and legend modules and
runs 26 DOM cases in Chromium without WebGPU: 40/60/80/120-pixel bars, three label sets, 10.5/15px
fonts, plus detached-to-connected and 15→18px typography changes with an unchanged camera. It
asserts actual text rectangles do not intersect, readable wide scales retain the midpoint, and
narrow scales retain both endpoint labels. Every nested SVG text `getBBox()` must lie within
its viewBox, and radii remain 46/32.5/23 pixels. The fixture caught an initial canvas-based attempt
because computed `font` was empty under tabular-number styling; it also caught fixed stacked
line heights under enlarged typography. Both regressions pass after the final changes. The
resulting screenshot was visually inspected: `/tmp/showcase-map-legend-spacing/legend-spacing.png`.
The report is in the same directory. Scoped Biome checks pass. No production GPU or rebuild was
run; root owns final integration.
