# web-text: WEB-INT-16, text in pictures

## Decision (WEB-INT-16)

Glyph atlas layer drawn on the GPU in the scene, not a DOM overlay.

1. Poster equals live picture: `snapshot-runs.mjs` takes `page.locator('canvas').screenshot()` in the harness, which has no DOM overlay, so DOM text could only reach posters by changing the harness and the poster pipeline. Text drawn into the canvas is in every capture, in both themes (colors come from `DemoTheme`).
2. Sizes measured on the running site (1440x900 window): card canvas 265x165 CSS (307x191 at 1024 wide, 111 wide on a 390 phone), hero 887x499 CSS (709x399 at 1280). Device pixel ratio 1 to 3. The harness renders posters at 960x540, dpr 1, shown at 265 CSS px.
3. Label size is in text pixels (CSS pixels on a card): ticks 9.5, labels 10, titles 10.5. On wider pictures it grows as width^0.45, capped at 1.45x (hero about 14-15 CSS px). Posters pass `displayWidth = 265`, so their labels have the card's proportions (checked: poster-on-card and the live card match).
4. No blur on scaling: signed distance field (24 texels per em, reach 5), outline computed per pixel, so text is sharp at 9 to 40 device pixels. Pivots and axis-aligned rules snap to whole pixels.
5. No network and no layout shift: the atlas is generated once at build time (`generate-font-atlas.mjs`) from KaTeX_SansSerif in the installed `katex` package and checked in as base64 (112 KiB of source, 81 KB of texels). Nothing is fetched and nothing is in the DOM.
6. Cost: one extra pipeline and one shared 81 KB atlas buffer per device. Measured in the preview: TextLayer creation plus layout 5 to 11 ms per picture (two layers), against today's 668 ms median start. Layout reruns only when the text or view changes.
7. Accessibility: labels repeat what the readout (`aria-label`) and the catalog text already say; runs must not put information only in labels.
8. Uses only `@luma.gl/core` (Buffer, Device, RenderPass) and `@luma.gl/engine` `Model` (through the kit's `DemoLayer`, plus `model.setBindings`), all public. The atlas is a storage buffer sampled by hand in WGSL, so no texture or sampler API is needed.

A DOM overlay was rejected: it would not appear in posters, would load the site's Google web fonts (late load, layout shift), and would need a second positioning path kept in sync with resize and the pointer glide.

## 1. Done

New files only, all under `website/src/gpu/runs/text/`:

- `generate-font-atlas.mjs`: build-time generator. Headless Chromium rasterizes each glyph at 4x from font bytes read from `node_modules/katex` (as an ArrayBuffer, so nothing is fetched). The script computes an exact Euclidean distance transform, shelf-packs 122 glyphs (ASCII, en and em dash, °, −, ×, ±, ⋅, …, μ, ′, ≤, ≥, arrows, √, ∞, ≈, Δ, Σ, σ, ρ, λ, α, β, θ, χ) into 512x159 texels, and writes `font-atlas.ts`. Characters neither KaTeX font has (` · ² ³ µ ≠) are replaced at layout time (`·` with `⋅`, `²` with `^2`, ...); anything else draws as `?`.
- `font-atlas.ts`: generated atlas and metrics. The header names the KaTeX fonts (MIT package; the upstream fonts are SIL OFL 1.1).
- `text-layout.ts`: the CPU half with the declarative API. Types: `RunLabel` (text, `at` in world or canvas space, `anchor`, `offset` in text px, `role` tick/label/title/value, `tone`, `swatch` as a category index or RGB, `angle` 0/90/-90, `priority`, `group`), `RunRule` (axis lines, ticks, dashed references, background panels, with `fromOffset`/`toOffset` in text px), and `RunText`. The layout slides a label that pokes off the canvas back in, thins each group evenly (every 2nd, 3rd...) until it does not overlap itself, then drops lower-priority labels that overlap one already placed. It returns `placed`/`dropped`. Halos are emitted in a pass before all ink, so a halo never covers a neighbouring glyph.
- `text-layer.ts`: `TextLayer extends DemoLayer`: one instanced quad per glyph, swatch or rule, field sampled bilinearly from an `array<u32>` atlas, background-colored halo (1.4 text px, 82%). One atlas buffer per device, reference counted. `getTextPalette(theme)` gives ink, muted ink (about 8:1 contrast on dark and 5.4:1 on light) and the categories.
- `labels.ts`: helpers. `createChartAxes({rectangle, x, y, references})` gives a value or category x axis and a y axis whose values sit inside the plot above its gridlines (no left margin needed). The y title goes on the top gridline, the x title at the right end, dashed reference lines get labels, and `grid` is returned separately so it can be drawn under the data. Also `createBarValueLabels`, `createLegend` (corner, title, swatches, panel), `createMatrixTicks` (rows, columns, optional rotated columns, titles; same rectangle convention as `createMatrixMesh`), `getChartRectangle` (the chart layer's default fractions), `getNiceTicks`, `formatTick` (true minus, k/M) and `mergeText`.
- `index.ts`: barrel, plus the `TextRunLayer` descriptor `{layer: 'text', context?, alpha?, clip?} & RunText`.
- `preview/text-preview.html`, `preview/text-preview.ts`: a standalone development page, outside the registry and the catalog, served by the harness Vite server: `http://localhost:3301/src/gpu/runs/text/preview/text-preview.html?case=axes|bars|legend|matrix&theme=dark|light&size=card|hero|poster`. It uses the real run layers (`ChartLayer`, `MeshLayer`, `SegmentListLayer`), `createTracts`, `createMatrixMesh` and two text layers (grid under the data, labels over it), all on the Chicago tract extract.
- `preview/capture-text-preview.mjs`: photographs all 24 combinations on WebGPU, plus each poster put through the site's pipeline (640 px WebP at q 0.62, then 530 px for a 265 CSS card at 2x), and writes a contact sheet and `results.json`.

Screenshots (read): `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/web-text/shots/`
- `contact-sheet.png`: one row per case and theme. Left to right: live card (265x166 at 2x), poster as shown on a card, hero (887x499 at 2x).
- `axes-*`: ranked tract poverty line with x and y ticks, `poverty (%)` on the top gridline, `tracts, ranked` at the end of the x axis, a dashed `median 27.0%` reference. The 150 tick is dropped because it collides with the x title.
- `bars-*`: histogram with category labels `0–10` ... `70+` (all eight fit on the card), counts over the bars, y ticks, and the titles `tracts` and `% in poverty`.
- `legend-*`: tract map in four poverty classes, with a `below poverty line` legend on a translucent panel in the top-right corner.
- `matrix-*`: 4x4 class join-count matrix with row names, short column names, `tract` and `contiguous neighbour` titles, and a count in each cell.
- Every picture reads at card size in both themes. Poster-on-card and the live card match in size and placement. The hero text is about 14 CSS px. The first capture showed `0` read as `C` before an en dash (the next glyph's halo covered it); the two-pass halo fixed that and the change is in the current captures.

## 2. Verified

- `npx tsc -p website/tsconfig.json --noEmit`: no errors in `website/src/gpu/runs/text/**`. The only errors are in other agents' files (`space-time-models/conformal-prediction-intervals.ts`, `raster-time-series/streaming-accumulator.ts`), in progress.
- `npx biome check --write website/src/gpu/runs/text/*.ts website/src/gpu/runs/text/preview/*.ts`: clean. The `.mjs` scripts are outside biome's includes.
- `node website/src/gpu/runs/text/generate-font-atlas.mjs`: 122 glyphs, 0 kerning pairs (the KaTeX font has none that canvas reports).
- `node website/src/gpu/runs/text/preview/capture-text-preview.mjs <shots>`: 32 pictures, 0 GPU validation errors (one earlier run failed while Vite reloaded after my own edit; the rerun passed).
- Wiring diff: applied to copies, which were typechecked inside the tree as temporary files together with a one-run usage (`draw: [{layer: 'text', ...createLegend(...)}]`). 0 errors; the temporary files are deleted. The patched `harness.ts` was not typechecked (it is a two-line change).
- Not run, as instructed: `yarn website-build`, `snapshot-runs.mjs`, `verify-runs.mjs`. No unit specs: the website has no test project. `layoutText` is pure and could get a node spec.

## 3. Not done

- Wiring into `defineOperationRun`: the files are frozen this wave. The exact diff is below and in `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/web-text/wiring/text-wiring.diff`.
- README section for run authors (`website/src/gpu/runs/README.md`): left for the wiring change. Suggested text: a `text` row in the layer table, and "list `{layer: 'text', ...axes.grid}` before a chart and `{layer: 'text', ...axes}` after it".
- No run uses labels yet.

## 4. Wave 2 requests

No renames. Changes in files I do not own: the diff below (`define-run.ts`, `types.ts`, `harness.ts`), then a poster regeneration for the runs that adopt labels.

```diff
--- a/website/src/gpu/runs/define-run.ts
+++ b/website/src/gpu/runs/define-run.ts
@@ -46,6 +46,7 @@
 } from './interaction';
 import {ChartLayer, EdgeLayer, MeshLayer} from './layers';
 import {getGlyphMetrics, getPixelOfWorld, PointerOverlay} from './pointer-overlay';
+import {TextLayer, type TextRunLayer} from './text';
 
@@ -301,7 +302,8 @@
   | PathsRunLayer
   | TrianglesRunLayer
   | PolygonsRunLayer
-  | ChartRunLayer;
+  | ChartRunLayer
+  | TextRunLayer;
 
@@ -1041,6 +1043,8 @@
             owned: []
           };
         }
+        case 'text':
+          return {layer: new TextLayer(device), owned: []};
       }
     };
 
@@ -1113,6 +1117,14 @@
             look: {...look, radius: layer.radius ?? (layer.kind === 'scatter' ? 3 : 1.6)}
           };
         }
+        case 'text':
+          // Laid out in TextLayer.update, which reruns the layout only when the text or view changes.
+          (layerObject as TextLayer).setText(layer, {
+            displayWidth: context.displayWidth,
+            context: layer.context,
+            alpha: layer.alpha
+          });
+          return {layer: layerObject, look: {}};
       }
     };
 
@@ -1509,5 +1521,7 @@
       return [layer.positions, layer.indices, layer.owners, layer.values, layer.triangleCount];
     case 'chart':
       return [layer.kind, layer.values, layer.xValues];
+    case 'text':
+      return [];
   }
 }
--- a/website/src/gpu/types.ts
+++ b/website/src/gpu/types.ts
@@ -15,6 +15,11 @@
   /** Size in device pixels. */
   width: number;
   height: number;
+  /**
+   * CSS width the picture is shown at, which sizes its text. Default: `width` over the device
+   * pixel ratio. Posters, captured large and shown on cards, pass the card width.
+   */
+  displayWidth?: number;
   theme: GpuTheme;
--- a/website/src/gpu/runs/harness.ts
+++ b/website/src/gpu/runs/harness.ts
@@ -11,6 +11,7 @@
 import type {GpuSceneInstance} from '../types';
 import type {RunDebug} from './define-run';
 import {getRunParameters, getRunScene, listRunIds} from './registry';
+import {CARD_DISPLAY_WIDTH} from './text';
 
@@ -138,6 +139,8 @@
     canvas,
     width: canvas.width,
     height: canvas.height,
+    // Posters are shown on catalog cards: size their text for the card, as the live card does.
+    displayWidth: poster ? CARD_DISPLAY_WIDTH : undefined,
     theme,
     reducedMotion: poster,
```

Usage after wiring:

```ts
draw: ({world, values}) => {
  const rectangle = getChartRectangle(world);
  const axes = createChartAxes({rectangle, x: {range: [0, maximumDistance], title: 'r (km)'},
    y: {range: [0, 1.2], title: 'K(r)'}, references: [{value: 1, label: 'CSR'}]});
  return [{layer: 'text', ...axes.grid}, {layer: 'chart', kind: 'line', values: outputs.k, range: [0, 1.2]},
          {layer: 'text', ...axes}];
}
```

Wiring notes:
- On the function-page hero, the poster (the fallback before the canvas is live) shows card-proportioned text, about 3x the live hero's size, until the canvas fades in. Separate hero posters would remove that jump.
- `verify-runs.mjs` counts a non-context layer as output. A run whose only non-context layer is `text` would pass the "output is drawn" check. Consider excluding `text` from that check.
- The 16:9 harness posters are cropped to 16:10 on cards (an existing behaviour). Labels near the left or right edge can be cut by the crop, so keep labels away from the outer 5%.

## 5. Roadmap

WEB-INT-16: replace the row with "Decided and built: glyph atlas text layer (`website/src/gpu/runs/text/`, preview page). Remaining: apply the wiring diff to `define-run.ts`, `types.ts` and `harness.ts` (from report web-text), add the README section, adopt it in the runs listed in the web-text report, and regenerate their posters." In section 7.8, the "Where glyphs are drawn" row: "Text too: SDF glyph atlas in the scene, sized for the display width; posters pass the card width".

Ranked: the 30 runs that most need labels (from `dev-docs/research/card-interaction-audit-2026-11-09/entries.md`, audit score in brackets; each file was re-checked for its chart, matrix or categorical layer):

1. `spatial-regression-diagnostics` [1]: bar labels LM lag, LM error, robust lag, robust error, SARMA, Moran z; a 3.84 reference labelled χ² 5%.
2. `network-statistics` [1]: category labels and values on its three bars (degree, length, ...).
3. `spatial-weights-summary` [2]: labels and values on S0, isolates, asymmetric pairs.
4. `class-breaks` [1]: y ticks on the sorted-value curve, a rule and value label at each GPU break.
5. `color-scale` [1]: legend strip with the domain stops ticked and a swatch per class.
6. Shared regression coefficient panel (`regression-weights/shared/regression.ts`: `ordinary-least-squares`, `spatial-error-gm`, `spatial-error-maximum-likelihood`, `spatial-lag-maximum-likelihood`, `spatial-two-stage-least-squares`, `spatial-panel-regression`, `seemingly-unrelated-regression`, `probit-regression`) [2-3]: coefficient names under bars, a zero reference, values.
7. `knox-test` [1]: shared permutation chart: x ticks of the statistic, "observed" reference with its value, pseudo p.
8. `mantel-test` [1]: same permutation chart.
9. `global-permutation-test` [2]: same permutation chart.
10. `network-cost-matrix` [2]: origin and destination matrix ticks, cost legend.
11. `adjacency-matrix-order` [1]: matrix ticks (node ranges) and "input order" / "ordered" titles.
12. `lisa-markov` [2]: HH/HL/LH/LL matrix ticks, from/to titles, quadrant legend.
13. `spatial-markov` [2]: title per matrix (low ... high lag), class ticks.
14. `transition-matrix` [3]: class ticks, from/to titles, probabilities in the cells.
15. `rank-markov` [2]: rank-class ticks, period label.
16. `instrumental-variables-covariance` [2]: a 3x3 matrix with coefficient names on rows and columns.
17. `spatial-lag-impacts` [2]: direct, indirect, total under the bars, a zero rule.
18. `restricted-regime-regression` [2]: bar labels, an F critical-value reference.
19. `weiss-landforms` [2]: 10-class legend (canyon ... mountain top).
20. `change-detection` [2]: three-entry legend decrease / none / increase.
21. `geometry-validity` [2]: legend of the five validity classes.
22. `canonical-geometry-equality` [2]: equal / not equal legend.
23. `population-support-alignment-recipe` [2]: matched / absent / repeated legend.
24. `spatial-weights-validation` [2]: four-class legend.
25. `multiple-testing-adjustment` [2]: alpha reference labelled, p ticks.
26. `point-horizon-profile` [3]: N/E/S/W x ticks, a 0° reference.
27. `rose-statistic` [2]: N/E/S/W ticks.
28. `calendar-buckets` [3]: hour ticks 0, 6, 12, 18 and weekday initials.
29. `space-time-variogram` [2]: distance and time-lag ticks with units.
30. `terrain-critical-points` [3]: peak / pit / saddle under its bars.

Next after these: `segregation-counterfactual-recipe` and `study-window-preparation` (bar labels), `viewshed` and `point-horizon-visibility` (visible / hidden legend), `network-k-function`, `network-inhomogeneous-k`, `ripley-*` and the variograms (r axis, CSR reference).

## 6. Noticed

- `website/src/gpu/runs/entries/space-time-models/conformal-prediction-intervals.ts` and `raster-time-series/streaming-accumulator.ts` fail the website typecheck (`unknown` outputs) while in progress in another stream.
- `THIRD-PARTY-DATA.md` does not list fonts. If the atlas ships, a line for the KaTeX fonts (SIL OFL 1.1; the atlas is a rendering of them) belongs there or in a site notices file. I did not edit it.
- The catalog card is 111 CSS px wide on a 390 px phone. Labels there fall to the 0.85x floor (about 8 CSS px) and collision thinning drops most ticks. Runs should keep labels few.

## Follow-up (review of the text layer)

All changes are in `website/src/gpu/runs/text/**`; the framework files are untouched.

### 1. Licence

What the font says. Both font files carry the notice in their own name table (record 0: copyright; 13: licence; 14: URL):
- `KaTeX_SansSerif-Regular.ttf`, Version 1.1, sha256 `f36ea897…a4ad`.
- `KaTeX_Main-Regular.ttf`, Version 1.1, sha256 `d0332f52…a830`.
- Both from `katex` 0.16.47, `node_modules/katex/dist/fonts`.
- Notice: "Copyright (c) 2009-2010, Design Science, Inc. (<www.mathjax.org>) / Copyright (c) 2014-2018 Khan Academy (<www.khanacademy.org>), with Reserved Font Name KaTeX_SansSerif [KaTeX_Main]. This Font Software is licensed under the SIL Open Font License, Version 1.1."
- The npm package ships no OFL text file; its `LICENSE` is the MIT licence of the KaTeX code.

OFL conclusion: the atlas is a Modified Version of the Font Software. The OFL defines a Modified Version to include changing formats, and the atlas both subsets the fonts and converts them to a distance field. So:
- Condition 5: the atlas files are distributed under OFL-1.1 only. `font-metrics.ts` and `font-pixels.ts` (advances and kerning are font data too) now carry `SPDX-License-Identifier: OFL-1.1` and one `SPDX-FileCopyrightText` line per holder (Design Science, Inc.; Khan Academy). Holders are read from the fonts at generation time, not from vis.gl.
- Condition 2: every copy must contain the copyright notice and the licence. The header quotes each font's licence record. The full OFL 1.1 text and the copyright are also kept as data (`FONT_LICENSE` in `font-metrics.ts`, a property of an object the layout uses, so minification cannot drop it). `font-pixels.ts` carries a short notice in its data. Bundled JavaScript loses comments, which is why the notice is also kept as data.
- Condition 3: the Modified Version must not use the Reserved Font Names KaTeX_SansSerif and KaTeX_Main. It is named "Gridlight Label Atlas" and presented under that name. Naming KaTeX only as the source is acknowledgement, which condition 4 allows.
- Condition 1 (not sold by itself) is met: it ships inside the site.
- Pictures and posters drawn with it are documents created with the font, and the OFL does not apply to them.
- The rest of the code (layout, layer, generator) stays MIT, because it is not derived from the font.

Generator changes:
- It reads the name records and refuses a font without an OFL notice.
- It writes the header described above and records the katex version, both file names, their font versions and SHA-256 hashes, and the Chromium build (`Chromium 151.0.7922.34 (playwright)`; review item 8).
- When the font is missing, it exits with: `generate-font-atlas: <path> is missing. The label font comes from the katex package: add it as a devDependency of the website workspace ("katex": "0.16.47" in website/package.json), run yarn install, then run this again.`

devDependency line for `website/package.json` (an exact pin, because the atlas records the version):

```json
    "katex": "0.16.47",
```

The `THIRD-PARTY-DATA.md` entry: the right place is its "Website demo extracts" section. The file covers distributed website assets and their terms. Its preamble excludes only ported source code, which carries SPDX headers, and the atlas is generated font data rather than ported source. Draft paragraph, to go after the 3DBAG paragraph:

> The label font of the website's pictures, "Gridlight Label Atlas" (`website/src/gpu/runs/text/font-metrics.ts` and `font-pixels.ts`), is a Modified Version (a subset rendered as a signed distance field) of KaTeX_SansSerif-Regular and KaTeX_Main-Regular from KaTeX 0.16.47. © 2009-2010 Design Science, Inc.; © 2014-2018 Khan Academy. Reserved Font Names KaTeX_SansSerif and KaTeX_Main, which the atlas does not use. Licensed under the [SIL Open Font License 1.1](https://openfontlicense.org) (`OFL-1.1`), whose full text ships with the data (`FONT_LICENSE`). Regenerate with `generate-font-atlas.mjs`. Pictures and posters drawn with it are not covered by the OFL.

### 2. Bundle cost

- The atlas is split in two:
  - `font-metrics.ts` (14 KB: advances, kerning, cells and the licence) loads with the layout.
  - `font-pixels.ts` (125 KB) is reached only through `import('./font-pixels')` in `font.ts` (`loadFontPixels`, `getFontPixels`).
- A `TextLayer` created before the texels arrive binds a placeholder buffer and draws nothing. When the texels load it rebinds them and calls `onReady`, so its owner can draw again.
- The wiring awaits `loadFontPixels()` before the first `render()` only when `draw` lists a `text` layer. The first frame and posters always have their labels (the harness awaits the scene), and runs without text never load the chunk.
- `CARD_DISPLAY_WIDTH` moved to `text/display.ts`, so the harness imports a single constant.
- The wiring imports `TextLayer` and `loadFontPixels` from their own files, and the `TextRunLayer` type from `./text`; the type import is erased.
- Remaining static cost for every run: the text layer code, the layout and the metrics, about 45 KB of unminified source. I did not measure minified sizes because a build was not allowed. Moving `TextLayer` behind the same dynamic import would remove that too, but the framework's `createLayer` is synchronous, so I left it static.

### 3. Own uniform block

The shader no longer includes `PALETTE_STRUCT` or reads `Look`. It has `TextUniforms` at binding 3 (canvas size, atlas size, field reach, halo width and strength, opacity, background), written by `TextLayer.update`, which no longer calls `DemoLayer.update`. Binding 0 is declared as an unused `vec4f` only because `DemoLayer` always binds its look buffer: without the declaration luma.gl logged "Binding look not set: Not found in shader layout". After the declaration, 0 console warnings and 0 errors across the 40 captures. A change to `kit.ts`'s `Look` cannot affect text.

### 4. Device pixel ratio 1

- Added a `card1` size (265x166 at ratio 1) to the preview.
- With `pixelsPerTexel < 0.75` the fragment shader now averages four field samples per pixel.
- The field reach went from 5 to 6 texels (atlas 512x180), so the halo at ratio 1 reaches about 1.5 px instead of being clamped at 1.1.
- I inspected magnified crops of `bars-light-card1` and `matrix-dark-card1` (`/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/web-text/mag-bars-light.png`, `mag-matrix-dark.png`). Stems are even and no glyph breaks up; text at ratio 1 is soft rather than aliased, and legible in both themes.
- I did not capture a ratio 1 "before" image to compare against.

### 5. Helper defects

- `createBarValueLabels` takes `baseline` (default `range[0]`, which is how the `chart` layer draws bars, from the bottom of the range). A value below the baseline is labelled under the bar's end (anchor `top`), so bars drawn from a zero line get correct labels.
- `createLegend` sizes swatches with the new `getSwatchWidth(role)`. It uses the atlas cap height (0.694) and the same rounding as the layout, instead of 0.72.
- The layout key now includes `theme.categories` (review item 5).

### Verified (follow-up)

- `node website/src/gpu/runs/text/generate-font-atlas.mjs`: 122 glyphs, 512x180; metrics 12.1 KiB, pixels 122.1 KiB before formatting.
- `npx biome check --write website/src/gpu/runs/text/`: clean.
- `npx tsc -p website/tsconfig.json --noEmit`: 0 errors in `runs/text`.
- `node website/src/gpu/runs/text/preview/capture-text-preview.mjs …/web-text/shots2`: 40 pictures (4 cases, 2 themes, card at 2x, card at 1x, hero, poster), 0 GPU errors, 0 console warnings.
  - Half of the pictures took the lazy path (layers first, atlas later, redraw on `onReady`). All of them show their labels.
  - Atlas load and decode took 3-7 ms when awaited. Text layers plus layout took 4-11 ms.
  - I read the contact sheet and the ratio 1 crops: labels match across card, poster-on-card and hero in both themes.
- Wiring diff, rebuilt from fresh copies of the frozen files: typechecked as temporary in-tree copies with a one-run usage, then deleted. `define-run` and `types` copies: 0 errors. The harness copy reports `displayWidth` unknown, but only because the check copy still sees the registry's original `GpuSceneContext`. With `types.ts` patched in place it resolves.
- Diff saved at `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/web-text/wiring/text-wiring.diff` and `dev-docs/research/parallel-pass-2026-11-10/text-wiring.diff` (104 lines).

Wiring diff changes against the first version:
- `define-run.ts`:
  - imports `TextRunLayer` (type), `loadFontPixels` (`./text/font`) and `TextLayer` (`./text/text-layer`) separately;
  - creates `new TextLayer(device, () => { framesToDraw = 3; scheduler.request(); })`;
  - adds `if (getLayerList().some(layer => layer.layer === 'text')) await loadFontPixels();` before the first `render()`.
- `harness.ts` imports `CARD_DISPLAY_WIDTH` from `./text/display`.
- `types.ts` is unchanged from the first version.

Screenshots: `/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-luma-contributors/310b8947-6a4f-4e58-adb4-aef2ba8200cd/scratchpad/web-text/shots2/` (`contact-sheet.png`, now with a ratio 1 column).

Files changed or added: `font-metrics.ts`, `font-pixels.ts` (generated, OFL-1.1; replace the removed `font-atlas.ts`), `font.ts`, `display.ts`, `generate-font-atlas.mjs`, `text-layer.ts`, `text-layout.ts`, `labels.ts`, `index.ts`, `preview/text-preview.ts`, `preview/capture-text-preview.mjs`.
