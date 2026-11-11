# w3-web-processing-trajectories

## Done
All under `website/src/gpu/runs/entries/`. Screenshots: `website/.reports/w3-web-processing-trajectories/<group>/<id>-{dark,light}.png`.
- `geometry-processing/line-loop-removal.ts`: before, a faint grey input under a white output; after, the input is red under a bright output, so only the removed loops show red; legend "kept by the operation / loop removed".
- `geometry-processing/normalize-geometry.ts`: before, the same tiny glyphs on both halves, colored by shuffled rank; after, each panel lists the 12 leaves in child-order slots (input shuffled, right panel in the order the operation wrote, via a GPU slot kernel from its `rank` output), colored by leaf kind (ring, line, point). The right panel visibly groups rings, then lines, then points. Hover snaps to the nearest slot (never lands on nothing), rings the slot in both panels and draws an arrow: "ring: input #2 -> canonical #1". Titles and kind legend.
- `geometry-processing/normalize-polygons.ts`: before, small white and orange dots; after, input ring starts are large orange dots (radius 10) that move with the aim, canonical starts are blue dots on top that stay put; legend.
- `recipes-trajectories/clock-encounters.ts`, `trajectory-encounters.ts`, `trajectory-closest-approach.ts` plus new `recipes-trajectories/lib/pair-links.ts` (GPU kernels over the operation's own pair columns `partners`, `firstBuckets`, `closestTimes`; parked rows off screen). The three are now different: clock = dots colored by clock instant (early to late) and white links between met pairs at the first shared instant, moved by the pad; encounters = grey resampled dots and red links at the first shared bucket; closest approach = raw fixes and orange links at the exact closest time, which lies between fixes. Each has its own legend.
- `geometry-processing/geometry-orientation.ts` and `shared/shape-segments.ts` (`addRingArrows` gains an optional `scale`, default 1): arrows 1.7x, input arrows now bright, input and output rings colored by the GPU `reversedRings` flag (orange reversed, blue kept), legend.
- `geometry-processing/lon-lat-polygon-preparation.ts`: already rests across the date line with split pieces (the score note is stale); added a legend only. The harness probe positions never cross the line, so its readouts say "whole"; the rest picture shows the split.
- `geometry-processing/minimum-clearance.ts`: dots at both ends of each witness line (GPU kernel), clearance printed on each polygon (readback of the operation's `clearances` through `afterRun`), legend.
- `recipes-trajectories/dasymetric-redistribution-recipe.ts`: source zones are heavy ink outlines with the total each hands down; legend for source, target fill, ancillary texture.
- `recipes-trajectories/hot-spot-analysis-recipe.ts`: epicentre dots are now ink and larger, legend. Hot cells only; no cold cell exists in the data so no cold entry.
- `recipes-trajectories/segregation-counterfactual-recipe.ts`: chart axes (0, 0.25, 0.5, "dissimilarity D"), bar names, values at the bar foot (read back from the operation), legend naming the scenario shown on the map.
- Labels added (text layer): `geometry-validity` (5-class legend; polygons now fill the bottom 80%, so the legend has a band), `canonical-geometry-equality`, `population-support-alignment-recipe`, plus the legends above.
- Catalog text: no entry describes the changed pictures; none edited.

## Verified
- `node scripts/verify-runs.mjs <15 ids> --themes dark,light --out .reports/w3-web-processing-trajectories`: 30 of 30 pass, no warnings. Read in both themes: line-loop-removal, normalize-geometry (dark, light, pointer sheet), normalize-polygons, geometry-orientation, lon-lat (dark, light), minimum-clearance, the three encounter runs (dark, one light), dasymetric, hot-spot, segregation, canonical equality, population alignment, geometry-validity; pointer contact sheets read for the recipes group.
- `npx tsc -p website/tsconfig.json --noEmit`: 0 errors. `npx biome check --write` on my files: clean. `check-catalog.ts`: 0 errors, 2 warnings (space-time-events, not mine).
- Not run: `snapshot-runs.mjs`, phone width, a normalize-geometry light-theme pointer sheet.

## Not done
- `study-window-preparation` bar labels (on the "next" list): not started.
- `geometry-orientation` gesture: it still reads "14 rings reversed" at every curtain position. I tried flipping `exteriorClockwise` by pointer y; the verifier then failed "no slider changes the picture" (the gesture overrides the slider while at rest), so I reverted it.
- `lon-lat-polygon-preparation` "reads whole at every probe position": not changed (rest already splits; probe positions are the harness's).

## Work list for others
Regenerate posters (both themes): line-loop-removal, normalize-geometry, normalize-polygons, geometry-orientation, lon-lat-polygon-preparation, minimum-clearance, geometry-validity, canonical-geometry-equality, clock-encounters, trajectory-encounters, trajectory-closest-approach, dasymetric-redistribution-recipe, hot-spot-analysis-recipe, segregation-counterfactual-recipe, population-support-alignment-recipe. No framework request.

## Roadmap
- WEB-PIC-12 (these groups): remaining is `study-window-preparation` labels and a gesture that changes what `geometry-orientation` computes.
- WEB-INT-16: remaining in my folders, `study-window-preparation`.
- WEB-PIC-15: the encounter trio is distinct now.

## Noticed
- A kernel binding more than 8 storage buffers fails ("binds 9 storage buffers, more than the device limit of 8"); the pair-link kernels are split for this.
- `createLegend` returns labels and rules: spreading it after your own `labels` key overwrites them; use `mergeText`.
- Canvas space for text has y down (0 at top), world space y up.
- A swipe or probe gesture that returns `values` for a slider parameter makes the verifier's slider check fail at rest.
