# w3-web-pointcloud-raster

Stream `web-pointcloud-raster`. Files are under `website/src/gpu/runs/entries/`. Screenshots: `website/.reports/w3-web-pointcloud-raster/<group>/<id>-<dark|light>.png` plus `-pointer-0..2.png`.

## 1. Done

All twelve runs and the two label runs were changed. Each now draws the operation's real output with a text layer; none of them used thicker strokes or constructed input to raise a pixel number.

- `raster-registration` (raster-bands-indices): before, an empty ruler grid with a dot. Now the moving near-infrared image (left) and reference minus moving (right, a diverging ramp, subtracted on the GPU by GPURasterArithmetic) side by side as one 128x64 raster, with the pixel-ruler inset (true shift hollow, recovered shift solid, recovered value read back from the GPU) in the lower right. The difference fringes grow with the shift; the pointer sets the shift across the frame (x) and down it (y). Titles and ruler labels added. The ramps are scaled from the data (image extent, 99th percentile of cell steps).
- `pointcloud-voxel-merge`: before, sparse dots on black. Now the voxel grid, points coloured by tile (blue, orange), the seam as a line, merged centroids as dots, voxels both tiles reduced outlined (from the GPU tile counts), legend with the shared-voxel count. The seam drag moves the outlines.
- `pointcloud-icp`: before, uniform speckle. Now target (large grey), source before ICP (blue), source after ICP (orange, read back from the GPU transform), a line joining each before and after point, 400 points drawn per cloud, RMSE curve with axes (iteration, RMSE) and a legend.
- `spectral-unmixing`: before, a dim purple field. Now three panels (water, soil, vegetation fractions), gathered by one GPURasterSampling pass into one raster, same ramp 0 to 1, with titles and a key. The probe outlines the cell in all three panels; the inset bars are named (water, soil, veg, misfit). The slider `Output shown` now has two options, `three fractions` (default) and `residual` (the old full-frame misfit map): edit of `spectral-unmixing.parameters.ts`.
- `pointcloud-statistical-filter`: before, a few red dots among orange. Now rejected points (the GPU `validity` mask) are large red discs over small grey kept points, in the map and the profile, with a legend and titles. The pointer-level preview now uses the mean distances the GPU wrote (the CPU neighbour search was removed) and is drawn in amber only while the pointer level differs from the applied one, labelled "estimate".
- `pointcloud-neighborhoods`: the hovered point's neighbours are now read from the GPU table (`indices`, `counts`) instead of a CPU search: cyan dots and spokes in the map and profile; the readout gives found, kept and capacity. The resting probe sits on a dense stretch (85th percentile of neighbour counts). The red "truncated" dots were dropped (the table does not hold them).
- `pointcloud-height`: ramp range now comes from the GPU's tallest height (readback), legend ("ground 0 m", "bright: up to N m", theme-aware swatch), titles. The probe already rested on a tree.
- `glcm-texture`: before, a 16% strip of input and a window box. Now a swipe: grey input left of the pointer, the chosen feature right of it, window outline kept; titles and a note that the added noise grows to the right. Gesture kind changed from `window` to `swipe`.
- `extrema-pyramid`: titles for the minimum and maximum halves with the block size. The picture already showed a visible difference today; I added labels only.
- `computation-record`: labels (graph, lifetimes, counts), named bars with the real counts above them, "record valid" label, a rule through the lifetime rows at the hovered node, readout now gives node number, label and op instead of "source none".
- `computation-metrics`: titles, ms axis on the pass chart, pass numbers, end value of the timeline; hovering a node also outlines its pass slot and draws a rule at the pass total.
- `snapshot-tap`: titles (the tap, the iteration coordinate row, "copied: state / residual / deviation").
- Labels (WEB-INT-16): `calendar-buckets` (hours 00 06 12 18, weekday names that rotate with `firstDayOfWeek`) and `change-detection` (significance legend with the three classes, title for the other statistics, label for the NDVI strip).

Files: `entries/dataframe-pointcloud-records/{pointcloud-voxel-merge,pointcloud-icp,pointcloud-statistical-filter,pointcloud-neighborhoods,pointcloud-height,computation-record,computation-metrics,snapshot-tap,calendar-buckets}.ts`, `entries/raster-bands-indices/{raster-registration,spectral-unmixing,spectral-unmixing.parameters,glcm-texture,extrema-pyramid,change-detection}.ts`. No catalog text was changed (no entry text describes the old pictures).

## 2. Verified

- `node scripts/verify-runs.mjs <14 ids> --themes dark,light --out .reports/w3-web-pointcloud-raster`: 28 of 28 pass. Warnings: `raster-registration` "upsampleFactor does not change the picture" (the recovered dot moves by at most 0.5/upsampleFactor px, below a pixel of the picture; it existed in a milder form before); `computation-metrics`, `computation-record`, `snapshot-tap` "picture differs after parameters return to defaults" (graph rebuild timings, also present before); `computation-metrics` "2 distinct pictures at 5 positions".
- Read both themes of: raster-registration, pointcloud-voxel-merge, spectral-unmixing (after a light-theme fix), pointcloud-icp, pointcloud-statistical-filter, pointcloud-neighborhoods, pointcloud-height, glcm-texture, extrema-pyramid, computation-record, computation-metrics, snapshot-tap (dark), calendar-buckets (light), change-detection (dark). Pointer sheets read for raster-registration and voxel-merge only. Not read: snapshot-tap light, change-detection light, calendar-buckets dark, hover sheets of the other runs.
- `npx tsc -p website/tsconfig.json --noEmit`: 0 errors. `node --experimental-strip-types scripts/check-catalog.ts`: 532/532, 0 errors, 2 warnings (space-time-events, not mine). `npx biome check --write` on the 15 files: clean.
- Not run: `snapshot-runs.mjs`, phone width (111 px), a hover-through of every gesture at card size.

## 3. Not done

- `pointcloud-smrf`, `pointcloud-ground`, `color-scale`, `class-breaks`: left alone as instructed.
- `snapshot-tap`: the note's "tapped view large, diffed against the final result" is not done; it needs the final state captured in a second arena slot, a change to the graph build. Only labels were added.
- `pointcloud-neighborhoods`: no enlarged inset; the neighbourhood is 27 px across on the live card. Spokes and brighter points were used instead.
- `computation-metrics` still gives 2 distinct pictures at 5 pointer positions: node bars are outlined per pass and there are only two passes.
- `raster-registration`: the recovered shift is below one pixel, so true and recovered dots overlap at default (that is the operation working); `upsampleFactor` is visible only in the readout tolerance.

## 4. Work list for others

Posters to regenerate (rest picture changed, both themes): `raster-registration`, `pointcloud-voxel-merge`, `pointcloud-icp`, `spectral-unmixing`, `pointcloud-statistical-filter`, `pointcloud-neighborhoods`, `pointcloud-height`, `glcm-texture`, `extrema-pyramid`, `computation-record`, `computation-metrics`, `snapshot-tap`, `calendar-buckets`, `change-detection`.

Renames in the library, runs that use the affected classes (all through the older built `dist`, names unchanged):
- `GPURasterRegistration`: `raster-registration.ts` (props `grid`, `reference`, `moving`, `upsampleFactor`, `output: {shift, peak, status}`). It reads `outputs.shift` via `afterRun` and `peek`.
- `GPURasterSpectralUnmixing`: `spectral-unmixing.ts` (props `cellCount`, `bandCount`, `endmemberCount`, `values`, `endmembers`, `constraint`, `output: {abundances, residual, status, iterationCounts}`).
- `GPURasterTexture`: `glcm-texture.ts` (props `grid`, `values`, `levels`, `range`, `radius`, `offset`, `symmetric`, `features`, `pairCount`).
- Other remote-sensing/raster classes in my folders: `GPURasterExtremaPyramid` (`extrema-pyramid.ts`), `GPURasterChangeDetection`-family in `change-detection.ts` (unchanged by me), `GPURasterSampling` (spectral-unmixing, glcm-texture), `GPURasterArithmetic` (raster-registration, through `support/indices.ts`).

Framework requests: none. Two things would have saved work: a raster layer with a placement rectangle (the registration and glcm panels use one wide raster plus `clip` because a raster always fills the world), and a text layer panel whose size is in world units (backing panels for insets are only possible in text pixels).

## 5. Roadmap

- WEB-PIC-12: 12 of the 40 worst pictures in my groups (raster-registration, pointcloud-voxel-merge, pointcloud-icp, spectral-unmixing, pointcloud-statistical-filter, pointcloud-neighborhoods, pointcloud-height, glcm-texture, extrema-pyramid, computation-record, computation-metrics, snapshot-tap) are reworked; they need a re-score from new posters. Remaining in these runs: the tapped-view-versus-final diff in `snapshot-tap`, an enlarged neighbourhood inset in `pointcloud-neighborhoods`, a pass-level hover in `computation-metrics` that gives more than 2 states.
- WEB-INT-16: add `calendar-buckets` and `change-detection` to the labelled list. Of the ranked runs in my two groups none remain unlabelled; other computation and point-cloud runs (`computation-captures`, `computation-recorder`, `snapshot-arena`, `pointcloud-surfaces`, `pointcloud-change`, `pointcloud-features`, ...) were not on the ranked list and carry no text.

## 6. Noticed

- `pointcloud-statistical-filter`: kept dots drawn in the background colour (the trick for large rejected discs) can clip a neighbouring red disc slightly in the map (small bites visible at 960 px); the red dots are redrawn small on top so they never disappear.
- The `dots` layer cannot draw an inverted mask or per-dot alpha with a categorical ramp; large rejected points therefore need two layers. A `radius` per category would remove that.
- `raster-registration` caption in `verify-runs` reports the upsampleFactor slider as inert; the recovered shift differs only in the third decimal.
- `pointcloud-height` shows heights up to 66 m in the lidar sample (the cloud `scale` is metres per unit); a tall outlier sets the ramp top, so most buildings use the lower half of the ramp.
