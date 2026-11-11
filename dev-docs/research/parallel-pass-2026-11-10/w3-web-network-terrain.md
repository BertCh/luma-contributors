# w3-web-network-terrain

## 1. Done (all under website/src/gpu/runs/entries/)
- `network-routing-access/shared/display.ts`: new GPU display kernels `addMatrixFootprint` (row or column of a cost matrix to a per-node value), `addZoneSegments` (roads above a threshold as a lines list), `addConsecutiveSegments` (track legs, optionally only where a flag matches). New `shared/band-text.ts` (band outlines and names for the adjacency matrices).
- `track-quality`: before, spaghetti of 7 tracks, "flagged 70" at every position, many real AIS legs flagged. After, 6 tracks, muted clean fixes, bold legs that arrive at a speed-flagged fix (from the recipe's flags, joined on the GPU), a legend; defaults maximumSpeed 0.5 to 0.8 (range 0.2-3), glitchRate 0.08 to 0.05. Flagged readout now 12-ish and the picture shows which legs.
- `network-kernel-density`: before, dim purple edges and a Euclidean ring that is not the network bandwidth; 1.4% change. After, brighter ramp, the dragged event's kernel footprint lit as a halo along the roads (GPU, from the operation's distance column), ring removed; 7.2% change.
- `catchment-accessibility`: before, dots and an arc that is not the catchment; 1.7% change. After, roads within costLimit of the dragged facility lit (its row of the cost matrix); 15.9% change.
- `network-snapping`: 150 to 70 points, bold ink snap links, bigger markers, slider range 0.004-0.05 to 0.004-0.1 (default 0.06); links visible at card size. Gesture change is small (1.5%): most points are within a few mm of a road.
- `adjacency-matrix`, `adjacency-matrix-order`: band blocks outlined and named, "rows: home tract, columns: work tract" note (also the label item for `adjacency-matrix-order`). The inset network of the proposed fix was NOT done.
- `map-matching`: 6 to 3 tracks; raw track and fixes blue, matched route (matched positions joined in order, GPU) orange and bold, fix-to-road links, legend. Gesture 1.8%.
- `cell-geometry`: default resolution 4 to 3, 120 points, data cells faded, pointer cell decoded large (fill, corner dots, center dot from GPUCellGeometry output). Corner dots are partly hidden by the existing white glyph dots.
- `terrain-dem-coregistration`: glacier moved to the curtain centre (x 0.5) so raw (dark disc) and corrected (blue disc) show at rest; titles, ramp legend, "glacier (unstable)", "fit"/"bias" bar names; readout now gives the stable-ground median of raw and corrected (read from the buffers): "raw 0.049 | corrected 0".
- `cell-grid-path`: readout and range, see Noticed. maximumPathLength range 3-14 to 3-24, default 14 to 20; readout "longest path N cells, all drawn whole" or "k of n drawn cut off".
- `terrain-critical-points`: broken log-count bars replaced by a legend with GPU-read counts (afterRun readback): peak 35, pit 53, saddle 252.
- Labels added: `weiss-landforms` (10-class legend), `viewshed` and `point-horizon-visibility` (visible / hidden legend), `point-horizon-profile` (N E S W N, 0 degree horizon, title), `network-k-function` (legend), `adjacency-matrix-order`, `terrain-critical-points`.

## 2. Verified
- `node scripts/verify-runs.mjs <16 ids> --themes dark,light --out .reports/w3-web-network-terrain`: 32 / 32 pass, 1 warning (track-quality slider maximumGap does not change the picture; the run's own gap flags are class colours only). One transient "canvas is blank" on cell-geometry light under load, passed on rerun.
- Read dark and light screenshots and pointer sheets for track-quality, network-kernel-density, catchment-accessibility, network-snapping, adjacency-matrix(-order), map-matching, cell-geometry, terrain-dem-coregistration, cell-grid-path, terrain-critical-points (the light shots of snapping, catchment, cell-geometry, grid-path and dark of order not all read).
- `npx tsc -p website/tsconfig.json --noEmit`: 0 errors. `node --experimental-strip-types scripts/check-catalog.ts`: 0 errors, 2 warnings (not mine). `npx biome check --write` on the folders: clean. Not run: snapshot-runs, phone width.

## 3. Not done
- adjacency-matrix inset network with hovered rows lit (needs a tract map; only text added).
- No catalog text edited.

## 4. Work list for others
- Regenerate posters: track-quality, network-kernel-density, catchment-accessibility, network-snapping, adjacency-matrix, adjacency-matrix-order, map-matching, cell-geometry, cell-grid-path, terrain-dem-coregistration, terrain-critical-points, weiss-landforms, viewshed, point-horizon-visibility, point-horizon-profile, network-k-function.
- Framework request: the `segments` layer draws a value faded to nothing as the opaque solid color (kit.ts SEGMENT_SHADER, `color.a == 0`), so `fade` cannot light part of a network; I worked round it with `addZoneSegments`.

## 5. Roadmap
- WEB-PIC-12: 10 of the 40 worst addressed here (items above); not re-scored. Remaining: adjacency-matrix inset, snapping gesture is weak, track-quality still crowded at the centre.
- WEB-INT-16: 7 more runs labelled; this stream's remaining unlabelled ranked runs: none known.

## 6. Noticed
- `cell-grid-path`: not an operation defect. `GPUCellGridPath` documents `counts[row]` as the true length (overflow when above maximumPathLength, cells cut to the first maximumPathLength). The old readout printed the true length next to the cap. Fixed the readout.
- `terrain-dem-coregistration`: not a defect. The blue disc appeared only when the curtain left the glacier on the corrected side; the readout never changed because the curtain does not change the fit. Fitted 0.049 vs bias 0.050 is the stable mask leaving a little glacier tail inside it.
- Previous quirk: `track-quality` readout "flagged" uses `peek` from before the rerun, so it lags a drag by one run.
