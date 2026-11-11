# w3-web-geometry-construction

## 1. Done
All files under `website/src/gpu/runs/entries/geometry-construction-overlay/`. New helper files: `shared/badge.ts` (text badge without swatches), `shared/ear-clip.ts` (CPU ear clipping, only to draw input cells), `getLogarithmicReach` and `getLogarithmicReachDistance` appended to `shared/reach.ts`.
- `parameter-buffer` (+ `.parameters.ts`: min 0.002, default 0.04): reach now logarithmic from the shore (0.002 at the shore, 0.1 at 0.3 units), so the pointer positions give 0.002 to 0.021 instead of 0.100 everywhere; rest tolerance 0.04 (visibly simplified shore); a badge reads `tolerance`, `graph built N×`, `parameter buffer written N×` (real counts from `build` and `update`). Before: shore with saturated tolerance, no sign of the per-frame write. After: counter rises (7× at pointer 1) while build stays 1×.
- `group-concave-hull`: gesture was already the reach on maximumCircumradius (hulls re-form strongly in the pointer sheets), so only a legend was added (title with the limit, one swatch per group hull).
- `line-segmentize`: original vertices are large muted discs, every written vertex an orange dot on top (lone orange dot = inserted), blue lines thicker, legend. Before: all dots blue and alike.
- `outline-geometry`: input drawn in ink over the outline (points r6), log reach (0.005 to 0.1 over 0.3 units) so the distance varies (was 0.100 at 2 of 3 positions), legend. Rest picture: discs with visible input points.
- `linear-referencing`: roads in ink, bigger query points, thicker links, legend (road, query point, link, foot point colored by measure).
- `coverage-validity`: cells filled (ear-clipped on the undistorted cells, shrink is a uniform scale so it stays valid), so real gaps show as background slivers and overlaps as darker; flagged edges stay orange; legend. No counts quoted, so the GEOS gap-rule change only changes which edges turn orange.
- `segment-ring-assembly`: loose input segments drawn clearly in ink, assembled rings colored per ring, legend. No counts.
- `building-adjacency-recipe`: probe snaps to the nearest building (all 5 test positions now hit, was 2 of 5 "no building hovered"), legend (terrace color, shared wall). Rest picture unchanged apart from the legend (restDifference 0).
- `polygon-triangulation` and `constrained-delaunay-triangulation` (shared `triangulation.ts`): outer rings blue, hole rings orange (the constraint edges for CDT), legend including "triangle color: smallest angle". No counts, so hole bridging changes do not make the caption false.
Labels: `geometry-validity` and `canonical-geometry-equality` are in `geometry-processing`, not my folder; no ranked run is in my folder. Labels were added as legends on the nine runs above.

## 2. Verified
- `node scripts/verify-runs.mjs <10 ids> --themes dark,light --out .reports/w3-web-geometry-construction`: 20 of 20 pass. Looked at dark and light screenshots and pointer sheets for all ten runs (not every pointer sheet of every run).
- `node --experimental-strip-types scripts/check-catalog.ts`: 0 errors, 2 warnings (space-time entries, not mine). `npx tsc -p website/tsconfig.json --noEmit`: no errors in my folder. `npx biome check --write` on my files: clean.
- Not run: snapshot-runs, phone width, a live-site card check.

## 3. Not done
- `polygon-triangulation` real-data input: no checked-in polygon data with holes (lakes and Amsterdam footprints have none); blob input kept, the picture explains it by ring type instead.
- `building-adjacency-recipe` fit: the Amsterdam frontages are a wide strip in two rows far apart; real data, left as is.
- Other gesture 1-2 runs (`buffer-surface`, `line-simplification`, `concave-hull`, `group-convex-hull`, `coverage-dissolve`, `dorling-cartogram`, `grid-generator`, `line-smooth`, `street-profiles-recipe`): not touched; no time. For `buffer-surface` and `line-simplification` the distance is the parameter by design (corridor), so the log mapping used elsewhere would change their meaning.
- Catalog text unchanged (no statement became false).

## 4. Work list for others
Posters to regenerate (rest picture changed, both themes): parameter-buffer, group-concave-hull, line-segmentize, outline-geometry, linear-referencing, coverage-validity, segment-ring-assembly, building-adjacency-recipe, polygon-triangulation, constrained-delaunay-triangulation. Framework request: none (a `peek`-free counter in `draw` works because `update` runs before `draw` in `render`).

## 5. Roadmap
WEB-PIC-12: ten pictures of the construction group reworked; remaining there: buffer-surface, line-simplification, concave-hull, group-convex-hull, coverage-dissolve, dorling-cartogram, grid-generator, line-smooth, street-profiles-recipe, group-geometry, shape-generator, offset-expansion, then a re-score. WEB-INT-16: no ranked run lives in this folder; `geometry-validity` and `canonical-geometry-equality` (geometry-processing) remain.

## 6. Noticed
- `constrained-delaunay-triangulation` at maximumFlips 3 draws the same fan as ear clipping; the two siblings still look alike (WEB-PIC-15).
- Light theme: sequential-ramp foot points at measure 0 (linear-referencing) are near the background color.
