# Fiji spherical boundaries implementation

## Delivered scene

`examples/deck/spatial-analysis-showcase/src/scenes/earth/spherical-boundaries.scene.ts` presents
**Fiji on a sphere**. It uses a light paper sheet and a close Fiji view so the source and computed
lines remain legible. Natural Earth 1:110m stores Fiji as three antimeridian-split shells (8, 9 and
5 listed positions including closure). All three source shells are visible; only the middle Viti
Levu shell is supplied to analysis. The Vanua Levu pieces are contextual outlines and are not
joined, reconstructed or analyzed. The synthetic analytic window and deterministic query grid are
mathematical probes, not observations, official boundaries or coastal hazard information.

The five story chapters cover the source shell, spherical containment, intersection, the four
Boolean operations and adaptive S2 support. Their prose, text alternatives and readouts describe
the geometry and its source limitations. The snippet shows only the method selected in the current
view. Optional convex intersection/contains and intrinsic arc overlay are not claimed as delivered.

## GPU operations and controls

- `GPUSphericalPolygonContains` evaluates the deterministic 23 × 19 grid plus eight exact source
  vertices under closed/open and selected-shell/complement rules. Query tolerance is `1e-6` in unit
  vector space; it is not polygon positional accuracy.
- `GPUSphericalPolygonOverlay` computes intersection, union, difference and symmetric difference
  between Viti Levu and the synthetic window in a gnomonic chart. Chart center is calculated once
  from 179° E, 17° S and shared by containment, overlay and cover. Great-circle edges are densified
  for display and longitudes are unwrapped around Fiji. `status.count` is output polygon count;
  `status.requiredCount` is the unclamped required ring count (shells plus holes), not polygon
  count. Candidate overflow, output/boundary overflow, uncertain pairs or the complement bit make
  the result incomplete for this renderer: success count is cleared and output geometry is
  withheld while written/required counts and causes remain in the status text.
- `GPUS2PolygonCovering` supports exterior/interior mode and user-selected minimum levels 3 or 5
  and maximum levels 6, 8 or 10. Every offered pair satisfies minimum ≤ maximum. A level change
  lazily builds a graph variant through `GraphSwap`; compilation never occurs in `encode`, stale
  cover geometry/readouts are hidden while it builds, and superseded builds are discarded. Cover
  status retains written, required-so-far and visited counts. Invalid input, capacity exhaustion,
  traversal exhaustion or measure errors withhold cover lines and area. Area is summed only after
  all measure status words pass, then converted from steradians using the IUGG mean sphere radius
  6,371,008.8 m.
- One selected-result publisher handles every relevant option update and matching asynchronous
  readback. It clears stale readouts before publishing a cached result or pending message. Summary
  readers are stopped before graph resources are destroyed; failed graph compilation retains
  resource cleanup.

## Verification and remaining scope

- Scoped `node scripts/check-cartography.mjs earth`: passed.
- Scoped `node scripts/check-story-steps.mjs earth`: passed.
- Showcase package typecheck: this scene reported no diagnostics. The project check remains blocked
  by unrelated concurrent edits in `geometry/repair-overlay.compute.ts`,
  `regression/county-panel.compute.ts/.scene.ts`, and
  `statistics/county-regionalization.compute.ts/.scene.ts`; exact diagnostics are in the review
  handoff.
- Browser GPU execution, readback values, scene screenshots, and independent comparison against a
  CPU spherical geometry oracle remain pending the planner's browser slot. The expected original
  Viti Levu grid checks are 437 grid probes plus 8 vertex probes; target classifications are
  inside 283, boundary 8, outside 154, with closed/open totals 291/283 and complement totals
  162/154. These are targets for GPU verification, not claimed browser results.
- Lazy level variants rebuild the complete scene graph, including the containment, overlay and
  both cover-mode branches. A narrower cover-only graph would reduce rebuild work. There is no
  source-shell selector: the story accurately keeps one selected analysis shell and shows all
  source parts as context. Optional convex and intrinsic arc operations remain outside this scene.

## References

- Natural Earth data and terms: <https://www.naturalearthdata.com/downloads/> and
  <https://www.naturalearthdata.com/about/terms-of-use/>
- S2 cell basics and hierarchy: <https://s2geometry.io/devguide/basic_types.html> and
  <https://s2geometry.io/devguide/s2cell_hierarchy.html>
- S2 RegionCoverer reference: <https://github.com/google/s2geometry/blob/master/src/s2/s2region_coverer.h>
