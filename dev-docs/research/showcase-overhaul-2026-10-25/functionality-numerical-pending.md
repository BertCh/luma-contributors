# Closed integration queue

No finite numerical, visual correction or root integration gate remains pending. The
[final integration packet](integration-final.json) records exit-zero `VITEST_MAX_WORKERS=2 yarn test`:
Node 1,246 files / 12,593 tests and all eight headless shards, 1,107 files / 12,620 tests. Skips,
nonfatal warnings and earlier failed attempts remain explicit. Source/build and focused checks are
separate evidence; all required gates have now completed.

The [complete wave-21 sweep](atlas-wave21-sweep.json) remains 173 passing stories and one mobile
legend failure across 174 stories and 994 steps on its unchanged bundle. Separately fingerprinted
[wave-22 focused checks](visual-corrections-wave22.json) pass seven shared-layout stories, 45 desktop
steps and seven mobile layouts. The wrong `population-support` invocation remains recorded as a
runner error; the correctly named population retry passes. Source-layout fixtures pass four map-chart
and four legend cases.

The [corrected missing-mask packet](earth-wave22-acceptance.json) repeats all 55 numerical checks
and passes exact finite-mask counts, a bounded pixel-area check, six steps and one mobile layout.
These accepted visual corrections do not erase the original failures or imply a second full-atlas
sweep on wave 22. The emitted manifest and source snapshot are pinned separately.

## Current bounded domain results

- Data: county 17 and fire 22 accepted on wave 14; OSM one three-state native draw regression accepted on wave 15.
- Statistics: wave-14 bounded checks retain their original evidence; population 14 on wave 15 closes its display failure and supersedes the earlier 11-case numerical-only run.
- Atlas: CTA 15 editing/affine and two windowing cases accepted on wave 15. Translation/inverse only; both subdivision paths conserve the same retained geometry.
- Movement: all 68 finite-queue cases accepted across waves 13, 15 and 16. BIXI selected-site visuals and wave-20 street-density first-entry publication are separate visual/lifecycle checks.
- Geometry: coverage 8, inhomogeneous patterns 11 and core topology 29 accepted on wave 15; county 10, footprint 27 and convex refinement 8 on wave 16; block 48 on wave 17. Sampling 28 was repeated on wave 21, with two additional render-role/source-containment checks. Block area acceptance is bounded and does not prove every Voronoi owner or vertex.
- Earth: tiled 58, topology 2, terrain 4, pixel 33, texture 46, perimeter 6, preparation 48 and regions 20 accepted on wave 16. Missing observations 55 on wave 17 supersedes 49 and is repeated on wave 22; display 68 on wave 18 includes six repeated nearest-policy records.
- Circuit: wave 19 closes with four independent mild numerical acceptances, four strict publication-withholding checks and one mild recovery. Strong cases remain uncertified numerical predictions; the separate 16 × 16 comparison cannot certify the full 64 × 64 solve.

Earlier partial counts are superseded here, not erased from their original packets. Repeated runs,
visual checks and numerical records must not be added into a single coverage total.
