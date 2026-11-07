> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/showcase-sprint-2026-10-24.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Sprint finished; outcomes recorded in research/showcase-overhaul-2026-10-25.

# Spatial analysis showcase expansion sprint

## Goals

Turn the expanded analysis API into inspectable, place-based stories. Each addition must answer a
question, expose the assumptions that change the answer, and show actual computed outputs. A named
contributor in a description is not implementation evidence.

The starting showcase contained 107 scene definitions and 64 dataset definitions. Existing work in
the checkout, including airline-network changes and analysis-library expansion, is outside this
sprint's change inventory.

The sprint adds five stories and expands three existing ones: 112 scene definitions and 66 dataset
definitions are now present. Twelve Luna agents worked on research, implementation, review, and
browser validation, with integration and final verification in the parent task.

## Research and design

- [API and story coverage audit](./showcase-sprint-api-audit-2026-10-24.md): public contributor and
  recipe coverage, opportunities, and implementation boundaries.
- [Dataset research](./showcase-sprint-datasets-2026-10-24.md): existing data inventory and future
  source candidates, with acquisition, attribution, preprocessing, and quality requirements.
- [Story design](./showcase-sprint-design-2026-10-24.md): narrative progression, visualization
  choices, scientific interpretation, and acceptance criteria.

These research documents distinguish proposed work from delivered work. The implementation and
verification ledger below is the authority for this sprint's delivery status.

## Implementation ledger

| Story | Intended addition | Data | Status |
| --- | --- | --- | --- |
| Helene rainfall | Spatial block holdout, training-only variogram fit, prediction-support evidence | Existing GHCN stations | Implemented; browser passed |
| Facility placement | Hypothetical three-site network, worst-case distance versus capacity-constrained allocation | Existing Chicago tracts and facility context | Implemented; browser passed |
| Habitat connectivity | Land-cover support, patches, core habitat and explicit movement-cost assumptions | Existing Dixie WorldCover and elevation | Implemented; browser passed |
| Street enclosure | Building footprints, movable observer visibility, measured profile endpoints | New sourced Chicago footprint extract | Implemented; browser passed |
| Watersheds | Along-flow distance and extracted stream reach structure | Existing Grand Canyon DEM | Implemented; browser passed |
| Local observation outliers | Neighborhood-relative sparsity, sample support, threshold sensitivity | Existing Chicago nature observations | Implemented; browser passed |
| Election dynamics | Relative rank mobility and transition-chain diagnostics | Existing county election panel; explicit state aggregation where required | Implemented; browser passed |
| Lidar ground | Ground estimation, height above ground, neighborhood support and voxel reduction | New sourced point-cloud subset | Implemented; browser and voxel reference passed |

The Chicago extract contains 1,706 active building footprints. The USGS King County 2021 extract
contains 116,213 original returns, with float64 geographic positions preserving the half-metre
sampling resolution. Both additions have reproducible builders, provenance, bounds, units, and
checksums. Together their runtime payloads are about 4.8 MiB; the full source LAZ is not bundled.

The API reporter measures 235 of 487 public class/recipe entries with direct runtime call evidence.
This includes helpers and excludes code-panel strings. It is coverage of call sites, not a claim
that every method is demonstrated or that every scene has been numerically validated.

## Acceptance criteria

1. Dataset provenance, units, geographic support, and any sampling rule are visible and reproducible.
2. Each story transition changes a map, chart, or other visible computed result. Controls match their
   declared parameter-write or graph-rebuild behavior.
3. Map geometry and attributes consume GPU outputs directly where the contributors produce them;
   summaries use bounded asynchronous readback. Pending work cannot overwrite newer state.
4. Missing, invalid, unconverged, capacity-limited, and unassigned outputs are distinguished from
   valid zero values. Legends and charts describe the same units and support as the map.
5. Scenario assumptions are labeled. Relative ranks are not magnitudes, observational patterns
   are not causal effects, and algorithmic optimality is claimed only where the contract provides it.
6. Scene metadata remains lightweight and compute imports lazy. Every new scene is discovered by
   the existing registry and every new dataset loads through the existing catalog.
7. Static checks, TypeScript, production bundling, and actual browser/GPU interactions are checked;
   any failures or unverified areas are recorded without substituting one check for another.

## Verification ledger

- `nvm use`: Node 22.22.1 selected.
- `yarn install`: completed with dependency peer warnings.
- `yarn lint fix`: passed across 7,944 files; lockfile valid.
- `yarn build`: passed, including every package's TypeScript declarations and CommonJS identity checks.
- Showcase `typecheck`: passed after the module build.
- Showcase `check`: passed across all 112 stories, including previously skipped storm scenes after
  correcting the check harness's graph-subpath import stub. WGSL scan passed 31,324 templates.
  Non-failing cartography/color findings remain, including the expanded watershed's nine steps.
- `yarn website:build`: passed; existing documentation-link, TypeDoc, dependency, and bundle-size
  warnings remain. This command invokes the website workspace build; `(cd website && yarn build)`
  was not repeated separately.
- `yarn test-node`: 12,491 passed, 433 skipped, one failed. The sole failure is the copyright header
  of the pre-existing snapshot
  a local research log (not checked in).
  No ownership was invented for that copied upstream material. Two worker-termination warnings
  also occurred. The new election rank tests passed.
- Full `yarn test` / all eight headless shards were not run; the targeted browser sweep below is
  distinct from the full repository GPU suite.
- Production showcase builds and native Chromium WebGPU interaction checks passed for all eight
  sprint tracks (47 story steps), plus a pre-existing choropleth baseline. Checks cover each story step, actual GPU queue
  submissions, nonuniform map pixels, diagnostics, screenshots, and selected control changes.

Browser review caught and corrected errors that static inspection alone missed: a missing selected
facility instance count, a rejected WGSL NaN constant in watershed display, and a habitat summary
readback offset. Screenshot review also corrected the unassigned-demand legend and numeric k
control. Lidar initialization exposed duplicate graph resource IDs and an indirect-count view
layout mismatch, both corrected before the final retest.

An additional numerical check found that passing voxel population counts directly as compaction
flags summed points rather than occupied cells. The scene now creates explicit binary occupancy
flags on the GPU. The independently computed reference, using the same projection and float32
GPU coordinates, is 9,489 occupied 2 m voxels from 116,213 returns, with no out-of-grid points.
The browser harness requires this exact count and a strict reduction. Follow-up for the shared
`GPUCompaction` API: its documented nonzero-flag acceptance is inconsistent with scanning raw
nonbinary flags; the scene uses binary flags and does not change that shared implementation.
The final GPU result matched 9,489 exactly; using the same camera and point radius in both views,
11.56% of map pixels changed between the full sample and the reduced sample. This check also
caught a missing layer-refresh callback. Height ramp and legend now use the same cividis range.

Local evidence is in `/tmp/luma-showcase-sprint-20261024` (command logs),
`/tmp/spatial-analysis-showcase-seven`, `/tmp/spatial-analysis-showcase-habitat-lidar`, and
`/tmp/spatial-analysis-showcase-final-sprint` (browser reports and screenshots; its earlier lidar
failure is superseded by `/tmp/spatial-analysis-showcase-final-realdata/report.json`). The latter
report verifies the final lidar reduction and enlarged habitat view on the standard production URL,
without a synthetic-data query override. The reproducible
browser command is `node examples/deck/spatial-analysis-showcase/scripts/check-sprint-scenes.mjs`;
use `--help` for URL, story selection, and output options.

## Next-wave planning

Use the coverage report and dataset research to select subsequent work by the question it unlocks,
not only by the number of APIs covered. Priority candidates include spatially validated regression,
segregation counterfactuals, regular seasonal imagery, network disruption and accessibility,
fractional raster summaries, and environmental monitoring. No planned source or story is counted
as shipped until its artifacts and runtime evidence exist.
