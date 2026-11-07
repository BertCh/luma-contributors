> **Note (2026-11-07).** Copied from the luma.gl `map-contributors` branch (`dev-docs/research/showcase-overhaul-2026-10-25/`). Paths refer to that branch. Large JSON snapshots, source-audit tarballs and Python/Node reference scripts were not copied; see [MANIFEST.md](./MANIFEST.md).

# Geospatial showcase overhaul

## Goal and scope

Turn the contributor showcase into a coherent atlas of scientific investigations, with distinctive
maps, strong source data, explicit assumptions, and inspectable computed evidence. New stories,
consolidation, retirement of redundant stories, and changes to the presentation are all authorized.
The existing 112 stories and 66 dataset definitions are the starting point, not a target format.

The domain documents below distinguish researched plans, implemented code, numerical references,
and browser-verified results. The finite domain queues are now complete within their stated limits;
the full wave-21 sweep completed with one mobile failure, now resolved by focused wave-22 checks;
the final full repository test command passes. Source presence alone
does not certify a numerical result or every available option.

## Parallel work

Six Astra planners directed and independently reviewed Luna implementers. The initial six
in-app workers were supplemented by bounded CLI lanes on disjoint file scopes. Existing shared
library edits were preserved. The parent coordinates integration, actual map captures, dataset
integrity, repository checks and serialized GPU browser verification.

| Track | Research and implementation record | Scope |
| --- | --- | --- |
| Atlas experience | [Experience](experience.md) | Question-led routes, searchable inventory, responsive and accessible reading |
| Statistical evidence | [Statistics](statistics.md) | Weights, inference, regression, interpolation, validation and uncertainty |
| Movement and networks | [Movement and networks](movement-networks.md) | Disruptions, access, routes, trajectories and flows |
| Earth systems | [Earth systems](earth-systems.md) | Raster evidence, time series, terrain, hydrology and spatial support |
| Geometry and observation | [Geometry and observation](geometry-observation.md) | Spherical cells, geometry, joins, patterns and observation support |
| Data and coverage | [Data and coverage](data-coverage.md) | Query workflows, public API/option inventory and evidence classification |

The experience team also owns point-cloud measurements. Domain ownership kept shared edits
coordinated. The wave-22 source freeze is audited and focused visual corrections are accepted;
the final full repository test command also passes.

## Implemented atlas structure

The current source contains 174 investigations, organized around seven short, question-led routes:
transit disruption, post-fire change, point-pattern geography, drainage, model transfer, survey
change, and query semantics. The complete inventory remains searchable, with direct story links
and a separate searchable source catalog. Each route connects three investigations and provides
an onward link at the end of a story.

The presentation separates observations, computed evidence, and interpretation limits. Maps retain
their subject-specific visual encodings while sharing responsive navigation, legends, attribution,
scale bars, and keyboard access. Phone layouts can move the legend into the reading panel when
the map has insufficient space. These implementation facts do not certify every numerical result;
the acceptance records below and in the domain ledgers remain authoritative.

## Scientific and visual acceptance

- Sources, licenses, dates, units, sampling, transformations and checksums must be reproducible.
  Source hashes and derived artifact hashes identify different objects.
- A story starts with a question, shows an observation, and lets a reader change an assumption.
  Each step changes visible evidence or meaningfully interrogates that evidence.
- Distinct aesthetics come from the subject and its visual encoding: timetable networks, satellite
  imagery, topographic relief, spherical support, observation traces and uncertainty surfaces.
- Color scales must match the data type and preserve interpretation. Sequential, diverging and
  categorical uses have different semantics; legends and rendered values must agree. This follows
  the evidence and recommendations in [Crameri et al. (2020)](https://www.nature.com/articles/s41467-020-19160-7).
- Keyboard focus, usable narrow layouts, textual alternatives and labels supplement maps and
  color, following applicable [WCAG 2.2 criteria](https://www.w3.org/TR/WCAG22/). Passing targeted
  checks does not constitute a full accessibility conformance audit.
- Missing, invalid, unsupported, unconverged and capacity-limited results cannot masquerade as zero.
  Hypothetical scenarios and validation experiments must be labeled as such.
- GPU outputs must drive the evidence. Bounded asynchronous readbacks need revision guards;
  controls must truthfully distinguish parameter changes and graph rebuilding.
- Independent numerical checks should verify consequential claims, alongside real browser/GPU
  interaction. A rendered background or successful TypeScript build alone is insufficient.

## Coverage evidence

`coverage-baseline.json` records the original class/recipe audit. The expanded inventory in
`functionality-baseline.json` and `functionality-gaps.json` includes functions, methods and option
variants, so its denominator is different. Its first expanded snapshot was taken during initial
implementation and is labeled accordingly. Do not compare those totals as a simple progress rate.

The audit distinguishes utility functions, graph protocol and lifecycle methods, public operations,
literal variants, dynamic expressions, module reachability and call sites. Static evidence cannot
prove GPU execution, numerical validity, or complete option coverage. Recipe composition and fluent
query factories need inspection rather than being counted automatically as missing functionality.

Regenerate the detailed report from the repository root:

```sh
node scripts/report-showcase-coverage.mjs --format=json --functionality
node --test scripts/showcase-functionality-inventory.test.mjs
node examples/deck/spatial-analysis-showcase/scripts/check-datasets.mjs --json
```

## Current acceptance and provenance

The [acceptance ledger](acceptance-ledger.md) is the current index. It separates source evidence,
bounded numerical checks, visual publication checks and the final integration gates. Its linked JSON
and domain packets retain commands, independent references, tolerances, report hashes, limitations
and failed attempts. Earlier accepted checks keep their actual build fingerprints; repeated runs are
not added into a coverage percentage.

All finite domain queues are closed within their documented scopes. Notable qualifications remain:

- Circuit verification accepts four mild numerical cases, four strict publication-withholding
  cases and one mild recovery. Strong cases remain uncertified predictions. The separate 16 × 16
  comparison does not certify the full 64 × 64 solve; rejected solver experiments are preserved.
- Population acceptance verifies observed → modeled → observed chart recovery. The 14-case final
  run supersedes its earlier numerical-only run rather than adding 14 new independent models.
- CTA affine checks cover translation and inverse only. Both tested subdivision modes conserve
  the same retained geometry; block area checks do not prove every owner or Voronoi vertex.
- Sampling verification reads all 2,048 render roles: 2,040 unused slots stay hidden and eight
  occupied positions remain source-contained. Its focused coordinate oracle does not certify
  every sampling option or distribution.
- Six of the 68 display/profile/transport records repeat an unused nearest-neighbor policy.
  Quantile-LISA membership is descriptive; its legacy pseudo-p values are not certified inference.

The final wave-22 [source inventory](functionality-source-final.md) has matching before/after
manifests and 91 checked manual call-path anchors. It records 174 stories and explicitly distinguishes
static calls, manual dispatch, composition, equivalent outputs, support helpers and infrastructure.
The [option-scope review](functionality-option-scope.md) preserves tested and untested semantics.
Zero unassigned top-level decisions is not complete runtime or option-union acceptance.

## Integration gates

| Gate | Current evidence |
| --- | --- |
| Node selection and dependencies | `nvm use` selected Node 22.23.4; `yarn install` completed with existing peer-dependency warnings. |
| Formatting and package build | Wave-22 formatting and package build passed after the test-only fixture correction and final display changes. |
| Showcase static and production checks | Wave 22 passes TypeScript, dataset/story/cartography/color/shader checks and production bundling. |
| Website builds | Both `yarn website:build` and `(cd website && yarn build)` passed on wave 20: 1,206 generated documents and 1,210 raw documentation pages each. Later runtime corrections are not relabeled as website runs. |
| Full repository tests | **Passed**, exit 0: Node 1,246 files / 12,593 tests; all eight headless shards, 1,107 files / 12,620 tests. Skipped: Node two files / 439 tests; headless eight files / 846 tests. |
| Complete atlas/mobile sweep | Wave 21 completed all 174 stories and 994 steps: 173 passed, one mobile legend interaction failed. The bundle remained unchanged. Focused wave-22 checks resolve the failure; no full sweep on the later bundle is claimed. |

The final wave-22 emitted-code fingerprint is
`7fc7fb3a993176cdbba18b723eb2fbc4719b53090829c7a1b48b9406db57278e`.
The [final integration packet](integration-final.json) preserves its manifest and build-log hashes.
All 1,418 manifest entries were checked against emitted bytes. Each browser suite saves
`build-evidence.json` and rejects a bundle change during its run. Public dataset files retain their
own checksums; the code fingerprint does not certify dataset contents.

The successful final command is `VITEST_MAX_WORKERS=2 yarn test`. Its complete log and per-shard
counts are preserved in [integration-final.json](integration-final.json). Post-fixture `yarn lint fix`
and `yarn build` pass; `git diff --check` passes. All 1,418 emitted entries and the served manifest
were rechecked after tests, with the runtime bundle unchanged.

Earlier test failures remain documented. The [registry diagnosis](lens-registry-node-order-diagnosis.md)
fixes a synthetic test ID that collided with a real scenario while preserving production duplicate
rejection. The [RS02 diagnosis](rs02-wave22-acceptance.json) replaces a brittle exact compact tuple
count with a dense 40-cell comparison at unchanged geometry bounds, including expected zeros and
strict ordered unique IDs. Neither correction changes runtime algorithms or tolerances. The focused
25-test RS02 run is supplementary; the final complete command passes afterward.

Two nonfatal warnings remain in the successful log: Vite cannot analyze a variable dynamic import
in the standalone Gaussian-splats loader, and headless shard six reports browser/Vite teardown
exceeding ten seconds. Subsequent shards complete and the overall command exits zero.

Dataset verification covers 715 assets across 70 bundled datasets, with two remote-loader datasets
explicitly classified. Provenance omissions remain notes rather than certified source quality.
The color and cartography audits retain 42 and six advisory findings respectively; the
[default-color audit](colour-advisory-default-audit.md) distinguishes unused registry entries from
displayed encodings. Targeted accessibility checks do not establish complete WCAG conformance.

## Visual and shell evidence

Wave-21 navigation checks pass inventory/search, cancellation, keyboard access, source catalog,
continuation and narrow-layout assertions with no browser errors or build drift. These are shell
checks, separate from rendering every step of all 174 investigations. The landing page requests no
GPU adapter, compute module or recorded-figure archive; its 248 script requests total 3,395,859
decoded bytes, distinct from the smaller entry chunk.

The [focused visual packet](visual-corrections-wave21.json) covers seven unique corrected stories
and 45 steps across waves 20 and 21. The [street-density publication check](street-density-publication-acceptance.json)
separately verifies first-entry and recovered map publication. The [complete wave-21 sweep](atlas-wave21-sweep.json) now records 173 passing stories and one mobile
legend failure across 174 stories and 994 steps. Independent review additionally found a support-mask
background defect. Both corrections pass focused verification on wave 22.
Earlier interrupted and failed atlas runs remain historical evidence.

Five featured previews were captured from actual wave-16 WebGPU renders and individually reviewed:
transit frequency, Dixie Fire severity, Chicago H3 observations, Grand Canyon drainage and Helene
rainfall interpolation. Their actual capture build and dimensions are in the showcase's
`public/images/story-previews/captures.json`. Preview refreshes do not certify the final atlas sweep.

Responsive checks cover compact legends, camera behavior, timelines at 320 and 390 pixels, and
legend/title/source-credit rectangles. Bounds refitting preserves a reader's moved camera. Detailed
shell and presentation evidence remains in [experience.md](experience.md).

The [wave-21 sweep packet](atlas-wave21-sweep.json) preserves the complete mixed-result report and
emitted manifest. The [wave-22 focused packet](visual-corrections-wave22.json) accepts seven shared-layout stories,
45 desktop steps and seven mobile layouts, with the incorrect `population-support` invocation
retained as a runner failure and the correct `chicago-population-support` retry accepted. The
[earth packet](earth-wave22-acceptance.json) separately repeats all 55 missing-observation numerical
checks and passes six desktop steps, one mobile layout and exact finite-mask/display-area checks.
The [final integration packet](integration-final.json) records the successful full node/headless
command, skips, warnings and log hashes. The finite verification queue is closed with the scientific
limits above. The earlier sweep is not relabeled as a later-build run and its failure is retained;
domain evidence keeps its original provenance.
