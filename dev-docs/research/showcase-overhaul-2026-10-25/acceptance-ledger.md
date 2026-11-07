# Durable acceptance ledger

The finite numerical domain queues are complete within their documented scientific limits. Overall
acceptance is **complete within explicit scientific limits**. The [complete wave-21 sweep](atlas-wave21-sweep.json) covered 174 stories
and 994 steps: 173 stories passed and one mobile legend check failed, with an unchanged bundle.
Independent review found a second support-mask visual defect. Both corrections pass separately
fingerprinted wave-22 checks. The final full `yarn test` command passes with exit code zero.

[Source inventory](functionality-source-final.md), bounded numerical checks, visual checks and
whole-atlas integration are different evidence. Counts below are test records, not a coverage
percentage. Earlier accepted implementations keep their actual tested builds; they are not relabeled
as wave-21 runs. Full commands, references, tolerances, report hashes and failed-attempt history are
in [the machine-readable ledger](acceptance-ledger.json) and the linked packets.

## Current domain suites

“Pass” means only the linked suite's bounded contract. The circuit row includes successful
withholding of uncertified predictions; it does not claim eight converged numerical solutions.

### Atlas

| Suite and evidence packet | Result | Records | Build |
| --- | --- | --- | --- |
| [CTA editing including affine translation](cta-acceptance-wave15/acceptance.json) | Pass within scope | 15 / 15 | [15](#build-fingerprints) |
| [CTA route windowing](cta-acceptance-wave15/acceptance.json) | Pass within scope | 2 / 2 | [15](#build-fingerprints) |

### Data

| Suite and evidence packet | Result | Records | Build |
| --- | --- | --- | --- |
| [county-query-semantics](data-wave14-acceptance.json) | Pass within scope | 17 / 17 | [14](#build-fingerprints) |
| [fire selection and basin merging](data-wave14-acceptance.json) | Pass within scope | 22 / 22 | [14](#build-fingerprints) |
| [OSM probe layer membership](data-osm-wave15-acceptance.json) | Pass within scope | 1 / 1 | [15](#build-fingerprints) |

### Statistics

| Suite and evidence packet | Result | Records | Build |
| --- | --- | --- | --- |
| [Meuse neighborhood statistics](wave14-statistics-acceptance.json) | Pass within scope | 12 / 12 | [14](#build-fingerprints) |
| [Massachusetts weights](wave14-statistics-acceptance.json) | Pass within scope | 9 / 9 | [14](#build-fingerprints) |
| [Meuse regimes](wave14-statistics-acceptance.json) | Pass within scope | 9 / 9 | [14](#build-fingerprints) |
| [Rate recipes](wave14-statistics-acceptance.json) | Pass within scope | 5 / 5 | [14](#build-fingerprints) |
| [Meuse descriptive quantile LISA](wave14-statistics-acceptance.json) | Pass within scope | 19 / 19 | [14](#build-fingerprints) |
| [Statistics display contracts](wave14-statistics-acceptance.json) | Pass within scope | 7 / 7 | [14](#build-fingerprints) |
| [Nature kernel density](wave14-statistics-acceptance.json) | Pass within scope | 24 / 24 | [14](#build-fingerprints) |
| [County classification](wave14-statistics-acceptance.json) | Pass within scope | 5 / 5 | [14](#build-fingerprints) |
| [Pointwise CSR envelope](wave14-statistics-acceptance.json) | Pass within scope | 6 / 6 | [14](#build-fingerprints) |
| [Rainfall fits and conditional LOO](wave14-statistics-acceptance.json) | Pass within scope | 18 / 18 | [14](#build-fingerprints) |
| [Point-pattern global evidence](wave14-statistics-acceptance.json) | Pass within scope | 7 / 7 | [14](#build-fingerprints) |
| [Chicago population support](wave15-population-acceptance.json) | Pass within scope | 14 / 14 | [15](#build-fingerprints) |
| [County IV covariance](wave14-statistics-acceptance.json) | Pass within scope | 5 / 5 | [14](#build-fingerprints) |
| [Hotspot projection initialization smoke](wave14-statistics-acceptance.json) | Pass within scope | 1 / 1 | [14](#build-fingerprints) |

### Movement and networks

| Suite and evidence packet | Result | Records | Build |
| --- | --- | --- | --- |
| [BIXI demand and service locations](movement-acceptance-final.json) | Pass within scope | 15 / 15 | [15](#build-fingerprints) |
| [Flight matrix](flight-wave21-settled-acceptance.json) | Numerical pass; title fix separately verified on 22 | 12 / 12 (10 repeated) | [21](#build-fingerprints) |
| [Taxi rose statistics](movement-acceptance-final.json) | Pass within scope | 8 / 8 | [15](#build-fingerprints) |
| [Aviation community and coarsening](movement-acceptance-final.json) | Pass within scope | 8 / 8 | [16](#build-fingerprints) |
| [Harbor trajectory quality](movement-acceptance-final.json) | Pass within scope | 10 / 10 | [13](#build-fingerprints) |
| [Chicago route workbench](movement-acceptance-final.json) | Pass within scope | 17 / 17 | [13](#build-fingerprints) |

### Geometry and observation

| Suite and evidence packet | Result | Records | Build |
| --- | --- | --- | --- |
| [County linework](geometry-wave16-acceptance.json) | Pass within scope | 10 / 10 | [16](#build-fingerprints) |
| [Footprint shape lab](geometry-wave16-acceptance.json) | Pass within scope | 27 / 27 | [16](#build-fingerprints) |
| [Sampling frames](geometry-wave21-acceptance.json) | Pass within scope | 28 / 28 | [21](#build-fingerprints) |
| [Block relationships](geometry-wave17-acceptance.json) | Pass within scope | 48 / 48 | [17](#build-fingerprints) |
| [Earthquake inhomogeneous patterns](geometry-wave15-acceptance.json) | Pass within scope | 11 / 11 | [15](#build-fingerprints) |
| [Geometry core topology regressions](geometry-wave15-acceptance.json) | Pass within scope | 29 / 29 | [15](#build-fingerprints) |
| [County coverage audit](geometry-wave15-acceptance.json) | Pass within scope | 8 / 8 | [15](#build-fingerprints) |
| [Geometry convex refinement regressions](geometry-wave16-acceptance.json) | Pass within scope | 8 / 8 | [16](#build-fingerprints) |

### Earth systems

| Suite and evidence packet | Result | Records | Build |
| --- | --- | --- | --- |
| [Pixel boundaries](earth-wave16-acceptance.json) | Pass within scope | 33 / 33 | [16](#build-fingerprints) |
| [Missing observations](earth-wave22-acceptance.json) | Numerical repeat and mask visual pass | 55 / 55 (repeated) | [22](#build-fingerprints) |
| [Spectral texture](earth-wave16-acceptance.json) | Pass within scope | 46 / 46 | [16](#build-fingerprints) |
| [Spectral display/profile/transport](earth-wave18-acceptance.json) | Pass within scope | 68 / 68 | [18](#build-fingerprints) |
| [Habitat circuit](earth-wave19-acceptance.json) | Pass with scientific withholding | 4 numerical; 4 withholding; 1 recovery | [19](#build-fingerprints) |
| [Habitat crop perimeter](earth-wave16-acceptance.json) | Pass within scope | 6 / 6 | [16](#build-fingerprints) |
| [Surface preparation](earth-wave16-acceptance.json) | Pass within scope | 48 / 48 | [16](#build-fingerprints) |
| [Spectral regional summaries](earth-wave16-acceptance.json) | Pass within scope | 20 / 20 | [16](#build-fingerprints) |
| [Grid projection accuracy correction](earth-wave14-priority-acceptance.json) | Pass within scope | 10 / 10 | [14](#build-fingerprints) |
| [Terrain boundaries](earth-wave16-acceptance.json) | Pass within scope | 4 / 4 | [16](#build-fingerprints) |
| [Tiled observations](earth-wave16-acceptance.json) | Pass within scope | 58 / 58 | [16](#build-fingerprints) |
| [Tiled observation topology](earth-wave16-acceptance.json) | Pass within scope | 2 / 2 | [16](#build-fingerprints) |

## Focused visual corrections

These checks are separate from numerical acceptance and the complete mixed-result atlas/mobile sweep.

| Suite and evidence packet | Result | Records | Build |
| --- | --- | --- | --- |
| [Street density first-entry publication](street-density-publication-acceptance.json) | Visual/lifecycle pass | 2 / 2 | [20](#build-fingerprints) |
| [Focused corrected-story visual tours](visual-corrections-wave21.json) | Focused visual pass | 7 unique stories; 45 steps | [20](#build-fingerprints), [21](#build-fingerprints) |
| [Sampling render roles](geometry-wave21-acceptance.json) | Visual/lifecycle pass | 2 / 2 | [21](#build-fingerprints) |
| [Wave 22 responsive chart and title verification](visual-corrections-wave22.json) | Focused visual pass | 7 stories; 45 steps; 7 mobile layouts | [22](#build-fingerprints) |

## Retained historical controls

The earlier three-case block bounds smoke is superseded by the 48-case block suite above. The
[Float32 projection baseline](earth-grid-float32-baseline.json) remains a rejected precision control:
correct scalar sampling at the GPU's coordinates did not establish correct projected coordinates.
The [adaptive projection check](earth-grid-adaptive-gpu-review.json) later passed all 32,768 support
decisions, with maximum coordinate error 0.000763260 m under the fixed 0.01 m gate. Failed geometry,
display and circuit experiments remain in the original domain packets and ledger history.

## Build fingerprints

These are emitted-code fingerprints, not dataset or source-tree hashes. Source manifests and public
dataset checksums are recorded separately. Each packet preserves its own before/after build check.

| Wave | SHA-256 |
| --- | --- |
| 13 | `658033867c43774cdb5ca9b9d9fa4aff28575cb70fc8b6d190d343ac32d0660d` |
| 14 | `872af62d507f3cae0bed2749a0b9f8c7736739cf2d99cab71ba8658bf5e817ae` |
| 15 | `5c13c77a7d5dd844146b724ae5c70d49272c63080f028da1228e99f5e22fac65` |
| 16 | `31e545703f8e9c2990ab47eb3c8ec879c9e3532b9a386c16da552c59782e7eaf` |
| 17 | `56f7715e655d8fe93b59b426b475b87848b7947fb8e4d454faff5353908b4a02` |
| 18 | `942c61078b2db07062a259c0bc88889cf064791c8126a6cbca07d7cca40a8307` |
| 19 | `f4742c261276e85b022651df7f3c72f419a7f6d68300c30803839629c67b3675` |
| 20 | `834b1464827bcbbd428b32cfd7950cdb383415c56730decfcdc106c98770fcb9` |
| 21 | `9a5f744c3e1344cc04be45c9626e293af98d36d77f2e058b27acbfe26c476f82` |
| 22 | `7fc7fb3a993176cdbba18b723eb2fbc4719b53090829c7a1b48b9406db57278e` |

## Final integration closure

The completed wave-21 report and emitted manifest are durable. Wave 22 receives focused correction
checks, not another complete atlas sweep: seven shared-layout stories (45 desktop steps and seven
mobile layouts), plus the separate missing-observations six-step/mobile check and 55-case numerical
repeat. The nonexistent `population-support` invocation remains a runner failure; the valid
`chicago-population-support` retry passes. Two four-case source-layout fixtures also pass.

The [final integration packet](integration-final.json) preserves the emitted manifest,
completed build-log hashes and focused-packet links. The source inventory has matching before/after
fingerprints. Full `yarn test` passes after the final test-only corrections: Node 1,246 files and
12,593 tests, plus all eight headless shards with 1,107 files and 12,620 tests. Skips remain explicit:
two files / 439 tests for Node and eight files / 846 tests for headless. The final command exits zero.

The packet preserves the registry and RS02 test-fixture failures, unchanged geometry bounds, and
the successful rerun. Vite dynamic-import and shard-six teardown warnings remain disclosed. Final
formatting, package build, emitted-manifest recheck and `git diff --check` pass. None of these gates
expands the numerical or scientific scope of the bounded domain suites.
