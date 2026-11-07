# Fire-scar topology: research, implementation and acceptance record

## Scope

Following the initial dataframe delivery, integration assigned this track the new
`raster/fire-scar-topology.scene.ts` and `.compute.ts` files. The same Luna implementer is reused;
this is coordinated with the earth-systems planner, who owns source acquisition and spectral/
fractional-remapping stories. No shared raster engine or public algorithm API is edited here.

The question is **what makes a spectral-change patch connected?** Thresholds, neighborhood
filters and connectivity change an analyst-defined mask. The story must make those decisions
visible rather than present one mask as an authoritative fire boundary.

## Source and support

The earth-systems track supplies `dixie-spectral`: pinned Sentinel-2 Level-2A acquisitions
`S2B_10TFK_20210713_0_L2A` and `S2B_10TFK_20210921_0_L2A`, calibrated float32 reflectance for six
bands on the same 250 × 250 grid. The exact UTM 10N extent is
`[662800, 4441040, 677800, 4456040]` metres, with 60 m cells and north-first rows. One cell is
3,600 m² or 0.36 hectares. The full window is 225 km², a small window within the Dixie Fire area.
This does not depict the whole fire.

The reproducible builder, catalog and checksums belong to the earth track at
`examples/deck/spatial-analysis-showcase/scripts/data/dixie-spectral/`. This scene uses NIR and
SWIR2 before/after plus **both** strict land-validity masks. Land validity requires every one of
nine contributing native 20 m SCL cells to be class 4 (vegetation) or 5 (non-vegetated), together
with finite calibrated spectral support. Water is excluded. Cloud, shadow, uncertain and other
rejected classes are missing analytical support, not unchanged land. Strict SCL filtering may
also omit genuinely burned dark surfaces; retaining only accepted land must not be described as
an exhaustive burn-area estimate.

Primary research consulted on 2026-10-25:

- [USGS Burn Severity glossary](https://burnseverity.cr.usgs.gov/glossary): dNBR is pre-fire NBR
  minus post-fire NBR. This scene uses that sign convention.
- [USGS NBR documentation](https://www.usgs.gov/landsat-missions/landsat-normalized-burn-ratio):
  the NIR/SWIR contrast motivates the index. We use Sentinel-2 bands with their own source
  calibration; the presence of an index alone does not establish a validated severity class.
- [USGS hazard assessment requirements](https://www.usgs.gov/programs/landslide-hazards/hazard-assessment-requirements):
  operational soil-burn-severity assessments require field-validated inputs. This exploratory
  spectral mask does not provide that validation or a hazard forecast.
- [Copernicus S2 processing](https://sentiwiki.copernicus.eu/web/s2-processing): SCL distinguishes
  vegetation, non-vegetated land, water and obscuring/uncertain classes. The source builder uses
  explicit native-pixel support rules, not a nearest-neighbor QA mask at the output resolution.

## Algorithms and interpretation

1. Compute NBR from calibrated NIR/SWIR2 on each date and subtract post from pre on the GPU.
   Summaries and map explicitly show the accepted support.
2. Compare a manual dNBR cutoff with a masked histogram/Otsu cutoff. Otsu separates this window's
   histogram statistically; it does not identify an ecological or soil-severity threshold.
   The theoretical dNBR domain is [−2, 2]; the actual maximum exceeds 1, so a [−1, 1] histogram
   would truncate real observations.
3. Compare rank/median filtering with binary opening/closing where supported. Explain footprint
   radius in cells and metres, and whether missing neighbors propagate or are ignored. Invalid
   centers remain invalid. Closing may connect nearby accepted pixels; opening can erase narrow
   but real features. Neither is automatically a correction.
4. Compare four- and eight-neighbor region definitions. Patch colors are nominal component IDs.
   Dense labels, convergence, capacity overflow and valid support feed region measurements;
   a failed label pass must not publish plausible-looking area statistics.
5. Skeleton, medial-axis and marker-watershed variants are a separate bounded focus view if
   implemented. The library's exact medial axis and watershed each require ≤16,384 cells; use
   a geographically explicit small crop, never the complete 62,500-cell grid. The medial axis
   retains maximal inscribed disks and can include plateaus; it is not synonymous with
   iterative binary thinning. Its radius is in pixels and requires conversion to
   metres. A spectral marker watershed segments an image; it is not a drainage-basin model.

`GPURasterSkeleton` publishes reserved word `0xfffffffe` when unconverged. This must be surfaced
as unavailable, never treated as a nonzero foreground mask. Marker placement and crop boundaries
must be explicit assumptions. These advanced variants are not credited until executed and shown.

## Independent numerical reference

`fire-topology-reference.py` uses only Python's standard library. It reads the actual bundled
float32 bytes, rounds NBR arithmetic to float32, intersects the strict land masks and independently
counts connected components by a CPU flood fill. It creates `fire-topology-reference.json`.

```sh
python3 dev-docs/research/showcase-overhaul-2026-10-25/fire-topology-reference.py
```

Both land masks plus finite bands and positive NIR/SWIR2 sums leave **35,583 / 62,500 cells**
(56.93% of the window). Before support is 59,709 land cells and after support is 35,695.
The source/earth track independently obtained the same intersection and verified all sixteen
binary payload checksums.

For **raw dNBR ≥ 0.27, with no smoothing or morphology**:

- Foreground: 22,375 cells, 8,055 hectares within accepted support.
- Four-neighbor patches: 953; largest 9,026 cells / 3,249.36 hectares.
- Eight-neighbor patches: 531; largest 11,209 cells / 4,035.24 hectares.

The reference also covers 0.10, 0.44 and 0.66 cutoffs. At 0.10, four-neighbor count is 1,117,
so a 1,024-region output would truncate this real case. Capacity must be larger or truncation must
be explicitly displayed. Reference values apply to the stated raw pipeline, not to median-filtered
or morphologically transformed masks.

A separate source oracle at cell 26,488 yields dNBR approximately 0.185309276. The center cell
31,250 is invalid on the after date and should not be an initial inspection target.

## Implementation and verification status

- Core Luna implementation: complete against registered `dixie-spectral`; arithmetic, histogram/Otsu, manual threshold, median/opening/closing, 4/8 connectivity, dense labels and measurements are reachable. Bounded focus extension is being developed in separate `.wip` drafts.
- Numerical source/support/flood-fill reference: completed and sent to implementer.
- Core scoped Biome, full showcase package TypeScript and story-step checker pass. Production build/native WebGPU numerical and control acceptance remain pending integration. Static review corrected dense-label rendering, legend scale mismatch, A→B→A graph-swap race, stale-summary association, and nonconvergence/overflow readout handling.
- No algorithm is counted as browser-covered solely because its name appears in metadata or code.

## Dataframe validation follow-up

Actual production browser testing of `earthquake-query-lab` found two initialization failures that
TypeScript did not detect. First, the event-type join required matching category dictionaries on
both key columns; the two right-side tables now carry the identical full category domain. Second,
the single-key table's `GPUVector.name` differed from its column key; both are now `typeCode`.
These failures are recorded as failures, with a new browser retest requested after each fix.
The underlying private query-graph inspection limitation remains separate from execution.

## Expanded independent acceptance

`python3 fire-topology-reference.py --extended` additionally writes
`fire-topology-extended-reference.json`: all opening/closing/median radius 1–3 scenarios,
using square footprints, ignored missing neighbors and missing-center preservation, with
both connectivity definitions. At the default 0.27 threshold, closing radius 1 and eight-neighbor
connectivity yields 23,920 foreground cells, 8,611.2 hectares, 408 patches and a 4,589.64-hectare
largest patch. These are references, not reported GPU observations.

The explicit 64×64 focus at column 176, row 56 has projected bounds
[673360, 4448840, 677200, 4452680], 3,647 accepted cells and 3,134 raw threshold foreground cells.
An independent Zhang–Suen implementation reaches 913 skeleton cells after 15 full rounds
including the final unchanged round. The maximum nearest-background/exterior-center radius
is √229 pixels, approximately 907.965 metres. These apply to the raw 0.27 mask only.

`examples/deck/spatial-analysis-showcase/scripts/check-data-topology-reference.mjs` drives
production UI controls and compares published GPU readouts to these independent references.
It tests eight raw threshold/connectivity scenarios, nine transformed masks, and Otsu’s raw
18,656-cell mask. It also tests dataframe all-type, earthquake-only and magnitude≥6 filters.
At creation this harness has passed syntax checking but awaits the parent’s serialized GPU slot.

## Dataframe production evidence

The parent’s production GPU retest passed all six `earthquake-query-lab` story steps with zero
reported errors after the dictionary and vector-name fixes. Evidence is in
`/tmp/luma-showcase-overhaul-20261025/wave3-fixes-browser/report.json`. Screenshots 01 and 06
were visually reviewed: 77,231 archived rows and 77,215 queried earthquakes agree with source
bytes; selected points and rank stars render. The chart text contrast problem was reported
to the parent and corrected in shared map inset CSS. The 626 magnitude≥6 case still requires
the targeted numeric harness; no claim that the default tour tested it.

## Core production acceptance passed

Production tour: four steps, zero reported errors in
`/tmp/luma-showcase-overhaul-20261025/wave4-browser/report.json`. Screenshots 01 and 04 were
visually reviewed: continuous dNBR with correct scale, actual categorical dense labels, accepted
support 35,583, and default 408 patches / 23,920 cells / 8,611.2 ha / largest 4,589.6 ha.

The combined numeric harness subsequently **passed all 21 scenarios** against production
`http://127.0.0.1:5313/`: three earthquake query cases, eight raw fire threshold/connectivity
cases, nine cleanup/radius cases, and raw Otsu. Every checked GPU count/area agreed with the
independent source-byte reference at displayed rounding. Report:
`/tmp/luma-showcase-overhaul-20261025/data-topology-reference.json`. No console or page errors
were recorded by this numerical harness. The separate tour harness supplies uncaptured GPU
validation instrumentation. The first numerical attempt failed only because the styled switch
intercepted Playwright's pointer check; native keyboard activation fixed the test selector.
The bounded focus extension and class-boundary RegionGraph remain a subsequent acceptance gate.

## Bounded extension source review

The same scene now has six story steps. Its additional source-reviewed paths are:

- `GPURasterRegionGraph` on accepted binary mask classes, after a GPU relabel pass excludes
  invalid support from adjacency entirely. The displayed product is shared four-neighbor
  contact length and mean dNBR gap; no merge mapping is applied and no full-perimeter claim
  is made. The raw 0.27 case has an independent 3,858-edge / 231.48 km reference.
- GPU row-wise crop copies retain the explicit 64×64 focus geography. `GPURasterSkeleton`
  with 64 iterations displays only converged nonreserved cells. `GPURasterMedialAxis`
  displays maximal-disk centers and reports maximum pixel radius converted to metres.
- `GPURasterWatershedSegmentation` uses continuous negative dNBR and accepted support,
  independently of binary threshold/cleanup. Sixteen analyst lattice sites each relocate
  to the first valid source cell within a four-cell square search. They are analytical
  marker choices, not observations. Invalid cells block flooding; unseeded disconnected
  regions remain transparent. No drainage interpretation is made.

Review corrected a seed-loop bug that added multiple markers per site, a shared readback
signature that could misattribute stale focus outputs, and a class-map branch accidentally
placed in `encode` rather than `getLayers`. Core and focus now use separate readback signatures.
The focus camera, high-contrast skeleton/seed colors, legends and cleaned-mask class wording
were reviewed. Scoped TypeScript and raster story checks pass. Runtime/numerical extension
acceptance is still pending, separately from the passed core. The harness’s `fire-focus` mode
checks class contact, skeleton count, accepted support, radius, markers and convergence while
capturing all focus map variants.

## Thinning research correction

The scene deliberately avoids claiming unconditional topology preservation for Zhang–Suen.
The primary [1986 ACM technical comment by Lü and Wang](https://www.csie.ntu.edu.tw/~hil/paper/cacm86.pdf)
documents lost small patterns, including certain four-pixel-square configurations, and altered
thin diagonal structures. The local contributor uses the original two-phase neighborhood
rules; convergence means no more pixels changed, not that every original component survives.
This correction is reflected in the focus body/caveat and does not change the algorithm or
independent 913-pixel reference. The original ACM DOI endpoint was blocked during browsing;
no unread full-paper details are claimed.

## Extended production tour evidence

Parent wave8 production sweep passed all six fire steps with zero scene errors and zero WebGPU errors (`/tmp/luma-showcase-overhaul-20261025/wave8-browser/report.json`). Screenshot06 was reviewed: the rendered skeleton is visible; 3,647 accepted focus cells, 913 converged skeleton cells, 908.0 m maximum discrete radius and 16 analyst markers match independent source references. Focus camera zoom was increased from12.6 to13.4 after this visual review to use more map space. All four focus-view interactions and class-contact numerical assertions remain queued for the dedicated harness.

Targeted production focus harness PASSED five scenarios; report `/tmp/luma-showcase-overhaul-20261025/fire-focus-reference-browser.json`, errors[]. Raw class-boundary231.48km and allfourfocusviews verified support3647,skeleton913,maximumradius908.0m,16markers/completequeue. Medial-axis andwatershed screenshots reviewed; no claim intrinsic watershedbasins validated against a separate fullgeometry oracle.

## Final selection-policy and adaptive-region closure (source work in progress)

The finite final extension remains inside this story. Selection controls compare above, below,
and bounded range tests, with inclusive/exclusive endpoints. Paired land QA stays fixed. These
are analytical subsets of measured spectral change, not observed burn classes. The default
0.27 and 0.66 endpoints coincide with no observed float32 dNBR value: the default above count
is 22,375, below is 13,208, and the interval contains 8,634 cells under either endpoint policy.
The genuine source-cell anchor 26,488 has dNBR approximately 0.185309276 and one exact match;
using that same GPU value as a threshold can illustrate equality without inventing data.

`fire-policy-reference.py` independently reads the archived bytes, reconstructs the documented
marker-controlled minimax/FIFO flood using Python's heap queue, and performs fixed and adaptive
region merging using count-weighted means. Four-neighbor flooding reaches 3,593 of the 3,647
accepted focus cells; eight-neighbor flooding reaches 3,595. Unseeded disconnected support stays
unlabeled. All 16 authored markers retain populated labels, although some seeded basins contain
only one cell. This is an explicit segmentation assumption, not sixteen observed ecological units.

At merge threshold 0.20 and 16 rounds, four-neighbor fixed original-mean single linkage leaves
six roots, whereas sequential closest-pair merging with recomputed observation-weighted means
leaves five. The smallest decision margin in the adaptive reference exceeds 0.003 dNBR, well
away from float32 rounding. The reference records all mappings, counts, pending flags, means,
shared edges and threshold margins for both connectivity choices, both merge modes, four
thresholds, and 1/4/16-round budgets. It does not derive expected results from GPU readbacks.
Actual implementation and new numerical acceptance remain pending; previous core/focus passes
above do not certify these added policies.
