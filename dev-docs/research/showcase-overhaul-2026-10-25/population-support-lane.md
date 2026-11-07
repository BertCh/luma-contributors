# Population support lane: Hyde Park

## Scope and question

This chapter asks how an areal population count can be compared across fine target cells when the
source data only report tract totals, and how that result changes under two explicit support
assumptions. It uses all 12 2020 Census tract polygons whose representative point is assigned to
Chicago community area 41 (Hyde Park). The tract polygons remain whole, as delivered in
`chicago-tracts`; they are not clipped to the community area or arbitrarily sampled.

The selected GEOIDs, in source order, are:

`17031410100`, `17031410200`, `17031410500`, `17031410600`, `17031410700`, `17031410800`,
`17031410900`, `17031411000`, `17031411100`, `17031411200`, `17031836200`, `17031836300`.

Their bounds are `[-87.606407, 41.785874, -87.575801, 41.802512]` and their SVI population sum is
29,559. Population and non-Hispanic Black counts come from CDC/ATSDR SVI 2022, using ACS 2018–2022
five-year estimates. This chapter does not use the separate ACS 2020–2024 income columns present in
the wider Chicago bundle. Source county and tract boundaries are the Census Bureau's
`cb_2022_17_tract_500k` geography.

## Support source and plan

The ancillary source is ESA WorldCover 2021 v200, tile
`ESA_WorldCover_10m_2021_v200_N39W090_Map.tif`. The source endpoint returned HTTP 200 in the prior
review; its expected ETag is `91546122dc8cf821894accc63dee8462-9` (74,618,789 bytes, last modified
2022-10-26). The intended analysis frame is ellipsoidal WGS84 Lambert azimuthal equal-area centered
at `[-87.59, 41.795]`. A 4 km square is divided into 64 × 64 fine cells (62.5 m); each target cell
is an 8 × 8 block (500 m). The full selected tracts must be inside the frame before any calculation.

For each target indicator, convert the native classes to a binary mask first: class 50 for built-up,
class 10 for tree cover, and nonzero for valid support. Area-average those indicator masks onto the
fine grid. Class codes themselves must never be averaged or replaced by a modal class when fractional
support is required. The expected resampling leaves valid fraction equal to one throughout the target
grid; a future missing cell must stay missing and be reported as reduced support.

## Analysis sequence

1. Show the observed tract counts and exact GEOID alignment. A deterministic shuffled source-row
   ordering is joined to the stable target GEOIDs by six-digit tract code inside fixed Cook County
   namespace `17031`; full 11-digit GEOIDs stay visible in provenance. Two groups are retained:
   non-Hispanic Black and all other residents. The alignment recipe's status and validity are gates
   for every downstream result.
2. Compare all-valid uniform area support (uniform within a tract and across every valid analysis
   cell, including mapped water) with built-up WorldCover fraction (residents assumed proportional
   to mapped built-up support). Both are allocation assumptions. Neither makes a fine-cell
   population measurement or a land-only estimate.
3. Transfer the returned GPU totals to 500 m target cells and show a mass ledger: source totals,
   allocated totals, unallocated totals, eligibility, transferred fraction, and completeness. Zero
   eligible support retains its source count. Incomplete coverage, invalid inputs, or capacity
   overflow must suppress published values, never silently truncate.
4. If the target recipe output is usable, compare its returned totals against mapped tree-cover
   fraction. The optional exposure recipe can estimate residents in cells below a chosen mapped-tree
   threshold (at or below the chosen low-tree fraction). This is an exposure-to-a-land-cover-class
   count only, not heat, health risk, public greenspace access, or a causal estimate.

The largest safe polygon tuple capacity is 49,152 (12 source polygons × 4,096 cells). At 64 target
cells, `targetCount × tupleCapacity = 3,145,728`, below the recipe limit of 4,194,304. Pair capacity
of 768 covers every possible tract/target pair. The actual coverage will be much smaller. The recipe's
`complete`, required-count, overflow and stage status are still required gates. The target-zone owner
is the complete fine-cell/500 m zone assignment, not a centroid shortcut.

## Interpretation limits

WorldCover class 50 describes mapped built-up land cover, not residence or occupancy. Class 10 is
mapped tree cover, not accessible or public green space. WorldCover's 2021 observation date and
classification error differ from the ACS 2018–2022 population support. Tract counts are estimates,
not individual locations. Any within-tract allocation depends on the chosen ancillary surface. The
shown 500 m cells are modeled allocations and must be visually distinct from the observed tract
inputs.

WorldCover is CC BY 4.0. Required credit: © ESA WorldCover project 2021 / Contains modified
Copernicus Sentinel data 2021 processed by WorldCover consortium. Product citation: Zanaga et al.
(2022), <https://doi.org/10.5281/zenodo.7254221>. Product manual:
<https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/docs/WorldCover_PUM_V2.0.pdf>.

## Delivery status

The pinned native WorldCover crop and the three 64 × 64 fractional-support bands are bundled under
`public/data/chicago-population-support`. The native crop is 432 × 578 cells and its SHA-256 is
`04247940e005d2f7997907a57b31f3bf0b8f9d0f41039ff82294e49cdc65471a`. The source URL, ETag, source
length and modification time are pinned in the builder. The builder verifies those pins before
rebuilding; the already bundled asset is the runtime input and does not need a source download.

The application frame uses the same ellipsoidal WGS84 Lambert azimuthal equal-area definition as
the builder: `+proj=laea +lon_0=-87.59 +lat_0=41.795 +ellps=WGS84`. Its runtime counterpart is
`createAnalysisFrameProjection({kind: 'lambert-equal-area', origin: [-87.59, 41.795]})`. The runtime
must project every source tract vertex through that projection before fractional coverage is
calculated.

### Implemented data path

The compute file explicitly fetches the bundled `tracts.geojson`; the raster dataset loader does not
provide a geometry contract. It validates the 12 feature GEOIDs and population fields against the
GEOID and population files, and validates every non-Hispanic Black count as finite and within its
population total. The bundled asset check reads 12 rows, population sum 29,559 (range 1,281–3,931),
and non-Hispanic Black sum 7,769 (range 102–1,719). All three fraction bands are finite in `[0,1]`;
the valid fraction is exactly 1 throughout the 64 × 64 grid.

The source counts are split into `[non-Hispanic Black, other residents]`, deliberately reversed,
and aligned by six-digit tract code in Cook County namespace `17031`. Full 11-digit GEOIDs remain in
the aligned ledger. The actual `addPopulationSupportAlignmentRecipe` output is summed in a GPU
kernel and supplies `addDasymetricRedistributionRecipe` source totals; the ledger reads the aligned
group values and alignment status from GPU output. Redistribution readback includes returned
eligible weights and transferred fractions as well as allocated/retained totals. Its compact
support ledger reports sources with positive eligible support and the population-weighted
transferred share. Zero eligible weight remains a complete outcome with source count retained.

The redistribution recipe uses polygon capacity 49,152 and pair capacity 768. One dirty-written
ancillary buffer switches between uniform ones and actual class-50 fractions. Uniform area covers
all analysis cells, including water. A fixed `[0, 3,500]` cividis scale covers the independent
reference maxima 2,606.71 (uniform) and 3,087.39 (built-up). Source mode uses measured tract counts
on its own `[0, 4,000]` cividis scale. Modeled maps show tract outlines only; source fills appear
only in the opening observed-count step.

`addExposureByZoneRecipe` consumes the returned 64 target totals and the actual WorldCover tree
fraction averaged over the same 8 × 8 fine-cell blocks. Its score is `1 - tree-cover fraction`;
thresholds correspond to tree fractions at or below 10%, 25%, or 50%. Four whole-cell grid
quadrants report estimated modeled residents in low mapped tree-cover cells, with the recipe's
observed coverage and zero-denominator cases represented explicitly. This is not heat, health,
public greenspace access, or an occupancy estimate. A verified valid fraction of 1 gates the
current crop; missing support remains a failure rather than zero-valued land cover.

The scene has six question-led steps: observed source counts, exact ID alignment, uniform-area
allocation, built-up allocation and mass ledger, tree-cover population-weighted summaries, and
interpretation limits. Map display is independent from allocation support: the support selector
chooses uniform area or built-up WorldCover weights, while the map selector chooses observed
counts, modeled allocations, the actual fine-grid built-up fractions, or target-grid tree cover.
Tree-cover steps set built-up support explicitly and expose both support choices, so direct
navigation and prior display history cannot change the estimate silently. The active support is
shown in a readout, and the code snippet names the selected ancillary input. The fine 64 × 64
built-up fraction view uses its actual 0–1 values; tree-cover quadrants have visible center
boundaries matching the NW/NE/SW/SE chart and readout. Charts and compact status/readouts use
GPU-returned allocation, alignment, and exposure values. Graph execution occurs only for a dirty
generation; each support/threshold change clears prior computed evidence, and failed status sets
the scene status before results can appear.

### Scoped verification and handoff

The population-support source handoff is source-ready after explicit support selection, visible
support readout, fine-grid built-up input display, and tree-quadrant locators. No GPU or browser
execution is claimed for this review update.

- `yarn workspace luma.gl-examples-deck-spatial-analysis-showcase typecheck`: passed with no
  diagnostics.
- `node scripts/check-story-steps.mjs statistics`: passed, including this scene.
- `node scripts/check-cartography.mjs --strict statistics`: passed, including this scene.
- `node scripts/check-wgsl-identifiers.mjs src/scenes/statistics`: passed (459 templates scanned).
- `node scripts/check-datasets.mjs`: passed, 70/72 datasets, 715 assets, 0 errors.
- `biome check` on the lane's TypeScript files: passed after formatting.
- No browser or GPU execution was performed; this handoff establishes source wiring and static
  checks, not WebGPU acceptance or a rendered visual review.

After the source and story checks passed, the scene metadata was promoted from `.scene.ts.wip` to
`.scene.ts` as the final lane change. No browser or GPU run was part of this authorization.
