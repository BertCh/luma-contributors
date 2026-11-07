# Earthquake reference populations and transformations

## Question and source

The proposed story asks what a standardized value is relative to. It uses the unchanged archived
USGS-derived `poopdeck-earthquakes` float32 magnitude/depth columns and type codes. All 77,215
type-zero earthquake rows form the default population; an M≥6 subset has 626 rows. A shallow
subset uses depth≤70 km, preserving negative source depths. All populations require finite
selected values. Magnitude and depth are different quantities; rescaling does not make them
scientifically interchangeable or infer hazard. The archive starts at magnitude four, and the
extract lacks magnitude-type metadata, so this cannot establish magnitude-scale comparability.

## Primary methods and interpretation

[GeoDa's variable-standardization workbook](https://geodacenter.github.io/workbook/01_datawrangling_1/lab1a.html#variable-standardization)
defines the five transforms used by the public contributor. Deviation subtracts the reference
mean; z-score also divides by its standard deviation. The MAD option divides that centered
value by **mean absolute deviation about the mean**, not median absolute deviation. Range-adjust
divides the original value by its observed range; range-standardize first subtracts the minimum.
The latter maps a nonconstant observed range to [0,1]; the former need not. The GPU API exposes
sample versus population variance for z-score. Changing the reference mask changes the summary
quantities and therefore the transformed values.

[USGS explains earthquake depth and negative depths](https://www.usgs.gov/faqs/what-does-it-mean-earthquake-occurred-a-depth-0-km-how-can-earthquake-have-a-negative-depth)
in terms of reference datum and varying network estimation practices. Negative source depths
must therefore not be silently treated as invalid. The source fields remain unchanged.

The producer's documented degenerate conventions are part of its API: constant-population
deviation/z-score outputs zero (sample z-score needs at least two included rows); MAD and range
methods preserve constant values. These are not general statistical identities. The real masks
in this story are nondegenerate; no demonstration of unobserved edge cases is claimed.

## Design and verification

Map colors and GPU histograms use explicit original or transformed units. Global summary and
masked support counts must come from GPU computation; any CPU mask adapter is disclosed. The
same population supports original and transformed views. Excluded rows remain NaN. The map is
a catalogue distribution, not an earthquake forecast. A bounded probe plus numerical summaries
will be compared with a source-byte Python reference using independent scalar formulas.

Status: implementation assigned to the existing Luna worker; no runtime acceptance yet.

## Reviewed implementation

All five transforms and both z-score conventions execute in one reusable GPU graph, followed by seven column profiles. Review corrected byte offsets, histogram domains versus map extent, population-SD labeling, mask exclusion versus within-mask null counts, missing chart registrations and stale charts. A fixed source row229 belongs to all three masks (M6.299986, depth16.999472km); its actual original/transformed GPU values are gathered into a bounded probe buffer. The36-case browser harness checks all2fields×3populations×6method/convention outputs against independent min/max/mean and fixed-probe references. Source/scoped checks passed; runtime pending.

## Production numerical acceptance

All36 configurations passed the independent source oracle on wave10: magnitude/depth × all/M≥6/shallow populations × deviation, sample/population z-score, mean-absolute-deviation scaling, range-adjust and range-standardize. Checks include finite population counts, transformed minimum/maximum/mean and the same source-row229 GPU probe. No page/console errors. Report: `/tmp/luma-showcase-overhaul-20261025/standardization-reference-browser.json`; rendered final control state reviewed. Acceptance applies to these configurations, not all possible inputs or parameter combinations.
