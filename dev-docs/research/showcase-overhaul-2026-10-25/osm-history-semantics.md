# Sampled contributor moments and exact temporal reductions

The unchanged `poopdeck-osm-nyc` archive contains 400,000 sampled node creations from a declared
901,827-node history. Anonymous contributor ranks describe accounts in that source, not identified
people, motives, expertise or import provenance. New methods extend the existing contributor and
creation-time stories rather than adding catalogue cards.

## Group moments

[NIST's skewness and kurtosis reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)
distinguishes unadjusted moment coefficients from small-sample corrections and distinguishes
ordinary kurtosis from excess kurtosis. The API computes g1=m3/m2^(3/2) and g2=m4/m2²−3 using
central moments divided by n. These are descriptive unadjusted coefficients; the configured
sample variance used for row z-scores does not change their definitions. Constant groups have
undefined skewness/kurtosis. A minimum of five sampled records is an analyst display gate,
not a guarantee of estimator stability or a hypothesis-test threshold.

Creation-day moments describe the time distribution of sampled node creations within an
anonymous account. They do not measure account quality or establish automated/manual work.
The existing float32-day adapter has roughly 40-second precision at the end of the interval.
The independent reference therefore uses those same stored float32-day values as inputs,
then computes moments by separate scalar formulas. No reference result feeds the scene.

## Exact archived time

Direct inspection of all 400,000 archived uint32 timestamp offsets confirms nondecreasing order;
293,879 adjacent pairs share a timestamp. The stored resolution is one second, with origin
2007-06-27T20:05:28Z, and offsets range from 3 to 596,079,008 seconds. Exact Int64 word conversion
can preserve this archived precision; it cannot recover missing upstream millisecond precision.

The public temporal reducer assigns fixed-duration buckets by exact integer division and drops
out-of-window records. Its sorted path trusts the monotonicity contract and limits dispatch to
the matching range; the full-scan path supplies an independent execution comparison. Earliest
and latest timestamp ties choose the lowest source row index. Source order is verified before
enabling that contract. Fixed-duration buckets must not be called calendar months.

Status: both extensions are actively implemented; no new numerical acceptance yet.

## Reviewed source delivery

Contributor extension adds actual grouped SD/g1/g2 and row-z outputs, finite-count n≥5 gate, bounded selected-row GPU probe, dense-capacity status, stale-summary signatures, and unavailable formatting for undefined moments. Its new shaderNaN constructors use the runtime-bit pattern required by the checker. The map states saturation beyond±3sampleSD. Temporal extension verifies monotonic source order, runs exact-second sorted/full-scan reducers on six kind slots in a fixed30day window, compares outputs/status onGPU, and displays actualsource identities/timestamps. The ordinary float32-day map remains expressly approximate context. No new catalogue card was added.

Independent group and temporal references plus a13-case combined browser harness are ready. The temporal cases include empty/one-kind/all-six-kind windows and latest-timestamp ties of multiplicity11. No new runtime acceptance yet.

## Production evidence and binding-limit repair

Wave10 numerical validation passed seven contributor-group scenarios: full-history ranks1/2/10/50 and the concentrated other-tag window ranks1(empty)/2/10. The independent reference checks counts, sample standard deviations, population-moment g1 and excess g2, and a bounded GPU row z-score. Report: `/tmp/luma-showcase-overhaul-20261025/osm-group-reference-browser.json`.

The temporal scene did not initialize: its custom comparison kernel bound11storage buffers against the device limit10. This was scene code, not a demonstrated contributor failure. The comparison is now split into7bindings(count/extrema) and6bindings(first/last), preserving the mismatch word by OR before the independent occupied-slot comparison. Source reviewed; rebuilt GPU and six exact-window cases remain required. The initialization failure is preserved at `/tmp/luma-showcase-overhaul-20261025/osm-temporal-initialization-failure.json`.

Verifier repairs: match the current finite-day wording, open the target story step explicitly, reset cumulative filter state, and fail promptly on a surfaced initialization error. These verifier changes do not alter GPU expectations.

## Wave 11 exact-time acceptance

After the storage-binding split, all six temporal scenarios passed on production WebGPU.
The independent archived-byte reference matches count, minimum/maximum source row, exact
first/last identities and timestamps, occupied-kind list and zero-overflow status for both
sorted-range and full-scan paths. The actual GPU comparison reports zero mismatches in every
case. Windows include empty support and repeated timestamp ties; the lowest source-row tie
policy is preserved. Report: `/tmp/luma-showcase-overhaul-20261025/osm-temporal-wave11.json`,
with empty page/console error arrays. The saved map screenshot was reviewed; float32-day
context remains explicitly separate from the exact-word audit. Both browser sessions were
closed before releasing the shared GPU slot. This establishes the six configured cases,
not every temporal word width, bucket or reduction option.
