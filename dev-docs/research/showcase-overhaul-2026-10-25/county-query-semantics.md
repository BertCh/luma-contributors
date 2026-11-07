# County query semantics: planned dataframe depth

Status: researched and assigned after fire-focus implementation, not yet implemented or credited.

## Question and real observations

“Where did the counties go?” is an audit of missing records and join definitions, using the
existing `us-counties` (3,109 contiguous-US counties/county equivalents) and source-aligned
`us-elections` data. It is not an election forecasting or political persuasion story.

Direct inspection of bundled float32 `demTwoParty{year}.bin` values finds 3,100 finite and nine
missing observations in 2016 and 2020, versus 2,859 finite and 250 missing in 2024. These numbers
are properties of the frozen local extraction. They do not describe the current completeness
of MEDSL’s upstream repository. The 2024 local table was rebuilt from precinct results with
state reconciliation failures excluded; Jackson County MO is separately missing in that year.
The packaged geography excludes Alaska and Hawaii. Connecticut’s nine planning-region units
are distinct from legacy county reporting geography; do not silently fabricate a crosswalk.

## Primary sources reviewed

- [Census county changes](https://www.census.gov/programs-surveys/geography/technical-documentation/county-changes.html): Connecticut adopted nine Councils of Governments as county-equivalent units; geographic vintage matters.
- [MEDSL 2024 repository](https://github.com/MEDSL/2024-elections-official): precinct-level source, with explicit instructions to avoid double-counting TOTAL plus disaggregated voting modes. Frozen local preprocessing is documented separately.
- [MEDSL 2024 codebook](https://raw.githubusercontent.com/MEDSL/2024-elections-official/main/codebook.md): county FIPS is a five-digit geography key; current Connecticut records use planning regions. The source can contain suppressed vote counts.
- [County returns 2000–2020 DOI](https://doi.org/10.7910/DVN/VOQCHQ): the declared source in the local manifest; direct webpage access hit a JavaScript verification screen, so no additional contents were inferred from it.

## Implementation design

Use genuine FIPS keys from `us-counties`, which the elections manifest explicitly aligns to.
For a unique-right join experiment, an adapter may construct a sparse right table containing
only finite source election observations. Document that this is extraction of observed rows,
not an original upstream batch layout. Then actually execute inner, left, semi and anti joins
and source-aligned lookup, show match/missing counts and bounded publication/index status, and
render accepted/missing county support. A missing result is not zero votes.

Maintain the source values, including NaNs. In the semantic analytical view, map unavailable
observations to explicit per-column validity sidecars. Show `isValid` and `isNull` predicates
with counts. A separate raw-NaN ordering view may preserve NaN as a valid numerical value to
explain why NaN placement and explicit null placement are separate controls; label the adapter
interpretation rather than claiming the source supplied two kinds of missing values.

Compare full global ordering with explicit application-created state batches. Stable row IDs
must keep the mapped county associated with its values. Expose ascending/descending and
nulls/NaNs first/last where they visibly affect outputs; do not substitute top-K for full sorting.
Use bounded row-ID/readout samples to show the order, with precise ranks and county names.
Derived changes `(share2024 - share2020) * 100` have percentage-point units and require both
observations valid; missing observations must remain missing. Grouped summaries must state
whether counties or votes are weighted. Do not call votes/population “turnout” without a valid
eligible-voter denominator. Preserve all source columns and geography provenance.

Graph APIs require unique uint32 right keys, symmetric dictionary metadata where dictionaries
exist, exact vector names matching table keys, explicit nullable schemas and aligned validity
sidecars. These were real production failures in the preceding earthquake lab. Review every
new table boundary before requesting a production test. Lifecycle/protocol methods, duplicate
pass-through authoring overloads and sort implementation strategy are not separate scientific
stories merely because coverage counts them.

## Source review and acceptance status

The implemented five-step `county-query-semantics` story now loads the actual binary polygon
layout, including multipart counties and holes, plus the aligned county-name file. It retains
source float32 NaNs and supplies explicit nullable fields and uint32 validity vectors. The
finite-observation right table has 2,859 unique FIPS keys; four joins and lookup execute on the
GPU, with bounded capacity and uniqueness diagnostics. Derived change, validity predicates,
grouped counts/means, four global full sorts and four state-batch full sorts also execute.

Review corrected an initial nonexistent GeoJSON assumption, initially hidden grouped means,
single-state sort excerpts, and NaN payload replacement. Group summaries now show named states
with unavailable AK/HI and clarify that 57 denotes numeric FIPS slots including gaps. Sort heads
show several state batches and respect the public compiler's `sourceRowIndexOffset` behavior.
The map deliberately remains a source-aligned lookup mask while join counts change; the UI
states that scope. Raw NaN placement independent of null validity is not demonstrated.

`county-query-reference.py` reads source bytes independently and records join denominators,
nullable float32 change, global rankings, state means/counts and state-batch ranking heads.
The browser harness `scripts/check-county-query-reference.mjs` checks these visible outcomes.
Scoped Biome checks passed on both scene files; GPU acceptance remains pending. An attempted
worker browser launch failed before Chromium started, so it supplies no runtime evidence.

### Worker command-scope incident

The correction worker improperly invoked `yarn lint fix` with the two scene paths as arguments.
The repository wrapper scanned the entire workspace and reported 24 fixes, plus two unrelated
unused-variable errors in point-cloud-change. It did not list affected paths. Parent integration
was notified immediately, with the exact command and `/tmp/luma-showcase-overhaul-20261025/cli-dataframe-review.jsonl`.
No shared changes were reset or reverted. Twenty other scene files shared the 16:18:33 local
modification time, which is circumstantial evidence only. Future worker prompts explicitly
prohibit root formatter wrappers as well as builds and unallocated browser/GPU tests.

## Production tour evidence

Parent wave8 production sweep passed all five county steps with zero scene errors and zero WebGPU errors (`/tmp/luma-showcase-overhaul-20261025/wave8-browser/report.json`). Screenshot05 was reviewed: the teal/plum availability map is visible, and default global/state-batch leading rows match the independent reference (including missing Connecticut). The duplicate long ranking paragraph was then removed from the evidence box; structured lists remain. All interactive numerical variants still await the targeted harness.

Targeted production numerical harness PASSED eight scenarios; report `/tmp/luma-showcase-overhaul-20261025/county-query-reference-browser.json`, errors[]. Source/null/derived/state-group values, fourjoins, bothdirection global andstateheads, andnull-first checked.

## Nullable expression extension

Source reviewed; numerical acceptance remains pending for the new ninth harness scenario.
The full 3,109-row nullable frame now actually executes Republican two-party complement
`-share + 1`, normalized Democratic margin `(share - 0.5) / 0.5`, and three-valued NOT,
null-OR-NOT, equality and inequality predicates. Source fractions are in [0,1]; displayed
percentages are conversions. The 250 unknown observations stay unknown unless explicitly
included by `isNull OR NOT(share > 0.5)`. Unweighted county averages are not vote totals.

`county-expression-reference.py` independently reads the archived float32 values. It finds
2,859 finite values, 402 Democratic majorities, 2,457 valid NOT-majorities and 2,707
null-or-NOT rows. There are no exact 0.5 ties in this archive, so the zero equality result
is not presented as empirical verification of a nonempty tie case. Mean Republican share
is 0.6826960746330496 and mean normalized margin is -0.36539215012087. The existing
`check-county-query-reference.mjs` now adds this ninth scenario; the original eight scenarios
remain accepted, and the new one awaits the rebuilt production bundle.

## Final ordering representation diagnostic

Source reviewed and frozen for the next production build. The default analytical validity
sidecar is derived from source finiteness and remains 2,859 valid / 250 null rows. A separately
selected diagnostic borrows the same full table and state batches with all-one 2024 validity
sidecars. Actual GPU predicates then report 3,109 valid / zero null rows, while all 250 archived
NaN payloads remain unavailable observations. This demonstrates that validity and IEEE
finiteness are different contracts; it does not replace missing shares with measurements.

Four diagnostic global and state-batch sort variants cover ascending/descending and NaN
first/last. They reuse the placement control in the existing ordering step, which explicitly
states whether placement applies to nulls or valid NaNs. The map, joins, analytical filters,
expression summaries and grouped means retain the default nullable frame. Separate GPU-to-GPU
copies gather only eight leading and eight trailing global row IDs into 64-byte preview
buffers per variant; state excerpts retain one row from each of eight batches. Source-null
rows retain explicit `null (archived NaN payload)` wording.

`county-nan-sort-reference.py` reads the archived bytes directly, including raw uint32 payload
bits, and independently sorts finite values and unavailable source rows with row-ID tie breaks.
All 250 missing payloads are 0x7fc00000. It records global heads/tails and state-local heads for
eight representation/direction/placement combinations. The production harness now checks
17 scenarios: the prior nine query/expression cases plus these eight representation states,
including actual GPU validity counts and preservation of 2,859 finite observations. No browser
acceptance is claimed for this extension until that rebuilt harness passes. The three touched
scene files pass scoped Biome; the harness passes syntax checking.
