# Analysis experience vocabulary

### `GPUAnalysisContiguityStatistics`

`GPUAnalysisContiguityStatistics` composes contiguity weights, a weights summary and a fixed
median into a full-population polygon profile. The [contiguity statistics guide](gpu-analysis-contiguity-statistics.md)
defines the topology, caller-owned layouts, capacity recovery, validity and named limits.
Check both candidate and final overflow before presenting any profile claims.
The composition fixes the median quantile to `[0.5]`; callers cannot supply quantile settings.

#### How it computes

{/* lens:GPUAnalysisContiguityStatistics */}

::lens{scenario=gpu-spatial-analysis.analysis-contiguity-statistics rung=result caption="Two adjacent squares and one island have cardinalities [1,1,0] and median 1."}
::lens{scenario=gpu-spatial-analysis.analysis-contiguity-statistics rung=formula caption="Count neighbors for every feature, including islands, then take the linear median."}
::lens{scenario=gpu-spatial-analysis.analysis-contiguity-statistics rung=graph caption="Trace contiguity, summary, quantile and overflow-validity stages in one graph."}

### `GPUAnalysisSpacing`

`GPUAnalysisSpacing` computes full-population point spacing from resident `float32x2` positions,
an optional `uint32` mask and stable identities, and a required planar/native or
longitude-latitude/great-circle spatial context. Geographic distances use the mean Earth sphere
in metres. It composes published graph sorting, scanning, reductions and `GPUNeighborSearch`
(which builds the published `GPUGridIndex`). A second indexed search over distinct locations
preserves `nearestNeighbor.nonZeroMedian` when locations repeat.
Longitude sorting preserves signed endpoints rather than adding 360 to small negative coordinates.
Preparation and longitude keys share neighbor search's integer-significand reduction, preserving
the exact remainder of large stored f32 longitudes (for example, stored `1e20` equals `-88°`).
Neighbour distances retain canonical f32 coordinates. Geographic grids use their canonical
longitude bounds, expanded to the world for spans above 180 degrees, because the current neighbour
API shares coordinates between indexing and distance evaluation. Dateline datasets may therefore
have higher search costs than a narrow planar/local-frame dataset. Locations
with at least `neighborRank + 1` rows publish their known zero distances without searching those
queries; they remain targets for every other query.

The caller owns `output.summary`, a packed `uint32` view of
`GPU_ANALYSIS_SPACING_SUMMARY_LENGTH` (544) words. Read those 2,176 bytes once after execution
and call `decodeGPUAnalysisSpacingStatistics` to obtain `GPUAnalysisSpacingStatistics`. There
is no readback inside the contributor. Neighbour CSR scratch exhaustion sets `candidateOverflow`
(word 27), independently of final `overflow` (word 2, zero for the fixed-size summary). Canonical
ports at `contributor.output.status` alias these summary words and `invalidCount`; they require no
additional readback. Either overflow refuses decoding; all-masked or invalid inputs
decode to `undefined`. Inputs are bounded at one million rows; rank is 1–64, and neighbour
scratch is bounded by the device storage limit. Only the k-distance display curve is reduced
(to 512 quantiles); searches use every valid row. Destroy the compiled graph to release scratch.

Position, mask and identity contents can change per frame; lengths, rank, grid shape,
capacities and context require rebuilding. F32 input quantization bounds geographic accuracy;
planar centre distances use coordinates relative to the extent minimum, and decoding adds
the origin to the relative mean in host f64 to preserve translated fractional centres.
Use a declared local planar frame for fine-scale work. Near cancellation of the geographic
mean uses a 1e-6 vector-norm threshold (the CPU reference uses 1e-9 with f64). Covering arcs
use the largest sorted longitude gap, with ties decided in ascending order; gaps within f32
rounding may tie. Nonfinite numerical summaries refuse decoding. Headless numerical, resource
and performance qualification remains required.

#### Parameters

`positions` is a packed `float32x2` view. Optional `mask` and stable `identities` have
one `uint32` per row. Declare `spatialContext` at construction; `neighborRank`
defaults to 3 (range 1–64), `gridSize` to `[128,128]` and `neighborCapacity` to
`positions.length * neighborRank`. `output.summary` has exactly 544 `uint32` words.
The composition packs its internal search bounds itself; there is no caller parameter packer.

#### How it computes

{/* lens:GPUAnalysisSpacing */}

::lens{scenario=gpu-spatial-analysis.analysis-spacing rung=result caption="Five planar locations, including a duplicate, have a maximum nearest-neighbor distance of 3 units."}
::lens{scenario=gpu-spatial-analysis.analysis-spacing rung=formula caption="The connectivity floor is the maximum nearest-other-row distance across the full population."}
::lens{scenario=gpu-spatial-analysis.analysis-spacing rung=graph caption="Trace indexed searches, duplicate handling and reductions into one bounded summary."}

`@luma-contributors/analysis/gpu-spatial-analysis` exports a host-neutral vocabulary for getting from
"this is my data" to "this is a suitable analysis, with sensible settings, read correctly". It is plain
TypeScript: types, tables and pure functions on the CPU. The optional `GPUAnalysisSpacing`
contributor described above computes resident statistics for this vocabulary. The
Kepler package builds its dataset profile, operation descriptors, suggestions and result reports on it
(see `modules/kepler/README.md`); a website or another host can use it the same way.

The design, the research behind it and its known limits are in
`dev-docs/research/analysis-workflow-ux-2026-11-14/README.md`.

## Questions

`GPU_ANALYSIS_QUESTIONS` lists twelve questions an analysis answers, from `where-concentrated` and
`where-significant` to `what-is-at-each-place` and `prepare-data`. `GPU_ANALYSIS_QUESTION_RECORDS`
gives each a label as a user would ask it, a summary and examples.

- `getGPUAnalysisOperatorQuestions(operator)` returns one to three questions for an operator of the
  [operator catalog](/docs/api/analysis-catalog), primary first. It resolves by an explicit override
  table, then name rules, then the capability family. An operator whose capability stage is `bridge`,
  `construct`, `transform` or `index` resolves to `prepare-data` unless a rule exempts it, so an
  operator added later gets an answer without this file changing.
- `queryGPUAnalysisOperatorsByQuestion(question)` and `getGPUAnalysisQuestionCoverage()` list and
  count operators per question.

## Data kinds

Three vocabularies describe what a dataset is: the ontology's data kinds, the website's data types and
the Kepler catalog's geometry kinds. `data-kind-vocabulary` translates between them
(`getGPUAnalysisWebsiteDataTypes`, `getGPUAnalysisDataKindsForWebsiteDataType`,
`getGPUAnalysisDataKindForCatalogGeometry`, `getGPUAnalysisCatalogGeometryForDataKind`) and
`GPU_ANALYSIS_DATA_KIND_RECORDS` says which kinds a user can load.

About 150 operator records have no kinds in the catalog. `GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS`
supplies them, each read from the operator's ports, and `resolveGPUAnalysisOperatorKinds(operator)`
returns the catalog's kinds, the supplement's, or `unknown`.

`findGPUAnalysisOperatorPaths(fromKind, toKind, maximumSteps)` searches chains of at most three
operators and returns `{paths, truncated, totalFound}`. A path is a discovery hint: kinds do not carry
required companions, layouts or capacities, and `attributes` is a hub through which many kind-valid
chains mean nothing.

Each path step is a `{package, exportName}` identity accepted by `getGPUAnalysisOperator`.
`getGPUAnalysisNextOperators(kind)` returns the same qualified identities, sorted by export name
and entry point. Both visibility recipes remain distinct in these results. Use `step.exportName`
for a label and retain `step.package` when resolving it.

## What a column measures

`guessGPUAnalysisMeasurementLevel(statistics)` guesses whether a column is a `count`, an `amount`, a
`rate`, `ordinal`, `nominal`, `cyclic` or an `identifier`, from its name, type, range and
cardinality, with a reason and a confidence. Values cannot prove a count from a rate, so a guess is
shown to the user and can be replaced by a declaration; a low-confidence guess never fills a role as
a preferred column.

- `matchGPUAnalysisFieldRole(role, columns)` and `assignGPUAnalysisFieldRoles(roles, columns)` offer
  and assign columns to the roles an operation declares (`GPUAnalysisFieldRoleSpec`), in the order of
  the role's accepted types, with a warning when the measurement level is not one the method is meant
  for.
- `suggestGPUAnalysisRatePairs(columns)` ranks numerator and denominator pairs (cases over population).
- `guessGPUAnalysisCoordinateModel(extent)` and `guessGPUAnalysisRowSemantics(profile)` guess whether
  coordinates are geographic and whether rows are events, objects or samples of a surface. Row
  semantics are always a low-confidence guess: a question to ask, not a fact.

## Requirements

An operation declares preconditions as `GPUAnalysisRequirementSpec` (`GPU_ANALYSIS_REQUIREMENT_IDS`).
`evaluateGPUAnalysisRequirements(requirements, profile, boundColumns)` settles each as `passed`,
`failed` or `unknown`, with what the answer rests on. Rules that keep it honest:

- A low-confidence guess never decides a requirement: it gives `unknown`.
- From sampled statistics a universal property (all positive, none missing) can be refuted but not
  established, and an existential one (varies) can be established but not refuted.
- `connected-weights` and `disjoint-coverage` cannot be read from a profile and are always `unknown`.

This is an early warning for a host. It does not replace a contributor's own construction-time checks.

## Defaults derived from the data

`computeGPUAnalysisSpacingStatistics(coordinates, options)` returns how locations are spread: extent
(the smallest covering longitude arc for geographic data, west greater than east across the
antimeridian), mean centre, standard distance, nearest-neighbour quartiles and upper fence, the
k-distance curve and duplicate locations. Distances are metres for geographic coordinates. It is exact
up to a sample limit and uses a hashed sample above it, reporting `sampled`.

`resolveGPUAnalysisDefault(rule, profile, factor, options)` derives one value with an explanation:

| Rule | Value |
| --- | --- |
| `maximum-nearest-neighbor-distance`, `outlier-resistant-nearest-neighbor-distance`, `median-nearest-neighbor-distance` | A neighbourhood reach from nearest-neighbour distances; the median uses distinct locations when repeated locations make it 0 |
| `k-distance-knee` | A clustering radius from the knee of the k-distance curve; no value when the curve has no clear knee |
| `extent-shorter-side-over-250`, `extent-shorter-side-over-250-in-source-units`, `mean-point-spacing` | Cell sizes from the extent or from point density |
| `extent-diagonal`, `extent-west`, `extent-south`, `extent-east`, `extent-north`, `standard-distance` | Sizes and bounds of the data |
| `normal-reference-bandwidth-scott`, `normal-reference-bandwidth-silverman` | The bandwidth `GPUKernelDensity` derives for the same points |
| `column-variance`, `column-standard-deviation`, `column-range`, `column-sum` | Statistics of one column |

A rule returns no value when the profile lacks what it needs; the caller keeps its constant and says so.

## Reading a result

- `getGPUAnalysisResultClasses(scheme, encoding)` gives the legend classes of the fixed class schemes
  (`GPU_ANALYSIS_CLASS_SCHEMES`: Gi* confidence bins, local Moran clusters, local Geary classes, cluster
  membership, flags). Producers encode the same classes differently (`GPU_ANALYSIS_CLASS_ENCODINGS`);
  `validateGPUAnalysisResultEncoding` gives the share of values a scheme recognises.
  For `validity-flag` with `contributor-integer`, zero is invalid and every other uint32 value is
  usable: `[0, 1, 2, 0xffffffff]` tallies three usable rows and one invalid row, with encoding share
  1. Negative numbers, fractions, out-of-range integers and strings remain unrecognized;
  significance flags retain their exact code meanings.
  `normalizeGPUAnalysisClassValue(scheme, encoding, value)` supplies the same conversion to
  publication encoders. Kepler uses it for row labels as well as legend tallies, so all three
  nonzero flags in this example publish as `Valid`; significance codes remain distinct.
- `summarizeGPUAnalysisResult(template, facts)` writes one sentence from typed facts for each of
  `GPU_ANALYSIS_SUMMARY_TEMPLATES`.
- `deriveGPUAnalysisCautions(facts)` and `getGPUAnalysisCautionText(caution, facts)` give the cautions
  that apply to a run: pseudo p-values, uncorrected multiple testing, fewer than thirty features,
  counts where rates are meant, rows without neighbours, small denominators, poor model fit and others.
- `getGPUAnalysisSignificanceAttainability` says whether any test can reach the chosen level with the
  given correction, test count and permutation count, so that "none significant" is not reported as a
  finding when it could not have been otherwise.
- `formatGPUAnalysisMeasuredValue` and `formatGPUAnalysisDistinctValues` print numbers so that
  distinct values stay distinct.
