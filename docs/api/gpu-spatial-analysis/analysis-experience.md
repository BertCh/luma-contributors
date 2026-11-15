# Analysis experience vocabulary

`@luma-contributors/analysis/gpu-spatial-analysis` exports a host-neutral vocabulary for getting from
"this is my data" to "this is a suitable analysis, with sensible settings, read correctly". It is plain
TypeScript: types, tables and pure functions on the CPU. Nothing here creates a GPU resource. The
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
