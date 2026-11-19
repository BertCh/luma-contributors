var e={slug:`api/gpu-spatial-analysis/analysis-experience`,title:`Analysis experience vocabulary`,html:`<h1 id="analysis-experience-vocabulary">Analysis experience vocabulary</h1>
<h3 id="gpuanalysiscontiguitystatistics"><code>GPUAnalysisContiguityStatistics</code></h3>
<p><code>GPUAnalysisContiguityStatistics</code> composes contiguity weights, a weights summary and a fixed
median into a full-population polygon profile. The <a href="/luma-contributors/docs/api/gpu-spatial-analysis/gpu-analysis-contiguity-statistics">contiguity statistics guide</a>
defines the topology, caller-owned layouts, capacity recovery, validity and named limits.
Check both candidate and final overflow before presenting any profile claims.
The composition fixes the median quantile to <code>[0.5]</code>; callers cannot supply quantile settings.</p>
<h4 id="how-it-computes">How it computes</h4>
<p>{/* lens:GPUAnalysisContiguityStatistics */}</p>
<p>::lens{scenario=gpu-spatial-analysis.analysis-contiguity-statistics rung=result caption="Two adjacent squares and one island have cardinalities [1,1,0] and median 1."}
::lens{scenario=gpu-spatial-analysis.analysis-contiguity-statistics rung=formula caption="Count neighbors for every feature, including islands, then take the linear median."}
::lens{scenario=gpu-spatial-analysis.analysis-contiguity-statistics rung=graph caption="Trace contiguity, summary, quantile and overflow-validity stages in one graph."}</p>
<h3 id="gpuanalysisspacing"><code>GPUAnalysisSpacing</code></h3>
<p><code>GPUAnalysisSpacing</code> computes full-population point spacing from resident <code>float32x2</code> positions,
an optional <code>uint32</code> mask and stable identities, and a required planar/native or
longitude-latitude/great-circle spatial context. Geographic distances use the mean Earth sphere
in metres. It composes published graph sorting, scanning, reductions and <code>GPUNeighborSearch</code>
(which builds the published <code>GPUGridIndex</code>). A second indexed search over distinct locations
preserves <code>nearestNeighbor.nonZeroMedian</code> when locations repeat.
Longitude sorting preserves signed endpoints rather than adding 360 to small negative coordinates.
Preparation and longitude keys share neighbor search's integer-significand reduction, preserving
the exact remainder of large stored f32 longitudes (for example, stored <code>1e20</code> equals <code>-88°</code>).
Neighbour distances retain canonical f32 coordinates. Geographic grids use their canonical
longitude bounds, expanded to the world for spans above 180 degrees, because the current neighbour
API shares coordinates between indexing and distance evaluation. Dateline datasets may therefore
have higher search costs than a narrow planar/local-frame dataset. Locations
with at least <code>neighborRank + 1</code> rows publish their known zero distances without searching those
queries; they remain targets for every other query.</p>
<p>The caller owns <code>output.summary</code>, a packed <code>uint32</code> view of
<code>GPU_ANALYSIS_SPACING_SUMMARY_LENGTH</code> (544) words. Read those 2,176 bytes once after execution
and call <code>decodeGPUAnalysisSpacingStatistics</code> to obtain <code>GPUAnalysisSpacingStatistics</code>. There
is no readback inside the contributor. Neighbour CSR scratch exhaustion sets <code>candidateOverflow</code>
(word 27), independently of final <code>overflow</code> (word 2, zero for the fixed-size summary). Canonical
ports at <code>contributor.output.status</code> alias these summary words and <code>invalidCount</code>; they require no
additional readback. Either overflow refuses decoding; all-masked or invalid inputs
decode to <code>undefined</code>. Inputs are bounded at one million rows; rank is 1–64, and neighbour
scratch is bounded by the device storage limit. Only the k-distance display curve is reduced
(to 512 quantiles); searches use every valid row. Destroy the compiled graph to release scratch.</p>
<p>Position, mask and identity contents can change per frame; lengths, rank, grid shape,
capacities and context require rebuilding. F32 input quantization bounds geographic accuracy;
planar centre distances use coordinates relative to the extent minimum, and decoding adds
the origin to the relative mean in host f64 to preserve translated fractional centres.
Use a declared local planar frame for fine-scale work. Near cancellation of the geographic
mean uses a 1e-6 vector-norm threshold (the CPU reference uses 1e-9 with f64). Covering arcs
use the largest sorted longitude gap, with ties decided in ascending order; gaps within f32
rounding may tie. Nonfinite numerical summaries refuse decoding. Headless numerical, resource
and performance qualification remains required.</p>
<h4 id="parameters">Parameters</h4>
<p><code>positions</code> is a packed <code>float32x2</code> view. Optional <code>mask</code> and stable <code>identities</code> have
one <code>uint32</code> per row. Declare <code>spatialContext</code> at construction; <code>neighborRank</code>
defaults to 3 (range 1–64), <code>gridSize</code> to <code>[128,128]</code> and <code>neighborCapacity</code> to
<code>positions.length * neighborRank</code>. <code>output.summary</code> has exactly 544 <code>uint32</code> words.
The composition packs its internal search bounds itself; there is no caller parameter packer.</p>
<h4 id="how-it-computes-1">How it computes</h4>
<p>{/* lens:GPUAnalysisSpacing */}</p>
<p>::lens{scenario=gpu-spatial-analysis.analysis-spacing rung=result caption="Five planar locations, including a duplicate, have a maximum nearest-neighbor distance of 3 units."}
::lens{scenario=gpu-spatial-analysis.analysis-spacing rung=formula caption="The connectivity floor is the maximum nearest-other-row distance across the full population."}
::lens{scenario=gpu-spatial-analysis.analysis-spacing rung=graph caption="Trace indexed searches, duplicate handling and reductions into one bounded summary."}</p>
<p><code>@luma-contributors/analysis/gpu-spatial-analysis</code> exports a host-neutral vocabulary for getting from
"this is my data" to "this is a suitable analysis, with sensible settings, read correctly". It is plain
TypeScript: types, tables and pure functions on the CPU. The optional <code>GPUAnalysisSpacing</code>
contributor described above computes resident statistics for this vocabulary. The
Kepler package builds its dataset profile, operation descriptors, suggestions and result reports on it
(see <code>modules/kepler/README.md</code>); a website or another host can use it the same way.</p>
<p>The design, the research behind it and its known limits are in
<code>dev-docs/research/analysis-workflow-ux-2026-11-14/README.md</code>.</p>
<h2 id="questions">Questions</h2>
<p><code>GPU_ANALYSIS_QUESTIONS</code> lists twelve questions an analysis answers, from <code>where-concentrated</code> and
<code>where-significant</code> to <code>what-is-at-each-place</code> and <code>prepare-data</code>. <code>GPU_ANALYSIS_QUESTION_RECORDS</code>
gives each a label as a user would ask it, a summary and examples.</p>
<ul>
<li><code>getGPUAnalysisOperatorQuestions(operator)</code> returns one to three questions for an operator of the
<a href="/luma-contributors/docs/api/analysis-catalog">operator catalog</a>, primary first. It resolves by an explicit override
table, then name rules, then the capability family. An operator whose capability stage is <code>bridge</code>,
<code>construct</code>, <code>transform</code> or <code>index</code> resolves to <code>prepare-data</code> unless a rule exempts it, so an
operator added later gets an answer without this file changing.</li>
<li><code>queryGPUAnalysisOperatorsByQuestion(question)</code> and <code>getGPUAnalysisQuestionCoverage()</code> list and
count operators per question.</li>
</ul>
<h2 id="data-kinds">Data kinds</h2>
<p>Three vocabularies describe what a dataset is: the ontology's data kinds, the website's data types and
the Kepler catalog's geometry kinds. <code>data-kind-vocabulary</code> translates between them
(<code>getGPUAnalysisWebsiteDataTypes</code>, <code>getGPUAnalysisDataKindsForWebsiteDataType</code>,
<code>getGPUAnalysisDataKindForCatalogGeometry</code>, <code>getGPUAnalysisCatalogGeometryForDataKind</code>) and
<code>GPU_ANALYSIS_DATA_KIND_RECORDS</code> says which kinds a user can load.</p>
<p>About 150 operator records have no kinds in the catalog. <code>GPU_ANALYSIS_OPERATOR_KIND_SUPPLEMENTS</code>
supplies them, each read from the operator's ports, and <code>resolveGPUAnalysisOperatorKinds(operator)</code>
returns the catalog's kinds, the supplement's, or <code>unknown</code>.</p>
<p><code>findGPUAnalysisOperatorPaths(fromKind, toKind, maximumSteps)</code> searches chains of at most three
operators and returns <code>{paths, truncated, totalFound}</code>. A path is a discovery hint: kinds do not carry
required companions, layouts or capacities, and <code>attributes</code> is a hub through which many kind-valid
chains mean nothing.</p>
<p>Each path step is a <code>{package, exportName}</code> identity accepted by <code>getGPUAnalysisOperator</code>.
<code>getGPUAnalysisNextOperators(kind)</code> returns the same qualified identities, sorted by export name
and entry point. Both visibility recipes remain distinct in these results. Use <code>step.exportName</code>
for a label and retain <code>step.package</code> when resolving it.</p>
<h2 id="what-a-column-measures">What a column measures</h2>
<p><code>guessGPUAnalysisMeasurementLevel(statistics)</code> guesses whether a column is a <code>count</code>, an <code>amount</code>, a
<code>rate</code>, <code>ordinal</code>, <code>nominal</code>, <code>cyclic</code> or an <code>identifier</code>, from its name, type, range and
cardinality, with a reason and a confidence. Values cannot prove a count from a rate, so a guess is
shown to the user and can be replaced by a declaration; a low-confidence guess never fills a role as
a preferred column.</p>
<ul>
<li><code>matchGPUAnalysisFieldRole(role, columns)</code> and <code>assignGPUAnalysisFieldRoles(roles, columns)</code> offer
and assign columns to the roles an operation declares (<code>GPUAnalysisFieldRoleSpec</code>), in the order of
the role's accepted types, with a warning when the measurement level is not one the method is meant
for.</li>
<li><code>suggestGPUAnalysisRatePairs(columns)</code> ranks numerator and denominator pairs (cases over population).</li>
<li><code>guessGPUAnalysisCoordinateModel(extent)</code> and <code>guessGPUAnalysisRowSemantics(profile)</code> guess whether
coordinates are geographic and whether rows are events, objects or samples of a surface. Row
semantics are always a low-confidence guess: a question to ask, not a fact.</li>
</ul>
<h2 id="requirements">Requirements</h2>
<p>An operation declares preconditions as <code>GPUAnalysisRequirementSpec</code> (<code>GPU_ANALYSIS_REQUIREMENT_IDS</code>).
<code>evaluateGPUAnalysisRequirements(requirements, profile, boundColumns)</code> settles each as <code>passed</code>,
<code>failed</code> or <code>unknown</code>, with what the answer rests on. Rules that keep it honest:</p>
<ul>
<li>A low-confidence guess never decides a requirement: it gives <code>unknown</code>.</li>
<li>From sampled statistics a universal property (all positive, none missing) can be refuted but not
established, and an existential one (varies) can be established but not refuted.</li>
<li><code>connected-weights</code> and <code>disjoint-coverage</code> cannot be read from a profile and are always <code>unknown</code>.</li>
</ul>
<p>This is an early warning for a host. It does not replace a contributor's own construction-time checks.</p>
<h2 id="defaults-derived-from-the-data">Defaults derived from the data</h2>
<p><code>computeGPUAnalysisSpacingStatistics(coordinates, options)</code> returns how locations are spread: extent
(the smallest covering longitude arc for geographic data, west greater than east across the
antimeridian), mean centre, standard distance, nearest-neighbour quartiles and upper fence, the
k-distance curve and duplicate locations. Distances are metres for geographic coordinates. It is exact
up to a sample limit and uses a hashed sample above it, reporting <code>sampled</code>.</p>
<p><code>resolveGPUAnalysisDefault(rule, profile, factor, options)</code> derives one value with an explanation:</p>
<table>
<thead>
<tr>
<th>Rule</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>maximum-nearest-neighbor-distance</code>, <code>outlier-resistant-nearest-neighbor-distance</code>, <code>median-nearest-neighbor-distance</code></td>
<td>A neighbourhood reach from nearest-neighbour distances; the median uses distinct locations when repeated locations make it 0</td>
</tr>
<tr>
<td><code>k-distance-knee</code></td>
<td>A clustering radius from the knee of the k-distance curve; no value when the curve has no clear knee</td>
</tr>
<tr>
<td><code>extent-shorter-side-over-250</code>, <code>extent-shorter-side-over-250-in-source-units</code>, <code>mean-point-spacing</code></td>
<td>Cell sizes from the extent or from point density</td>
</tr>
<tr>
<td><code>extent-diagonal</code>, <code>extent-west</code>, <code>extent-south</code>, <code>extent-east</code>, <code>extent-north</code>, <code>standard-distance</code></td>
<td>Sizes and bounds of the data</td>
</tr>
<tr>
<td><code>normal-reference-bandwidth-scott</code>, <code>normal-reference-bandwidth-silverman</code></td>
<td>The bandwidth <code>GPUKernelDensity</code> derives for the same points</td>
</tr>
<tr>
<td><code>column-variance</code>, <code>column-standard-deviation</code>, <code>column-range</code>, <code>column-sum</code></td>
<td>Statistics of one column</td>
</tr>
</tbody>
</table>
<p>A rule returns no value when the profile lacks what it needs; the caller keeps its constant and says so.</p>
<h2 id="reading-a-result">Reading a result</h2>
<ul>
<li><code>getGPUAnalysisResultClasses(scheme, encoding)</code> gives the legend classes of the fixed class schemes
(<code>GPU_ANALYSIS_CLASS_SCHEMES</code>: Gi* confidence bins, local Moran clusters, local Geary classes, cluster
membership, flags). Producers encode the same classes differently (<code>GPU_ANALYSIS_CLASS_ENCODINGS</code>);
<code>validateGPUAnalysisResultEncoding</code> gives the share of values a scheme recognises.
For <code>validity-flag</code> with <code>contributor-integer</code>, zero is invalid and every other uint32 value is
usable: <code>[0, 1, 2, 0xffffffff]</code> tallies three usable rows and one invalid row, with encoding share
<ol>
<li>Negative numbers, fractions, out-of-range integers and strings remain unrecognized;
significance flags retain their exact code meanings.
<code>normalizeGPUAnalysisClassValue(scheme, encoding, value)</code> supplies the same conversion to
publication encoders. Kepler uses it for row labels as well as legend tallies, so all three
nonzero flags in this example publish as <code>Valid</code>; significance codes remain distinct.</li>
</ol>
</li>
<li><code>summarizeGPUAnalysisResult(template, facts)</code> writes one sentence from typed facts for each of
<code>GPU_ANALYSIS_SUMMARY_TEMPLATES</code>.</li>
<li><code>deriveGPUAnalysisCautions(facts)</code> and <code>getGPUAnalysisCautionText(caution, facts)</code> give the cautions
that apply to a run: pseudo p-values, uncorrected multiple testing, fewer than thirty features,
counts where rates are meant, rows without neighbours, small denominators, poor model fit and others.</li>
<li><code>getGPUAnalysisSignificanceAttainability</code> says whether any test can reach the chosen level with the
given correction, test count and permutation count, so that "none significant" is not reported as a
finding when it could not have been otherwise.</li>
<li><code>formatGPUAnalysisMeasuredValue</code> and <code>formatGPUAnalysisDistinctValues</code> print numbers so that
distinct values stay distinct.</li>
</ul>`};export{e as default};