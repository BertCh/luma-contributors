var e={slug:`api/analysis-catalog`,title:`Analysis catalog`,html:`<h1 id="analysis-catalog">Analysis catalog</h1>
<p>Browse the <a href="/luma-contributors/explore/">interactive catalog</a> for contributors across dataframe, spatial analysis,
network, raster and terrain analysis. Contributor cards include examples, parameters and recorded
input/output figures where available. For example, open <a href="/luma-contributors/fn/point-density">point density</a> for its
recorded GPU computation on a small real dataset. Entries without a recording remain discoverable
in the catalog.</p>
<p>The Kepler example’s <a href="/luma-contributors/docs/../examples/deck/kepler-compute/VALIDATION-CATALOG">validation and regression catalog</a> binds the existing GTWR, space-time kriging, Vecchia, applicability, conformal, scoring and FDR contributors with executable forms and diagnostics tables.</p>
<h2 id="the-operator-catalog">The operator catalog</h2>
<p>The interactive catalog links to recorded runs. The operator catalog is a data model that lists
every exported contributor class and every <code>add*Recipe</code> function of all nine entry
points: <code>analysis-core</code>, <code>computation-record</code>, <code>gpu-spatial-analysis</code>, <code>gpu-network</code>,
<code>gpu-terrain</code>, <code>raster-analysis</code>, <code>dataframe-analysis</code>, <code>pointcloud-analysis</code> and
<code>geospatial-analysis</code>. <code>analysis-core</code> includes the four fitting producers
<code>GPUFittingCoordinateEncoder</code>, <code>GPUFittingGradientSums</code>, <code>GPUFittingMinibatchOrder</code> and
<code>GPUAdamStep</code>. A test fails when an exported contributor or recipe has no entry.</p>
<p>Each operator has a capability family named for what it does (<code>network-routing</code>,
<code>terrain-hydrology</code>, <code>raster-focal</code>, <code>raster-time-series</code>, <code>dataframe-aggregation</code>,
<code>pointcloud-ground</code>), its package, input and output data kinds, an optional one-line <code>summary</code>,
<code>dimensionality</code> (<code>'2d+time'</code> for temporal operators, <code>'3d'</code> for point clouds),
and the tests and documentation that cover it.</p>
<p>All 261 original spatial descriptors now have summaries from public TSDoc, with editorial overrides
where the opening documentation is absent or chiefly provenance. Both spatial and all-entry text
queries search these summaries. The static metadata is prepared by
<code>node scripts/generate-spatial-catalog-summaries.mjs</code>; review its printed module before updating
<code>spatial-analysis-operator-summaries.ts</code>. Runtime discovery imports no constructors or source files.
The type remains optional for other catalog extensions; consumers should fall back to <code>exportName</code>
when a record has no summary.</p>
<p>An operator's identity is its full entry-point <code>package</code> plus <code>exportName</code>. The spatial and network
entry points each export <code>addVisibilityGraphRecipe</code>: the spatial CSR recipe builds adjacency and
degree, while the network CSR recipe adds exact or sampled network metrics. Pass both fields to
<code>getGPUAnalysisOperator</code> for either recipe. A bare name still resolves when unique and throws when
ambiguous. Catalog records, evidence, limits and arrays are frozen and their types are readonly;
copy a descriptor before adding application-specific fields.</p>
<p>The canonical data-kind tuple and <code>REUSED_ANALYSIS_OPERATORS</code> are also frozen at declaration;
mutating them cannot change a later guard or discovery call.</p>
<p>Neither visibility recipe produces polygons. The spatial recipe's CSR mode reports <code>count</code>,
<code>requiredCount</code>, <code>overflow</code>, <code>candidateOverflow</code> and <code>invalidCount</code>; network CSR adds its network
metric statuses, including <code>zeroWeightCycle</code>. The dense reference overload retains its packed
status and result layout.</p>
<p>The website inventory retains each overload's props and return fields. Its checker selects the
contract from the example call's inline literal <code>mode</code>; for visibility an omitted mode selects
the reference overload. Required options distinguish overloads when multiple modes allow omission.
Explicit CSR calls are checked against CSR return fields. Overload examples
with an unresolved or unknown mode fail validation. Documentation and same-name lens evidence
use the full entry-point identity. Extraction deduplicates actual re-exports and reports eligible
declarations with missing qualified metadata, including public factories.</p>
<p><code>getGPUAnalysisCapabilityCatalog()</code> derives each family's <code>operators</code> and <code>operatorIdentities</code> from
the finalized records. Use <code>operatorIdentities</code> to retain entry-point information. Historical
<code>GPU_SPATIAL_ANALYSIS_CAPABILITIES</code> membership stays unchanged.</p>
<h3 id="tier-and-limits">Tier and limits</h3>
<p>Every operator has a <code>tier</code>:</p>
<ul>
<li><code>'scale'</code>: the kernels are parallel over the data and sized by device limits. Limits that still
exist (for example a neighbor capacity of 64) are listed in <code>limits</code>.</li>
<li><code>'reference'</code>: a single-invocation, exhaustive or small-dense design with a hard maximum size. Use
these for small problems, oracles and teaching, not for million-row inputs. <code>limits</code> lists each
maximum as <code>{name, maximum, reason, source}</code>, where <code>source</code> is the file that enforces it, or
<code>limitation</code> describes a serial step that has no numeric maximum.</li>
<li><code>'unknown'</code>: the execution tier has not been audited. It does not imply a scale implementation.</li>
</ul>
<p>For unaudited metadata, <code>status: 'unknown'</code> differs from a confirmed empty status list and
<code>readback: 'unknown'</code> differs from confirmed <code>'none'</code>. Queries can select <code>'unknown'</code> for <code>tier</code>,
<code>status</code> or <code>readback</code>; a query for a known status excludes unknown records. Audited exceptions
include K-shortest paths' convergence, iteration and candidate-overflow status and the snapshot
arena's <code>'explicit-adapter'</code> readback.</p>
<p>Audited compact ports advertise <code>candidateOverflow</code> independently of final-capacity <code>overflow</code>.
For publishers that count retained candidates before compaction, <code>requiredCount</code> is a lower bound
after candidate loss. Check the publisher's contract: visibility CSR instead publishes <code>0xffffffff</code>
for an unknown requirement after candidate search exhaustion; never use it as an allocation size.
The <a href="https://github.com/visgl/luma-contributors/blob/master/modules/analysis/test/gpu-spatial-analysis/urban-ecology/visibility-graph-csr.spec.ts">visibility CSR regression</a>,
“CSR failures suppress asymmetric graphs and misleading metrics”, distinguishes four required
directed slots from that unknown sentinel. Polygonize and make-valid publish <code>requiredRingCount</code>,
with no feature <code>requiredCount</code>. Gather publishes clamped <code>rowCount</code>, final-capacity <code>overflow</code>,
optional inherited <code>candidateOverflow</code> and optional <code>matchedCount</code>, with no <code>requiredCount</code>.
Nearest-join <code>neighborRows</code> and gather's <code>neighbors.rows</code> are original feature rows;
<code>neighborIds</code> may be caller identities.</p>
<p>Local Moran, local Geary, hot-spot analysis and global spatial statistics advertise the separate
input-weight <code>overflow</code> and <code>candidateOverflow</code> flags. Local statistics also publish <code>invalidCount</code>.
Point-in-polygon joins expose <code>candidateOverflow</code> separately; their top-level <code>status.overflow</code>
aggregates BVH, candidate and final-match capacity loss, while <code>matches.status.overflow</code> reports
only final-match capacity loss. Optional compact matches also publish <code>matches.status.count</code> and
<code>matches.status.requiredCount</code>.</p>
<p>Nearest-feature weights publish <code>outputWeights.status.overflow</code> and optional
<code>outputWeights.status.candidateOverflow</code>; they do not publish <code>requiredCount</code>. Local variability
retains the input weights' capacity flags in <code>status.stages</code> alongside per-stage <code>invalidCount</code>.
Cell-flow topology and surface preparation publish stage <code>invalidCount</code>, without an <code>overflow</code>
port. Discovery status lists describe these actual output or recipe-stage ports, including optional
ports, rather than inheriting the whole recipe family's status vocabulary.</p>
<p>SKATER, Ward, AZP and Max-p enforce a 32,768 estimated-command-node construction budget. This is
admission for the selected rows and budgets, rather than a universal row cap. SKATER compiles at
most <code>min(maximumRegionCount - 1, rows - 1)</code> cuts. Tile reprojection estimates
<code>3 + 2 * sources.length</code> nodes, admitting at most 16,382 source tiles under the same budget.</p>
<p>Raster limitations state the selected area and grid conditions: terrain derivatives require
orthogonal axes, and ground cell-size modes require axis alignment. Fractional zonal aggregation
supports uniform planar, ground and explicit cell-area units; explicit <code>cellAreas</code> require uniform
mode. Embedding pyramid sums/counts retain original support, including cancelling children;
direction validity is separate from support counts. Evidence may include <code>additionalTests</code>
beside its primary <code>tests</code> path for distinct supported modes.</p>
<p>For example <code>GPUNetworkFacilityOptimization</code> is <code>'reference'</code>: it enumerates 2^facilities subsets, or
facilities^demand assignments for the capacitated p-median, and rejects more than 2^20 candidates.
The tier describes the code as it is now. When a contributor is reworked to remove its limit, its entry
in <code>all-entry-points/operator-annotations.ts</code> changes with it.</p>
<p>Public annotations are applied at final assembly and checked against exactly one public identity.
Internal contributor annotations are kept separately. Limits for <code>GPUNetworkInhomogeneousK</code> apply
in every correction mode: at most 1,024 events, 256 radii, 65,536 nodes, 65,536 edges, 1,048,576
event pairs, and 16,777,216 event-by-node distance rows. Event pairs must also fit the caller's
<code>maximumPairCount</code> budget. Their reasons name the enforcing source constants.</p>
<p><code>GPUParameterBuffer</code> is listed because its ID is stable, but it is a helper, not an operation. It has
<code>role: 'helper'</code> and is left out of <code>queryGPUAnalysisOperators</code> unless <code>includeHelpers</code> is true.</p>
<h3 id="querying-it">Querying it</h3>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUAnalysisCapabilityCatalog,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUAnalysisOperator,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUAnalysisOperatorCatalog,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  queryGPUAnalysisOperators</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis/ontology'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Everything that is limited to small inputs.</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">queryGPUAnalysisOperators</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({tier: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'reference'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// Temporal operators of the raster entry point.</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">queryGPUAnalysisOperators</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  package: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  dimensionality: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'2d+time'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// One operator, with its limits.</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUAnalysisOperator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'GPUNetworkFlow'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)?.limits;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// A name shared by two entry points needs its import subpath.</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">getGPUAnalysisOperator</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  package: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'@luma-contributors/analysis/gpu-network'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  exportName: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'addVisibilityGraphRecipe'</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span></code></pre>
<p><code>queryGPUAnalysisOperators</code> also accepts the filters of <code>queryGPUSpatialAnalysisOperators</code>
(<code>primaryCapability</code>, <code>input</code>, <code>output</code>, <code>coordinateModel</code>, <code>semantics</code>, <code>evidence</code>, <code>search</code>),
plus <code>status</code>, <code>readback</code> and <code>tier</code>. Gaussian random fields are discoverable through both point
and surface inputs and outputs, reflecting their point-sampling and grid modes.</p>
<p><code>GPU_SPATIAL_ANALYSIS_OPERATORS</code>, <code>GPU_SPATIAL_ANALYSIS_CAPABILITIES</code> and the functions that read them
still describe <code>gpu-spatial-analysis</code> (and <code>GPUTerrainPeakProminence</code>) only. They remain historical
entry-point APIs; the website inventory generator uses <code>getGPUAnalysisOperatorCatalog()</code> and
<code>getGPUAnalysisCapabilityCatalog()</code> for all entries. Operators of the other entry points come from
<code>getGPUAnalysisOperatorCatalog()</code>. The original operators also carry <code>tier</code>,
<code>limits</code> and <code>role</code> now.</p>
<p>Operators added to the catalog after the first families do not repeat the family scale figures
(<code>representativeSourceRows</code>, <code>transientMemory</code>, <code>dispatch</code>); their <code>evidence</code> records tests and
documentation, with <code>status: 'partial'</code> and <code>contract: 'unaudited'</code> until a contract is assigned.</p>`};export{e as default};