var e={slug:`api/forest-inference`,title:`Forest inference`,html:`<h1 id="forest-inference">Forest inference</h1>
<p>Import from <code>@luma-contributors/analysis/dataframe-analysis</code>.</p>
<h2 id="gpu-inference">GPU inference</h2>
<h3 id="gpuforestinference"><code>GPUForestInference</code></h3>
<p>Applies a caller-supplied random forest or gradient-boosted model to GPU-resident
rows or a stack of co-registered raster bands. It returns source-aligned
predictions, validity, optional scores and optional original-model leaf indices.
The props type requires exactly one input source: <code>features</code> (with optional <code>featureLayout</code>)
or <code>rasterStack</code>. JavaScript callers receive the same construction-time either-or check;
<code>featureLayout</code> is rejected with raster inputs because only a feature matrix has that layout choice.
There is no training, readback, bundled model or dataset. A CPU inference twin is
not provided: inference is row-parallel and intended for large GPU-resident inputs.</p>
<p>Upload the four arrays returned by <code>readScikitLearnForestModel</code>,
<code>readXGBoostForestModel</code> or <code>readLightGBMForestModel</code> to packed graph views:</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Format</th>
<th>Length</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>nodes</code></td>
<td>uint32</td>
<td><code>nodeCount * 6</code></td>
</tr>
<tr>
<td><code>trees</code></td>
<td>uint32</td>
<td><code>treeCount * 4</code></td>
</tr>
<tr>
<td><code>leafValues</code></td>
<td>float32</td>
<td><code>nodeCount * outputCount</code></td>
</tr>
<tr>
<td><code>parameters</code></td>
<td>float32</td>
<td><code>outputCount + 1</code></td>
</tr>
<tr>
<td><code>features</code></td>
<td>float32</td>
<td><code>rowCount * featureCount</code></td>
</tr>
<tr>
<td><code>mask</code> (optional)</td>
<td>uint32</td>
<td><code>rowCount</code></td>
</tr>
<tr>
<td><code>output.predictions</code></td>
<td>float32</td>
<td><code>rowCount</code></td>
</tr>
<tr>
<td><code>output.validity</code></td>
<td>uint32</td>
<td><code>rowCount</code></td>
</tr>
<tr>
<td><code>output.scores</code> (optional)</td>
<td>float32</td>
<td><code>rowCount * outputCount</code></td>
</tr>
<tr>
<td><code>output.leafIndices</code> (optional)</td>
<td>uint32</td>
<td><code>rowCount * treeCount</code></td>
</tr>
</tbody>
</table>
<p>Pass the returned model as <code>model</code> to supply its compile-time shape. Model views
(<code>nodes</code>, <code>trees</code>, <code>leafValues</code>, <code>parameters</code>) are single packed buffers. Row views
(<code>features</code>, <code>mask</code>, predictions, validity, scores and leaf indices) may be aligned
<code>GraphVectorView</code> chunks at identical source-row boundaries. Each feature chunk
contains whole rows; feature-major layout is local to each batch. Scores and leaf
chunks contain the batch row count times <code>outputCount</code> and <code>treeCount</code> respectively.
Views may have nonzero byte offsets. Output buffers must be distinct from all input buffers,
even when their logical ranges would be disjoint. Output views may share a physical buffer when
their logical output ranges do not overlap. Views bound together in a dispatch must also have
nonoverlapping aligned storage-binding ranges. For example, with 256-byte storage-binding alignment,
one-row predictions at byte 0 and validity at byte 256 of the same buffer have separate aligned
ranges; validity at byte 4 shares the aligned binding range and is rejected. Use the device's storage-binding alignment
when choosing offsets. <code>features</code> is row-major unless
<code>featureLayout: 'feature-major'</code> is selected. Changing buffer contents, base scores,
sigmoid, features or mask replays the compiled graph without recompilation.
Changing shape, task, link, layouts or optional views requires a new graph.</p>
<p>For classification, predictions are indices into the imported model's CPU
<code>classes</code> array; scores are row-major class probabilities. Ties choose the first
class. For regression, predictions and the single score are the fitted value.
Leaf indices are row-major in exported tree order, including class-specific
boosting trees. sklearn and XGBoost retain tree-local node ids; LightGBM retains
<code>leaf_index</code>. Each invalid row receives NaN predictions/scores, validity zero and
<code>0xffffffff</code> in <strong>every</strong> leaf slot. A zero mask excludes padding rows. Empty row
input is supported.</p>
<p>NaN follows the imported model's missing branch. sklearn gradient boosting has
no supported missing branch, so a visited NaN invalidates the row. A visited
infinity, failed traversal, nonfinite score or exhausted traversal depth also
invalidates the row. Unvisited features do not affect the result. Model writers
must maintain valid topology and finite leaves/weights; GPU bounds checks prevent
unbounded traversal but do not certify model semantics.</p>
<p>For rasters, supply <code>rasterStack: {bands, grid}</code> instead of <code>features</code>. These are
published <code>GPURasterBand&#x3C;'float32'></code> and <code>GPURasterMetadata</code> objects. All bands
must be calibrated buffer bands in feature order, with <code>scale</code> 1, <code>offset</code> 0 and
no <code>noDataValue</code> metadata. Prepare raw or texture bands with upstream raster
operations first. Each band's validity zero becomes NaN and follows the model's
missing rule; it does not automatically exclude the whole pixel. Use <code>mask</code> for
whole-pixel exclusion. <code>rowCount</code> must equal <code>grid.width * grid.height</code>. The output
is pixel-aligned. The affine is retained by the caller; no coordinates, distances
or areas are computed, so cell size and coordinate system do not affect inference.</p>
<h4 id="limits-and-execution">Limits and execution</h4>
<table>
<thead>
<tr>
<th>Limit</th>
<th>Maximum</th>
<th>Reason</th>
</tr>
</thead>
<tbody>
<tr>
<td>Trees</td>
<td>1,024</td>
<td>Bounded per-row traversal</td>
</tr>
<tr>
<td>Sparse nodes</td>
<td>1,048,576</td>
<td>24 MiB node binding</td>
</tr>
<tr>
<td>Depth (splits)</td>
<td>64</td>
<td>Bounded walk, including a depth-zero stump</td>
</tr>
<tr>
<td>Features</td>
<td>256</td>
<td>Bounded feature index</td>
</tr>
<tr>
<td>Outputs/classes</td>
<td>32</td>
<td>Local score array</td>
</tr>
<tr>
<td>Raster bands</td>
<td>16</td>
<td>At most 18 dispatches per batch</td>
</tr>
<tr>
<td>Total rows</td>
<td>4,294,967,295</td>
<td>uint32 source-row bases; per-batch indexing, work and binding limits apply separately</td>
</tr>
<tr>
<td>Scalar elements per row-view chunk</td>
<td>4,294,967,295</td>
<td>uint32 flat addressing; whole rows must fit</td>
</tr>
<tr>
<td>Worst-case work per batch</td>
<td>5,000,000,000</td>
<td><code>batchRows * trees * (depth + outputs + 1)</code></td>
</tr>
</tbody>
</table>
<p>Each nonempty batch uses two dispatches; raster stacks add one preparation dispatch per band per batch.
The largest kernel uses eight storage bindings. Scratch uses
<code>4 * batchRows * (outputs + 2)</code> bytes, plus <code>4 * batchRows * features</code> for raster packing;
scratch is reused across batches at the largest batch size. Optional leaf output uses
<code>4 * batchRows * trees</code> bytes per batch. Every actual input/output chunk binding and
scratch allocation is checked against the device storage-binding limit before
nodes are created (128 MiB on a default WebGPU device). Batch rows when needed;
leaf output often sets the practical batch size. <code>estimatedCommandNodeCount</code>
provides the validated command budget.</p>
<h4 id="numerics-and-formulas">Numerics and formulas</h4>
<p>Each row walks sparse children from a tree's root to a leaf, accumulating scores
in fixed tree order. Random-forest regression computes the mean tree value;
classification normalizes each leaf's class counts and averages probabilities,
then selects the largest probability. This follows <a href="https://scikit-learn.org/stable/modules/ensemble.html#forest">sklearn's forest
semantics</a>.</p>
<p>Boosting computes <code>raw[k] = base[k] + sum(weight[t] * leaf[t,k])</code>.
sklearn uses <code>learning_rate</code> as the tree weight. XGBoost and LightGBM saved leaf
values already contain shrinkage; the importer uses weight one. Binary log-loss
uses <code>p = sigmoid(multiplier * raw[1])</code>, with scores <code>[1-p, p]</code>. The multiplier is
two for sklearn exponential loss, LightGBM's exported <code>sigmoid</code> for its binary
objective, and one otherwise. Multiclass boosting uses a stable softmax after
subtracting the largest raw score. See <a href="https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation">sklearn's boosting
formulas</a>
and <a href="https://github.com/microsoft/LightGBM/blob/master/src/objective/binary_objective.hpp">LightGBM's output conversion</a>.</p>
<p>sklearn and LightGBM numerical splits use <code>&#x3C;=</code>; XGBoost uses <code>&#x3C;</code>.
Double <code>&#x3C;=</code> thresholds are rounded <strong>down</strong> to the next float32 threshold,
preserving branch selection for float32 features, even when nearest rounding
would change the branch. XGBoost's serialized float32 thresholds are used directly.
LightGBM <code>missing_type: None</code> converts NaN to zero before comparison; <code>Zero</code>
treats NaN and values with absolute magnitude at most <code>1e-35</code> as missing, and
<code>NaN</code> uses the declared default child for NaN. These rules follow the upstream
<a href="https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/tree/_tree.pyx">sklearn traversal</a>,
<a href="https://github.com/dmlc/xgboost/blob/master/src/predictor/predict_fn.h">XGBoost traversal</a>
and <a href="https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/tree.h">LightGBM traversal</a>.</p>
<p>Features and accumulation are float32, so probabilities and predictions can
differ from CPU float64 sums. Tests use relative/absolute score tolerance <code>2e-5</code>
and compare leaf indices exactly. Very close class margins may change the winning
class after float32 rounding; subnormal-sensitive GPU behavior is outside parity.</p>
<p>{/* parameters: GPUForestInference generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h4 id="gpuforestinference-parameters"><code>GPUForestInference</code> parameters</h4>
<ul>
<li>Packer: <code>getGPUForestInferenceParameterValues(settings)</code>, writes <code>getGPUForestInferenceParameterLength(...)</code> <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUForestInferenceParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>baseScores</code></td>
<td><code>readonly number[]</code></td>
<td>Initial raw score per output; zero for a random forest.</td>
</tr>
<tr>
<td><code>sigmoid</code> (optional)</td>
<td><code>number</code></td>
<td>Positive logistic link multiplier; LightGBM's sigmoid, otherwise one.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="importing-models">Importing models</h2>
<h3 id="readscikitlearnforestmodel"><code>readScikitLearnForestModel</code></h3>
<p>sklearn has no native public JSON model format. The checked-in script
<code>modules/analysis/test/gpu-dataframe/forest-inference/export_sklearn_forest.py</code>
exports a documented <code>sklearn-forest</code> version-1 schema:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">/usr/bin/python3</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> modules/analysis/test/gpu-dataframe/forest-inference/export_sklearn_forest.py</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> model.joblib</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> model.json</span></span></code></pre>
<p>Load only trusted joblib input. The browser reads JSON, never pickle. The script
supports single-target <code>RandomForestRegressor</code>, <code>RandomForestClassifier</code>,
<code>GradientBoostingRegressor</code>, and <code>GradientBoostingClassifier</code> (log-loss or
exponential). Initial estimators must be <code>'zero'</code>, an exact <code>DummyClassifier</code>
with <code>prior</code>, <code>most_frequent</code>, <code>uniform</code> or <code>constant</code> probability strategy, or an
exact <code>DummyRegressor</code> with <code>mean</code>, <code>median</code>, <code>quantile</code> or <code>constant</code> strategy.
<code>stratified</code> classification samples row-dependent probabilities and is rejected.
Dummy subclasses, custom initial estimators and multi-target models are rejected.
String and numeric class labels remain in <code>classes</code> on the CPU.</p>
<p>The schema has <code>format</code>, <code>version</code>, <code>estimator</code>, <code>n_features_in</code>, optional
<code>classes</code>, and <code>trees</code>. Each tree carries parallel <code>children_left</code>,
<code>children_right</code>, <code>feature</code>, <code>threshold</code>, flattened-per-node <code>value</code> and optional
<code>missing_go_to_left</code> arrays. Infinite thresholds are encoded as the JSON strings
<code>"Infinity"</code> or <code>"-Infinity"</code>; finite thresholds remain numbers. The reader also
accepts numeric infinities in in-memory objects and rejects NaN or null thresholds.
This preserves sklearn's missing-versus-present splits: all finite values compare
<code>&#x3C;= Infinity</code>, while NaN follows <code>missing_go_to_left</code>. No threshold is replaced by
null. Leaves have both children <code>-1</code>. Boosted exports add
model <code>base_scores</code>, <code>learning_rate</code>, <code>sigmoid</code>, and each tree's <code>output</code> channel.
The exporter derives the initial raw score from the fitted estimator, including
its class priors and regression loss. The importer rejects cyclic, shared,
unreachable, out-of-range or over-depth nodes and invalid class/array shapes.</p>
<h3 id="readxgboostforestmodel"><code>readXGBoostForestModel</code></h3>
<p>Reads <code>save_model('model.json')</code> learner/gbtree JSON, including scalar or vector
string <code>base_score</code> metadata. It does not read <code>get_dump()</code> JSON. Supported
objectives are <code>reg:squarederror</code>, <code>binary:logistic</code>, <code>multi:softprob</code> and
<code>multi:softmax</code>; both multiclass forms publish probabilities and class indices,
including one-class models (class index <code>0</code>, probability <code>[1]</code>, classes <code>[0]</code>).
Task and link follow the objective regardless of class count, matching <a href="https://github.com/dmlc/xgboost/blob/v3.1.0/src/objective/multiclass_obj.cu#L142-L169">XGBoost's
classification and transform implementation</a>.
Model shape limits, including <code>outputCount</code> in <code>[1, 32]</code>, are checked before
base-score expansion or model-dependent allocation. A binary probability base score
is converted to log-odds. Numerical, scalar-leaf
<code>gbtree</code> models only: DART, categorical splits, deleted nodes, multi-target/vector
leaves and other objectives are rejected. See the official <a href="https://xgboost.readthedocs.io/en/stable/python/examples/model_parser.html">JSON model
parser</a>.</p>
<h3 id="readlightgbmforestmodel"><code>readLightGBMForestModel</code></h3>
<p>Reads <code>Booster.dump_model()</code> JSON with numerical <code>&#x3C;=</code> splits and constant leaves.
Supports identity-link regression objectives, binary and multiclass; preserves
the exported sigmoid and missing-value policy. Saved leaves include the initial
score and shrinkage, so neither is added again. Categorical splits, linear leaves,
<code>average_output</code> random-forest mode, multiclass-OVA and other links are rejected.</p>
<h2 id="independent-oracles">Independent oracles</h2>
<p>Regenerate the small synthetic sklearn fixtures with:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">/usr/bin/python3</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> modules/analysis/test/gpu-dataframe/forest-inference/export_sklearn_forest.py</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> --fixtures</span></span></code></pre>
<p>The fixture records sklearn predictions, probabilities and leaf indices for random
forest regression/classification and gradient boosting regression, binary,
multiclass and exponential loss, including unbalanced trees, NaNs for forests, and
infinite-threshold missing-only splits for both random-forest types.
Headless tests compare GPU output directly to these stored results. XGBoost and
LightGBM readers use hand-written native JSON and hand-derived outputs; those
libraries are not required. Additional cases cover thresholds, ties, masked replay,
all leaf slots of invalid rows, band validity, empty input, offsets and sentinels.</p>
<p>Run the exporter initializer and strict-JSON/CLI regression tests with:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">/usr/bin/python3</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> -B</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> modules/analysis/test/gpu-dataframe/forest-inference/test_export_sklearn_forest.py</span></span></code></pre>`};export{e as default};