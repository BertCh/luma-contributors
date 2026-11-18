# Forest inference

Import from `@luma-contributors/analysis/dataframe-analysis`.

## GPU inference

### `GPUForestInference`

Applies a caller-supplied random forest or gradient-boosted model to GPU-resident
rows or a stack of co-registered raster bands. It returns source-aligned
predictions, validity, optional scores and optional original-model leaf indices.
The props type requires exactly one input source: `features` (with optional `featureLayout`)
or `rasterStack`. JavaScript callers receive the same construction-time either-or check;
`featureLayout` is rejected with raster inputs because only a feature matrix has that layout choice.
There is no training, readback, bundled model or dataset. A CPU inference twin is
not provided: inference is row-parallel and intended for large GPU-resident inputs.

Upload the four arrays returned by `readScikitLearnForestModel`,
`readXGBoostForestModel` or `readLightGBMForestModel` to packed graph views:

| Prop | Format | Length |
| --- | --- | --- |
| `nodes` | uint32 | `nodeCount * 6` |
| `trees` | uint32 | `treeCount * 4` |
| `leafValues` | float32 | `nodeCount * outputCount` |
| `parameters` | float32 | `outputCount + 1` |
| `features` | float32 | `rowCount * featureCount` |
| `mask` (optional) | uint32 | `rowCount` |
| `output.predictions` | float32 | `rowCount` |
| `output.validity` | uint32 | `rowCount` |
| `output.scores` (optional) | float32 | `rowCount * outputCount` |
| `output.leafIndices` (optional) | uint32 | `rowCount * treeCount` |

Pass the returned model as `model` to supply its compile-time shape. Model views
(`nodes`, `trees`, `leafValues`, `parameters`) are single packed buffers. Row views
(`features`, `mask`, predictions, validity, scores and leaf indices) may be aligned
`GraphVectorView` chunks at identical source-row boundaries. Each feature chunk
contains whole rows; feature-major layout is local to each batch. Scores and leaf
chunks contain the batch row count times `outputCount` and `treeCount` respectively.
Views may have nonzero byte offsets. Output buffers must be distinct from all input buffers,
even when their logical ranges would be disjoint. Output views may share a physical buffer when
their logical output ranges do not overlap. Views bound together in a dispatch must also have
nonoverlapping aligned storage-binding ranges. For example, with 256-byte storage-binding alignment,
one-row predictions at byte 0 and validity at byte 256 of the same buffer have separate aligned
ranges; validity at byte 4 shares the aligned binding range and is rejected. Use the device's storage-binding alignment
when choosing offsets. `features` is row-major unless
`featureLayout: 'feature-major'` is selected. Changing buffer contents, base scores,
sigmoid, features or mask replays the compiled graph without recompilation.
Changing shape, task, link, layouts or optional views requires a new graph.

For classification, predictions are indices into the imported model's CPU
`classes` array; scores are row-major class probabilities. Ties choose the first
class. For regression, predictions and the single score are the fitted value.
Leaf indices are row-major in exported tree order, including class-specific
boosting trees. sklearn and XGBoost retain tree-local node ids; LightGBM retains
`leaf_index`. Each invalid row receives NaN predictions/scores, validity zero and
`0xffffffff` in **every** leaf slot. A zero mask excludes padding rows. Empty row
input is supported.

NaN follows the imported model's missing branch. sklearn gradient boosting has
no supported missing branch, so a visited NaN invalidates the row. A visited
infinity, failed traversal, nonfinite score or exhausted traversal depth also
invalidates the row. Unvisited features do not affect the result. Model writers
must maintain valid topology and finite leaves/weights; GPU bounds checks prevent
unbounded traversal but do not certify model semantics.

For rasters, supply `rasterStack: {bands, grid}` instead of `features`. These are
published `GPURasterBand<'float32'>` and `GPURasterMetadata` objects. All bands
must be calibrated buffer bands in feature order, with `scale` 1, `offset` 0 and
no `noDataValue` metadata. Prepare raw or texture bands with upstream raster
operations first. Each band's validity zero becomes NaN and follows the model's
missing rule; it does not automatically exclude the whole pixel. Use `mask` for
whole-pixel exclusion. `rowCount` must equal `grid.width * grid.height`. The output
is pixel-aligned. The affine is retained by the caller; no coordinates, distances
or areas are computed, so cell size and coordinate system do not affect inference.

#### Limits and execution

| Limit | Maximum | Reason |
| --- | --- | --- |
| Trees | 1,024 | Bounded per-row traversal |
| Sparse nodes | 1,048,576 | 24 MiB node binding |
| Depth (splits) | 64 | Bounded walk, including a depth-zero stump |
| Features | 256 | Bounded feature index |
| Outputs/classes | 32 | Local score array |
| Raster bands | 16 | At most 18 dispatches per batch |
| Total rows | 4,294,967,295 | uint32 source-row bases; per-batch indexing, work and binding limits apply separately |
| Scalar elements per row-view chunk | 4,294,967,295 | uint32 flat addressing; whole rows must fit |
| Worst-case work per batch | 5,000,000,000 | `batchRows * trees * (depth + outputs + 1)` |

Each nonempty batch uses two dispatches; raster stacks add one preparation dispatch per band per batch.
The largest kernel uses eight storage bindings. Scratch uses
`4 * batchRows * (outputs + 2)` bytes, plus `4 * batchRows * features` for raster packing;
scratch is reused across batches at the largest batch size. Optional leaf output uses
`4 * batchRows * trees` bytes per batch. Every actual input/output chunk binding and
scratch allocation is checked against the device storage-binding limit before
nodes are created (128 MiB on a default WebGPU device). Batch rows when needed;
leaf output often sets the practical batch size. `estimatedCommandNodeCount`
provides the validated command budget.

#### Numerics and formulas

Each row walks sparse children from a tree's root to a leaf, accumulating scores
in fixed tree order. Random-forest regression computes the mean tree value;
classification normalizes each leaf's class counts and averages probabilities,
then selects the largest probability. This follows [sklearn's forest
semantics](https://scikit-learn.org/stable/modules/ensemble.html#forest).

Boosting computes `raw[k] = base[k] + sum(weight[t] * leaf[t,k])`.
sklearn uses `learning_rate` as the tree weight. XGBoost and LightGBM saved leaf
values already contain shrinkage; the importer uses weight one. Binary log-loss
uses `p = sigmoid(multiplier * raw[1])`, with scores `[1-p, p]`. The multiplier is
two for sklearn exponential loss, LightGBM's exported `sigmoid` for its binary
objective, and one otherwise. Multiclass boosting uses a stable softmax after
subtracting the largest raw score. See [sklearn's boosting
formulas](https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation)
and [LightGBM's output conversion](https://github.com/microsoft/LightGBM/blob/master/src/objective/binary_objective.hpp).

sklearn and LightGBM numerical splits use `<=`; XGBoost uses `<`.
Double `<=` thresholds are rounded **down** to the next float32 threshold,
preserving branch selection for float32 features, even when nearest rounding
would change the branch. XGBoost's serialized float32 thresholds are used directly.
LightGBM `missing_type: None` converts NaN to zero before comparison; `Zero`
treats NaN and values with absolute magnitude at most `1e-35` as missing, and
`NaN` uses the declared default child for NaN. These rules follow the upstream
[sklearn traversal](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/tree/_tree.pyx),
[XGBoost traversal](https://github.com/dmlc/xgboost/blob/master/src/predictor/predict_fn.h)
and [LightGBM traversal](https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/tree.h).

Features and accumulation are float32, so probabilities and predictions can
differ from CPU float64 sums. Tests use relative/absolute score tolerance `2e-5`
and compare leaf indices exactly. Very close class margins may change the winning
class after float32 rounding; subnormal-sensitive GPU behavior is outside parity.

{/* parameters: GPUForestInference generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}

#### `GPUForestInference` parameters

- Packer: `getGPUForestInferenceParameterValues(settings)`, writes `getGPUForestInferenceParameterLength(...)` `float32` values.

Settings of `getGPUForestInferenceParameterValues`:

| Field | Type | Description |
|---|---|---|
| `baseScores` | `readonly number[]` | Initial raw score per output; zero for a random forest. |
| `sigmoid` (optional) | `number` | Positive logistic link multiplier; LightGBM's sigmoid, otherwise one. |

{/* /parameters */}

## Importing models

### `readScikitLearnForestModel`

sklearn has no native public JSON model format. The checked-in script
`modules/analysis/test/gpu-dataframe/forest-inference/export_sklearn_forest.py`
exports a documented `sklearn-forest` version-1 schema:

```sh
/usr/bin/python3 modules/analysis/test/gpu-dataframe/forest-inference/export_sklearn_forest.py model.joblib model.json
```

Load only trusted joblib input. The browser reads JSON, never pickle. The script
supports single-target `RandomForestRegressor`, `RandomForestClassifier`,
`GradientBoostingRegressor`, and `GradientBoostingClassifier` (log-loss or
exponential). Initial estimators must be `'zero'`, an exact `DummyClassifier`
with `prior`, `most_frequent`, `uniform` or `constant` probability strategy, or an
exact `DummyRegressor` with `mean`, `median`, `quantile` or `constant` strategy.
`stratified` classification samples row-dependent probabilities and is rejected.
Dummy subclasses, custom initial estimators and multi-target models are rejected.
String and numeric class labels remain in `classes` on the CPU.

The schema has `format`, `version`, `estimator`, `n_features_in`, optional
`classes`, and `trees`. Each tree carries parallel `children_left`,
`children_right`, `feature`, `threshold`, flattened-per-node `value` and optional
`missing_go_to_left` arrays. Infinite thresholds are encoded as the JSON strings
`"Infinity"` or `"-Infinity"`; finite thresholds remain numbers. The reader also
accepts numeric infinities in in-memory objects and rejects NaN or null thresholds.
This preserves sklearn's missing-versus-present splits: all finite values compare
`<= Infinity`, while NaN follows `missing_go_to_left`. No threshold is replaced by
null. Leaves have both children `-1`. Boosted exports add
model `base_scores`, `learning_rate`, `sigmoid`, and each tree's `output` channel.
The exporter derives the initial raw score from the fitted estimator, including
its class priors and regression loss. The importer rejects cyclic, shared,
unreachable, out-of-range or over-depth nodes and invalid class/array shapes.

### `readXGBoostForestModel`

Reads `save_model('model.json')` learner/gbtree JSON, including scalar or vector
string `base_score` metadata. It does not read `get_dump()` JSON. Supported
objectives are `reg:squarederror`, `binary:logistic`, `multi:softprob` and
`multi:softmax`; both multiclass forms publish probabilities and class indices,
including one-class models (class index `0`, probability `[1]`, classes `[0]`).
Task and link follow the objective regardless of class count, matching [XGBoost's
classification and transform implementation](https://github.com/dmlc/xgboost/blob/v3.1.0/src/objective/multiclass_obj.cu#L142-L169).
Model shape limits, including `outputCount` in `[1, 32]`, are checked before
base-score expansion or model-dependent allocation. A binary probability base score
is converted to log-odds. Numerical, scalar-leaf
`gbtree` models only: DART, categorical splits, deleted nodes, multi-target/vector
leaves and other objectives are rejected. See the official [JSON model
parser](https://xgboost.readthedocs.io/en/stable/python/examples/model_parser.html).

### `readLightGBMForestModel`

Reads `Booster.dump_model()` JSON with numerical `<=` splits and constant leaves.
Supports identity-link regression objectives, binary and multiclass; preserves
the exported sigmoid and missing-value policy. Saved leaves include the initial
score and shrinkage, so neither is added again. Categorical splits, linear leaves,
`average_output` random-forest mode, multiclass-OVA and other links are rejected.

## Independent oracles

Regenerate the small synthetic sklearn fixtures with:

```sh
/usr/bin/python3 modules/analysis/test/gpu-dataframe/forest-inference/export_sklearn_forest.py --fixtures
```

The fixture records sklearn predictions, probabilities and leaf indices for random
forest regression/classification and gradient boosting regression, binary,
multiclass and exponential loss, including unbalanced trees, NaNs for forests, and
infinite-threshold missing-only splits for both random-forest types.
Headless tests compare GPU output directly to these stored results. XGBoost and
LightGBM readers use hand-written native JSON and hand-derived outputs; those
libraries are not required. Additional cases cover thresholds, ties, masked replay,
all leaf slots of invalid rows, band validity, empty input, offsets and sentinels.


Run the exporter initializer and strict-JSON/CLI regression tests with:

```sh
/usr/bin/python3 -B modules/analysis/test/gpu-dataframe/forest-inference/test_export_sklearn_forest.py
```
