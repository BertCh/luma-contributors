# Authoring walkthrough: coregionalization fit

Follow `GPUCoregionalizationFit` from its input contract to the example a reader can inspect. This
is a worked path through an existing contributor, with the files to update when changing it. The
[contributor contract](./contributor-contract.md) supplies the rules; this guide connects them.
Use published luma.gl APIs through the entry points permitted by [AGENTS.md](../AGENTS.md).
Keep analysis behavior in this repository and propose generic graph changes upstream.

The fit estimates positive semidefinite (PSD) structure matrices for a linear model of
coregionalization (LMC). It accepts empirical direct and cross semivariograms on common lag bins,
fits coefficients for supplied covariance models and ranges, and returns matrices and loading
factors usable by local co-kriging. It does not search for ranges. Its small, serial dense solves
make it a reference-tier contributor with a preferred CPU twin when the bins are already on the
CPU. Start with its [class documentation](../modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts#L58)
and [user-facing contract](../docs/api/dataframe-analysis/coregionalization-fit.md).

## 1. Define props and caller-owned output before the kernel

Read the [props, output and status types](../modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts#L30)
and [shape options](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-parameters.ts#L40).
The caller owns the graph views and buffers. The contributor declares work against them; submission,
readback and buffer destruction belong to the caller.

For two variables and one structure, the layout is concrete:

| View | Format | Length and meaning |
| --- | --- | --- |
| `distances` | `float32` | One positive lag per bin |
| `pairCounts` | `uint32` | One shared pair weight per bin; zero excludes the complete bin |
| `semivariances` | `float32` | Three entries per bin: `(0,0), (0,1), (1,1)`; cross values may be negative |
| `parameters` | `float32` | At least five elements: tolerance, model code, nugget, sill, range |
| `output.matrices` | `float32` | Four result elements: the structure's row-major square matrix |
| `output.loadings` | `float32` | Four result elements: a square factor `L` satisfying `B = L Lᵀ` |
| `output.diagnostics` | `float32x4` | One row: weighted objective, relative change, usable bins, total pair weight |
| Each `output.status` view | `uint32` | One scalar result; the five fields are described below |

In general, each bin contains `variableCount * (variableCount + 1) / 2` triangle entries, and
each matrix/factor result contains `structureCount * variableCount²` elements. Results are
structure-major, then row-major. Outputs use distinct buffers disjoint from inputs. Extra output
capacity is permitted and stays untouched beyond the result prefix.

Declare which changes rebuild the graph. Here `variableCount`, `structureCount`,
`maximumIterations` and view shapes are compile-time; all view contents are per-frame.
The [parameter packer](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-parameters.ts#L76)
encodes models and tolerance, validating their float32 representation. Its default tolerance is
`1e-5`. Changing those values in the parameter buffer must affect replay without recompilation.
Do not turn a per-frame value into a shader literal when extending the fit.

Coregionalization reads lags rather than coordinates: lags and ranges must share units. For a
coordinate-reading contributor, resolve its spatial context at construction. For time input, use
the [temporal-context resolver guide](../docs/api/gpu-spatial-analysis/temporal-context.md#resolving-and-checking)
to declare encoding, unit and epoch.

## 2. Choose an input that distinguishes right from wrong

Before changing a result, name a plausible wrong implementation and choose the smallest input
whose caller-visible answer differs. Write that test, run it against the old code, record the
failure, then fix the implementation and rerun it. GPU/CPU agreement alone cannot detect an
algorithmic mistake they share.

The existing [one-bin input](../modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.node.spec.ts#L78)
is a useful starting point: lag `1`, pair count `10`, upper triangle `[2, -1, 3]`, and a unit-sill
spherical structure of range `1`. Its shape at this lag is `1`, so the fitted matrix is
`[[2, -1], [-1, 3]]`. This matrix is positive definite (positive leading entry and determinant
`5`), with objective zero. A wrong implementation that drops signed cross values cannot produce
the expected off-diagonal `-1`.

Also choose a case that needs PSD projection: the [independent fixture generator](../modules/analysis/test/gpu-dataframe/pair-statistics/fixtures/generate-coregionalization.py#L91)
uses triangle `[1, 2, 1]` at spherical shape `1`. The matrix has eigenvalues `3` and `-1`;
clipping the negative one gives `[[1.5, 1.5], [1.5, 1.5]]`, with Frobenius squared residual `1`.
Returning the empirical matrix unchanged would give residual zero and fail this oracle.

For an iteration change, include an explicit zero budget. With the usable one-bin input above,
the expected matrices and factors are zero, `validity = 1`, `iterationCount = 0`,
`converged = 0`, `iterationLimitReached = 1`, and the objective is
`2² + 2 * (-1)² + 3² = 15`. Both forms must honor the caller's budget.
The committed fixture set includes this `zero-budget` case.

## 3. Validate construction and preflight device limits

Trace the [constructor](../modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts#L81),
starting with the [shared named bounds](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-parameters.ts#L29):

| Subject and execution route | Accepted bound | Reason |
| --- | --- | --- |
| Variables in either form | 2–4 | Supplied local co-kriging loading layout |
| Structures in either form | 1–4 | Supplied local co-kriging structure layout |
| GPU input lag matrices | 0–256 | Reference-tier serial sufficient-statistics pass |
| GPU `maximumIterations` | 0–1024, default 128 | Two command nodes per sweep |

Use `validateCapacity` and `validateRequirement` for ID-prefixed errors that name the prop and
bound. Check packed formats, related input lengths, output lengths, and output aliasing once.
Do not validate input rows on the CPU inside the contributor; unusable per-frame data gets a GPU
result and status. Snapshot compile-time options and port maps before validation while retaining
view/buffer identities for per-frame contents.

The fit's constructor snapshots compile-time options and input/output/status port maps before
validation. Constructing with `variableCount = 2`, changing the original props to `5`, then
declaring nodes still emits `VARIABLE_COUNT = 2u` in every kernel. Its existing
[CORE-08 mutation regressions](../modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.node.spec.ts#L29)
check the original shape, sweep budget, getter-backed ports and nested status view identities.
For another authoring example, use the
[working nested-port snapshot](../modules/analysis/src/analysis-core/ring-assembly/gpu-segment-ring-assembly.ts#L510)
and its [mutation regression](../modules/analysis/test/gpu-spatial-analysis/ring-assembly/gpu-segment-ring-assembly.node.spec.ts#L54)
as the authoring pattern. The
[walkthrough review spec](../modules/analysis/test/analysis-core/authoring-walkthrough.node.spec.ts)
performs the same two-to-five mutation and requires dimension `2u` in every emitted kernel;
it checks graph declaration without claiming GPU execution results.

The constructor exposes `estimatedCommandNodeCount = 2 * maximumIterations + 3` and calls
`validateCommandNodeBudget`. A changed graph must update the estimate and its test together.
The [node builder's storage preflight](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-fit-nodes.ts#L26)
checks the fit-state byte length with `validateGPUStorageBindingByteLength` before allocating
transient state or declaring nodes. Checks needing the device go here; shape errors belong in
construction. Graph membership is checked when [declaring command nodes](../modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit.ts#L194).

## 4. Build bounded kernels and publish status on every run

The implementation is split between the producer and an internal
[node builder](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-fit-nodes.ts#L21):

1. Initialize transient state and form weighted sufficient statistics from usable bins.
2. Emit a gated sweep for each budgeted iteration: update each structure conditional on the
   others, project to PSD, reconstruct its matrix, and measure coefficient change.
3. Emit a gate update after each sweep to disarm later work after convergence or invalidity.
4. Publish the fit and diagnostics, then publish every named status scalar.

Initialization is [bounded by the reference-tier caps](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-fit-nodes.ts#L140).
The [sweep](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-fit-nodes.ts#L195)
uses the shared symmetric eigensolver and `createWGSLDispatchGate`. This bounded dense fit is
not a template for serial scans over arbitrary datasets: use published parallel primitives for
data-sized reductions and scans. Keep graph resources, binding access and transient ownership
explicit through the shared kernel builder.

The [publication kernels](../modules/analysis/src/dataframe-analysis/pair-statistics/coregionalization-fit-nodes.ts#L265)
give a caller enough information to decide whether to use the fit:

| Status field | Meaning |
| --- | --- |
| `validity` | Nonzero means a usable fit; zero means invalid data/parameters or numerical failure |
| `iterationCount` | Sweeps actually performed |
| `converged` | Valid fit met the relative coefficient-change criterion |
| `iterationLimitReached` | Valid fit exhausted the budget without convergence |
| `invalidCount` | Complete lag matrices excluded for bad lag, zero weight or any nonfinite triangle entry |

An invalid fit publishes NaN coefficients, objective and relative change, with convergence and
limit flags zero. Usable-bin count and weight remain diagnostic outputs. A valid budget exit
publishes its current PSD fit, even if unconverged. Consumers must inspect that condition and
propagate it, suppress dependent results, or refuse/fall back. They must also decide whether the
reported excluded bins are acceptable; validity does not mean every supplied bin participated.
Rewrite statuses on replay so an earlier successful result cannot survive a later invalid input.

## 5. Connect the operation form and CPU twin

The [GPU operation](../modules/analysis/src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit-operation.ts#L9)
replaces views with graph-value names in `CoregionalizationFitOperationProps`. It declares the
same formats and lengths, then constructs the contributor from resolved views. It adds no
statistical behavior. Compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`.

The [pure CPU function](../modules/analysis/src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts#L41)
takes typed arrays; the [CPU operation](../modules/analysis/src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts#L279)
adapts the same graph names to that function and its results. This lets a caller substitute forms
without changing its ports. The pure function checks array lengths and computes in float64,
rounding to float32 at output. It drops GPU-only bin and sweep caps while retaining layout bounds
and honoring the supplied iteration budget exactly.

Document differences beside the pure function under “What differs from the GPU contributor”.
Here float32 accumulation can change convergence counts and eigenvector signs/bases. Compare
`B` and reconstructed `L Lᵀ`, rather than requiring equal factor entries. The
[host convenience function](../modules/analysis/src/dataframe-analysis/pair-statistics/cpu-coregionalization-fit-operation.ts#L243)
packages fitted loadings for local kriging, but still requires a caller decision about validity
and convergence. For graph execution and cleanup, follow the
[operation-form walkthrough](../docs/api/operation-forms-and-cpu-twins.md) and destroy an operation
that owns resources after destroying its compiled graph.

## 6. Export the API and describe it in the catalog

The [pair-statistics barrel](../modules/analysis/src/dataframe-analysis/pair-statistics/index.ts#L182)
exports the producer, its props/output/status types, both operations, the pure function, the
convenience function, and public parameter helpers. The
[family barrel](../modules/analysis/src/dataframe-analysis/index.ts#L12) exports that barrel via
`./pair-statistics/index`. Keep internal node builders and validators internal. Consumers import
from `@luma-contributors/analysis/dataframe-analysis`; do not re-export luma.gl APIs.

The catalog is assembled from several records. Follow all of these connections when changing
the contributor:

- [Entry details](../modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/entry-point-operator-details.ts#L1607):
  public entry point, summary, spec and documentation path.
- [Annotations](../modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/operator-annotations.ts#L116):
  reference tier, recommendation to use the CPU twin, and each named limit with its reason and source.
- [Capabilities](../modules/analysis/src/gpu-spatial-analysis/ontology/all-entry-points/entry-point-capabilities.ts#L713)
  and [input/output kinds](../modules/analysis/src/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-records.ts#L661):
  discoverability in the family and the data shapes a user can connect.

Check the catalog's documentation target as well as its summary. The entry points to the
[canonical detailed guide](../docs/api/dataframe-analysis/coregionalization-fit.md), which owns the
numerical contract for the contributor and both operation forms. The family API section links
there and keeps its generated Parameters block. During a coordinated pass where ontology files
belong to another stream, request the exact record change from its owner.

## 7. Register a maximal kernel audit and refusal companions

The [audit props](../modules/analysis/test/analysis-core/kernel-audit/cases-coregionalization-fit.ts#L11)
wire four variables, four structures, 256 bins, 1024 sweeps and every status view. The
[above-limit cases](../modules/analysis/test/analysis-core/kernel-audit/cases-coregionalization-fit.ts#L47)
go one beyond each cap. Their [rejection records](../modules/analysis/test/analysis-core/kernel-audit/rejections-coregionalization-fit.ts#L7)
specify the expected ID-prefixed errors and distinguish binding limits from dispatch limits.

Cases are connected through the [central case map](../modules/analysis/test/analysis-core/kernel-audit/kernel-audit-cases.ts#L70),
the [rejection registry](../modules/analysis/test/analysis-core/kernel-audit/kernel-audit-rejections.ts#L5)
and the [scoped node spec](../modules/analysis/test/analysis-core/kernel-audit-coregionalization-fit.node.spec.ts#L5).
Add both accepted and rejected shapes when changing a bound. This audit examines graph declarations
and WGSL on a test context; passing it does not establish GPU numerical correctness.

## 8. Run independent node checks and compare both forms headlessly

The [node spec](../modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.node.spec.ts#L24)
compares the CPU function with committed NumPy fixtures. The
[generator](../modules/analysis/test/gpu-dataframe/pair-statistics/fixtures/generate-coregionalization.py#L3)
forms residuals directly and uses NumPy's eigensolver, whereas production caches sufficient
statistics and uses the shared Jacobi solver. That independence matters. Preserve generator
provenance and hand-derived expectations; a copied production function is not an oracle.

The node spec also checks malformed arrays, nonfinite bins, supplied-LMC compatibility, named
caps, output alias rejection, the node estimate and GPU gates. Its
[constructor test](../modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.node.spec.ts#L216)
does not exercise a wrong-format input: every supplied view has the expected format.
A discriminating validation case uses `context.view('uint32', props.distances.length)` as
`distances`, keeping the correct length and every other prop valid. At an untyped caller boundary,
construction must reject it with `coregionalization-fit distances must be packed, uint32-aligned
float32 GPU data`. The walkthrough review spec exercises this input with a deliberate type cast;
removing the format validator would fail that assertion. Add equivalent cases when extending
validation of other input formats or strides.

The
[headless spec](../modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.spec.ts#L78)
runs GPU and CPU operations with the same props through `runOperationGraph`, compares all
outputs, reconstructs factors, and states tolerances. It includes empty inputs, zero weights,
zero budget and correlated structures. The
[replay case](../modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.spec.ts#L211)
alternates valid and invalid data/models and checks untouched trailing capacity; the
[co-kriging case](../modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.spec.ts#L140)
tests the fitted model at its next consumer.

For every new numeric assertion, reject unexpected NaN explicitly with `Number.isFinite` or a
comparator that fails on NaN. Match integers and flags exactly; state absolute/relative tolerances
for floats. Use fixed small GPU inputs. Call `requireTestDevice`; do not return green when the
device is absent. Record the named test's pre-fix failure and post-fix result, and distinguish tests
you ran from headless tests that need a GPU runner.

## 9. Write the API page and generate Parameters

Update the [canonical coregionalization guide](../docs/api/dataframe-analysis/coregionalization-fit.md)
with input/output layouts, model definition, limits and their conditions, compile-time versus
per-frame values, status meanings, and CPU differences for the contributor and both operation
forms. Check that the catalog's documentation target still names that guide. Keep the
[family API heading](../docs/api/gpu-dataframe-analysis.md#gpucoregionalizationfit) as a summary
with a link to the canonical guide, rather than a second numerical contract. Link the real
consumer path. An allocated view alone is not a runnable example: point a first-time reader to
the operation walkthrough for submission, readback and cleanup.

The [generated Parameters block](../docs/api/gpu-dataframe-analysis.md#gpucoregionalizationfit-parameters)
comes from public types/packers and their TSDoc through
[doc-parameters.ts](../modules/analysis/test/analysis-core/doc-parameters.ts#L15).
Edit the TSDoc, then regenerate rather than hand-editing between the Parameters markers:

```sh
UPDATE_DOC_PARAMETERS=1 node_modules/.bin/vitest run --project node --maxWorkers=2 modules/analysis/test/analysis-core/doc-parameters.node.spec.ts
```

Review every generated diff; this command can update multiple API pages. Rerun that named spec
without the environment variable to check synchronization. A new API page also needs navigation
and snippet-check coverage. The
[shared API-page discovery](../modules/analysis/test/analysis-core/doc-program.ts#L17)
finds Markdown pages automatically; coregionalization lives on the existing dataframe page.
Ensure each discovered page has exactly one sidebar entry.

## 10. Make lens coverage demonstrate the same result

The [scenario definition](../modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/coregionalization-fit.lens.ts#L7)
names the element, dataset provenance/license and runnable module. Its
[family registration](../modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/index.ts#L29)
makes the scenario discoverable. The [example](../modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/coregionalization-fit.example.ts#L11)
adds the real contributor. The [runner](../modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/coregionalization-fit.run.ts#L44)
supplies three hand-derived spherical lag matrices for `B = [[2, -1], [-1, 3]]` and a fourth,
unrelated zero-pair bin. It sets an eight-sweep budget and records actual GPU readback when run;
the lens spec expects `iterationCount = 2`, distinguishing the budget from completed work.

The [capture declarations](../modules/lenses/src/computation-lenses/scenarios/gpu-dataframe/coregionalization-fit.run.ts#L92)
separate lag bins, triangle entries, parameters, coefficients and whole-fit diagnostics into
different key spaces. Do not align columns from those domains merely by row position. Include
every named status and its meaning. The [lens spec](../modules/lenses/test/computation-lenses/coregionalization-fit.spec.ts#L10)
checks readback provenance, expected matrix, reconstructed factors, zero residual, three usable
bins, weight `60`, and the excluded-bin count.

The API page's result and graph directives both link to the scenario's recording page. The
[website compiler](../website/build/markdown-plugin.ts#L175) renders the links, and the recording
page displays the graph. The committed capture is
`docs/lens-records/gpu-dataframe.coregionalization-fit.json`. Running the lens spec checks live
GPU readback. To regenerate the capture for publication, use the
[headless record-capture spec](../modules/lenses/test/computation-lenses/capture-records.spec.ts)
to capture this scenario on a GPU runner, with capture enabled and the scenario selected:

```sh
VITE_CAPTURE_LENS_RECORDS=1 node_modules/.bin/vitest run --project headless modules/lenses/test/computation-lenses/capture-records.spec.ts -t '^capture lens record gpu-dataframe[.]coregionalization-fit$'
```

Verify that the scenario record exists and render this API section through the active website
compiler. Both links should point to the same recording page, where readers can inspect
the capture and its graph.

When graph nodes or capture layouts change, arrange to recapture the committed lens records with
the GPU runner. Review the resulting tables and diagnostics against these expectations. A
documentation change that only links this existing scenario does not require new captures.

## 11. Validate the complete change

For changes to this walkthrough, run its evidence checks, including the website rendering probe,
mutable-props probe, canonical-guide comparison and wrong-format validation case:

```sh
node_modules/.bin/vitest run --project node --maxWorkers=2 modules/analysis/test/analysis-core/authoring-walkthrough.node.spec.ts
```

For this contributor, the smallest relevant node run is:

```sh
node_modules/.bin/vitest run --project node --maxWorkers=2 modules/analysis/test/gpu-dataframe/pair-statistics/coregionalization-fit.node.spec.ts modules/analysis/test/analysis-core/kernel-audit-coregionalization-fit.node.spec.ts
```

Run the named analysis and lens headless specs on a GPU runner. Typecheck each changed source/spec
through the scoped configuration supplied for the work, and apply the permitted formatter to the
owned files. Check public exports, catalog records, docs links and generated Parameters alongside
the numerical tests. Run `node scripts/check-upstream-first.mjs` and
`node scripts/check-contributor-contract.mjs`; report findings owned by other streams without
editing their files. Complete the repository's broader pre-merge gates when the coordinator allows
them. During a constrained review pass, hand the GPU and broad gates to the lead and report the
checks you actually observed.
