# TypeScript surface review — props, outputs and inference

## Summary

All 498 public command-node producers have an exported `<Class>Props` type, and all 67 generic producers have defaults for every class type parameter. A caller can usually construct a contributor without writing a type argument.

The largest gap is between **declaring an output and getting the promised value**. I executed a CPU operation that silently exchanged two named scores, a variogram fit that returned `undefined` for a field typed `number`, and a GPU operation factory that advertised a misspelled output it never requested from the lowering.

Direct GPU view typing is substantially better. Among 421 producers with a directly resolved, top-level `GraphDataView` field, `GPUTransferMask` was the sole producer exposing the unrestricted format on such a field. The operation bridge loses that precision again: its caller can choose a return format that disagrees with the actual view.

The highest-value change is to derive each operation’s declared outputs, lowering bindings and CPU return keys from **one validated snapshot of its selected ports**. This fixes a demonstrated wrong answer and gives TypeScript the information it needs to preserve output inference.

## Census

`S/` below means `modules/analysis/src/`. Counts are from the working checkout. I resolved the eleven `modules/analysis/package.json` export targets to source with the installed TypeScript 6.0.3 API, followed aliases with `getAliasedSymbol`, and queried constructor, property and return types. Public names are deduplicated across barrels except where I explicitly distinguish the two visibility recipes. I checked consumer snippets in memory with `strict: true`, without invoking `tsc` or emitting files.

| Question | Count and variants |
|---|---|
| Public classes | 761 distinct names. 498 have `getCommandNodes`. |
| Named producer props | 498/498 export `<Class>Props`. First constructor parameter is that type directly in 489, an inferred whole-props parameter in 8, and a stricter intersection in 1. There are 699 public `*Props` type aliases overall. |
| Class inference | 431 producers have no class type parameters. All 67 others default every parameter. All 498 expose one public construct signature. Among 294 direct constructions of the eight whole-props classes, **zero** supplied an explicit class type argument. |
| GPU outputs | 488/498 have a constructor `output` property. 317/498 export `<Class>Output`, zero export `<Class>Outputs`, and 181 expose their output another way or have no such property. Across all exports, 14 type names end in `Outputs`: 13 recipe types and `RasterIterationStatusOutputs`. A single view or an existing domain port does not need another alias. |
| CPU outputs | 131 public `compute<Name>OnCPU` functions. 120 explicitly return an exported `CPU<Name>Result`, 10 return anonymous shapes, and 1 reuses another public result type. Fourteen have at least one optional top-level result member. |
| Required selections | `{}` is assignable to the resolved `output` type of 80 producers. **34** of those also contain “requires at least one output” in the constructor. Some permit a separate output group, so 34 is a review inventory, not a count of unconditional failures. |
| Element formats and modes | 421 producers have at least one top-level field resolving directly to `GraphDataView` after removing `undefined`. Only `GPUTransferMask.source` and `.target` resolve to the unrestricted format. All 51 producers with a common top-level `mode`, `method`, `algorithm`, `strategy` or `policy` field use string-literal unions. This count excludes nested fields and mode-specific field names. |
| Readonly | 494 producers expose public `readonly props`, including inherited properties. `GPUTransferMask.props` is private. `GPUDotDensity`, `GPURandomPointsInPolygon` and `GPULocalFrameProjection` have no `props` property. `readonly props` alone says nothing about assignment to a member of `props`. |
| Open types | Direct public declaration syntax contained explicit `any` in 7 distinct export names, `unknown` in 27, `Record<string, …>` in 42, and zero explicit string index signatures in that scan. These categories overlap. Internal `as any`, local binding dictionaries, inferred types and transitive upstream declarations are excluded. |
| Overloads | Six qualified public functions have overloads, under five distinct names. All six have a selection or return-type issue below. |
| Operation bridge | 128 public classes directly extend `GPUContributorOperation`. All inherit the same string-indexed `outputs` declaration. |

Some directly repeatable counts:

```sh
rg 'extends GPUContributorOperation' modules/analysis/src --stats
# 128 matches in 128 files

rg -o --no-filename "getView<'[^']+'>" modules/analysis/src | wc -l
# 1328 single-format type arguments; two further calls specify format unions

rg -n 'Object.entries\(props.output\)' modules/analysis/src -g '*operation.ts' | wc -l
# 22 lines; these are candidates for inspection, not 22 confirmed phantom outputs

rg -n ': any\b|as any\b|<any>|Record<string' modules/analysis/src -g '*.ts'
# Candidate search. Counts above retain public signatures, not every textual hit.
```

For the assignability counts, the relevant operations were `checker.getExportsOfModule`, `checker.getDeclaredTypeOfSymbol(classSymbol).getProperty('getCommandNodes')`, and `checker.isTypeAssignableTo(typeOfEmptyObjectLiteral, resolvedOutputType)`. Using the type of a real `{}` expression matters. The tests for optional CPU results used the return signature and `SymbolFlags.Optional`, which also finds properties behind aliases.

### Five calls from the documentation

The variables below stand for the views, arrays or graph prepared in the corresponding page. These are executed **compiler** probes. Runtime observations are identified separately.

| Starting page | Caller code | What the editor caught or missed |
|---|---|---|
| `docs/api/gpu-dataframe-analysis.md`, variogram fit | `const options = {model: 'spherical' as const, diagnostics: showDetails}; const nugget: number = fitVariogramModel(bins, options).nugget;` | Accepted with `showDetails: boolean`. When true, the top-level `nugget` was `undefined`. |
| `docs/api/gpu-spatial-analysis/operators-dynamics.md`, class assignment | `new GPUClassAssignment({values, breaks, classCount, output, intervalClousure: 'right'})` | The fresh literal’s misspelling compiled. Replacing float32 `values` with a uint32 view correctly produced TS2322. |
| `docs/api/README.md`, first-run terrain | `new GPUTerrainDerivatives({...inputs, output: {slope}}).props.output.slope.length` | A correct float32 slope produced TS2532, “possibly undefined.” A uint32 slope correctly produced TS2322. `output: {}` compiled and the constructor rejected it. Keeping the original `slope` local, as the walkthrough does, works. |
| `docs/api/gpu-spatial-analysis/scoring-rules.md` and the CPU-twin convention | `const r = computeEnsembleScoresOnCPU(input, {memberCount: 2, dimension: 1, scores: ['crps']}); r.crps[0]` | TS18048 on the explicitly requested score. Conversely, `r.energyScore?.[0]` compiled even though this literal selection cannot produce energy. |
| `docs/api/gpu-network/operators.md`, path extraction | `computeNetworkPathExtractionOnCPU({...input, edges}).edges.count[0]` | Removing required `capacity` correctly produced TS2345. Supplying a complete edge resolver still produced TS2532 on `.edges`. The host returned edge count `1` for a two-node, one-edge path. |

## Proposed convention

A public constructor should accept its exported props type, and a public result should preserve each port name, format and presence guarantee established by the caller’s selection. Capture port names once at construction, reject unknown fixed-schema ports, and require a runtime format check wherever the format cannot be derived from the declaration.

Check this with public-entry consumer fixtures containing a valid literal, a widened variable, a bad format or key, and a read of an unrequested output. Pair the type fixture with a host assertion that the declared keys and returned or resolved keys agree.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| TYP-01 | W | UNUSUAL | ADDITIVE | S | CPU ensemble scores | Mutating the caller’s names swaps two valid scalar scores. |
| TYP-02 | W | ORDINARY | BREAKING | S | Variogram fit | A widened diagnostics flag makes a `number` field disappear. |
| TYP-03 | W | ORDINARY | ADDITIVE | S | GPU ensemble operation | An extra enumerable key creates a declared graph write with no corresponding binding. |
| TYP-04 | W | ORDINARY | BREAKING | M | Eight generic producers | Fresh literals accept misspelled options, including interval closure. |
| TYP-05 | W | UNUSUAL | ADDITIVE | S | CPU row readers | A valid structural input can return the wrong typed-array class. |
| TYP-06 | W | UNUSUAL | BREAKING | S | Computation recorder | An awaited callback loses its recording scope. |
| TYP-07 | C | ORDINARY | ADDITIVE | M | Operation view getter | The caller can select an element format independently of the resolved buffer. |
| TYP-08 | C | ORDINARY | ADDITIVE | M | Operation descriptors and CPU results | Open maps promise absent ports, while selected CPU ports remain optional. |
| TYP-09 | C | ORDINARY | BREAKING | S | Transfer mask | Five uint32-only bindings advertise every element format. |
| TYP-10 | E | ORDINARY | ADDITIVE | S | Five qualified overload sets | Valid union-typed configurations require a redundant caller branch. |
| TYP-11 | C | ORDINARY | BREAKING | M | Terrain derivatives and distance field | Constructors accept configurations that cannot request a computation. |
| TYP-12 | C | ORDINARY | BREAKING | S | Public lookup tables | A missing confidence level or operator key has a definite result type. |
| TYP-13 | C | ORDINARY | BREAKING | S | Frozen descriptors and spectral props | Assignments approved by the editor throw on frozen objects. |
| TYP-14 | C | UNUSUAL | BREAKING | S | Recorder graph getter | Reading a recorded graph back erases its frame type to `any`. |
| TYP-15 | E | ORDINARY | BREAKING | S | Population support alignment | The class’s named props type cannot be passed to its constructor. |
| TYP-16 | E | ORDINARY | ADDITIVE | S | Terrain catchment and CPU result barrels | One constructor type and ten CPU result shapes are missing from normal imports. |

### TYP-01: Two labelled CPU scores can exchange values after construction

**Where.** `S/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation.ts:172`, `:197`, `:218`.

**Evidence — EXECUTED.** Members `[[1,1],[3,3]]`, observation `[2,2]`, variogram pair `[0,1]`. I constructed the operation with `{energyScore:'energy', variogramScore:'variogram'}` and called its public CPU executor. It returned `{energy:0.7071067690849304, variogram:0}`. I swapped the two strings in the original output object and executed the **same operation on the same inputs**. It returned `{energy:0, variogram:0.7071067690849304}`. Its declared output names were still `['energy','variogram']`. Both outputs have the same length and element type.

**Now.** `output.energyScore = 'variogram'` changes which mathematical quantity the existing graph value receives. The constructor captures the score selections, then the executor rereads the caller’s names. This extends `KNOWN(PTS-133)` with a measured, shape-preserving output swap in a CPU operation.

**Wanted.** Capture immutable `[score, valueName]` pairs and use those same pairs in `super({outputs})` and `execute`. Capture the optional pair/weight input selections used by the callback at the same time.

**In-repository callers that change.** None. **Value and cost.** Users can reuse or edit a graph-building config without relabelling old results. S.

### TYP-02: The diagnostics overload can turn a typed variogram coefficient into NaN

**Where.** `S/dataframe-analysis/pair-statistics/variogram-model.ts:253–270`, `S/dataframe-analysis/pair-statistics/index.ts:19`. This is a consequence of the diagnostics change following `KNOWN(PTS-113)`.

**Evidence — EXECUTED.** The first caller example compiled. For bins `distances=[1,2,3]`, `semivariances=[1,2,3]`, `pairCounts=[1,1,1]`, with diagnostics true, the host returned keys `model, logRangeBracket, atBoundary, converged`. `fit.nugget` was `undefined`, and `2 * fit.nugget` was `NaN`. The plain-model overload accepts the widened options object by structural assignment.

**Now.** Changing `diagnostics: true` to a boolean from a UI toggle changes what TypeScript promises, even on the run where the toggle is true.

**Wanted.** Publish the existing `fitVariogramModelWithDiagnostics(input, options): VariogramModelFitDiagnostics` and its result type. Keep `fitVariogramModel(input, options): VariogramModel` single-shaped and remove the flag before alpha. A compatible interim fix is a `diagnostics?: false` plain-model overload plus an explicit boolean overload returning the union.

**In-repository callers that change.** **3 of 25** direct TypeScript calls for the recommended split. All three are in `modules/analysis/test/gpu-dataframe/pair-statistics/r19-search-diagnostics.node.spec.ts`. The other 22 omit the flag. **Value and cost.** A plotting or kriging caller gets the coefficient shape its editor promises. S.

### TYP-03: A reusable output object can invent a graph write

**Where.** `S/gpu-spatial-analysis/scoring-rules/gpu-ensemble-scores-operation.ts:29`, `:44`; the same family’s CPU validation is at `cpu-ensemble-scores-operation.ts:46`.

**Evidence — EXECUTED.** `const output = {crps:'c', energyScroe:'energy'}` compiled when passed to `GPUEnsembleScoresOperation`. The operation declared ports `['crps','energyScroe']`. I invoked its real contributor factory with a recording view getter. The complete resolution sequence was `['ensemble','observations','crps']`. Nothing requested the advertised energy output. The CPU operation rejected the identical object with `scores must request crps, energyScore or variogramScore`.

**Now.** Moving the output literal into a variable is enough to lose TypeScript’s excess-property check. The GPU wrapper enumerates arbitrary keys when declaring writes and uses explicit names when resolving writes. This is host evidence about the contract and bindings; I did not infer a particular readback value.

**Wanted.** Validate against `crps`, `energyScore`, `variogramScore` before publishing the contract. Produce both descriptors and bindings from the validated selection. Report `ensemble-scores output.energyScroe is unknown` for this input.

**In-repository callers that change.** None with valid ports. **Value and cost.** A typo fails where the graph is authored, with the offending name. S.

### TYP-04: Whole-props inference disables typo checking on fresh constructor literals

**Where.** All eight constructors are listed under **Departures by name**. The consequential example is `S/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment.ts:138`, `:148`, `:211`.

**Evidence — EXECUTED.** For **8/8**, `new Class({...validProps, optoin:true})` compiled. For class assignment specifically, `intervalClousure:'right'` compiled with no spread, and construction succeeded. The intended field remained absent and resolved to `'left'`. Passing a uint32 `values` view still failed at compile time. The loss is key checking, not general type checking.

**SOURCE, smallest GPU probe.** The generated comparison at `:211` selects `<` for right closure and `<=` otherwise. Submit `values=[1]`, `breaks=[0,1,2]`, `classCount=[2]`. Compare the correctly spelled right-closure call with the misspelled call. Expected classes are respectively `0` and `1`.

**Wanted.** Preserve inference of the independent row-view types while excluding extra top-level keys in the constructor parameter. For discriminated props, compute allowed keys across the branches so Fourier-specific or forest-specific keys remain legal. Show `satisfies GPUClassAssignmentProps` as the immediately available workaround.

**In-repository callers that change.** **0 of 294** direct TypeScript constructions of these eight supplied a top-level key outside the union of declared branch keys. **Value and cost.** Map authors get a diagnostic before an edge value changes class. M, chiefly to preserve the existing chunked-view inference fixtures.

### TYP-05: The CPU row readers assert an array class which their input type does not establish

**Where.** `S/analysis-core/operations/cpu-operation-rows.ts:13`, `:47`, `:66`, `:85`. Affects `getCPUOperationFloat32Rows`, `getCPUOperationUint32Rows`, `getCPUOperationInt32Rows`.

**Evidence — EXECUTED.** A structural `GPUCPUVector` with `format:'float32'`, one chunk containing `Uint32Array.of(1)`, and `pack()` returning that array passed strict TypeScript. `getCPUOperationFloat32Rows` returned a `Uint32Array`. Calling the nonmutating `rows.map(x => x / 2)[0]` returned **0**, where the promised `Float32Array` gives **0.5**. In the reverse probes, the uint32 and sint32 readers returned `Float32Array` and preserved `0.5`.

**Now.** `const rows: Float32Array = getCPUOperationFloat32Rows(...)` is accepted. The tag and the actual array are independently typed upstream. This matters for custom host adapters and direct calls to the public helpers; I did not observe the standard mixed executor constructing an inconsistent vector.

**Wanted.** Check the returned chunk or packed array’s actual element class before returning it. Preserve the one-chunk zero-copy path. Include the operation, port, expected class and observed class in the failure.

**In-repository callers that change.** None for consistent vectors. **Value and cost.** Custom CPU stages cannot silently switch arithmetic representation. S.

### TYP-06: An async recording callback silently loses its namespace

**Where.** `S/computation-record/computation-recorder.ts:71–76`; usage at `docs/api/computation-record.md:99`.

**Evidence — EXECUTED.** `await recorder.withScope({namespace:'basin'}, async () => { await Promise.resolve(); recorder.recordNode(node); })` compiled. `getNodeRecord(node.id)` returned `{namespace:''}`. Expected `basin`. The `finally` pops the scope when the promise is returned.

**Wanted.** Make `withScope` explicitly synchronous. Exclude `PromiseLike` callback results in its public signature and document the synchronous extent. If async graph authoring is needed, use a separately scoped recorder object passed to the callback. Keeping a shared stack pushed across `await` would mix concurrent scopes.

**In-repository callers that change.** **0 of 5** executable TypeScript calls examined returned a promise. **Value and cost.** Recorded nodes keep attributable provenance. S for the synchronous contract.

### TYP-07: The operation view getter’s format parameter behaves like a cast

**Where.** `S/analysis-core/operations/gpu-contributor-operation.ts:48`, `:211–226`.

**Evidence — EXECUTED, COUNT.** `const format: 'float32' = getView<'float32'>('mask').format` compiled. I ran the actual `GPU_CONTRIBUTOR_OPERATION_LOWERING.lower` with a context resolving `mask` to one packed uint32 view. The observed `format` was **`uint32`**. The host checks the port name and chunk count, then returns the chunk. Across source there are 1,328 single-format type arguments and two union-format arguments.

**Wanted.** Add checked forms `getView(port, 'float32')` and `getView(port, ['float32','uint32'] as const)` which infer their return format from the actual check. For declared output ports, derive the expected format from the descriptor. Start migrating the hand-written assertions at this one bridge.

**In-repository callers that change.** None to introduce the overloads. Retiring the unchecked form entails **1,330 explicitly parameterized uses** before considering contextually inferred calls. **Value and cost.** Operation authors can trust a format returned by the lowering. M.

### TYP-08: Unrequested operation ports look definite, while requested CPU results look optional

**Where.** `S/analysis-core/operations/gpu-contributor-operation.ts:99–101`; `S/gpu-spatial-analysis/scoring-rules/cpu-ensemble-scores-operation.ts:36`, `:73`; `S/gpu-network/network-analysis/cpu-network-path-extraction-operation.ts:56`, `:82`.

**Evidence — EXECUTED.** For an operation declaring only `crps`, `op.outputs.unsupplied.value` type-checked. `op.outputs.unsupplied` was `undefined`. In the other direction, with members `[1,3]`, observation `[2]` and `scores:['crps']`, the pure function returned exactly `{crps:Float32Array[0.5]}`. The compiler required an undefined check on `crps`. For network extraction, adding an edge resolver to predecessors `[0xffffffff,0]`, costs `[0,1]`, target `[1]` changed `.edges` from `undefined` to a result with count `1`. Its type was unchanged.

**Wanted.** Add literal-selection overloads to those two pure functions. A literal score tuple should return its selected keys as required, with a conservative overload for a dynamic array. A supplied non-optional edge resolver should return required `edges`. For operation extensions, add a typed `getOutput(port)` that preserves known keys and returns `undefined` for a dynamic absent key. New concrete wrappers should preserve their actual port map.

**In-repository callers that change.** None for the additive overloads and accessor. The inherited declaration is shared by all 128 wrappers, so change the bridge once. **Value and cost.** CPU users can read the result they asked for, and graph consumers can detect absence before dereferencing it. M.

### TYP-09: Selection transfer has five unparameterized uint32 views

**Where.** `S/computation-record/gpu-transfer-mask.ts:24–37`, `:51`. Fields are `source`, `target`, index `targets`, CSR `offsets` and CSR `neighbors`. `KNOWN(ERG-103)` concerns audit coverage; the departure here is the public element type.

**Evidence — EXECUTED.** `new GPUTransferMask({id:'mask', source:f32, relation:{layout:'index',targets:f32}, target:f32})` compiled. The constructor rejected the first binding with `mask sourceMask must be packed, uint32-aligned uint32 GPU data`. The neighbouring terrain slope type rejects the wrong format in the editor.

**Wanted.** Type all five fields as `GraphDataView<'uint32'>`. Keep the runtime validation for JavaScript, strides and alignment.

**In-repository callers that change.** **3 of 15** executable constructor sites intentionally pass a float32 view. All three are negative cases at `modules/analysis/test/computation-record/gpu-transfer-mask.node.spec.ts:93–96`; mark the expected compile errors. **Value and cost.** Selection wiring fails at the offending prop. S.

### TYP-10: Overloads accept each arm separately and reject a valid union variable

**Where.** `S/gpu-network/edge-bundling/gpu-edge-bundling.ts:118`; `S/gpu-network/visibility-graph/visibility-graph.ts:129`; `S/gpu-spatial-analysis/urban-morphology/visibility-graph.ts:120`, `building-tessellation.ts:121`; `S/gpu-spatial-analysis/landscape-ecology/patch-proximity.ts:81`.

**Evidence — EXECUTED.** `getGPUEdgeBundlingParameterValues({}, format)` produced TS2769 for `format:'float32'|'uint32'`. Executing the packer with the two values returned `Float32Array` and `Uint32Array` respectively. Passing the declared reference/CSR, reference/local or reference/boundary props unions gave TS2769 at **all four** recipe entries. Each implementation already takes the union.

**Wanted.** Publish a final overload with the implementation’s union parameter and union result, after the specific overloads. Literal callers keep their precise results. A caller reading a mode from UI state can dispatch once.

**In-repository callers that change.** None. **Value and cost.** Five public APIs become usable in typed dispatchers without duplicate branches or assertions. S.

### TYP-11: Nonempty output and seed requirements are missing from first-run props

**Where.** `S/gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts:107`, `:137`, `:198`; `S/raster-analysis/distance-field/gpu-distance-field.ts:97`, `:109`, `:192`, `:199`.

**Evidence — EXECUTED.** On a `1×1` grid, `output:{}` passed strict TypeScript for terrain derivatives and threw `terrain-derivatives requires at least one output`. A distance field with packed parameters, a distance output and neither seed source compiled and threw `distance-field requires seedPositions or seedMask`. As a control, `GPURasterNDWI` with `method:'gao'` and no `shortwaveInfrared` produced TS2345 before construction.

**Wanted.** Require at least one of the five derivative output ports. For distance fields, encode a point-seed arm, allowing an additional mask, and a mask-only arm which excludes `seedIds` and `seedCount`. Use the effective writer condition when extending the convention. In particular, `GPUGeometryMeasures` accepts group outputs as an alternative to feature outputs.

**In-repository callers that change.** An in-memory assignability probe found **3 of 46** direct terrain arguments and **8 of 28** direct distance-field arguments did not establish the proposed complete type. All eleven were in tests, principally `terrain-analysis` and `gpu-raster/distance-field`, including conditional fixture builders and negative cases. These are places to narrow the fixture shape or assert the intended error. **Value and cost.** Incorrect first-run configurations become visible before resource setup. M including migration.

### TYP-12: Three exported lookup tables claim every possible key exists

**Where.** `S/gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spot-parameters.ts:98`; `S/gpu-spatial-analysis/ontology/analysis-experience/analysis-operator-kind-records.ts:24`, `:65`. The KI critical-value table has the same type at `:103`.

**Evidence — EXECUTED.** `const z:number = GPU_EMERGING_HOT_SPOTS_CRITICAL_Z_SCORES['0.975']` compiled and `2*z` was `NaN`. The table has three keys. `GPU_ANALYSIS_OPERATOR_KIND_RECORDS.missing.outputs` compiled despite the lookup returning `undefined`. The measured table contained 377 keys. `GPU_ANALYSIS_OPERATOR_EMPTY_KIND_EXCEPTIONS.GPUTransferMask` was likewise `undefined`.

**Wanted.** Give the three critical levels literal keys and a checked lookup for a dynamic `number`. Type the dynamic operator tables as `Readonly<Partial<Record<string, Value>>>` or provide a lookup returning `Value | undefined`. Retain definite types for known keys. A consumer should get this protection with ordinary `strict`, independently of `noUncheckedIndexedAccess`.

**In-repository callers that change.** `rg` found **one** dynamic critical-value indexing expression, in the packer at `:209`, plus **four** dynamic operator/exception indexing expressions. The latter already guard or handle absence. **Value and cost.** An unsupported confidence level cannot silently become a numeric threshold. S.

### TYP-13: Several frozen objects advertise writable members

**Where.** `S/analysis-core/operations/gpu-contributor-operation.ts:36`, `:109`; `S/raster-analysis/spectral-indices/raster-spectral-index.ts:49–53`, `:75`. The spectral base serves `GPURasterEVI`, `GPURasterNBR`, `GPURasterNDWI`, `GPURasterSAVI`.

**Evidence — EXECUTED.** `evi.props.epsilon = 0.25` compiled. An actual one-pixel EVI instance reported `Object.isFrozen(evi.props) === true`, and the assignment threw `Cannot add property epsilon, object is not extensible`. Similarly, `operation.outputs.crps.length = 5` threw `Cannot assign to read only property 'length'`. `Readonly<Record<…>>` protects dictionary entries but leaves their descriptor members writable in the type.

**Wanted.** Expose `Readonly<Properties>` on the spectral base and `Readonly<GPUContributorOperationOutput>` as the descriptor value on constructed operations. Keep caller-owned configuration objects and buffer contents usable according to their own contracts.

**In-repository callers that change.** **0** direct writes found to the surveyed spectral policy fields or operation descriptor fields. **Value and cost.** The editor flags an assignment that the actual object rejects. S.

### TYP-14: `recordKernels` preserves a frame type that `recorder.graph` immediately erases

**Where.** `S/computation-record/computation-recorder.ts:45`, `:64`; `S/computation-record/describe-command-graph.ts:56`, `:96`.

**Evidence — EXECUTED.** With `graph: GPUCommandGraph<{year:number}>` and `wrong: GPUCommandNode<{name:string}>`, `recorder.recordKernels(graph).add(wrong)` gave TS2345. `recorder.graph?.add(wrong)` compiled. It is the same graph object.

**Wanted.** Keep the graph returned by `recordKernels` typed for authoring. Expose a recording lookup or graph-identity handle for inspection, with no writable graph methods. Replace the recorder fallback in `describeGPUCommandGraph` with that lookup. The concrete leak is the getter returning `GPUCommandGraph<any>`; a generic constraint that returns the caller’s original `Graph` has a different effect.

**In-repository callers that change.** **2** direct getter reads found: the description fallback and an identity assertion in `modules/analysis/test/computation-record/wgsl-kernel-records.node.spec.ts:187`. **Value and cost.** Adding recording cannot bypass the graph’s frame contract. S.

### TYP-15: The population-alignment props type belongs to the recipe, but bears the class name

**Where.** `S/gpu-spatial-analysis/landscape-ecology/gpu-population-support-alignment.ts:60`, `:155`; `segregation-counterfactual.ts:225`.

**Evidence — EXECUTED.** `function build(p: GPUPopulationSupportAlignmentProps) { return new GPUPopulationSupportAlignment(p); }` gave TS2345. The alias allows `output?: Partial<…>`; the constructor requires all four views. The recipe can allocate missing views and legitimately takes the looser shape. This was the **only** stricter constructor intersection in the 498-producer census.

**Wanted.** Give `GPUPopulationSupportAlignmentProps` the contributor’s required `output`. Export `GPUPopulationSupportAlignmentRecipeProps` for the existing optional-output shape and use it on the recipe and its preflight helper.

**In-repository callers that change.** **0** external TypeScript caller annotations found by `rg`. Update the two defining modules, the family barrel and the recipe’s catalog contract label. **Value and cost.** A wrapper accepting the documented class props can construct the class. S.

### TYP-16: One CPU constructor and ten pure functions need public type names

**Where.** `S/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation.ts:58`, `:307`, and `terrain-processing/index.ts:37`. The ten pure functions are listed below.

**Evidence — EXECUTED, COUNT.** Importing `CPUTerrainCatchmentSummaryOperationProps` from the public terrain entry gave TS2724. The compiler suggested `TerrainCatchmentSummaryOperationProps`, but the CPU-specific type adds optional named status outputs. `ConstructorParameters<typeof CPUTerrainCatchmentSummaryOperation>[0]` works. For the ten anonymous CPU results, the analogous workaround is `ReturnType<typeof compute…OnCPU>`.

**Wanted.** Re-export the existing CPU constructor props. Add and return named `CPU<Name>Result` aliases for the ten shape-returning functions. `computePolygonContiguityOnCPU` already returns the public `CPUContiguityWeightsResult`; keep that useful shared type. The two Vecchia likelihood functions also already have exported result aliases, even though the checker can print their shared underlying `CPUVecchiaLikelihoodRows`.

**In-repository callers that change.** None. **Value and cost.** Typed pipeline stages can import a result type by the same rule as their siblings. S.

## Departures by name

- **Whole-props constructors, TYP-04.** `GPUClassAssignment` (`distribution-dynamics/gpu-class-assignment.ts:138`), `GPUCalendarBuckets` (`calendar-buckets/gpu-calendar-buckets.ts:243`), `GPUGroupStatistics` (`group-statistics/gpu-group-statistics.ts:383`), `GPUPeriodIndex` (`period-index/gpu-period-index.ts:212`), `GPUTimeWordsRelative` (`time-words-relative/gpu-time-words-relative.ts:117`), `GPUCellPeriodKeys` (`value-cube/gpu-cell-period-keys.ts:161`), `GPUForestInference` (`forest-inference/gpu-forest-inference.ts:147`), `GPUFittingCoordinateEncoder` (`analysis-core/kernels/gpu-fitting-coordinate-encoder.ts:133`). I checked the KI copies of class assignment, period index and cell-period keys; each retains the whole-props constructor.
- **Optional CPU result members, TYP-08.** The fourteen `compute<Name>OnCPU` stems are `EnsembleScores`, `SeeminglyUnrelatedRegression`, `SpatialWeightsTraces`, `MultiscaleGeographicallyWeightedRegression`, `ProbitRegression`, `SpatialTwoStageLeastSquares`, `SpatialPanelLagMaximumLikelihood`, `SpatialWeightsAlgebra`, `TrajectoryKalman`, `TrajectoryRestriction`, `SpatialCrossValidation`, `SpatialInteraction`, `NetworkFlow` and `NetworkPathExtraction`. The ensemble and path cases are the executed recommendations. For the others, strengthen only facts determined by the supplied inputs or options.
- **Empty output type plus explicit constructor guard, TYP-11.** Spatial: `GPUGeometryMeasures`, `GPUShapeDescriptors`, `GPUTrajectoryMetrics`, `GPUTrajectoryPlayhead`, `GPUMobilityIndicators`, `GPUZonalStatistics`. Raster: `GPURasterJoin`, `GPURasterExtremaPyramid`, `GPURasterZonalStatistics`. Terrain and illumination: `GPUGeomorphons`, `GPUTerrainFlow`, `GPUTerrainHeightAboveDrainage`, `GPUTerrainHydrologicIndices`, `GPUPointHorizonProfile`, `GPUSimpleLocalRelief`, `GPUMultiScaleRelief`, `GPULocalDominance`, `GPUReliefBlend`, `GPUTerrainDerivatives`, `GPUTerrainViewshed`, `GPUTerrainCurvature`, `GPUTerrainSummits`, `GPUProfilePeaks`, `GPUSolarPosition`, `GPUTerrainHorizon`, `GPUTerrainCastShadow`, `GPUSolarIrradiance`, `GPUSolarShadowMask`, `GPUReliefShading`, `GPUTextureShading`, `GPUTerrainRuggedness`, `GPUTerrainVectorRuggedness`, `GPUTerrainTopographicPosition`, `GPUTerrainWeissLandforms`. The KI geometry-measures and terrain-flow copies retain the guard at lines 453 and 576 respectively. This list identifies the validation sites to compare with the types; alternative writer groups need their own branch.
- **Anonymous CPU result shapes, TYP-16.** Under `gpu-spatial-analysis`: `computeHealpixPointToCellOnCPU` (`healpix-cells/cpu-healpix-point-to-cell-operation.ts:37`), `computeMultipleTestingAdjustmentOnCPU` (`multiple-testing/cpu-multiple-testing-adjustment-operation.ts:27`), `computeSpatialPredicateJoinOnCPU` and `computeNearestFeatureJoinOnCPU` (`spatial-join/cpu-feature-join.ts:356`, `:485`), `computeSpatialLagImpactsOnCPU` (`spatial-regression/cpu-spatial-lag-impacts-operation.ts:70`), `computeSpatialWeightsValidationOnCPU` (`spatial-weights/cpu-spatial-weights-validation-operation.ts:27`), `computeBrownianBridgeUtilizationOnCPU` (`trajectory-home-range/cpu-brownian-bridge-utilization-operation.ts:63`). Under `gpu-network`: `computeNetworkTimeRoutingOnCPU` (`network-optimization/cpu-network-time-routing-operation.ts:224`), `computeRadiationFlowModelOnCPU` (`flow-aggregation/cpu-radiation-flow-model-operation.ts:44`), `computeNetworkTravelTimePercentilesOnCPU` (`network-time-geography/cpu-network-travel-time-percentiles-operation.ts:79`). I checked the KI copies of the two joins, lag impacts and weights validation; their corresponding result aliases are still absent.
- **Explicit public `any`.** `GPUComputationRecorder`, `DescribeGPUCommandGraphOptions`, `GPUSnapshotTap`, `WGSLKernelProps`, `recordWGSLKernelNodes`, `isRecordingWGSLKernelNodes`, `getWGSLKernelNodeRecord`. TYP-14 demonstrates the caller-visible loss. `GPUSnapshotTap.getCommandNodes<Parameters>` itself preserves the frame despite `any` in its `implements` clause. `WGSLKernelProps.modules` uses `ShaderModule<any,any,any>` for heterogeneous modules. I found no corresponding wrong-result probe for those declarations. Accepting `unknown` at a validator or importer is appropriate.
- **Inherited operation descriptors.** The single declaration in TYP-08 and TYP-13 reaches 79 spatial, 17 network, 13 raster, 11 terrain, 7 dataframe and 1 analysis-core operation. `rg -n 'extends GPUContributorOperation' modules/analysis/src` lists every affected class and file. The repair belongs in the base declaration and the wrapper’s selected-port type.

## Quick wins

- [ ] Snapshot ensemble score/name pairs. Pin the two-score swap with the numeric fixture in TYP-01.
- [ ] Reject an unknown ensemble output before creating either twin’s contract. Assert that every declared output is requested by the GPU contributor factory.
- [ ] Check the actual packed typed-array class in the three CPU row readers.
- [ ] Add the five union overloads from TYP-10 and preserve literal return types with paired consumer fixtures.
- [ ] Export `CPUTerrainCatchmentSummaryOperationProps` and the ten missing CPU result aliases.
- [ ] Add literal-result overloads for CRPS-only scoring and network path extraction with edges. Include a dynamic-selection case so the general type stays honest.

## Larger proposals

1. **Migrate the operation bridge first.** Add checked `getView(port, format)` and a presence-aware output accessor. The existing string descriptors already carry output formats. Move wrappers incrementally, beginning with ensemble scores. This gives the 128 wrappers a shared fix and avoids asking each author to maintain a separate assertion.
2. **Take the small variogram break before alpha.** Two named functions give stable result shapes, including through wrappers which widen options. Three in-repository test calls change. Keeping the flag with a truthful boolean overload also works for directly typed callers.
3. **Use stricter props where execution already depends on the distinction.** Start with transfer-mask formats, derivative outputs, distance-field seeds and class-assignment keys. Preserve legitimate alternate output groups and independent chunk layouts. The NDWI method union is an existing pattern authors can follow.
4. **Extend the public-surface check through the actual barrels.** `KNOWN(ERG-104)` identified missed members in syntax-based vocabulary checks. Add type fixtures for widened booleans, reusable objects with extra keys, omitted outputs and readonly writes. A successful literal-only documentation example misses all four.

## Not read

I read the ensemble CPU operation, its GPU operation wrapper, the contributor-operation bridge, computation recorder and CPU row readers in full. I read the public declarations, overload bodies and relevant constructor or return branches for the other findings. I sampled documentation through the first-run, variogram, scoring, dynamics, network, spectral-index and recording paths. The census visited public symbols across all eleven entries. I checked the synthesis, relevant area reports, API/catalog and direction reports, the decision list, and staging-owned copies for the cited departures.

I did not inspect every shader, every implementation behind the 128 inherited wrappers, every dynamic `Record<string, …>` use, or every conditional CPU result branch. The 22 `Object.entries(props.output)` matches are not all defects. No GPU, browser, network, build, `tsc`, Vitest or file writes were used. The class-boundary shader outcome in TYP-04 remains **SOURCE**; the numeric swaps, missing fields, concrete array classes, constructor failures and TypeScript diagnostics reported above were executed locally.