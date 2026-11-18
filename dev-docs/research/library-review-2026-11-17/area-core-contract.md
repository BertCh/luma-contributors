# Analysis core and the contributor contract

`analysis-core` earns its place as a shared implementation layer. Raster and spatial contributors really do share triangulation, and three families import ring assembly. Its public entry point has a different problem. It puts contributor-authoring contracts, Arrow conversion, graph primitives, and model-specific results behind one import, while some of its most-used authoring helpers remain private.

I would fix the public data-contract defects before reorganizing files. The most consequential are a slice that can address a neighboring logical allocation, an exact conversion that silently loses data, and two parameter APIs that can deliver different values from the ones they validated.

## What I checked

I reviewed the working tree at `bdc715ea`, read the contributor contract and repository instructions, and checked candidate findings against the roadmap and the four earlier review collections named in the brief. Both permitted checkers completed successfully. `check-contributor-contract.mjs --all` reported **zero new findings**. `check-upstream-first.mjs` reported **no violations**. I made no file changes and ran no repository compilation, tests, or GPU workloads.

**Scope map.** Counts are TypeScript files and physical lines, including documentation and WGSL strings.

| Under `modules/analysis/src/analysis-core` | Files | Lines | Contents |
|---|---:|---:|---|
| `contracts/` | 17 | 3,492 | Status, ports, coordinates, time, parameter schemas, inference, model results, capacity and partition planning |
| `kernels/` | 55 | 11,391 | WGSL numerics, graph construction, scans, sorts, fitting contributors, CPU numerics and validators |
| `graph-support/` | 4 | 563 | Arrow normalization, precision conversion and time upload |
| `operations/` | 6 | 664 | Graph lowering, CPU row access, trajectory math and network routing |
| `polygon-triangulation/` | 7 | 4,669 | GPU contributor, parallel fallback, CPU twin and operation adapters |
| `ring-assembly/` | 5 | 2,853 | GPU contributor, quantization, shell indexing and diagnostics |
| Root | 7 | 1,210 | Barrel plus cell keys, device limits, longitude/latitude rings, topology, raster grid and spherical cell area |
| **Total** | **101** | **24,842** | **Geometry implementations alone account for 30.3% of core.** |

**Depth.** I read the core barrel, contract barrel, `wgsl-kernel-nodes.ts`, capacity planner, parameter schema, temporal context, status and inference contracts, model-result builder, device limits, Arrow normalization and conversion, chunk gather, minibatch-order contributor, and layering spec in full. I sampled the largest two contributors at their public interfaces, validation, scratch sizing and generated-node paths. I also sampled the CPU triangulation twin, graph lowering, spatial context, chunk-group builder, testing utilities, checker fixtures, kernel audit and catalogue specs. The newest implementation sample came from `69305eb4`. I did not audit every shader’s math or establish GPU outcomes from source text.

For compact citations below, `core/` means `modules/analysis/src/analysis-core/`, `testing/` means `modules/analysis/src/testing/`, and `core-test/` means `modules/analysis/test/analysis-core/`. Other paths are repository-relative.

### Helper adoption

These are **direct named-import calls or constructions / distinct calling files** across 1,688 analysis source files, excluding `testing` and `computation-record`. I parsed source without typechecking, followed local import aliases, and counted calls to symbols imported directly from a core path. Calls within the defining file, calls through a family wrapper and external consumers are excluded. In particular, applications usually own `GPUParameterBuffer`, so its internal count measures a different thing from the kernel-builder count.

| Helper | Calls / files | Helper | Calls / files |
|---|---:|---|---:|
| `createWGSLKernelNode` | 2,396 / 551 | `getWGSLKernelBinding` | 10 / 4 |
| `createFillNode` | 199 / 117 | `createPublishNode` | 42 / 33 |
| `createWGSLDispatchGate` | 64 / 52 | `createWGSLActiveCountDispatch` | 10 / 7 |
| `GPUParameterBuffer` | 7 / 6 | `captureGraphCommandNodes` | 22 / 19 |
| `validateCapacity` | 1,892 / 608 | `validateMaximumCount` | 334 / 189 |
| `validateRequirement` | 1,678 / 449 | `validateFiniteNumber` | 233 / 129 |
| `validateViewLength` | 375 / 100 | `validateCommandNodeBudget` | 169 / 166 |
| `validateCompactOutput` | 33 / 30 | `validateGPUStorageBindingByteLength` | 334 / 241 |
| `validateSinglePackedView` | 1,790 / 435 | `getStorageBindingElementLimit` | 86 / 60 |
| `validateGraphOutputsDisjointFromInputs` | 354 / 315 | `validateDisjointWritableViews` | 176 / 125 |
| `validateGraphViewsBelongToGraph` | 296 / 279 | `validateGraphViewTopology` | 14 / 9 |
| `validateGraphChunkBindingRanges` | 235 / 227 | `validateGraphChunkDispatchBindings` | 30 / 30 |
| `validatePowerOfTwo` | 28 / 25 | `validateNonZeroFiniteNumber` | 14 / 10 |
| `validateExtent` | 1 / 1 | `validateGPUStatusPort` | 30 / 29 |
| `validateGPUCompactPairPort` | 3 / 3 | `validateGPUFeatureGeometryPort` | 1 / 1 |
| `validateGPURecipeStatusPort` | 0 / 0 | `validateGPUPartitionDescriptor` | 1 / 1 |
| `resolveGPUSpatialContext` | 225 / 211 | `resolveGPUSecondarySpatialContext` | 20 / 16 |
| `resolveGPUTemporalContext` | 30 / 28 | `validateGPUTemporalContext` | 1 / 1 |
| `assertCompatibleGPUSpatialContexts` | 7 / 7 | `assertCompatibleGPUTemporalContexts` | 1 / 1 |
| `defineGPUSpatialParameterSchema` | 7 / 6 | `packGPUSpatialParameterValues` | 4 / 3 |
| `createGPURegressionModelResult` | 5 / 5 | `getGPURegressionPredictorNames` | 4 / 4 |

The ten generic capacity planning and recovery functions had **zero direct imported calls** in that population. The public barrel intentionally exposes only the two plan types. Spatial joins and weights have live domain planners. Inference is mainly a type contract: 21 files directly import `GPUInferenceMetadata`, 28 import `GPUIterationStatusPort`, nine import `GPURecipeStatusPort`, and one imports `GPUPermutationStatisticAdapter`. These numbers support keeping the shared vocabulary and exercising restraint about publishing more machinery. `core/contracts/index.ts:177`, `core/contracts/inference.ts:30`, `modules/analysis/src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join.ts:998`.

Of 492 `GPU*` classes directly declaring `getCommandNodes`, 148 had no direct core kernel-builder call in their file, 337 had no direct budget-helper call, and 319 had no direct byte-length-helper call. These are **screening counts**, since delegation and fixed-size work are legitimate. A concrete §4 departure is `GPUSegmentRingAssembly`: it derives `rounds` from the segment count and pushes nodes per round, but exposes no estimate. A legitimate delegation is `GPUFittingMinibatchOrder`, which calls `validateFittingGraph` and uses published `GPUSort`. `core/ring-assembly/gpu-segment-ring-assembly.ts:1250`, `:1262`; `core/kernels/gpu-fitting-minibatch-order.ts:80`, `:102`.

Representative census commands, including a reproducible single-symbol version of the direct-call census:

```sh
node scripts/check-contributor-contract.mjs --all
node scripts/check-upstream-first.mjs
rg --files modules/analysis/src/analysis-core modules/analysis/src/testing
rg -n 'contract-allow:\s*single-invocation' modules/analysis/src
rg -n '0x7feb352d|fn \w*(lowerBound|upperBound)\w*\(' modules/analysis/src -i
```

```sh
SYMBOL=createPublishNode node -e '
const fs = require("fs"), path = require("path"), ts = require("typescript");
const root = "modules/analysis/src", symbol = process.env.SYMBOL;
let calls = 0, files = 0;
for (const name of fs.readdirSync(root, {recursive: true})) {
  if (!name.endsWith(".ts") || /^(testing|computation-record)\//.test(name)) continue;
  const file = path.join(root, name);
  const tree = ts.createSourceFile(file, fs.readFileSync(file, "utf8"), ts.ScriptTarget.Latest);
  const aliases = new Set();
  for (const item of tree.statements) {
    if (!ts.isImportDeclaration(item)) continue;
    const target = path.join(path.dirname(file), item.moduleSpecifier.text);
    if (!target.startsWith(root + "/analysis-core")) continue;
    for (const binding of item.importClause?.namedBindings?.elements ?? []) {
      if ((binding.propertyName ?? binding.name).text === symbol) aliases.add(binding.name.text);
    }
  }
  const before = calls;
  function visit(node) {
    if ((ts.isCallExpression(node) || ts.isNewExpression(node)) &&
        ts.isIdentifier(node.expression) && aliases.has(node.expression.text)) calls++;
    ts.forEachChild(node, visit);
  }
  visit(tree);
  if (calls > before) files++;
}
console.log(symbol, calls, files);'
```

## Findings

### CORE-01 A chunk subview can address rows outside its parent [P0] [correctness]

- **Evidence.** `sliceGraphChunkGroupView` computes `view.byteOffset + rowStart * view.rowByteLength` and creates a view on `view.buffer`. It never checks the parent’s length or stride. In a 32-byte buffer, a two-row `uint32` parent starting at byte 8 occupies bytes 8–15. Requesting `(rowStart=2, rowCount=1)` produced a child descriptor at byte 16 in an extracted host probe. Published `createDataView` validates against the backing handle’s 32 bytes. The helper is publicly exported. `core/kernels/graph-chunk-groups.ts:129`, `:136`; `core/index.ts:77`; `node_modules/@luma.gl/graph/src/gpu-core/gpu-command-graph.ts:2375`.
- **Why it matters.** The child can be passed on as a writable view of the neighboring allocation. A valid strided parent also produces the wrong row address when stride exceeds row width. The local kriging caller bounds its own slices, so this finding concerns the public helper. `modules/analysis/src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging.ts:813`.
- **Proposed change.** Require a packed parent as documented, validate integer row bounds against `view.length`, and allow an empty slice at the end. Add a test using a parent in the middle of a larger buffer, with sentinels on both sides. **S.** Rejects formerly accepted invalid slices.
- **Confidence.** High. The incorrect descriptor is observable on the host.

### CORE-02 Exact Arrow Decimal conversion silently loses a unit [P0] [correctness]

- **Evidence.** Decimal words are assembled as a `bigint`, then converted with `Number(integer)` before the precision policy sees them. The policy immediately accepts when `Object.is(value, Math.fround(value))`. With an actual Arrow Decimal128 vector containing scale-zero `9007199254740993`, the extracted conversion returned `9007199254740992`. That rounded value passes the exact check and leaves reported error at zero. `core/graph-support/arrow-analytics-wide-numeric.ts:29`, `:37`; `core/graph-support/arrow-analytics-normalization.ts:165`, `:225`.
- **Why it matters.** A caller explicitly requesting lossless preparation gets changed data and a clean conversion report. Conversely, the Int64 branch throws before even a permissive round policy can evaluate the original integer. `core/graph-support/arrow-analytics-wide-numeric.ts:22`.
- **Proposed change.** Carry the integer coefficient and decimal scale through the comparison with the candidate float32. Compare exact rational values for exact mode and compute rounding error against the original value for round mode. Cover scale zero, a scaled decimal, negative values, and `2^53 + 1`. **M.** Corrects observable conversion behavior.
- **Confidence.** High. Reproduced with CPU arithmetic and Arrow’s real accessor.

### CORE-03 Parameter buffers accept a different scalar representation from their declared format [P0] [correctness]

- **Evidence.** Both `values` and `write` accept all three typed arrays regardless of `Format`. The constructor calls `this.buffer.write(props.values)` directly. Thus `{format: 'uint32', length: 1, values: new Float32Array([1])}` passes the public type and supplies the word `1065353216`. The published WebGPU buffer forwards the underlying bytes. Initial values also bypass the range check on later writes, and that check accepts fractional `firstElement`. `core/kernels/gpu-contributor-utils.ts:28`, `:57`, `:66`, `:85`; `node_modules/@luma.gl/core/src/webgpu/adapter/resources/webgpu-buffer.ts:93`.
- **Why it matters.** A numeric argument that appears to be `1` can become a billion-scale count or seed. The caller gets no format diagnostic.
- **Proposed change.** Use the existing format-to-array mapping for construction and writes, validate representation and integer bounds before allocation or upload, and share the checked write path. Give intentional bit reinterpretation an explicit API only if a consumer needs it. **S.** Source-breaking for mismatched arrays.
- **Confidence.** High. Verified the word representation on the host.

### CORE-04 The parameter schema can emit infinity from a finite-only field [P0] [correctness]

- **Evidence.** The packer calls `validateFieldValue` on the JavaScript number, then assigns it to a typed array. A float32 field with no upper bound accepts `1e39` as finite and stores `Infinity`. This reaches a real packer: trajectory `stopSpeedThreshold` has `minimum: 0`, documents a finite threshold and delegates directly to the schema. The existing `validateFiniteNumber` already checks the rounded value. `core/contracts/parameter-schema.ts:99`, `:126`; `core/kernels/gpu-contributor-utils.ts:551`; `modules/analysis/src/gpu-spatial-analysis/trajectory-analysis/trajectory-metrics-parameters.ts:19`, `:50`, `:80`.
- **Why it matters.** The packed value violates the declared domain before the graph executes. The same discrepancy applies to schema defaults.
- **Proposed change.** Apply finite and range checks to the encoded float32 value as well as any documented input constraints. Preserve explicit `allowNonFinite` behavior. Add the trajectory example and a value that rounds across a field bound. **S.** Rejects values that previously produced an out-of-contract word.
- **Confidence.** High. `Math.fround(1e39)` is `Infinity`.

### CORE-05 Corrected AIC can reward a model for having too few observations [P0] [correctness]

- **Evidence.** The shared result builder divides by `rowCount - criterionParameterCount - 1` without checking its sign. With `n=3`, `k=3` and log likelihood `−10`, it returns AIC `26` and corrected AIC `2`. At `n=k+1` it divides by zero. The OLS decoder uses the shared calculation. Another model family already has a regression case requiring rejection of a nonpositive correction denominator. `core/contracts/model-result.ts:202`, `:208`, `:216`; `modules/analysis/src/gpu-spatial-analysis/spatial-regression/ordinary-least-squares-statistics.ts:216`; `modules/analysis/test/gpu-spatial-analysis/spatial-regression/generalized-gwr-selection.spec.ts:537`.
- **Why it matters.** A caller sorting public model summaries by corrected AIC can prefer a fit for which the correction is undefined.
- **Proposed change.** Compute the denominator once. Return `null` when the derived correction is undefined, document that convention, and preserve explicit estimator overrides. Test both zero and negative denominators. **S.** Observable result correction within the existing nullable type.
- **Confidence.** High. The public builder and calculation are host-side.

### CORE-06 The compact publisher gives `overflow` two incompatible meanings [P1] [consistency]

- **Evidence.** The base port defines `overflow` as insufficient **final output** capacity. The compact output’s own comment says “any capacity.” The publisher ORs `overflowSources` into that same word. Region statistics passes `candidates.candidateFlag` explicitly. Meanwhile §9 requires `candidateOverflow` to remain separate, and the capacity recovery model needs that distinction to decide which allocation to grow. `core/kernels/gpu-contributor-types.ts:29`, `:45`; `core/kernels/wgsl-kernel-nodes.ts:897`, `:961`; `modules/analysis/src/gpu-spatial-analysis/region-statistics/gpu-region-statistics.ts:520`; `dev-docs/contributor-contract.md:55`; `core/contracts/capacity-planning.ts:57`.
- **Why it matters.** Increasing `ids.length` cannot recover candidates discarded earlier. The contract also calls `requiredCount` exact even though the field is optional, candidate loss can make the global requirement unknowable, and the shared scan can saturate at `0xffffffff`. `core/kernels/gpu-contributor-types.ts:25`; `core/kernels/scan-total.ts:15`.
- **Proposed change.** Give publish inputs separate final and candidate overflow destinations. Document which stage a requirement counts and whether it is exact or saturated. Migrate callers with a case where candidates overflow and final capacity is ample. **M.** Changing the legacy aggregate bit requires a compatibility period.
- **Confidence.** High. **KNOWN(STD-04)** is under-scoped: the contradiction lives in the shared producer and type, so migrating individual port names cannot fix it.

### CORE-07 Fill and inferred binding helpers accept layouts their implementations cannot honor [P1] [ergonomics]

- **Evidence.** `createFillNode` promises to fill a view, defaults `componentCount` to `view.length`, and writes one scalar per index. For a packed two-row `float32x4`, it generates two scalar writes for eight components. `getWGSLKernelBinding` accepts unrestricted `GraphDataView` and chooses `f32` for everything outside the `uint32*` and `sint32*` prefixes, including packed `uint16x2`. The public graph format union includes vertex and list formats. `core/kernels/wgsl-kernel-nodes.ts:814`, `:819`, `:1000`, `:1013`; `node_modules/@luma.gl/graph/src/gpu-data/gpu-vector-format.ts:51`.
- **Why it matters.** Both APIs accept a normal-looking view that changes the meaning of “element.” Only 10 direct calls use the inferred binding helper. Across the seven domain families I counted 10,148 literal `type`/`access` pairs in 580 files, so the library’s predominant pattern still asks authors to specify the representation manually.
- **Proposed change.** Restrict inferred bindings to supported packed 32-bit scalar/vector layouts. Derive full-fill word count from the validated format, and range-check explicit prefix fills. Keep a separately named raw-word binding for deliberate bit reinterpretation. **M.** Audit existing vector fills for intentional prefix behavior.
- **Confidence.** High for generated types and index counts. **UNVERIFIED** on GPU.

### CORE-08 Construction-time validation can be invalidated through the caller’s props object [P1] [architecture]

- **Evidence.** All six core `GPUCommandNodeProducer` classes retain the caller’s props object. `GPUFittingMinibatchOrder` validates `rowCount` against `output.order.length`, then consults `props.rowCount` again during graph declaration. Changing the original object’s `rowCount` to zero after construction makes `getCommandNodes` return `[]` for a previously nonempty fit. Across production analysis source, `rg -n 'this\.props\s*=\s*props\b'` finds 390 assignments in 383 files. The operation adapter already copies and freezes its output descriptors. `core/kernels/gpu-fitting-minibatch-order.ts:66`, `:79`; `core/operations/gpu-contributor-operation.ts:102`; `core/ring-assembly/gpu-segment-ring-assembly.ts:504`.
- **Why it matters.** The contract depends on validating shape once. A `readonly props` member protects the reference from reassignment, while the caller can still mutate the object or its nested output map.
- **Proposed change.** Snapshot shape-driving scalars, options and port maps at construction. Preserve the identities of graph views and buffers, whose contents are intentionally dynamic. Add a mutation-after-construction case to the authoring fixture. Start with the six core classes. **M.** Changes behavior for callers relying on undocumented mutation.
- **Confidence.** High for the demonstrated case. The package-wide count measures the pattern, not 390 independently proven defects.

### CORE-09 Importing the recommended authoring API requires the optional Arrow peer [P1] [packaging]

- **Evidence.** The contract directs authors to `/analysis-core` for `validateCapacity` and kernel helpers. Its barrel eagerly re-exports `graph-support`, which re-exports Arrow normalization; that module imports runtime values from `apache-arrow`. The manifest marks Arrow optional. Existing built CJS also contains `require("./graph-support/index.cjs")` and `require("apache-arrow")`. `dev-docs/contributor-contract.md:5`; `core/index.ts:6`; `core/graph-support/index.ts:5`; `core/graph-support/arrow-analytics-normalization.ts:5`; `modules/analysis/package.json:94`; `modules/analysis/dist/analysis-core/index.cjs:53`; `modules/analysis/dist/analysis-core/graph-support/arrow-analytics-normalization.cjs:23`.
- **Why it matters.** An author using only validation or WGSL construction has a runtime dependency that the package manager is explicitly allowed to omit.
- **Proposed change.** Add an Arrow-free authoring entry and an explicit Arrow integration entry. Keep the existing barrel for compatibility, update the contract’s import example, and verify the new entry in an install without Arrow. **M.** Additive first phase.
- **Confidence.** High for the dependency chain. I did not create a clean installation.

### CORE-10 The contract baseline permits replacement regressions [P1] [testing]

- **Evidence.** For count rules, `isNewFinding` asks only `finding.count > recorded`. Fixing one loop and introducing an unrelated one in the same file can go green. Only 61 of the 85 loop files have a reason. Eleven additional inline `contract-allow: single-invocation` markers across nine files sit outside the baseline. The checker itself explains that storage preflight verifies presence/order, not every allocation formula. `scripts/check-contributor-contract.mjs:775`, `:1113`, `:1124`, `:1240`; `scripts/contributor-contract-baseline.json:36`, `:128`. One justified bounded-loop example is `modules/analysis/src/gpu-spatial-analysis/permutation-inference/permutation-inference-kernels.ts:601`.
- **Why it matters.** “Zero new findings” currently means the measured file totals did not grow. It gives weaker assurance than a maintainer reviewing a new kernel might infer.

| Live baseline rule | Files | Reported total | Unit |
|---|---:|---:|---|
| `single-invocation-data-loop` | 85 | 118 | Flagged kernel sites, not proven data-sized loops |
| `multiple-classes` | 9 | 21 | Classes in those files, not 21 excess classes |
| `root-file` | 21 | 21 | Files in the five checked roots |
| `storage-binding-preflight` | 1 | 1 | `GPUNetworkIsochrones` |
| Other seven rules | 0 | 0 | No live findings |

- **Proposed change.** Key loop allowances by file, containing symbol and stable kernel or variant identity, with a reason or roadmap ID. Fail on stale allowances. Keep the six strict zero-baseline rules. Report the checked bound or stage in storage diagnostics. **M.** Checker-only.
- **Confidence.** High. `scripts/check-contributor-contract.mjs:34`, `:56`, `:1300`; `scripts/contributor-contract-baseline.json:193`.

### CORE-11 Two structural tests can pass without enforcing their titles [P1] [testing]

- **Evidence.** The layering spec initializes `DOCUMENTED_ENTRIES = {}` and loops over its entries to check documented contributors. The runtime-boundary spec walks imports, drops external specifiers, and never asserts against a forbidden target. Its separate parser fixture has assertions, but the seven entry traversals have no rejection condition. The catalogue spec checks that an added operator’s documentation *file* exists, and the kernel audit enumerates seven domain barrels; the four fitting contributors in core depend on a separate manually maintained suite. `core-test/analysis-layering.node.spec.ts:43`, `:188`; `core-test/analysis-runtime-boundaries.node.spec.ts:93`, `:104`, `:111`; `modules/analysis/test/gpu-spatial-analysis/ontology/analysis-operator-catalog.node.spec.ts:150`; `core-test/kernel-audit/analysis-contributors.ts:15`; `core-test/kernel-audit-fitting-kernels.node.spec.ts:9`.
- **Why it matters.** A removed documentation heading, a forbidden runtime import, or a new core contributor can escape the particular guard whose name suggests coverage. Existing fitting cases are valuable and do run in their own suite.
- **Proposed change.** Feed heading checks from the maintained docs inventory, require a nonempty mapping, and add one failing fixture for each boundary. Include core in contributor discovery, with explicit classifications for helpers, and reuse the fitting cases. **M.** Test-only. `core-test/doc-program.ts:21`.
- **Confidence.** High. **KNOWN(STD-13)** understates the immediate issue: two existing guards lack an effective assertion over their intended subjects.

### CORE-12 Test utilities undercount shader memory while calling the result an upper bound [P1] [testing]

- **Evidence.** `getTypeBytes` recognizes an array length only if it is a decimal literal, then falls back to four bytes. `var<workgroup> scratch: array<u32, N>;` with `const N = 8192u` therefore counts as **4 bytes** instead of **32,768**. The kernel audit already resolves constants, overrides and structs. Textual call-site searches found `getPrivateMemoryBytes(...)` in 50 analysis test files and `getWorkgroupMemoryBytes(...)` in 44. `testing/gpu-contributor-test-utils.ts:159`, `:168`, `:180`, `:192`; `core-test/kernel-audit/kernel-audit-wgsl.ts:808`; `modules/analysis/test/gpu-network/network-optimization/network-flow-parallel.node.spec.ts:79`.
- **Why it matters.** A passing resource-budget assertion can reassure a reviewer about a kernel far beyond the stated budget.
- **Proposed change.** Move the audit’s WGSL layout accounting into private test support, use it from both paths, pass pipeline constants, and return an explicit unresolved result if an upper bound cannot be established. **M.** Test-only.
- **Confidence.** High for the host estimator. No claim about actual device allocation.

### CORE-13 Section 14 attributes a local one-chunk restriction to the upstream executor [P2] [docs]

- **Evidence.** Section 14 labels its one-packed-chunk limit “luma.gl 10.0.0-alpha.12.” The workspace pins alpha.14; `AGENTS.md` still says alpha.13. The installed executor preserves `chunkLengths`, accepts chunked CPU results and handles scalar inputs. The **analysis lowering** explicitly enforces `chunks.length === 1`, while the CPU accessors can call `input.pack()`. Textures and state vectors are still rejected by the mixed executor. `dev-docs/contributor-contract.md:96`, `:101`; `package.json:42`; `AGENTS.md:21`; `node_modules/@luma.gl/graph/src/gpu-core/gpu-cpu-execution.ts:37`, `:48`, `:189`, `:203`; `core/operations/gpu-contributor-operation.ts:216`; `core/operations/cpu-operation-rows.ts:27`.
- **Why it matters.** “A contributor that needs anything else has no operation form yet” discourages work for reasons the current dependency no longer imposes. The alpha.12 timings are valid as dated measurements.
- **Proposed change.** Split §14 into current executor capabilities, restrictions of the analysis adapter, and linked benchmark evidence. Replace §6’s “roadmap TC-01” with the live temporal resolver and support type. Move the historical compiler narrative in §11 to its cited evidence, retaining the actionable rules. **S.** Documentation-only. `dev-docs/contributor-contract.md:38`, `:71`; `dev-docs/README.md:22`.
- **Confidence.** High. Verified against the installed dependency source.

### CORE-14 The root rule skips 30 files across three public entries [P2] [architecture]

- **Evidence.** The checker names five roots. It omits `analysis-core`, `pointcloud-analysis` and `geospatial-analysis`, all published entries. They have **6, 21 and 3** non-index TypeScript files respectively. The six in core are `cell-key-wgsl.ts`, `device-limits.ts`, `lon-lat-ring-wgsl.ts`, `network-topology.ts`, `raster-grid.ts` and `spherical-cell-area.ts`. The vocabulary spec independently omits pointcloud from its layer list. `dev-docs/contributor-contract.md:11`; `scripts/check-contributor-contract.mjs:24`, `:1174`; `modules/analysis/package.json:59`; `core/index.ts:68`, `:74`; `core-test/analysis-vocabulary.node.spec.ts:30`.
- **Why it matters.** A contributor’s location determines whether a structural rule applies. The reported 21 root files cannot serve as a repository-wide migration count.
- **Proposed change.** Derive the checked public entry set from `package.json`, with explicit exceptions for infrastructure such as `computation-record`. Classify the six core files by owner and preserve barrel imports through any move. **S** for enforcement, **M** for relocation. No public import break required.
- **Confidence.** High. **KNOWN(STD-12)** tracks loose roots; its checker coverage and denominator need correction.

### CORE-15 The public frame-unit helper can produce a context its validator rejects [P2] [ergonomics]

- **Evidence.** `getGPUSpatialFrameContextUnits` returns `'kilometers'` when `metersPerUnit === 1000`. Passing that result into a projected planar context with `metric: 'native'` throws “native metrics must use native or meters units.” Both functions are public. `core/contracts/spatial-frame.ts:87`, `:96`; `core/contracts/spatial-context.ts:145`; `core/contracts/index.ts:78`, `:137`.
- **Why it matters.** Following the helper’s name and documentation produces an invalid descriptor for a legitimate linear frame. Consumers must know which metadata to discard.
- **Proposed change.** Choose the supported planar unit policy in one place. If kilometre-native planar results are supported, accept and propagate them. Otherwise have the helper return `undefined` for that case and retain the original unit in frame metadata. Add a frame-to-context round-trip test. **S.** The first option widens accepted input.
- **Confidence.** High.

## Cross-cutting patterns

### What each contract section actually enforces

“Partial” includes the named structural specs where indicated. “Unchecked” means I found no general enforcement in the audited checker and structural suite. Individual contributors often have relevant tests.

| § | Status | What is enforced or missing |
|---|---|---|
| 1. Shape | Partial | Producer declaration, multiple exported classes and five roots are checked. File/class correspondence, prop/output naming and recipe composition are incomplete. `scripts/check-contributor-contract.mjs:42`, `:1174`; `dev-docs/contributor-contract.md:9`. |
| 2. Parallel construction | Partial | The checker examines recognized `createWGSLKernelNode` calls with statically resolved `invocationCount === 1`. A loop over all rows inside **each** of many invocations escapes this heuristic; a bounded loop in a one-invocation kernel can trigger it. The catalogue separately checks reference-tier limit metadata. `scripts/check-contributor-contract.mjs:1082`, `:1113`; `modules/analysis/test/gpu-spatial-analysis/ontology/analysis-operator-catalog.node.spec.ts:118`. |
| 3. Indexed candidates | Unchecked | No general check establishes a default index, cap or opt-in for exhaustive search. This needs contributor-specific algorithm and boundary evidence. `dev-docs/contributor-contract.md:19`; `scripts/check-contributor-contract.mjs:42`. |
| 4. Iteration budget | Partial | Selected fixtures compare actual node count with the estimate. General estimate presence, gate applicability and limit-result publication remain open. Fixed-depth pointer jumping and convergence-driven rounds should have different requirements. `core-test/fitting/fitting-kernels.node.spec.ts:179`; `core/ring-assembly/gpu-segment-ring-assembly.ts:1250`; `dev-docs/contributor-contract.md:25`. |
| 5. Per-frame parameters | Partial | Vocabulary checks some parameter names. No general check links a field’s mutability to its packer, binding and pipeline stability. `core-test/analysis-vocabulary.node.spec.ts:63`; `dev-docs/contributor-contract.md:31`. |
| 6. Declared context | Partial | Spatial and temporal resolvers enforce support when called. There is no general proof that every coordinate/time consumer calls them or that every metric uses the declared context. `core/contracts/spatial-context.ts:260`; `core/contracts/temporal-context.ts:113`; `dev-docs/contributor-contract.md:36`. |
| 7. No-data | Partial | The vocabulary spec catches selected row-filter names. It cannot establish polarity, propagation of band validity, or missing-key semantics. Its suggestion says `outputMask`, while the contract prescribes `validity` for output validity. Separate **selection** from **validity** explicitly. `core-test/analysis-vocabulary.node.spec.ts:87`; `dev-docs/contributor-contract.md:42`. |
| 8. Output layout | Partial | Geometry and pair validators check selected formats and lengths. Stage meaning, usable prefix, provenance and exactness need behavioral fixtures. `core/contracts/ports.ts:215`, `:257`; `dev-docs/contributor-contract.md:48`. |
| 9. Named status | Partial | Status placement is checked for selected type shapes. The port validator accepts any `uint32` view with `length >= 1` despite its “one uint32 row” diagnostic. No general checker establishes bit semantics or bans readback. `core-test/analysis-vocabulary.node.spec.ts:53`; `core/contracts/status.ts:117`; `dev-docs/contributor-contract.md:54`. |
| 10. Named limits | Partial | Bare assertions and preflight presence/order have rules. Limit arithmetic, exact allocation coverage and message quality still depend on review. The isochrones exception documents ownership rather than the technical safety of the allocation. `scripts/check-contributor-contract.mjs:775`; `scripts/contributor-contract-baseline.json:193`; `dev-docs/contributor-contract.md:62`. |
| 11. Shared numerics | Partial | Kernel audit covers selected compiler-sensitive idioms, declarations and resource limits. Equivalent solver, distance, RNG and search implementations require semantic review. `core-test/kernel-audit.node.spec.ts:274`; `core-test/kernel-audit/kernel-audit-wgsl.ts:880`; `dev-docs/contributor-contract.md:69`. |
| 12. Registry | Partial | The CLI searches the ontology source for a name, so a comment can satisfy it. The catalogue spec performs the stronger runtime export-to-record and record-to-export checks. Heading coverage has the gap in CORE-11. `scripts/check-contributor-contract.mjs:88`, `:1194`; `modules/analysis/test/gpu-spatial-analysis/ontology/analysis-operator-catalog.node.spec.ts:55`, `:65`. |
| 13. Public surface | Partial | Internal exports and luma re-exports are strict rules, and upstream-first passed. The checker does not enforce the policy that domain entries own contributors; the core barrel intentionally exports fitting contributors. `scripts/check-contributor-contract.mjs:56`; `core/index.ts:55`; `dev-docs/contributor-contract.md:85`. |
| 14. Operations and twins | Partial | The variogram walkthrough and CPU/GPU comparison provide good examples. There is no general check for interchangeable props, independent oracle, documented differences or complete output comparison. `core-test/operation-forms-docs.node.spec.ts:36`; `core-test/mixed-execution/variogram-model-fit-operation.spec.ts:84`; `dev-docs/contributor-contract.md:93`. |

### Consolidation opportunities

Across the seven domain families, a case-insensitive WGSL declaration-name scan found **12 lower/upper-bound declarations in 10 files**, **81 distance-named functions in 44**, and **4 solver-name matches in 4** for `solveMatrix|solveSystem|cholesky|factorInformation`. There were **21 calls in 13 files** to `getBinarySearchWGSL`, **142 `new GPUScan` sites in 95 files**, and **no copies of the core lowbias32 `0x7feb352d` constant**. A `function\s+validate\w*\s*[<(]` scan found **350 local validator definitions in 236 files**. These are candidate inventories. A cell-step distance, an indexed lookup through another column and a contiguous ordered-key search have different contracts. `core/kernels/binary-search-wgsl.ts:23`, `:29`; `core/kernels/hash-uint32.ts:12`; `modules/analysis/src/gpu-spatial-analysis/line-simplification/line-simplification-kernels.ts:799`; `modules/analysis/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics.ts:723`; `modules/analysis/src/gpu-spatial-analysis/geometry-edit/gpu-geometry-cleanup.ts:509`.

One consolidation is immediately actionable. Core itself contains two CPU mirrors of `geodesicLongitudeDelta`, including the same seam constants `131072` and `47185920`. Only one bearing implementation has the explicit near-antipodal branch. Have trajectory distance call the shared longitude-delta routine first; compare antipodal and seam fixtures before sharing the central-angle path. That is narrower and safer than a bulk distance-function codemod. `core/operations/cpu-trajectory-distance.ts:12`, `:48`; `core/kernels/cpu-geodesic.ts:18`, `:56`.

The highest-value public/private decision is `validateSinglePackedView`. It has **1,790 direct calls in 435 files**, names the contributor’s prop, and delegates the packed-layout check to published luma.gl. Exposing this small adapter would let external authors follow the same convention as internal contributors. **KNOWN(STD-09)** is marked low priority despite that adoption. `core/kernels/gpu-contributor-utils.ts:206`, `:216`; `core/index.ts:27`; `dev-docs/roadmap.md:421`.

## Keep

- **The split between host planning and graph declaration.** `planGraphChunkDispatch` checks row topology without allocating. `createGraphChunkGather` validates topology, copy usage, byte range and budget before declaring scratch. Its documentation tells callers exactly which input is borrowed, which route copies, and who owns the result. This is a good template for new helpers. `core/kernels/graph-chunk-dispatch.ts:54`; `core/kernels/graph-chunk-gather.ts:29`, `:103`, `:118`.
- **The semantic model-result and inference types.** Named residual-variance conventions, tested-family descriptions and coefficient roles save every consumer from decoding estimator-specific numeric slots. Fix the AICc edge centrally and retain the shared result builder. `core/contracts/model-result.ts:41`, `:97`, `:172`; `core/contracts/inference.ts:30`.
- **The temporal support contract.** Unit, epoch, representation and sampling axis are separate, and an operation can declare exactly what it accepts. This deserves a direct example in §6. `core/contracts/temporal-context.ts:41`, `:57`, `:113`.
- **The existing resource-lifecycle fixtures.** They compare buffer and texture deltas separately and exercise repeated construction and cleanup. **KNOWN(LUM-30)** is stale: the roadmap says “No spec checks for leaks,” while the scoped spec already calls `validateNoGPUResourceLeaks` after each repetition. Update that row to identify families still lacking coverage. `testing/gpu-resource-counts.ts:15`, `:24`; `core-test/gpu-resource-counts.spec.ts:43`; `dev-docs/roadmap.md:447`.
- **Shared geometry below the domain entries.** Raster and spatial code import triangulation; raster, spatial and network import ring assembly, including type-only network uses. Preserve that dependency direction. CPU network routing, by contrast, has six direct importing files, all in `gpu-network`, and is a straightforward move to that family. `modules/analysis/src/raster-analysis/fractional-coverage/raster-fractional-coverage-index.ts:18`; `modules/analysis/src/raster-analysis/isolines/gpu-isoband-rings.ts:19`; `modules/analysis/src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay.ts:53`; `modules/analysis/src/gpu-network/network-isochrones/gpu-network-isochrones.ts:38`; `modules/analysis/src/gpu-network/network-reachability/cpu-network-reachability-operation.ts:23`.

## Future

1. **Publish a short contract v2 organized around the lifecycle.** Specify construction and immutable shape, device-dependent preflight, graph declaration, per-encoding inputs, readback, and destruction. Include an ownership table for borrowed, graph-owned and contributor-owned resources. The existing operation supports one live prepared execution and destroys its last owned contributor separately; surface that restriction where authors choose an operation form. Move dated measurements and compiler investigations to evidence pages. The trade-off is a small amount of explicit metadata in exchange for fewer lifecycle surprises. `core/operations/gpu-contributor-operation.ts:88`, `:175`, `:187`; `dev-docs/contributor-contract.md:93`.
2. **Standardize result interpretation before standardizing all output shapes.** Define primary row unit, usable prefix, per-stage capacity, exact versus saturated requirement, and a pure decoder for host-read status. Leave readback scheduling with the caller. Compact IDs, packed pairs, fixed-stride geometry and scalar model results can then share a vocabulary without pretending to share a layout. The main migration cost is legacy aggregate overflow. `core/kernels/gpu-contributor-types.ts:22`; `core/contracts/ports.ts:179`; `core/contracts/capacity-planning.ts:57`.
3. **Build one descriptor-driven conformance inventory.** Start with static contributor ID, public entry, kind, topology props, runtime parameter schema, primary output unit, tier and audit fixture. Generate registry/heading checks and cheap construction fixtures from it. Keep numerical oracles and algorithmic complexity review handwritten. Pilot on the four fitting contributors before extending to hundreds of classes. `core/kernels/gpu-fitting-minibatch-order.ts:23`, `:32`, `:61`; `core-test/kernel-audit/analysis-contributors.ts:29`; `core-test/kernel-audit-fitting-kernels.node.spec.ts:9`.
4. **Give core three clear owners.** Keep contracts and analysis semantics in the authoring surface. Keep shared geometry in a named internal geometry layer reachable by raster and spatial entries. Move single-family CPU algorithms back to their families. For upstream, propose checked packed-view slicing, graph-native parameter buffers and a supported way to collect nodes from `addToGraph`; the present bridge temporarily replaces three graph methods with `Object.defineProperty`. Adopt upstream implementations only after they ship in a public entry. The trade-off is import churn internally, which stable public barrels can absorb. `core/kernels/gpu-contributor-utils.ts:414`, `:427`; `core/kernels/graph-chunk-groups.ts:129`; `AGENTS.md:22`.

## Top 5

1. Enforce parent-view bounds and packed layout in `sliceGraphChunkGroupView`.
2. Make Arrow exact and round policies compare against the original Decimal or Int64 value.
3. Make parameter schemas and buffers agree on the representation actually written, covering both CORE-03 and CORE-04.
4. Guard undefined AICc in the shared result builder.
5. Separate candidate loss from final-output overflow in the shared publisher, then migrate its callers.

## Questions for the lead

- May a compatibility release change the meaning of the existing aggregate `overflow` bit, or should the migration introduce a separate aggregate field before changing it?
- Is Arrow intended to be optional for third-party contributor authors? That determines whether to add an Arrow-free authoring entry or make the dependency mandatory.
- Should `analysis-core` be a supported home for reusable contributors such as fitting kernels, as the current exports imply? An explicit decision would let §13, the catalogue and the audit enumerate the same public surface.