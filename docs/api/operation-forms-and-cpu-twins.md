---
title: Operation forms and CPU twins
description: Run a CPU variogram fit or a complete GPU point-density computation, then choose a contributor, graph operation or CPU twin.
---

# Operation forms and CPU twins

An analysis can expose three forms with the same output layout. Choose the form according to where
your data lives and who should manage execution.

| Form | Inputs and execution | When to use it |
| --- | --- | --- |
| GPU contributor, such as `GPUVariogramModelFit` | Takes caller-owned `GraphDataView`s and produces command nodes for a `GPUCommandGraph`. | Data is already on the GPU and you own allocation, submission and readback. |
| GPU operation, such as `GPUVariogramModelFitOperation` | Takes names of `GPUGraph` values. Extends `GPUContributorOperation`, which resolves those names to views and builds the contributor. | You want the graph to plan intermediate vectors, including a contributor between CPU stages. It runs the same GPU algorithm. |
| CPU twin, such as `CPUVariogramModelFitOperation` | Extends the published `GPUCPUOperation`. Takes the same operation props and reads and writes the same named values. | The work is small, serial or sensitive to precision, and moving its vectors across the CPU boundary is worthwhile. |

Operation forms and twins are exported from their contributor's domain entry point. The bridge,
`GPU_CONTRIBUTOR_OPERATION_LOWERING` and `getCPUOperationFloat32Rows` /
`getCPUOperationInt32Rows` / `getCPUOperationUint32Rows` come from
`@luma-contributors/analysis/analysis-core`. luma.gl owns `GPUGraph`, `GPUCPUOperation`,
`CPUVector` and `GPUVector`; see its [GPU Core documentation](https://luma.gl/docs/api-reference/experimental/gpu-core).
When an existing class already uses the operation name, the wrapper is named
`GPU<Name>GraphOperation`, for example `GPULocationAllocationGraphOperation`.

Family contracts describe the named inputs, outputs and CPU/GPU differences for
[movement coverage](./gpu-spatial-analysis/movement-coverage.md#operation-forms),
[home range and recursion](./gpu-spatial-analysis/home-ranges-and-recursions.md#operation-forms),
[HEALPix point-to-cell](./gpu-spatial-analysis/healpix-cells-dggs-json.md#graph-operation-forms),
[cell indexing and measures](./gpu-spatial-analysis/operators-cells.md#operation-forms-indexing-and-measures),
[S2 covers](./gpu-spatial-analysis/global-coordinates.md#operation-forms-s2-covers-and-intrinsic-arc-overlay),
[arc overlay](./gpu-spatial-analysis/global-coordinates.md#operation-forms-s2-covers-and-intrinsic-arc-overlay),
[trajectory cleaning](./gpu-spatial-analysis/movement-semantics.md#operation-forms-cleaning),
[stays, segments and locations](./gpu-spatial-analysis/movement-semantics.md#operation-forms-stays-triplegs-and-locations),
and [track similarity](./gpu-spatial-analysis/track-similarity-measures.md#operation-forms).

## Choosing CPU or GPU

Quantile and Diggle operations require a positive integer `rowCount` at construction. GPU lowering checks the resolved values or positions against that exact population, just as CPU execution does; a longer input cannot silently add rows when optional outputs are omitted.

Both Diggle wrappers capture spatial and temporal context descriptors at construction, including frame origin/units/extent and calendar-axis steps. Later edits to the caller's encoding, epoch, frame or axis cannot change validation or execution on a subsequent run.

The region graph, coregionalization and variogram fit operations capture shape, options, model order and
port names at construction. Reusing an operation preserves that contract even if the caller
later edits its props, model array or nested output maps. Input graph values still change
per run; construct a new operation when changing its configuration or names.

Both multiple-testing adjustment wrappers snapshot `rowCount`, `method`, `minimumFamilySize`
and the declared input and output names at construction, including nested output names.
`CPUMultipleTestingAdjustmentOperation` and `GPUMultipleTestingAdjustmentOperation` therefore
retain those settings after the caller edits its props. Construct a new operation to change the
method, sizes or destinations. Input p-value contents remain per-frame and can change between runs.

Region-graph wrappers copy grid metadata, including its affine array, and nested output/status names at construction. The direct `GPURasterRegionGraph` contributor likewise captures its route, budget, threshold, grid and output map while retaining the bound graph views. Editing the caller's construction objects afterward cannot change the declared contract, merge-node budget or output bindings.

Region CRS snapshots read the declared `authority`, `wellKnownText` and `projectionJson` fields as plain data. Transparent proxy descriptors are accepted; nested projection arrays and records are copied, so later edits to their backing objects do not change the captured metadata.

Many CPU twins compute in float64 and round once to the declared output format; consult the
producer's arithmetic contract. CPU network routing currently has these producer-specific rules:

| Public CPU form | Retained default | Selectable precision |
| --- | --- | --- |
| `computeNetworkCostTableOnCPU` / `CPUNetworkCostTableOperation` | `'float32-per-edge'` before cutoff and label comparisons. | Pure input `costPrecision` / operation `props.costPrecision`: `'float32-per-edge'` or `'float64'`. |
| `computeNetworkCostMatrixOnCPU` / `CPUNetworkCostMatrixOperation` | `'float64'`, then Float32 output. | Same two values for completed and bounded searches. |
| `computeNetworkReachabilityOnCPU` / `CPUNetworkReachabilityOperation` | `'float64'`, then Float32 output. | Same two values, including predecessor tight-edge tests. |
| `computeNetworkServiceAreasOnCPU` / `CPUNetworkServiceAreasOperation` | `'float64'`, then Float32 output. | Same two values, including facility-label and predecessor comparisons. |

All four pure CPU inputs accept `costPrecision`; their named CPU operation forms capture the same option from construction props. Unknown values are rejected. GPU operations accept only `'float32-per-edge'`. Precision changes addition and cutoff membership, not units or GPU scheduling. Import these public functions and operations from `@luma-contributors/analysis/gpu-network`.

Only `undefined` selects a default; unknown values, including JavaScript `null`, are rejected.
Sparse float64 tables omit costs that are not representable in float32 and clear `complete`,
even when the search converges and storage fits.

For the three-node chain `0 → 1 → 2`, stored Float32 edge weights `[0.1, 0.4]` and inclusive
cutoff `0.5`, `'float32-per-edge'` admits node 2 at cost `0.5`; `'float64'` excludes it because
the exact binary64 sum of the stored edges is `0.5000000074505806`. Thus the default sparse table
includes node 2, while the default dense matrix and reachability exclude it. The distinction holds
with completed searches and explicit sufficient iteration budgets. GPU forms use Float32 arithmetic;
selecting CPU precision does not change GPU arithmetic or scheduling. See
[routing arithmetic and cutoffs](./gpu-network/operators.md#gpunetworkcosttable).

An exported pure
`compute<Name>OnCPU` function also works with arrays outside a graph. Prefer that function when
the data is already on the CPU and there are no GPU stages to compose. A reference-tier contributor
with a twin should generally use that twin; check the family's documentation for its limits.

Keep large parallel workloads on the GPU. A mixed graph reads a GPU value back when a CPU stage
first needs it, and uploads a CPU result when a later GPU stage needs it. Those transfers can
outweigh the computation. Historical measurements on the test GPU with luma.gl 10.0.0-alpha.12
put one readback/identity/upload boundary at about 0.7 ms for 1,000 `float32x2` rows, 11 ms for
100,000, and 90 ms for one million. In the same experiment a 20-bin variogram fit between CPU
stages took about 0.5 ms with the twin versus 1.6–2.5 ms with the GPU contributor. These are
measurements of that setup, not a portable crossover rule or an automatic choice made by the API.

The operation bridge supports one packed chunk per value. Mixed execution uses `float32`,
`sint32` or `uint32` vectors with one to four components; it does not support textures, state
vectors, or CPU input null bitmaps and list offsets. Each GPU operation in a mixed graph has
its own submission. Use asynchronous `compiled.run()`, rather than `prepare()`/`encode()`.
Mixed outputs belong to the invocation; caller-provided output targets are unsupported.

## Variogram fitting without a graph

In this checkout, after the repository's `nvm use` and `yarn install` setup, run
`node_modules/.bin/vitest run --project node --maxWorkers=2 modules/analysis/test/analysis-core/first-run-docs.node.spec.ts -t 'CPU variogram walkthrough'`.
This executes the first block below against workspace sources, prints the fitted model and checks
the expected parameters, without a GPU or a package build. The package is currently unpublished; until an alpha is released,
use the workspace for this first run.
In an installed alpha, use the same imports from your TypeScript application with the required
`apache-arrow` peer and luma.gl peers installed.

The empirical variogram step, [`GPUVariogram`](./gpu-dataframe-analysis.md), produces
mean lag distances, semivariances and pair counts. Once those bins have been read back, pass them
to `computeVariogramModelFitOnCPU`; skip fitting if the empirical step's `output.status.overflow`
is nonzero. The example below creates 20 synthetic bins directly, so no dataset or GPU is needed
for this first step. It fits three families to an exponential model with nugget 0.2, partial sill
1.5 and effective range 2.2.

```ts
import {
  computeVariogramModelFitOnCPU,
  fitVariogramModel,
  fitVariogramModelWithDiagnostics,
  type VariogramModelFitDiagnostics
} from '@luma-contributors/analysis/dataframe-analysis';

const models = ['gaussian', 'exponential', 'spherical'] as const;
const distances = Float32Array.from({length: 20}, (_, index) => (index + 1) * 0.2);
const bins = {
  distances,
  semivariances: Float32Array.from(
    distances,
    distance => 0.2 + 1.5 * (1 - Math.exp((-3 * distance) / 2.2))
  ),
  pairCounts: Uint32Array.from({length: 20}, (_, index) => 10 + index)
};
const cpuFit = computeVariogramModelFitOnCPU(bins, {models, weighting: 'pairs'});
// Row 1 is exponential because output rows follow models.
const exponentialParameters = cpuFit.parameters.subarray(4, 8);
console.log({
  model: 'exponential',
  nugget: exponentialParameters[0],
  partialSill: exponentialParameters[1],
  effectiveRange: exponentialParameters[2],
  status: cpuFit.status[1]
});
// Approximately {model: 'exponential', nugget: 0.2, partialSill: 1.5,
//                effectiveRange: 2.2, status: 0}.

const fitOptions = {model: 'exponential', weighting: 'pairs'} as const;
const coefficients = fitVariogramModel(bins, fitOptions); // No flag: coefficients directly.
const diagnosticFit: VariogramModelFitDiagnostics = fitVariogramModelWithDiagnostics(bins, fitOptions);
// The dedicated helper always returns diagnostics, with coefficients in .model.
console.log(coefficients.nugget, diagnosticFit.model.nugget);
console.log(diagnosticFit.atBoundary, diagnosticFit.logRangeBracket, diagnosticFit.converged);
// Both nuggets are approximately 0.2; atBoundary is false and converged is true.
```

This function calls the existing `fitVariogramModel` CPU fitter. Use `fitVariogramModel` directly
when you want a model object for one family; `computeVariogramModelFitOnCPU` returns the typed-array
rows used by the operation contract. `fitVariogramModel(input, options)` returns coefficients.
For boundary and convergence diagnostics, the dedicated function is
`fitVariogramModelWithDiagnostics(input, options)`; read its `.model` for coefficients and
`.atBoundary`, `.logRangeBracket` and `.converged` for search diagnostics. The dedicated helper
and its `VariogramModelFitDiagnostics` result type are publicly exported from
`@luma-contributors/analysis/dataframe-analysis`.

## The same fit inside a graph

Pass `bins` and `models` from the first example to the function below. It takes an existing
WebGPU `Device` and returns CPU output vectors and the selected model name. Calling it with `'gpu'` runs the GPU fit followed
by a CPU selection stage; `'cpu'` replaces just the fit with its twin. The same
`VariogramModelFitOperationProps` object builds either one. Even the CPU-only choice here needs a
WebGPU device because it uses `GPUGraph`; the pure function above has no device requirement.

```ts
import type {Device} from '@luma.gl/core';
import {GPUCPUOperation, GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {CPUData, CPUVector} from '@luma.gl/graph/gpu-data';
import {
  GPU_CONTRIBUTOR_OPERATION_LOWERING,
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '@luma-contributors/analysis/analysis-core';
import {
  CPUVariogramModelFitOperation,
  GPUVariogramModelFitOperation,
  type VariogramModelType,
  type VariogramModelFitOperationProps
} from '@luma-contributors/analysis/dataframe-analysis';

export async function fitVariogramInGraph(
  device: Device,
  bins: {distances: Float32Array; semivariances: Float32Array; pairCounts: Uint32Array},
  models: readonly VariogramModelType[],
  location: 'cpu' | 'gpu' = 'cpu'
) {
  const graph = new GPUGraph({id: 'variogram-example'});
  const inputs: Record<string, CPUVector> = {};
  const binValues: Record<string, Float32Array | Uint32Array> = bins;
  for (const [name, values] of Object.entries(binValues)) {
    const format = values instanceof Uint32Array ? 'uint32' : 'float32';
    graph.addInput(name, gpuGraph.vector(format, {length: values.length}));
    inputs[name] = new CPUVector({
      format,
      data: [new CPUData({buffer: values, format, length: values.length})]
    });
  }
  const props: VariogramModelFitOperationProps = {
    id: 'fit',
    distances: 'distances',
    semivariances: 'semivariances',
    pairCounts: 'pairCounts',
    models,
    weighting: 'pairs',
    output: {parameters: 'parameters', diagnostics: 'diagnostics', status: 'status'}
  };
  const fit = location === 'gpu'
    ? new GPUVariogramModelFitOperation(props)
    : new CPUVariogramModelFitOperation(props);
  graph.add(fit);
  graph.add(new GPUCPUOperation({
    id: 'select-model',
    inputs: {parameters: 'parameters', status: 'status'},
    outputs: {bestModel: gpuGraph.vector('uint32', {length: 1})},
    execute({inputs}) {
      const parameters = getCPUOperationFloat32Rows(
        'select-model', inputs, 'parameters', 'float32x4'
      );
      const status = getCPUOperationUint32Rows('select-model', inputs, 'status');
      let bestModel = 0xffffffff;
      let bestResidual = Number.POSITIVE_INFINITY;
      for (let row = 0; row < models.length; row++) {
        const residual = parameters[row * 4 + 3];
        // This example accepts only identified interior fits; status 3 needs review.
        if (status[row] === 0 && Number.isFinite(residual) && residual < bestResidual) {
          bestModel = row;
          bestResidual = residual;
        }
      }
      // Equal residuals keep the first model; no acceptable fit returns the missing-key code.
      return {bestModel: Uint32Array.of(bestModel)};
    }
  }));
  const compiled = graph.compile<Record<string, CPUVector>>(device, {
    outputs: ['bestModel', 'parameters', 'diagnostics', 'status'],
    outputLocations: {bestModel: 'cpu', parameters: 'cpu', diagnostics: 'cpu', status: 'cpu'},
    lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
  });
  try {
    const outputs = await compiled.run({inputs});
    const selected = outputs['bestModel'].data[0];
    const bestModel = new Uint32Array(selected.buffer, selected.byteOffset, 1)[0];
    return {outputs, model: bestModel === 0xffffffff ? undefined : models[bestModel]};
  } finally {
    compiled.destroy();
    // The GPU wrapper may retain its last contributor's resources after a run.
    if (fit instanceof GPUVariogramModelFitOperation) fit.destroy();
  }
}
```

`await fitVariogramInGraph(device, bins, models, 'cpu')` returns `model: 'exponential'` on these bins;
`'gpu'` should select the same family within the fitting tolerance below. Returned `outputs` are
`CPUVector`s, with data available through their `CPUData` chunks; the selected scalar above reads
its chunk's buffer at `byteOffset`. They remain usable after compiled-graph cleanup. If outputs
are requested on the GPU instead, the caller must destroy the returned GPU vectors when finished.

In an application, a preceding graph stage can write the three named bin values instead of
supplying CPU inputs. A following GPU stage can consume the fit values directly; the executor
uploads a CPU twin's result only if such a consumer needs it. Include
`GPU_CONTRIBUTOR_OPERATION_LOWERING` whenever any contributor operation is present. It resolves
the named ports and delegates to the existing contributor; it does not implement a second GPU fit.

## Variogram ports, limits and numerical differences

The three input vectors have equal lengths: `distances` and `semivariances` are `float32`;
`pairCounts` is `uint32`. Distances and fitted ranges share units. A bin with zero pairs, a
nonfinite lag/semivariance or a nonpositive lag is skipped. Fitting needs at least two usable bins
at distinct positive lags. An all-zero pair-count input produces status 1 and NaN parameter and
diagnostic rows in both forms.

| Output | Format and length | Components |
| --- | --- | --- |
| `parameters` | `float32x4`, one row per model | Nugget, partial sill, effective range, weighted squared residual. |
| `diagnostics` | `float32x4`, one row per model | Usable bins, unweighted squared residual, residual AIC, final relative log-range bracket. |
| `status` | `uint32`, one row per model | 0: interior fit; 1: insufficient bins; 2: numerical failure; 3: search boundary or budget result in either form. |

Both classes default to spherical, exponential and Gaussian rows, in that order; the example
explicitly changes their order. They also accept `matern-3/2` and `matern-5/2`, with one to five
model rows. Weighting is `'cressie'` by default, or `'pairs'` / `'none'`. All props are compile-time;
input vector contents may change between runs without rebuilding the operation. For the
[fitting objective and effective-range convention](./gpu-spatial-analysis/interpolation-expansion.md#resident-variogram-fitting),
see the resident variogram fitting contract. AIC is an unweighted residual diagnostic for comparing
fits to the same bins, not a spatial likelihood. The example selects by weighted residual using the
same bins and weighting for each family.

The interchangeable props and layouts do not promise bit identity:

- Both forms honor explicit `rangeSteps` (3–1024) and `maximumIterations` (0–64) for every
  weighting pass, including zero refinements. Both default to 200 ladder points; GPU defaults
  to 32 refinements and CPU retains its 60-step default when omitted. The pure CPU fitter and
  `computeVariogramModelFitOnCPU` accept the same search options. CPU uses float64 internally
  and rounds once to the output format.
- The CPU can fit some bins that the GPU refuses because its float32 objective cannot be evaluated.
  It still returns status 2 and NaN rows when fitted outputs cannot be represented as finite float32.
- Both search `[minimum positive lag / 10, 3 * maximum lag]` and assign status 3 when the best
  ladder point is an endpoint or the final log-range bracket is wider than `1e-5`. With the
  walkthrough's bins and 200 ladder points, `maximumIterations: 0` leaves a bracket of about
  `0.06429` and status 3 in either form. The selection stage above suppresses such rows.
  Treat a budget result as a candidate needing review, rather than an identified interior fit.
- The CPU bracket diagnostic measures the work actually performed. GPU may stop refinement at
  bracket `1e-5`; CPU continues up to its budget in float64. Cressie weights use the
  smallest float64 denominator floor on CPU versus `1e-20` on GPU, so very small semivariances
  can give different fits.

The existing mixed-execution parity spec checks ordinary synthetic bins with
`abs(gpu - cpu) / max(1, abs(cpu)) <= 3e-4` for parameters and diagnostics, with exact status
agreement on that case. This is a tested fitting tolerance, not a bound for every input. It treats
the exact-fit AIC separately: AIC takes the logarithm of residual rounding noise, so the CPU and GPU
values need not agree; both residuals must be below `1e-10` per bin. Check statuses before using a
row and avoid ranking nearly indistinguishable fits by tiny residual differences.

For repeated GPU runs, reuse a compiled graph sequentially. A contributor operation owns its last
resource-owning contributor and releases it when the next run replaces it. Keep each operation
instance with one compiled graph and one prepared execution at a time. Destroy the compiled graph
first, then call the GPU operation's `destroy()` to release its final contributor.

## Point density from device to readback

Run this in a browser TypeScript entry point served on localhost or HTTPS with WebGPU enabled
(for example, a Vite application in this workspace). It imports only published luma.gl and analysis
entry points; the WebGPU adapter comes from the public `@luma.gl/core/webgpu` entry.
Paste the complete block into that entry point. No existing device, buffers, graph or dataset is required.

For a minimal workspace browser page, save the block as `first-density.ts` in the repository
root, add an `index.html` containing `<script type="module" src="/first-density.ts"></script>`,
and run `node_modules/.bin/vite --host localhost`. Open the printed localhost URL and the browser
console. Package-name imports use the package build in this setup; run `yarn build` first if
`modules/analysis/dist` is absent or stale. The source-backed headless regression runs the same
program without relying on a package build.

Five positions are binned into a 2 × 2 grid over `[0, 0, 2, 2]`. Four are inside the bounds;
`[3, 3]` is outside. Each output row is one cell, with X varying fastest, starting at the minimum Y.
The expected counts are `[2, 0, 1, 1]`, and the grid overflow flag is `0`.

```ts
import {Buffer, luma} from '@luma.gl/core';
import {webgpuAdapter} from '@luma.gl/core/webgpu';
import {GPUCommandGraph} from '@luma.gl/graph';
import {GPUPointDensity} from '@luma-contributors/analysis/gpu-spatial-analysis';

export async function runPointDensityExample() {
  const device = await luma.createDevice({
    type: 'webgpu',
    adapters: [webgpuAdapter]
  });
  const buffers: Buffer[] = [];
  let compiled: Awaited<ReturnType<GPUCommandGraph['compileAsync']>> | undefined;
  try {
    const graph = new GPUCommandGraph(device, {id: 'first-density'});
    const createBuffer = (id: string, data: Float32Array | Uint32Array) => {
      const buffer = device.createBuffer({
        id,
        data,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      const handle = graph.importBuffer(
        {id, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      return {buffer, handle};
    };
    const positions = graph.createDataView(
      createBuffer(
        'positions',
        Float32Array.of(0.25, 0.25, 0.75, 0.75, 0.25, 1.25, 1.25, 1.25, 3, 3)
      ).handle,
      {format: 'float32x2', length: 5}
    );
    const bounds = graph.createDataView(createBuffer('bounds', Float32Array.of(0, 0, 2, 2)).handle, {
      format: 'float32',
      length: 4
    });
    const valuesAllocation = createBuffer('values', new Float32Array(4));
    const overflowAllocation = createBuffer('overflow', new Uint32Array(1));
    const values = graph.createDataView(valuesAllocation.handle, {format: 'float32', length: 4});
    const overflow = graph.createDataView(overflowAllocation.handle, {format: 'uint32', length: 1});
    graph.add(
      new GPUPointDensity({
        positions,
        bounds,
        gridSize: [2, 2],
        statistic: 'count',
        output: {values, status: {overflow}}
      })
    );
    compiled = await graph.compileAsync();
    const commandEncoder = device.createCommandEncoder();
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const valueBytes = await valuesAllocation.buffer.readAsync();
    const overflowBytes = await overflowAllocation.buffer.readAsync();
    const result = {
      values: new Float32Array(valueBytes.buffer, valueBytes.byteOffset, 4).slice(),
      overflow: new Uint32Array(overflowBytes.buffer, overflowBytes.byteOffset, 1)[0]
    };
    if (result.overflow !== 0) throw new Error('Point density output is incomplete');
    return result;
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
}

const density = await runPointDensityExample();
console.log(Array.from(density.values), density.overflow);
```

Named allocations retain each output buffer for readback; `buffers` is used only for cleanup.
Each raw buffer is imported exactly once and then given a typed view. `float32x2` length counts
positions, so five positions occupy ten floats. `values` contains four `float32` rows, while
`output.status.overflow` is a separate one-row `uint32` view. Storage, upload and readback need
`STORAGE | COPY_DST | COPY_SRC` usage here. If your application already owns `GPUData`, use
`graph.importGPUData(id, data)` instead of importing its underlying buffer again.

`compileAsync()` creates the executable graph. `encode()` records commands; `device.submit()`
executes them. `readAsync()` waits for the copied result. Typed views honor the readback's
`byteOffset`; the returned values are copied so they survive cleanup. Imported buffers remain
caller-owned: destroy the compiled graph before the buffers, then release a device you created.
In an application that keeps its device, keep that device alive and release only this computation's
resources. Updating bounds or positions with `buffer.write()` before another submission does not
require recompilation; changing view lengths or grid dimensions does.

For other families, keep this allocation/import/compile/submit/readback/cleanup sequence and use
the family quick start's input formats, parameter packer and output layout. Allocate separate
buffers for writable outputs, including status scalars. Read completeness status together with
the result: propagate incomplete work, suppress dependent results, or refuse/fall back.

