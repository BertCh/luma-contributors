---
title: Operation forms and CPU twins
description: Choose a GPU contributor, a named-value graph operation or its CPU twin, with a complete variogram fitting example.
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

## Choosing CPU or GPU

A CPU twin computes in float64 and rounds once to the declared output format. Its exported pure
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

The empirical variogram step, [`GPUVariogram`](./gpu-dataframe-analysis.md), produces
mean lag distances, semivariances and pair counts. Once those bins have been read back, pass them
to `computeVariogramModelFitOnCPU`; skip fitting if the empirical step's `output.status.overflow`
is nonzero. The example below creates 20 synthetic bins directly, so no dataset or GPU is needed
for this first step. It fits three families to an exponential model with nugget 0.2, partial sill
1.5 and effective range 2.2.

```ts
import {computeVariogramModelFitOnCPU} from '@luma-contributors/analysis/dataframe-analysis';

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
// Approximately [0.2, 1.5, 2.2, 0], with cpuFit.status[1] === 0.
```

This function calls the existing `fitVariogramModel` CPU fitter. Use `fitVariogramModel` directly
when you want a model object for one family; `computeVariogramModelFitOnCPU` returns the typed-array
rows used by the operation contract.

## The same fit inside a graph

Continue with `bins` and `models` above. The function takes an existing WebGPU `Device` and returns
CPU output vectors and the selected model name. Calling it with `'gpu'` runs the GPU fit followed
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
  type VariogramModelFitOperationProps
} from '@luma-contributors/analysis/dataframe-analysis';

export async function fitVariogramInGraph(device: Device, location: 'cpu' | 'gpu' = 'cpu') {
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

`await fitVariogramInGraph(device, 'cpu')` returns `model: 'exponential'` on these bins;
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
| `status` | `uint32`, one row per model | 0: interior fit; 1: insufficient bins; 2: numerical failure; 3: search boundary or GPU budget result. |

Both classes default to spherical, exponential and Gaussian rows, in that order; the example
explicitly changes their order. They also accept `matern-3/2` and `matern-5/2`, with one to five
model rows. Weighting is `'cressie'` by default, or `'pairs'` / `'none'`. All props are compile-time;
input vector contents may change between runs without rebuilding the operation. For the
[fitting objective and effective-range convention](./gpu-spatial-analysis/interpolation-expansion.md#resident-variogram-fitting),
see the resident variogram fitting contract. AIC is an unweighted residual diagnostic for comparing
fits to the same bins, not a spatial likelihood. The example selects by weighted residual using the
same bins and weighting for each family.

The interchangeable props and layouts do not promise bit identity:

- The GPU uses float32 reductions and a configurable search: `rangeSteps` is 3–1024 (default 200),
  and `maximumIterations` is 0–64 (default 32). The CPU twin ignores these two GPU budgets and uses
  the CPU fitter's fixed 200-point ladder and 60 refinement steps in float64, rounding its output
  once to float32.
- The CPU can fit some bins that the GPU refuses because its float32 objective cannot be evaluated.
  It still returns status 2 and NaN rows when fitted outputs cannot be represented as finite float32.
- Both search `[minimum positive lag / 10, 3 * maximum lag]`. The CPU assigns status 3 when the
  fitted range lies in the first or last ladder interval. The GPU also uses status 3 for a search
  budget exit. Treat that row as a candidate needing review, rather than an identified interior fit.
- The CPU bracket diagnostic is an upper bound after its fixed 60 steps. Cressie weights use the
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
