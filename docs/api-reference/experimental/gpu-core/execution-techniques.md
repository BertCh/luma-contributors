import {GPUCoreCompilerAnatomy} from '@site/src/components/docs/gpu-core-compiler-anatomy';

# GPU Command Graph Execution Techniques

## Overview

GPU Core compiles resource declarations and compute, copy, and render nodes into a reusable plan.
These techniques keep intermediate data on the GPU while making memory use and scheduling visible.

## Conceptual architecture and sample graph

```text
Application + reusable contributors
            ↓ resources, nodes, explicit dependencies
Compiler: validate → infer hazards → schedule → plan lifetimes → compile nodes
            ↓ immutable compiled graph
Encode with current parameters → caller submits → GPU executes
            ↓                                     ↓
CPU encoding statistics                     optional GPU timings
            └──────────── inspector and trace ─────┘
```

Explore a sample selection-and-render graph below. Change the compiler view to follow logical
resources, dependency edges, execution order, lifetimes, and physical allocation reuse. This is a
teaching model of compilation, not a live GPU timing capture.

<GPUCoreCompilerAnatomy />

## Resource management

Import long-lived buffers and textures whose ownership stays with the application. Declare
scratch resources as transients so the compiled graph can allocate and destroy them. Compatible
transients with non-overlapping scheduled lifetimes may share a physical allocation; logical
resource count therefore differs from physical allocation count.

Declare all uses and retain data for as long as a consumer needs it. Do not treat transient storage
as persistent history across encodings. Use [GPUTextureHistory](./gpu-texture-history) for rotating
texture state. Destroy compiled graphs when replacing them; imported resources remain caller-owned.

## Hazard scheduling

The compiler combines explicit dependencies with read-after-write, write-after-read, and
write-after-write hazards. Buffer hazards are conservative at whole-handle granularity; texture
hazards use overlapping mip, layer, and aspect ranges. Two differently named views do not make
underlying overlapping storage independent.

Independent nodes retain stable declaration order. A schedule provides dependency correctness;
it does not promise parallel hardware execution. Missing declarations cannot be repaired by the
scheduler. Declare the indirect argument buffer as a write in its producer and a read in its consumer,
as well as declaring the result buffer itself.

## Parallel compilation

Use `await graph.compileAsync()` during preparation. The compiler starts node compilation across
the scheduled nodes and collects asynchronous pipeline work before returning the compiled graph.
Nodes can provide `compileAsync`; nodes with only `compile` keep their synchronous compatibility path.
Actual compilation concurrency depends on the backend and browser.

Compile once for a fixed topology, formats, and capacities. Update parameters and compatible imported
resources while encoding. Recompile when the validated resource shape or topology changes.

## Pass coalescing

Consecutive compute nodes can share one physical compute pass. Copy and render nodes create pass
boundaries. Per-node GPU timestamp collection disables compute-pass coalescing for that encoding,
so a profiled capture may have more physical passes than normal execution.

Compare logical nodes, dispatches, and physical pass counts separately. Profiling every frame can
change the workload being measured; sample diagnostics at a deliberate cadence.

## Bounded work and asynchronous readback

Use GPU-written indirect arguments for data-dependent work counts. Keep a fixed capacity plus a
count and overflow diagnostics for variable-sized outputs. A zero indirect dispatch count can avoid
shader work without reading a predicate back to the CPU.

Use resumable execution and explicit budgets when an algorithm spans frames. A skipped writer does
not clear its output or automatically skip consumers: condition the dependent chain or ensure that
retained data is valid. For interaction and diagnostics, use a
[GPUReadbackRing](./gpu-readback-ring) to bound asynchronous readback rather than waiting for a
large result in the frame loop.

## Pipeline-constant kernels

Primitives that dispatch one WGSL kernel over graph views create that node with
`createGPUKernelNode`. Problem-size values (row counts, view offsets, grid sizes, literal domains
and bounds, dispatch shape) are passed as pipeline override constants instead of being written into
the shader text, so one compiled shader module serves every size and offset of the same kernel
shape. `getGPUKernelOverrideDeclarations(constants, types)` and
`getGPUKernelBindingDeclarations(bindings)` generate the `override` and `var<storage>` declarations
from the same lists passed to the node, so binding order cannot drift from the WGSL.

```ts
import {
  createGPUKernelNode,
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations
} from '@luma-contributors/graph/gpu-core';

const bindings = [
  {name: 'inputValues', view: input, access: 'read'},
  {name: 'outputValues', view: output, access: 'write'}
] as const;
const dispatchLayout = getBoundedDispatchLayout('MyOperation', input.length, 256, maximum);
const constants = {ELEMENT_COUNT: input.length, ...getBoundedDispatchOverrides(dispatchLayout)};

graph.add(createGPUKernelNode(graph, {
  id: 'my-operation',
  operation: 'MyOperation',
  source: `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(256) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(256).body}
  if (index < ELEMENT_COUNT) { outputValues[index] = inputValues[index]; }
}`,
  constants,
  bindings,
  dispatchLayout
}));
```

Binding `access` drives both the storage declaration and the graph's hazard analysis. Constants
default to `u32`; pass a type map to `getGPUKernelOverrideDeclarations` for `f32` or `i32` values.
The helper also caches bind groups for recurring graphs and annotates the node on recording graphs.

Some kernels still change their shader text with the problem shape: the number of rows each thread
accumulates (reduction), the subgroup strategy, literal histogram edges, the number of mask inputs,
the dimension of a spatial index, the workgroup width of a segmented sort, and the per-segment
descriptor table of a segmented sort.

## Measure the result

Track CPU encoding, GPU duration, transient bytes, allocation reuse, output counts, and overflow
separately. See [debugging GPU graphs](./debugging) for the GPU Graph and GPU Trace panels,
and [execution and composition](./concepts) for the detailed execution contract.
