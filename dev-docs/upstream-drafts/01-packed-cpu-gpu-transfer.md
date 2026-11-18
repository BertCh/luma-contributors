# Avoid per-row host copies for packed mixed CPU/GPU vectors

Status: draft, not filed. Target: `@luma.gl/graph` 10.0.0-alpha.14.

## Problem

`GPUGraph` can execute a `GPUCPUOperation` between GPU stages. For fixed-width numeric vectors whose
`byteStride` equals the format's byte length, crossing that boundary still performs host copies one
row at a time. This adds allocation and JavaScript work proportional to the row count even though
the bytes already form a contiguous range.

Verified in the installed release:

- `dist/gpu-core/gpu-cpu-execution.js`, `readCPUVector`: one readback per nonempty chunk, followed by
  a row loop calling `bytes.subarray(...)` and `packed.set(...)`.
- The same file, `getCPUView`: allocates a destination array and creates a `Uint8Array` per row before
  copying it. `ensureCPU` converts the readback to a `CPUVector` and then calls `getCPUView`, so the
  GPU-to-CPU route visits both copy loops.
- `ensureGPU` calls `ensureCPU` and constructs `GPUUploadOperation`; its constructor snapshots the
  chunks. CPU callback outputs also construct an upload operation for validation before later upload.
- `dist/gpu-data/cpu-gpu-vector-transfer.js`, `makeGPUVectorFromCPUVector`: the explicit physical-vector
  upload helper also copies each row into packed bytes. This is a separate route worth testing.

The transfer lowerer's `lowerGPUUpload` already emits one buffer copy for a packed destination. The
request concerns host packing/snapshot overhead, not a claim that every GPU upload emits one copy
command per row. Chunked numeric vectors are supported today; this is not a request to add chunks.

## Existing benchmark evidence

The dated CPU suitability measurement (repository `BertCh/luma-contributors`, commit
`92416edf4981c81f5fa83d4135ae4ed163041249`, file
`dev-docs/research/cpu-suitability-2026-11-11/README.md`, section 3)
reported one readback → identity CPU callback → upload boundary, best of four or five runs, in
headless Chromium on the development Mac's Apple GPU with **alpha.12**:

| `float32x2` rows | Payload in each direction | Boundary time |
| --- | --- | --- |
| 1,000 | 8,000 bytes | 0.7 ms |
| 100,000 | 800,000 bytes | 10.7 ms |
| 1,000,000 | 8,000,000 bytes | 90 ms |
| 4,000,000 | 32,000,000 bytes | 352 ms |

At one million rows this is about 11.25 ns per payload byte (or 5.625 ns per byte when counting both
directions). These are end-to-end historical boundary timings, not isolated JavaScript copy costs,
not alpha.14 results and not a prediction of the speedup. The original measurement script and full
browser/adapter version details were not retained in that report. The following benchmark is a new,
standalone reproduction protocol, not a claim to recreate that script exactly.

## Minimal reproduction and benchmark

Run this module with a WebGPU `Device`. Use alpha.14 for all three luma.gl packages. It times a
resident GPU input → identity CPU operation → GPU copy result. The final copy forces CPU output
back onto the GPU. Construction, input upload, warm-up, validation readback and destruction are
outside the recorded interval. `run()` resolves after GPU submission, so the metric is mixed-run
latency through submission, not a GPU-completion fence. Validation drains each run before the next.

```typescript
import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUGraph, GPUCPUOperation, GPUCopyOperation, gpuGraph
} from '@luma.gl/graph/gpu-core';
import {GPUVector} from '@luma.gl/graph/gpu-data';

export async function benchmarkPackedBoundary(device: Device) {
  const results = [];
  for (const rowCount of [1_000, 100_000, 1_000_000, 4_000_000]) {
    const values = Float32Array.from(
      {length: rowCount * 2}, (_, index) => (index % 1024) - 512
    );
    const buffer = device.createBuffer({
      data: values, usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    const input = new GPUVector({
      type: 'buffer', name: 'input', buffer, format: 'float32x2',
      length: rowCount, ownsBuffer: false
    });
    const graph = new GPUGraph();
    graph.addInput('input', gpuGraph.vector('float32x2', {length: rowCount}));
    graph.add(new GPUCPUOperation({
      id: 'identity', inputs: {rows: 'input'},
      outputs: {middle: gpuGraph.vector('float32x2', {length: rowCount})},
      execute: ({inputs}) => {
        const rows = inputs['rows'];
        if (typeof rows === 'number' || rows.chunks.length !== 1) {
          throw new Error('Expected one numeric chunk');
        }
        return {middle: rows.chunks[0].values};
      }
    }));
    graph.add(new GPUCopyOperation({input: 'middle', output: 'result'}));
    const compiled = graph.compile(device, {outputs: ['result']});
    const samples: number[] = [];
    const rawReadbackSamples: number[] = [];
    try {
      for (let iteration = 0; iteration < 7; iteration++) {
        // Write outside both timers; stale snapshots must fail on the second invocation.
        values[0] = iteration + 1;
        buffer.write(values.subarray(0, 1), 0);
        const rawStart = performance.now();
        await buffer.readAsync(0, values.byteLength);
        const rawMilliseconds = performance.now() - rawStart;
        const start = performance.now();
        const outputs = await compiled.run({inputs: {input}});
        const milliseconds = performance.now() - start;
        const result = outputs['result'];
        try {
          const chunk = result.data[0];
          const bytes = await chunk.buffer.readAsync(chunk.byteOffset, values.byteLength);
          const actual = new Float32Array(
            bytes.buffer, bytes.byteOffset, values.length
          );
          if (!actual.every((value, index) => value === values[index])) {
            throw new Error('Identity round trip changed a value');
          }
          if (iteration >= 2) {
            samples.push(milliseconds);
            rawReadbackSamples.push(rawMilliseconds);
          }
        } finally {
          result.destroy();
        }
      }
      samples.sort((left, right) => left - right);
      rawReadbackSamples.sort((left, right) => left - right);
      results.push({
        rowCount, payloadBytes: values.byteLength,
        minimumMixedSubmissionMs: samples[0], medianMixedSubmissionMs: samples[2],
        medianRawReadbackMs: rawReadbackSamples[2],
        mixedSubmissionSamplesMs: samples, rawReadbackSamplesMs: rawReadbackSamples
      });
    } finally {
      compiled.destroy();
      input.destroy();
      buffer.destroy();
    }
  }
  console.table(results);
  return results;
}
```

Report browser/OS, adapter, luma.gl versions and all five measured samples when comparing revisions.
The raw readback is a control, not a number to subtract to claim pure packing time. The benchmark
contains allocations and stage setup performed internally by `run()`; profile the named loops to
attribute improvements. No alpha.14 timings have been collected for this draft.

## Proposed change and alternatives

Keep the public API and snapshot semantics. For each supported fixed-width numeric chunk, branch
once on packed layout and copy its complete byte range with a single bulk operation. Preserve the
strided fallback, chunk order, nonzero byte offsets and empty chunks. A borrowed typed view is an
optional later improvement only where alignment and the documented callback lifetime permit it;
bulk copying into an owned snapshot is the conservative first fix.

Alternatives are leaving the row loop, making callers repack, or introducing a transfer pool first.
Caller repacking cannot remove internal copies; pooling alone cannot remove per-row view objects.
Do not change CPU callback mutability or ownership as part of this performance fix.

Acceptance: compare packed and strided inputs with identical logical values, offset subarrays with
sentinel padding, empty and multiple chunks, all three numeric scalar types and component counts
1–4. Assert exact bytes and chunk shapes after round trip and repeated runs. Mutation after a
snapshot must retain the existing promised isolation. Record both performance and correctness;
a speed-only benchmark would miss an offset or stride regression.

## Downstream deletion

No local transfer replacement can be deleted: this repository already delegates execution to the
published executor and deliberately carries no fork. After new measurements, remove the dated
performance caveat from the CPU-twin guidance in `dev-docs/contributor-contract.md` section 14 and
`docs/api/operation-forms-and-cpu-twins.md`, replacing it with measured guidance for the new release.
The domain operation lowering and CPU twins remain necessary.
