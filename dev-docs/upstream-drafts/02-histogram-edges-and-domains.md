# Make equal-width GPUHistogram stable at exact edges and wide GPU domains

Status: draft, not filed. Target: `@luma.gl/graph` 10.0.0-alpha.16.

Alpha.16 public API disposition: **not satisfied** for this equal-width arithmetic request.
`GPUHistogram` from `@luma.gl/graph/gpu-core` already supports views, offsets and explicit edges;
the equal-width float quotient remains. No alpha.16 GPU failure is claimed by source inspection.

## Problem and verified scope

The equal-width float32 path in `GPUHistogram` computes a normalized quotient and scales it back to
a bin index. Exact interior edge values can fall in the preceding bin, and historical runs gave
different answers for a literal domain and the same domain read from a GPU view.

Installed `dist/gpu-core/gpu-histogram.js`, `addHistogramPass`, still generates:

```wgsl
let ratio = (value - minimum) / (maximum - minimum);
binIndex = min(u32(ratio * f32(BIN_COUNT)), BIN_COUNT - 1u);
```

Alpha.16 already has literal, view and automatic domains, explicit irregular `edges`, masks,
chunked inputs, domain/output view offsets and non-finite input filtering. Those are not missing
APIs. This filing is about the remaining **equal-width float32** arithmetic. The integer path uses
`multiplyDivideFloor` and is outside this reproduction.

The committed downstream spec records observations on Apple M3 Pro / headless Chromium with
**alpha.12**. Its formula remains in installed alpha.16, but that is source evidence only: this
draft has no fresh GPU run, alpha.16 failure confirmation or second-adapter confirmation.

## Minimal reproduction using published packages

The function below submits a histogram and reads it back. Call it twice with the same WebGPU
`Device`, changing only `domainForm`. It uses no downstream helper.

```typescript
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUHistogram} from '@luma.gl/graph/gpu-core';

export async function reproduceHistogram(
  device: Device,
  values = Float32Array.of(0, 41, 82, 123),
  domain: readonly [number, number] = [0, 123],
  binCount = 3,
  domainForm: 'literal' | 'view' = 'literal'
) {
  const graph = new GPUCommandGraph<undefined>(device);
  const usage = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
  const inputBuffer = device.createBuffer({data: values, usage});
  const domainBuffer = device.createBuffer({data: Float32Array.from(domain), usage});
  const outputBuffer = device.createBuffer({byteLength: binCount * 4, usage});
  const inputHandle = graph.importBuffer(
    {id: 'input', byteLength: inputBuffer.byteLength, usage}, inputBuffer
  );
  const domainHandle = graph.importBuffer(
    {id: 'domain', byteLength: domainBuffer.byteLength, usage}, domainBuffer
  );
  const outputHandle = graph.importBuffer(
    {id: 'output', byteLength: outputBuffer.byteLength, usage}, outputBuffer
  );
  graph.add(new GPUHistogram({
    input: graph.createDataView(inputHandle, {format: 'float32', length: values.length}),
    output: graph.createDataView(outputHandle, {format: 'uint32', length: binCount}),
    domain: domainForm === 'literal' ? domain : graph.createDataView(
      domainHandle, {format: 'float32', length: 2}
    )
  }));
  const compiled = graph.compile();
  const encoder = device.createCommandEncoder();
  try {
    compiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    const bytes = await outputBuffer.readAsync(0, binCount * 4);
    return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, binCount));
  } finally {
    encoder.destroy();
    compiled.destroy();
    inputBuffer.destroy();
    domainBuffer.destroy();
    outputBuffer.destroy();
  }
}
```

For `values = [0, 41, 82, 123]`, domain `[0, 123]`, three bins, the intended half-open bins with a
closed last bin give **`[1, 1, 2]`** in both forms. The historical observed result was `[2, 1, 1]`
in both forms. Every value, edge and width in this case is exactly representable in float32.

For `values = 0..49`, domain `[0, 49]`, 49 bins, expect **48 ones followed by two**, in both
forms. These expectations follow by counting `[k, k + 1)` and closing the final bin; they do not
come from the quotient implementation. The runner below adds immediate float32 neighbors of 1,
27, 41 and 82. The adjacent words differ by one in IEEE-754 binary32 encoding, so values below an
interior edge must stay in the preceding bin and the edge and value above it belong in the upper
bin. In particular, the triplet at 1 catches an incorrect float32 implementation that scales by
`f32(binCount / range)` and adds `2^-24` before integer conversion: it moves `0x3f7fffff`
(`1 - 2^-24`) into bin 1, producing `[0,3,0,…]` instead of `[1,2,0,…]`. The 27/41/82 controls
alone do not reject that implementation; all ten earlier literal/view cases passed it.

Append this to the reproduction above and supply the actual installed version and device adapter
description (including backend), plus browser/version. It logs every input and every bin before
asserting, including failures; do not replace the arrays with totals or abbreviated output.

```typescript
export async function collectHistogramEdgeEvidence(
  device: Device,
  evidence: {packageVersion: string; adapter: string; browser: string}
) {
  const fromWords = (...words: number[]) => new Float32Array(Uint32Array.from(words).buffer);
  // Independent half-open interval counts: one value just below, two at/above each edge.
  const adjacent1Counts = [1, 2, ...new Array<number>(47).fill(0)];
  const adjacent27Counts = new Array<number>(49).fill(0);
  adjacent27Counts[26] = 1;
  adjacent27Counts[27] = 2;
  const fixtures = [
    {
      name: 'integer-edges-49', values: Float32Array.from({length: 50}, (_, index) => index),
      domain: [0, 49] as const, binCount: 49, expected: [...new Array<number>(48).fill(1), 2]
    },
    {
      name: 'integer-edges-3', values: Float32Array.of(0, 41, 82, 123),
      domain: [0, 123] as const, binCount: 3, expected: [1, 1, 2]
    },
    {
      name: 'adjacent-1', values: fromWords(0x3f7fffff, 0x3f800000, 0x3f800001),
      domain: [0, 49] as const, binCount: 49, expected: adjacent1Counts
    },
    {
      name: 'adjacent-27', values: fromWords(0x41d7ffff, 0x41d80000, 0x41d80001),
      domain: [0, 49] as const, binCount: 49, expected: adjacent27Counts
    },
    {
      name: 'adjacent-41', values: fromWords(0x4223ffff, 0x42240000, 0x42240001),
      domain: [0, 123] as const, binCount: 3, expected: [1, 2, 0]
    },
    {
      name: 'adjacent-82', values: fromWords(0x42a3ffff, 0x42a40000, 0x42a40001),
      domain: [0, 123] as const, binCount: 3, expected: [0, 1, 2]
    }
  ];
  const records = [];
  for (const fixture of fixtures) {
    for (const domainForm of ['literal', 'view'] as const) {
      const counts = await reproduceHistogram(
        device, fixture.values, fixture.domain, fixture.binCount, domainForm
      );
      const record = {
        ...evidence, name: fixture.name, domainForm, domain: fixture.domain,
        binCount: fixture.binCount, values: Array.from(fixture.values),
        counts, expected: fixture.expected
      };
      records.push(record);
      console.log(JSON.stringify(record));
    }
  }
  if (records.some(record => record.counts.length !== record.expected.length ||
      record.counts.some((count, index) => count !== record.expected[index]))) {
    throw new Error('Histogram edge mismatch; retain all version-labelled records above');
  }
  return records;
}
```

This runner is queued, not executed here. Alpha.16 outcomes remain **unverified** until device
readback, even if a CPU float32 model predicts a misplaced edge.

For the wide case, call with `Float32Array.of(0, 1, -5e37, 5e37)`, domain `[-1e38, 1e38]` and
16 bins. Assert both forms equal:

```text
[0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]
```

Historically the literal form gave that vector and the view form gave `[4, 0, ..., 0]`. Reciprocal
lowering and subnormal flushing are a possible explanation, inferred from the counts, not verified
generated GPU code. The reciprocal of a range around `2e38` is subnormal. Do not treat this as a
proven compiler diagnosis.

## Assertions to carry upstream

The committed spec (repository `BertCh/luma-contributors`, commit
`fbb78ec5dad721b9ff48783142d6f0ed61bbbc46`, file
`modules/analysis/test/gpu-core/gpu-histogram-domain-view.spec.ts`)
contains 12 cases. Five assert desired counts; six pin the observed misplaced edges; the last pins
the incorrect wide-view result. A passing downstream suite therefore does **not** prove correctness.
For the upstream regression suite, replace observation assertions with these expectations:

| Bins | Width | Minimum | Values | Expected, for literal and view |
| --- | --- | --- | --- | --- |
| 7 | 1 | 0 | Every edge, inclusive | One per bin, two in last |
| 12 | 0.25 | -16 | Every edge, inclusive | One per bin, two in last |
| 30 | 8 | 1024 | Every edge, inclusive | One per bin, two in last |
| 255 | 1 | 0 | Every edge, inclusive | One per bin, two in last |
| 1000 | 1 | 0 | Every edge, inclusive | One per bin, two in last |
| 49 | 1 | 0 | Every edge, inclusive | One per bin, two in last |
| 49 | 8 | 1024 | Every edge, inclusive | One per bin, two in last |
| 3 | 41 | 0 | Every edge, inclusive | `[1, 1, 2]` |
| 6 | 41 | 0 | Every edge, inclusive | `[1, 1, 1, 1, 1, 2]` |
| 12 | 41 | 0 | Every edge, inclusive | Eleven ones, then two |
| 7 | 83 | 0 | Every edge, inclusive | Six ones, then two |
| 16 | — | -1e38 | `[0, 1, -5e37, 5e37]`, maximum 1e38 | Bin 4 = 1, bin 8 = 2, bin 12 = 1; others zero |

For edge fixtures, `values[k] = minimum + k * width` for `k = 0..binCount`, and
`maximum = minimum + binCount * width`. Assert `counts` equals an array of `binCount` ones with its
last element replaced by two, not just the total count. The 1000-bin case exercises global atomics;
the smaller cases exercise local atomics. Missing devices should fail these cases explicitly.

## Proposed behavior and alternatives

Retain `GPUHistogram`'s props. Specify interior edges in the upper bin and the maximum in the last
bin; require literal/view agreement for the same representable domain. Use division to estimate a
candidate and verify it with boundary comparisons rather than trusting the quotient near an integer.
For suitable finite products, the desired comparison is
`k * range <= (value - minimum) * binCount < (k + 1) * range`.

This expression alone is not a complete fix: the subtraction and products can overflow, and product
rounding limits exact classification. Scale operands by powers of two *before* dangerous arithmetic
and document the representable-edge rule. Verify the remedy on GPU; no fix is validated by this draft.

Alternatives: binary search over explicit `edges` (already supported, and a useful comparison oracle),
integer arithmetic for integer input (already a separate path), or an epsilon added to the quotient.
A fixed epsilon moves values genuinely below an edge, so it is not an acceptable general remedy.
Literal-only domains would discard a supported GPU-resident workflow.

Also test values immediately below/above edges, bin centers, narrow domains far from zero, domains
bracketing `2^126`, domains whose raw width overflows float32, nonzero view offsets, repeated
submissions and `'auto'` where it yields the same domain. Keep filtering and degenerate-domain
behavior covered. These are requested additions, not claims about committed test coverage.

## Downstream deletion

After the fixed release passes these cases and column-profile parity, delegate numeric histogram
counting in `modules/analysis/src/dataframe-analysis/column-profile/column-profile-kernels.ts` to
published `GPUHistogram` and delete its private `BIN_WGSL` / `getBinIndex` implementation and local
bin-boundary workaround constants where no longer used. Category histograms and other profile
statistics remain. Flip the downstream observed-failure assertions to the correct results rather
than deleting regression coverage.
