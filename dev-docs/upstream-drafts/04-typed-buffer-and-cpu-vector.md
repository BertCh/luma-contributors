# Add typed raw-buffer graph import and packed CPUVector construction

Status: draft, not filed. Target: `@luma.gl/graph` 10.0.0-alpha.16.

Alpha.16 public API disposition: **not satisfied** for either constructor. Public
`getGPUVectorWGSLLayout` from `@luma.gl/graph/gpu-data` supplies useful packed/native layout
metadata (vec3 packed stride 12, native storage stride 16), but no raw-buffer import or
`CPUVector.fromTypedArray` convenience API.

## Problem and current public APIs

Two frequent entry points require repetitive metadata plumbing:

1. A raw luma.gl `Buffer` needs `graph.importBuffer(descriptor, buffer)` followed by
   `graph.createDataView(handle, layout)` before it can be passed to a graph-native primitive.
2. A packed numeric typed array needs a `CPUData` object with a format and row count, then a
   `CPUVector` with the same format and a one-element chunk list.

These are ergonomics requests. Both workflows already work using published entry points; no new
executor or downstream graph vocabulary is needed.

Checked in installed `dist/gpu-core/gpu-command-graph.d.ts` and implementation:
`importBuffer`, `createDataView`, `importGPUData` and `importGPUVector` exist; `importBufferView` does
not. The last two accept physical GPU metadata containers, not a raw `Buffer` plus a typed layout.
Existing graph import rules require one handle per physical buffer when any access may write.

Checked in `dist/gpu-data/cpu-vector.{d.ts,js}` and `cpu-data.{d.ts,js}`: `CPUVector` accepts
`{name?, format, data: readonly CPUData[]}` only. `CPUData` borrows an `ArrayBuffer` or view and
records offsets/strides without copying. A packed typed-array factory is absent. The executor's
`GPUCPUVector` callback value is a different interface (with `chunks` and `pack()`), not the
physical `CPUVector` input container requested here.

## Minimal reproductions using published packages

The raw-buffer path below is valid today. It exposes two typed, disjoint ranges using one logical
handle; the caller owns and eventually destroys the physical buffer.

```typescript
import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';

export function createRawBufferViews(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({
    byteLength: 32, usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const handle = graph.importBuffer({
    id: 'values', byteLength: buffer.byteLength, usage: buffer.usage
  }, buffer);
  const positions = graph.createDataView(handle, {
    format: 'float32x2', length: 2, byteOffset: 0
  });
  const codes = graph.createDataView(handle, {
    format: 'uint32', length: 2, byteOffset: 16
  });
  return {graph, buffer, positions, codes};
}
```

The CPU path below is also valid today and runnable in Node without a GPU. Its nonzero-offset
subarray demonstrates why a convenience constructor must preserve the supplied view's logical
range rather than wrapping its entire backing array.

```typescript
import {CPUData, CPUVector} from '@luma.gl/graph/gpu-data';

export function createPackedCPUInput() {
  const storage = Float32Array.of(999, 10, 20, 30, 40, 999);
  const values = storage.subarray(1, 5);
  const vector = new CPUVector({
    name: 'positions', format: 'float32x2',
    data: [new CPUData({buffer: values, format: 'float32x2', length: values.length / 2})]
  });
  const chunk = vector.data[0];
  if (vector.length !== 2 || chunk.byteOffset !== values.byteOffset ||
      chunk.buffer !== values.buffer || chunk.byteStride !== 8) {
    throw new Error('CPU input layout changed');
  }
  return vector;
}
```

## Proposed API: graph.importBufferView

Add a method to `GPUCommandGraph` returning `GraphDataView<Format>`:

```text
graph.importBufferView<Format>({
  id: string,
  buffer: Buffer,
  format: Format,
  length: number,
  byteOffset?: number,
  byteStride?: number,
  rowByteLength?: number
}): GraphDataView<Format>
```

Derive descriptor capacity and usage from the buffer. Check format alignment and minimum row width
**before** delegating integer/range validation to `createDataView`. Alpha.16's `createDataView`
accepts a scalar float32 at offset 1 and a one-byte row at offset 31 in a 32-byte buffer; a packed
consumer such as `GPUHistogram` rejects them later. Delegation alone is insufficient.
For the fixed-width 32-bit numeric formats proposed here, offset and stride must be multiples of
4, `rowByteLength` must be at least the format's full byte length (for example 8 for `float32x2`),
and stride must be at least row width. This is component alignment, not a 256-byte storage-binding
offset requirement; existing graph binding helpers align the binding and expose the view's offset.
Other formats need their own format-derived alignment, rather than assuming this numeric rule.

The proposed numeric layout precheck can be expressed without any new runtime dependency:

```typescript
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import type {GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';

export function validateImportLayout(props: {
  format: GPUGraphVectorFormat;
  byteOffset?: number;
  rowByteLength?: number;
  byteStride?: number;
}) {
  const formatByteLength = getGPUVectorFormatInfo(props.format).byteLength;
  const byteOffset = props.byteOffset ?? 0;
  const rowByteLength = props.rowByteLength ?? formatByteLength;
  const byteStride = props.byteStride ?? rowByteLength;
  if (byteOffset % 4 !== 0 || byteStride % 4 !== 0) {
    throw new Error('importBufferView numeric offset and stride must be aligned to 4 bytes');
  }
  if (rowByteLength < formatByteLength || byteStride < rowByteLength) {
    throw new Error('importBufferView row width and stride must fit the complete format');
  }
  return {byteOffset, rowByteLength, byteStride};
}
```

This is a proposed precheck, not an installed method. Apply the existing `createDataView` checks
afterward for non-negative safe-integer layout values and the final row's buffer bound.
On a 32-byte buffer, reject `{format: 'float32', length: 1, byteOffset: 1}` and
`{format: 'float32', length: 1, byteOffset: 31, rowByteLength: 1}`; accept the exact-end packed
view `{format: 'float32', length: 1, byteOffset: 28}`. Also reject an aligned offset 28 with
`rowByteLength: 1, byteStride: 4`, which isolates minimum row width from the alignment condition.

Require logical `length`; spare allocation capacity must not become extra rows.
Retain the existing two-call form for explicit descriptors, deferred per-encoding bindings and
multiple views of an already-declared handle. Supporting `DynamicBuffer` can follow the same
contract as existing `importGPUData`; a raw fixed `Buffer` is sufficient for a first version.

The method borrows the buffer and performs no allocation, upload, readback or submission. All
views of the same physical buffer that can participate in writes must resolve to the same graph
handle, including imports through existing import methods. Reuse compatible imports; reject
incompatible descriptor/ID declarations instead of creating an alias that hides hazards. Preserve
per-encoding replacement behavior through a documented canonical resource ID. A convenience call
must not weaken graph resource tracking.

For example, the proposed first range above becomes:

```text
const positions = graph.importBufferView({
  id: 'values', buffer, format: 'float32x2', length: 2
});
// More ranges can still use createDataView(positions.buffer, ...).
```

## Proposed API: CPUVector.fromTypedArray

Add a static constructor on the physical CPU vector class, exported through the existing
`@luma.gl/graph/gpu-data` entry point:

```text
CPUVector.fromTypedArray<Format>({
  name?: string,
  format: Format,
  values: matching typed array
}): CPUVector<Format>

const vector = CPUVector.fromTypedArray({
  name: 'positions', format: 'float32x2', values
});
```

Start with fixed-width numeric graph formats `float32`, `sint32`, `uint32` and their 2–4 component
forms. Require `Float32Array`, `Int32Array`, `Uint32Array` respectively. Infer row count from
`values.length / components`, reject a partial trailing row and mismatched element types, preserve
subarray byte offsets, and construct one borrowed packed `CPUData` chunk. Keep the format explicit
because one typed array cannot tell scalar rows from multi-component rows. Empty arrays are valid.

Document borrowing: later writes to `values` are visible through the vector before invocation
snapshotting; this constructor does not transfer ownership or make an immutable copy. Existing
`CPUData`/`CPUVector` construction remains the general route for strided data, multiple chunks,
lists and validity metadata. No change is requested to `GPUCPUOperation.execute`'s borrowed
read-only snapshot contract.

## Alternatives and acceptance

Leaving the existing calls is functional but forces applications to repeat format sizes and row
count arithmetic. A `GPUData` or `GPUVector` wrapper already enables graph imports; constructing
another physical metadata container solely to obtain a view adds an unnecessary intermediate.
A free `makeCPUVectorFromTypedArray` factory would also work; a static method keeps the constructor
discoverable without widening the graph execution API. An overloaded `CPUVector` constructor is
another option, with a discriminator to preserve its existing chunk form. Inferring format solely
from array type cannot determine component count.

Acceptance tests for raw import: inferred return format, offset and stride preservation, exact
end-of-buffer range, empty range, invalid length/alignment/range, shared-handle hazard tracking,
conflicting IDs and borrowed-buffer lifetime after compiled graph destruction. Include two
disjoint writable ranges of one physical allocation and an overlapping read/write case; a helper
that simply imports a fresh handle on each call would fail the hazard test.

Acceptance tests for CPU construction: `[10,20,30,40]` as `float32x2` produces two rows, a nonzero
subarray offset is retained, padding `[999]` is excluded, a three-element array for `float32x2`
throws, a `Uint32Array` for `float32` is rejected at typecheck and runtime, and zero rows remain
zero. Test all supported element/component combinations and mutation before snapshotting. These
checks distinguish a raw-byte reinterpretation or whole-backing-buffer wrapper from the proposal.

## Downstream deletion

Replace calls to and delete `importGraphBuffer` in
`modules/analysis/src/analysis-core/kernels/gpu-contributor-utils.ts`; keep `GPUParameterBuffer`
and other domain utilities in that file. Its current implementation is exactly the public
`importBuffer` / `createDataView` pair, with a default length inferred from physical capacity.
Migration must pass each logical length explicitly, especially for buffers with spare capacity.

Replace and delete `makeCPUVector` in
`modules/analysis/test/utils/mixed-execution-test-utils.ts`. Other examples and tests that construct
the same one-chunk CPU input can use the upstream factory. `getCPUVectorRows`, operation lowering,
domain contributors and the more general chunk helpers remain. The proposal does not promise to
remove mixed-executor copies; that is the separate packed-transfer filing.
