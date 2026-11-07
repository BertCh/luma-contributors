# GPUScalarScale

`GPUScalarScale`, from `@luma-contributors/graph/gpu-dataframe`, maps a prepared scalar column to numeric rendering attributes without reading source rows back to the CPU. It accepts packed float32 graph views (or uint32 category codes for point mode), optional uint32 source validity, shared domain/range/parameter buffers, and caller-owned output values with optional output validity.

Create one contributor per source chunk and share the controls across them. Output rows retain source identity; empty chunks emit no commands. The contributor does not allocate or destroy borrowed input/output buffers. Display selection masks remain separate from source validity.

```typescript
const controls = prepareGPUScalarScaleControls({
  scale: 'sqrt', domain: [0, 100], range: [1, 20], nullValue: 0
});
// Allocate domainView with maximumDomainCount entries, at least two.
// Upload controls.domain to its prefix, controls.range, and controls.parameters.
graph.add(new GPUScalarScale({
  values: sourceValues,
  validity: sourceValidity,
  domain: domainView,
  range: rangeView,
  parameters: parameterView,
  maximumDomainCount: 2,
  output: {values: radiusValues, validity: radiusValidity}
}));
```

Changing control buffer contents replays the existing graph. Changing view topology or capacity requires recompilation. `getGPUScalarScaleParameterValues({scale, domainCount, nullValue, invert}, target?)` fills four floats: scale code, active domain count, fallback value, and fixed-scale inversion flag (zero by default).

| Mode | Mapping |
| --- | --- |
| `linear` | Two endpoints, ascending or descending; extrapolation; equal domain maps to range midpoint. |
| `sqrt` | Signed square root, including negative domains and values; otherwise the same normalization. |
| `log` | Positive or negative same-sign nonzero domain endpoints; either direction; same-sign extrapolation. No positive floor is substituted. |
| `identity` | Native fixed linear scale with range equal to domain. Unequal endpoints return the prepared value; equal endpoints return that domain value. With `invert: true`, the GPU domain is reversed as the output range, matching native fixed-height inversion; equal endpoints still return that value. |
| `point` | Unique ordered categories, d3's default padding 0 and alignment 0.5; descending ranges supported. One category maps to midpoint; empty domains and unknown categories are invalid. |

`invert: true` is supported only for `identity`. It maps unequal endpoints as `domain[0] + domain[1] - value`, using normalized interpolation to avoid overflowing the endpoint sum. Domain endpoints may be produced on the GPU; no host copy of the domain or source rows is required. Other modes express inversion by reversing their explicit range.

For a nondegenerate log domain, zero and wrong-sign input values are invalid. An equal valid log domain maps every finite input to midpoint, including otherwise invalid signs, matching constant native normalization.

`prepareGPUScalarScaleControls` rejects nonfinite or inexact float32 domain/range values; normalize higher-precision controls explicitly before calling it. Its domain array contains only active entries, including zero entries for an empty point domain. Allocate a graph domain view with capacity `maximumDomainCount >= max(2, domainCount)` and upload that prefix. Capacity is bounded at 256; point lookup scans active domain entries per row. GPU producers of point domains must guarantee uniqueness.

Null/nonfinite source values, unknown categories, malformed replay controls, unsupported log inputs, collapsed transformed domains and nonfinite float32 results write the fallback value with output validity zero. Successful mappings write validity one. Output validity is optional; without it, fallback values cannot diagnose unsupported arithmetic.

This is a finite prepared-float32 contract, not bitwise double-precision d3 equivalence. Raw timestamp words, arbitrary doubles, piecewise domains, clamping, categorical strings, custom point padding/alignment and threshold sizes are outside the contract. Transcendental rounding and subnormal-sensitive arithmetic need explicit tolerances and domain policy. Upstream code owns precision normalization and dictionary encoding. Direct uint32 point inputs accept codes through 2^24 inclusive; larger codes and sentinel values are invalid before conversion, so distinct integers never alias. Other uint32 scalar modes are unsupported and produce invalid outputs. The ordered point domain can remap dictionary order, for example `[2, 0, 1]`.

Read-only inputs may share buffers. Writable output values and validity must use separate buffers from each other and every input, even for disjoint ranges. Nonzero offsets in separate buffers are supported. Allocate rendered outputs with `STORAGE | VERTEX` usage when binding them directly as attributes.
