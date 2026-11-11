# GPUVariableStandardization

`GPUVariableStandardization` is a command-node producer for five named variable transformations.
It composes shared `GPUReduction` passes and elementwise WGSL. It never submits, reads back,
or computes statistics on the CPU. Add it to a `GPUCommandGraph` and compile/encode normally.

```typescript
const recipe = new GPUVariableStandardization({
  values, // GraphDataView<'float32'> or GraphVectorView<'float32'>
  mask, // optional uint32 view/vector: nonzero includes a row
  method: 'z-score',
  variance: 'sample', // default; 'population' also available
  output // caller-owned float32 view/vector
});
graph.add(recipe);
```

For the finite, unmasked rows of the entire logical column, let `n` be count, `mean` the
arithmetic mean, `minimum`/`maximum` the extrema, and `range = maximum - minimum`:

| Method | Formula | Constant population |
| --- | --- | --- |
| `deviation-from-mean` | `x - mean` | zero |
| `z-score` (default sample) | `(x - mean) / sqrt(sum((x - mean)^2) / (n - 1))` | zero when `n >= 2` |
| `z-score`, population | `(x - mean) / sqrt(sum((x - mean)^2) / n)` | zero |
| `mad` | `(x - mean) / (sum(abs(x - mean)) / n)` | original value |
| `range-adjust` | `x / range` | original value |
| `range-standardize` | `(x - minimum) / range` | original value |

MAD means **mean absolute deviation about the arithmetic mean**. It is not median absolute
deviation and has no normal consistency factor. Sample z-score with fewer than two included
rows produces NaN. Excluded rows (mask zero, NaN, positive or negative infinity) always output
NaN, including in constant populations. An all-excluded column therefore outputs all NaN.
Null validity must be represented by NaN or the caller's mask; this API does not inspect Arrow
validity bitmaps. Every output row is rewritten on each encoding.

`values`, `mask` and `output` must each be a single packed chunk; a column with more than one
chunk is rejected at construction (`values single chunk`), and the three columns must have the same
length. There is no implicit packing, realignment or independent normalization per chunk. Source and
output buffers must be disjoint. All views must belong to the graph. The logical row count is 0
through 2^24. An empty column emits no output rows. Lengths, method, variance convention, and mask presence are topology;
buffer contents are per-frame data. Buffers remain caller-owned and scratch is graph-owned.

Arithmetic uses float32 shared hierarchical reductions, not GeoDa's float64 loops. Summation order
is a hierarchical reduction, not GeoDa's sequential loop, so compare within a float32 tolerance, not bitwise. A shared extent reduction supplies an origin and radius. The origin is the minimum for same-sign extrema, otherwise zero. Values are normalized around
that origin before mean and dispersion reductions; z-scores and MAD are computed in normalized
space. Extreme divisors are rescaled before division to avoid reciprocal underflow on GPUs
that flush subnormal results. This avoids raw-sum/squared-deviation overflow and cancellation from large common offsets.
Underflow and float32 rounding can still reduce accuracy. Unrepresentable final deviations or
range-adjusted values become NaN. This is not a float64 statistics API.

## Audited reference

The formulas were inspected in GeoDa `GenUtils.cpp` at commit
[`37510860bd9f7b8ae145c15759f7abc5ab217cf3`](https://github.com/GeoDaCenter/geoda/blob/37510860bd9f7b8ae145c15759f7abc5ab217cf3/GenUtils.cpp#L1343):
`DeviationFromMean`, `StandardizeData` (sample denominator `n - 1`),
`MeanAbsoluteDeviation` (mean-centered absolute deviations divided by `n`), `RangeAdjust`, and
`RangeStandardize`. These independently implemented recipes have no GeoDa runtime dependency.

The named mathematical formulas agree with GeoDa for finite nonconstant data, with these explicit
policies: singleton sample z-score is NaN; excluded values are NaN; infinities are excluded.
GeoDa's zero-MAD and zero-range early returns preserve original values, retained here. Its constant
z-score centers to zero before returning failure, retained for `n >= 2`. GeoDa's range routines
initialize maximum with `DBL_MIN` and use an `else if` update, which can miscompute extrema for
negative-only or strictly decreasing input. These recipes compute the actual extrema instead of
reproducing those implementation defects. No claim of byte-for-byte GeoDa parity is made.

{/* lens:GPUVariableStandardization */}

::lens{scenario=gpu-dataframe.variable-standardization rung=graph}
