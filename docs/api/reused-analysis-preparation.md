# Source topology and wide numeric profiling

### `prepareArrowTableForGPUAnalytics`

Import from `@luma-contributors/analysis/analysis-core`. Float64, Int64, Uint64 and wide Decimal
columns (`bitWidth >= 64`, including Decimal64, Decimal128 and Decimal256) require an explicit
`float64` precision policy before narrowing to Float32. Decimal32 is unsupported. For wide
integers and decimals, preparation rounds the original signed coefficient and decimal scale
directly to nearest-even Float32. `mode: 'exact'` rejects any changed value, including
`9007199254740993` and Decimal128 `1.0000000000000001`.

Round mode compares the error of a finite candidate with
`maximumAbsoluteError + maximumRelativeError * abs(original)`. Int64, Uint64 and supported Decimal
columns compare budgets with exact rational arithmetic. Float64 columns compute the error and
budget expression with JavaScript binary64 arithmetic. For example, with relative budget
`1 / 16777281` and absolute budget `0`, Float64 `16777281` accepts candidate `16777280`, while
Decimal128 rejects it because the binary64 relative budget is slightly too small.

Wide integer and Decimal error reports round each exact rational error once to nearest-even
binary64, including subnormals. An Int64 value of `9007199254740993` accepts an absolute budget
of `1` and reports an error of `1`. Extreme Decimal scales do not require constructing an enormous
power of ten: zero remains exact, definite overflow rejects, and definite underflow respects the
budget and preserves the sign of zero. An absolute error below half `Number.MIN_VALUE` reports
zero while its relative error is still `1`. Nulls and record-batch topology are preserved.
The standalone `convertWideNumericValue` / `convertWideNumericVector` helpers convert through JavaScript numbers;
they do not establish an exact Float32 precision policy for Decimal values.

### `preparePolygonContiguityTopology`

Import from `@luma-contributors/analysis/gpu-spatial-analysis`. Accepts source-aligned Arrow
polygon lists, retaining empty and invalid rows, batches and diagnostics. Original binary64 XY
coordinates are interned into Float32 integer identity tokens. They are equality-only values:
never use them for rendering, metric distances or positive snap tolerance. Z is ignored; signed
zero compares equal. No snapping, noding, repair or Float32 coordinate rounding changes adjacency.
Limits are 4096 source rows, 65536 inspected vertex occurrences (including closures and rejected
rows), and 1 MiB of batch metadata. Temporary preparation reserves a conservative 24 MiB.

### `computePolygonContiguityOnCPU`

Optional CPU route on the prepared tokens, for queen or whole-edge rook adjacency. GPU consumers
use `GPUContiguityWeights` with zero snap tolerance. CPU final-capacity overflow is explicit and
requiredCount is exact; there is no intermediate candidate overflow. geometryMask still controls
which source rows participate in downstream statistics.

### `GPUWideNumericScan`

Import from `@luma-contributors/analysis/dataframe-analysis`. Chunked packed word phases run in
parallel over rows. The caller initializes result words. Props and chunk layout are compile-time;
no per-frame CPU row validation occurs. Command-node budget and device binding limits are checked.
Fewer than 2^32 total rows keep counters and carry sums exact. No readback occurs in the contributor.

### Exact aggregates and CPU references

`aggregateInteger64OnGpu` / `aggregateInteger64Range` preserve signed or unsigned 64-bit extrema,
null counts and bigint sums (44-byte summaries). `aggregateDecimal128OnGpu` /
`aggregateDecimal128Column` preserve 128-bit unscaled bigint values, decimal scale and exact
scale-normalized comparisons (76-byte summaries). Optional selection skips unselected nulls.
`summarizeInteger64Reference` and `summarizeDecimal128Reference` are pure CPU reference routes.
Integer reference inputs are normalized to the declared word width, as resident storage is.

`makeDecimal128WordVectors(device, column)` accepts an Arrow `Vector<Decimal>` only when its
physical `bitWidth` is 128. Every chunk must also be Decimal128 and have the column's scale;
mixed-width or mixed-scale concatenations reject before any device allocation. The upload rejects
other types and widths, including Decimal256. It retains each source chunk, including empty chunks.
When any chunk has nulls, `validity` has the same topology as `words`; otherwise it is absent and consumes no
device memory. Call the returned `destroy()` to free all uploaded vectors.
`aggregateDecimal128Column` owns and frees its upload.

### Narrowing diagnostics

`diagnoseFloat32Narrowing` and `scanWideNumericColumnOnGpu` share `NumericNarrowingReport`.
`maximumAbsoluteError` is an exact bigint for Int64/Uint64, including zero; subtraction happens
before conversion to Number. Integer rounding is nearest-even and avoids binary64 double rounding.
For other formats it is a number: CPU JS absolute difference, or a GPU Float64 lower bound within
relative 2^-31. Overflow and nonfinite counts are separate from the maximum finite error. The GPU
reads six words (24 bytes) over two phases. `mergeNumericNarrowingReports` keeps the maximum across
chunks. `summarizeNarrowingReference` independently rounds integer candidates for qualification.
Arithmetic, WGSL-expression and upload helpers are public building blocks, not catalog operators.
