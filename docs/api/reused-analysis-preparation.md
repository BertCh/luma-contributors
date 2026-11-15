# Source topology and wide numeric profiling

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

### Narrowing diagnostics

`diagnoseFloat32Narrowing` and `scanWideNumericColumnOnGpu` share `NumericNarrowingReport`.
`maximumAbsoluteError` is an exact bigint for Int64/Uint64, including zero; subtraction happens
before conversion to Number. Integer rounding is nearest-even and avoids binary64 double rounding.
For other formats it is a number: CPU JS absolute difference, or a GPU Float64 lower bound within
relative 2^-31. Overflow and nonfinite counts are separate from the maximum finite error. The GPU
reads six words (24 bytes) over two phases. `mergeNumericNarrowingReports` keeps the maximum across
chunks. `summarizeNarrowingReference` independently rounds integer candidates for qualification.
Arithmetic, WGSL-expression and upload helpers are public building blocks, not catalog operators.
