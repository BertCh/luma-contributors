var e={slug:`api/reused-analysis-preparation`,title:`Source topology and wide numeric profiling`,html:`<h1 id="source-topology-and-wide-numeric-profiling">Source topology and wide numeric profiling</h1>
<h3 id="preparearrowtableforgpuanalytics"><code>prepareArrowTableForGPUAnalytics</code></h3>
<p>Import from <code>@luma-contributors/analysis/analysis-core</code>. Float64, Int64, Uint64 and wide Decimal
columns (<code>bitWidth >= 64</code>, including Decimal64, Decimal128 and Decimal256) require an explicit
<code>float64</code> precision policy before narrowing to Float32. Decimal32 is unsupported. For wide
integers and decimals, preparation rounds the original signed coefficient and decimal scale
directly to nearest-even Float32. <code>mode: 'exact'</code> rejects any changed value, including
<code>9007199254740993</code> and Decimal128 <code>1.0000000000000001</code>.</p>
<p>Round mode compares the error of a finite candidate with
<code>maximumAbsoluteError + maximumRelativeError * abs(original)</code>. Int64, Uint64 and supported Decimal
columns compare budgets with exact rational arithmetic. Float64 columns compute the error and
budget expression with JavaScript binary64 arithmetic. For example, with relative budget
<code>1 / 16777281</code> and absolute budget <code>0</code>, Float64 <code>16777281</code> accepts candidate <code>16777280</code>, while
Decimal128 rejects it because the binary64 relative budget is slightly too small.</p>
<p>Wide integer and Decimal error reports round each exact rational error once to nearest-even
binary64, including subnormals. An Int64 value of <code>9007199254740993</code> accepts an absolute budget
of <code>1</code> and reports an error of <code>1</code>. Extreme Decimal scales do not require constructing an enormous
power of ten: zero remains exact, definite overflow rejects, and definite underflow respects the
budget and preserves the sign of zero. An absolute error below half <code>Number.MIN_VALUE</code> reports
zero while its relative error is still <code>1</code>. Nulls and record-batch topology are preserved.
The standalone <code>convertWideNumericValue</code> / <code>convertWideNumericVector</code> helpers convert through JavaScript numbers;
they do not establish an exact Float32 precision policy for Decimal values.</p>
<h3 id="preparepolygoncontiguitytopology"><code>preparePolygonContiguityTopology</code></h3>
<p>Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>. Accepts source-aligned Arrow
polygon lists, retaining empty and invalid rows, batches and diagnostics. Original binary64 XY
coordinates are interned into Float32 integer identity tokens. They are equality-only values:
never use them for rendering, metric distances or positive snap tolerance. Z is ignored; signed
zero compares equal. No snapping, noding, repair or Float32 coordinate rounding changes adjacency.
Limits are 4096 source rows, 65536 inspected vertex occurrences (including closures and rejected
rows), and 1 MiB of batch metadata. Temporary preparation reserves a conservative 24 MiB.</p>
<h3 id="computepolygoncontiguityoncpu"><code>computePolygonContiguityOnCPU</code></h3>
<p>Optional CPU route on the prepared tokens, for queen or whole-edge rook adjacency. GPU consumers
use <code>GPUContiguityWeights</code> with zero snap tolerance. CPU final-capacity overflow is explicit and
requiredCount is exact; there is no intermediate candidate overflow. geometryMask still controls
which source rows participate in downstream statistics.</p>
<h3 id="gpuwidenumericscan"><code>GPUWideNumericScan</code></h3>
<p>Import from <code>@luma-contributors/analysis/dataframe-analysis</code>. Chunked packed word phases run in
parallel over rows. The caller initializes result words. Props and chunk layout are compile-time;
no per-frame CPU row validation occurs. Command-node budget and device binding limits are checked.
Fewer than 2^32 total rows keep counters and carry sums exact. No readback occurs in the contributor.</p>
<h3 id="exact-aggregates-and-cpu-references">Exact aggregates and CPU references</h3>
<p><code>aggregateInteger64OnGpu</code> / <code>aggregateInteger64Range</code> preserve signed or unsigned 64-bit extrema,
null counts and bigint sums (44-byte summaries). <code>aggregateDecimal128OnGpu</code> /
<code>aggregateDecimal128Column</code> preserve 128-bit unscaled bigint values, decimal scale and exact
scale-normalized comparisons (76-byte summaries). Optional selection skips unselected nulls.
<code>summarizeInteger64Reference</code> and <code>summarizeDecimal128Reference</code> are pure CPU reference routes.
Integer reference inputs are normalized to the declared word width, as resident storage is.</p>
<p><code>makeDecimal128WordVectors(device, column)</code> accepts an Arrow <code>Vector&#x3C;Decimal></code> only when its
physical <code>bitWidth</code> is 128. Every chunk must also be Decimal128 and have the column's scale;
mixed-width or mixed-scale concatenations reject before any device allocation. The upload rejects
other types and widths, including Decimal256. It retains each source chunk, including empty chunks.
When any chunk has nulls, <code>validity</code> has the same topology as <code>words</code>; otherwise it is absent and consumes no
device memory. Call the returned <code>destroy()</code> to free all uploaded vectors.
<code>aggregateDecimal128Column</code> owns and frees its upload.</p>
<h3 id="narrowing-diagnostics">Narrowing diagnostics</h3>
<p><code>diagnoseFloat32Narrowing</code> and <code>scanWideNumericColumnOnGpu</code> share <code>NumericNarrowingReport</code>.
<code>maximumAbsoluteError</code> is an exact bigint for Int64/Uint64, including zero; subtraction happens
before conversion to Number. Integer rounding is nearest-even and avoids binary64 double rounding.
For other formats it is a number: CPU JS absolute difference, or a GPU Float64 lower bound within
relative 2^-31. Overflow and nonfinite counts are separate from the maximum finite error. The GPU
reads six words (24 bytes) over two phases. <code>mergeNumericNarrowingReports</code> keeps the maximum across
chunks. <code>summarizeNarrowingReference</code> independently rounds integer candidates for qualification.
Arithmetic, WGSL-expression and upload helpers are public building blocks, not catalog operators.</p>`};export{e as default};