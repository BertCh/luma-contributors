var e={slug:`api/analysis-core/fitting-kernels`,title:`Shared fitting kernels`,html:`<h1 id="shared-fitting-kernels">Shared fitting kernels</h1>
<p>See the <a href="/luma-contributors/docs/api">API index</a> for the other analysis entry points.</p>
<h2 id="overview">Overview</h2>
<p>Import these building blocks from <code>@luma-contributors/analysis/analysis-core</code>. They fit models supplied
by the caller; no model, pretrained weights or dataset is included. These are GPU building blocks,
without a CPU twin: their work is parallel and a CPU stage would force gradient/state readbacks.</p>
<p>Compile-time shapes and loop bounds are validated at construction. Mutable values live in graph
views. Packed scalar bindings retain the 128 MiB fitting cap, except coordinate-encoder row chunks,
which are bounded by the device storage-binding limit. Every binding is checked against the device
at the start of <code>getCommandNodes</code>. Fitted models have at most 100,000 parameters. Use minibatches to bound
row-gradient and encoded-feature storage. Encoder work is capped at 5 billion bounded operations.
The limits are planning bounds, not throughput measurements on additional devices.</p>
<h2 id="gpufittinggradientsums"><code>GPUFittingGradientSums</code></h2>
<p>Takes <code>rowCount</code>, <code>parameterCount</code>, row-major scalar <code>gradients</code>, optional uint32 <code>mask</code>, and
<code>output.sums</code> (one float per parameter). A zero mask skips the row. Empty input writes zero sums.
Finite retained gradients are required; non-finite terms propagate. For a mean gradient, scale
contributions by the retained batch size in the derivative kernel. A loss can be reduced as a term.</p>
<p>One invocation sums one parameter over at most 256 rows. Separate dispatches merge adjacent tile
partials in a fixed binary tree, including odd final tiles. The node count is
<code>1 + ceil(log2(max(1, ceil(rowCount / 256))))</code> (at most 17). Padding invocations never access rows.
Results are repeatable on a device/compiler; floating-point reassociation and trig precision mean
there is no cross-device bit identity guarantee. No atomics or compensated-sum identity is used.</p>
<h2 id="gpuadamstep"><code>GPUAdamStep</code></h2>
<p>Implements <a href="https://arxiv.org/pdf/1412.6980">Kingma and Ba, Algorithm 1</a>:</p>
<p><code>first = beta1 * first + (1 - beta1) * gradient</code>,
<code>second = beta2 * second + (1 - beta2) * gradient²</code>,
<code>weight -= learningRate * (first / (1 - beta1^t)) / (sqrt(second / (1 - beta2^t)) + epsilon)</code>.</p>
<p>The second-moment contribution uses a significand/exponent decomposition for gradients above
2^63, so a compiler cannot square the unscaled gradient before its small multiplier. The corrected
square root likewise divides bounded significands, takes a square root in a bounded range, and
restores half the exponent difference. Integer bit operations separate these calculations from
the original magnitudes and prevent floating-point reassociation across the decomposition.
For a first gradient of 1e20, the stored moments and the roughly 1e20 denominator are representable,
while both an unscaled square and <code>second / secondCorrection</code> would require roughly 1e40.
The final update also multiplies/divides bounded significands, combines the learning rate,
first-moment correction and denominator exponents, and restores only the final update's scale.
This preserves representable updates when <code>learningRate * correctedFirst</code> would overflow or
underflow. Scale restoration uses integer float32 encoding, with nearest-even rounding at
subnormal boundaries; device flushing of subnormal arithmetic results can still apply.</p>
<p>Pass finite current gradients, a scalar current loss and <code>parameters</code> packed by
<code>getGPUAdamStepParameterValues</code> (a float32 <code>GPUParameterBuffer</code> can host it). The packer accepts
<code>learningRate</code>, <code>beta1</code>, <code>beta2</code>, <code>epsilon</code>, <code>tolerance</code> and <code>skipZeroGradient</code>. Defaults are
0.001, 0.9, 0.999, 1e-8, zero (convergence disabled) and false. Values can change without recompiling.
Changing betas during a fit uses the new beta raised to the global step count; keep them fixed for
standard Adam. Zero betas are supported. Epsilon must be at least 1e-30.</p>
<p>Bias correction keeps the gap <code>C(t) = 1 - beta^t</code> directly, starting from <code>C(1) = 1 - beta</code>.
It composes blocks using <code>C(2n) = C(n) * (2 - C(n))</code> and
<code>C(n+m) = C(n) + (1 - C(n)) * C(m)</code>, by the bits of t (at most 14 rounds for the budget).
The sum has nonnegative terms and never subtracts an approximate power from one;
<a href="https://www.w3.org/TR/WGSL/#floating-point-accuracy">WGSL pow accuracy</a> inherits the error of
<code>exp2(t * log2(beta))</code>, which that subtraction would amplify near one. Headless regressions
compare both the corrections and single-step weights to NumPy at the first step, binary block
boundaries and the 10,000-step budget, including the largest float32 beta below one.</p>
<p>The output owns persistent <code>weights</code>, <code>firstMoment</code>, <code>secondMoment</code>, <code>lossHistory</code>, and required
uint32 status scalars <code>iterationCount</code>, <code>converged</code>, <code>iterationLimitReached</code>, <code>invalidCount</code>.
Use separate buffers. Initialize moments and all status scalars to zero and supply initial weights.
The history has <code>maximumIterations + 1</code> floats (budget 1–10,000). Entries through the latest evaluated
iterate are valid; later entries have unspecified contents. Reset state explicitly to start a new fit.</p>
<p>Each call emits two nodes: record the loss and decide whether to step, then an indirect-gated update.
The caller evaluates the model and its derivatives between steps. Call once more after the final
update with its loss to record <code>lossHistory[iterationCount]</code>; no additional update is performed.
A complete fit composes evaluation and step nodes for the budget, followed by this final evaluation
and step. Return the current weights even at the limit. The last update sets <code>converged=0</code> and
<code>iterationLimitReached=1</code>; a plateau detected at the final evaluation sets converged instead.
The helper itself never unrolls a fit or reads status back to the CPU. A composing contributor must
budget its whole graph and gate its evaluation nodes too, keeping each run to tens of dispatches.</p>
<p>A positive tolerance stops before an update when
<code>abs(loss - previousLoss) &#x3C;= tolerance * max(1, abs(previousLoss))</code>. The implementation compares
a normalized relative change to tolerance, using a magnitude ratio and the losses' signs when
<code>abs(previousLoss) >= 1</code>. The ratio uses exponent/significand decomposition, so large opposite
losses cannot overflow the difference or tolerance product and falsely indicate convergence.
Same-sign losses use their finite difference before normalization, preserving small changes
between adjacent float32 values. For smaller previous losses the denominator is one and the
direct difference stays in range.
The current loss is recorded.
Finished calls preserve optimizer state. Invalid loss or packed parameters sets <code>invalidCount=1</code>
and skips the update. Gradients, weights and moments must be finite and second moments nonnegative;
they are not scanned by the control node. <code>skipZeroGradient</code> preserves weights and moments at zero
gradient, using the global bias-correction clock (not a sparse per-entry clock).</p>
<p>{/* parameters: GPUAdamStep generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuadamstep-parameters"><code>GPUAdamStep</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUAdamStepParameterValues(settings?)</code>, writes <code>GPU_ADAM_STEP_PARAMETER_LENGTH</code> (6) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUAdamStepParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>learningRate</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>beta1</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>beta2</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>epsilon</code> (optional)</td>
<td><code>number</code></td>
<td>-</td>
</tr>
<tr>
<td><code>tolerance</code> (optional)</td>
<td><code>number</code></td>
<td>Relative loss plateau threshold; zero disables early convergence.</td>
</tr>
<tr>
<td><code>skipZeroGradient</code> (optional)</td>
<td><code>boolean</code></td>
<td>-</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpufittingminibatchorder"><code>GPUFittingMinibatchOrder</code></h2>
<p>Takes <code>rowCount</code>, three uint32 <code>parameters</code> packed by
<code>getGPUFittingMinibatchOrderParameterValues([seedLow, seedHigh], epoch)</code>, and <code>output.order</code>.
A registered Philox 4x32-10 purpose generates one 32-bit key per original row; published stable
<code>GPUSort</code> sorts keys and original row IDs. Equal keys retain source order. Every row occurs exactly
once, even on collisions; finite-key ties introduce a small bias relative to an exactly uniform
permutation. Empty input is a no-op. Change the epoch for a different reproducible order. Slice
consecutive ranges of this permutation into minibatches in the model's GPU derivative kernel;
the final batch may be short. There is no row-sized CPU shuffling. The published sort has a
128-node conservative allowance, verified at the maximum row count by the node audit.</p>
<p>{/* parameters: GPUFittingMinibatchOrder generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpufittingminibatchorder-parameters"><code>GPUFittingMinibatchOrder</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUFittingMinibatchOrderParameterValues(seed, epoch)</code>, writes the <code>uint32</code> parameter view.
<ul>
<li><code>seed</code> (<code>readonly [number, number]</code>)</li>
<li><code>epoch</code> (<code>number</code>)</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h2 id="gpufittingcoordinateencoder"><code>GPUFittingCoordinateEncoder</code></h2>
<p>Takes scalar-packed <code>coordinates</code>, <code>rowCount</code>, optional <code>mask</code>, <code>spatialContext</code>, and
<code>output.features</code>/<code>output.validity</code>. Invalid/masked rows are zero-filled with validity zero. The
encoding, row count, widths and spatial context are compile-time; coordinates, masks and caller
frequency vectors are per-frame. Output is row-major, one scalar float per feature.</p>
<p>For <code>encoding: 'fourier'</code>, specify <code>dimensions</code> (2 or 3), <code>frequencyCount</code> (1–256) and caller-supplied
<code>frequencies</code> shaped <code>[frequencyCount, dimensions]</code>, in cycles per normalized coordinate unit.
The output interleaves <code>[cos(2π b·x), sin(2π b·x)]</code> per frequency, unit amplitude, as in
<a href="https://arxiv.org/pdf/2006.10739">Tancik et al., page 2</a>. Use a planar context and normalized local
coordinates (with an optional normalized third time/height component). No CRS transformation,
normalization or frequency sampling is implicit. A non-finite coordinate or phase, or absolute
phase above 8192 radians, invalidates the whole row.</p>
<p>For <code>encoding: 'spherical-harmonics'</code>, specify <code>maximumDegree</code> (0–39) and explicitly supply a
longitude/latitude context with metric <code>none</code> or <code>great-circle</code>. Coordinates are longitude then
latitude in <strong>degrees</strong>, in [-180,180] × [-90,90]. There are <code>(maximumDegree + 1)²</code> features.
The degree-39 limit supports 1600 basis values but does not imply compatibility with a pretrained
model's basis convention. No pretrained model is supplied.</p>
<p>The complex basis is <a href="https://dlmf.nist.gov/14.30.E1">DLMF 14.30.1</a>, including the Condon-Shortley
phase. For each degree l, write orders m=-l..l at index <code>l*l+l+m</code>:</p>
<table>
<thead>
<tr>
<th>Order</th>
<th>Real feature</th>
</tr>
</thead>
<tbody>
<tr>
<td>m &#x3C; 0</td>
<td>sqrt(2) Im(Y_l^abs(m))</td>
</tr>
<tr>
<td>m = 0</td>
<td>Y_l^0</td>
</tr>
<tr>
<td>m > 0</td>
<td>sqrt(2) Re(Y_l^m)</td>
</tr>
</tbody>
</table>
<p>The associated Legendre recurrence is <a href="https://dlmf.nist.gov/14.10.E3">DLMF 14.10.3</a>, with
normalization carried through each step to avoid factorial overflow. The diagonal follows
<a href="https://dlmf.nist.gov/14.7.E8">Rodrigues' formula</a>. Latitude uses <code>sin(latitude)</code> and <code>cos(latitude)</code>
to preserve polar offsets. Both poles and both representations of the antimeridian are covered.
The NumPy/SciPy fixture generator under <code>test/analysis-core/fitting</code> compares this convention to
<a href="https://docs.scipy.org/doc/scipy-1.15.1/reference/generated/scipy.special.sph_harm.html"><code>scipy.special.sph_harm</code></a>,
whose arguments are azimuth then colatitude in radians. Converting from another basis convention
is the caller's responsibility. GPU tolerances are stated in the headless specs.</p>`};export{e as default};