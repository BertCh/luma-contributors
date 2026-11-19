var e={slug:`api/gpu-raster/raster-embeddings`,title:`Embedding rasters`,html:`<h1 id="embedding-rasters">Embedding rasters</h1>
<p>Contributors for rasters of learned embedding vectors, such as the AlphaEarth Satellite Embedding V1
annual layers (64 bands, 10 m, unit-length vectors): decoding the int8 files, similarity search, change
between years, clustering, classification, principal components and per-polygon descriptors, and
overviews that average vectors correctly. Import from <code>@luma-contributors/analysis/raster-analysis</code>.</p>
<p>K-means and nearest-pixel search follow the same producer API as the other embedding contributors:
use <code>graph.add(contributor)</code> or obtain unscheduled passes with <code>contributor.getCommandNodes(graph)</code>.
Their former <code>addToGraph</code> method has been removed.
Inspecting their nodes repeatedly on the same graph reuses the materialized passes; an inspection
can be followed by <code>graph.add(contributor)</code> without consuming the producer or duplicating scratch.
The cached node lists are frozen and returned as readonly arrays; inspecting them cannot change
the passes that a later <code>graph.add</code> schedules. The same rule applies to the four spectral indices.</p>
<p>Layout. A raster is <code>pixelCount</code> pixels, row-major, each with <code>dimensions</code> float32 components (1–1,024 for streaming contributors; 1–128 for the reference covariance family):
<code>vectors[pixel * dimensions + band]</code>. An invalid pixel is marked by <code>mask</code> (zero) or by a non-finite
component; the decoder writes NaN for every component of a NoData pixel. int8 data is <code>uint32</code> words of four
little-endian signed bytes, pixel-major, as an <code>Int8Array</code> uploaded unchanged. Band-sequential files are
interleaved by the host adapter. The element count <code>pixelCount * dimensions</code> must stay below 2^32, and a
128-band float32 raster of 262,144 pixels fills one 128 MiB storage binding on a default device; tile larger
rasters.</p>
<p>Reference. NumPy in binary64, written from the documented definitions
(<code>test/oracles/rts03-embedding-rasters-numpy.py</code>). scikit-learn was not installed, so the kNN, logistic
regression and PCA semantics are written from its documentation and nothing was run or copied from it.
Additional verified TESSERA int8 and binary16 decoding, masks, and AlphaEarth int8 encoding are
described in <a href="/luma-contributors/docs/api/gpu-raster/raster-embedding-codecs">Embedding raster codecs</a>. Model inference
(TerraMind, Prithvi, Clay) and training are outside these codecs.</p>
<h2 id="gpurasterembeddingdequantize"><code>GPURasterEmbeddingDequantize</code></h2>
<p>Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.</p>
<p>AlphaEarth stores each component as an int8 <code>q</code> and decodes it as <code>sign(q) * (q / 127.5)^2</code>, with <code>q = -128</code>
as NoData; a pixel with NoData in any band is invalid. The decoded vector has length close to one, not
exactly one, because of the 8-bit rounding; set <code>normalize</code> to divide by the length so dot products are
exact cosines. <code>output.validity</code> is optional.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterEmbeddingDequantize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pixelCount: width </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> height,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  dimensions: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  quantized: int8Words, </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// an uploaded Int8Array</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  normalize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {vectors, validity}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<h2 id="gpurasterembeddingsimilarity"><code>GPURasterEmbeddingSimilarity</code></h2>
<p>Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.</p>
<p>Cosine (or dot) similarity of every pixel to up to 64 query vectors: <code>similarity</code> for every pixel and
query, the <code>maximum</code> over the queries, the <code>bestQuery</code> (the lowest on ties) and a <code>matches</code> mask for a
per-frame threshold. Queries come from clicked pixels or from <code>GPURasterEmbeddingZoneMean</code> (the mean of a
polygon, renormalized). Cosine follows the published <code>GPUSimilaritySearch</code>: two zero vectors have
similarity one and exactly one gives zero.</p>
<p>{/* parameters: GPURasterEmbeddingSimilarity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterembeddingsimilarity-parameters"><code>GPURasterEmbeddingSimilarity</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterEmbeddingSimilarityParameterValues(settings)</code>, writes <code>GPU_RASTER_EMBEDDING_SIMILARITY_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterEmbeddingSimilarityParameterValues</code>:</p>
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
<td><code>matchThreshold</code></td>
<td><code>number</code></td>
<td>A pixel matches when its best similarity is at least this value.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterembeddingnearestpixels"><code>GPURasterEmbeddingNearestPixels</code></h2>
<p>"Find places like this one": the <code>resultCount</code> pixels most similar to each query, ranked by cosine, as pixel
indices and scores. It runs the published exact <code>GPUSimilaritySearch</code> of
<code>@luma.gl/graph/gpu-vector-search</code> on the raster's vectors, so equal scores order by pixel index and the
query-by-pixel matrix is never stored.
<code>output.pixelRows</code> contains physical zero-based raster rows, with <code>0xffffffff</code> for missing matches.
Kepler exposes these as the <code>pixel_row</code> column; these values are not caller identities.</p>
<h2 id="gpurasterembeddingchange"><code>GPURasterEmbeddingChange</code></h2>
<p>Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.</p>
<p>The use Google recommends for AlphaEarth: the dot product between two years. Outputs are the cosine
similarity, the angle between the vectors (computed as <code>2 atan2(|u - v|, |u + v|)</code> of the two unit
vectors, not <code>acos</code> of the float32 cosine, so angles far below 5e-4 rad are resolved), the length of the
change vector <code>|after - before|</code> and a mask where the
cosine is below a per-frame threshold. A pixel invalid in either year is NaN. Combine the similarity with
<code>GPURasterFalseDiscoveryRateMask</code> or a percentile threshold when a map must show only significant change.</p>
<p>{/* parameters: GPURasterEmbeddingChange generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterembeddingchange-parameters"><code>GPURasterEmbeddingChange</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterEmbeddingChangeParameterValues(settings)</code>, writes <code>GPU_RASTER_EMBEDDING_CHANGE_PARAMETER_LENGTH</code> (1) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterEmbeddingChangeParameterValues</code>:</p>
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
<td><code>similarityThreshold</code></td>
<td><code>number</code></td>
<td>A pixel is changed when its cosine similarity between the two years is below this value.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterembeddingkmeans"><code>GPURasterEmbeddingKMeans</code></h2>
<p>Unsupervised clustering with the published <code>GPUKMeans</code> of <code>@luma.gl/graph/gpu-vector-search</code>: deterministic
evenly spaced valid rows as seeds (not k-means++), up to 64 Lloyd iterations encoded and skipped once the
labels stop changing, Euclidean centres and an optional <code>[iterations, changedLabels, converged]</code> status.
Normalize the vectors first to cluster by direction. This is a different contract from the planar
<code>GPUSpatialKMeans</code> of <code>gpu-spatial-analysis</code>. The check compares a converged run on separated blobs with NumPy's
blob means; it does not reproduce scikit-learn's seeding.
The optional status is <code>[executedIterations, changedLabels, converged]</code>, with <code>converged = 1</code> only
when labels stopped changing. Reaching the iteration budget with <code>converged = 0</code> leaves a provisional
clustering. Kepler requests this status for its fixed 32-round budget and refuses to publish the
label column if convergence was not reached; the error reports executed rounds and changed labels.</p>
<h2 id="gpurasterembeddingknearestclassifier"><code>GPURasterEmbeddingKNearestClassifier</code></h2>
<p>Default searchStructure tiled performs exact exhaustive training in 256-row tiles. Brute-force
retains 4,096 training rows. Training count times dimensions must fit uint32 addressing, device
bindings and the command budget; aligned whole-pixel chunks support 1–1,024 dimensions.</p>
<p>Exact tiled k-nearest-neighbour classification against a labelled training set, with
scikit-learn's <code>KNeighborsClassifier</code> semantics: uniform votes among the <code>neighborCount</code> (at most 16)
nearest, the smallest class on a vote tie, the lower sample on equal distances. <code>metric</code> is <code>'euclidean'</code>
or <code>'cosine'</code> (distance <code>1 - cosine</code>; two zero vectors have similarity 1 and exactly one zero vector gives
0, as in the published <code>GPUSimilaritySearch</code>). Outputs are the label, the vote share of the winner and the share of
every class. The cost is <code>pixelCount x trainingCount x dimensions</code>.</p>
<h2 id="gpurasterembeddinglinearprobe"><code>GPURasterEmbeddingLinearProbe</code></h2>
<p>Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.</p>
<p>Inference of a host-trained linear probe: <code>softmax(W x + b)</code>, the prediction of a multinomial
<code>LogisticRegression</code> (<code>coef_</code>, <code>intercept_</code>) with up to 64 classes, computed with the maximum subtracted so
large logits do not overflow. A binary scikit-learn model has one row; pass the two rows <code>[0, w]</code> with
intercepts <code>[0, b]</code>.</p>
<h2 id="pca-to-rgb">PCA to RGB</h2>
<p><code>GPURasterEmbeddingProjection</code>: Aligned GraphVectorView pixel records support chunked input with
1–1,024 components; whole embedding records stay together. Reference covariance-family dimensions
remain 128.</p>
<p>Three steps, with only the eigenproblem on the CPU:</p>
<ol>
<li><code>GPURasterEmbeddingCovariance</code> writes the mean vector and the sample covariance (divisor <code>n - 1</code>) of the
usable pixels. The sums are two-pass and in fixed-order partial ranges (<code>rangeCount</code> up to 256), so the
result is reproducible.</li>
<li><code>getRasterEmbeddingPrincipalComponents(covariance, dimensions, componentCount)</code> finds the components in
binary64 (cyclic Jacobi, at most 128 x 128). The sign of each component makes its largest loading
positive, scikit-learn's <code>svd_flip(u_based_decision=False)</code> convention, so maps agree with a NumPy or
scikit-learn reference. <code>getRasterEmbeddingStretchRanges(eigenvalues, 2)</code> gives a two-sigma stretch.</li>
<li><code>GPURasterEmbeddingProjection</code> writes the scores <code>(x - mean) . component</code> and, for three or more
components, an RGB image stretched to <code>[0, 1]</code>.</li>
</ol>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">components</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">eigenvalues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getRasterEmbeddingPrincipalComponents</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(covarianceReadback, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> stretchRanges</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getRasterEmbeddingStretchRanges</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(eigenvalues, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> parameters</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> getGPURasterEmbeddingProjectionParameterValues</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({stretchRanges});</span></span></code></pre>
<p>{/* parameters: GPURasterEmbeddingProjection generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpurasterembeddingprojection-parameters"><code>GPURasterEmbeddingProjection</code> parameters</h3>
<ul>
<li>Packer: <code>getGPURasterEmbeddingProjectionParameterValues(settings)</code>, writes <code>GPU_RASTER_EMBEDDING_PROJECTION_PARAMETER_LENGTH</code> (6) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPURasterEmbeddingProjectionParameterValues</code>:</p>
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
<td><code>stretchRanges</code></td>
<td><code>readonly (readonly [number, number])[]</code></td>
<td><code>[low, high]</code> of the first three components, mapped to <code>[0, 1]</code> and clamped. Use <code>getRasterEmbeddingStretchRanges(eigenvalues)</code> for a two-sigma stretch.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpurasterembeddingzonemean"><code>GPURasterEmbeddingZoneMean</code></h2>
<p>The mean embedding of each polygon, weighted by coverage and renormalized to unit length: a descriptor to
search with, or to compare zones. Pixels come from a <code>zoneRaster</code> (one id per pixel, weight one) or from
the compact tuples of <code>GPURasterFractionalCoverage</code> (<code>zoneIds</code>, <code>cellIds</code>, <code>fractions</code>, <code>status.count</code>), so a
border pixel shared by two polygons counts toward each in proportion to its overlap. Sums are 96-bit
fixed-point integers added atomically, so they do not depend on thread order; the only rounding is the
conversion to float32. <code>output.meanLengths</code> is the length of the mean before renormalization, 1 when the
zone points one way and near 0 when its pixels cancel. A zone without usable weight is NaN.</p>
<h2 id="gpurasterembeddingpyramidlevel"><code>GPURasterEmbeddingPyramidLevel</code></h2>
<p>The unit-vector-mean overview. The generic mean overview (<code>GPURasterOverview</code>, <code>gdaladdo -r average</code>) is wrong
for these data: on raw int8 values it averages a nonlinear code, and on decoded vectors the mean of unit
vectors is shorter than one, which biases later dot products. AlphaEarth's own rule is to average the
decoded vectors and renormalize to unit length; this contributor does that for each footprint (partial
footprints at odd edges, scale 1 to 16 per axis) from either <code>vectors</code> or the packed <code>quantized</code> input.
<code>output.meanLength</code> and <code>validCount</code> say how coherent and how complete each footprint was.</p>
<p>Compute each overview from the original vectors, or request <code>output.vectorSums</code> and
<code>output.validCount</code> and pass that pair to the next level as
<code>aggregation: {vectorSums, validCount}</code>.</p>
<p>Give exactly one input mode: <code>vectors</code>, <code>quantized</code>, or <code>aggregation</code>. Aggregation sums contain
<code>sourceWidth * sourceHeight * dimensions</code> float32 components; counts contain one uint32 per child
pixel. Original-pixel counts must total at most 2^32 - 1 in each output footprint. Every level
retains the 1–16 scale limit per axis and 1–128 dimensions; sums/counts allow arbitrarily many
levels within the count and storage limits. Chaining preserves original-pixel weights up to
float32 summation rounding. Feeding normalized overview vectors into <code>vectors</code> gives each child
equal weight and loses both support size and concentration: <code>[east, east, north]</code> becomes
<code>(east + north) / sqrt(2)</code> after two scale-two reductions, instead of <code>(2 east + north) / sqrt(5)</code>.</p>
<p>An empty footprint has sums zero, count zero, mean length NaN and vector validity 0. An exactly
cancelling footprint has sums zero, a positive count, mean length zero and vector validity 0.
Its normalized vector is NaN, but its sums/counts remain usable for the next level. Do not use
the child's vector validity as the aggregation mask: that would discard cancellation support.
An explicit aggregation mask excludes whole child footprints.
Normalization scales by the largest absolute sum component before computing the squared norm.
For sums <code>[1, 0]</code> and <code>[-1, 2^-80]</code> with one supporting pixel each, the result retains count 2,
sum <code>[0, 2^-80]</code>, valid north direction <code>[0, 1]</code> and mean length <code>2^-81</code>; squaring that mean
directly in float32 would incorrectly classify it as cancellation. Sizes and nested aggregation/output
port maps are captured at construction, while GPU views retain their per-frame contents.</p>
<p>Writing an overview back as int8 needs the inverse of the decoder,
<code>q = sign(v) round(127.5 sqrt|v|)</code>, provided by <code>GPURasterEmbeddingQuantize</code> (nearest-even rounding with saturation).</p>`};export{e as default};