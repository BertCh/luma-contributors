var e={slug:`api/gpu-raster/raster-embedding-codecs`,title:`Embedding raster codecs`,html:`<h1 id="embedding-raster-codecs">Embedding raster codecs</h1>
<p>Import from <code>@luma-contributors/analysis/raster-analysis</code>. These codecs convert caller-supplied,
pixel-major arrays; they do not run a model or include weights or datasets. Related analysis operators
are described in <a href="/luma-contributors/docs/api/gpu-raster/raster-embeddings">Embedding rasters</a>.</p>
<h2 id="gpurasterembeddingdequantize"><code>GPURasterEmbeddingDequantize</code></h2>
<p>Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.</p>
<p>The existing decoder accepts <code>format</code> (compile-time, default <code>'alphaearth-int8'</code>):</p>
<table>
<thead>
<tr>
<th>Format</th>
<th>Packed <code>quantized</code> input</th>
<th>Decoded component</th>
<th>Invalid pixel</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>alphaearth-int8</code></td>
<td>Four signed bytes per uint32, low byte first</td>
<td><code>sign(code) * (code / 127.5)^2</code></td>
<td>Any byte is -128</td>
</tr>
<tr>
<td><code>tessera-int8</code></td>
<td>Four signed bytes per uint32, low byte first</td>
<td><code>float32(code) * scales[pixel]</code></td>
<td>Scale or decoded component is non-finite</td>
</tr>
<tr>
<td><code>float16</code></td>
<td>Two IEEE binary16 components per uint32, low half first</td>
<td>IEEE binary16 to float32</td>
<td>Any component has exponent 31 (NaN or infinity)</td>
</tr>
</tbody>
</table>
<p>Every format also accepts a per-pixel uint32 <code>mask</code>; zero invalidates the entire pixel. Invalid pixels
write NaN to every output component and zero to optional <code>output.validity</code>. Other pixels write validity
one, including all-zero vectors. TESSERA uses all signed byte codes: -128 is a value, not a sentinel.
Zero and negative finite scales are multiplied as supplied. A TESSERA scale is one float32 per pixel;
per-component scales are not supported. <code>scales</code> is required only for TESSERA and rejected otherwise.</p>
<p>The TESSERA formula matches the authors' <a href="https://github.com/ucam-eo/geotessera/blob/main/geotessera/core.py#L30-L61">GeoTessera <code>dequantize_embedding</code></a>
(verified 2026-11-14). Binary16 follows the <a href="https://www.w3.org/TR/WGSL/#floating-point-types">WGSL floating-point representation</a>:
normal magnitudes are <code>(1024 + fraction) * 2^(exponent - 25)</code> and subnormals are <code>fraction * 2^-24</code>.
The decoder reads integer fields, preserving half-float subnormals without <code>shader-f16</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterEmbeddingDequantize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pixelCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  dimensions: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  format: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'tessera-int8'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  quantized: packedSignedBytes,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  scales: pixelScales,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mask,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  normalize: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {vectors, validity}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>normalize: true</code> divides finite nonzero vectors by their length, scaling by the largest magnitude
first to avoid overflow. A zero vector stays zero. The default preserves decoded magnitudes. Each word
array has exactly <code>ceil(pixelCount * dimensions / 4)</code> words for int8 or
<code>ceil(pixelCount * dimensions / 2)</code> for binary16. Unused lanes of the final word are ignored, even when
they contain a sentinel or NaN.</p>
<h2 id="gpurasterembeddingquantize"><code>GPURasterEmbeddingQuantize</code></h2>
<p>Aligned GraphVectorView pixel records support chunked input with 1–1,024 components; whole embedding
records stay together. Reference covariance-family dimensions remain 128.</p>
<p>{/* lens:GPURasterEmbeddingQuantize */}</p>
<p>::lens{scenario=gpu-raster.raster-embedding-quantize rung=formula}</p>
<p>The synthetic lens encodes three three-component pixels, including saturation, a masked pixel, a nonfinite pixel and zeroed unused final bytes.</p>
<p>Encodes float32 vectors to the AlphaEarth int8 format, including vectors produced by
<code>GPURasterEmbeddingPyramidLevel</code>. The formula is
<code>clip(roundToEven(sign(value) * sqrt(abs(value)) * 127.5), -127, 127)</code>, from
<a href="https://arxiv.org/html/2507.22291v2#S22.SS1">AlphaEarth Foundations, supplementary S22.1</a>
(power 2, scale 127.5; verified 2026-11-14). Rounding follows nearest-even, including negative ties.
Finite components outside [-1, 1] saturate. The encoder does not normalize vectors.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterEmbeddingQuantize</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  pixelCount,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  dimensions: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">64</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  vectors: overviewVectors,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  mask: overviewValidity,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  output: {quantized: packedSignedBytes, validity}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}));</span></span></code></pre>
<p><code>vectors</code> has exactly <code>pixelCount * dimensions</code> float32 entries. <code>mask</code> and <code>output.validity</code> are
optional, each one uint32 per pixel. A masked pixel or any non-finite input component writes -128 in
every band and validity zero. Otherwise validity is one. <code>output.quantized</code> has exactly
<code>ceil(pixelCount * dimensions / 4)</code> uint32 words; unused final bytes are zero. The validity pass uses
one invocation per pixel, then the packing pass uses one per word, so odd vector lengths cannot cause
concurrent writes to the same word. Omitting <code>output.validity</code> allocates a pixelCount-word scratch.</p>
<h2 id="limits-and-verification">Limits and verification</h2>
<p>Both classes accept positive integer pixel counts and 1–1,024 dimensions; total pixelCount is at most 4,294,967,295
across all chunks. Chunking cannot bypass this total uint32 row limit. The flattened component
count is at most 2^32 - 1 per chunk for uint32 addressing; the total across chunks may exceed it.
All supplied pixel fields must have matching chunk counts and complete-pixel boundaries. Packed
int8/binary16 words cannot cross a chunk boundary; only the final nonempty chunk may have word padding.
Each chunk binding, including encoder scratch, is checked against the device's storage binding limit
before nodes are created. A 262,144-pixel, 128-component float32 chunk occupies 128 MiB; split larger
arrays into aligned chunks on default devices. Widths 192, 256, 384 and 1,024 are covered by the
declaration and shader-audit scale spec; the 128-dimension limit belongs to reference covariance
operators, not these codecs. Decode takes one dispatch and encode takes two per nonempty chunk.
The per-contributor command budget is <code>1 + dispatchesPerChunk × nonemptyChunks ≤ 32,768</code>:</p>
<table>
<thead>
<tr>
<th>Codec</th>
<th>Dispatches per chunk</th>
<th>Maximum nonempty chunks</th>
<th>First rejected chunk count</th>
</tr>
</thead>
<tbody>
<tr>
<td>Encode</td>
<td>2</td>
<td>16,383</td>
<td>16,384</td>
</tr>
<tr>
<td>Decode</td>
<td>1</td>
<td>32,767</td>
<td>32,768</td>
</tr>
</tbody>
</table>
<p>Empty chunks do not consume dispatches; every binding and native packed-word boundary must still fit.
Shapes, chunk topology, formats, normalization and optional view presence are
compile-time; all view contents are per-frame. No parameter packer is needed. These raster operations
remain GPU-only, as recommended by the CPU suitability survey.</p>
<p>The independent NumPy fixture generator is
<code>modules/analysis/test/oracles/emb01-embedding-codecs-numpy.py</code>; run it with <code>/usr/bin/python3</code>.
The headless specs compare decoded/normalized vectors and exact packed output on signed extrema,
nearest-even rounding, saturation, zero and non-finite scales, masks, half-float subnormals and
exceptional values, partial workgroups and final-word padding. The existing AlphaEarth suite remains
regression coverage.</p>
<p>Presto uint16 decoding is <strong>NOT DONE</strong>: the <a href="https://nasaharvest.github.io/presto-embeddings/">author's pipeline page</a>
and <a href="https://huggingface.co/datasets/izvonkov/Togo_Presto_Embeddings">dataset card</a> confirm storage type,
but provide no scale/offset formula. The linked generation script redirects to Google sign-in. A
verified quantization formula is required before adding a Presto format; no scaling is inferred here.</p>`};export{e as default};