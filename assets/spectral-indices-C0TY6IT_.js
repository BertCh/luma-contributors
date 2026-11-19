var e={slug:`api/gpu-raster/spectral-indices`,title:`Calibrated spectral indices`,html:`<h1 id="calibrated-spectral-indices">Calibrated spectral indices</h1>
<p><code>GPURasterNDWI</code>, <code>GPURasterNBR</code>, <code>GPURasterEVI</code>, and <code>GPURasterSAVI</code> compose
<code>GPURasterBandMath</code> into graph-native recipes. They accept co-registered buffer bands,
caller-owned float32 outputs, and a separate uint32 validity output. They do not upload,
submit, read back, resample, choose sensor bands, or decode cloud/quality bits.
<code>GPURasterNDVI</code> remains the existing vegetation normalized-difference contributor.</p>
<p>Add these producers with <code>graph.add(new GPURasterNBR(props))</code> (likewise NDWI, EVI and SAVI).
<code>getCommandNodes(graph)</code> returns their ordered passes without scheduling them; <code>addToGraph</code> has
been removed. EVI returns six band-math passes, SAVI three, and NDWI/NBR one each.
Repeated inspection on the same graph reuses those passes. It is safe to inspect with
<code>getCommandNodes(graph)</code> and then schedule once with <code>graph.add(producer)</code>.
The returned list is frozen and readonly; attempts to remove or reorder inspected passes cannot
change what a subsequent add schedules.</p>
<p>Each producer is implemented in its own file under <code>raster-analysis/spectral-indices</code>, sharing the
band-math adapter. The established <code>gpu-raster-spectral-indices</code> source module forwards the same
four classes and props types; package imports, pass IDs and pass order are preserved.</p>
<h2 id="formula-and-band-contracts">Formula and band contracts</h2>
<p>All symbols below denote calibrated, dimensionless reflectance. Outputs are dimensionless
ratios, without the integer scaling used by some delivered satellite products.</p>
<table>
<thead>
<tr>
<th>Contributor</th>
<th>Required named bands</th>
<th>Formula</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>GPURasterNDWI</code>, <code>method: 'mcfeeters'</code></td>
<td><code>green</code>, <code>nearInfrared</code></td>
<td><code>(green - nearInfrared) / (green + nearInfrared)</code></td>
</tr>
<tr>
<td><code>GPURasterNDWI</code>, <code>method: 'gao'</code></td>
<td><code>nearInfrared</code>, <code>shortwaveInfrared</code></td>
<td><code>(nearInfrared - shortwaveInfrared) / (nearInfrared + shortwaveInfrared)</code></td>
</tr>
<tr>
<td><code>GPURasterNBR</code></td>
<td><code>nearInfrared</code>, <code>shortwaveInfrared2</code></td>
<td><code>(nearInfrared - shortwaveInfrared2) / (nearInfrared + shortwaveInfrared2)</code></td>
</tr>
<tr>
<td><code>GPURasterEVI</code></td>
<td><code>nearInfrared</code>, <code>red</code>, <code>blue</code></td>
<td><code>2.5 * (nearInfrared - red) / (nearInfrared + 6 * red - 7.5 * blue + 1)</code></td>
</tr>
<tr>
<td><code>GPURasterSAVI</code></td>
<td><code>nearInfrared</code>, <code>red</code></td>
<td><code>(1 + soilAdjustment) * (nearInfrared - red) / (nearInfrared + red + soilAdjustment)</code></td>
</tr>
</tbody>
</table>
<p>NDWI requires an explicit method. OTB calls the Gao recipe <code>Water:NDWI</code> and the
McFeeters open-water recipe <code>Water:NDWI2</code>; those names describe different band choices.
See <a href="https://www.orfeo-toolbox.org/CookBook/Applications/app_RadiometricIndices.html">OTB RadiometricIndices 10.0</a>.
The original Gao recipe uses channels near 0.86 and 1.24 micrometers. Substituting a
1.6-micrometer sensor band is an adaptation, and does not establish equivalent spectral
response. See <a href="https://doi.org/10.1016/S0034-4257(96)00067-3">Gao's original paper</a>.</p>
<p>NBR specifically uses SWIR2, around 2.2 micrometers; EVI fixes gain 2.5, red coefficient 6,
blue coefficient 7.5, and background constant 1. SAVI exposes <code>soilAdjustment</code> in <code>[0, 1]</code>,
with default 0.5. These follow the published
<a href="https://www.usgs.gov/landsat-missions/landsat-normalized-burn-ratio">USGS NBR</a>,
<a href="https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index">EVI</a>, and
<a href="https://www.usgs.gov/landsat-missions/landsat-soil-adjusted-vegetation-index">SAVI</a> formulas.
SAVI with zero adjustment has the NDVI formula; it does not replace the existing NDVI API.</p>
<p>For a concrete sensor mapping, the Landsat 8/9 OLI bands are:</p>
<table>
<thead>
<tr>
<th>Input</th>
<th>OLI band</th>
<th>Wavelength, micrometers</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>blue</code></td>
<td>2</td>
<td>0.45–0.51</td>
</tr>
<tr>
<td><code>green</code></td>
<td>3</td>
<td>0.53–0.59</td>
</tr>
<tr>
<td><code>red</code></td>
<td>4</td>
<td>0.64–0.67</td>
</tr>
<tr>
<td><code>nearInfrared</code></td>
<td>5</td>
<td>0.85–0.88</td>
</tr>
<tr>
<td>SWIR1, a possible adapted Gao input</td>
<td>6</td>
<td>1.57–1.65</td>
</tr>
<tr>
<td><code>shortwaveInfrared2</code></td>
<td>7</td>
<td>2.11–2.29</td>
</tr>
</tbody>
</table>
<p>These are the <a href="https://www.usgs.gov/faqs/what-are-best-landsat-spectral-bands-use-my-research">USGS sensor band ranges</a>,
not universal wavelength bounds or automatic sensor selection. The recipe cannot infer a
buffer's actual spectral response or physical units.</p>
<h2 id="calibration-masks-and-denominators">Calibration, masks, and denominators</h2>
<p>Each <code>GPURasterBufferBand</code> supplies <code>scale</code> and <code>offset</code>, applying
<code>reflectance = raw * scale + offset</code>. Defaults are one and zero for already calibrated
inputs. Raw integer nodata is compared in its native format before it can be lost to
float32 conversion. Each source's <code>validity</code> accepts any nonzero flag. The output
intersects every required source domain and rejects nonfinite source, calibrated,
intermediate, and result values. Invalid output samples are canonical quiet NaNs,
with an output validity flag of zero; accepted flags are one.</p>
<p>Calibration must express reflectance as a fraction, not percent, radiance, or unchanged
scaled digital numbers. EVI and SAVI contain additive reflectance constants and therefore
are not invariant to arbitrary input scaling. Even normalized differences change when
an additive calibration offset is omitted. Landsat Collection 2 surface reflectance,
for example, uses scale <code>0.0000275</code>, offset <code>-0.2</code>, and raw fill value <code>0</code>.
See the <a href="https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products">USGS scaling guidance</a>.</p>
<p><code>epsilon</code> defaults to zero. A pixel is valid only when the absolute value of its final
formula denominator is <strong>strictly greater</strong> than <code>epsilon</code>; equality is rejected.
EVI tests <code>nearInfrared + 6 * red - 7.5 * blue + 1</code>, and SAVI tests
<code>nearInfrared + red + soilAdjustment</code>. Intermediate sums are not independently subjected
to that threshold. Negative denominators are allowed. No source reflectance or output
index is implicitly clipped, so valid results can lie outside <code>[-1, 1]</code>. An optional
<code>clamp: [minimum, maximum]</code> explicitly clips finite output results after validity checks.</p>
<h2 id="usage">Usage</h2>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPURasterEVI} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/raster-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPUCommandGraph, GraphDataView} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {GPURasterBufferBand} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma.gl/graph/gpu-raster'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> addVegetationIndex</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  graph</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUCommandGraph</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  width</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  height</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  nearInfrared</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterBufferBand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  red</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterBufferBand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  blue</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterBufferBand</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  output</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'float32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">>,</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  outputValidity</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GraphDataView</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">&#x3C;</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'uint32'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">></span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> void</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  graph.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">add</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">new</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPURasterEVI</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    width,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    height,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    nearInfrared,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    red,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    blue,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    output,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    outputValidity,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">    epsilon: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">0.000001</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  }));</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">}</span></span></code></pre>
<p>Inputs and outputs must belong to the same graph and contain one packed sample per pixel.
Outputs must use separate buffers from all input samples, masks, and each other. Shared
read-only source buffers and masks are allowed. Separation uses the underlying physical buffer,
including separate graph imports: reimporting the same bytes cannot make an output independent
of a source. Supply distinct <code>id</code> values for multiple
recipes in one graph. Re-encoding a compiled graph consumes current input values and masks;
format, dimensions, calibration, methods, and scalar options are fixed at construction.
Each producer captures those fields, output views and nested source-band/storage descriptors once,
including inherited getters. Later caller-map mutation cannot replace a validated output with a
physical source alias or change the declared band bindings. Input buffer contents remain live per run.
All imported buffers remain caller-owned. Intermediate buffers belong to the graph.</p>
<h2 id="composition-and-verification">Composition and verification</h2>
<p>NDWI and NBR each add one band-math pass; SAVI adds three. EVI adds six, including explicit
red and blue calibration stages before weighting. Every pass uses at most six storage
bindings, including all masks. Operations retain the primitive's float32 arithmetic;
this composition does not claim a fused-kernel performance advantage. Dimensions and
buffer sizes remain subject to the public raster primitive and device limits.</p>
<p>The checked-in <code>raster-spectral-indices.py</code> oracle calls <code>spyndex.computeIndex</code> from
<code>spyndex==0.12.0</code> with <code>numpy==2.2.6</code> and its bundled, offline formula catalog. Its thirteen
cases include supplied calibration, independent masks, input updates, exact denominator
boundaries, signed denominators, explicit clamping, and SAVI adjustment endpoints.
Calibration and validity are explicit adapters around those independent formula calls.
The Gao case uses the algebraically identical library NDMI formula on the caller-selected
shortwave samples; this is formula-composition evidence, not direct OTB execution or
proof of sensor-response equivalence. See the <a href="https://spyndex.readthedocs.io/en/latest/reference.html">Spyndex API</a>.</p>
<p>GPU tests compare against these independent float64 values with absolute and relative
tolerances of <code>1e-5</code>, accounting for float32 calibration and intermediate rounding.
The fixtures keep accepted stress denominators away from zero; the epsilon policy is
not an error-bound certificate for arbitrary ill-conditioned inputs. Additional tests
exercise exact uint32 nodata, nonfinite samples, nonzero buffer offsets, canonical NaNs,
re-encoding, and borrowed-buffer ownership. Node tests inspect all composed shader
sources and validation boundaries. Numerical GPU execution is a separate verification
gate from those node construction tests.</p>
<h2 id="executable-examples">Executable examples</h2>
<p>These examples use explicit synthetic inputs and expose real GPU graph captures.</p>
<h3 id="gpurasterndwi">GPURasterNDWI</h3>
<p>{/* lens:GPURasterNDWI */}</p>
<p>::lens{scenario=gpu-raster.raster-ndwi rung=graph}</p>
<h3 id="gpurasternbr">GPURasterNBR</h3>
<p>{/* lens:GPURasterNBR */}</p>
<p>::lens{scenario=gpu-raster.raster-nbr rung=graph}</p>
<h3 id="gpurasterevi">GPURasterEVI</h3>
<p>{/* lens:GPURasterEVI */}</p>
<p>::lens{scenario=gpu-raster.raster-evi rung=graph}</p>
<h3 id="gpurastersavi">GPURasterSAVI</h3>
<p>{/* lens:GPURasterSAVI */}</p>
<p>::lens{scenario=gpu-raster.raster-savi rung=graph}</p>`};export{e as default};