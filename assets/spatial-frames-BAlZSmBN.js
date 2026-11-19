var e={slug:`api/gpu-spatial-analysis/spatial-frames`,title:`Spatial frames`,html:`<h1 id="spatial-frames">Spatial frames</h1>
<p>A <code>GPUSpatialFrame</code> describes the projected planar coordinate space of a set of rows. It is carried
by <code>GPUSpatialContext.frame</code> and is how planar operators and projection graphs reject data that was
projected differently.</p>
<h2 id="frame-model">Frame model</h2>
<p>A frame is an identity plus optional math.gl CRS metadata:</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">type</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialFrame</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  id</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> string</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// identity; equal ids mean interchangeable coordinates</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  kind</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'local-azimuthal-equidistant'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'local-lambert-equal-area'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'crs'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  origin</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">]; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// [longitude, latitude] in degrees</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  crs</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> string</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// identifier as written, for kind 'crs'</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  signature</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> string</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// execution signature of resolved math.gl CRS metadata</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  unit</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> GPUSpatialFrameUnit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// math.gl CRSUnit: name, metersPerUnit, degreesPerUnit</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  axisOrder</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> string</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">; </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// for example 'enu'</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  areaOfUse</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> readonly</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> [</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">west</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">south</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">east</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">north</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">: </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">number</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">];</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  anchor</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'projection-center'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'relative-origin'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#E36209;--shiki-dark:#FFAB70">  precision</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">?:</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'float32'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'float32-origin-relative'</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> |</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> 'float64-split'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">};</span></span></code></pre>
<p>Compatibility between two contexts is decided by comparing frame <code>id</code> only. Every other field is
descriptive metadata and never changes whether two frames are compatible. Two frames with
different ids are incompatible even if they describe the same CRS; build both with the same
constructor and options to get the same id.</p>
<ul>
<li><code>anchor: 'projection-center'</code> means the projection is centered on <code>origin</code> (local frames).</li>
<li><code>anchor: 'relative-origin'</code> means coordinates are the CRS's own, stored relative to <code>origin</code> to
keep float32 precision.</li>
</ul>
<h3 id="signature-and-semantic-equivalence">Signature and semantic equivalence</h3>
<p><code>signature</code> is the math.gl <code>getCRSSignature</code> of resolved CRS metadata. It compares execution
conventions only (the parameters that determine how coordinates are computed). It is not a proof
that two CRS definitions are semantically the same CRS, and it is not used for compatibility.
Use it to tell whether two frames execute identically; use <code>id</code> to decide whether data may be
mixed. Datum or area-of-use differences that do not change execution are not reflected in it.</p>
<h2 id="units-and-precision">Units and precision</h2>
<p><code>unit</code> is math.gl's <code>CRSUnit</code>. The library checks declared units and never converts them.
<code>getGPUSpatialFrameContextUnits</code> returns <code>'meters'</code> for <code>metersPerUnit === 1</code> and otherwise leaves
<code>GPUSpatialContext.units</code> undeclared (native), as required by the native planar context validator.
Kilometers and other units remain in <code>frame.unit</code>; coordinates and thresholds continue to use
those native frame units without conversion.</p>
<p>Local frames are in meters, centered on their origin and held origin-relative in float32
(<code>precision: 'float32-origin-relative'</code>). The origin is subtracted in binary64 on the CPU, so float32
frame coordinates keep about 1e-7 of the frame extent (millimeters at 10 km, centimeters at 100 km),
instead of the 0.4 to 1 m quantization of float32 longitude/latitude.</p>
<h2 id="distance-frames-and-area-frames">Distance frames and area frames</h2>
<p>Local frames come in two kinds, listed in <code>GPU_LOCAL_FRAME_KINDS</code>:</p>
<table>
<thead>
<tr>
<th><code>kind</code></th>
<th>Frame kind</th>
<th>Projection</th>
<th>Use for</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>'azimuthal-equidistant'</code></td>
<td><code>local-azimuthal-equidistant</code></td>
<td>Spherical AEQD on the mean Earth radius (or <code>sphereRadius</code>)</td>
<td>Distance bands, neighbor radii, buffers. Distances and bearings from the origin are exact great-circle values; other distances are within about 0.1% up to 200 km.</td>
</tr>
<tr>
<td><code>'lambert-equal-area'</code></td>
<td><code>local-lambert-equal-area</code></td>
<td>Ellipsoidal (WGS84) LAEA, <code>+proj=laea +ellps=WGS84</code></td>
<td>Densities, areal weights, dot sampling. Areas are exact ellipsoidal areas at any scale.</td>
</tr>
</tbody>
</table>
<p><code>GPU_LOCAL_FRAME_KINDS</code> maps each local kind to its <code>GPUSpatialFrameKind</code>. Distance operators
reject Lambert frames and area operators reject AEQD frames.</p>
<h2 id="creating-frames">Creating frames</h2>
<h3 id="creategpulocalframe">createGPULocalFrame</h3>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {createGPULocalFrame} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> frame</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> createGPULocalFrame</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">({origin: [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">122.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">37.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">], kind: </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'azimuthal-equidistant'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">});</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> frame.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">project</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">122.39</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">37.81</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// meters, binary64</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> positions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> frame.</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0">projectArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(lngLats, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">); </span><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// interleaved float32 [x, y]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// frame.spatialContext carries units 'meters' and the frame identity.</span></span></code></pre>
<p><code>createGPULocalFrame(props)</code> returns a binary64 CPU-side frame with <code>project</code>, <code>unproject</code>,
<code>projectArray</code> and <code>unprojectArray</code>. Props: <code>origin</code> (required, finite, <code>|latitude| &#x3C;= 90</code>), <code>kind</code>
(default <code>'azimuthal-equidistant'</code>), <code>sphereRadius</code> (AEQD only; default mean Earth radius),
<code>id</code> and <code>datum</code>. The <code>datum</code> is copied to <code>spatialContext.datum</code> and does not change the frame id.
Antipodes project to NaN.</p>
<h3 id="creategpulocalspatialframe">createGPULocalSpatialFrame</h3>
<p>Creates only the <code>GPUSpatialFrame</code> descriptor (meters, <code>axisOrder: 'enu'</code>,
<code>anchor: 'projection-center'</code>, <code>precision: 'float32-origin-relative'</code>) without a transform. The
default id is <code>local-azimuthal-equidistant(lng,lat)</code> or <code>local-lambert-equal-area(lng,lat)</code>; an AEQD
frame with a non-default <code>sphereRadius</code> appends <code>,R=&#x3C;radius></code>.</p>
<h3 id="creategpuspatialframefromcrs">createGPUSpatialFrameFromCRS</h3>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {createGPUSpatialFrameFromCRS} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/analysis-core'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">const</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF"> frame</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583"> =</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0"> createGPUSpatialFrameFromCRS</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">(</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF">'urn:ogc:def:crs:EPSG::32633'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">);</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D">// frame.id === 'EPSG:32633'</span></span></code></pre>
<p>Creates a <code>'crs'</code> frame synchronously, without a CRS provider. The id is the normalized authority
identifier (<code>epsg:3857</code> and <code>urn:ogc:def:crs:EPSG::3857</code> both give <code>EPSG:3857</code>), else the metadata
signature, else the input string. <code>crs</code> keeps the input as written. Options: <code>metadata</code> (resolved
math.gl metadata, for example from <code>projectionEngine.getCRS</code>, giving the signature, unit, axis
order and area of use), <code>id</code> and <code>precision</code>. The unit comes from <code>metadata</code>, then the PROJ string
<code>+units</code>/<code>+to_meter</code>, then a known registry family; it is omitted when unknown.</p>
<h3 id="creategpuspatialframefromidentifier-and-getknowngpuspatialframeunit">createGPUSpatialFrameFromIdentifier and getKnownGPUSpatialFrameUnit</h3>
<p><code>createGPUSpatialFrameFromIdentifier(crs)</code> creates a <code>'crs'</code> frame with no math.gl runtime code:
the id and <code>crs</code> are the input as written. It is what <code>GPUProjection</code> uses for a plain <code>targetCrs</code>
string, so spellings are not normalized.</p>
<p><code>getKnownGPUSpatialFrameUnit(identifier)</code> returns the unit of a recognized family: <code>EPSG:4326</code> and
<code>OGC:CRS84</code> (degrees), <code>EPSG:3857</code> and WGS84 UTM <code>EPSG:326xx</code>/<code>EPSG:327xx</code> (meters). It returns
<code>undefined</code> for anything else.</p>
<h2 id="what-mathgl-provides">What math.gl provides</h2>
<p>The frame facade delegates to <code>@math.gl/geospatial/crs</code>: <code>normalizeCRSIdentifier</code>,
<code>getCRSSignature</code>, <code>parsePROJString</code> and <code>linearUnits</code>, plus the <code>CRSMetadata</code> and <code>CRSUnit</code>
types. Local frame transforms use <code>projectionEngine</code> from <code>@math.gl/geospatial/projection</code>. The
<code>gpu-project</code> execution bundle imports math.gl types only, so it stays free of math.gl CRS runtime
code; only <code>createGPUSpatialFrameFromCRS</code> pulls the runtime in.</p>
<h2 id="frames-and-the-published-projection-classes">Frames and the published projection classes</h2>
<p>The published projection entry points (<code>prepareCRSProjection</code> of <code>@luma.gl/graph/gpu-project/crs</code>,
and the plans and programs of <code>@luma.gl/graph/gpu-project</code>) take CRS strings and an optional
<code>destinationOrigin</code>. They do not take a <code>GPUSpatialFrame</code>. To project into a frame, pass the frame's <code>crs</code> as the target CRS. Pass the
frame's <code>origin</code> as <code>destinationOrigin</code> only when that origin is a point in the target CRS that
the coordinates are stored relative to: <code>anchor: 'relative-origin'</code>, or
<code>precision: 'float32-origin-relative'</code> with an anchor other than <code>'projection-center'</code>. A frame
anchored at <code>'projection-center'</code> (a local frame) holds its origin as longitude and latitude, the
centre of the projection, and its coordinates are already relative to it: pass no
<code>destinationOrigin</code> for such a frame. Without a <code>destinationOrigin</code> the compiled plan keeps the
origin the projection compiler chooses.</p>
<h2 id="rendering-frames">Rendering frames</h2>
<p>Configure the renderer's coordinate frame, projection and origin explicitly from the stored data
coordinates. This package exports frame descriptors and projection helpers; it does not export a
deck.gl layer adapter. The former <code>@deck.gl-community/gpu-layers</code> import is not a published
dependency of this workspace. The <a href="https://github.com/visgl/luma-contributors/blob/master/examples/deck/kepler-compute/README.md">Kepler example</a>
describes its local rendering integration and pinned deck.gl version.</p>
<p>A local AEQD or LAEA projection is distinct from an east/north/up tangent offset approximation.
Reusing the same longitude/latitude origin alone does not make their coordinates interchangeable.
Project the data into the renderer's frame, or choose a rendering projection that preserves the
analysis frame; keep the declared origin and units consistent in both paths.</p>`};export{e as default};