import{r as e}from"./kernel-C3tfQOn4.js";import{E as t,f as n,v as r,y as i}from"./gpu-contributor-utils-BIwLQON5.js";import{c as a,o}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as s,g as c,v as l}from"./spatial-context-DwZEd4EZ.js";import{c as u,s as d}from"./wgsl-common-DfyXiYB0.js";import{t as f}from"./geodesic-wgsl-I0CYDKcz.js";import{a as p}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as m}from"./vessel-tracks-DYc3-NZu.js";import{t as h}from"./define-run-B-brkCe4.js";import{d as g,s as _}from"./pointer-overlay-Qh8qflcA.js";import{m as v,p as y,r as b,t as x}from"./lines-kx5qHdti.js";import{Mo as S}from"./entry-preview-D70TdSbK.js";var C=`GPUShapeGenerator`,w=Object.freeze({...s,measure:`distance`});function T(e,t=new Float32Array(2)){if(t.length<2)throw Error(`Shape generator target must hold 2 elements`);let{segmentCount:n,radiusScale:r=1}=e;if(!Number.isFinite(Math.fround(n))||!Number.isFinite(Math.fround(r)))throw Error(`Shape generator settings must be finite`);if(n<1||r<0)throw Error(`Shape generator segmentCount must be at least 1 and radiusScale >= 0`);return t.set([Math.floor(n),r]),t}function E(e){return e===`sector`?1:3}function D(e,t){return e===`sector`?t+3:t+1}var O=class{id;props;spatialContext;coordinateSystem;sphereRadius;featureCount;maximumSegments;constructor(e){let r=e.id??`shape-generator`;if(this.id=r,this.props=e,e.shape!==`circle`&&e.shape!==`sector`&&e.shape!==`ellipse`)throw Error(`${r} unknown shape ${String(e.shape)}`);if(this.spatialContext=l(r,e.spatialContext,w),this.coordinateSystem=this.spatialContext.coordinateSpace===`longitude-latitude`?`geodesic`:`planar`,this.sphereRadius=this.coordinateSystem===`geodesic`?c(this.spatialContext):1,e.ellipseSpacing!==void 0&&(e.shape!==`ellipse`||e.ellipseSpacing!==`parameter`&&e.ellipseSpacing!==`arc-length`))throw Error(`${r} ellipseSpacing must be 'parameter' or 'arc-length' for ellipses`);this.maximumSegments=e.maximumSegments??64;let a=E(e.shape);if(n(r,`maximumSegments`,this.maximumSegments,a,65536),t(e.centers,[`float32x2`],`${r} centers`),this.featureCount=e.centers.length,this.featureCount<1)throw Error(`${r} centers must hold at least one row`);if(t(e.radii,[e.shape===`ellipse`?`float32x2`:`float32`],`${r} radii`),e.radii.length!==this.featureCount)throw Error(`${r} radii must have one row per center`);if(e.shape===`sector`){if(!e.bearings)throw Error(`${r} sector shapes need bearings`);if(t(e.bearings,[`float32x2`],`${r} bearings`),e.bearings.length!==this.featureCount)throw Error(`${r} bearings must have one row per center`)}else if(e.bearings)throw Error(`${r} ${e.shape} shapes have no bearings`);if(e.rotations){if(e.shape!==`ellipse`)throw Error(`${r} only ellipse shapes have rotations`);if(t(e.rotations,[`float32`],`${r} rotations`),e.rotations.length!==this.featureCount)throw Error(`${r} rotations must have one row per center`)}if(t(e.parameters,[`float32`],`${r} parameters`),e.parameters.length<2)throw Error(`${r} parameters must hold 2 float32 values`);let{positions:o,offsets:s,vertexCount:u}=e.output;t(o,[`float32x2`],`${r} output.positions`);let d=this.featureCount*D(e.shape,this.maximumSegments);if(o.length<d)throw Error(`${r} output.positions must hold at least ${d} rows`);if(t(s,[`uint32`],`${r} output.offsets`),s.length!==this.featureCount+1)throw Error(`${r} output.offsets must hold featureCount + 1 rows`);if(u&&(t(u,[`uint32`],`${r} output.vertexCount`),u.length<1))throw Error(`${r} output.vertexCount must hold one uint32`);i(r,[o,s,u],[e.centers,e.radii,e.bearings,e.rotations,e.parameters])}getCommandNodes(t){let{props:n,id:i,featureCount:s,maximumSegments:c}=this,{positions:l,offsets:m,vertexCount:h}=n.output;p(t,i,{centers:n.centers,radii:n.radii,bearings:n.bearings,rotations:n.rotations,parameters:n.parameters,positions:l,offsets:m,vertexCount:h}),n.ellipseSpacing===`arc-length`&&r(t.device,i,`ellipse arc-length scratch`,s*257*4,`reduce centers rows or use parameter spacing`);let{shape:g}=n,_=this.coordinateSystem===`geodesic`,v=n.ellipseSpacing===`arc-length`?e(t,`${i}-arc-lengths`,`float32`,s*257):void 0,y=[],b=`
const QUARTER_TURN: f32 = 1.5707963267948966;
const ARC_INTERVALS: u32 = 256u;

fn getEllipseRadii(feature: u32) -> vec2<f32> {
  var axes = vec2<f32>(
    radii[radiiOffset + 2u * feature], radii[radiiOffset + 2u * feature + 1u]) *
    parameters[parametersOffset + 1u];
  if ((bitcast<u32>(axes.x) & 0x7f800000u) == 0x7f800000u || axes.x < 0.0) {
    axes.x = 0.0;
  }
  if ((bitcast<u32>(axes.y) & 0x7f800000u) == 0x7f800000u || axes.y < 0.0) {
    axes.y = 0.0;
  }
  return axes;
}`;v&&y.push(o(t,{id:`${i}-arc-lengths`,operation:C,variant:`ellipse-arc-lengths`,bindings:[{name:`radii`,view:n.radii,type:`f32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`arcLengths`,view:v,type:`f32`,access:`read_write`}],invocationCount:s,declarations:b,body:`let axes = getEllipseRadii(index);
  // Normalize before squaring to keep the table independent of world-coordinate scale.
  let maximumRadius = max(axes.x, axes.y);
  var normalizedAxes = vec2<f32>(0.0);
  if (maximumRadius > 0.0) { normalizedAxes = axes / maximumRadius; }
  let base = arcLengthsOffset + index * (ARC_INTERVALS + 1u);
  arcLengths[base] = 0.0;
  var total = 0.0;
  for (var interval = 1u; interval <= ARC_INTERVALS; interval++) {
    // The exact chord uses the midpoint derivative times 2 sin(delta / 2), avoiding
    // cancellation from subtracting nearby ellipse coordinates. Quarter symmetry lets
    // every vertex reuse one table. The polygonal length converges quadratically.
    let halfStep = QUARTER_TURN / (2.0 * f32(ARC_INTERVALS));
    let midpoint = (2.0 * f32(interval) - 1.0) * halfStep;
    total += 2.0 * sin(halfStep) * length(normalizedAxes * vec2<f32>(sin(midpoint), cos(midpoint)));
    arcLengths[base + interval] = total;
  }`}));let x=D(g,c),S=`const FEATURE_COUNT: u32 = ${s}u;
const MAXIMUM_SEGMENTS: u32 = ${c}u;
const MINIMUM_SEGMENTS: u32 = ${E(g)}u;
const MAXIMUM_VERTICES: u32 = ${x}u;
const IS_SECTOR: bool = ${g===`sector`};

fn getSegmentCount(rawSegments: f32) -> u32 {
  return u32(clamp(rawSegments, f32(MINIMUM_SEGMENTS), f32(MAXIMUM_SEGMENTS)));
}

fn getVertexCount(segments: u32) -> u32 {
  return select(segments + 1u, segments + 3u, IS_SECTOR);
}`,w=[{name:`centers`,view:n.centers,type:`f32`,access:`read`},{name:`radii`,view:n.radii,type:`f32`,access:`read`},...n.bearings?[{name:`bearings`,view:n.bearings,type:`f32`,access:`read`}]:[],...n.rotations?[{name:`rotations`,view:n.rotations,type:`f32`,access:`read`}]:[],{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},...v?[{name:`arcLengths`,view:v,type:`f32`,access:`read`}]:[],{name:`positionsOut`,view:l,type:`f32`,access:`read_write`}];return y.push(o(t,{id:`${i}-vertices`,operation:C,variant:`${g}-${_?`geodesic`:`planar`}${n.rotations?`-rotated`:``}${v?`-arc-length`:``}`,bindings:w,invocationCount:s*x,declarations:`${S}
const EARTH_RADIUS: f32 = ${a(this.sphereRadius)};
const DEGREES: f32 = 0.017453292519943295;
const TWO_PI: f32 = 6.283185307179586;
const NOT_A_NUMBER: u32 = 0x7fc00000u;
${g===`ellipse`?b:``}
${v?`
fn getEllipsePhase(feature: u32, step: u32, segments: u32, axes: vec2<f32>) -> f32 {
  let fraction = f32(step) / f32(segments);
  // Equal axes already have equal arc intervals. This also handles collapsed points.
  if (axes.x == axes.y) { return TWO_PI * fraction; }
  // Integer quadrant arithmetic keeps cardinal vertices and the closure exact.
  let quadrant = (4u * step) / segments;
  let quarterFraction = f32((4u * step) % segments) / f32(segments);
  let reverse = (quadrant & 1u) != 0u;
  let targetFraction = select(quarterFraction, 1.0 - quarterFraction, reverse);
  let base = arcLengthsOffset + feature * (ARC_INTERVALS + 1u);
  let targetLength = targetFraction * arcLengths[base + ARC_INTERVALS];
  var lower = 0u;
  var upper = ARC_INTERVALS;
  for (var iteration = 0u; iteration < 8u; iteration++) {
    let middle = (lower + upper) / 2u;
    if (arcLengths[base + middle] < targetLength) { lower = middle; } else { upper = middle; }
  }
  let start = arcLengths[base + lower];
  let span = arcLengths[base + upper] - start;
  var weight = 0.0;
  if (span > 0.0) { weight = clamp((targetLength - start) / span, 0.0, 1.0); }
  let phase = QUARTER_TURN * (f32(lower) + weight) / f32(ARC_INTERVALS);
  return f32(quadrant) * QUARTER_TURN + select(phase, QUARTER_TURN - phase, reverse);
}`:``}

${u}
${d}
${_?f:``}

// Point at the given distance (world units, or meters for geodesic) and bearing (radians
// clockwise from north) from the center.
fn getDestination(center: vec2<f32>, distance: f32, bearing: f32) -> vec2<f32> {
${_?`  // The shared half-angle destination (geodesicDestination): sin(lat2) - sin(lat) is formed from
  // half-angle terms and the latitude change is recovered as 2 asin(delta / (2 cos(mid latitude))).
  // The textbook asin(sin(lat) cos(d) + cos(lat) sin(d) cos(b)) in f32 loses a 50 m radius to
  // rounding (0.8 m at latitude 45, 10 m at 88.6).
  return geodesicDestination(center, bearing * GEODESIC_RADIANS_TO_DEGREES, distance / EARTH_RADIUS);`:`  return center + distance * vec2<f32>(sin(bearing), cos(bearing));`}
}`,body:`let feature = index / MAXIMUM_VERTICES;
  let local = index % MAXIMUM_VERTICES;
  let segments = getSegmentCount(parameters[parametersOffset]);
  let vertices = getVertexCount(segments);
  if (local >= vertices) {
    return;
  }
  let radiusScale = parameters[parametersOffset + 1u];
  let center = vec2<f32>(
    centers[centersOffset + 2u * feature], centers[centersOffset + 2u * feature + 1u]);
  var position = center;
${g===`ellipse`?`  let radii2 = getEllipseRadii(feature);
  // Counter-clockwise from the x semi-axis, which points along bearing 90 degrees + rotation.
  let phase = ${v?`getEllipsePhase(feature, local % segments, segments, radii2)`:`TWO_PI * f32(local % segments) / f32(segments)`};
  let east = radii2.x * cos(phase);
  let north = radii2.y * sin(phase);
  ${n.rotations?`let tilt = rotations[rotationsOffset + feature] * DEGREES;`:`let tilt: f32 = 0.0;`}
  // Rotate the (east, north) offset clockwise by tilt.
  let rotatedEast = east * cos(tilt) + north * sin(tilt);
  let rotatedNorth = north * cos(tilt) - east * sin(tilt);
  ${v&&!_?`position = center + vec2<f32>(rotatedEast, rotatedNorth);`:`let distance = length(vec2<f32>(rotatedEast, rotatedNorth));
  // The north offset is -0.0 for a collapsed axis in the lower half turn, and both offsets are
  // zero for a zero radius: the guarded form keeps the east side and returns bearing 0.
  var bearing = guardedAtan2(rotatedEast, rotatedNorth);
  ${v?`// Avoid backend atan2 signed-zero ambiguity for an east/west degenerate axis.
  if (rotatedNorth == 0.0) {
    bearing = select(-QUARTER_TURN, QUARTER_TURN, rotatedEast >= 0.0);
  }`:``}
  position = getDestination(center, distance, bearing);`}`:`  var radius = radii[radiiOffset + feature] * radiusScale;
  if (!isFiniteBits(radius) || radius < 0.0) { radius = 0.0; }
${g===`sector`?`  var bearing1 = bearings[bearingsOffset + 2u * feature];
  var bearing2 = bearings[bearingsOffset + 2u * feature + 1u];
  bearing1 = bearing1 - 360.0 * floor(bearing1 / 360.0);
  bearing2 = bearing2 - 360.0 * floor(bearing2 / 360.0);
  let sweep = select(bearing2 - bearing1 + 360.0, bearing2 - bearing1, bearing1 < bearing2);
  // Slot 0 and the last slot are the center; slots 1..=segments+1 are the arc.
  if (local != 0u && local != vertices - 1u) {
    let step = f32(local - 1u) / f32(segments);
    position = getDestination(center, radius, (bearing1 + sweep * step) * DEGREES);
  }`:`  // Counter-clockwise ring, bearings 0, -360 / n, ...; the last vertex repeats the first.
  let ringStep = f32(local % segments) / f32(segments);
  position = getDestination(center, radius, -ringStep * TWO_PI);`}`}
  if (!isFiniteBits(center.x) || !isFiniteBits(center.y)) {
    position = vec2<f32>(bitcast<f32>(NOT_A_NUMBER | (feature & 0u)));
  }
  let base = positionsOutOffset + 2u * (feature * vertices + local);
  positionsOut[base] = position.x;
  positionsOut[base + 1u] = position.y;`}),o(t,{id:`${i}-offsets`,operation:C,variant:`offsets`,bindings:[{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`offsetsOut`,view:m,type:`u32`,access:`read_write`},...h?[{name:`vertexCountOut`,view:h,type:`u32`,access:`read_write`}]:[]],invocationCount:s+1,declarations:S,body:`let vertices = getVertexCount(getSegmentCount(parameters[parametersOffset]));
  offsetsOut[offsetsOutOffset + index] = index * vertices;
  ${h?`if (index == FEATURE_COUNT) { vertexCountOut[vertexCountOutOffset] = index * vertices; }`:``}`})),y}},k=[`circle`,`sector`,`ellipse`],A=8,j=48,M=.1,N=.4,P=h({parameters:S,caption:`Modeled search areas around the last San Francisco Bay AIS fixes. Sectors follow the last observed course.`,createInputs:({world:e})=>{let t=m(e,A,2,72),n=new Float32Array(16),r=new Float32Array(A).fill(M),i=new Float32Array(16),a=new Float32Array(16),o=new Float32Array(A);for(let e=0;e<A;e++){let r=t.trackOffsets[e],s=t.trackOffsets[e+1]-1,c=t.points[s*2],l=t.points[s*2+1],u=s-1;for(;u>r&&t.points[u*2]===c&&t.points[u*2+1]===l;)u--;let d=Math.atan2(c-t.points[u*2],l-t.points[u*2+1])*180/Math.PI;n.set([c,l],e*2),i.set([.14,.055],e*2),a.set([d-45,d+45],e*2),o[e]=d-90}return{centers:n,radii:r,ellipseRadii:i,bearings:a,rotations:o,tracks:t.segments,world:e}},build(e,t,n){let r=k[n.shape],i=D(r,j),a=e.upload(`centers`,`float32x2`,t.centers),o=e.upload(`radii`,`float32`,t.radii),s=e.upload(`ellipse-radii`,`float32x2`,t.ellipseRadii),c=e.upload(`bearings`,`float32x2`,t.bearings),l=e.upload(`rotations`,`float32`,t.rotations),u=e.allocate(`ring-positions`,`float32x2`,A*i),d=e.allocate(`ring-offsets`,`uint32`,9),f=e.allocate(`shape-parameters`,`float32`,2),p={centers:a.view,maximumSegments:j,parameters:f.view,output:{positions:u.view,offsets:d.view}};e.graph.add(new O(r===`ellipse`?{shape:r,radii:s.view,rotations:l.view,...p}:r===`sector`?{shape:r,radii:o.view,bearings:c.view,...p}:{shape:r,radii:o.view,...p}));let m=b(e,`rings`,u,d,{closed:r!==`sector`});return{outputs:{centers:a,tracks:e.upload(`observed-tracks`,`float32x4`,t.tracks),...m},update:e=>f.write(T({segmentCount:e.segmentCount,radiusScale:e.radiusScale}))}},rebuildOn:[`shape`],interaction:{kind:`reach`,hint:`move off a vessel`,rest:({inputs:e,toUnit:t})=>{let n=_(S,`radiusScale`);return t(v([e.centers[0],e.centers[1]],(n.default-n.min)/(n.max-n.min)*N,(t,n)=>y(e.centers,t,n)?.distance??0,e.world))},resolve(e,{inputs:t,values:n}){let r=y(t.centers,e.x,e.y);if(!r)return{};let i=_(S,`radiusScale`),a=g(i,i.min+(i.max-i.min)*Math.min(1,r.distance/N));return{values:{radiusScale:a},glyphs:[{shape:`line`,from:r.point,to:[e.x,e.y],dashed:!0},{shape:`dot`,at:r.point}],readout:`radiusScale ${a.toFixed(2)} · ${n.segmentCount} segments`}}},draw:({outputs:e,values:t})=>[{layer:`lines`,segments:e.tracks,context:!0,radius:.9,alpha:.6},{layer:`segments`,nodes:e.points,edges:e.edges,values:e.lineIds,radius:2.4},...t.segmentCount<=16?[{layer:`dots`,positions:e.points,color:x,radius:2.6}]:[],{layer:`dots`,context:!0,positions:e.centers,radius:3}]});export{P as default};