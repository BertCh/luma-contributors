import{r as e}from"./gpu-command-graph-BdzdmV6u.js";import{n as t,t as n}from"./gpu-dispatch-utils-CbRzFAgT.js";import{c as r,d as i,o as a,t as o}from"./kernel-C3tfQOn4.js";import{n as s}from"./earth-DODowzP2.js";import{o as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as l}from"./wgsl-common-DfyXiYB0.js";import{E as u,T as d,c as f,f as p,g as m,p as h,y as g}from"./gpu-contributor-utils-BIwLQON5.js";import{a as _,i as v,n as y,o as b,r as x,t as S}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as C}from"./define-run-B-brkCe4.js";import{d as w,o as T,s as E}from"./pointer-overlay-Qh8qflcA.js";import{t as D}from"./lightning-C0gvbE9a.js";import{jc as O}from"./entry-preview-D70TdSbK.js";var k=256,A=16777216,j=s;function M(e,t=new Float32Array(2)){return m(`GPUPointBuffer`,`radius`,e),d(`GPUPointBuffer`,e>0,`radius must be positive`),t[0]=e,t[1]=0,t}var N=class{props;id;estimatedCommandNodeCount;chunkPlan;constructor(e){this.props=e,this.id=e.id??`point-buffer`,this.chunkPlan=x(this.id,{points:e.points});for(let[t,n]of[[`points`,e.points],[`offsets`,e.offsets]])for(let e of f(n))i(e,[`float32x2`],`${this.id} ${t}`);p(this.id,`verticesPerRing`,e.verticesPerRing,3);let t=f(e.offsets);d(this.id,t.length===this.chunkPlan.chunks.length,`offsets must match points chunk count with whole rings per chunk`),p(this.id,`offsets.length`,e.points.length*e.verticesPerRing,0,S),d(this.id,e.offsets.length===e.points.length*e.verticesPerRing,`offsets must have verticesPerRing rows per point`);for(let n of this.chunkPlan.chunks)d(this.id,t[n.chunkIndex].length===n.length*e.verticesPerRing,`offsets chunk ${n.chunkIndex} must have verticesPerRing rows per points chunk row`);v(this.id,{offsets:e.offsets}),g(this.id,[e.offsets],[e.points,e.parameters]),this.estimatedCommandNodeCount=1+this.chunkPlan.nonemptyChunkCount,h(this.id,this.estimatedCommandNodeCount),u(e.parameters,[`float32`],`${this.id} parameters`),d(this.id,e.parameters.length>=2,`parameters must have at least 2 rows`)}getCommandNodes(t){let{points:i,parameters:s,offsets:u,verticesPerRing:d,coordinateSystem:p}=this.props;if(b(t,this.id,this.chunkPlan),_(t,this.id,{parameters:s,offsets:u}),this.chunkPlan.nonemptyChunkCount===0)return[];let m={id:`${this.id}-offsets`,operation:`GPUPointBuffer`,variant:p,bindings:[{name:`pointValues`,view:f(i)[0],type:`f32`,access:`read`},{name:`settings`,view:s,type:`f32`,access:`read`},{name:`offsetValues`,view:f(u)[0],type:`f32`,access:`read_write`}],invocationCount:u.length,declarations:`const VERTICES_PER_RING: u32 = ${d}u;
const EARTH_RADIUS_METERS: f32 = ${j};
const TAU: f32 = 6.283185307179586;
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
const RADIANS_TO_DEGREES: f32 = 57.29577951308232;
${l}`,body:`let pointIndex = index / VERTICES_PER_RING;
  let vertexIndex = index % VERTICES_PER_RING;
  let bearing = TAU * f32(vertexIndex) / f32(VERTICES_PER_RING);
  let radius = settings[settingsOffset];
  let centerLongitude = pointValues[pointValuesOffset + 2u * pointIndex];
  let centerLatitude = pointValues[pointValuesOffset + 2u * pointIndex + 1u];
  ${p===`geographic`?`let clampedLatitude = clamp(centerLatitude, -90.0, 90.0);
  let phi = clampedLatitude * DEGREES_TO_RADIANS;
  let delta = radius / EARTH_RADIUS_METERS;
  let sinDelta = sin(delta);
  let cosDelta = cos(delta);
  let halfSine = sin(delta * 0.5);
  let versine = 2.0 * halfSine * halfSine;
  let nearPole = abs(clampedLatitude) > 45.0;
  let coLatitude = (90.0 - abs(clampedLatitude)) * DEGREES_TO_RADIANS;
  let sinPhi = select(sin(phi), sign(clampedLatitude) * cos(coLatitude), nearPole);
  let cosPhi = select(cos(phi), sin(coLatitude), nearPole);
  let cosTheta = cos(bearing);
  let sinTheta = sin(bearing);
  var latitudeOffset: f32;
  if (cosPhi < 0.05 || delta > 0.5 * cosPhi) {
    // Absolute destination: sin(latitude') and cos(latitude') times the longitude difference.
    let sineTarget = sinPhi * cosDelta + cosPhi * sinDelta * cosTheta;
    let northTerm = cosPhi * cosDelta - sinPhi * sinDelta * cosTheta;
    let targetLatitude = atan2(sineTarget, length(vec2f(northTerm, sinTheta * sinDelta)));
    latitudeOffset = targetLatitude - phi;
  } else {
    // sin(latitude') - sin(phi), without the cancellation of cos(delta) - 1.
    let sineDifference = -sinPhi * versine + cosPhi * sinDelta * cosTheta;
    latitudeOffset = sineDifference / cosPhi;
    for (var iteration = 0u; iteration < 2u; iteration++) {
      let cosMean = max(cos(phi + 0.5 * latitudeOffset), 1e-6);
      latitudeOffset = 2.0 * asin(clamp(sineDifference / (2.0 * cosMean), -1.0, 1.0));
    }
  }
  // The common cosPhi factor of the standard atan2 arguments is divided out, so the longitude
  // stays defined (and finite) at the poles, where cosPhi is exactly zero.
  let longitudeOffset = guardedAtan2(
    sinTheta * sinDelta,
    cosPhi * (1.0 - versine) - sinPhi * sinDelta * cosTheta
  );
  // Clamp only when the ring reaches past a pole, so small offsets keep their precision.
  let rawLatitudeOffset = latitudeOffset * RADIANS_TO_DEGREES;
  let latitudeOffsetDegrees = select(
    rawLatitudeOffset,
    clamp(clampedLatitude + rawLatitudeOffset, -90.0, 90.0) - clampedLatitude,
    abs(clampedLatitude + rawLatitudeOffset) > 90.0
  );
  offsetValues[offsetValuesOffset + 2u * index] = longitudeOffset * RADIANS_TO_DEGREES;
  offsetValues[offsetValuesOffset + 2u * index + 1u] = latitudeOffsetDegrees;`:`offsetValues[offsetValuesOffset + 2u * index] = radius * sin(bearing);
  offsetValues[offsetValuesOffset + 2u * index + 1u] = radius * cos(bearing);`}`};if(`buffer`in i&&`buffer`in u)return[c(t,m)];let h=f(u),g=h.map(e=>n(this.id,e.length,k,t.device.limits.maxComputeWorkgroupsPerDimension)),v=`
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
@group(0) @binding(1) var<storage, read> pointValues: array<f32>;
@group(0) @binding(2) var<storage, read> settings: array<f32>;
@group(0) @binding(3) var<storage, read_write> offsetValues: array<f32>;
const settingsOffset: u32 = ${r(s)}u;
${m.declarations}
@compute @workgroup_size(${k})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  _ = &pointValues;
  _ = &settings;
  _ = &offsetValues;
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${A}u) { return; }
  let index = workgroupIndex * ${k}u + localInvocationIndex;
  let vertexCount = chunkParameters[1] * VERTICES_PER_RING;
  if (index >= vertexCount) { return; }
  let pointValuesOffset = arrayLength(&pointValues) - 2u * chunkParameters[1];
  let offsetValuesOffset = arrayLength(&offsetValues) - 2u * vertexCount;
  ${m.body}
}`,{slots:x,nodes:S}=y(t,{id:`${this.id}-chunks`,plan:this.chunkPlan});return[...S,...this.chunkPlan.chunks.filter(e=>e.length>0).map(t=>{let n=x[t.chunkIndex],r=t.views.points,i=h[t.chunkIndex];return e({id:`${this.id}-offsets-${t.chunkIndex}`,workload:{operation:`GPUPointBuffer`,variant:p,commandCount:1,maximumInvocationCount:i.length,maximumWorkgroupCount:g[t.chunkIndex].x*g[t.chunkIndex].y*g[t.chunkIndex].z},resources:[{buffer:n,usage:`storage-read`},{buffer:r,usage:`storage-read`},{buffer:s,usage:`storage-read`},{buffer:i,usage:`storage-read-write`}],compile:({device:e})=>{let c=new o(e,{id:`point-buffer-chunk`,source:v,shaderLayout:{bindings:[{name:`chunkParameters`,group:0,location:0,type:`read-only-storage`},{name:`pointValues`,group:0,location:1,type:`read-only-storage`},{name:`settings`,group:0,location:2,type:`read-only-storage`},{name:`offsetValues`,group:0,location:3,type:`storage`}]}});return{encode:({computePass:e,getBuffer:o})=>c.dispatch(e,{bindings:{chunkParameters:a(n,o),pointValues:a(r,o),settings:a(s,o),offsetValues:a(i,o)},...g[t.chunkIndex]}),destroy:()=>c.destroy()}}})})]}},P=64,F=class{id;centers;offsets;segments;ringSize;constructor(e,t,n,r,i){this.id=e,this.centers=t,this.offsets=n,this.segments=r,this.ringSize=i}getCommandNodes(i){let{id:s,centers:c,offsets:l,segments:u,ringSize:d}=this,f=n(s,u.length,P,i.device.limits.maxComputeWorkgroupsPerDimension),p=`
const centersOffset: u32 = ${r(c)}u;
const offsetsOffset: u32 = ${r(l)}u;
const segmentsOffset: u32 = ${r(u)}u;
@group(0) @binding(0) var<storage, read> centers: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<f32>;
@group(0) @binding(2) var<storage, read_write> segments: array<f32>;
@compute @workgroup_size(${P})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${t(f,P)}
  if (index >= ${u.length}u) { return; }
  let ring = index / ${d}u;
  let vertex = index % ${d}u;
  let next = (vertex + 1u) % ${d}u;
  let center = vec2f(centers[centersOffset + ring * 2u], centers[centersOffset + ring * 2u + 1u]);
  let first = ring * ${d}u + vertex;
  let second = ring * ${d}u + next;
  let start = center + vec2f(offsets[offsetsOffset + first * 2u], offsets[offsetsOffset + first * 2u + 1u]);
  let end = center + vec2f(offsets[offsetsOffset + second * 2u], offsets[offsetsOffset + second * 2u + 1u]);
  segments[segmentsOffset + index * 4u] = start.x;
  segments[segmentsOffset + index * 4u + 1u] = start.y;
  segments[segmentsOffset + index * 4u + 2u] = end.x;
  segments[segmentsOffset + index * 4u + 3u] = end.y;
}`;return[e({id:s,resources:[{buffer:c,usage:`storage-read`},{buffer:l,usage:`storage-read`},{buffer:u,usage:`storage-read-write`}],workload:{operation:`RingSegments`,variant:`ring`,commandCount:1,maximumWorkgroupCount:f.x*f.y*f.z,maximumInvocationCount:f.x*f.y*f.z*P,readByteLength:u.length*16,writeByteLength:u.length*16},compile:({device:e})=>{let t=new o(e,{id:s,source:p,shaderLayout:{bindings:[`centers`,`offsets`,`segments`].map((e,t)=>({name:e,type:`storage`,group:0,location:t}))}});return{encode:({computePass:e,getBuffer:n})=>{let r={centers:a(c,n),offsets:a(l,n),segments:a(u,n)};t.dispatch(e,{bindings:r,...f})},destroy:()=>t.destroy()}}})]}};function I(e,t,n,r,i){let a=e.allocate(t,`float32x4`,r.length);return e.graph.add(new F(`${t}-ring`,n.view,r.view,a.view,i)),a}var L=24;function R(e,t,n,r){let i=0,a=1/0;for(let o=0;o<t;o++){let t=Math.hypot(e[o*2]-n,e[o*2+1]-r);t<a&&(i=o,a=t)}return{index:i,distance:a}}var z=C({parameters:O,caption:`Fixed-radius buffers around measured Midwest lightning flashes.`,createInputs:({world:e})=>({centers:D(e,L).positions}),build(e,t,n){let r=n.pointCount,i=e.upload(`centers`,`float32x2`,t.centers.slice(0,r*2)),a=e.allocate(`parameters`,`float32`,2),o=e.allocate(`offsets`,`float32x2`,r*n.verticesPerRing);return e.graph.add(new N({points:i.view,parameters:a.view,offsets:o.view,verticesPerRing:n.verticesPerRing,coordinateSystem:`planar`})),{outputs:{centers:i,rings:I(e,`rings`,i,o,n.verticesPerRing)},update:e=>a.write(M(e.radius))}},rebuildOn:[`verticesPerRing`,`pointCount`],interaction:{kind:`reach`,hint:`pull a ring out`,rest:({inputs:e,values:t,toUnit:n})=>{let r=E(O,`radius`).default,i=[.5,.5],a=-1;for(let o=0;o<t.pointCount;o++){let s=e.centers[o*2],c=e.centers[o*2+1];for(let l of[0,Math.PI,Math.PI/2,-Math.PI/2]){let u=[s+Math.cos(l)*r,c+Math.sin(l)*r],d=R(e.centers,t.pointCount,u[0],u[1]),[f,p]=n(u);if(d.index===o&&f>.03&&f<.97&&p>.03&&p<.97){let e=Math.min(f,1-f,p,1-p);e>a&&(a=e,i=[f,p])}}}return i},resolve(e,{inputs:t,values:n}){let r=R(t.centers,n.pointCount,e.x,e.y),i=E(O,`radius`),a=w(i,T(r.distance,i.min,i.max)),o=[t.centers[r.index*2],t.centers[r.index*2+1]];return{values:{radius:a},glyphs:[{shape:`line`,from:o,to:[e.x,e.y],dashed:!0},{shape:`dot`,at:[e.x,e.y]}],readout:`radius ${a.toFixed(3)} · ${n.verticesPerRing} vertices per ring`}}},draw:({outputs:e})=>[{layer:`dots`,context:!0,positions:e.centers,color:[.9,.9,1,.9],radius:3},{layer:`lines`,segments:e.rings,color:[.3,.75,1,1],radius:1.8}]});export{z as default};