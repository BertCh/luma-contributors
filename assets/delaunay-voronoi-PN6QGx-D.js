import{a as e,n as t,r as n}from"./gpu-delaunay-tessellation-D0udu4Pw.js";import{t as r}from"./common-CshkX-d_.js";import{v as i}from"./spatial-context-DwZEd4EZ.js";import{r as a}from"./kernel-C3tfQOn4.js";import{t as o}from"./gpu-compaction-B6IzcF0P.js";import{E as s,T as c,f as l,v as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{t as f}from"./stable-lexicographic-index-sort-DYazZEY1.js";import{o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as m}from"./status-aOwAAXek.js";import{t as h}from"./exact-orientation-wgsl-B_jI-HEc.js";import{n as g,t as _}from"./triangle-circumcircle-wgsl-Dn21B-0X.js";import{a as v,i as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{s as b}from"./point-observations-CeqP-hIo.js";import{t as x}from"./pointer-driver-Bk4iFPMM.js";import{C as S,m as C}from"./index-DIfTuwM1.js";import{A as w,C as T,E as ee,S as E,T as D,b as te,h as ne,k as O,m as k,p as A,w as j,x as M}from"./pointer-overlay-Qh8qflcA.js";var N=357913941,P=1073741823;function F(e){return{segments:e.segments,siteIds:e.siteIds,count:e.status.count,requiredCount:e.status.requiredCount,overflow:e.status.overflow,invalidCount:e.status.invalidCount}}function I(e){if(!Number.isSafeInteger(e)||e<0)throw Error(`triangleCapacity must be a non-negative safe integer`);return 3*e}var re=class{id;props;spatialContext;constructor(t){this.id=t.id??`voronoi-diagram`,this.props=t,this.spatialContext=i(this.id,t.spatialContext);let{id:n}=this;if(s(t.positions,[`float32x2`],`${n} positions`),s(t.triangles,[`uint32x3`],`${n} triangles`),s(t.triangleCount,[`uint32`],`${n} triangleCount`),t.triangleCount.length<1)throw Error(`${n} triangleCount must contain one row`);if(s(t.clipBounds,[`float32x4`],`${n} clipBounds`),t.clipBounds.length!==1)throw Error(`${n} clipBounds must contain one row`);if(s(t.output.segments,[`float32x4`],`${n} output.segments`),s(t.output.siteIds,[`uint32x2`],`${n} output.siteIds`),t.output.segments.length!==t.output.siteIds.length)throw Error(`${n} output segment and site capacities must match`);m(`${n} output.status`,t.output.status),c(n,!!t.output.status.requiredCount,`output.status.requiredCount is required`),l(n,`triangles rows`,t.triangles.length,0,N),l(n,`output.segments rows`,t.output.segments.length,0,P);let r=F(t.output);e(n,r),y(n,{requiredCount:r.requiredCount,count:r.count,overflow:r.overflow}),y(n,{requiredCount:r.requiredCount,segments:r.segments}),y(n,{requiredCount:r.requiredCount,siteIds:r.siteIds}),d(n,Object.values(r),[t.positions,t.triangles,t.triangleCount,t.clipBounds])}getCommandNodes(e){let{id:t,props:n}=this,{output:i}=n;v(e,t,{positions:n.positions,triangles:n.triangles,triangleCount:n.triangleCount,clipBounds:n.clipBounds,...F(i)}),u(e.device,t,`candidate segment scratch`,Math.max(n.triangles.length*48,4),`reduce triangles rows`);let s=n.triangles.length,c=3*s,l=a(e,`${t}-centers`,`float32`,3*s),d=a(e,`${t}-candidate-ids`,`uint32`,c),m=a(e,`${t}-candidate-flags`,`uint32`,c),h=a(e,`${t}-edge-minimum-keys`,`uint32`,c),g=a(e,`${t}-edge-maximum-keys`,`uint32`,c),_=r(32-Math.clz32(n.positions.length),1,32),y=f(e,{id:`${t}-edge-sort`,operation:`GPUVoronoiDiagram`,indices:d,keys:[{view:h,keyBits:_},{view:g,keyBits:_}]}),b=a(e,`${t}-neighbors`,`uint32`,c),x=a(e,`${t}-candidate-segments`,`float32`,4*c),S=a(e,`${t}-compact-candidate-ids`,`uint32`,c),C=i.status.invalidCount??a(e,`${t}-invalid-count`,`uint32`,1),w=[p(e,{id:`${t}-initialize`,operation:`GPUVoronoiDiagram`,variant:`initialize`,bindings:[{name:`invalidCount`,view:C,type:`u32`,access:`read_write`}],invocationCount:1,body:`invalidCount[invalidCountOffset] = 0u;`}),p(e,{id:`${t}-circumcenters`,operation:`GPUVoronoiDiagram`,variant:`parallel-circumcenters`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`triangles`,view:n.triangles,type:`u32`,access:`read`},{name:`triangleCountIn`,view:n.triangleCount,type:`u32`,access:`read`},{name:`centers`,view:l,type:`f32`,access:`read_write`},{name:`invalidCount`,view:C,type:`atomic<u32>`,access:`read_write`}],invocationCount:s,declarations:L({pointCount:n.positions.length,triangleCapacity:s}),body:`let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let center = circumcenter(index);
  let base = centersOffset + 3u * index;
  centers[base] = center.x;
  centers[base + 1u] = center.y;
  centers[base + 2u] = select(0.0, center.z, index < triangleCount);
  if (index < triangleCount && center.z == 0.0) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }`}),p(e,{id:`${t}-edge-keys`,operation:`GPUVoronoiDiagram`,variant:`parallel-edge-keys`,bindings:[{name:`triangles`,view:n.triangles,type:`u32`,access:`read`},{name:`triangleCountIn`,view:n.triangleCount,type:`u32`,access:`read`},{name:`candidateIds`,view:d,type:`u32`,access:`read_write`},{name:`edgeMinimumKeys`,view:h,type:`u32`,access:`read_write`},{name:`edgeMaximumKeys`,view:g,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const POINT_COUNT: u32 = ${n.positions.length}u;
const TRIANGLE_CAPACITY: u32 = ${s}u;`,body:`let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let triangle = index / 3u;
  let edge = index % 3u;
  candidateIds[candidateIdsOffset + index] = index;
  var minimum = POINT_COUNT;
  var maximum = POINT_COUNT;
  if (triangle < triangleCount) {
    let a = triangles[trianglesOffset + 3u * triangle + edge];
    let b = triangles[trianglesOffset + 3u * triangle + (edge + 1u) % 3u];
    if (a < POINT_COUNT && b < POINT_COUNT) {
      minimum = min(a, b);
      maximum = max(a, b);
    }
  }
  edgeMinimumKeys[edgeMinimumKeysOffset + index] = minimum;
  edgeMaximumKeys[edgeMaximumKeysOffset + index] = maximum;`}),...y.nodes,p(e,{id:`${t}-classify-edges`,operation:`GPUVoronoiDiagram`,variant:`sorted-edge-classification`,bindings:[{name:`sortedCandidates`,view:y.sortedIndices,type:`u32`,access:`read`},{name:`sortedMinimumKeys`,view:y.sortedPrimaryKeys,type:`u32`,access:`read`},{name:`edgeMaximumKeys`,view:g,type:`u32`,access:`read`},{name:`centers`,view:l,type:`f32`,access:`read`},{name:`neighbors`,view:b,type:`u32`,access:`read_write`},{name:`candidateFlags`,view:m,type:`u32`,access:`read_write`}],invocationCount:c,declarations:`const POINT_COUNT: u32 = ${n.positions.length}u;
const TRIANGLE_CAPACITY: u32 = ${s}u;
const CANDIDATE_COUNT: u32 = ${c}u;
fn sortedTriangle(row: u32) -> u32 {
  return sortedCandidates[sortedCandidatesOffset + row] / 3u;
}
fn sortedMaximumKey(row: u32) -> u32 {
  return edgeMaximumKeys[edgeMaximumKeysOffset + sortedCandidates[sortedCandidatesOffset + row]];
}
fn sortedKeyEquals(row: u32, minimum: u32, maximum: u32) -> bool {
  return sortedMinimumKeys[sortedMinimumKeysOffset + row] == minimum && sortedMaximumKey(row) == maximum;
}
fn isValidCenter(triangle: u32) -> bool {
  return centers[centersOffset + 3u * triangle + 2u] != 0.0;
}`,body:`let candidate = sortedCandidates[sortedCandidatesOffset + index];
  let triangle = candidate / 3u;
  let minimum = sortedMinimumKeys[sortedMinimumKeysOffset + index];
  var neighbor = TRIANGLE_CAPACITY;
  var accepted = false;
  if (minimum < POINT_COUNT) {
    let maximum = sortedMaximumKey(index);
    // Lower bound of this key's run in [0, index].
    var low = 0u;
    var high = index;
    while (low < high) {
      let middle = (low + high) >> 1u;
      let middleMinimum = sortedMinimumKeys[sortedMinimumKeysOffset + middle];
      if (middleMinimum < minimum || (middleMinimum == minimum && sortedMaximumKey(middle) < maximum)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    // Skip this triangle's own rows (at most three, contiguous) at the head of the run.
    var row = low;
    while (row < CANDIDATE_COUNT && sortedTriangle(row) == triangle && sortedKeyEquals(row, minimum, maximum)) {
      row++;
    }
    if (row < CANDIDATE_COUNT && sortedKeyEquals(row, minimum, maximum)) {
      neighbor = sortedTriangle(row);
    }
    accepted = isValidCenter(triangle) &&
      (neighbor == TRIANGLE_CAPACITY || (neighbor > triangle && isValidCenter(neighbor)));
  }
  neighbors[neighborsOffset + candidate] = neighbor;
  candidateFlags[candidateFlagsOffset + candidate] = select(0u, 1u, accepted);`})];return w.push(p(e,{id:`${t}-clip-segments`,operation:`GPUVoronoiDiagram`,variant:`clip-segments`,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`triangles`,view:n.triangles,type:`u32`,access:`read`},{name:`clipBounds`,view:n.clipBounds,type:`f32`,access:`read`},{name:`centers`,view:l,type:`f32`,access:`read`},{name:`neighbors`,view:b,type:`u32`,access:`read`},{name:`candidateFlags`,view:m,type:`u32`,access:`read_write`},{name:`candidateSegments`,view:x,type:`f32`,access:`read_write`}],invocationCount:c,declarations:`${R({pointCount:n.positions.length,triangleCapacity:s})}
fn centerAt(triangle: u32) -> vec2<f32> {
  return vec2<f32>(centers[centersOffset + 3u * triangle], centers[centersOffset + 3u * triangle + 1u]);
}
// Liang-Barsky clip of a segment (0 <= t <= 1) or ray (t >= 0); returns (t0, t1, 1) or zeros.
fn clipParametric(origin: vec2<f32>, direction: vec2<f32>, isRay: bool, bounds: vec4<f32>) -> vec3<f32> {
  var enter = 0.0;
  var leave = 1.0;
  var hasExit = !isRay;
  for (var axis = 0u; axis < 2u; axis++) {
    let low = bounds[axis];
    let high = bounds[axis + 2u];
    if (direction[axis] == 0.0) {
      if (origin[axis] < low || origin[axis] > high) { return vec3<f32>(0.0); }
    } else {
      let first = (low - origin[axis]) / direction[axis];
      let second = (high - origin[axis]) / direction[axis];
      enter = max(enter, min(first, second));
      let exit = max(first, second);
      if (hasExit) { leave = min(leave, exit); } else { leave = exit; hasExit = true; }
    }
  }
  if (!hasExit || !(enter <= leave)) { return vec3<f32>(0.0); }
  return vec3<f32>(enter, leave, 1.0);
}`,body:`let triangle = index / 3u;
let edge = index % 3u;
let slot = candidateSegmentsOffset + 4u * index;
candidateSegments[slot] = 0.0; candidateSegments[slot + 1u] = 0.0; candidateSegments[slot + 2u] = 0.0; candidateSegments[slot + 3u] = 0.0;
if (candidateFlags[candidateFlagsOffset + index] == 0u) { return; }
let bounds = vec4<f32>(clipBounds[clipBoundsOffset], clipBounds[clipBoundsOffset + 1u], clipBounds[clipBoundsOffset + 2u], clipBounds[clipBoundsOffset + 3u]);
let neighbor = neighbors[neighborsOffset + index];
let center = centerAt(triangle);
var origin = center;
var direction = vec2<f32>(0.0);
let isRay = neighbor >= TRIANGLE_CAPACITY;
if (neighbor < TRIANGLE_CAPACITY) {
  direction = centerAt(neighbor) - center;
} else {
  let a = site(vertex(triangle, edge));
  let b = site(vertex(triangle, (edge + 1u) % 3u));
  let edgeDirection = getScaledTriangleEdge(a, b).direction;
  let edgeExtent = max(abs(edgeDirection.x), abs(edgeDirection.y));
  // The third center word carries the certified triangle orientation, including clockwise input.
  direction = centers[centersOffset + 3u * triangle + 2u] * vec2f(edgeDirection.y, -edgeDirection.x) / edgeExtent;
}
let clipped = clipParametric(origin, direction, isRay, bounds);
if (clipped.z == 0.0) { candidateFlags[candidateFlagsOffset + index] = 0u; return; }
let start = origin + clipped.x * direction;
let end = origin + clipped.y * direction;
candidateSegments[slot] = start.x; candidateSegments[slot + 1u] = start.y; candidateSegments[slot + 2u] = end.x; candidateSegments[slot + 3u] = end.y;`}),...new o({id:`${t}-compact-edges`,input:d,flags:m,output:S,count:i.status.requiredCount}).getCommandNodes(e),p(e,{id:`${t}-materialize`,operation:`GPUVoronoiDiagram`,variant:`parallel-materialization`,bindings:[{name:`compactCandidateIds`,view:S,type:`u32`,access:`read`},{name:`candidateSegments`,view:x,type:`f32`,access:`read`},{name:`requiredCount`,view:i.status.requiredCount,type:`u32`,access:`read`},{name:`segmentsOut`,view:i.segments,type:`f32`,access:`read_write`}],invocationCount:i.segments.length,body:`if (index >= requiredCount[requiredCountOffset]) { return; }
  let candidate = compactCandidateIds[compactCandidateIdsOffset + index];
  let source = candidateSegmentsOffset + 4u * candidate;
  let segmentBase = segmentsOutOffset + 4u * index;
  segmentsOut[segmentBase] = candidateSegments[source];
  segmentsOut[segmentBase + 1u] = candidateSegments[source + 1u];
  segmentsOut[segmentBase + 2u] = candidateSegments[source + 2u];
  segmentsOut[segmentBase + 3u] = candidateSegments[source + 3u];`}),p(e,{id:`${t}-materialize-site-ids`,operation:`GPUVoronoiDiagram`,variant:`parallel-site-id-materialization`,bindings:[{name:`triangles`,view:n.triangles,type:`u32`,access:`read`},{name:`compactCandidateIds`,view:S,type:`u32`,access:`read`},{name:`requiredCount`,view:i.status.requiredCount,type:`u32`,access:`read`},{name:`siteIdsOut`,view:i.siteIds,type:`u32`,access:`read_write`}],invocationCount:i.siteIds.length,declarations:`fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}`,body:`if (index >= requiredCount[requiredCountOffset]) { return; }
  let candidate = compactCandidateIds[compactCandidateIdsOffset + index];
  let triangle = candidate / 3u;
  let edge = candidate % 3u;
  let a = vertex(triangle, edge);
  let b = vertex(triangle, (edge + 1u) % 3u);
  let siteBase = siteIdsOutOffset + 2u * index;
  siteIdsOut[siteBase] = min(a, b);
  siteIdsOut[siteBase + 1u] = max(a, b);`}),p(e,{id:`${t}-publish-status`,operation:`GPUVoronoiDiagram`,variant:`publish-status`,bindings:[{name:`requiredCountIn`,view:i.status.requiredCount,type:`u32`,access:`read`},{name:`countOut`,view:i.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:i.status.overflow,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const OUTPUT_CAPACITY: u32 = ${i.segments.length}u;`,body:`let requiredCount = requiredCountIn[requiredCountInOffset];
  countOut[countOutOffset] = min(requiredCount, OUTPUT_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, requiredCount > OUTPUT_CAPACITY);`})),w}};function L(e){return`
const POINT_COUNT: u32 = ${e.pointCount}u;
const TRIANGLE_CAPACITY: u32 = ${e.triangleCapacity}u;
${h}
${_}
fn site(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}
fn circumcenter(triangle: u32) -> vec3<f32> {
  let aRow = vertex(triangle, 0u);
  let bRow = vertex(triangle, 1u);
  let cRow = vertex(triangle, 2u);
  if (aRow >= POINT_COUNT || bRow >= POINT_COUNT || cRow >= POINT_COUNT) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  let a = site(aRow);
  let b = site(bRow);
  let c = site(cRow);
  let circle = getTriangleCircumcircle(a, b, c);
  if (circle.orientation == 0) { return vec3f(0.0); }
  let center = circle.center;
  // Nearly collinear triangles can overflow; NaN fails both comparisons.
  if (!(abs(center.x) <= ORIENT_FLOAT_MAXIMUM && abs(center.y) <= ORIENT_FLOAT_MAXIMUM)) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return vec3<f32>(center, f32(circle.orientation));
}
`}function R(e){return`
${g}
const POINT_COUNT: u32 = ${e.pointCount}u;
const TRIANGLE_CAPACITY: u32 = ${e.triangleCapacity}u;
fn site(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}`}var z=C,B=96,V=async e=>{let{device:r}=e;E(r);let i=ee(e.theme),a=S(z,e.parameters),o=e.width/e.height,s={worldWidth:o,worldHeight:1,width:e.width,height:e.height},c=w(e.seed),{positions:l}=b(c,{width:o,height:1},B,!0),u=Float32Array.from({length:192},()=>c()*Math.PI*2),d=Float32Array.from(l),f=n(B),p=I(f),m=j(r,d),h=T(r,f*3),g=T(r,1),_=T(r,1),v=T(r,1),y=j(r,Float32Array.of(0,0,o,1)),C=T(r,p*4),N=T(r,p*2),P=T(r,1),F=T(r,1),L=T(r,1),R=T(r,B),V=e=>ne.create(r,`demo-delaunay-voronoi-${e}`,r=>{let i=n(e),a=I(i),o=O(r,`positions`,m,`float32x2`,e),s=O(r,`triangle-count`,g,`uint32`,1),c=O(r,`triangles`,h,`uint32x3`,i);r.add(new t({positions:o,output:{triangles:c,status:{count:s,requiredCount:O(r,`triangle-required`,_,`uint32`,1),overflow:O(r,`triangle-overflow`,v,`uint32`,1)}}})),r.add(new re({positions:o,triangles:c,triangleCount:s,clipBounds:O(r,`clip-bounds`,y,`float32x4`,1),output:{segments:O(r,`segments`,C,`float32x4`,a),siteIds:O(r,`segment-sites`,N,`uint32x2`,a),status:{count:O(r,`segment-count`,P,`uint32`,1),requiredCount:O(r,`segment-required`,F,`uint32`,1),overflow:O(r,`segment-overflow`,L,`uint32`,1)}}}))}),H=new Map([[B,await V(B)]]),U=new Set,W=B,G=!1,K=new M(r,m,h,g,f),q=new te(r,C,P,p),J=new A(r,m,R,B),Y=null,X=!1,Z=e.reducedMotion?4:0,Q=new k(()=>{let e=Z,t=a.wander/100,n=Math.round(a.sites),o=H.get(n);o?W=n:(U.has(n)||(U.add(n),V(n).then(e=>{G?e.destroy():(H.set(n,e),Q.request())})),o=H.get(W));for(let n=1;n<B;n++)d[n*2]=l[n*2]+Math.sin(e*.6+u[n*2])*t,d[n*2+1]=l[n*2+1]+Math.cos(e*.5+u[n*2+1])*t;let[c,f]=Y??[l[0],l[1]];d[0]=c,d[1]=f,m.write(d),o?.run(),J.setInstanceCount(W);let p=i.categories[0],h=i.categories[1];K.update(s,i,{radius:.7,solid:[p[0],p[1],p[2],.45]}),q.update(s,i,{radius:1.1,solid:[h[0],h[1],h[2],.95]}),J.update(s,i,{mode:-1,radius:3,solid:[...i.ink,1]}),D(r,i,[K,q,J,...$.getLayers()])}),$=x({device:r,view:s,theme:i,reducedMotion:e.reducedMotion,kind:`drag`,hint:`drag a site`,rest:e=>e.toUnit([l[0],l[1]]),reportReadout:e.reportReadout,requestDraw:()=>Q.request(),resolve(e,t){return Y=[e.x,e.y],Q.request(),{glyphs:[{shape:`ring`,at:Y,radius:t.pixelSize*9},{shape:`crosshair`,at:Y,size:5}],readout:`site 1 ${X?`moved`:`at its gauge`} · ${Math.round(a.sites)} sites`}}});return{gesture:$.gesture,render(t){e.reducedMotion||(Z=t/1e3),$.advance(t),Q.flush()},setParameters(e){Object.assign(a,e),Q.request()},resize(e,t){s.width=e,s.height=t,$.refresh(),Q.request()},setPointer(e){e&&(X=!0),$.setPointer(e)},resetPointer:$.resetPointer,destroy(){G=!0,Q.cancel(),$.destroy(),K.destroy(),q.destroy(),J.destroy();for(let e of H.values())e.destroy();for(let e of[m,h,g,_,v,y,C,N,P,F,L,R])e.destroy()}}};export{V as default,z as parameters};