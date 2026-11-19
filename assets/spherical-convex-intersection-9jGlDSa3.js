import{o as e}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as t,T as n,f as r,x as i,y as a}from"./gpu-contributor-utils-BIwLQON5.js";import{v as o}from"./spatial-context-DwZEd4EZ.js";import{s}from"./wgsl-common-DfyXiYB0.js";import{a as c,i as l}from"./graph-chunk-dispatch-zBt8k2-o.js";import{a as u,c as d,i as f,n as p,s as m}from"./spherical-limits-C7rAYBu5.js";import{t as h}from"./define-run-B-brkCe4.js";import{d as g,s as _}from"./pointer-overlay-Qh8qflcA.js";import{i as v,m as y}from"./kernels-IDOsx_br.js";import{i as b}from"./polygons--0zAAHu9.js";import{a as x,i as S,n as C,o as w,r as T}from"./spherical-surveys-Vp2aL65w.js";import{Zc as E}from"./entry-preview-D70TdSbK.js";var D=class{props;id;spatialContext;constructor(e){this.props=e,this.id=e.id??`spherical-convex-intersection`,this.spatialContext=o(this.id,e.spatialContext??p,m);for(let n of[e.firstPositions,e.secondPositions,e.output.positions])t(n,[`float32x3`],this.id);for(let n of[e.firstOffsets,e.secondOffsets,e.output.status])t(n,[`uint32`],this.id);t(e.output.counts,[`uint32x2`],this.id),e.output.measures&&(t(e.output.measures,[`float32x4`],this.id),n(this.id,e.output.measures.length===e.firstOffsets.length-1,`output.measures must hold one entry per row`));let i=e.firstOffsets.length-1;n(this.id,i>=0&&e.secondOffsets.length===i+1,`firstOffsets and secondOffsets must describe the same number of rows`),r(this.id,`capacity`,e.capacity,1,64),n(this.id,e.output.positions.length===i*e.capacity&&e.output.counts.length===i&&e.output.status.length===i,`output.positions must hold rows times capacity entries, and output.counts and output.status one entry per row`),d(this.id,e.tolerance??1e-6,u,f),a(this.id,Object.values(e.output),[e.firstPositions,e.secondPositions,e.firstOffsets,e.secondOffsets]),l(this.id,e.output)}getCommandNodes(t){let{props:n,id:r}=this;return c(t,r,{firstPositions:n.firstPositions,secondPositions:n.secondPositions,firstOffsets:n.firstOffsets,secondOffsets:n.secondOffsets,...n.output}),i(r,t,[n.firstPositions,n.secondPositions,n.firstOffsets,n.secondOffsets,...Object.values(n.output)]),[e(t,{id:r,operation:`GPUSphericalConvexIntersection`,variant:`closed-hemisphere-convex`,invocationCount:n.firstOffsets.length-1,bindings:[{name:`firstPositions`,view:n.firstPositions,type:`f32`,access:`read`},{name:`secondPositions`,view:n.secondPositions,type:`f32`,access:`read`},{name:`firstOffsets`,view:n.firstOffsets,type:`u32`,access:`read`},{name:`secondOffsets`,view:n.secondOffsets,type:`u32`,access:`read`},{name:`output`,view:n.output.positions,type:`f32`,access:`read_write`},{name:`counts`,view:n.output.counts,type:`u32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`},...n.output.measures?[{name:`measures`,view:n.output.measures,type:`f32`,access:`read_write`}]:[]],declarations:`const SPHERICAL_TOLERANCE: f32 = ${n.tolerance??1e-6};
${s}
fn spherePosition(second: bool, row: u32) -> vec3f {
  if (second) { let offset = secondPositionsOffset + row * 3u; return vec3f(secondPositions[offset], secondPositions[offset + 1u], secondPositions[offset + 2u]); }
  let offset = firstPositionsOffset + row * 3u; return vec3f(firstPositions[offset], firstPositions[offset + 1u], firstPositions[offset + 2u]);
}
fn sphereContains(second: bool, begin: u32, end: u32, point: vec3f) -> bool {
  for (var edge = begin; edge < end; edge++) {
    let next = select(edge + 1u, begin, edge + 1u == end);
    let normal = normalize(cross(spherePosition(second, edge), spherePosition(second, next)));
    if (dot(normal, point) < -SPHERICAL_TOLERANCE) { return false; }
  }
  return true;
}
fn sphereValid(second: bool, begin: u32, end: u32) -> bool {
  let capacity = select(${n.firstPositions.length}u, ${n.secondPositions.length}u, second);
  if (begin > end || end > capacity || end - begin < 3u || end - begin > 32u) { return false; }
  var center = vec3f(0.0);
  for (var vertex = begin; vertex < end; vertex++) {
    let point = spherePosition(second, vertex);
    let squared = dot(point, point);
    if ((bitcast<u32>(squared) & 0x7f800000u) == 0x7f800000u || abs(squared - 1.0) > 1e-4) { return false; }
    let next = spherePosition(second, select(vertex + 1u, begin, vertex + 1u == end));
    if (length(cross(point, next)) <= SPHERICAL_TOLERANCE || !sphereContains(second, begin, end, point)) { return false; }
    center += point;
  }
  if (length(center) <= SPHERICAL_TOLERANCE) { return false; }
  center = normalize(center);
  for (var vertex = begin; vertex < end; vertex++) { if (dot(center, spherePosition(second, vertex)) <= SPHERICAL_TOLERANCE) { return false; } }
  for (var edge = begin; edge < end; edge++) {
    let next = select(edge + 1u, begin, edge + 1u == end);
    if (dot(center, normalize(cross(spherePosition(second, edge), spherePosition(second, next)))) <= SPHERICAL_TOLERANCE) { return false; }
  }
  return true;
}
fn sphereTriangleArea(first: vec3f, second: vec3f, third: vec3f) -> f32 {
  return 2.0 * atan2(abs(dot(first, cross(second - first, third - first))), 1.0 + dot(first, second) + dot(second, third) + dot(third, first));
}
fn spherePolygonArea(second: bool, begin: u32, end: u32) -> f32 {
  var area = 0.0;
  for (var vertex = begin + 1u; vertex + 1u < end; vertex++) { area += sphereTriangleArea(spherePosition(second, begin), spherePosition(second, vertex), spherePosition(second, vertex + 1u)); }
  return area;
}
fn sphereOnArc(first: vec3f, second: vec3f, point: vec3f) -> bool {
  let normal = normalize(cross(first, second));
  return dot(cross(first, point), normal) >= -SPHERICAL_TOLERANCE && dot(cross(point, second), normal) >= -SPHERICAL_TOLERANCE;
}
fn sphereAppend(candidates: ptr<function, array<vec3f, 64>>, count: ptr<function, u32>, point: vec3f) {
  for (var previous = 0u; previous < min(*count, 64u); previous++) { if (length((*candidates)[previous] - point) <= SPHERICAL_TOLERANCE * 4.0) { return; } }
  if (*count < 64u) { (*candidates)[*count] = point; }
  *count += 1u;
}`,body:`let firstBegin = firstOffsets[firstOffsetsOffset + index]; let firstEnd = firstOffsets[firstOffsetsOffset + index + 1u];
let secondBegin = secondOffsets[secondOffsetsOffset + index]; let secondEnd = secondOffsets[secondOffsetsOffset + index + 1u];
var failure = 0u; var count = 0u; var candidates: array<vec3f, 64>;
if (!sphereValid(false, firstBegin, firstEnd) || !sphereValid(true, secondBegin, secondEnd)) { failure = 1u; }
if (failure == 0u) {
  for (var vertex = firstBegin; vertex < firstEnd; vertex++) { let point = spherePosition(false, vertex); if (sphereContains(true, secondBegin, secondEnd, point)) { sphereAppend(&candidates, &count, point); } }
  for (var vertex = secondBegin; vertex < secondEnd; vertex++) { let point = spherePosition(true, vertex); if (sphereContains(false, firstBegin, firstEnd, point)) { sphereAppend(&candidates, &count, point); } }
  for (var firstEdge = firstBegin; firstEdge < firstEnd; firstEdge++) {
    let firstStart = spherePosition(false, firstEdge); let firstStop = spherePosition(false, select(firstEdge + 1u, firstBegin, firstEdge + 1u == firstEnd));
    let firstNormal = normalize(cross(firstStart, firstStop));
    for (var secondEdge = secondBegin; secondEdge < secondEnd; secondEdge++) {
      let secondStart = spherePosition(true, secondEdge); let secondStop = spherePosition(true, select(secondEdge + 1u, secondBegin, secondEdge + 1u == secondEnd));
      let secondNormal = normalize(cross(secondStart, secondStop));
      let crossing = cross(firstNormal, secondNormal); let magnitude = length(crossing);
      // Coincident planes are handled by contained endpoints. Near-parallel distinct planes are uncertain.
      if (magnitude <= SPHERICAL_TOLERANCE) {
        if (abs(dot(firstNormal, secondStart)) > SPHERICAL_TOLERANCE) { failure |= 4u; }
        continue;
      }
      for (var side = 0u; side < 2u; side++) {
        let point = crossing / magnitude * select(1.0, -1.0, side == 1u);
        if (sphereOnArc(firstStart, firstStop, point) && sphereOnArc(secondStart, secondStop, point) && sphereContains(false, firstBegin, firstEnd, point) && sphereContains(true, secondBegin, secondEnd, point)) { sphereAppend(&candidates, &count, point); }
      }
    }
  }
}
if (count > 64u) { failure |= 4u; }
if (count > ${n.capacity}u) { failure |= 2u; }
var center = vec3f(0.0);
for (var vertex = 0u; vertex < min(count, 64u); vertex++) { center += candidates[vertex]; }
if (count > 2u && length(center) > SPHERICAL_TOLERANCE) {
  center = normalize(center);
  let tangent = normalize(candidates[0] - center * dot(center, candidates[0]));
  let bitangent = cross(center, tangent);
  for (var vertex = 1u; vertex < min(count, 64u); vertex++) {
    let point = candidates[vertex]; let angle = guardedAtan2(dot(point, bitangent), dot(point, tangent)); var insertion = vertex;
    while (insertion > 0u) {
      let previous = candidates[insertion - 1u];
      if (guardedAtan2(dot(previous, bitangent), dot(previous, tangent)) <= angle) { break; }
      candidates[insertion] = previous; insertion -= 1u;
    }
    candidates[insertion] = point;
  }
}
let written = select(min(count, ${n.capacity}u), 0u, (failure & 5u) != 0u);
for (var slot = 0u; slot < ${n.capacity}u; slot++) {
  var point = vec3f(0.0); if (slot < written) { point = candidates[slot]; }
  let offset = outputOffset + (index * ${n.capacity}u + slot) * 3u;
  output[offset] = point.x; output[offset + 1u] = point.y; output[offset + 2u] = point.z;
}
counts[countsOffset + index * 2u] = written; counts[countsOffset + index * 2u + 1u] = count;
status[statusOffset + index] = failure;
${n.output.measures?`var intersectionArea = 0.0; var sourceFraction = 0.0; var targetFraction = 0.0; var unionArea = 0.0;
if (failure == 0u) {
  for (var vertex = 1u; vertex + 1u < count; vertex++) { intersectionArea += sphereTriangleArea(candidates[0], candidates[vertex], candidates[vertex + 1u]); }
  let firstArea = spherePolygonArea(false, firstBegin, firstEnd); let secondArea = spherePolygonArea(true, secondBegin, secondEnd);
  sourceFraction = intersectionArea / firstArea; targetFraction = intersectionArea / secondArea; unionArea = firstArea + secondArea - intersectionArea;
} else { intersectionArea = bitcast<f32>(0x7fc00000u | (index & 0u)); sourceFraction = intersectionArea; targetFraction = intersectionArea; unionArea = intersectionArea; }
measures[measuresOffset + index * 4u] = intersectionArea; measures[measuresOffset + index * 4u + 1u] = sourceFraction; measures[measuresOffset + index * 4u + 2u] = targetFraction; measures[measuresOffset + index * 4u + 3u] = unionArea;`:``}`})]}},O=32,k=_(E,`separation`),A=_(E,`surveyLatitude`),j=h({parameters:E,geographicWindow:C,caption:`A modeled survey footprint intersects the convex envelope of Lake Superior. The mapped shoreline provides context.`,createInputs:()=>{let e=S(`Lake Superior`);return{shore:e,hull:T(e)}},build(e,{shore:t,hull:n},r){let i=r.sides,a=e.allocate(`second-positions`,`float32x3`,i),o=e.allocate(`intersection`,`float32x3`,O),s=e.allocate(`counts`,`uint32x2`,1);e.graph.add(new D({firstPositions:e.upload(`hull`,`float32x3`,b([n])).view,secondPositions:a.view,firstOffsets:e.upload(`first-offsets`,`uint32`,Uint32Array.of(0,n.length)).view,secondOffsets:e.upload(`second-offsets`,`uint32`,Uint32Array.of(0,i)).view,capacity:O,output:{positions:o.view,counts:s.view,status:e.allocate(`status`,`uint32`,1).view}}));let c=y(e,`intersection`,o),l=e.allocate(`survey-outline`,`float32x4`,i);return{outputs:{shore:e.upload(`shore`,`float32x4`,x([t])),hull:e.upload(`hull-outline`,`float32x4`,x([n])),outline:l,counts:s,ring:v(e,`intersection`,c,s,O)},update:e=>{let t=w(e.separation,e.surveyLatitude,130,i,e.rotation);a.write(b([t])),l.write(x([t]))}}},rebuildOn:[`sides`],interaction:{kind:`drag`,hint:`drag the footprint`,rest:({toUnit:e})=>e([k.default,A.default]),resolve(e,{inputs:t,outputs:n,values:r,peek:i}){let a=g(k,e.x),o=g(A,e.y),s=i(n.counts);return{values:{separation:a,surveyLatitude:o},glyphs:[{shape:`crosshair`,at:[a,o]}],readout:`intersection ${s?s[0]:`…`} vertices`}}},draw:({outputs:e,theme:t})=>[{layer:`lines`,context:!0,segments:e.shore,radius:1.3,color:[...t.muted,.9]},{layer:`lines`,context:!0,segments:e.hull,radius:1.2,color:[.45,.65,1,.85]},{layer:`lines`,context:!0,segments:e.outline,radius:1.2,color:[1,.6,.35,.85]},{layer:`lines`,segments:e.ring,radius:3,color:[...t.ink,1]}]});export{j as default};