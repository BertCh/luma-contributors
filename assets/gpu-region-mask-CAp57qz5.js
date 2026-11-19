import{t as e}from"./spatial-statistics-collective-node-BlCi_ESE.js";import{i as t}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as n}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{l as r}from"./gpu-command-graph-BdzdmV6u.js";import{r as i,u as a}from"./kernel-C3tfQOn4.js";import{n as o,o as s,t as c}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as l,c as u,x as d,y as f}from"./gpu-contributor-utils-BIwLQON5.js";import{t as p}from"./status-aOwAAXek.js";import{_ as m,n as h,v as g}from"./spatial-context-DwZEd4EZ.js";import{n as _,r as v}from"./lon-lat-ring-wgsl-DpZBIfuE.js";import{u as y}from"./wgsl-common-DfyXiYB0.js";import{t as b}from"./exact-orientation-wgsl-B_jI-HEc.js";var x={selectedCount:0,valueCount:1,sum:2,mean:3,minimum:4,maximum:5,histogramOutsideCount:6,flags:7,histogram:8},S={selectionTruncated:1,regionTruncated:2,candidatesTruncated:4},C=Object.freeze({edgeModels:h,coordinateSpaces:Object.freeze([`planar`,`longitude-latitude`]),metrics:Object.freeze([`none`,`native`])}),w=Object.freeze({...C,measure:`distance`});function T(e,t,n){let r=g(e,t,n===`radius`?w:C);if(n===`radius`&&r.coordinateSpace===`longitude-latitude`)throw Error(`${e} radius regions need planar coordinates`);return r}function E(e,t,n,r){if(t!==void 0&&(r.kind===`pick-region`||r.kind===`mask`||`screenTransform`in r&&r.screenTransform!==void 0))throw Error(`${e} regionSpatialContext needs a region in position coordinates`);m(e,`region`,t,n,C.edgeModels)}var D=y,O=class{id;props;spatialContext;constructor(e){this.id=e.id??`region-mask`,this.props=e,this.spatialContext=T(this.id,e.spatialContext,e.region.kind),E(this.id,e.regionSpatialContext,this.spatialContext,e.region);let{id:n}=this,{positions:r,region:i,output:o}=e;for(let e of u(r))l(e,[`float32x2`],`${n} positions`);if(a(o.mask,`${n} output.mask`),o.mask.length!==r.length)throw Error(`${n} output.mask length must equal positions length`);if(p(`${n} output.status`,o.status),!o.status.overflow)throw Error(`${n} output.status.overflow is required`);if(i.kind===`rectangle`){if(l(i.bounds,[`float32`],`${n} bounds`),i.bounds.length!==4)throw Error(`${n} bounds must contain four float32 values`)}else{if(l(i.vertices,[`float32x2`],`${n} vertices`),i.vertices.length<3)throw Error(`${n} vertices must hold at least three vertices`);if(a(i.vertexCount,`${n} vertexCount`),i.vertexCount.length<1)throw Error(`${n} vertexCount must contain one uint32 row`)}if(i.screenTransform&&(l(i.screenTransform,[`float32`],`${n} screenTransform`),i.screenTransform.length!==20))throw Error(`${n} screenTransform must contain 20 float32 values`);if(t(n,{mask:o.mask,...o.status}),f(n,[o.mask,o.status.overflow,o.status.antimeridianUncertainCount,o.status.poleUncertainCount],k(e)),o.mask.buffer===o.status.overflow.buffer)throw Error(`${n} output.mask and overflow must use separate buffers`)}getCommandNodes(t){n(t,this.id,this.props,{});let{id:i,props:a}=this,{positions:o,region:c}=a,l=a.output.mask,{overflow:f}=a.output.status;d(i,t,[l,f,a.output.status.antimeridianUncertainCount,a.output.status.poleUncertainCount,...k(a)]);let p=c.kind===`polygon`,m=c.kind===`rectangle`?c.bounds:c.vertices,h=c.kind===`polygon`?c.vertices.length:0,g=A(t,i,`GPURegionMask`,this.spatialContext,c,a.output.status),_=[{name:`overflow`,view:f,type:`u32`,access:`read_write`}];c.kind===`polygon`&&_.push({name:`vertexCount`,view:c.vertexCount,type:`u32`,access:`read`}),g.push(s(t,{id:`${i}-overflow`,operation:`GPURegionMask`,variant:`overflow`,bindings:_,invocationCount:1,body:p?`overflow[overflowOffset] = select(0u, 1u, vertexCount[vertexCountOffset] > ${h}u);`:`overflow[overflowOffset] = 0u;`}));let v=c.screenTransform?`
struct RegionPoint { position: vec2f, valid: bool }
fn toRegionSpace(world: vec2f) -> RegionPoint {
  let t = screenTransformOffset;
  let column0 = vec4f(screenTransform[t], screenTransform[t + 1u], screenTransform[t + 2u], screenTransform[t + 3u]);
  let column1 = vec4f(screenTransform[t + 4u], screenTransform[t + 5u], screenTransform[t + 6u], screenTransform[t + 7u]);
  let column3 = vec4f(screenTransform[t + 12u], screenTransform[t + 13u], screenTransform[t + 14u], screenTransform[t + 15u]);
  let clip = column0 * world.x + column1 * world.y + column3;
  if (!(clip.w > 0.0)) { return RegionPoint(vec2f(0.0), false); }
  let ndc = clip.xy / clip.w;
  let pixel = vec2f(
    (ndc.x * 0.5 + 0.5) * screenTransform[t + 16u],
    (0.5 - ndc.y * 0.5) * screenTransform[t + 17u]
  );
  return RegionPoint(pixel, isFiniteValue(pixel.x) && isFiniteValue(pixel.y));
}`:`
struct RegionPoint { position: vec2f, valid: bool }
fn toRegionSpace(world: vec2f) -> RegionPoint { return RegionPoint(world, true); }`,y=p?`
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(shape[shapeOffset + vertexIndex * 2u], shape[shapeOffset + vertexIndex * 2u + 1u]);
}
${b}
// Inclusive even-odd test, the boundary rule of the rectangle region and of published
// GPUPointSpatialQuery: a point on an edge or a vertex is inside. Elsewhere it is the crossing
// test (PNPOLY, half-open in y) with an exact side test: an edge that straddles the point's y is
// crossed when the point lies strictly left of the upward edge (right of a downward one), decided
// by the sign of the exact orientation rather than an f32 crossing x. A point is on an edge when
// it is inside the edge's bounding box and the exact orientation is zero. Edges whose sign cannot
// be certified count no crossing and no boundary.
fn isInsideRegion(point: vec2f) -> bool {
  let activeCount = min(vertexCount[vertexCountOffset], ${h}u);
  if (activeCount < 3u) { return false; }
  var inside = false;
  var previous = readVertex(activeCount - 1u);
  for (var vertexIndex = 0u; vertexIndex < activeCount; vertexIndex++) {
    let current = readVertex(vertexIndex);
    let straddles = (current.y > point.y) != (previous.y > point.y);
    let inBox = all(point >= min(previous, current)) && all(point <= max(previous, current));
    if (straddles || inBox) {
      let sign = orientSign(previous, current, point);
      if (inBox && sign == 0) { return true; }
      if (straddles && sign != 2 && sign != 0 && (sign > 0) == (current.y > previous.y)) { inside = !inside; }
    }
    previous = current;
  }
  return inside;
}`:`
// Inclusive bounds; invalid bounds (non-finite or min > max) select nothing.
fn isInsideRegion(point: vec2f) -> bool {
  let minimum = vec2f(shape[shapeOffset], shape[shapeOffset + 1u]);
  let maximum = vec2f(shape[shapeOffset + 2u], shape[shapeOffset + 3u]);
  let validBounds = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) && all(minimum <= maximum);
  return validBounds && all(point >= minimum) && all(point <= maximum);
}`,x=o instanceof r,S=0;for(let[n,r]of u(o).entries()){if(r.length>0){let a=[{name:`positions`,view:r,type:`f32`,access:`read`},{name:`shape`,view:m,type:`f32`,access:`read`},{name:`outputMask`,view:l,type:`u32`,access:`read_write`}];c.kind===`polygon`&&a.push({name:`vertexCount`,view:c.vertexCount,type:`u32`,access:`read`}),c.screenTransform&&a.push({name:`screenTransform`,view:c.screenTransform,type:`f32`,access:`read`}),g.push(e(t,{id:x?`${i}-chunk-${n}`:i,operation:`GPURegionMask`,variant:`${c.kind}${c.screenTransform?`-screen`:``}`,bindings:a,invocationCount:r.length,parameterValues:[S],declarations:`
${D}
${v}
${y}`,body:`if (index >= INVOCATION_COUNT) { return; }
  let ROW_START = getChunkParameter(36u);
  let world = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  var selected = false;
  if (isFiniteValue(world.x) && isFiniteValue(world.y)) {
    let regionPoint = toRegionSpace(world);
    selected = regionPoint.valid && isInsideRegion(regionPoint.position);
  }
  outputMask[outputMaskOffset + ROW_START + index] = select(0u, 1u, selected);`}))}S+=r.length}return g}};function k(e){let{region:t}=e;return[e.positions,...t.kind===`rectangle`?[t.bounds]:[t.vertices,t.vertexCount],...t.screenTransform?[t.screenTransform]:[]]}function A(e,t,n,r,a,s){if(r.coordinateSpace!==`longitude-latitude`||a.kind!==`polygon`||a.screenTransform||!s||!v(s))return[];let l=[];for(let r of[`antimeridianUncertainCount`,`poleUncertainCount`]){let i=s[r];i&&l.push(c(e,{id:`${t}-clear-${r}`,operation:n,view:i,type:`u32`,value:`0u`,componentCount:1}))}let u=i(e,`${t}-lasso-ring-offsets`,`uint32`,2),d=e.createDataView(u.buffer,{format:`uint32`,length:1}),f=e.createDataView(u.buffer,{format:`uint32`,length:1,byteOffset:4});l.push(c(e,{id:`${t}-lasso-ring-start`,operation:n,view:d,type:`u32`,value:`0u`,componentCount:1}));let p=e.createDataView(a.vertices.buffer,{format:`uint32`,length:a.vertices.length,byteOffset:a.vertices.byteOffset});return l.push(o(e,{id:`${t}-lasso-ring-end`,operation:n,requiredCount:a.vertexCount,output:{ids:p,status:{count:f,overflow:i(e,`${t}-lasso-check-overflow`,`uint32`,1)}}})),l.push(_(e,{id:`${t}-lon-lat-polygon-check-0`,operation:n,positions:a.vertices,ringOffsets:u,status:s})),l}export{x as a,S as i,D as n,T as o,A as r,E as s,O as t};