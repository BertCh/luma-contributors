import{r as e}from"./kernel-C3tfQOn4.js";import{n as t,t as n}from"./vertical-datum-BTN03tZP.js";import{o as r,t as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as a,S as o,u as s,x as c}from"./gpu-contributor-utils-BIwLQON5.js";import{a as l,f as u,r as d,u as f}from"./raster-grid-VB0AM_GY.js";import{c as p,d as m,h,l as g,n as _,o as v,r as y,t as b,u as x}from"./raster-band-utils-BcTxc8_Y.js";import{a as S,i as C,n as w,o as T,r as E}from"./terrain-features-utils-BLqx9E-Z.js";import{t as D}from"./define-run-B-brkCe4.js";import{t as O}from"./pixel-to-world-cNchXLzW.js";import{a as k,f as A,i as j,m as M,s as N,u as P}from"./terrain-Cc9dW1uq.js";import{V as F}from"./entry-preview-D70TdSbK.js";var I={unchanged:0,snapped:1,onRing:2,rejectedMove:3,rejectedHeight:4,noData:5,outside:6};function L(e){return s(e,20)}var R={cellSizeIndex:3,northEdgeIndex:5};function z(e,t,n=new Float32Array(8)){let r=`getGPUTerrainPeakSnapParameterValues`;v(r,n,8);let i=e.radius??150,a=e.maximumMove??300,o=e.maximumHeightChange??80;return T(r,`radius`,i,0,!0),T(r,`maximumMove`,a,0),T(r,`maximumHeightChange`,o,0),C(r,d(t).cellSize),n[0]=i,n[1]=a,n[2]=o,n[7]=e.interior===!1?0:1,u(n,t,R),n}var B=class{id;props;constructor(e){this.id=e.id??`terrain-peak-snap`,this.props=e;let{id:r}=this;if(e.candidateVerticalDatum!==void 0&&!e.candidateHeights)throw Error(`${r} candidateVerticalDatum needs candidateHeights`);n(r,e.candidateVerticalDatum,t(e.elevation,e.elevationVerticalDatum),`candidate`);let{output:i}=e,o=i.status?.radiusClamped;x(r,e.grid),f(r,e.cellSizeMode??`uniform`),m(r,e.grid,e.cellSizeMode),S(r,e.maximumRadiusPixels),a(e.candidates,[`float32x2`],`${r} candidates`);let s=e.candidates.length;if(s<1)throw Error(`${r} candidates must contain at least one row`);h(r,e.parameters,8);for(let[t,n]of[[`candidateHeights`,e.candidateHeights],[`candidateRadii`,e.candidateRadii],[`output.heights`,i.heights],[`output.snapDistance`,i.snapDistance]])if(n&&(a(n,[`float32`],`${r} ${t}`),n.length!==s))throw Error(`${r} ${t} must contain one value per candidate`);if(a(i.positions,[`float32x2`],`${r} output.positions`),i.positions.length!==s)throw Error(`${r} output.positions must contain one value per candidate`);if(a(i.snapStatus,[`uint32`],`${r} output.snapStatus`),i.snapStatus.length!==s)throw Error(`${r} output.snapStatus must contain one value per candidate`);if(o&&(a(o,[`uint32`],`${r} output.status.radiusClamped`),o.length<1))throw Error(`${r} output.status.radiusClamped must contain one row`);g(r,[i.positions,i.heights,i.snapStatus,i.snapDistance,o],[..._(e.elevation),e.parameters,e.candidates,e.candidateHeights,e.candidateRadii])}getCommandNodes(t){let{id:n,props:a}=this,{output:s}=a,{width:u,height:d}=a.grid,f=s.status?.radiusClamped;p(n,t,a.elevation,[]),c(n,t,[a.parameters,a.candidates,a.candidateHeights,a.candidateRadii,s.positions,s.heights,s.snapStatus,s.snapDistance,f]);let m=a.candidates.length;o(n,`candidates rows`,m,L(t.device));let h=a.cellSizeMode??`uniform`,g=S(n,a.maximumRadiusPixels),_=y(t,n,a.elevation,a.grid,!0),v=[..._.nodes],x=_.band.validity,C=e(t,`${n}-results`,`uint32`,m*5);f&&v.push(i(t,{id:`${n}-overflow-clear`,operation:`GPUTerrainPeakSnap`,view:f,type:`u32`,value:`0u`,componentCount:1}));let T=[{name:`elevationValues`,view:_.band.storage.values,type:`f32`,access:`read`},{name:`elevationValidity`,view:x,type:`u32`,access:`read`},{name:`settings`,view:a.parameters,type:`f32`,access:`read`},{name:`candidates`,view:a.candidates,type:`f32`,access:`read`},...a.candidateHeights?[{name:`candidateHeights`,view:a.candidateHeights,type:`f32`,access:`read`}]:[],...a.candidateRadii?[{name:`candidateRadii`,view:a.candidateRadii,type:`f32`,access:`read`}]:[],{name:`results`,view:C,type:`u32`,access:`read_write`},...f?[{name:`overflowValues`,view:f,type:`atomic<u32>`,access:`read_write`}]:[]],D=I;return v.push(r(t,{id:`${n}-search`,operation:`GPUTerrainPeakSnap`,variant:`disc-${h}`,bindings:T,invocationCount:m*256,workgroupSize:256,guardIndex:!1,declarations:`const WIDTH: u32 = ${u}u;
const HEIGHT: u32 = ${d}u;
const CANDIDATE_COUNT: u32 = ${m}u;
var<workgroup> reduceIndex: array<u32, 256>;
var<workgroup> reduceHeight: array<f32, 256>;
// True when (height, index) beats the incumbent: higher first, lower index on ties, any pixel beats none.
fn isBetterPixel(height: f32, pixelIndex: u32, incumbentHeight: f32, incumbentIndex: u32) -> bool {
  if (pixelIndex == NO_INDEX) { return false; }
  return incumbentIndex == NO_INDEX || height > incumbentHeight ||
    (height == incumbentHeight && pixelIndex < incumbentIndex);
}
const MAXIMUM_RADIUS_PIXELS: i32 = ${g};
const NO_INDEX: u32 = 0xffffffffu;
const STATUS_UNCHANGED: u32 = ${D.unchanged}u;
const STATUS_SNAPPED: u32 = ${D.snapped}u;
const STATUS_ON_RING: u32 = ${D.onRing}u;
const STATUS_REJECTED_MOVE: u32 = ${D.rejectedMove}u;
const STATUS_REJECTED_HEIGHT: u32 = ${D.rejectedHeight}u;
const STATUS_NO_DATA: u32 = ${D.noData}u;
const STATUS_OUTSIDE: u32 = ${D.outside}u;
${E}
${b}
${l(h,R)}
${w}
var<private> snapStatus: u32;
var<private> snapPosition: vec2<f32>;
var<private> snapHeight: f32;
var<private> snapDistanceValue: f32;
fn isValidPixel(column: u32, row: u32) -> bool {
  return elevationValidity[elevationValidityOffset + row * WIDTH + column] != 0u;
}
fn getElevation(column: u32, row: u32) -> f32 {
  return elevationValues[elevationValuesOffset + row * WIDTH + column];
}
// Bilinear DEM at an in-grid position; NaN unless all four surrounding pixels are valid.
fn sampleElevation(position: vec2<f32>, seed: u32) -> f32 {
  let base = vec2<u32>(floor(position));
  let next = min(base + vec2<u32>(1u), vec2<u32>(WIDTH - 1u, HEIGHT - 1u));
  let fraction = position - floor(position);
  if (!isValidPixel(base.x, base.y) || !isValidPixel(next.x, base.y) ||
      !isValidPixel(base.x, next.y) || !isValidPixel(next.x, next.y)) {
    return getQuietNaN(seed);
  }
  let top = mix(getElevation(base.x, base.y), getElevation(next.x, base.y), fraction.x);
  let bottom = mix(getElevation(base.x, next.y), getElevation(next.x, next.y), fraction.x);
  return mix(top, bottom, fraction.y);
}
var<private> snapNearest: vec2<i32>;
var<private> snapCell: vec2<f32>;
var<private> snapRadiusSquared: f32;
var<private> snapExtent: vec2<i32>;
var<private> snapDemHeight: f32;
// Validates the candidate and sets up the disc search. Returns false when snapStatus is final.
fn prepareSnap(index: u32, candidate: vec2<f32>, isLeader: bool) -> bool {
  snapStatus = STATUS_OUTSIDE;
  snapPosition = candidate;
  snapHeight = getQuietNaN(index);
  snapDistanceValue = 0.0;
  if (!(isFiniteValue(candidate.x) && isFiniteValue(candidate.y) &&
      candidate.x >= 0.0 && candidate.y >= 0.0 &&
      candidate.x <= f32(WIDTH - 1u) && candidate.y <= f32(HEIGHT - 1u))) {
    return false;
  }
  let demHeight = sampleElevation(candidate, index);
  snapHeight = demHeight;
  snapStatus = STATUS_NO_DATA;
  let nearest = vec2<i32>(i32(floor(candidate.x + 0.5)), i32(floor(candidate.y + 0.5)));
  let cell = getGroundCellSize(u32(nearest.y));
  var requested = settings[settingsOffset];
  ${a.candidateRadii?`let candidateRadius = candidateRadii[candidateRadiiOffset + index];
  if (isFiniteValue(candidateRadius) && candidateRadius > 0.0) { requested = candidateRadius; }`:``}
  if (!(isFiniteValue(cell.x) && isFiniteValue(cell.y) && cell.x > 0.0 && cell.y > 0.0 &&
      isFiniteValue(requested) && requested > 0.0)) {
    return false;
  }
  ${f?`if (isLeader && isRadiusClamped(requested, cell)) { atomicMax(&overflowValues[overflowValuesOffset], 1u); }`:``}
  let radius = getEffectiveRadius(requested, cell);
  let radiusSquared = radius * radius;
  let extentX = getDiscExtent(radius, cell.x);
  let extentY = getDiscExtent(radius, cell.y);
  snapNearest = nearest;
  snapCell = cell;
  snapRadiusSquared = radiusSquared;
  snapExtent = vec2<i32>(extentX, extentY);
  snapDemHeight = demHeight;
  return true;
}
// Applies the move, height, and ring policies to the disc maximum found by the cooperative scan.
fn finishSnap(index: u32, candidate: vec2<f32>, bestIndex: u32, bestHeight: f32) {
  let nearest = snapNearest;
  let cell = snapCell;
  let radiusSquared = snapRadiusSquared;
  let demHeight = snapDemHeight;
  if (bestIndex == NO_INDEX) { return; }
  let bestPixel = vec2<i32>(i32(bestIndex % WIDTH), i32(bestIndex / WIDTH));
  if (settings[settingsOffset + 7u] != 0.0 &&
      isOnDiscRing(bestPixel, candidate, cell, radiusSquared)) {
    snapStatus = STATUS_ON_RING;
    return;
  }
  if (bestPixel.x == nearest.x && bestPixel.y == nearest.y) {
    snapStatus = STATUS_UNCHANGED;
    return;
  }
  let moveX = (f32(bestPixel.x) - candidate.x) * cell.x;
  let moveY = (f32(bestPixel.y) - candidate.y) * cell.y;
  let moveSquared = moveX * moveX + moveY * moveY;
  let maximumMove = settings[settingsOffset + 1u];
  if (moveSquared > maximumMove * maximumMove) {
    snapStatus = STATUS_REJECTED_MOVE;
    return;
  }
  var reference = demHeight;
  ${a.candidateHeights?`let candidateHeight = candidateHeights[candidateHeightsOffset + index];
  if (isFiniteValue(candidateHeight)) { reference = candidateHeight; }`:``}
  if (isFiniteValue(reference) &&
      abs(bestHeight - reference) > settings[settingsOffset + 2u]) {
    snapStatus = STATUS_REJECTED_HEIGHT;
    return;
  }
  snapStatus = STATUS_SNAPPED;
  snapPosition = vec2<f32>(f32(bestPixel.x), f32(bestPixel.y));
  snapHeight = bestHeight;
  snapDistanceValue = sqrt(moveSquared);
}`,body:`let candidateIndex = workgroupIndex;
  let isLeader = localInvocationIndex == 0u;
  let inRange = candidateIndex < CANDIDATE_COUNT;
  var candidate = vec2<f32>(0.0, 0.0);
  if (inRange) {
    candidate = vec2<f32>(candidates[candidatesOffset + 2u * candidateIndex], candidates[candidatesOffset + 2u * candidateIndex + 1u]);
  }
  snapStatus = STATUS_OUTSIDE;
  snapPosition = candidate;
  snapHeight = getQuietNaN(candidateIndex);
  snapDistanceValue = 0.0;
  let prepared = inRange && prepareSnap(candidateIndex, candidate, isLeader);
  // The whole workgroup scans one candidate's disc; the (height, -index) maximum is order independent.
  var bestIndex = NO_INDEX;
  var bestHeight = 0.0;
  if (prepared) {
    let extent = snapExtent;
    let spanX = u32(2 * extent.x + 1);
    let scanCount = spanX * u32(2 * extent.y + 1);
    for (var scan = localInvocationIndex; scan < scanCount; scan += 256u) {
      let offsetX = i32(scan % spanX) - extent.x;
      let offsetY = i32(scan / spanX) - extent.y;
      let pixel = snapNearest + vec2<i32>(offsetX, offsetY);
      if (pixel.x < 0 || pixel.y < 0 || pixel.x >= i32(WIDTH) || pixel.y >= i32(HEIGHT)) { continue; }
      if (!isInDisc(pixel, candidate, snapCell, snapRadiusSquared)) { continue; }
      if (!isValidPixel(u32(pixel.x), u32(pixel.y))) { continue; }
      let pixelIndex = u32(pixel.y) * WIDTH + u32(pixel.x);
      let pixelHeight = elevationValues[elevationValuesOffset + pixelIndex];
      if (isBetterPixel(pixelHeight, pixelIndex, bestHeight, bestIndex)) {
        bestIndex = pixelIndex;
        bestHeight = pixelHeight;
      }
    }
  }
  reduceIndex[localInvocationIndex] = bestIndex;
  reduceHeight[localInvocationIndex] = bestHeight;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      let otherIndex = reduceIndex[localInvocationIndex + stride];
      let otherHeight = reduceHeight[localInvocationIndex + stride];
      if (isBetterPixel(otherHeight, otherIndex, reduceHeight[localInvocationIndex], reduceIndex[localInvocationIndex])) {
        reduceIndex[localInvocationIndex] = otherIndex;
        reduceHeight[localInvocationIndex] = otherHeight;
      }
    }
    workgroupBarrier();
  }
  if (isLeader && inRange) {
    if (prepared && reduceIndex[0] != NO_INDEX) {
      finishSnap(candidateIndex, candidate, reduceIndex[0], reduceHeight[0]);
    }
    let resultBase = resultsOffset + 5u * candidateIndex;
    results[resultBase] = snapStatus;
    results[resultBase + 1u] = bitcast<u32>(snapPosition.x);
    results[resultBase + 2u] = bitcast<u32>(snapPosition.y);
    results[resultBase + 3u] = bitcast<u32>(snapHeight);
    results[resultBase + 4u] = bitcast<u32>(snapDistanceValue);
  }`}),r(t,{id:`${n}-unpack`,operation:`GPUTerrainPeakSnap`,variant:`unpack`,bindings:[{name:`results`,view:C,type:`u32`,access:`read`},{name:`positionValues`,view:s.positions,type:`f32`,access:`read_write`},{name:`heightValues`,view:s.heights,type:`u32`,access:`read_write`},{name:`statusValues`,view:s.snapStatus,type:`u32`,access:`read_write`},...s.snapDistance?[{name:`distanceValues`,view:s.snapDistance,type:`u32`,access:`read_write`}]:[]],invocationCount:m,body:`let resultBase = resultsOffset + 5u * index;
  statusValues[statusValuesOffset + index] = results[resultBase];
  positionValues[positionValuesOffset + 2u * index] = bitcast<f32>(results[resultBase + 1u]);
  positionValues[positionValuesOffset + 2u * index + 1u] = bitcast<f32>(results[resultBase + 2u]);
  heightValues[heightValuesOffset + index] = results[resultBase + 3u];
  ${s.snapDistance?`distanceValues[distanceValuesOffset + index] = results[resultBase + 4u];`:``}`})),v}},V=11,H=[`unchanged`,`snapped`,`on a ring`,`move too long`,`height off`,`no data`,`outside`],U=[[.55,.58,.65],[.2,.8,.4],[.95,.65,.2],[.9,.25,.25],[.65,.3,.85],[.4,.4,.4],[.4,.4,.4]],W=D({parameters:F,caption:`Elevation and rough peak positions in; positions moved onto the actual summits (or rejected) out.`,createInputs:({random:e,world:t})=>{let n=j(t),r=N(n,V,8).map(t=>({column:t%n.columns,row:Math.floor(t/n.columns),height:n.elevation[t]+(e()-.5)*160,direction:e()*Math.PI*2,amount:.4+.6*e()})),i=r[0],a=[Math.min(i.column+2,n.columns-1),Math.min(i.row+1,n.rows-1)];return{...n,peaks:r,carried:a}},build(e,t){let{grid:n,band:r,elevation:i}=M(e,t),a=t.peaks.length+1,o=e.allocate(`candidates`,`float32x2`,a),s=e.allocate(`candidate-heights`,`float32`,a),c=e.allocate(`candidates-drawn`,`float32x2`,a),l=e.allocate(`parameters`,`float32`,8),u=e.allocate(`snapped`,`float32x2`,a),d=e.allocate(`heights`,`float32`,a),f=e.allocate(`snap-status`,`uint32`,a);return e.graph.add(new B({grid:n,elevation:r,maximumRadiusPixels:24,candidates:o.view,candidateHeights:s.view,parameters:l.view,output:{positions:u.view,heights:d.view,snapStatus:f.view}})),{outputs:{elevation:i,snapped:u,heights:d,snappedDrawn:O(e,`snapped-drawn`,u,t,t.world,.5),snapStatus:f,candidatesDrawn:c},update:e=>{let r=t.peaks.map(n=>[Math.min(Math.max(n.column+Math.cos(n.direction)*n.amount*e.offset,0),t.columns-1),Math.min(Math.max(n.row+Math.sin(n.direction)*n.amount*e.offset,0),t.rows-1)]),[i,a]=t.carried;r.push([i,a]),o.write(Float32Array.from(r.flat())),s.write(Float32Array.from([...t.peaks.map(e=>e.height),t.elevation[Math.round(a)*t.columns+Math.round(i)]])),c.write(Float32Array.from(r.flatMap(([e,n])=>P(t,t.world,e,n)))),l.write(z({radius:e.radius,maximumMove:e.maximumMove,maximumHeightChange:e.maximumHeightChange},n))}}},interaction:{kind:`drag`,hint:`drop a survey point`,rest:({inputs:e})=>[(e.carried[0]+.5)/e.columns,(e.carried[1]+.5)/e.rows],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){t.carried=[Math.min(Math.max(e.x/i.width*t.columns-.5,0),t.columns-1),Math.min(Math.max((1-e.y/i.height)*t.rows-.5,0),t.rows-1)];let o=A(t,i),s=[{shape:`ring`,at:[e.x,e.y],radius:r.radius*o},{shape:`ring`,at:[e.x,e.y],radius:r.maximumMove*o,dashed:!0}],c=a(n.snapped),l=a(n.snapStatus),u=a(n.heights),d=t.peaks.length,f=a(n.candidatesDrawn);c&&l&&f&&t.peaks.forEach((e,n)=>{let a=[f[n*2],f[n*2+1]];if(s.push({shape:`ring`,at:a,radius:r.radius*o}),l[n]!==1)return;let u=P(t,i,c[n*2],c[n*2+1]);s.push({shape:`arrow`,from:a,to:u})});let p=`radius ${r.radius} m · maximumMove ${r.maximumMove} m`;return c&&l&&u&&(l[d]===1&&s.push({shape:`arrow`,from:[e.x,e.y],to:P(t,i,c[d*2],c[d*2+1])}),p=`${H[l[d]]??`unknown`} · top ${Math.round(u[d])} m · radius ${r.radius} m`),{rerun:!0,glyphs:s,readout:p}}},draw:({inputs:e,outputs:t})=>[k(e,t.elevation),{layer:`dots`,context:!0,positions:t.candidatesDrawn,radius:5},{layer:`dots`,positions:t.snappedDrawn,values:t.snapStatus,ramp:`categorical`,categories:U,radius:13}]});export{W as default};