import{n as e}from"./raster-storage-preflight-CIxFc-0g.js";import{t}from"./status-aOwAAXek.js";import{S as n,T as r,p as i,s as a}from"./gpu-contributor-utils-BIwLQON5.js";import{r as o}from"./kernel-C3tfQOn4.js";import{o as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as c,i as l,r as u}from"./raster-expansion-utils-AZOLseCG.js";import{a as d,c as f,f as p,s as m}from"./raster-relaxation-Di9wjtxp.js";import{n as h,r as g,t as _}from"./gpu-distance-field-B7VYu8Kj.js";import{o as v}from"./gpu-command-graph-BdzdmV6u.js";function y(t,n,r){let{grid:i,output:a}=r,c=i.width*i.height,l=h(i),u=Math.max(c,8,4*l.columnBandCount*i.width,2*l.rowBandCount*i.height),d=new v(a.radius.buffer,{format:a.radius.format,byteOffset:a.radius.byteOffset,byteStride:a.radius.byteStride,rowByteLength:a.radius.rowByteLength,length:c});e(t,n,[r.input,a.medialAxis,d],{"distance-transform planes":u*4,"distance-transform merge slots":l.mergeRoundCount>0?(l.mergeRoundCount+1)*256:0});let f=o(t,`${n}-background`,`uint32`,c),p=o(t,`${n}-nearest-background`,`uint32`,c),m=o(t,`${n}-distance-parameters`,`float32`,8),g=t.createDataView(a.radius.buffer,{...a.radius,length:c});return[s(t,{id:`${n}-background`,operation:`GPURasterMedialAxis`,variant:`background`,invocationCount:c,bindings:[{name:`input`,view:r.input,type:`u32`,access:`read`},{name:`seeds`,view:f,type:`u32`,access:`read_write`}],body:`seeds[seedsOffset + index] = u32(input[inputOffset + index] == 0u);`}),s(t,{id:`${n}-distance-parameters`,operation:`GPURasterMedialAxis`,variant:`pixel-metric`,invocationCount:8,bindings:[{name:`settings`,view:m,type:`f32`,access:`read_write`}],body:`var value = select(0.0, 1.0, index >= 2u && index <= 5u);
  if (index == 6u) { value = bitcast<f32>(0x7f800000u | (index & 0u)); }
  settings[settingsOffset + index] = value;`}),...new _({id:`${n}-distance`,grid:i,parameters:m,seedMask:f,mode:`exact`,output:{distances:g,nearestCells:p}}).getCommandNodes(t),s(t,{id:`${n}-maximal-disks`,operation:`GPURasterMedialAxis`,variant:`primitive-step`,invocationCount:c,bindings:[{name:`nearest`,view:p,type:`u32`,access:`read`},{name:`radius`,view:a.radius,type:`f32`,access:`read_write`},{name:`axis`,view:a.medialAxis,type:`u32`,access:`read_write`}],declarations:`const WIDTH: u32 = ${i.width}u;
const HEIGHT: u32 = ${i.height}u;
fn squaredLength(value: vec2<i32>) -> u32 {
  let absolute = vec2<u32>(abs(value));
  return absolute.x * absolute.x + absolute.y * absolute.y;
}
// Four nearest virtual exterior centers suffice; farther exterior centers cannot win.
fn getBackgroundOffset(cell: u32) -> vec2<i32> {
  let column = i32(cell % WIDTH);
  let row = i32(cell / WIDTH);
  var offset = vec2<i32>(column + 1, 0);
  if (i32(WIDTH) - column < offset.x) { offset = vec2<i32>(column - i32(WIDTH), 0); }
  if (row + 1 < abs(offset.x)) { offset = vec2<i32>(0, row + 1); }
  if (i32(HEIGHT) - row < abs(offset.x) + abs(offset.y)) {
    offset = vec2<i32>(0, row - i32(HEIGHT));
  }
  let background = nearest[nearestOffset + cell];
  if (background != 0xffffffffu) {
    let interior = vec2<i32>(column - i32(background % WIDTH), row - i32(background / WIDTH));
    if (squaredLength(interior) < squaredLength(offset)) { offset = interior; }
  }
  return offset;
}
fn greatestCommonDivisor(first: u32, second: u32) -> u32 {
  var dividend = first;
  var divisor = second;
  loop {
    if (divisor == 0u) { return dividend; }
    let remainder = dividend % divisor;
    dividend = divisor;
    divisor = remainder;
  }
}`,body:`let column = index % WIDTH;
  let row = index / WIDTH;
  let exteriorRadius = min(min(column + 1u, WIDTH - column), min(row + 1u, HEIGHT - row));
  radius[radiusOffset + index] = min(radius[radiusOffset + index], f32(exteriorRadius));
  let offset = getBackgroundOffset(index);
  let squaredRadius = squaredLength(offset);
  var maximal = squaredRadius > 0u;
  if (maximal) {
    let divisor = greatestCommonDivisor(u32(abs(offset.x)), u32(abs(offset.y)));
    let step = offset / i32(divisor);
    let candidate = vec2<i32>(i32(column), i32(row)) + step;
    if (all(candidate >= vec2<i32>(0)) && candidate.x < i32(WIDTH) && candidate.y < i32(HEIGHT)) {
      let candidateCell = u32(candidate.y) * WIDTH + u32(candidate.x);
      // q and its background b are within [−1, dimension], so the two squares sum to < 2^32.
      maximal = squaredLength(getBackgroundOffset(candidateCell)) != squaredLength(offset + step);
    }
  }
  axis[axisOffset + index] = u32(maximal);`})]}var b=64,x=class{id;props;estimatedCommandNodeCount;constructor(e){this.id=e.id??`raster-skeleton`,this.props=e;let{id:n}=this,o=c(n,e.grid,[e.input],a(e.output));u(e.input,o,n,`input`),u(e.output.skeleton,o,n,`output.skeleton`),u(e.output.status.converged,1,n,`output.status.converged`),u(e.output.status.iterationLimitReached,1,n,`output.status.iterationLimitReached`),u(e.output.status.iterationCount,1,n,`output.status.iterationCount`),t(n,e.output.status);let s=e.maximumIterations??b;r(n,Number.isInteger(s)&&s>0,`maximumIterations must be a positive integer`),this.estimatedCommandNodeCount=4+3*(s+1),i(n,this.estimatedCommandNodeCount)}getCommandNodes(t){let n=this.props,r=n.grid.width*n.grid.height;e(t,this.id,[n.input,...a(n.output)],{"thinning planes":r*4,"iteration state":16});let{id:i,props:c}=this,{grid:u,output:h}=c,g=u.width*u.height;l(i,t,[c.input,...a(h)]);let _=o(t,`${i}-first`,`uint32`,g),v=o(t,`${i}-second`,`uint32`,g),y=`GPURasterSkeleton`,x=(c.maximumIterations??b)+1,S=f(t,`${i}-rounds`,y,g,i),C=`const WIDTH: u32 = ${u.width}u;
const HEIGHT: u32 = ${u.height}u;

// Value of the neighbor on side 0..7 (clockwise from north); outside the raster is background.
fn getNeighborValue(index: u32, side: u32) -> u32 {
  let columns = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let rows = array<i32, 8>(-1, -1, 0, 1, 1, 1, 0, -1);
  let column = i32(index % WIDTH) + columns[side];
  let row = i32(index / WIDTH) + rows[side];
  if (column < 0 || row < 0 || column >= i32(WIDTH) || row >= i32(HEIGHT)) { return 0u; }
  return previous[previousOffset + u32(row) * WIDTH + u32(column)];
}

// Zhang-Suen deletion test for sub-iteration phase 0 or 1.
fn shouldRemove(index: u32, phase: u32) -> bool {
  var neighborhood: array<u32, 8>;
  var neighbors = 0u;
  var transitions = 0u;
  for (var side = 0u; side < 8u; side++) {
    neighborhood[side] = getNeighborValue(index, side);
    neighbors += neighborhood[side];
  }
  for (var side = 0u; side < 8u; side++) {
    transitions += u32(neighborhood[side] == 0u && neighborhood[(side + 1u) % 8u] != 0u);
  }
  if (neighbors < 2u || neighbors > 6u || transitions != 1u) { return false; }
  if (phase == 0u) {
    return neighborhood[0] * neighborhood[2] * neighborhood[4] == 0u
      && neighborhood[2] * neighborhood[4] * neighborhood[6] == 0u;
  }
  return neighborhood[0] * neighborhood[2] * neighborhood[6] == 0u
    && neighborhood[0] * neighborhood[4] * neighborhood[6] == 0u;
}`,w=[s(t,{id:`${i}-initialize`,operation:y,invocationCount:g,bindings:[{name:`input`,view:c.input,type:`u32`,access:`read`},{name:`first`,view:_,type:`u32`,access:`read_write`}],body:`first[firstOffset+index]=u32(input[inputOffset+index]!=0u);`}),m(t,{id:`${i}-reset`,operation:y,state:S})];for(let e=0;e<x;e++){for(let n=0;n<2;n++){let r=`${i}-${e}-${n}`,{condition:a,extraResources:o}=p(S,r);w.push(s(t,{id:r,operation:y,variant:`thinning-phase-${n}`,invocationCount:g,bindings:[{name:`previous`,view:n===0?_:v,type:`u32`,access:`read`},{name:`next`,view:n===0?v:_,type:`u32`,access:`read_write`},{name:`status`,view:S.status,type:`atomic<u32>`,access:`read_write`}],declarations:C,body:`let original = previous[previousOffset + index];
  let remove = original != 0u && shouldRemove(index, ${n}u);
  next[nextOffset + index] = select(original, 0u, remove);
  if (remove) { atomicOr(&status[statusOffset], 1u); }`,condition:a,extraResources:o}))}w.push(d(t,{id:`${i}-gate-${e}`,operation:y,state:S,maxIterations:x}))}let T=[{name:`status`,view:S.status,type:`u32`,access:`read`},{name:`converged`,view:h.status.converged,type:`u32`,access:`read_write`},...h.status.iterationLimitReached?[{name:`iterationLimitReached`,view:h.status.iterationLimitReached,type:`u32`,access:`read_write`}]:[],...h.status.iterationCount?[{name:`iterationCount`,view:h.status.iterationCount,type:`u32`,access:`read_write`}]:[]];return w.push(s(t,{id:`${i}-status`,operation:y,variant:`iteration-status`,invocationCount:1,bindings:T,body:`converged[convergedOffset] = status[statusOffset + 3u];
  ${h.status.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = 1u - status[statusOffset + 3u];`:``}
  ${h.status.iterationCount?`iterationCount[iterationCountOffset] = status[statusOffset + 2u];`:``}`}),s(t,{id:`${i}-publish`,operation:y,invocationCount:g,bindings:[{name:`first`,view:_,type:`u32`,access:`read`},{name:`skeleton`,view:h.skeleton,type:`u32`,access:`read_write`}],body:`skeleton[skeletonOffset+index]=first[firstOffset+index];`})),w}},S=class{id;props;constructor(e){this.id=e.id??`raster-medial-axis`,this.props=e;let t=c(this.id,e.grid,[e.input],a(e.output));n(this.id,`grid.width`,e.grid.width,g),n(this.id,`grid.height`,e.grid.height,g),u(e.input,t,this.id,`input`),u(e.output.medialAxis,t,this.id,`output.medialAxis`),u(e.output.radius,t,this.id,`output.radius`)}getCommandNodes(e){let{id:t,props:n}=this;return l(t,e,[n.input,...a(n.output)]),y(e,t,n)}};export{x as n,S as t};