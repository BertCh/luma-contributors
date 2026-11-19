import{r as e,t}from"./terrain-binding-preflight-laSEObQE.js";import{i as n,r,s as i}from"./kernel-C3tfQOn4.js";import{n as a,r as o,t as s}from"./raster-grid-utils-D2opjNmg.js";import{a as c,c as l,d as u,f as d,i as f,l as p,m,p as h,r as g,s as _,u as v}from"./raster-relaxation-Di9wjtxp.js";import{E as y,x as b,y as x}from"./gpu-contributor-utils-BIwLQON5.js";import{c as S,d as C,f as w,h as T,n as E,o as D,r as O,u as k}from"./raster-band-utils-BcTxc8_Y.js";import{u as A}from"./raster-grid-VB0AM_GY.js";import{a as j,o as M,t as N}from"./wgsl-kernel-nodes-DagJklH2.js";import{a as P,i as ee,n as F,o as te,r as ne,t as I}from"./terrain-flow-spill-utils-BvdNovhC.js";import{u as L}from"./wgsl-common-DfyXiYB0.js";function re(e){let t=Math.max(2*e,2),n=32-Math.clz32(t-1);return Math.ceil((n+1)/4)*4}function R(e,t){return`
const TOUR_WIDTH: u32 = ${e}u;
const TOUR_HEIGHT: u32 = ${t}u;
const TOUR_NONE: u32 = 0xffffffffu;

fn getTourNeighbor(cell: u32, direction: u32) -> u32 {
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let column = i32(cell % TOUR_WIDTH) + columnOffsets[direction];
  let row = i32(cell / TOUR_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(TOUR_WIDTH) || row >= i32(TOUR_HEIGHT)) {
    return TOUR_NONE;
  }
  return u32(row) * TOUR_WIDTH + u32(column);
}

fn getTourFirstChild(cell: u32) -> u32 {
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getTourNeighbor(cell, direction);
    if (neighbor != TOUR_NONE && getTourParent(neighbor) == cell) { return neighbor; }
  }
  return TOUR_NONE;
}

// The child of 'parent' that follows 'cell' in the direction order of 'parent'.
fn getTourNextSibling(cell: u32, parent: u32) -> u32 {
  var passed = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getTourNeighbor(parent, direction);
    if (neighbor == TOUR_NONE) { continue; }
    if (neighbor == cell) { passed = true; continue; }
    if (passed && getTourParent(neighbor) == parent) { return neighbor; }
  }
  return TOUR_NONE;
}`}var z=`
fn isTourEdgeResolved(edge: u32) -> bool {
  let terminal = tourSucc[tourSuccOffset + edge];
  // A terminal is the exit edge (odd) of a cell without a parent. A cycle edge that jumped onto
  // itself has a parent, so it is rejected.
  return (terminal & 1u) == 1u && getTourParent(terminal >> 1u) == TOUR_NONE;
}`;function B(e,t){let n=t.width*t.height,r=2*n,i=n=>s(e,t.operation,t.contributorId,`${t.id}-tour-${n}`,`uint32`,r),a=i(`successor`),o=i(`distance`),c=i(`successor-scratch`),u=i(`distance-scratch`),d=l(e,`${t.id}-tour`,t.operation,r,t.contributorId);return{nodes:[M(e,{id:`${t.id}-tour-link`,operation:t.operation,variant:`tour-link`,bindings:[...t.parentBindings,{name:`tourSucc`,view:a,type:`u32`,access:`read_write`},{name:`tourDist`,view:o,type:`u32`,access:`read_write`}],invocationCount:n,declarations:`${R(t.width,t.height)}\n${t.declarations}`,body:`let parent = getTourParent(index);
  let firstChild = getTourFirstChild(index);
  let exitEdge = 2u * index + 1u;
  var enterSuccessor = exitEdge;
  if (firstChild != TOUR_NONE) { enterSuccessor = 2u * firstChild; }
  // A root exits to itself: its exit edge is the terminal, at distance 0.
  var exitSuccessor = exitEdge;
  var exitDistance = 0u;
  if (parent != TOUR_NONE) {
    let sibling = getTourNextSibling(index, parent);
    exitSuccessor = select(2u * parent + 1u, 2u * sibling, sibling != TOUR_NONE);
    exitDistance = 1u;
  }
  tourSucc[tourSuccOffset + 2u * index] = enterSuccessor;
  tourSucc[tourSuccOffset + exitEdge] = exitSuccessor;
  tourDist[tourDistOffset + 2u * index] = 1u;
  tourDist[tourDistOffset + exitEdge] = exitDistance;`}),_(e,{id:`${t.id}-tour-reset`,operation:t.operation,state:d}),...g(e,{id:`${t.id}-tour-round`,gateId:`${t.id}-tour-gate`,operation:t.operation,state:d,maxIterations:re(n),groupSize:4,createRound:({nodeId:n,iteration:i,slotBinding:s,markChangedWGSL:l,condition:f,extraResources:p})=>{let[m,h,g,_]=i%2==0?[a,o,c,u]:[c,u,a,o];return M(e,{id:n,operation:t.operation,variant:`tour-round`,bindings:[{name:`succIn`,view:m,type:`u32`,access:`read`},{name:`distIn`,view:h,type:`u32`,access:`read`},{name:`succOut`,view:g,type:`u32`,access:`read_write`},{name:`distOut`,view:_,type:`u32`,access:`read_write`},{name:`status`,view:d.status,type:`atomic<u32>`,access:`read_write`},s],invocationCount:r,body:`let next = succIn[succInOffset + index];
  let jumped = succIn[succInOffset + next];
  var rankedDistance = distIn[distInOffset + index];
  if (next != jumped) {
    rankedDistance = rankedDistance + distIn[distInOffset + next];
    ${l}
  }
  succOut[succOutOffset + index] = jumped;
  distOut[distOutOffset + index] = rankedDistance;`,condition:f,extraResources:p})}})],successor:a,distance:o,state:d,scratch:[c,u]}}var V=`GPUTerrainFlow`,H=1024;function ie(e,t){let n={id:`${t.id}-fill`,operation:V,width:t.width,height:t.height,cellSizeMode:t.cellSizeMode,maxIterations:t.maxIterations},{relaxation:r,resetNodes:i}=p(e,n,!1),o=M(e,{id:`${t.id}-fill-init`,operation:V,variant:`fill-init`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`filled`,view:t.filled,type:`u32`,access:`read_write`},{name:`tileStamps`,view:r.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.width*t.height,declarations:`${a(t)}\n${h(t.width)}`,body:`_ = settings[settingsOffset];
  let elevationValue = elevation[elevationOffset + index];
  if (!isFiniteValue(elevationValue)) {
    filled[filledOffset + index] = bitcast<u32>(getQuietNaN());
    return;
  }
  var boundary = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || !isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      break;
    }
  }
  filled[filledOffset + index] = bitcast<u32>(select(getInfinity(), elevationValue, boundary));
  if (boundary) {
    atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(index)], 1u);
  }`}),s=v(e,{...n,relaxation:r,sweepAfterIteration:96,values:t.filled,auxiliary:t.elevation,settings:t.settings,declarations:`
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  var epsilon = settings[settingsOffset + 4u];
  if (!(epsilon >= 0.0) || !isFiniteValue(epsilon)) { epsilon = 0.0; }
  return max(centerAuxiliary, neighborValue + epsilon);
}`}),c=[...i,o,...s];return(t.converged||t.iterationCount)&&c.push(f(e,{id:`${t.id}-fill-finalize`,operation:V,state:r.state,converged:t.converged,iterationCount:t.iterationCount})),{nodes:c,status:r.state.status}}function ae(e,t){let n=[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];for(let[e,r]of[[`directions`,t.directions],[`classes`,t.classes],[`receivers`,t.receivers]])r&&n.push({name:e,view:r,type:`u32`,access:`read_write`});let r=(e,n,r)=>[t.directions?`directions[directionsOffset + index] = ${e};`:``,t.classes?`classes[classesOffset + index] = ${n};`:``,t.receivers?`receivers[receiversOffset + index] = ${r};`:``].join(`
  `),{cellClass:i}=t;return M(e,{id:t.id,operation:V,variant:`flow-direction`,bindings:n,invocationCount:t.width*t.height,declarations:a(t),body:`let row = index / GRID_WIDTH;
  if (!isFiniteValue(elevation[elevationOffset + index])) {
    ${r(`GRID_NONE`,`${i.invalid}u`,`GRID_NONE`)}
    return;
  }
  let centerSurface = surface[surfaceOffset + index];
  var bestDrop = 0.0;
  var bestDirection = 8u;
  var bestNeighbor = GRID_NONE;
  var boundary = false;
  var hasEqualNeighbor = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || !isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      continue;
    }
    let neighborSurface = surface[surfaceOffset + neighbor];
    if (neighborSurface == centerSurface) { hasEqualNeighbor = true; }
    let drop = (centerSurface - neighborSurface) / getD8Distance(direction, row);
    // Strictly greater: ties keep the lowest direction index.
    if (drop > bestDrop) {
      bestDrop = drop;
      bestDirection = direction;
      bestNeighbor = neighbor;
    }
  }
  if (bestDirection < 8u) {
    ${r(`getD8Code(bestDirection)`,`${i.draining}u`,`bestNeighbor`)}
    return;
  }
  var terminalClass = ${i.pit}u;
  if (boundary) { terminalClass = ${i.outlet}u; }
  else if (hasEqualNeighbor) { terminalClass = ${i.flat}u; }
  ${r(`0u`,`terminalClass`,`GRID_NONE`)}`})}function oe(e,t){return!t.runoff&&!t.area&&8*t.width*t.height<=e.device.limits.maxStorageBufferBindingSize}function se(e,t){let n=t.width*t.height,r=B(e,{id:`${t.id}-accumulate`,operation:V,contributorId:t.id,width:t.width,height:t.height,declarations:`fn getTourParent(cell: u32) -> u32 { return receivers[receiversOffset + cell]; }`,parentBindings:[{name:`receivers`,view:t.receivers,type:`u32`,access:`read`}]}),i=M(e,{id:`${t.id}-accumulate-count`,operation:V,variant:`tour-accumulation`,bindings:[{name:`acc`,view:t.accumulation,type:`u32`,access:`read_write`},{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`tourSucc`,view:r.successor,type:`u32`,access:`read`},{name:`tourDist`,view:r.distance,type:`u32`,access:`read`},{name:`status`,view:r.state.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,declarations:`${R(t.width,t.height)}
fn getTourParent(cell: u32) -> u32 { return receivers[receiversOffset + cell]; }
${z}`,body:`let isValid = (bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) < 0x7f800000u;
  if (!isValid) { acc[accOffset + index] = 0xffffffffu; return; }
  if (!isTourEdgeResolved(2u * index)) {
    // The list never reached a root: a receiver cycle. Not converged.
    acc[accOffset + index] = 0xffffffffu;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  // The edges from E(v) to X(v) inclusive are the tour of v's subtree: 2 * size(v) of them.
  let size = (tourDist[tourDistOffset + 2u * index] - tourDist[tourDistOffset + 2u * index + 1u] + 1u) / 2u;
  acc[accOffset + index] = bitcast<u32>(f32(size));`}),a=[...r.nodes,i];return(t.converged||t.iterationCount)&&a.push(f(e,{id:`${t.id}-accumulate-finalize`,operation:V,state:r.state,converged:t.converged,iterationCount:t.iterationCount})),a}function U(e,t){if(oe(e,t))return se(e,t);let n=t.width*t.height,r=l(e,`${t.id}-accumulate`,V,n),i=[N(e,{id:`${t.id}-accumulate-init`,operation:V,view:e.createDataView(t.accumulation.buffer,{format:`uint32`,byteOffset:t.accumulation.byteOffset,length:t.accumulation.length}),type:`u32`,value:`0xffffffffu`}),_(e,{id:`${t.id}-accumulate-reset`,operation:V,state:r})],o=[{name:`acc`,view:t.accumulation,type:`atomic<u32>`,access:`read_write`},{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`}];t.runoff&&o.push({name:`runoff`,view:t.runoff,type:`f32`,access:`read`}),o.push({name:`status`,view:r.status,type:`atomic<u32>`,access:`read_write`});let s=`${a(t)}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${H}u;

fn getWeight(cell: u32) -> f32 {
  var weight: f32 = 1.0;
  ${t.runoff?`let runoffValue = runoff[runoffOffset + cell];
  weight = select(0.0, runoffValue, isFiniteValue(runoffValue) && runoffValue >= 0.0);`:``}
  ${t.area?`let ground = getGroundCellSizeAtRowCoordinate(f32(cell / GRID_WIDTH) + 0.5);
  weight = weight * (ground.x * ground.y);`:``}
  return weight;
}

// Returns true when this thread wrote the final value of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&acc[accOffset + cell]) != SENTINEL) { return false; }
  var sum = getWeight(cell);
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(cell, direction);
    if (neighbor != GRID_NONE && receivers[receiversOffset + neighbor] == cell) {
      let donorBits = atomicLoad(&acc[accOffset + neighbor]);
      if (donorBits == SENTINEL) { return false; }
      sum = sum + bitcast<f32>(donorBits);
    }
  }
  return atomicCompareExchangeWeak(&acc[accOffset + cell], SENTINEL, bitcast<u32>(sum)).exchanged;
}`,c=e=>`_ = settings[settingsOffset];
  if (!isFiniteValue(elevation[elevationOffset + index])) { return; }
  if (atomicLoad(&acc[accOffset + index]) != SENTINEL) { return; }
  if (tryFinalize(index)) {
    var walk = receivers[receiversOffset + index];
    for (var step = 0u; step < MAX_WALK_LENGTH; step++) {
      if (walk == GRID_NONE || !tryFinalize(walk)) { break; }
      walk = receivers[receiversOffset + walk];
    }
  }
  if (atomicLoad(&acc[accOffset + index]) == SENTINEL) {
    ${e}
  }`;return i.push(...g(e,{id:`${t.id}-accumulate-round`,gateId:`${t.id}-accumulate-gate`,operation:V,state:r,maxIterations:t.maxIterations,createRound:({nodeId:t,slotBinding:r,markChangedWGSL:i,condition:a,extraResources:l})=>M(e,{id:t,operation:V,variant:`accumulate-round`,bindings:[...o,r],invocationCount:n,declarations:s,body:c(i),condition:a,extraResources:l})})),(t.converged||t.iterationCount)&&i.push(f(e,{id:`${t.id}-accumulate-finalize`,operation:V,state:r,converged:t.converged,iterationCount:t.iterationCount})),i}function ce(e,t){return M(e,{id:t.id,operation:V,variant:`streams`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`accumulation`,view:t.accumulation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`streams`,view:t.streams,type:`u32`,access:`read_write`}],invocationCount:t.cellCount,body:`let value = accumulation[accumulationOffset + index];
  let isResolved = (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u;
  let isStream = isResolved && value >= settings[settingsOffset + 5u];
  var stream = select(0u, 1u, isStream);
  // A valid cell without a value was cut off by the accumulation round limit. It is usually the
  // downstream trunk, so 0 ("not a stream") would erase the main river.
  let isValid = (bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) < 0x7f800000u;
  if (isValid && !isResolved) { stream = ${t.unresolvedValue}u; }
  streams[streamsOffset + index] = stream;`})}function le(e,t){let{cellCount:n}=t,r=n+1,i=2*n,a=(n,r)=>s(e,t.operation,t.contributorId,`${t.id}-spill-tree-${n}`,`uint32`,r),o=a(`list-next`,i),c=a(`successor`,i),l=a(`distance`,i),u=a(`successor-scratch`,i),d=a(`distance-scratch`,i),f=a(`parent-scratch`,r),p=a(`parent-key-scratch`,r),m=(n,r,i,a,o)=>M(e,{id:`${t.id}-spill-tree-${n}`,operation:t.operation,variant:`spill-tree-${n}`,bindings:i,invocationCount:r,workgroupSize:256,declarations:`const TREE_NONE: u32 = 0xffffffffu;`,body:a,...o}),h=[{name:`mstFirst`,view:t.mstFirst,type:`u32`,access:`read`},{name:`mstSecond`,view:t.mstSecond,type:`u32`,access:`read`}],_={name:`status`,view:t.mstStatus,type:`u32`,access:`read`},v=m(`lists`,i,[...h,{name:`head`,view:t.head,type:`atomic<u32>`,access:`read_write`},{name:`listNext`,view:o,type:`u32`,access:`read_write`},_],`if (index >= 2u * status[statusOffset + 4u]) { return; }
  let edge = index >> 1u;
  let origin = select(mstFirst[mstFirstOffset + edge], mstSecond[mstSecondOffset + edge], (index & 1u) == 1u);
  // Push this half-edge on the list of its origin; the previous head becomes its successor.
  listNext[listNextOffset + index] = atomicExchange(&head[headOffset + origin], index);`,t.getHalfEdgeGate(`${t.id}-spill-tree-lists`)),y=m(`link`,i,[...h,{name:`head`,view:t.head,type:`atomic<u32>`,access:`read_write`},{name:`listNext`,view:o,type:`u32`,access:`read`},{name:`tourSucc`,view:c,type:`u32`,access:`read_write`},{name:`tourDist`,view:l,type:`u32`,access:`read_write`},_],`if (index >= 2u * status[statusOffset + 4u]) {
    // Unused slots are their own terminals, so the ranking rounds never read out of range.
    tourSucc[tourSuccOffset + index] = index;
    tourDist[tourDistOffset + index] = 0u;
    return;
  }
  let edge = index >> 1u;
  // The half-edge arrives at the other endpoint than its origin.
  let arrival = select(mstSecond[mstSecondOffset + edge], mstFirst[mstFirstOffset + edge], (index & 1u) == 1u);
  var following = listNext[listNextOffset + (index ^ 1u)];
  if (following == TREE_NONE) { following = atomicLoad(&head[headOffset + arrival]); }
  let rootStart = atomicLoad(&head[headOffset + ${n}u]);
  // The tour is cut where it would return to the first half-edge of the ocean.
  if (following == rootStart) {
    tourSucc[tourSuccOffset + index] = index;
    tourDist[tourDistOffset + index] = 0u;
  } else {
    tourSucc[tourSuccOffset + index] = following;
    tourDist[tourDistOffset + index] = 1u;
  }`,t.getHalfEdgeGate(`${t.id}-spill-tree-link`)),b=I(e,{id:`${t.id}-spill-tree-rank`,operation:t.operation,contributorId:t.contributorId,shapes:[[i,256]]}),x=F(e,{id:`${t.id}-spill-tree-rank-reset`,operation:t.operation,state:b,enabledBindings:t.enabledBindings,enabledExpression:t.enabledExpression}),S=g(e,{id:`${t.id}-spill-tree-rank-round`,gateId:`${t.id}-spill-tree-rank-gate`,operation:t.operation,state:b,maxIterations:te(i),createRound:({nodeId:n,iteration:r,slotBinding:a,markChangedWGSL:o,condition:s,extraResources:f})=>{let[p,m,h,g]=r%2==0?[c,l,u,d]:[u,d,c,l];return M(e,{id:n,operation:t.operation,variant:`spill-tree-rank`,bindings:[{name:`succIn`,view:p,type:`u32`,access:`read`},{name:`distIn`,view:m,type:`u32`,access:`read`},{name:`succOut`,view:h,type:`u32`,access:`read_write`},{name:`distOut`,view:g,type:`u32`,access:`read_write`},{name:`status`,view:b.status,type:`atomic<u32>`,access:`read_write`},a],invocationCount:i,workgroupSize:256,body:`let next = succIn[succInOffset + index];
  let jumped = succIn[succInOffset + next];
  var rankedDistance = distIn[distInOffset + index];
  if (next != jumped) {
    rankedDistance = rankedDistance + distIn[distInOffset + next];
    ${o}
  }
  succOut[succOutOffset + index] = jumped;
  distOut[distOutOffset + index] = rankedDistance;`,condition:s,extraResources:f})}}),C=m(`root`,n,[...h,{name:`mstKey`,view:t.mstKey,type:`u32`,access:`read`},{name:`tourDist`,view:l,type:`u32`,access:`read`},{name:`parent`,view:t.parent,type:`u32`,access:`read_write`},{name:`parentKey`,view:t.parentKey,type:`u32`,access:`read_write`},_],`if (index >= status[statusOffset + 4u]) { return; }
  // The half-edge that comes first in the tour leads away from the root (larger distance).
  let forward = tourDist[tourDistOffset + 2u * index] > tourDist[tourDistOffset + 2u * index + 1u];
  let child = select(mstFirst[mstFirstOffset + index], mstSecond[mstSecondOffset + index], forward);
  let owner = select(mstSecond[mstSecondOffset + index], mstFirst[mstFirstOffset + index], forward);
  parent[parentOffset + child] = owner;
  parentKey[parentKeyOffset + child] = mstKey[mstKeyOffset + index];`,t.getEdgeGate(`${t.id}-spill-tree-root`)),w=I(e,{id:`${t.id}-spill-tree-path`,operation:t.operation,contributorId:t.contributorId,shapes:[[r,256]]}),T=F(e,{id:`${t.id}-spill-tree-path-reset`,operation:t.operation,state:w,enabledBindings:t.enabledBindings,enabledExpression:t.enabledExpression}),E=g(e,{id:`${t.id}-spill-tree-path-round`,gateId:`${t.id}-spill-tree-path-gate`,operation:t.operation,state:w,maxIterations:te(r),createRound:({nodeId:n,iteration:i,slotBinding:a,markChangedWGSL:o,condition:s,extraResources:c})=>{let[l,u,d,m]=i%2==0?[t.parent,t.parentKey,f,p]:[f,p,t.parent,t.parentKey];return M(e,{id:n,operation:t.operation,variant:`spill-tree-path`,bindings:[{name:`parentIn`,view:l,type:`u32`,access:`read`},{name:`keyIn`,view:u,type:`u32`,access:`read`},{name:`parentOut`,view:d,type:`u32`,access:`read_write`},{name:`keyOut`,view:m,type:`u32`,access:`read_write`},{name:`status`,view:w.status,type:`atomic<u32>`,access:`read_write`},a],invocationCount:r,workgroupSize:256,body:`let ancestor = parentIn[parentInOffset + index];
  let jumped = parentIn[parentInOffset + ancestor];
  // The path from here to 'jumped' is the path to 'ancestor' followed by the path from 'ancestor'.
  keyOut[keyOutOffset + index] = max(keyIn[keyInOffset + index], keyIn[keyInOffset + ancestor]);
  parentOut[parentOutOffset + index] = jumped;
  if (jumped != ancestor) {
    ${o}
  }`,condition:s,extraResources:c})}});return{nodes:[v,y,x,...S,C,T,...E]}}var W=`GPUTerrainFlow`,ue=32,de=12320,fe=0,pe=1,me=2,he=3,ge=4,_e=0,ve=1;function ye(e){return e.device.limits.maxComputeWorkgroupStorageSize>=de?ue:16}function be(e){return ne(Math.max(e,1))+1}function xe(e,t){return 8*(t.width*t.height+1)<=e.device.limits.maxStorageBufferBindingSize}function Se(e,t){if(!xe(e,t))return ie(e,t);let{id:n,width:r,height:i}=t,o=r*i,l=o+1,u=5*o,d=ye(e),f=Math.ceil(r/d),m=f*Math.ceil(i/d),_=m*256,y=(t,r,i)=>s(e,W,n,`${n}-spill-${t}`,r,i),b=y(`mode`,`uint32`,1),x=y(`local-fill`,`float32`,o),S=y(`labels`,`uint32`,o),C=y(`comp`,`uint32`,l),w=y(`best-key`,`uint32`,l),T=y(`best-edge`,`uint32`,l),E=y(`hook-pointer`,`uint32`,l),D=y(`hook-pointer-scratch`,`uint32`,l),O=y(`mst-first`,`uint32`,o),k=y(`mst-second`,`uint32`,o),A=y(`mst-key`,`uint32`,o),N=y(`head`,`uint32`,l),ne=y(`parent`,`uint32`,l),L=y(`parent-key`,`uint32`,l),re=y(`status`,`uint32`,4),R={name:`mode`,view:b,type:`u32`,access:`read`},z=`mode[modeOffset] == 0u`,B=j(e,{id:`${n}-spill-mode-dispatch`,operation:W,shapes:[[_,256],[l,256],[2*o,256],[o,256],[o,256]]}),V=e=>t=>P(B,e,t),H=(t,r,i,a,o,s,c=!0)=>M(e,{id:`${n}-spill-${t}`,operation:W,variant:`spill-${t}`,bindings:i,invocationCount:r,workgroupSize:256,guardIndex:c,declarations:a,body:o,...s}),ae=e=>e.map(e=>`dispatch[dispatchOffset + ${3*e}u] = 0u;`).join(`
    `),oe=M(e,{id:`${n}-spill-select`,operation:W,variant:`spill-select`,bindings:[{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`mode`,view:b,type:`u32`,access:`read_write`},B.binding],invocationCount:1,body:`let epsilon = settings[settingsOffset + 4u];
  // Invalid, negative and non-finite epsilon mean 0, as in the relaxation.
  let useRelaxation = (bitcast<u32>(epsilon) & 0x7fffffffu) < 0x7f800000u && epsilon > 0.0;
  mode[modeOffset] = select(0u, 1u, useRelaxation);
  ${B.armWGSL}
  if (useRelaxation) {
    ${ae([fe,pe,me,he])}
  } else {
    ${ae([ge])}
  }`}),se={id:`${n}-fill`,operation:W,width:r,height:i,cellSizeMode:t.cellSizeMode,maxIterations:t.maxIterations},{relaxation:U,resetNodes:ce}=p(e,se,!1),ue=ce.find(e=>e.id===`${n}-fill-tile-reset`);if(!ue)throw Error(`${n} relaxation tile reset node is missing`);let de=F(e,{id:`${n}-fill-relax-reset`,operation:W,state:U.state,enabledBindings:[R],enabledExpression:`mode[modeOffset] == 1u`}),Se=M(e,{id:`${n}-fill-init`,operation:W,variant:`fill-init`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`filled`,view:t.filled,type:`u32`,access:`read_write`},{name:`tileStamps`,view:U.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:o,workgroupSize:256,declarations:`${a(t)}\n${h(r)}`,body:`_ = settings[settingsOffset];
  let elevationValue = elevation[elevationOffset + index];
  if (!isFiniteValue(elevationValue)) {
    filled[filledOffset + index] = bitcast<u32>(getQuietNaN());
    return;
  }
  var boundary = false;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE) {
      boundary = true;
      break;
    }
    if (!isFiniteValue(elevation[elevationOffset + neighbor])) {
      boundary = true;
      break;
    }
  }
  filled[filledOffset + index] = bitcast<u32>(select(getInfinity(), elevationValue, boundary));
  if (boundary) {
    atomicMax(&tileStamps[tileStampsOffset + getRelaxationTile(index)], 1u);
  }`,...P(B,ge,`${n}-fill-init`)}),G=v(e,{...se,relaxation:U,sweepAfterIteration:96,values:t.filled,auxiliary:t.elevation,settings:t.settings,declarations:`
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  var epsilon = settings[settingsOffset + 4u];
  if (!(epsilon >= 0.0) || !isFiniteValue(epsilon)) { epsilon = 0.0; }
  return max(centerAuxiliary, neighborValue + epsilon);
}`}),K=ee(r,i),q=d*d,J=H(`tile-fill`,_,[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},{name:`localFill`,view:x,type:`u32`,access:`read_write`},{name:`labels`,view:S,type:`u32`,access:`read_write`}],`${K}
const TILE: u32 = ${d}u;
const TILE_CELLS: u32 = ${q}u;
const CELLS_PER_THREAD: u32 = ${q/256}u;
const TILES_X: u32 = ${f}u;
const TILE_COUNT: u32 = ${m}u;
var<workgroup> tileKey: array<atomic<u32>, ${q}>;
var<workgroup> tileLabel: array<atomic<u32>, ${q}>;
var<workgroup> tileCategory: array<u32, ${q}>;
var<workgroup> tileChanged: atomic<u32>;
var<workgroup> tileContinue: u32;`,`if (workgroupIndex >= TILE_COUNT) { return; }
  let tileColumn = workgroupIndex % TILES_X;
  let tileRow = workgroupIndex / TILES_X;
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);

  for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
    let local = localInvocationIndex + slot * 256u;
    let localColumn = local % TILE;
    let localRow = local / TILE;
    let column = tileColumn * TILE + localColumn;
    let row = tileRow * TILE + localRow;
    var category = 0u;
    var key = SPILL_INFINITE_KEY;
    var label = SPILL_NONE;
    if (column < SPILL_WIDTH && row < SPILL_HEIGHT) {
      let cell = row * SPILL_WIDTH + column;
      let value = elevation[elevationOffset + cell];
      if (isFiniteValue(value)) {
        var boundary = false;
        for (var direction = 0u; direction < 8u; direction++) {
          let neighbor = getSpillNeighbor(cell, direction);
          if (neighbor == SPILL_NONE) {
            boundary = true;
            break;
          }
          if (!isFiniteValue(elevation[elevationOffset + neighbor])) {
            boundary = true;
            break;
          }
        }
        let perimeter = localColumn == 0u || localRow == 0u || localColumn == TILE - 1u || localRow == TILE - 1u;
        if (boundary) {
          category = 3u;
          key = getSpillKey(value);
          label = cell | SPILL_BOUNDARY_FLAG;
        } else if (perimeter) {
          category = 2u;
          key = getSpillKey(value);
          label = cell;
        } else {
          category = 1u;
        }
      }
    }
    tileCategory[local] = category;
    atomicStore(&tileKey[local], key);
    atomicStore(&tileLabel[local], label);
  }
  workgroupBarrier();

  // Fill: key = max(own, lowest neighbor key) to the fixpoint. Keys only decrease (atomicMin), so
  // the update order does not matter; the cap T * T + 1 passes is the longest simple path plus one
  // pass that confirms the fixpoint, and each changing pass lowers a key, so it always exits.
  for (var fillPass = 0u; fillPass <= TILE_CELLS; fillPass++) {
    if (localInvocationIndex == 0u) { atomicStore(&tileChanged, 0u); }
    workgroupBarrier();
    for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
      let local = localInvocationIndex + slot * 256u;
      if (tileCategory[local] != 1u) { continue; }
      let column = tileColumn * TILE + local % TILE;
      let row = tileRow * TILE + local / TILE;
      let ownKey = getSpillKey(elevation[elevationOffset + row * SPILL_WIDTH + column]);
      var lowest = SPILL_INFINITE_KEY;
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborLocal = u32(i32(local) + rowOffsets[direction] * i32(TILE) + columnOffsets[direction]);
        lowest = min(lowest, atomicLoad(&tileKey[neighborLocal]));
      }
      let candidate = max(ownKey, lowest);
      if (candidate < atomicLoad(&tileKey[local])) {
        atomicMin(&tileKey[local], candidate);
        atomicStore(&tileChanged, 1u);
      }
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) { tileContinue = atomicLoad(&tileChanged); }
    if (workgroupUniformLoad(&tileContinue) == 0u) { break; }
  }

  // Labels: the smallest outlet index reachable through cells of non-increasing key. A neighbor
  // passes its label on when its key is at most the own key (the chain stays within the own key).
  for (var labelPass = 0u; labelPass <= TILE_CELLS; labelPass++) {
    if (localInvocationIndex == 0u) { atomicStore(&tileChanged, 0u); }
    workgroupBarrier();
    for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
      let local = localInvocationIndex + slot * 256u;
      if (tileCategory[local] != 1u) { continue; }
      let ownKey = atomicLoad(&tileKey[local]);
      var best = SPILL_NONE;
      for (var direction = 0u; direction < 8u; direction++) {
        let neighborLocal = u32(i32(local) + rowOffsets[direction] * i32(TILE) + columnOffsets[direction]);
        let neighborLabel = atomicLoad(&tileLabel[neighborLocal]);
        if (neighborLabel != SPILL_NONE && atomicLoad(&tileKey[neighborLocal]) <= ownKey) {
          best = min(best, neighborLabel & SPILL_ID_MASK);
        }
      }
      if (best < atomicLoad(&tileLabel[local])) {
        atomicMin(&tileLabel[local], best);
        atomicStore(&tileChanged, 1u);
      }
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) { tileContinue = atomicLoad(&tileChanged); }
    if (workgroupUniformLoad(&tileContinue) == 0u) { break; }
  }

  for (var slot = 0u; slot < CELLS_PER_THREAD; slot++) {
    let local = localInvocationIndex + slot * 256u;
    let column = tileColumn * TILE + local % TILE;
    let row = tileRow * TILE + local / TILE;
    if (column >= SPILL_WIDTH || row >= SPILL_HEIGHT) { continue; }
    let cell = row * SPILL_WIDTH + column;
    if (tileCategory[local] == 0u) {
      localFill[localFillOffset + cell] = 0x7fc00000u;
      labels[labelsOffset + cell] = SPILL_NONE;
    } else {
      localFill[localFillOffset + cell] = getSpillValueBits(atomicLoad(&tileKey[local]));
      labels[labelsOffset + cell] = atomicLoad(&tileLabel[local]);
    }
  }`,P(B,fe,`${n}-spill-tile-fill`),!1),Y=`${K}
const SPILL_OCEAN: u32 = ${o}u;
const SPILL_FORWARD_SLOTS: u32 = ${4*o}u;

// The decoded edge lives in module-scope private variables (no struct return value).
var<private> edgeValid: bool;
var<private> edgeFirst: u32;
var<private> edgeSecond: u32;
var<private> edgeKey: u32;

// Edge slots [0, 4 n): cell = slot / 4 and one of the four forward directions (E, SE, S, SW),
// an edge between the labels of two adjacent cells. Slots [4 n, 5 n): the ocean edge of cell
// slot - 4 n, present when that cell is a boundary outlet. The slot is the tie-break of equal keys.
fn loadSpillEdge(slot: u32) {
  edgeValid = false;
  if (slot < SPILL_FORWARD_SLOTS) {
    let cell = slot >> 2u;
    let neighbor = getSpillNeighbor(cell, slot & 3u);
    if (neighbor == SPILL_NONE) { return; }
    let firstWord = labels[labelsOffset + cell];
    let secondWord = labels[labelsOffset + neighbor];
    if (firstWord == SPILL_NONE || secondWord == SPILL_NONE) { return; }
    edgeFirst = firstWord & SPILL_ID_MASK;
    edgeSecond = secondWord & SPILL_ID_MASK;
    if (edgeFirst == edgeSecond) { return; }
    edgeKey = max(
      getSpillKey(bitcast<f32>(localFill[localFillOffset + cell])),
      getSpillKey(bitcast<f32>(localFill[localFillOffset + neighbor]))
    );
    edgeValid = true;
    return;
  }
  let cell = slot - SPILL_FORWARD_SLOTS;
  let word = labels[labelsOffset + cell];
  if (word == SPILL_NONE || (word & SPILL_BOUNDARY_FLAG) == 0u) { return; }
  edgeFirst = cell;
  edgeSecond = SPILL_OCEAN;
  edgeKey = getSpillKey(bitcast<f32>(localFill[localFillOffset + cell]));
  edgeValid = true;
}`,X=[{name:`labels`,view:S,type:`u32`,access:`read`},{name:`localFill`,view:x,type:`u32`,access:`read`}],Z=be(o),Ce=te(l),Q=I(e,{id:`${n}-spill-boruvka`,operation:W,contributorId:n,shapes:[[l,256],[u,256]],statusLength:8}),we=F(e,{id:`${n}-spill-boruvka-reset`,operation:W,state:Q,statusLength:8,enabledBindings:[R],enabledExpression:z}),Te=H(`init`,l,[{name:`comp`,view:C,type:`u32`,access:`read_write`},{name:`head`,view:N,type:`u32`,access:`read_write`},{name:`parent`,view:ne,type:`u32`,access:`read_write`},{name:`parentKey`,view:L,type:`u32`,access:`read_write`}],``,`comp[compOffset + index] = index;
  head[headOffset + index] = 0xffffffffu;
  parent[parentOffset + index] = index;
  parentKey[parentKeyOffset + index] = 0u;`,P(B,pe,`${n}-spill-init`)),$=I(e,{id:`${n}-spill-boruvka-jump`,operation:W,contributorId:n,shapes:[[l,256]]}),Ee=[];for(let t=0;t<Z;t++){let r=`boruvka-${t}`,i=e=>P(Q.gate,_e,`${n}-spill-${r}-${e}`),a=e=>P(Q.gate,ve,`${n}-spill-${r}-${e}`);Ee.push(H(`${r}-clear`,l,[{name:`bestKey`,view:w,type:`u32`,access:`read_write`},{name:`bestEdge`,view:T,type:`u32`,access:`read_write`}],``,`bestKey[bestKeyOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`,i(`clear`)),H(`${r}-pick-key`,u,[...X,{name:`comp`,view:C,type:`u32`,access:`read`},{name:`bestKey`,view:w,type:`atomic<u32>`,access:`read_write`}],Y,`loadSpillEdge(index);
  if (!edgeValid) { return; }
  let firstComponent = comp[compOffset + edgeFirst];
  let secondComponent = comp[compOffset + edgeSecond];
  if (firstComponent == secondComponent) { return; }
  atomicMin(&bestKey[bestKeyOffset + firstComponent], edgeKey);
  atomicMin(&bestKey[bestKeyOffset + secondComponent], edgeKey);`,a(`pick-key`)),H(`${r}-pick-edge`,u,[...X,{name:`comp`,view:C,type:`u32`,access:`read`},{name:`bestKey`,view:w,type:`u32`,access:`read`},{name:`bestEdge`,view:T,type:`atomic<u32>`,access:`read_write`}],Y,`loadSpillEdge(index);
  if (!edgeValid) { return; }
  let firstComponent = comp[compOffset + edgeFirst];
  let secondComponent = comp[compOffset + edgeSecond];
  if (firstComponent == secondComponent) { return; }
  if (edgeKey == bestKey[bestKeyOffset + firstComponent]) {
    atomicMin(&bestEdge[bestEdgeOffset + firstComponent], index);
  }
  if (edgeKey == bestKey[bestKeyOffset + secondComponent]) {
    atomicMin(&bestEdge[bestEdgeOffset + secondComponent], index);
  }`,a(`pick-edge`)),H(`${r}-hook`,l,[...X,{name:`comp`,view:C,type:`u32`,access:`read`},{name:`bestEdge`,view:T,type:`u32`,access:`read`},{name:`hookPointer`,view:E,type:`u32`,access:`read_write`},{name:`status`,view:Q.status,type:`atomic<u32>`,access:`read_write`}],Y,`var hookTarget = index;
  if (comp[compOffset + index] == index && index != SPILL_OCEAN) {
    let slot = bestEdge[bestEdgeOffset + index];
    if (slot != SPILL_NONE) {
      loadSpillEdge(slot);
      let firstComponent = comp[compOffset + edgeFirst];
      let secondComponent = comp[compOffset + edgeSecond];
      let other = select(firstComponent, secondComponent, firstComponent == index);
      let mutual = bestEdge[bestEdgeOffset + other] == slot;
      if (!(mutual && index > other)) {
        hookTarget = other;
        atomicStore(&status[statusOffset + 0u], 1u);
      }
    }
  }
  hookPointer[hookPointerOffset + index] = hookTarget;`,i(`hook`)),H(`${r}-record`,l,[...X,{name:`bestEdge`,view:T,type:`u32`,access:`read`},{name:`hookPointer`,view:E,type:`u32`,access:`read`},{name:`mstFirst`,view:O,type:`u32`,access:`read_write`},{name:`mstSecond`,view:k,type:`u32`,access:`read_write`},{name:`mstKey`,view:A,type:`u32`,access:`read_write`},{name:`status`,view:Q.status,type:`atomic<u32>`,access:`read_write`}],Y,`if (hookPointer[hookPointerOffset + index] == index) { return; }
  loadSpillEdge(bestEdge[bestEdgeOffset + index]);
  let position = atomicAdd(&status[statusOffset + 4u], 1u);
  // A spanning tree of at most n + 1 nodes has at most n edges.
  if (position < ${o}u) {
    mstFirst[mstFirstOffset + position] = edgeFirst;
    mstSecond[mstSecondOffset + position] = edgeSecond;
    mstKey[mstKeyOffset + position] = edgeKey;
  }`,i(`record`)),F(e,{id:`${n}-spill-${r}-jump-reset`,operation:W,state:$,enabledBindings:[{name:`outer`,view:Q.status,type:`u32`,access:`read`}],enabledExpression:`outer[outerOffset + 1u] == 1u && outer[outerOffset + 0u] != 0u`}),...g(e,{id:`${n}-spill-${r}-jump`,gateId:`${n}-spill-${r}-jump-gate`,operation:W,state:$,maxIterations:Ce,createRound:({nodeId:t,iteration:n,slotBinding:r,markChangedWGSL:i,condition:a,extraResources:o})=>{let[s,c]=n%2==0?[E,D]:[D,E];return M(e,{id:t,operation:W,variant:`spill-jump`,bindings:[{name:`pointerIn`,view:s,type:`u32`,access:`read`},{name:`pointerOut`,view:c,type:`u32`,access:`read_write`},{name:`status`,view:$.status,type:`atomic<u32>`,access:`read_write`},r],invocationCount:l,workgroupSize:256,body:`let next = pointerIn[pointerInOffset + index];
  let jumped = pointerIn[pointerInOffset + next];
  pointerOut[pointerOutOffset + index] = jumped;
  if (jumped != next) {
    ${i}
  }`,condition:a,extraResources:o})}}),H(`${r}-apply`,l,[{name:`comp`,view:C,type:`u32`,access:`read_write`},{name:`hookPointer`,view:E,type:`u32`,access:`read`}],``,`comp[compOffset + index] = hookPointer[hookPointerOffset + comp[compOffset + index]];`,i(`apply`)),c(e,{id:`${n}-spill-${r}-gate`,operation:W,state:Q,maxIterations:Z}))}let De=le(e,{id:n,operation:W,contributorId:n,cellCount:o,mstFirst:O,mstSecond:k,mstKey:A,head:N,mstStatus:Q.status,parent:ne,parentKey:L,enabledBindings:[R],enabledExpression:z,getHalfEdgeGate:V(me),getEdgeGate:V(he)}),Oe=H(`final`,o,[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},...X,{name:`parent`,view:ne,type:`u32`,access:`read`},{name:`parentKey`,view:L,type:`u32`,access:`read`},{name:`filled`,view:t.filled,type:`u32`,access:`read_write`},{name:`status`,view:Q.status,type:`atomic<u32>`,access:`read_write`}],`${K}
const SPILL_OCEAN: u32 = ${o}u;`,`let value = elevation[elevationOffset + index];
  if (!isFiniteValue(value)) {
    filled[filledOffset + index] = 0x7fc00000u;
    return;
  }
  let word = labels[labelsOffset + index];
  if (word == SPILL_NONE || parent[parentOffset + (word & SPILL_ID_MASK)] != SPILL_OCEAN) {
    filled[filledOffset + index] = 0x7fc00000u;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  let key = max(
    getSpillKey(bitcast<f32>(localFill[localFillOffset + index])),
    parentKey[parentKeyOffset + (word & SPILL_ID_MASK)]
  );
  filled[filledOffset + index] = getSpillValueBits(key);`,P(B,he,`${n}-spill-final`)),ke=[R,{name:`relaxationStatus`,view:U.state.status,type:`u32`,access:`read`},{name:`spillStatus`,view:Q.status,type:`u32`,access:`read`},{name:`combined`,view:re,type:`u32`,access:`read_write`}];t.converged&&ke.push({name:`converged`,view:t.converged,type:`u32`,access:`read_write`}),t.iterationCount&&ke.push({name:`iterationCount`,view:t.iterationCount,type:`u32`,access:`read_write`});let Ae=M(e,{id:`${n}-spill-publish`,operation:W,variant:`spill-publish`,bindings:ke,invocationCount:1,body:`let useRelaxation = mode[modeOffset] == 1u;
  let roundCount = select(spillStatus[spillStatusOffset + 2u], relaxationStatus[relaxationStatusOffset + 2u], useRelaxation);
  let isConverged = select(spillStatus[spillStatusOffset + 3u], relaxationStatus[relaxationStatusOffset + 3u], useRelaxation);
  combined[combinedOffset + 0u] = 0u;
  combined[combinedOffset + 1u] = 0u;
  combined[combinedOffset + 2u] = roundCount;
  combined[combinedOffset + 3u] = isConverged;
  ${t.converged?`converged[convergedOffset] = isConverged;`:``}
  ${t.iterationCount?`iterationCount[iterationCountOffset] = roundCount;`:``}`});return{nodes:[oe,de,ue,Se,...G,we,J,Te,...Ee,...De.nodes,Oe,Ae],status:re}}var G=`GPUTerrainFlow`;function K(e,t){let n=[...t.fillStatus?[t.fillStatus]:[],...t.flatStatuses];if(n.length===0)return[];let r=[];for(let[e,n]of[[`filledElevation`,t.filledElevation],[`directions`,t.directions],[`classes`,t.classes],[`accumulation`,t.accumulation],[`streams`,t.streams]])n&&r.push({name:e,view:n,type:`u32`,access:`read_write`});if(r.length===0)return[];let i=l(e,`${t.id}-unresolved`,G,t.cellCount,t.id),{x:a,y:o,z:s}=i.layout,{condition:c,extraResources:u}=d(i,`${t.id}-unresolved-cells`),f=n.map((e,t)=>({name:`status${t}`,view:e,type:`u32`,access:`read`})),p=n.map((e,t)=>`status${t}[status${t}Offset + 3u] != 0u`),m=(e,t,n)=>t?`${e}[${e}Offset + index] = ${n};`:``,h=M(e,{id:`${t.id}-unresolved-arm`,operation:G,variant:`unresolved-arm`,bindings:[...f,{name:`dispatch`,view:i.dispatch,type:`u32`,access:`read_write`}],invocationCount:1,body:`let converged = ${p.join(` && `)};
  dispatch[dispatchOffset] = select(${a}u, 0u, converged);
  dispatch[dispatchOffset + 1u] = ${o}u;
  dispatch[dispatchOffset + 2u] = ${s}u;`}),g=t.fillStatus&&t.filledElevation?[{name:`fillStatus`,view:t.fillStatus,type:`u32`,access:`read`}]:[];return[h,M(e,{id:`${t.id}-unresolved-cells`,operation:G,variant:`unresolved-cells`,bindings:[{name:`elevation`,view:t.elevation,type:`f32`,access:`read`},...g,...r],invocationCount:i.invocationCount,workgroupSize:256,body:`if ((bitcast<u32>(elevation[elevationOffset + index]) & 0x7fffffffu) >= 0x7f800000u) { return; }
  ${m(`directions`,t.directions,`${t.codes.direction}u`)}
  ${m(`classes`,t.classes,`${t.codes.cellClass}u`)}
  ${m(`accumulation`,t.accumulation,`0x7fc00000u`)}
  ${m(`streams`,t.streams,`${t.codes.stream}u`)}
  ${g.length>0?`if (fillStatus[fillStatusOffset + 3u] == 0u) { ${m(`filledElevation`,t.filledElevation,`0x7fc00000u`)} }`:``}`,condition:c,extraResources:u})]}var q=`GPUTerrainFlow`,J=`
${L}
fn getInfinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }`;function Y(e,t){let n=t.width*t.height,{cellClass:r}=t,i=r=>s(e,q,t.id,`${t.id}-flats-${r}`,`float32`,n),o=t.continuation?.lower.values??i(`lower-values`),c=i(`lower-aux`),l=t.continuation?.higher.values??i(`higher-values`),u=i(`higher-aux`),d=t.continuation?.maximum.values??i(`maximum-values`),f=i(`mask`),m=t.continuation?i(`lower-seeds`):o,g=t.continuation?i(`higher-seeds`):l,_=M(e,{id:`${t.id}-flats-prepare`,operation:q,variant:`flats-prepare`,bindings:[{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`classes`,view:t.classes,type:`u32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`lowerValues`,view:m,type:`f32`,access:`read_write`},{name:`lowerAuxiliary`,view:c,type:`f32`,access:`read_write`},{name:`higherValues`,view:g,type:`f32`,access:`read_write`},{name:`higherAuxiliary`,view:u,type:`f32`,access:`read_write`}],invocationCount:n,declarations:a(t),body:`_ = settings[settingsOffset];
  let centerClass = classes[classesOffset + index];
  let centerSurface = surface[surfaceOffset + index];
  let isFlat = centerClass == ${r.flat}u;
  let hasFlow = centerClass == ${r.draining}u || centerClass == ${r.outlet}u;
  var isLowEdge = false;
  var isHighEdge = false;
  if (isFlat || hasFlow) {
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getD8Neighbor(index, direction);
      if (neighbor == GRID_NONE) { continue; }
      let neighborClass = classes[classesOffset + neighbor];
      if (neighborClass == ${r.invalid}u) { continue; }
      let neighborSurface = surface[surfaceOffset + neighbor];
      if (hasFlow && neighborClass == ${r.flat}u && neighborSurface == centerSurface) {
        isLowEdge = true;
      }
      if (isFlat && neighborSurface > centerSurface) { isHighEdge = true; }
    }
  }
  lowerAuxiliary[lowerAuxiliaryOffset + index] = select(getQuietNaN(), centerSurface, isFlat || isLowEdge);
  lowerValues[lowerValuesOffset + index] = select(getInfinity(), 1.0, isLowEdge);
  higherAuxiliary[higherAuxiliaryOffset + index] = select(getQuietNaN(), centerSurface, isFlat);
  higherValues[higherValuesOffset + index] = select(getInfinity(), 1.0, isHighEdge);`}),y=(r,i,a,o)=>{let s={id:`${t.id}-flats-${r}`,operation:q,width:t.width,height:t.height,cellSizeMode:t.cellSizeMode,maxIterations:t.maxIterations,continuation:t.continuation&&{state:t.continuation[r].state,retainConverged:!0,prerequisite:r===`maximum`?t.continuation.higher.state:void 0,restartShapes:[[n]]}},{relaxation:c,resetNodes:l}=p(e,s,!0);return{relaxation:c,resetNodes:l,relaxNodes:v(e,{...s,relaxation:c,sweepAfterIteration:96,values:i,auxiliary:a,settings:t.settings,declarations:`
fn getRelaxationCandidate(neighborValue: f32, neighborAuxiliary: f32, centerAuxiliary: f32, centerRow: u32, direction: u32) -> f32 {
  if (neighborAuxiliary == centerAuxiliary) { return ${o}; }
  return getInfinity();
}`})}},b=y(`lower`,o,c,`neighborValue + 1.0`),x=y(`higher`,l,u,`neighborValue + 1.0`),S=y(`maximum`,d,t.surface,`neighborValue`),C=h(t.width),w=M(e,{id:`${t.id}-flats-seed-edges`,operation:q,variant:`flats-seed-edges`,bindings:[{name:`lowerValues`,view:o,type:`f32`,access:`read`},{name:`higherValues`,view:l,type:`f32`,access:`read`},{name:`lowerStamps`,view:b.relaxation.tileStamps,type:`atomic<u32>`,access:`read_write`},{name:`higherStamps`,view:x.relaxation.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,declarations:`${J}\n${C}`,body:`if (isFiniteValue(lowerValues[lowerValuesOffset + index])) {
    atomicMax(&lowerStamps[lowerStampsOffset + getRelaxationTile(index)], 1u);
  }
  if (isFiniteValue(higherValues[higherValuesOffset + index])) {
    atomicMax(&higherStamps[higherStampsOffset + getRelaxationTile(index)], 1u);
  }`}),T=M(e,{id:`${t.id}-flats-prepare-maximum`,...S.relaxation.continuation?.getRestartGate(0,`${t.id}-flats-prepare-maximum`),operation:q,variant:`flats-prepare-maximum`,bindings:[{name:`higherValues`,view:l,type:`f32`,access:`read`},{name:`maximumValues`,view:d,type:`f32`,access:`read_write`},{name:`maximumStamps`,view:S.relaxation.tileStamps,type:`atomic<u32>`,access:`read_write`}],invocationCount:n,declarations:`${J}\n${C}`,body:`let higher = higherValues[higherValuesOffset + index];
  let reached = isFiniteValue(higher);
  maximumValues[maximumValuesOffset + index] = select(0.0, -higher, reached);
  if (reached) {
    atomicMax(&maximumStamps[maximumStampsOffset + getRelaxationTile(index)], 1u);
  }`}),E=M(e,{id:`${t.id}-flats-mask`,operation:q,variant:`flats-mask`,bindings:[{name:`lowerValues`,view:o,type:`f32`,access:`read`},{name:`higherValues`,view:l,type:`f32`,access:`read`},{name:`maximumValues`,view:d,type:`f32`,access:`read`},{name:`mask`,view:f,type:`f32`,access:`read_write`}],invocationCount:n,declarations:J,body:`let towardLower = lowerValues[lowerValuesOffset + index];
  if (!isFiniteValue(towardLower)) {
    mask[maskOffset + index] = getInfinity();
    return;
  }
  let awayFromHigher = higherValues[higherValuesOffset + index];
  var awayTerm = 0.0;
  if (isFiniteValue(awayFromHigher)) {
    awayTerm = -maximumValues[maximumValuesOffset + index] - awayFromHigher;
  }
  mask[maskOffset + index] = 2.0 * towardLower + awayTerm;`}),D=[{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`mask`,view:f,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`classes`,view:t.classes,type:`u32`,access:`read_write`}];t.directions&&D.push({name:`directions`,view:t.directions,type:`u32`,access:`read_write`}),t.receivers&&D.push({name:`receivers`,view:t.receivers,type:`u32`,access:`read_write`});let O=M(e,{id:`${t.id}-flats-route`,operation:q,variant:`flats-route`,bindings:D,invocationCount:n,declarations:a(t),body:`_ = settings[settingsOffset];
  if (classes[classesOffset + index] != ${r.flat}u) { return; }
  let centerMask = mask[maskOffset + index];
  if (!isFiniteValue(centerMask)) { return; }
  let centerSurface = surface[surfaceOffset + index];
  var bestMask = centerMask;
  var bestDirection = 8u;
  var bestNeighbor = GRID_NONE;
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(index, direction);
    if (neighbor == GRID_NONE || surface[surfaceOffset + neighbor] != centerSurface) { continue; }
    let neighborMask = mask[maskOffset + neighbor];
    // Strictly smaller: ties keep the lowest direction index.
    if (isFiniteValue(neighborMask) && neighborMask < bestMask) {
      bestMask = neighborMask;
      bestDirection = direction;
      bestNeighbor = neighbor;
    }
  }
  if (bestDirection < 8u) {
    classes[classesOffset + index] = ${r.draining}u;
    ${t.directions?`directions[directionsOffset + index] = getD8Code(bestDirection);`:``}
    ${t.receivers?`receivers[receiversOffset + index] = bestNeighbor;`:``}
  }`}),k=[];if(t.continuation){k.push(M(e,{id:`${t.id}-flats-continuation-reset`,operation:q,bindings:[{name:`lower`,view:t.continuation.lower.state,type:`u32`,access:`read_write`},{name:`higher`,view:t.continuation.higher.state,type:`u32`,access:`read_write`},{name:`maximum`,view:t.continuation.maximum.state,type:`u32`,access:`read_write`}],invocationCount:1,body:`let restart = lower[lowerOffset + 5u] != 0u || higher[higherOffset + 5u] != 0u || maximum[maximumOffset + 5u] != 0u ||
        (lower[lowerOffset + 3u] != 0u && higher[higherOffset + 3u] != 0u && maximum[maximumOffset + 3u] != 0u);
        if (restart) {
          lower[lowerOffset + 2u] = 0u; lower[lowerOffset + 3u] = 0u;
          higher[higherOffset + 2u] = 0u; higher[higherOffset + 3u] = 0u;
          maximum[maximumOffset + 2u] = 0u; maximum[maximumOffset + 3u] = 0u;
          lower[lowerOffset + 5u] = 0u; higher[higherOffset + 5u] = 0u; maximum[maximumOffset + 5u] = 0u;
        }`})),k.push(_);for(let[r,i,a,s]of[[b,m,o,`lower`],[x,g,l,`higher`]])k.push(...r.resetNodes,M(e,{id:`${t.id}-flats-${s}-initialize`,operation:q,...r.relaxation.continuation?.getRestartGate(0,`${t.id}-flats-${s}-initialize`),bindings:[{name:`seeds`,view:i,type:`f32`,access:`read`},{name:`values`,view:a,type:`f32`,access:`read_write`}],invocationCount:n,body:`values[valuesOffset + index] = seeds[seedsOffset + index];`}),...r.relaxNodes);k.push(...S.resetNodes,T,...S.relaxNodes,E,O)}else k.push(_,...b.resetNodes,...x.resetNodes,...S.resetNodes,w,...b.relaxNodes,...x.relaxNodes,T,...S.relaxNodes,E,O);return(t.converged||t.iterationCount)&&k.push(M(e,{id:`${t.id}-flats-finalize`,operation:q,variant:`flats-finalize`,bindings:[{name:`lowerStatus`,view:b.relaxation.state.status,type:`u32`,access:`read`},{name:`higherStatus`,view:x.relaxation.state.status,type:`u32`,access:`read`},{name:`maximumStatus`,view:S.relaxation.state.status,type:`u32`,access:`read`},...t.converged?[{name:`converged`,view:t.converged,type:`u32`,access:`read_write`}]:[],...t.iterationCount?[{name:`iterationCount`,view:t.iterationCount,type:`u32`,access:`read_write`}]:[]],invocationCount:1,body:`${t.converged?`converged[convergedOffset] = select(0u, 1u,
    lowerStatus[lowerStatusOffset + 3u] == 1u &&
    higherStatus[higherStatusOffset + 3u] == 1u &&
    maximumStatus[maximumStatusOffset + 3u] == 1u);`:``}
  ${t.iterationCount?`iterationCount[iterationCountOffset] = lowerStatus[lowerStatusOffset + 2u] +
    higherStatus[higherStatusOffset + 2u] + maximumStatus[maximumStatusOffset + 2u];`:``}`})),{nodes:k,statuses:[b,x,S].map(({relaxation:e})=>e.state.status)}}var X=`GPUTerrainFlow`;function Z(e,t){let n=`
// Fraction of 'cell' flow that goes to the neighbor in 'direction' when no lower neighbor exists.
fn getFallbackFraction(cell: u32, direction: u32) -> f32 {
  let receiver = receivers[receiversOffset + cell];
  return select(0.0, 1.0, receiver != GRID_NONE && receiver == getD8Neighbor(cell, direction));
}`;if(e===`d-infinity`)return`${n}
// Tarboton (1997) facets in the documented order: index -> (cardinal, diagonal).
fn getFacetCardinal(facet: u32) -> u32 { return (facet / 2u) * 2u; }
fn getFacetDiagonal(facet: u32) -> u32 {
  let cardinal = getFacetCardinal(facet);
  let lower = select(cardinal - 1u, 1u, cardinal == 0u);
  let upper = select(cardinal + 1u, 7u, cardinal == 0u);
  return select(lower, upper, (facet & 1u) == 1u);
}

${t?`// Writes the cell's D-infinity routing record: [cardinal | diagonal << 4 (8 = no lower facet),
// diagonal fraction bits], or ROUTING_INVALID for an invalid cell.
fn computeRouting(cell: u32) {
  let center = surface[surfaceOffset + cell];
  if (!isFiniteValue(center)) {
    routing[routingOffset + 2u * cell] = ROUTING_INVALID;
    routing[routingOffset + 2u * cell + 1u] = 0u;
    return;
  }
  let row = cell / GRID_WIDTH;
  var bestSlope = 0.0;
  var bestCardinal = 8u;
  var bestDiagonal = 8u;
  var bestDiagonalFraction = 0.0;
  for (var facet = 0u; facet < 8u; facet++) {
    let cardinal = getFacetCardinal(facet);
    let diagonal = getFacetDiagonal(facet);
    let cardinalNeighbor = getD8Neighbor(cell, cardinal);
    let diagonalNeighbor = getD8Neighbor(cell, diagonal);
    var cardinalUsable = false;
    var diagonalUsable = false;
    var e1 = 0.0;
    var e2 = 0.0;
    if (cardinalNeighbor != GRID_NONE) {
      e1 = surface[surfaceOffset + cardinalNeighbor];
      cardinalUsable = isFiniteValue(e1);
    }
    if (diagonalNeighbor != GRID_NONE) {
      e2 = surface[surfaceOffset + diagonalNeighbor];
      diagonalUsable = isFiniteValue(e2);
    }
    if (!cardinalUsable && !diagonalUsable) { continue; }
    let d1 = getD8Distance(cardinal, row);
    let diagonalDistance = getD8Distance(diagonal, row);
    let diagonalRow = f32(row) + f32(getD8RowOffset(diagonal));
    let ground = getGroundCellSizeAtRowCoordinate((f32(row) + diagonalRow + 1.0) * 0.5);
    let d2 = select(ground.x, ground.y, cardinal == 0u || cardinal == 4u);
    var slope = 0.0;
    var fractionDiagonal = 0.0;
    if (cardinalUsable && diagonalUsable) {
      let s1 = (center - e1) / d1;
      let s2 = (e1 - e2) / d2;
      if (s1 > 0.0 && s2 == 0.0) {
        // Exactly cardinal: no sqrt/atan, so equal-height ties compare bit-exactly.
        slope = s1;
      } else if (s1 > 0.0 && s2 > 0.0) {
        let angle = atan2(s2, s1);
        let maximumAngle = atan2(d2, d1);
        if (angle <= maximumAngle) {
          slope = sqrt(s1 * s1 + s2 * s2);
          fractionDiagonal = angle / maximumAngle;
        } else {
          slope = (center - e2) / diagonalDistance;
          fractionDiagonal = 1.0;
        }
      } else if (s1 > 0.0) {
        slope = s1;
      } else if (s2 > 0.0) {
        slope = (center - e2) / diagonalDistance;
        fractionDiagonal = 1.0;
      }
    } else if (cardinalUsable) {
      slope = (center - e1) / d1;
    } else {
      slope = (center - e2) / diagonalDistance;
      fractionDiagonal = 1.0;
    }
    // Strictly greater: ties keep the first facet.
    if (slope > bestSlope) {
      bestSlope = slope;
      bestCardinal = cardinal;
      bestDiagonal = diagonal;
      bestDiagonalFraction = fractionDiagonal;
    }
  }
  routing[routingOffset + 2u * cell] = bestCardinal | (bestDiagonal << 4u);
  routing[routingOffset + 2u * cell + 1u] = bitcast<u32>(bestDiagonalFraction);
}

`:``}
fn getFlowFraction(cell: u32, direction: u32) -> f32 {
  let code = routing[routingOffset + 2u * cell];
  if (code == ROUTING_INVALID) { return 0.0; }
  let bestCardinal = code & 15u;
  if (bestCardinal == 8u) { return getFallbackFraction(cell, direction); }
  let bestDiagonal = code >> 4u;
  let bestDiagonalFraction = bitcast<f32>(routing[routingOffset + 2u * cell + 1u]);
  if (direction == bestDiagonal) { return bestDiagonalFraction; }
  if (direction == bestCardinal) { return 1.0 - bestDiagonalFraction; }
  return 0.0;
}`;let r=e===`mfd-quinn`;return`${n}
fn getFlowExponent() -> f32 {
  let exponent = settings[settingsOffset + 6u];
  return select(${r?`1.0`:`1.1`}, exponent, isFiniteValue(exponent) && exponent > 0.0);
}

// Slope (Freeman) or slope times contour length (Quinn) of one lower neighbor.
fn getMultipleFlowScore(drop: f32, direction: u32, row: u32) -> f32 {
  let tangent = drop / getD8Distance(direction, row);
  ${r?`let ground = getGroundCellSizeAtRowCoordinate(f32(row) + 0.5);
  var contourLength = 0.25 * getD8Distance(direction, row);
  if (direction == 0u || direction == 4u) { contourLength = 0.5 * ground.y; }
  if (direction == 2u || direction == 6u) { contourLength = 0.5 * ground.x; }
  return tangent * contourLength;`:`return tangent;`}
}

${t?`// Writes the cell's MFD routing record: [largest score bits (-1 = no lower neighbor), weight total
// bits], or ROUTING_INVALID for an invalid cell.
fn computeRouting(cell: u32) {
  let center = surface[surfaceOffset + cell];
  if (!isFiniteValue(center)) {
    routing[routingOffset + 2u * cell] = ROUTING_INVALID;
    routing[routingOffset + 2u * cell + 1u] = 0u;
    return;
  }
  let row = cell / GRID_WIDTH;
  var hasLower = false;
  var maximumScore = 0.0;
  for (var other = 0u; other < 8u; other++) {
    let neighbor = getD8Neighbor(cell, other);
    if (neighbor == GRID_NONE) { continue; }
    let drop = center - surface[surfaceOffset + neighbor];
    if (isFiniteValue(drop) && drop > 0.0) {
      hasLower = true;
      maximumScore = max(maximumScore, getMultipleFlowScore(drop, other, row));
    }
  }
  if (!hasLower) {
    routing[routingOffset + 2u * cell] = bitcast<u32>(-1.0);
    routing[routingOffset + 2u * cell + 1u] = 0u;
    return;
  }
  var total = 0.0;
  for (var other = 0u; other < 8u; other++) {
    total = total + getMultipleFlowWeight(cell, center, other, maximumScore);
  }
  routing[routingOffset + 2u * cell] = bitcast<u32>(maximumScore);
  routing[routingOffset + 2u * cell + 1u] = bitcast<u32>(total);
}

`:``}
// Unnormalized weight of the move from 'cell' toward 'direction'; 0 unless that neighbor is lower.
fn getMultipleFlowWeight(cell: u32, center: f32, direction: u32, maximumScore: f32) -> f32 {
  let neighbor = getD8Neighbor(cell, direction);
  if (neighbor == GRID_NONE) { return 0.0; }
  let drop = center - surface[surfaceOffset + neighbor];
  if (!(isFiniteValue(drop) && drop > 0.0)) { return 0.0; }
  let score = getMultipleFlowScore(drop, direction, cell / GRID_WIDTH);
  let ratio = select(1.0, score / maximumScore, maximumScore > 0.0);
  return select(0.0, pow(ratio, getFlowExponent()), ratio > 0.0);
}

fn getFlowFraction(cell: u32, direction: u32) -> f32 {
  let code = routing[routingOffset + 2u * cell];
  if (code == ROUTING_INVALID) { return 0.0; }
  let maximumScore = bitcast<f32>(code);
  if (maximumScore < 0.0) { return getFallbackFraction(cell, direction); }
  let total = bitcast<f32>(routing[routingOffset + 2u * cell + 1u]);
  let weight = getMultipleFlowWeight(cell, surface[surfaceOffset + cell], direction, maximumScore);
  return select(0.0, weight / total, total > 0.0);
}`}function Ce(e,t){let n=t.width*t.height,i=l(e,`${t.id}-accumulate`,X,n),o=r(e,`${t.id}-routing-records`,`uint32`,2*n),s=e=>`${a(t)}
const ROUTING_INVALID: u32 = 0xffffffffu;
${Z(t.routing,e)}`,c=[M(e,{id:`${t.id}-routing-prepare`,operation:X,variant:`routing-prepare-${t.routing}`,bindings:[{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`routing`,view:o,type:`u32`,access:`read_write`}],invocationCount:n,declarations:s(!0),body:`_ = receivers[receiversOffset];
  _ = settings[settingsOffset];
  computeRouting(index);`}),N(e,{id:`${t.id}-accumulate-init`,operation:X,view:e.createDataView(t.accumulation.buffer,{format:`uint32`,byteOffset:t.accumulation.byteOffset,length:t.accumulation.length}),type:`u32`,value:`0xffffffffu`}),_(e,{id:`${t.id}-accumulate-reset`,operation:X,state:i})],u=[{name:`acc`,view:t.accumulation,type:`atomic<u32>`,access:`read_write`},{name:`receivers`,view:t.receivers,type:`u32`,access:`read`},{name:`surface`,view:t.surface,type:`f32`,access:`read`},{name:`settings`,view:t.settings,type:`f32`,access:`read`},{name:`routing`,view:o,type:`u32`,access:`read`}];t.runoff&&u.push({name:`runoff`,view:t.runoff,type:`f32`,access:`read`}),u.push({name:`status`,view:i.status,type:`atomic<u32>`,access:`read_write`});let d=`${s(!1)}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${H}u;

fn getWeight(cell: u32) -> f32 {
  var weight: f32 = 1.0;
  ${t.runoff?`let runoffValue = runoff[runoffOffset + cell];
  weight = select(0.0, runoffValue, isFiniteValue(runoffValue) && runoffValue >= 0.0);`:``}
  ${t.area?`let ground = getGroundCellSizeAtRowCoordinate(f32(cell / GRID_WIDTH) + 0.5);
  weight = weight * (ground.x * ground.y);`:``}
  return weight;
}

// Returns true when this thread wrote the final value of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&acc[accOffset + cell]) != SENTINEL) { return false; }
  var sum = getWeight(cell);
  for (var direction = 0u; direction < 8u; direction++) {
    let neighbor = getD8Neighbor(cell, direction);
    if (neighbor == GRID_NONE) { continue; }
    let fraction = getFlowFraction(neighbor, (direction + 4u) & 7u);
    if (fraction > 0.0) {
      let donorBits = atomicLoad(&acc[accOffset + neighbor]);
      if (donorBits == SENTINEL) { return false; }
      sum = sum + fraction * bitcast<f32>(donorBits);
    }
  }
  return atomicCompareExchangeWeak(&acc[accOffset + cell], SENTINEL, bitcast<u32>(sum)).exchanged;
}`,p=e=>`if (!isFiniteValue(surface[surfaceOffset + index])) { return; }
  if (atomicLoad(&acc[accOffset + index]) != SENTINEL) { return; }
  var walk = index;
  var finalized = tryFinalize(index);
  for (var step = 0u; step < MAX_WALK_LENGTH && finalized; step++) {
    // Try every receiver in direction order; continue from the first one this thread finalized.
    var next = GRID_NONE;
    for (var direction = 0u; direction < 8u; direction++) {
      let receiver = getD8Neighbor(walk, direction);
      if (receiver != GRID_NONE && getFlowFraction(walk, direction) > 0.0) {
        if (tryFinalize(receiver) && next == GRID_NONE) { next = receiver; }
      }
    }
    walk = next;
    finalized = next != GRID_NONE;
  }
  if (atomicLoad(&acc[accOffset + index]) == SENTINEL) {
    ${e}
  }`;return c.push(...g(e,{id:`${t.id}-accumulate-round`,gateId:`${t.id}-accumulate-gate`,operation:X,state:i,maxIterations:t.maxIterations,createRound:({nodeId:r,slotBinding:i,markChangedWGSL:a,condition:o,extraResources:s})=>M(e,{id:r,operation:X,variant:`accumulate-round-${t.routing}`,bindings:[...u,i],invocationCount:n,declarations:d,body:p(a),condition:o,extraResources:s})})),(t.converged||t.iterationCount)&&c.push(f(e,{id:`${t.id}-accumulate-finalize`,operation:X,state:i,converged:t.converged,iterationCount:t.iterationCount})),c}var Q=4294967295,we=4294967294,Te=4294967295,$={draining:0,flat:1,pit:2,outlet:3,invalid:4,unresolved:5};function Ee(e,t,n=new Float32Array(8)){return D(`getGPUTerrainFlowParameterValues`,n,8),o(n,t),n[4]=e.fillEpsilon??0,n[5]=e.streamThreshold??1/0,n[6]=e.flowExponent??0,n[7]=0,n}function De(e,t){let{output:r}=t,a=[[`filledElevation`,r.filledElevation],[`flowDirections`,r.flowDirections],[`cellClasses`,r.cellClasses],[`accumulation`,r.accumulation],[`streams`,r.streams],[`status.fillConverged`,r.status?.fillConverged],[`status.accumulationConverged`,r.status?.accumulationConverged],[`status.flatsConverged`,r.status?.flatsConverged],[`status.fillIterationCount`,r.status?.fillIterationCount],[`status.flatsIterationCount`,r.status?.flatsIterationCount],[`status.accumulationIterationCount`,r.status?.accumulationIterationCount]];x(e,a.map(([,e])=>e),[...E(t.elevation),t.parameters,t.runoff]);let o=[[`filledElevation`,`flowDirections`,`cellClasses`],[`accumulation`,`streams`]];for(let[t,[r,s]]of a.entries())for(let[c,l]of a.slice(t+1)){if(!s||!l||s.buffer!==l.buffer)continue;if(n(s,l))throw Error(`${e} outputs must not share buffers with overlapping byte ranges (${r} and ${c})`);let t=i(s),a=i(l);if(o.some(e=>e.includes(r)&&e.includes(c))&&t.offset<a.offset+a.size&&a.offset<t.offset+t.size)throw Error(`${e} ${r} and ${c} are written by one node and must be at least 256 bytes apart when they share a buffer`)}}var Oe=class{id;props;constructor(e){this.id=e.id??`terrain-flow`,this.props=e;let{id:t}=this,{output:n}=e,r=n.status,i=k(t,e.grid);if(T(t,e.parameters,8),A(t,e.cellSizeMode),C(t,e.grid,e.cellSizeMode),w(t,e.grid),![`cells`,`area`].includes(e.accumulationUnits??`cells`))throw Error(`${t} accumulationUnits must be cells or area`);if(m(t,`maximumFillIterations`,e.maximumFillIterations??512),m(t,`maximumFlatIterations`,e.maximumFlatIterations??512),e.flatContinuation){if(!e.resolveFlats)throw Error(`${t} flatContinuation requires resolveFlats`);let r=Object.values(e.flatContinuation).flatMap(e=>[e.state,e.values]);for(let n of Object.values(e.flatContinuation))if(y(n.state,[`uint32`],`${t} flatContinuation.state`),y(n.values,[`float32`],`${t} flatContinuation.values`),n.state.length!==u(e.grid)||n.values.length!==i)throw Error(`${t} flatContinuation has the wrong length`);x(t,r,[...E(e.elevation),e.parameters,e.runoff,n.filledElevation,n.flowDirections,n.cellClasses,n.accumulation,n.streams,...Object.values(n.status??{})]);for(let e=0;e<r.length;e++)for(let n=e+1;n<r.length;n++)if(r[e].buffer===r[n].buffer)throw Error(`${t} flatContinuation buffers must be distinct`)}if(![`d8`,`d-infinity`,`mfd-freeman`,`mfd-quinn`].includes(e.flowRouting??`d8`))throw Error(`${t} flowRouting must be d8, d-infinity, mfd-freeman, or mfd-quinn`);if((r?.flatsConverged||r?.flatsIterationCount)&&!e.resolveFlats)throw Error(`${t} output.status.flatsConverged and flatsIterationCount require resolveFlats`);if(r?.fillIterationCount&&!e.fillDepressions)throw Error(`${t} output.status.fillIterationCount requires fillDepressions`);if(r?.accumulationIterationCount&&!n.accumulation&&!n.streams)throw Error(`${t} output.status.accumulationIterationCount requires output.accumulation or output.streams`);if(m(t,`maximumAccumulationIterations`,e.maximumAccumulationIterations??64),!n.filledElevation&&!n.flowDirections&&!n.cellClasses&&!n.accumulation&&!n.streams)throw Error(`${t} requires at least one output`);if((n.filledElevation||r?.fillConverged)&&!e.fillDepressions)throw Error(`${t} output.filledElevation and output.status.fillConverged require fillDepressions`);if(r?.accumulationConverged&&!n.accumulation&&!n.streams)throw Error(`${t} output.status.accumulationConverged requires output.accumulation or output.streams`);for(let[r,a]of[[`output.filledElevation`,n.filledElevation],[`output.accumulation`,n.accumulation],[`runoff`,e.runoff]])if(a&&(y(a,[`float32`],`${t} ${r}`),a.length!==i))throw Error(`${t} ${r} must contain one value per cell`);for(let[e,r]of[[`output.flowDirections`,n.flowDirections],[`output.cellClasses`,n.cellClasses],[`output.streams`,n.streams]])if(r&&(y(r,[`uint32`],`${t} ${e}`),r.length!==i))throw Error(`${t} ${e} must contain one value per cell`);for(let[e,n]of[[`fillConverged`,r?.fillConverged],[`accumulationConverged`,r?.accumulationConverged],[`flatsConverged`,r?.flatsConverged],[`fillIterationCount`,r?.fillIterationCount],[`flatsIterationCount`,r?.flatsIterationCount],[`accumulationIterationCount`,r?.accumulationIterationCount]])if(n&&(y(n,[`uint32`],`${t} output.status.${e}`),n.length!==1))throw Error(`${t} output.status.${e} must contain one row`);De(t,e)}getCommandNodes(n){let i=this.props.grid.width*this.props.grid.height,a=!!(this.props.output.accumulation||this.props.output.streams),o=!!(this.props.output.flowDirections||this.props.output.cellClasses||a),s=!!(this.props.resolveFlats&&o),c=!!(this.props.fillDepressions&&8*(i+1)<=n.device.limits.maxStorageBufferBindingSize),l=a&&(this.props.flowRouting??`d8`)===`d8`&&!this.props.runoff&&(this.props.accumulationUnits??`cells`)===`cells`&&8*i<=n.device.limits.maxStorageBufferBindingSize,u=Math.max(this.props.fillDepressions?this.props.maximumFillIterations??512:0,s?this.props.maximumFlatIterations??(this.props.flatContinuation?64:512):0,a?this.props.maximumAccumulationIterations??64:0,c||l?4:0);e(n,this.id,{...this.props,runoff:a?this.props.runoff:void 0,flatContinuation:s?this.props.flatContinuation:void 0},{"canonical and relaxation planes":4*i,"spill tree":c?8*(i+1):0,"Euler tour":l?8*i:0,"fractional routing records":a&&(this.props.flowRouting??`d8`)!==`d8`?8*i:0,"gate table":u?t(u):0});let{id:d,props:f}=this,{output:p}=f,{status:m}=p,{width:h,height:g}=f.grid,_=f.cellSizeMode??`uniform`;S(d,n,f.elevation,[]),b(d,n,[f.parameters,f.runoff,...Object.values(f.flatContinuation??{}).flatMap(e=>[e.state,e.values]),p.filledElevation,p.flowDirections,p.cellClasses,p.accumulation,p.streams,m?.fillConverged,m?.accumulationConverged,m?.flatsConverged,m?.fillIterationCount,m?.flatsIterationCount,m?.accumulationIterationCount]);let v=h*g,y={width:h,height:g,cellSizeMode:_},x=O(n,d,f.elevation,f.grid,!0),C=[...x.nodes],w=x.band.storage.values,T=w,E,D=[],k=()=>K(n,{id:d,cellCount:v,elevation:w,fillStatus:E,flatStatuses:D,filledElevation:p.filledElevation,directions:p.flowDirections,classes:p.cellClasses,accumulation:p.accumulation,streams:p.streams,codes:{direction:we,cellClass:$.unresolved,stream:Te}});if(f.fillDepressions){let e=p.filledElevation??r(n,`${d}-filled`,`float32`,v),t=Se(n,{...y,id:d,maxIterations:f.maximumFillIterations??512,elevation:w,filled:e,settings:f.parameters,converged:m?.fillConverged,iterationCount:m?.fillIterationCount});C.push(...t.nodes),E=t.status,T=e}let A=!!(p.accumulation||p.streams);if(!p.flowDirections&&!p.cellClasses&&!A)return C.push(...k()),C;let j=A?r(n,`${d}-receivers`,`uint32`,v):void 0,M=p.cellClasses??(f.resolveFlats?r(n,`${d}-classes`,`uint32`,v):void 0);if(C.push(ae(n,{...y,id:`${d}-flow-direction`,elevation:w,surface:T,settings:f.parameters,directions:p.flowDirections,classes:M,receivers:j,cellClass:$})),f.resolveFlats&&M){let e=Y(n,{...y,id:d,maxIterations:f.maximumFlatIterations??(f.flatContinuation?64:512),continuation:f.flatContinuation,elevation:w,surface:T,settings:f.parameters,classes:M,directions:p.flowDirections,receivers:j,converged:m?.flatsConverged,iterationCount:m?.flatsIterationCount,cellClass:$});C.push(...e.nodes),D=e.statuses}if(j){let e=p.accumulation??r(n,`${d}-accumulation`,`float32`,v),t=f.flowRouting??`d8`,i={...y,id:d,maxIterations:f.maximumAccumulationIterations??64,elevation:w,receivers:j,accumulation:e,settings:f.parameters,runoff:f.runoff,area:(f.accumulationUnits??`cells`)===`area`,converged:m?.accumulationConverged,iterationCount:m?.accumulationIterationCount};C.push(...t===`d8`?U(n,i):Ce(n,{...i,routing:t,surface:T})),p.streams&&C.push(ce(n,{id:`${d}-streams`,cellCount:v,elevation:w,accumulation:e,streams:p.streams,settings:f.parameters,unresolvedValue:Te}))}return C.push(...k()),C}};export{Z as a,B as c,Ee as i,R as l,Q as n,H as o,Te as r,z as s,Oe as t};