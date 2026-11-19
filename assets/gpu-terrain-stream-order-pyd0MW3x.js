import{r as e,t}from"./terrain-binding-preflight-laSEObQE.js";import{t as n}from"./common-CshkX-d_.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{c as a,i as o,m as s,r as c,s as l}from"./raster-relaxation-Di9wjtxp.js";import{a as u,o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{l as m,u as h}from"./raster-band-utils-BcTxc8_Y.js";import{c as g,l as _,o as v,r as y,s as b}from"./gpu-terrain-flow-BhpjXagd.js";import{a as x,i as S,n as C,o as w,r as T}from"./terrain-flow-tracing-BfyLX2kv.js";var E=`GPUTerrainStreamOrder`,D=class{id;props;constructor(e){this.id=e.id??`terrain-stream-order`,this.props=e;let{id:t}=this,{output:n}=e,r=h(t,e.grid);s(t,`maximumIterations`,e.maximumIterations??64),S(t,e.directionEncoding),x(t,`flowDirections`,e.flowDirections,`uint32`,r),x(t,`streams`,e.streams,`uint32`,r),x(t,`output.streamOrder`,n.streamOrder,`uint32`,r),w(t,n.status);let i=[n.streamOrder,n.status?.converged,n.status?.iterationLimitReached,n.status?.iterationCount];p(t,i,[e.flowDirections,e.streams]),m(t,i,[])}getCommandNodes(n){e(n,this.id,this.props,{"forest scratch":8*this.props.grid.width*this.props.grid.height<=n.device.limits.maxStorageBufferBindingSize?8*this.props.grid.width*this.props.grid.height:4*this.props.grid.width*this.props.grid.height,"gate table":t(Math.max(this.props.maximumIterations??64,4))});let{width:r,height:i}=this.props.grid;return 8*r*i<=n.device.limits.maxStorageBufferBindingSize?this.getTourCommandNodes(n):this.getPullCommandNodes(n)}getTourCommandNodes(e){let{id:t,props:a}=this,{output:s}=a,{width:c,height:l}=a.grid;f(t,e,[a.flowDirections,a.streams,s.streamOrder,s.status?.converged,s.status?.iterationLimitReached,s.status?.iterationCount]);let p=c*l,m=2*p,h=(n,i)=>r(e,`${t}-order-${n}`,`uint32`,i),v=h(`member`,p),x=h(`root-length`,p),S=h(`root-offset`,p),w=h(`position`,m),D=h(`level-status`,4),O=[{name:`flowDirections`,view:a.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:a.streams,type:`u32`,access:`read`}],k=T(c,l,a.directionEncoding),A=`fn isStreamCell(cell: u32) -> bool {
  return streams[streamsOffset + cell] != 0u && flowDirections[flowDirectionsOffset + cell] != NO_CELL;
}

fn getTourParent(cell: u32) -> u32 {
  if (!isStreamCell(cell)) { return NO_CELL; }
  let receiver = getFlowReceiver(cell, getFlowDirectionCode(flowDirections[flowDirectionsOffset + cell]));
  if (receiver == NO_CELL || !isStreamCell(receiver)) { return NO_CELL; }
  return receiver;
}`,j=`${k}
${_(c,l)}
${A}`,M=`${j}
${b}`,N=g(e,{id:t,operation:E,contributorId:t,width:c,height:l,declarations:`${k}\n${A}`,parentBindings:O}),P=[{name:`tourSucc`,view:N.successor,type:`u32`,access:`read`},{name:`tourDist`,view:N.distance,type:`u32`,access:`read`}],[F,I]=N.scratch,L=[d(e,{id:`${t}-order-init`,operation:E,variant:`order-tour-init`,bindings:[...O,{name:`order`,view:s.streamOrder,type:`u32`,access:`read_write`},{name:`member`,view:v,type:`u32`,access:`read_write`}],invocationCount:p,declarations:`${k}\n${A}`,body:`let isStream = isStreamCell(index);
  order[orderOffset + index] = select(0u, 1u, isStream);
  member[memberOffset + index] = select(0u, 1u, isStream);`}),...N.nodes,...s.status?.iterationLimitReached?[C(e,{id:`${t}-order-limit`,operation:E,state:N.state,iterationLimitReached:s.status.iterationLimitReached})]:[],d(e,{id:`${t}-order-root-length`,operation:E,variant:`order-root-length`,bindings:[...O,...P,{name:`rootLength`,view:x,type:`u32`,access:`read_write`}],invocationCount:p,declarations:M,body:`let isRoot = isTourEdgeResolved(2u * index) && getTourParent(index) == TOUR_NONE;
  rootLength[rootLengthOffset + index] = select(0u, tourDist[tourDistOffset + 2u * index] + 1u, isRoot);`}),...new i({id:`${t}-order-root-scan`,input:x,output:S,mode:`exclusive`}).getCommandNodes(e),d(e,{id:`${t}-order-position`,operation:E,variant:`order-position`,bindings:[...O,...P,{name:`rootOffset`,view:S,type:`u32`,access:`read`},{name:`position`,view:w,type:`u32`,access:`read_write`}],invocationCount:m,declarations:M,body:`if (!isTourEdgeResolved(index)) {
    position[positionOffset + index] = TOUR_NONE;
    return;
  }
  let root = tourSucc[tourSuccOffset + index] >> 1u;
  position[positionOffset + index] = rootOffset[rootOffsetOffset + root] +
    tourDist[tourDistOffset + 2u * root] - tourDist[tourDistOffset + index];`})],R=n(31-Math.clz32(Math.max(p,1)),1,32),z=u(e,{id:`${t}-order-level-dispatch`,operation:E,shapes:[[p,256],[m,256]]});L.push(d(e,{id:`${t}-order-level-reset`,operation:E,variant:`order-level-reset`,bindings:[{name:`status`,view:D,type:`u32`,access:`read_write`},z.binding],invocationCount:1,body:`status[statusOffset + 0u] = 0u;
  status[statusOffset + 1u] = 1u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${z.armWGSL}`}));let B=(e,t)=>{let{condition:n,extraResources:r}=z.getGate(t);return{condition:{...n,id:`${e}-gate`},extraResources:r}};for(let n=0;n<R;n++){let r=`${t}-order-level-${n}-scatter`,a=`${t}-order-level-${n}-gather`;L.push(d(e,{id:r,operation:E,variant:`order-level-scatter`,bindings:[...O,{name:`position`,view:w,type:`u32`,access:`read`},{name:`member`,view:v,type:`u32`,access:`read`},{name:`indicator`,view:F,type:`u32`,access:`read_write`}],invocationCount:m,declarations:j,body:`let place = position[positionOffset + index];
  if (place == TOUR_NONE) { return; }
  var junction = 0u;
  if ((index & 1u) == 0u) {
    let cell = index >> 1u;
    var childCount = 0u;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getTourNeighbor(cell, direction);
      if (neighbor != TOUR_NONE && getTourParent(neighbor) == cell && member[memberOffset + neighbor] != 0u) {
        childCount = childCount + 1u;
      }
    }
    junction = select(0u, 1u, childCount >= 2u);
  }
  indicator[indicatorOffset + place] = junction;`,...B(r,1)}),...new i({id:`${t}-order-level-${n}-scan`,input:F,output:I,mode:`exclusive`}).getCommandNodes(e),d(e,{id:a,operation:E,variant:`order-level-gather`,bindings:[{name:`position`,view:w,type:`u32`,access:`read`},{name:`prefix`,view:I,type:`u32`,access:`read`},{name:`member`,view:v,type:`u32`,access:`read_write`},{name:`order`,view:s.streamOrder,type:`u32`,access:`read_write`},{name:`status`,view:D,type:`atomic<u32>`,access:`read_write`}],invocationCount:p,declarations:`const TOUR_NONE: u32 = 0xffffffffu;`,body:`if (member[memberOffset + index] == 0u) { return; }
  let enterPlace = position[positionOffset + 2u * index];
  if (enterPlace == TOUR_NONE) { return; }
  let exitPlace = position[positionOffset + 2u * index + 1u];
  if (prefix[prefixOffset + exitPlace] - prefix[prefixOffset + enterPlace] > 0u) {
    order[orderOffset + index] = order[orderOffset + index] + 1u;
    atomicAdd(&status[statusOffset + 0u], 1u);
  } else {
    member[memberOffset + index] = 0u;
  }`,...B(a,0)}),d(e,{id:`${t}-order-level-${n}-gate`,operation:E,variant:`order-level-gate`,bindings:[{name:`status`,view:D,type:`u32`,access:`read_write`},z.binding],invocationCount:1,body:`if (status[statusOffset + 1u] == 0u) { return; }
  let joined = status[statusOffset + 0u];
  status[statusOffset + 0u] = 0u;
  status[statusOffset + 2u] = status[statusOffset + 2u] + 1u;
  if (joined == 0u) {
    status[statusOffset + 1u] = 0u;
    ${z.disarmWGSL}
  }`}))}return L.push(d(e,{id:`${t}-order-unknown-scatter`,operation:E,variant:`order-unknown-scatter`,bindings:[...O,{name:`position`,view:w,type:`u32`,access:`read`},{name:`indicator`,view:F,type:`u32`,access:`read_write`}],invocationCount:m,declarations:j,body:`let place = position[positionOffset + index];
  if (place == TOUR_NONE) { return; }
  var seed = 0u;
  let cell = index >> 1u;
  if ((index & 1u) == 0u && isStreamCell(cell)) {
    var isSeed = streams[streamsOffset + cell] == ${y}u ||
      flowDirections[flowDirectionsOffset + cell] == UNRESOLVED_DIRECTION;
    for (var direction = 0u; direction < 8u; direction++) {
      let neighbor = getTourNeighbor(cell, direction);
      if (neighbor != TOUR_NONE && isStreamCell(neighbor) &&
          flowDirections[flowDirectionsOffset + neighbor] == UNRESOLVED_DIRECTION) {
        isSeed = true;
      }
    }
    seed = select(0u, 1u, isSeed);
  }
  indicator[indicatorOffset + place] = seed;`}),...new i({id:`${t}-order-unknown-scan`,input:F,output:I,mode:`exclusive`}).getCommandNodes(e),d(e,{id:`${t}-order-unknown-gather`,operation:E,variant:`order-unknown-gather`,bindings:[...O,{name:`position`,view:w,type:`u32`,access:`read`},{name:`prefix`,view:I,type:`u32`,access:`read`},{name:`order`,view:s.streamOrder,type:`u32`,access:`read_write`},{name:`status`,view:N.state.status,type:`atomic<u32>`,access:`read_write`}],invocationCount:p,declarations:j,body:`if (!isStreamCell(index)) { return; }
  let enterPlace = position[positionOffset + 2u * index];
  if (enterPlace == TOUR_NONE) {
    order[orderOffset + index] = NO_CELL;
    atomicStore(&status[statusOffset + 3u], 0u);
    return;
  }
  let exitPlace = position[positionOffset + 2u * index + 1u];
  if (prefix[prefixOffset + exitPlace] - prefix[prefixOffset + enterPlace] > 0u) {
    order[orderOffset + index] = NO_CELL;
  }`})),(s.status?.converged||s.status?.iterationCount)&&L.push(o(e,{id:`${t}-order-finalize`,operation:E,state:N.state,converged:s.status.converged,iterationCount:s.status.iterationCount})),L}getPullCommandNodes(e){let{id:t,props:n}=this,{output:r}=n,{width:i,height:s}=n.grid;f(t,e,[n.flowDirections,n.streams,r.streamOrder,r.status?.converged,r.status?.iterationLimitReached,r.status?.iterationCount]);let u=i*s,p=n.maximumIterations??64,m=a(e,`${t}-order`,E,u,t),h=T(i,s,n.directionEncoding),g=[d(e,{id:`${t}-order-init`,operation:E,variant:`order-init`,bindings:[{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:n.streams,type:`u32`,access:`read`},{name:`order`,view:r.streamOrder,type:`u32`,access:`read_write`}],invocationCount:u,declarations:h,body:`let isStream = streams[streamsOffset + index] != 0u &&
    flowDirections[flowDirectionsOffset + index] != NO_CELL;
  order[orderOffset + index] = select(0u, NO_CELL, isStream);`}),l(e,{id:`${t}-order-reset`,operation:E,state:m})],_=[{name:`order`,view:r.streamOrder,type:`atomic<u32>`,access:`read_write`},{name:`flowDirections`,view:n.flowDirections,type:`u32`,access:`read`},{name:`streams`,view:n.streams,type:`u32`,access:`read`},{name:`status`,view:m.status,type:`atomic<u32>`,access:`read_write`}],b=`${h}
const SENTINEL: u32 = 0xffffffffu;
const MAX_WALK_LENGTH: u32 = ${v}u;

// Unknown stream status (GPUTerrainFlow ran out of accumulation rounds): never finalized, so the
// cell and every stream cell downstream of it keep the unresolved sentinel.
fn isUnresolvedStream(cell: u32) -> bool {
  return streams[streamsOffset + cell] == ${y}u ||
    flowDirections[flowDirectionsOffset + cell] == UNRESOLVED_DIRECTION;
}

// Returns true when this thread wrote the final order of 'cell'.
fn tryFinalize(cell: u32) -> bool {
  if (atomicLoad(&order[orderOffset + cell]) != SENTINEL || isUnresolvedStream(cell)) { return false; }
  var maximumOrder = 0u;
  var maximumCount = 0u;
  for (var direction = 0u; direction < 8u; direction++) {
    // The neighbor in 'direction' donates when its own code points back at this cell.
    let donor = getFlowReceiver(cell, 1u << direction);
    if (donor == NO_CELL) { continue; }
    if (streams[streamsOffset + donor] == 0u) { continue; }
    // A stream neighbor with an unknown direction might donate: this order cannot be known.
    if (flowDirections[flowDirectionsOffset + donor] == UNRESOLVED_DIRECTION) { return false; }
    if (getFlowDirectionCode(flowDirections[flowDirectionsOffset + donor]) != (1u << ((direction + 4u) & 7u))) { continue; }
    let donorOrder = atomicLoad(&order[orderOffset + donor]);
    if (donorOrder == SENTINEL) { return false; }
    if (donorOrder > maximumOrder) {
      maximumOrder = donorOrder;
      maximumCount = 1u;
    } else if (donorOrder == maximumOrder) {
      maximumCount = maximumCount + 1u;
    }
  }
  var result = 1u;
  if (maximumCount > 0u) {
    result = maximumOrder + select(0u, 1u, maximumCount >= 2u);
  }
  return atomicCompareExchangeWeak(&order[orderOffset + cell], SENTINEL, result).exchanged;
}`,x=e=>`if (atomicLoad(&order[orderOffset + index]) != SENTINEL) { return; }
  // Stays unresolved without marking a change: no round can resolve it.
  if (isUnresolvedStream(index)) { return; }
  if (tryFinalize(index)) {
    var walk = getFlowReceiver(index, getFlowDirectionCode(flowDirections[flowDirectionsOffset + index]));
    for (var step = 0u; step < MAX_WALK_LENGTH; step++) {
      if (walk == NO_CELL || !tryFinalize(walk)) { break; }
      walk = getFlowReceiver(walk, getFlowDirectionCode(flowDirections[flowDirectionsOffset + walk]));
    }
  }
  if (atomicLoad(&order[orderOffset + index]) == SENTINEL) {
    ${e}
  }`;return g.push(...c(e,{id:`${t}-order-round`,gateId:`${t}-order-gate`,operation:E,state:m,maxIterations:p,createRound:({nodeId:t,slotBinding:n,markChangedWGSL:r,condition:i,extraResources:a})=>d(e,{id:t,operation:E,variant:`order-round`,bindings:[..._,n],invocationCount:u,declarations:b,body:x(r),condition:i,extraResources:a})})),(r.status?.converged||r.status?.iterationCount)&&g.push(o(e,{id:`${t}-order-finalize`,operation:E,state:m,converged:r.status.converged,iterationCount:r.status.iterationCount})),r.status?.iterationLimitReached&&g.push(C(e,{id:`${t}-order-limit`,operation:E,state:m,iterationLimitReached:r.status.iterationLimitReached})),g}};export{D as t};