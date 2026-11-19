import{E as e,T as t}from"./gpu-contributor-utils-BIwLQON5.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as r,i,r as a,s as o}from"./raster-relaxation-Di9wjtxp.js";var s=[`esri`,`whitebox`,`taudem`];function c(e,n){t(e,n===void 0||s.includes(n),`directionEncoding must be 'esri', 'whitebox' or 'taudem' (got ${String(n)})`)}function l(e){switch(e){case`esri`:return`fn getFlowDirectionCode(word: u32) -> u32 { return word; }`;case`whitebox`:return`fn getFlowDirectionCode(word: u32) -> u32 {
  if (word == 0u || word >= UNRESOLVED_DIRECTION) { return word; }
  if (word > 128u || (word & (word - 1u)) != 0u) { return INVALID_DIRECTION_CODE; }
  return select(word >> 1u, 128u, word == 1u);
}`;case`taudem`:return`fn getFlowDirectionCode(word: u32) -> u32 {
  if (word == 0u || word >= UNRESOLVED_DIRECTION) { return word; }
  if (word > 8u) { return INVALID_DIRECTION_CODE; }
  return 1u << ((9u - word) % 8u);
}`}}function u(e,t,n=`esri`){return`
const GRID_WIDTH: u32 = ${e}u;
const GRID_HEIGHT: u32 = ${t}u;
const NO_CELL: u32 = 0xffffffffu;
// GPU_TERRAIN_FLOW_UNRESOLVED: a valid cell whose direction is unknown. It has no receiver but is
// not a terminal cell, so consumers must not read it as an outlet or a sink.
const UNRESOLVED_DIRECTION: u32 = 0xfffffffeu;
// A direction word outside the contributor's encoding: nonzero and without a receiver.
const INVALID_DIRECTION_CODE: u32 = 0x100u;

${l(n)}

fn getFlowReceiver(cell: u32, code: u32) -> u32 {
  if (code == 0u || code > 128u || (code & (code - 1u)) != 0u) { return NO_CELL; }
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let direction = firstTrailingBit(code);
  let column = i32(cell % GRID_WIDTH) + columnOffsets[direction];
  let row = i32(cell / GRID_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) {
    return NO_CELL;
  }
  return u32(row) * GRID_WIDTH + u32(column);
}`}function d(e){switch(e.kind){case`none`:return{declarations:`fn isStopCell(cell: u32, code: u32) -> bool { return false; }`};case`streams`:return{binding:{name:`stops`,view:e.streams,type:`u32`,access:`read`},declarations:`fn isStopCell(cell: u32, code: u32) -> bool {
  return code != NO_CELL && stops[stopsOffset + cell] != 0u;
}`};case`markers`:return{binding:{name:`stops`,view:e.markers,type:`u32`,access:`read`},declarations:`fn isStopCell(cell: u32, code: u32) -> bool {
  return stops[stopsOffset + cell] != NO_CELL;
}`}}}function f(e,t){let s=t.width*t.height,c=r(e,`${t.id}-pointer`,t.operation,s,t.contributorId),l=d(t.stops),f=u(t.width,t.height,t.directionEncoding),m=[{name:`flowDirections`,view:t.flowDirections,type:`u32`,access:`read`},...l.binding?[l.binding]:[],{name:`pointer`,view:t.pointer,type:`u32`,access:`read_write`}],h=[n(e,{id:`${t.id}-pointer-init`,operation:t.operation,variant:`pointer-init`,bindings:m,invocationCount:s,declarations:`${f}\n${l.declarations}`,body:`let code = getFlowDirectionCode(flowDirections[flowDirectionsOffset + index]);
  var jumpTarget = index;
  if (!isStopCell(index, code)) {
    let receiver = getFlowReceiver(index, code);
    if (receiver != NO_CELL) { jumpTarget = receiver; }
  }
  pointer[pointerOffset + index] = jumpTarget;`}),o(e,{id:`${t.id}-pointer-reset`,operation:t.operation,state:c})],g=[{name:`pointer`,view:t.pointer,type:`atomic<u32>`,access:`read_write`},{name:`status`,view:c.status,type:`atomic<u32>`,access:`read_write`}],_=e=>`let current = atomicLoad(&pointer[pointerOffset + index]);
  let next = atomicLoad(&pointer[pointerOffset + current]);
  if (next != current) {
    atomicStore(&pointer[pointerOffset + index], next);
    ${e}
  }`;return h.push(...a(e,{id:`${t.id}-pointer-round`,gateId:`${t.id}-pointer-gate`,operation:t.operation,state:c,maxIterations:t.maxIterations,createRound:({nodeId:r,slotBinding:i,markChangedWGSL:a,condition:o,extraResources:c})=>n(e,{id:r,operation:t.operation,variant:`pointer-round`,bindings:[...g,i],invocationCount:s,body:_(a),condition:o,extraResources:c})})),(t.converged||t.iterationCount)&&h.push(i(e,{id:`${t.id}-pointer-finalize`,operation:t.operation,state:c,converged:t.converged,iterationCount:t.iterationCount})),t.iterationLimitReached&&h.push(p(e,{id:`${t.id}-pointer-limit`,operation:t.operation,state:c,iterationLimitReached:t.iterationLimitReached})),h}function p(e,t){return n(e,{id:t.id,operation:t.operation,variant:`iteration-limit`,bindings:[{name:`status`,view:t.state.status,type:`u32`,access:`read`},{name:`iterationLimitReached`,view:t.iterationLimitReached,type:`u32`,access:`read_write`}],invocationCount:1,body:`iterationLimitReached[iterationLimitReachedOffset] = 1u - status[statusOffset + 3u];`})}function m(t,n,r,i,a){if(e(r,[i],`${t} ${n}`),r.length!==a)throw Error(`${t} ${n} must contain one value per cell`)}function h(t,n){for(let[r,i]of[[`converged`,n?.converged],[`iterationLimitReached`,n?.iterationLimitReached],[`iterationCount`,n?.iterationCount]])if(i&&(e(i,[`uint32`],`${t} output.status.${r}`),i.length!==1))throw Error(`${t} output.status.${r} must contain one row`)}export{m as a,c as i,p as n,h as o,u as r,f as t};