import{t as e}from"./gpu-dispatch-utils-CbRzFAgT.js";import{c as t}from"./raster-relaxation-Di9wjtxp.js";import{t as n}from"./raster-grid-utils-D2opjNmg.js";import{a as r,o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{u as a}from"./wgsl-common-DfyXiYB0.js";function o(e){return 32-Math.clz32(e)}function s(e){let t=o(Math.max(e,2)-1);return Math.ceil((t+1)/4)*4}function c(e,t,n){let{condition:r,extraResources:i}=e.getGate(t);return{condition:{...r,id:`${n}-gate`},extraResources:i}}function l(i,a){if(a.shapes.length===1&&(a.statusLength??4)===4)return t(i,a.id,a.operation,a.shapes[0][0],a.contributorId);let o=r(i,{id:`${a.id}-dispatch`,operation:a.operation,shapes:a.shapes});return{status:n(i,a.operation,a.contributorId,`${a.id}-status`,`uint32`,a.statusLength??4),dispatch:o.view,gate:o,layout:e(a.operation,Math.max(a.shapes[0][0],1),a.shapes[0][1],i.device.limits.maxComputeWorkgroupsPerDimension),invocationCount:a.shapes[0][0]}}function u(e,t){let n=[];for(let e=0;e<(t.statusLength??4);e++)n.push(e===1?`status[statusOffset + 1u] = select(0u, 1u, enabled);`:`status[statusOffset + ${e}u] = 0u;`);return i(e,{id:t.id,operation:t.operation,variant:`spill-reset`,bindings:[...t.enabledBindings,{name:`status`,view:t.state.status,type:`u32`,access:`read_write`},t.state.gate.binding],invocationCount:1,body:`let enabled = ${t.enabledExpression};
  ${n.join(`
  `)}
  if (enabled) {
    ${t.state.gate.armWGSL}
  } else {
    ${t.state.gate.disarmWGSL}
  }`})}function d(e,t){return`
const SPILL_WIDTH: u32 = ${e}u;
const SPILL_HEIGHT: u32 = ${t}u;
const SPILL_NONE: u32 = 0xffffffffu;
const SPILL_BOUNDARY_FLAG: u32 = 0x80000000u;
const SPILL_ID_MASK: u32 = 0x7fffffffu;
const SPILL_INFINITE_KEY: u32 = 0xffffffffu;

fn getSpillNeighbor(cell: u32, direction: u32) -> u32 {
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let column = i32(cell % SPILL_WIDTH) + columnOffsets[direction];
  let row = i32(cell / SPILL_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(SPILL_WIDTH) || row >= i32(SPILL_HEIGHT)) {
    return SPILL_NONE;
  }
  return u32(row) * SPILL_WIDTH + u32(column);
}

${a}
fn getSpillKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, bits >= 0x80000000u);
}

fn getSpillValueBits(key: u32) -> u32 {
  return select(~key, key & 0x7fffffffu, key >= 0x80000000u);
}`}export{c as a,d as i,u as n,s as o,o as r,l as t};