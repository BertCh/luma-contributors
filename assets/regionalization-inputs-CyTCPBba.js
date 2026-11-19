import{r as e}from"./kernel-C3tfQOn4.js";import{o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{l as n}from"./wgsl-common-DfyXiYB0.js";import{E as r,S as i,f as a,u as o,v as s}from"./gpu-contributor-utils-BIwLQON5.js";import{t as c}from"./gpu-spatial-weights-transpose-BOTZ1Oz3.js";import{t as l}from"./spatial-weights-CqWysstN.js";var u=256,d=256,f=16;function p(e,t,n,r){if(n.information?.symmetric===!0)return;let i=n.neighbors.length;for(let[n,a]of[[`adjacency offsets`,(r+1)*4],[`adjacency neighbors and weights`,Math.max(1,2*i)*4],[`transpose sort histogram`,i>d?Math.ceil(i/d)*f*4:0],[`transpose parameters`,4]])s(e.device,t,n,a,`reduce rows or weights slot capacity`)}function m(e,t=0){return a(e,`seed`,t,0),Uint32Array.of(t)}function h(e){return o(e,8)}function g(e,t,n){i(e,`weights.neighbors rows`,t.neighbors.length,n)}function _(e,t,n){if(r(t,[`uint32`],`${e} parameters`),t.length<n)throw Error(`${e} parameters must hold ${n} words`)}function v(e,r){let{rowCount:i,columnCount:a}=r;return t(e,{id:r.id,operation:r.operation,variant:`standardize`,bindings:[{name:`values`,view:r.values,type:`f32`,access:`read`},{name:`scaled`,view:r.scaled,type:`f32`,access:`read_write`}],workgroupSize:u,invocationCount:a*u,guardIndex:!1,declarations:`const ROWS: u32 = ${i}u;
const COLUMNS: u32 = ${a}u;
const LANES: u32 = ${u}u;
${n}
var<workgroup> partials: array<f32, ${u}>;

// Sum of one value per lane over the workgroup, available to every lane.
fn reduceLanes(lane: u32, value: f32) -> f32 {
  partials[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partials[lane] = partials[lane] + partials[lane + stride];
    }
    workgroupBarrier();
  }
  let total = partials[0];
  workgroupBarrier();
  return total;
}`,body:`let column = index / LANES;
  let lane = localInvocationIndex;
  // Center before reducing: a rounded mean of a constant column can otherwise
  // create a nonzero deviation and normalize every row to +1 or -1. The origin is the first
  // finite entry; statistics skip non-finite entries, which stay as they are.
  var origin = 0.0;
  for (var row = 0u; row < ROWS; row++) {
    let candidate = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(candidate)) { origin = candidate; break; }
  }
  var sum = 0.0;
  var count = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    let value = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(value)) { sum += value - origin; count += 1.0; }
  }
  let finiteCount = max(reduceLanes(lane, count), 1.0);
  let mean = reduceLanes(lane, sum) / finiteCount;
  var squares = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    let value = values[valuesOffset + row * COLUMNS + column];
    if (isFiniteFloat(value)) {
      let delta = (value - origin) - mean;
      squares += delta * delta;
    }
  }
  let deviation = sqrt(reduceLanes(lane, squares) / finiteCount);
  let inverse = select(0.0, 1.0 / deviation, deviation > 1e-20);
  for (var row = lane; row < ROWS; row += LANES) {
    let value = values[valuesOffset + row * COLUMNS + column];
    scaled[scaledOffset + row * COLUMNS + column] =
      select(value, ((value - origin) - mean) * inverse, isFiniteFloat(value));
  }`})}function y(n,r){let{id:i,operation:a,weights:o,rowCount:s}=r;if(o.information?.symmetric===!0)return{weights:o,nodes:[]};let u=o.neighbors.length,d=e(n,`${i}-adjacency-t-offsets`,`uint32`,s+1),f=e(n,`${i}-adjacency-t-neighbors`,`uint32`,u),p=e(n,`${i}-adjacency-t-weights`,`float32`,u),m=e(n,`${i}-adjacency-offsets`,`uint32`,s+1),h=e(n,`${i}-adjacency-neighbors`,`uint32`,2*u),g=e(n,`${i}-adjacency-weights`,`float32`,2*u),_=`const ROWS: u32 = ${s}u;
const CAPACITY: u32 = ${2*u}u;`,v=[...new c({id:`${i}-adjacency-transpose`,weights:{offsets:o.offsets,neighbors:o.neighbors,weights:o.weights},output:{offsets:d,neighbors:f,weights:p}}).getCommandNodes(n),t(n,{id:`${i}-adjacency-offsets`,operation:a,variant:`adjacency-offsets`,bindings:[{name:`offsets`,view:o.offsets,type:`u32`,access:`read`},{name:`transposeOffsets`,view:d,type:`u32`,access:`read`},{name:`adjacencyOffsets`,view:m,type:`u32`,access:`read_write`}],invocationCount:s+1,body:`adjacencyOffsets[adjacencyOffsetsOffset + index] = offsets[offsetsOffset + index] + transposeOffsets[transposeOffsetsOffset + index];`}),t(n,{id:`${i}-adjacency-neighbors`,operation:a,variant:`adjacency-neighbors`,bindings:[{name:`offsets`,view:o.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:o.neighbors,type:`u32`,access:`read`},{name:`transposeOffsets`,view:d,type:`u32`,access:`read`},{name:`transposeNeighbors`,view:f,type:`u32`,access:`read`},{name:`adjacencyOffsets`,view:m,type:`u32`,access:`read`},{name:`adjacencyNeighbors`,view:h,type:`u32`,access:`read_write`}],invocationCount:2*u,declarations:`${_}
// Row that owns adjacency slot \`slot\`: the first row whose range ends after it.
fn findAdjacencyRow(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (adjacencyOffsets[adjacencyOffsetsOffset + middle + 1u] > slot) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  return low;
}`,body:`var neighbor = 0u;
  if (index < adjacencyOffsets[adjacencyOffsetsOffset + ROWS]) {
    let row = findAdjacencyRow(index);
    let local = index - adjacencyOffsets[adjacencyOffsetsOffset + row];
    let ownCount = offsets[offsetsOffset + row + 1u] - offsets[offsetsOffset + row];
    if (local < ownCount) {
      neighbor = neighbors[neighborsOffset + offsets[offsetsOffset + row] + local];
      if (neighbor >= ROWS) { neighbor = row; }
    } else {
      neighbor = transposeNeighbors[transposeNeighborsOffset + transposeOffsets[transposeOffsetsOffset + row] + local - ownCount];
    }
  }
  adjacencyNeighbors[adjacencyNeighborsOffset + index] = neighbor;`})];return{weights:l({offsets:m,neighbors:h,weights:g},{information:{rowCount:s,transform:`unknown`,symmetric:!0}}),nodes:v}}export{_ as a,m as i,y as n,g as o,h as r,p as s,v as t};