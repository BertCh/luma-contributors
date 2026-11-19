import{o as e,s as t}from"./network-view-utils-B1U8oNKx.js";import{r as n}from"./scan-total-C4QddMmi.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{t as a}from"./status-aOwAAXek.js";import{t as o}from"./spatial-weights-CqWysstN.js";import{E as s,f as c,y as l}from"./gpu-contributor-utils-BIwLQON5.js";import{c as u}from"./wgsl-common-DfyXiYB0.js";import{o as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as f}from"./scaled-quotient-wgsl-UDyhQag6.js";var p=`GPUNetworkCostWeights`,m=256,h={none:0,exponential:1,power:2,gaussian:3};function g(e){let t=e.decay??`none`,n=e.beta??0,r=e.minimumCost??1,{costLimit:i}=e;if(!(i>=0))throw Error(`${p} costLimit must be a non-negative number`);if(!(t in h))throw Error(`${p} decay must be 'none', 'exponential', 'power' or 'gaussian'`);if(t===`gaussian`&&!(Number.isFinite(i)&&i>0))throw Error(`${p} 'gaussian' decay needs a finite positive costLimit`);if(!Number.isFinite(n)||n<0)throw Error(`${p} beta must be a non-negative finite number`);if(!Number.isFinite(r)||!(r>0))throw Error(`${p} minimumCost must be a positive finite number`);return Float32Array.from([i,h[t],n,r])}var _=class{id;props;orientation;rowCount;columnCount;locationCount;weightsRowCount;capacity;outputWeights;constructor(e){let n=e.id??`network-cost-weights`;if(this.id=n,this.props=e,t(this.id,e.output,[{outputs:{neighbors:e.output.neighbors,weights:e.output.weights,distances:e.output.distances},inputs:{offsets:e.output.offsets}},{outputs:{offsets:e.output.offsets,status:e.output.status}}]),this.orientation=e.orientation??`rows`,this.orientation!==`rows`&&this.orientation!==`columns`)throw Error(`${n} orientation must be 'rows' or 'columns'`);let{output:r}=e;if(s(e.costs,[`float32`],`${n} costs`),s(e.parameters,[`float32`],`${n} parameters`),e.parameters.length!==4)throw Error(`${n} parameters must contain 4 rows`);if(this.rowCount=e.rowCount??1,c(n,`rowCount`,this.rowCount,1,e.costs.length),e.costs.length%this.rowCount!==0)throw Error(`${n} costs length must be a multiple of rowCount`);if(this.columnCount=e.costs.length/this.rowCount,e.columns&&s(e.columns,[`uint32`],`${n} columns`),this.locationCount=e.columns?.length??this.columnCount,c(n,`columns length`,this.locationCount),this.weightsRowCount=this.orientation===`rows`?this.rowCount:this.locationCount,s(r.offsets,[`uint32`],`${n} output.offsets`),r.offsets.length!==this.weightsRowCount+1)throw Error(`${n} output.offsets must contain one row per weights row plus one`);s(r.neighbors,[`uint32`],`${n} output.neighbors`),this.capacity=r.neighbors.length,c(n,`output.neighbors length`,this.capacity);for(let[e,t]of[[`output.weights`,r.weights],[`output.distances`,r.distances]])if(t&&(s(t,[`float32`],`${n} ${e}`),t.length!==this.capacity))throw Error(`${n} ${e} length must equal output.neighbors length`);a(n,{overflow:r.status?.overflow,requiredCount:r.status?.requiredCount}),l(n,y(e),v(e)),this.outputWeights=o({offsets:r.offsets,neighbors:r.neighbors,weights:r.weights,distances:r.distances},{information:{rowCount:this.weightsRowCount,transform:`kernel`,symmetric:`unknown`,provenance:{source:p,criterion:`network-cost`}},status:{overflow:r.status?.overflow,requiredCount:r.status?.requiredCount}})}getCommandNodes(t){let{id:a,props:o}=this,{output:s}=o;e(a,t,[...v(o),...y(o)]),e(a,t,[],{rowCounts:(this.weightsRowCount+1)*4});let c=r(t,`${a}-row-counts`,`uint32`,this.weightsRowCount+1),l=r(t,`${a}-scan-total`,`uint32`,1),u=this.getDeclarations(),h=[{name:`costs`,view:o.costs,type:`f32`,access:`read`},{name:`parameters`,view:o.parameters,type:`f32`,access:`read`}];o.columns&&h.push({name:`columns`,view:o.columns,type:`u32`,access:`read`});let g=m,_=d(t,{id:`${a}-count`,operation:p,variant:`count`,bindings:[...h,{name:`counts`,view:c,type:`u32`,access:`read_write`}],workgroupSize:g,invocationCount:(this.weightsRowCount+1)*g,guardIndex:!1,declarations:`${u}
var<workgroup> partials: array<u32, ${g}>;`,body:`let row = index / ${g}u;
  let costLimit = parameters[parametersOffset];
  var count = 0u;
  if (row < WEIGHTS_ROWS) {
    for (var neighbor = localInvocationIndex; neighbor < NEIGHBOR_SPACE; neighbor += ${g}u) {
      if (isWithinCostLimit(getEntryCost(row, neighbor), costLimit)) {
        count += 1u;
      }
    }
  }
  partials[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${g/2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partials[localInvocationIndex] += partials[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && row <= WEIGHTS_ROWS) {
    counts[countsOffset + row] = partials[0];
  }`}),b=new i({id:`${a}-offsets`,input:c,output:s.offsets,mode:`exclusive`}).getCommandNodes(t),x=[...h,{name:`offsets`,view:s.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:s.neighbors,type:`u32`,access:`read_write`},{name:`weights`,view:s.weights,type:`f32`,access:`read_write`}];s.distances&&x.push({name:`distances`,view:s.distances,type:`f32`,access:`read_write`});let S=d(t,{id:`${a}-fill`,operation:p,variant:`fill`,bindings:x,workgroupSize:g,invocationCount:this.weightsRowCount*g,guardIndex:!1,declarations:`${u}
${f}
var<workgroup> prefix: array<u32, ${g}>;
fn getDecayWeight(cost: f32, costLimit: f32, decay: u32, beta: f32, minimumCost: f32) -> f32 {
  if (decay == 1u) {
    return exp(-beta * cost);
  }
  if (decay == 2u) {
    return pow(max(cost, minimumCost), -beta);
  }
  if (decay == 3u) {
    let ratio = getScaledQuotient(cost, costLimit, 1.0, 0);
    let edge = exp(-0.5);
    return max((exp(-0.5 * ratio * ratio) - edge) / (1.0 - edge), 0.0);
  }
  return 1.0;
}`,body:`let row = index / ${g}u;
  let isActiveRow = row < WEIGHTS_ROWS;
  let costLimit = parameters[parametersOffset];
  let decay = u32(max(parameters[parametersOffset + 1u], 0.0));
  let beta = parameters[parametersOffset + 2u];
  let minimumCost = parameters[parametersOffset + 3u];
  var base = 0u;
  if (isActiveRow) {
    base = offsets[offsetsOffset + row];
  }
  for (var chunk = 0u; chunk < NEIGHBOR_SPACE; chunk += ${g}u) {
    let neighbor = chunk + localInvocationIndex;
    var cost = -1.0;
    if (isActiveRow && neighbor < NEIGHBOR_SPACE) {
      cost = getEntryCost(row, neighbor);
    }
    let isIncluded = isWithinCostLimit(cost, costLimit);
    prefix[localInvocationIndex] = select(0u, 1u, isIncluded);
    workgroupBarrier();
    for (var stride = 1u; stride < ${g}u; stride = stride * 2u) {
      var addend = 0u;
      if (localInvocationIndex >= stride) {
        addend = prefix[localInvocationIndex - stride];
      }
      workgroupBarrier();
      prefix[localInvocationIndex] += addend;
      workgroupBarrier();
    }
    let slot = base + prefix[localInvocationIndex] - 1u;
    if (isIncluded && slot < CAPACITY) {
      neighbors[neighborsOffset + slot] = neighbor;
      weights[weightsOffset + slot] = getDecayWeight(cost, costLimit, decay, beta, minimumCost);
      ${s.distances?`distances[distancesOffset + slot] = cost;`:``}
    }
    base += prefix[255u];
    workgroupBarrier();
  }`}),C=[{name:`scanTotal`,view:l,type:`u32`,access:`read`},{name:`offsets`,view:s.offsets,type:`u32`,access:`read_write`}];s.status?.overflow&&C.push({name:`overflow`,view:s.status.overflow,type:`u32`,access:`read_write`}),s.status?.requiredCount&&C.push({name:`requiredCount`,view:s.status.requiredCount,type:`u32`,access:`read_write`});let w=d(t,{id:`${a}-finalize`,operation:p,variant:`finalize`,bindings:C,invocationCount:this.weightsRowCount+1,declarations:`const WEIGHTS_ROWS: u32 = ${this.weightsRowCount}u;
const CAPACITY: u32 = ${this.capacity}u;`,body:`let value = offsets[offsetsOffset + index];
  if (index == WEIGHTS_ROWS) {
    let total = scanTotal[scanTotalOffset];
    ${s.status?.overflow?`overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);`:``}
    ${s.status?.requiredCount?`requiredCount[requiredCountOffset] = total;`:``}
  }
  offsets[offsetsOffset + index] = min(value, CAPACITY);`});return[_,...b,...n(t,{id:a,operation:p,counts:c,offsets:s.offsets,rowCount:this.weightsRowCount,total:l}),S,w]}getDeclarations(){let e=this.orientation===`rows`,t=e?this.locationCount:this.rowCount,n=e?`neighbor`:`row`,r=e?`row`:`neighbor`,i=this.props.columns?`columns[columnsOffset + ${n}]`:n;return`const WEIGHTS_ROWS: u32 = ${this.weightsRowCount}u;
const NEIGHBOR_SPACE: u32 = ${t}u;
const MATRIX_COLUMNS: u32 = ${this.columnCount}u;
const CAPACITY: u32 = ${this.capacity}u;
${u}
fn getEntryCost(row: u32, neighbor: u32) -> f32 {
  let column = ${i};
  if (column >= MATRIX_COLUMNS) {
    return -1.0;
  }
  return costs[costsOffset + ${r} * MATRIX_COLUMNS + column];
}
fn isWithinCostLimit(cost: f32, costLimit: f32) -> bool {
  // Unreached entries are +Infinity; an infinite limit must still reject them.
  return isFiniteBits(cost) && cost >= 0.0 && cost <= costLimit;
}`}};function v(e){return[e.costs,e.parameters,e.columns]}function y(e){let{output:t}=e;return[t.offsets,t.neighbors,t.weights,t.distances,t.status?.overflow,t.status?.requiredCount]}export{g as n,_ as t};