import{t as e}from"./common-CshkX-d_.js";import{r as t,u as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./statistics-kernel-node-lacKNERm.js";import{E as i,f as a,x as o,y as s}from"./gpu-contributor-utils-BIwLQON5.js";import{n as c,t as l}from"./float32-values-60_XMjYS.js";import{i as u,s as d,t as f}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{o as p,s as m}from"./spatial-weights-CqWysstN.js";import{i as h}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as g}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{t as _}from"./spatial-statistics-collective-node-BlCi_ESE.js";import{t as v}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{i as y,n as b,o as x,s as S,t as C}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{n as w,t as T}from"./spatial-autocorrelation-status-aaIIJ64M.js";var E=256,D=256,O=6,k=class{id;props;islands;constructor(e){this.id=e.id??`local-geary`,this.props=e,this.islands=m(this.id,e.islands);let t=S({...e,id:this.id,globalCounts:e.output.globalCounts,globalStatistics:e.output.globalStatistics});if(i(e.output.statistic,[`float32`],`${this.id} output.statistic`),e.output.statistic.length!==t)throw Error(`${this.id} output.statistic length must equal the row count`);w(this.id,e.output.status,[e.output.statistic,e.output.globalStatistics]),s(this.id,[e.output.statistic,e.output.globalStatistics,e.output.globalCounts,...Object.values(e.output.status??{})],[...u(e.weights),e.weights.status?.overflow,e.weights.status?.candidateOverflow,e.values,e.parameters,e.mask]),h(this.id,{statistic:e.output.statistic,globalStatistics:e.output.globalStatistics,globalCounts:e.output.globalCounts,status:void 0,...e.output.status})}getCommandNodes(e){let{id:t,props:n}=this,r=f(n.weights);o(t,e,[...u(n.weights),n.weights.status?.overflow,n.weights.status?.candidateOverflow,n.values,n.parameters,n.mask,n.output.statistic,n.output.globalStatistics,n.output.globalCounts,...Object.values(n.output.status??{})]),x(e,{...n,id:t},[n.output.statistic,n.output.globalStatistics,n.output.globalCounts,...Object.values(n.output.status??{})]);let i=T(e,t,`GPULocalGeary`,n),a=n.values.length,s=b(e,{...n,globalCounts:n.output.globalCounts,globalStatistics:n.output.globalStatistics,id:t,islands:this.islands,operation:`GPULocalGeary`});return s.nodes.push(...r.map(r=>v(e,{rowBase:r.rowStart,id:`${t}-statistic${r.idSuffix}`,operation:`GPULocalGeary`,variant:`local-geary`,invocationCount:r.rowCount,bindings:[{name:`offsets`,view:r.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:r.weights.weights,type:`f32`,access:`read`},{name:`statistics`,view:s.statistics,type:`f32`,access:`read`},{name:`result`,view:n.output.statistic,type:`f32`,access:`read_write`}],declarations:`${C}
const ROWS: u32 = ${a}u;
const MOMENTS: u32 = ${a}u;
const STANDARDIZE: bool = ${p(n.weights)};`,body:`let focus = statistics[statisticsOffset + index];
  if (!isFiniteFloat(focus)) {
    result[resultOffset + index] = getQuietNaN(index);
    return;
  }
  let variance = statistics[statisticsOffset + MOMENTS + 2u];
  var sum = 0.0;
  var weightSum = 0.0;
  ${y(`let other = statistics[statisticsOffset + neighbor];
    let difference = focus - other;
    sum += weight * difference * difference;
    weightSum += weight;`,`WEIGHTS_ROW_START`)}
  if (STANDARDIZE && weightSum > 0.0) {
    // Row-standardized weights are divided by what remains of their row sum.
    sum = sum / weightSum;
  }
  // A constant variable has no standardized values: the statistic is undefined, not zero.
  result[resultOffset + index] = select(getQuietNaN(index), sum / variance, variance > 0.0 && isFiniteFloat(variance));`}))),[...i,...s.nodes]}},A=class{id;props;constructor(e){this.id=e.id??`spatial-pearson`,this.props=e,M(this.id,e)}getCommandNodes(e){return N(e,this.id,this.props,`pearson`)}},j=class{id;props;operation;constructor(e){if(this.id=e.id??`gamma-statistic`,this.props=e,this.operation=e.operation??`cross-product`,![`cross-product`,`squared-difference`,`absolute-difference`].includes(this.operation))throw Error(`${this.id} unknown operation ${this.operation}`);M(this.id,e)}getCommandNodes(e){return N(e,this.id,this.props,this.operation)}};function M(e,t){let r=d(e,t.weights);if(c(t.values,`${e} values`)!==r)throw Error(`${e} values length must equal weights rows`);if(t.secondValues&&c(t.secondValues,`${e} secondValues`)!==r)throw Error(`${e} secondValues length must equal weights rows`);if(t.mask&&(n(t.mask,`${e} mask`),t.mask.length!==r))throw Error(`${e} mask length must equal weights rows`);i(t.output.statistic,[`float32`],`${e} output.statistic`),a(e,`output.statistic length`,t.output.statistic.length),s(e,[t.output.statistic],[...u(t.weights),t.values,t.secondValues,t.mask])}function N(n,i,a,s){let c=f(a.weights),d=c.map(t=>e(Math.ceil(t.rowCount/E),1,D)),p=d.reduce((e,t)=>e+t,0);g(n,i,a,{"converted values":a.values.length*4,"edge moment partials":p*O*4}),o(i,n,[...u(a.weights),a.values,a.secondValues,a.mask,a.output.statistic]);let m=a.values.length,h=l(n,`${i}-values`,a.values),v=a.secondValues?l(n,`${i}-second-values`,a.secondValues):h,y=t(n,`${i}-partials`,`float32`,p*O),b=e=>[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:e.weights.weights,type:`f32`,access:`read`},{name:`values`,view:h.view,type:`f32`,access:`read`},{name:`secondValues`,view:v.view,type:`f32`,access:`read`},...a.mask?[{name:`mask`,view:a.mask,type:`u32`,access:`read`}]:[]],x=s===`cross-product`||s===`pearson`?`(x - meanX) * (y - meanY)`:s===`squared-difference`?`(x - y) * (x - y)`:s===`absolute-difference`?`abs(x - y)`:`0.0`,S=s===`pearson`?`GPUSpatialPearson`:`GPUGammaStatistic`,w=()=>`${C}
const ROWS: u32 = ${m}u;
var<private> CHUNK_ROWS: u32;
var<private> ROW_START: u32;
const LANES: u32 = ${E}u;
var<private> GROUPS: u32;
var<private> GROUP_BASE: u32;
const TOTAL_GROUPS: u32 = ${p}u;
const PARTIAL_WORDS: u32 = ${O}u;
var<workgroup> sharedA: array<f32, ${E}>;
var<workgroup> sharedB: array<f32, ${E}>;
var<workgroup> sharedC: array<f32, ${E}>;
`,T=e=>e.replaceAll(`%ROW_OK%`,`${a.mask?`mask[maskOffset + row] == 0u || `:``}!isFiniteFloat(values[valuesOffset + row])`).replaceAll(`%NEIGHBOR_OK%`,`neighbor >= ROWS || neighbor == row || !(weights[weightsOffset + slot] > 0.0) || ${a.mask?`mask[maskOffset + neighbor] == 0u || `:``}!isFiniteFloat(secondValues[secondValuesOffset + neighbor])`),k=`workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sharedA[lane] += sharedA[lane + stride];
      sharedB[lane] += sharedB[lane + stride];
      sharedC[lane] += sharedC[lane + stride];
    }
    workgroupBarrier();
  }`,A=e=>`{
  var accumulatedA = 0.0;
  var accumulatedB = 0.0;
  var accumulatedC = 0.0;
  for (var partialGroup = lane; partialGroup < TOTAL_GROUPS; partialGroup += LANES) {
    accumulatedA += partials[partialsOffset + partialGroup * PARTIAL_WORDS + ${e}u];
    accumulatedB += partials[partialsOffset + partialGroup * PARTIAL_WORDS + ${e+1}u];
    accumulatedC += partials[partialsOffset + partialGroup * PARTIAL_WORDS + ${e+2}u];
  }
  sharedA[lane] = accumulatedA;
  sharedB[lane] = accumulatedB;
  sharedC[lane] = accumulatedC;
  ${k}
  }`,j=e=>({name:`partials`,view:y,type:`f32`,access:e}),M=0,N=c.map((e,t)=>{let n={dispatch:e,groups:d[t],groupBase:M};return M+=n.groups,n}),P=N.map(({dispatch:e,groups:t,groupBase:r})=>_(n,{id:`${i}-moments${e.idSuffix}`,operation:S,variant:`${s}-moments`,workgroupSize:E,invocationCount:t*E,parameterValues:[e.rowCount,e.rowStart,t,r],bindings:[...b(e),j(`read_write`)],declarations:w(),body:T(`CHUNK_ROWS = getChunkParameter(36u);
  ROW_START = getChunkParameter(37u);
  GROUPS = getChunkParameter(38u);
  GROUP_BASE = getChunkParameter(39u);
  let lane = localInvocationIndex;
  let group = GROUP_BASE + index / LANES;
  var laneWeight = 0.0;
  var laneX = 0.0;
  var laneY = 0.0;
  for (var localRow = index; localRow < CHUNK_ROWS; localRow += GROUPS * LANES) {
    let row = ROW_START + localRow;
    if (%ROW_OK%) { continue; }
    for (var slot = offsets[offsetsOffset + localRow]; slot < offsets[offsetsOffset + localRow + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (%NEIGHBOR_OK%) { continue; }
      let weight = weights[weightsOffset + slot];
      laneWeight += weight;
      laneX += weight * values[valuesOffset + row];
      laneY += weight * secondValues[secondValuesOffset + neighbor];
    }
  }
  sharedA[lane] = laneWeight;
  sharedB[lane] = laneX;
  sharedC[lane] = laneY;
  ${k}
  if (lane == 0u) {
    partials[partialsOffset + group * PARTIAL_WORDS] = sharedA[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 1u] = sharedB[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 2u] = sharedC[0];
  }`)})),F=N.map(({dispatch:e,groups:t,groupBase:r})=>_(n,{id:`${i}-centered${e.idSuffix}`,operation:S,variant:`${s}-centered`,workgroupSize:E,invocationCount:t*E,parameterValues:[e.rowCount,e.rowStart,t,r],bindings:[...b(e),j(`read_write`)],declarations:w(),body:T(`CHUNK_ROWS = getChunkParameter(36u);
  ROW_START = getChunkParameter(37u);
  GROUPS = getChunkParameter(38u);
  GROUP_BASE = getChunkParameter(39u);
  let lane = localInvocationIndex;
  let group = GROUP_BASE + index / LANES;
  ${A(0)}
  let weightTotal = sharedA[0];
  let meanX = sharedB[0] / weightTotal;
  let meanY = sharedC[0] / weightTotal;
  workgroupBarrier();
  var laneNumerator = 0.0;
  var laneSquareX = 0.0;
  var laneSquareY = 0.0;
  for (var localRow = index; localRow < CHUNK_ROWS; localRow += GROUPS * LANES) {
    let row = ROW_START + localRow;
    if (%ROW_OK%) { continue; }
    for (var slot = offsets[offsetsOffset + localRow]; slot < offsets[offsetsOffset + localRow + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (%NEIGHBOR_OK%) { continue; }
      let weight = weights[weightsOffset + slot];
      let x = values[valuesOffset + row];
      let y = secondValues[secondValuesOffset + neighbor];
      laneNumerator += weight * ${x};
      laneSquareX += weight * (x - meanX) * (x - meanX);
      laneSquareY += weight * (y - meanY) * (y - meanY);
    }
  }
  sharedA[lane] = laneNumerator;
  sharedB[lane] = laneSquareX;
  sharedC[lane] = laneSquareY;
  ${k}
  if (lane == 0u) {
    partials[partialsOffset + group * PARTIAL_WORDS + 3u] = sharedA[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 4u] = sharedB[0];
    partials[partialsOffset + group * PARTIAL_WORDS + 5u] = sharedC[0];
  }`)})),I=r(n,{id:`${i}-statistic`,operation:S,variant:s,workgroupSize:E,invocationCount:E,guardIndex:!1,bindings:[j(`read`),{name:`result`,view:a.output.statistic,type:`f32`,access:`read_write`}],declarations:w(),body:`let lane = localInvocationIndex;
  ${A(0)}
  let weightTotal = sharedA[0];
  workgroupBarrier();
  ${A(3)}
  if (lane == 0u) {
    let numerator = sharedA[0];
    let squareX = sharedB[0];
    let squareY = sharedC[0];
    var answer = getQuietNaN(0u);
    ${s===`pearson`?`if (weightTotal > 0.0 && squareX > 0.0 && squareY > 0.0) { answer = clamp(numerator / sqrt(squareX * squareY), -1.0, 1.0); }`:`if (weightTotal > 0.0) { answer = numerator / weightTotal; }`}
    result[resultOffset] = answer;
  }`});return[...v===h?h.nodes:[...h.nodes,...v.nodes],...P,...F,I]}export{k as n,A as r,j as t};