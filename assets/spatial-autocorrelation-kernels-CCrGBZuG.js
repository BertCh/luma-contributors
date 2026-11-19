import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./statistics-kernel-node-lacKNERm.js";import{t as r}from"./spatial-statistics-counts-wgsl-B8N7XLDs.js";import{r as i,s as a}from"./spatial-weights-CqWysstN.js";import{i as o,s,t as c}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{t as l}from"./parameter-schema-BEeBYH77.js";import{E as u,T as d,f,v as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m,l as h,n as g}from"./wgsl-common-DfyXiYB0.js";import{a as _}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as v}from"./graph-chunk-groups-BB8ACpQO.js";import{t as y}from"./spatial-statistics-row-dispatch-DNaskhxv.js";import{n as b,t as x}from"./float32-values-60_XMjYS.js";var S=4294967295,C=2**24,w=l({id:`spatial-autocorrelation`,format:`float32`,wordLength:8,fields:[{name:`significanceLevel`,format:`float32`,wordOffset:0,defaultValue:.05,minimum:Number.MIN_VALUE,maximum:1,dynamic:!0},{name:`useFixedMoments`,format:`float32`,wordOffset:1,defaultValue:0,minimum:0,maximum:1,dynamic:!0},{name:`fixedCount`,format:`float32`,wordOffset:2,defaultValue:0,minimum:0,dynamic:!0},{name:`fixedMean`,format:`float32`,wordOffset:3,defaultValue:0,dynamic:!0},{name:`fixedVariance`,format:`float32`,wordOffset:4,defaultValue:0,minimum:0,dynamic:!0},{name:`fixedCountHigh`,format:`float32`,wordOffset:5,defaultValue:0,minimum:0,maximum:65535,dynamic:!0},{name:`fixedCountLow`,format:`float32`,wordOffset:6,defaultValue:0,minimum:0,maximum:65535,dynamic:!0},{name:`useExtendedFixedCount`,format:`float32`,wordOffset:7,defaultValue:0,minimum:0,maximum:1,dynamic:!0}]}),T=[1.6448536269514722,1.959963984540054,2.5758293035489004],E=[.1,.05,.01],D={NOT_SIGNIFICANT:0,HIGH_HIGH:1,LOW_HIGH:2,LOW_LOW:3,HIGH_LOW:4};function O(e={},t=new Float32Array(8)){f(`getGPUSpatialAutocorrelationParameterValues`,`target length`,t.length,8);let n=e.significanceLevel??.05;if(!(Math.fround(n)>0&&Math.fround(n)<1))throw Error(`Spatial autocorrelation significanceLevel must be in (0, 1)`);let r=e.fixedMoments;if(r&&(f(`spatial-autocorrelation`,`fixedMoments.count`,r.count,2,S),!Number.isFinite(r.count)||!Number.isFinite(Math.fround(r.mean))||!Number.isFinite(Math.fround(r.variance))||r.count<2||Math.fround(r.variance)<=0))throw Error(`Spatial autocorrelation fixedMoments are invalid`);let i=!!(r&&r.count>C);return t.set([n,+!!r,r?.count??0,r?.mean??0,r?.variance??0,i?Math.floor(r.count/65536):0,i?r.count%65536:0,+!!i]),t}var k=4096,A=4294963199,j=256,M=0,N=1,P=2;function F(e,t){let n=t??1;return f(e,`minimumNeighbors`,n),n}function I(e){let{id:n}=e,r=s(n,e.weights);if(`chunks`in e.weights)for(let[t,r]of e.weights.chunks.entries())d(n,!r.weights.status?.overflow&&!r.weights.status?.candidateOverflow,`weights.chunks[${t}].weights.status loss flags must be aggregated on the outer weights.status port`);if(b(e.values,`${n} values`),u(e.parameters,[`float32`],`${n} parameters`),e.islands!==void 0&&a(n,e.islands),e.weights.status?.overflow&&t(e.weights.status.overflow,`${n} weights.status.overflow`),e.weights.status?.candidateOverflow&&(u(e.weights.status.candidateOverflow,[`uint32`],`${n} weights.status.candidateOverflow`),f(n,`weights.status.candidateOverflow length`,e.weights.status.candidateOverflow.length,1)),f(n,`weights row count`,r,1,A),e.values.length!==r)throw Error(`${n} values length must equal the weights row count`);if(f(n,`parameters length`,e.parameters.length,8),e.mask&&(t(e.mask,`${n} mask`),e.mask.length!==r))throw Error(`${n} mask length must equal the weights row count`);return e.globalCounts&&(u(e.globalCounts,[`uint32`],`${n} output.globalCounts`),f(n,`output.globalCounts length`,e.globalCounts.length,2,2)),e.globalStatistics&&(u(e.globalStatistics,[`float32`],`${n} output.globalStatistics`),f(n,`output.globalStatistics length`,e.globalStatistics.length,5)),r}function L(e,t,n=[]){v(e,t.id,8),_(e,t.id,Object.fromEntries([...o(t.weights),t.weights.status?.overflow,t.weights.status?.candidateOverflow,t.values,t.parameters,t.mask,t.globalStatistics,t.globalCounts,...n].map((e,t)=>[`binding ${t}`,e]))),p(e.device,t.id,`statistics scratch`,(t.values.length+6)*4,`reduce packed values rows or request larger device bindings`)}var R=`
${h}
${r}

// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
${m}

// Two-sided normal p-value erfc(|z| / sqrt(2)) through the shared erfcValue (Numerical Recipes
// erfcc Chebyshev fit, fractional error below 1.2e-7), so tiny p-values keep relative accuracy.
${g}

// Floor of every analytic (normal) p-value: the smallest normal f32. f32 exp underflows beyond
// |z| of about 13, where erfc would read exactly 0; a floor keeps such p-values positive, so they
// stay ordered against larger ones in a multiple-testing correction and safe under log(). One
// rule for the global, local, hot-spot and emerging-hot-spot statistics.
const ANALYTIC_P_VALUE_FLOOR: f32 = 1.17549435e-38;

fn getTwoSidedPValue(z: f32) -> f32 {
  return clamp(erfcValue(abs(z) * 0.70710678118654752), ANALYTIC_P_VALUE_FLOOR, 1.0);
}
`;function z(){return`
fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

${R}
`}function B(e,t){return i(`isFiniteFloat(statistics[statisticsOffset + neighbor])`,e,`index`,t)}function V(t,r){L(t,r);let{id:a,operation:o,parameters:s,mask:l,weights:u}=r,d=x(t,`${a}-values`,r.values),f=d.view,p=f.length,m=Math.ceil(p/k),h=z(),g=[...d.nodes],_=r.islands!==void 0,v=r.islands===`keep`,b=u.status?.overflow,S=u.status?.candidateOverflow,C=c(u),w=e(t,`${a}-value-partials`,`float32`,m),T=e(t,`${a}-count-partials`,`uint32`,m),E=_?e(t,`${a}-island-partials`,`uint32`,m):void 0,D=e(t,`${a}-deviation-partials`,`float32`,m),O=e(t,`${a}-square-partials`,`float32`,m),A=e(t,`${a}-statistics`,`float32`,p+6),F=r.globalCounts??e(t,`${a}-counts`,`uint32`,2),I=_?e(t,`${a}-row-states`,`uint32`,p):void 0,B=l?[{name:`mask`,view:l,type:`u32`,access:`read`}]:[],V=`fn isRawRowValid(row: u32) -> bool {
  return ${l?`mask[maskOffset + row] != 0u && `:``}isFiniteFloat(values[valuesOffset + row]);
}`,H=I?[{name:`rowStates`,view:I,type:`u32`,access:`read`}]:B,U=`${R}
const MOMENTS: u32 = ${p}u;
const ROW_COUNT: u32 = ${p}u;
const BLOCK_ROWS: u32 = ${k}u;
const WORKGROUP_SIZE: u32 = ${j}u;
${I?`fn isRowValid(row: u32) -> bool {
  let state = rowStates[rowStatesOffset + row];
  return state == ${N}u || (state == ${P}u && ${v});
}`:V.replace(`isRawRowValid`,`isRowValid`)}`;return I&&g.push(...C.map(e=>y(t,{rowBase:e.rowStart,id:`${a}-row-states${e.idSuffix}`,operation:o,variant:`row-states`,bindings:[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:e.weights.weights,type:`f32`,access:`read`},{name:`values`,view:f,type:`f32`,access:`read`},...B,{name:`rowStates`,view:I,type:`u32`,access:`read_write`}],invocationCount:e.rowCount,declarations:`${R}
const ROWS: u32 = ${p}u;
${V}`,body:`if (!isRawRowValid(index)) {
    rowStates[rowStatesOffset + index] = ${M}u;
    return;
  }
  var hasNeighbor = false;
  ${i(`isRawRowValid(neighbor)`,`hasNeighbor = true; break;`,`index`,`WEIGHTS_ROW_START`)}
  rowStates[rowStatesOffset + index] = select(${P}u, ${N}u, hasNeighbor);`}))),g.push(n(t,{id:`${a}-moments-blocks`,operation:o,variant:`moments-blocks`,bindings:[{name:`values`,view:f,type:`f32`,access:`read`},...H,{name:`valuePartials`,view:w,type:`f32`,access:`read_write`},{name:`countPartials`,view:T,type:`u32`,access:`read_write`},...E?[{name:`islandPartials`,view:E,type:`u32`,access:`read_write`}]:[]],workgroupSize:j,invocationCount:m*j,guardIndex:!1,declarations:`${U}
var<workgroup> partialSums: array<f32, ${j}>;
var<workgroup> partialCounts: array<u32, ${j}>;
var<workgroup> partialIslands: array<u32, ${j}>;`,body:`let blockIndex = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  var islands = 0u;
  if (isInRange) {
    let end = min((blockIndex + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = blockIndex * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      if (isRowValid(row)) {
        sum += values[valuesOffset + row];
        count++;
      }
      ${I?`islands += select(0u, 1u, rowStates[rowStatesOffset + row] == ${P}u);`:``}
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  partialIslands[localInvocationIndex] = islands;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialIslands[localInvocationIndex] += partialIslands[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    valuePartials[valuePartialsOffset + blockIndex] = partialSums[0];
    countPartials[countPartialsOffset + blockIndex] = partialCounts[0];
    ${E?`islandPartials[islandPartialsOffset + blockIndex] = partialIslands[0];`:``}
  }`}),n(t,{id:`${a}-moments-mean`,operation:o,variant:`moments-mean`,bindings:[{name:`valuePartials`,view:w,type:`f32`,access:`read`},{name:`countPartials`,view:T,type:`u32`,access:`read`},...E?[{name:`islandPartials`,view:E,type:`u32`,access:`read`}]:[],{name:`statistics`,view:A,type:`f32`,access:`read_write`},{name:`counts`,view:F,type:`u32`,access:`read_write`}],workgroupSize:j,invocationCount:j,guardIndex:!1,declarations:`${R}
const MOMENTS: u32 = ${p}u;
const BLOCK_COUNT: u32 = ${m}u;
const WORKGROUP_SIZE: u32 = ${j}u;
var<workgroup> partialSums: array<f32, ${j}>;
var<workgroup> partialCounts: array<u32, ${j}>;
var<workgroup> partialIslands: array<u32, ${j}>;`,body:`var sum = 0.0;
  var count = 0u;
  var islands = 0u;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += WORKGROUP_SIZE) {
    sum += valuePartials[valuePartialsOffset + blockIndex];
    count += countPartials[countPartialsOffset + blockIndex];
    ${E?`islands += islandPartials[islandPartialsOffset + blockIndex];`:``}
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  partialIslands[localInvocationIndex] = islands;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
      partialIslands[localInvocationIndex] += partialIslands[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    // The count is exact in u32 and converted to f32 only here. The mean is provisional: the
    // deviation pass corrects it.
    counts[countsOffset] = partialCounts[0];
    counts[countsOffset + 1u] = partialIslands[0];
    let total = f32(partialCounts[0]);
    statistics[statisticsOffset + MOMENTS] = total;
    statistics[statisticsOffset + MOMENTS + 1u] = select(getQuietNaN(index), divideBySpatialCount(partialSums[0], partialCounts[0]), total >= 1.0);
    statistics[statisticsOffset + MOMENTS + 4u] = f32(partialIslands[0]);
  }`}),n(t,{id:`${a}-moments-deviations`,operation:o,variant:`moments-deviations`,bindings:[{name:`values`,view:f,type:`f32`,access:`read`},...H,{name:`statistics`,view:A,type:`f32`,access:`read_write`},{name:`deviationPartials`,view:D,type:`f32`,access:`read_write`}],workgroupSize:j,invocationCount:m*j,guardIndex:!1,declarations:`${U}
var<workgroup> partialSums: array<f32, ${j}>;`,body:`let blockIndex = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  if (isInRange) {
    let provisionalMean = statistics[statisticsOffset + MOMENTS + 1u];
    let end = min((blockIndex + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = blockIndex * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      let valid = isRowValid(row);
      let deviation = values[valuesOffset + row] - provisionalMean;
      statistics[statisticsOffset + row] = select(getQuietNaN(row), deviation, valid);
      sum += select(0.0, deviation, valid);
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    deviationPartials[deviationPartialsOffset + blockIndex] = partialSums[0];
  }`}),n(t,{id:`${a}-moments-correct`,operation:o,variant:`moments-correct`,bindings:[{name:`deviationPartials`,view:D,type:`f32`,access:`read`},{name:`parameters`,view:s,type:`f32`,access:`read`},{name:`statistics`,view:A,type:`f32`,access:`read_write`},{name:`counts`,view:F,type:`u32`,access:`read_write`},...b?[{name:`overflow`,view:b,type:`u32`,access:`read`}]:[],...S?[{name:`candidateOverflow`,view:S,type:`u32`,access:`read`}]:[]],workgroupSize:j,invocationCount:j,guardIndex:!1,declarations:`${h}
const MOMENTS: u32 = ${p}u;
const BLOCK_COUNT: u32 = ${m}u;
const WORKGROUP_SIZE: u32 = ${j}u;
var<workgroup> partialSums: array<f32, ${j}>;`,body:`var sum = 0.0;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += WORKGROUP_SIZE) {
    sum += deviationPartials[deviationPartialsOffset + blockIndex];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    var total = statistics[statisticsOffset + MOMENTS];
    var mean = statistics[statisticsOffset + MOMENTS + 1u];
    // Mean of the first-pass deviations: what the center pass subtracts from each stored deviation.
    var residualMean = 0.0;
    if (total >= 1.0) {
      residualMean = divideBySpatialCount(partialSums[0], counts[countsOffset]);
    }
    if (readParameter(1u) != 0.0) {
      // Fixed moments: center on the given mean, again through the stored deviations.
      residualMean = readParameter(3u) - mean;
      let packedCount = (u32(readParameter(5u)) << 16u) | u32(readParameter(6u));
      if (readParameter(7u) == 1.0) { counts[countsOffset] = packedCount; }
      else { counts[countsOffset] = u32(readParameter(2u)); }
      total = f32(counts[countsOffset]);
      mean = readParameter(3u);
    } else {
      mean += residualMean;
    }
    ${b||S?`// Incomplete weights: suppress statistics and inference.
    if (${[b?`overflow[overflowOffset] != 0u`:void 0,S?`candidateOverflow[candidateOverflowOffset] != 0u`:void 0].filter(Boolean).join(` || `)}) {
      mean = getQuietNaN(index);
      residualMean = getQuietNaN(index);
    }`:``}
    statistics[statisticsOffset + MOMENTS] = total;
    statistics[statisticsOffset + MOMENTS + 1u] = mean;
    statistics[statisticsOffset + MOMENTS + 5u] = residualMean;
  }`}),n(t,{id:`${a}-center`,operation:o,variant:`center`,bindings:[{name:`values`,view:f,type:`f32`,access:`read`},...H,{name:`statistics`,view:A,type:`f32`,access:`read_write`},{name:`squarePartials`,view:O,type:`f32`,access:`read_write`}],workgroupSize:j,invocationCount:m*j,guardIndex:!1,declarations:`${U}
var<workgroup> partialSums: array<f32, ${j}>;`,body:`let blockIndex = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  if (isInRange) {
    let residualMean = statistics[statisticsOffset + MOMENTS + 5u];
    let end = min((blockIndex + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = blockIndex * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      let valid = isRowValid(row);
      // The stored first-pass deviation minus the mean of those deviations. It is read back from
      // memory, not recomputed as (x - mean1) - mean2, which a shader compiler may reassociate to
      // x - (mean1 + mean2).
      let centered = statistics[statisticsOffset + row] - residualMean;
      // Rows outside the analysis store quiet NaN, which every later kernel reads as "not a focus
      // row and not a neighbor".
      statistics[statisticsOffset + row] = select(getQuietNaN(row), centered, valid);
      sum += select(0.0, centered * centered, valid);
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + blockIndex] = partialSums[0];
  }`}),n(t,{id:`${a}-moments-variance`,operation:o,variant:`moments-variance`,bindings:[{name:`squarePartials`,view:O,type:`f32`,access:`read`},{name:`parameters`,view:s,type:`f32`,access:`read`},{name:`statistics`,view:A,type:`f32`,access:`read_write`},{name:`counts`,view:F,type:`u32`,access:`read_write`},...r.globalStatistics?[{name:`globalStatistics`,view:r.globalStatistics,type:`f32`,access:`read_write`}]:[]],workgroupSize:j,invocationCount:j,guardIndex:!1,declarations:`${h}
const MOMENTS: u32 = ${p}u;
const BLOCK_COUNT: u32 = ${m}u;
const WORKGROUP_SIZE: u32 = ${j}u;
var<workgroup> partialSums: array<f32, ${j}>;`,body:`var sum = 0.0;
  for (var blockIndex = localInvocationIndex; blockIndex < BLOCK_COUNT; blockIndex += WORKGROUP_SIZE) {
    sum += squarePartials[squarePartialsOffset + blockIndex];
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let count = statistics[statisticsOffset + MOMENTS];
    let mean = statistics[statisticsOffset + MOMENTS + 1u];
    var sumOfSquares = partialSums[0];
    var variance = select(getQuietNaN(index), divideBySpatialCount(sumOfSquares, counts[countsOffset]), count >= 1.0);
    if (readParameter(1u) != 0.0) {
      variance = readParameter(4u);
      sumOfSquares = multiplyBySpatialCount(variance, counts[countsOffset]);
    }
    statistics[statisticsOffset + MOMENTS + 2u] = variance;
    statistics[statisticsOffset + MOMENTS + 3u] = sumOfSquares;
    ${r.globalStatistics?`globalStatistics[globalStatisticsOffset] = count;
    globalStatistics[globalStatisticsOffset + 1u] = mean;
    globalStatistics[globalStatisticsOffset + 2u] = variance;
    globalStatistics[globalStatisticsOffset + 3u] = sqrt(variance);
    globalStatistics[globalStatisticsOffset + 4u] = statistics[statisticsOffset + MOMENTS + 4u];`:``}
  }`})),{nodes:g,statistics:A,counts:F}}export{F as a,T as c,w as d,O as f,B as i,E as l,V as n,L as o,z as r,I as s,R as t,D as u};