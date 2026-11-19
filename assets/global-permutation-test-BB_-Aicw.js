import{c as e,d as t,i as n,l as r,n as i,o as a,r as o,s,u as c}from"./permutation-inference-kernels-COguC6vP.js";import{r as l,u}from"./kernel-C3tfQOn4.js";import{a as d,t as f}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as p}from"./statistics-kernel-node-lacKNERm.js";import{E as m,S as h,f as g,p as _,v,x as y,y as b}from"./gpu-contributor-utils-BIwLQON5.js";import{r as x,s as S}from"./spatial-weights-CqWysstN.js";import{i as ee,t as C}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{i as w}from"./graph-chunk-dispatch-zBt8k2-o.js";import{t as te}from"./spatial-statistics-collective-node-BlCi_ESE.js";import{t as ne}from"./graph-chunk-groups-BB8ACpQO.js";import{t as re}from"./permutation-random-SiwCiDNI.js";import{t as ie}from"./permutation-shuffle-zu7WXW69.js";import{t as ae}from"./philox-random-DfVxWNPn.js";import{i as T,n as oe,r as E,t as se}from"./permutation-alternative-B1Iqm_lI.js";import{t as D}from"./define-run-B-brkCe4.js";import{l as O,s as k,u as A}from"./pointer-overlay-Qh8qflcA.js";import{m as j}from"./inputs-CJKKIRKK.js";import{n as ce,t as le}from"./permutation-chart-dNDRNqOt.js";import{a as ue,m as M,o as N}from"./point-field-CLZ0m0cx.js";import{Bt as de}from"./entry-preview-D70TdSbK.js";var P=`GPUGlobalPermutationTest`,F=256,I=2048,L=256,R=256,z=128,B=64,fe=2**25,pe=4096,me=96,V={base:4,greater:5,lesser:6,stoppedAt:7,done:8,length:12};function H(e,t){let n=Math.ceil(e/I);return Math.min(t,Math.max(Math.ceil(z/n),Math.ceil(t/B),1))}var U=Math.floor((2**32-1)/L),W={observed:0,pseudoPValue:1,exceedances:2,permutationCount:3,simulatedMean:4,simulatedStandardDeviation:5,zSimulated:6,pZSimulated:7,minimum:8,maximum:9,islandCount:10,analysisCount:11,length:12};function he(e){return C(e).reduce((e,t)=>e+Math.ceil(t.rowCount/I),0)}var G=class{id;props;estimatedCommandNodeCount;constructor(e){let t=e.id??`global-permutation-test`;if(this.id=t,this.props=e,c({...e,id:t,operation:P,analysisCount:e.output.analysisCount,islandCount:e.output.islandCount}),![`moran`,`geary`,`getisOrdG`,`bivariateMoran`,`leesL`].includes(e.statistic))throw Error(`${t} unknown statistic ${e.statistic}`);if(T(t,e.stopAfterExceedances,E(t,e.alternative),e.maximumPermutations),S(t,e.islands),e.permutationBatchSize!==void 0&&(g(t,`permutationBatchSize`,e.permutationBatchSize,1,e.maximumPermutations+1),h(t,`permutationBatchSize x rows`,e.permutationBatchSize*e.values.length,fe),h(t,`permutation batches`,Math.ceil((e.maximumPermutations+ +(e.stopAfterExceedances===void 0))/e.permutationBatchSize),pe)),(e.statistic===`bivariateMoran`||e.statistic===`leesL`)&&!e.secondValues)throw Error(`${t} ${e.statistic} requires secondValues`);h(t,`permutations x row blocks`,(e.maximumPermutations+1)*he(e.weights),U);let n=C(e.weights).length,r=e.stopAfterExceedances!==void 0,i=e.permutationBatchSize??(r?H(e.values.length,e.maximumPermutations):e.maximumPermutations+1),a=Math.ceil((e.maximumPermutations+ +!r)/i);this.estimatedCommandNodeCount=me+3*n+a*(n+3),_(t,this.estimatedCommandNodeCount);let{output:o}=e;if(o.status?.invalidCount&&(u(o.status.invalidCount,`${t} output.status.invalidCount`),g(t,`output.status.invalidCount length`,o.status.invalidCount.length)),m(o.summary,[`float32`],`${t} output.summary`),g(t,`output.summary length`,o.summary.length,W.length),o.referenceDistribution&&(m(o.referenceDistribution,[`float32`],`${t} output.referenceDistribution`),o.referenceDistribution.length!==e.maximumPermutations))throw Error(`${t} referenceDistribution length must equal maximumPermutations`);o.histogram&&(u(o.histogram,`${t} output.histogram`),g(t,`output.histogram length`,o.histogram.length)),w(t,{...e.output,status:void 0,...e.output.status}),b(t,[e.output.summary,e.output.analysisCount,e.output.islandCount,e.output.referenceDistribution,e.output.histogram,e.output.status?.invalidCount],[...ee(e.weights),e.values,e.secondValues,e.mask,e.parameters,e.weights.status?.overflow])}getParameterValues(e){return t(e,{maximumPermutations:this.props.maximumPermutations})}getInferenceMetadata(e){this.getParameterValues(e);let t=S(this.id,this.props.islands)===`keep`;return{alternative:E(this.id,this.props.alternative),permutationCount:e.permutationCount,seed:e.seed,multipleTesting:`none`,testedRowsDefinition:`one global statistic over the analysis rows (nonzero mask, finite values; islands, `+(t?`rows without a positive-weight neighbor, are kept in n)`:`rows without a positive-weight neighbor, are excluded)`)}}getCommandNodes(t){let{id:c,props:u}=this,{weights:d,parameters:m,statistic:h,maximumPermutations:g}=u,_=C(d);y(c,t,[...ee(d),u.values,u.secondValues,u.mask,m,d.status?.overflow,u.output.summary,u.output.analysisCount,u.output.islandCount,u.output.referenceDistribution,u.output.histogram,u.output.status?.invalidCount]),u.permutationBatchSize!==void 0&&v(t.device,c,`materialized permutation values`,u.permutationBatchSize*u.values.length*Float32Array.BYTES_PER_ELEMENT,`reduce permutationBatchSize or rows`),r(t,{...u,id:c,operation:P},[u.output.summary,u.output.analysisCount,u.output.islandCount,u.output.referenceDistribution,u.output.histogram,u.output.status?.invalidCount]),v(t.device,c,`pair partials scratch`,(g+1)*he(d)*4,`reduce maximumPermutations or packed rows`);let b=u.values.length,S=h===`leesL`,w=h===`bivariateMoran`||S,T=E(c,u.alternative),D=se(T),O=s(t,{...u,analysisCount:u.output.analysisCount,islandCount:u.output.islandCount,secondValues:w?u.secondValues:void 0,islands:u.islands,id:`${c}-inputs`,operation:P,centerX:h!==`getisOrdG`}),k=O.nodes,{rowPositions:A,compactX:j,compactY:ce,totals:le,islandCount:ue}=O,M=he(d),N=g+1,de=l(t,`${c}-partials`,`float32`,N*M),R=l(t,`${c}-pair-sums`,`float32`,N),z=u.permutationBatchSize,B=z!==void 0,fe=w?`compactY`:`compactX`,pe=S?B?`lag += weight * permutedX[permutedXOffset + slotBase + neighborPosition];
        lagY += weight * permutedY[permutedYOffset + slotBase + neighborPosition];`:`let permutedNeighbor = getPermutedPosition(neighborPosition, permutation, count);
        lag += weight * compactX[compactXOffset + permutedNeighbor];
        lagY += weight * compactY[compactYOffset + permutedNeighbor];`:h===`geary`?`let difference = focus - neighborValue;
        lag += weight * difference * difference;`:`lag += weight * neighborValue;`,me=S?`partialSum += lag * lagY;`:h===`geary`?`partialSum += lag;`:`partialSum += focus * lag;`,H=u.stopAfterExceedances,U=H!==void 0,G=U||B?l(t,`${c}-batch-state`,`uint32`,V.length):void 0,K=G??m,ge=B&&h!==`bivariateMoran`,_e=B&&w,q=ge?l(t,`${c}-permuted-x`,`float32`,z*b):void 0,J=_e?l(t,`${c}-permuted-y`,`float32`,z*b):void 0,ve=[{name:`rowPositions`,view:A,type:`u32`,access:`read`},...B?[...h===`bivariateMoran`?[{name:`compactX`,view:j,type:`f32`,access:`read`}]:[],...q?[{name:`permutedX`,view:q,type:`f32`,access:`read`}]:[],...J?[{name:`permutedY`,view:J,type:`f32`,access:`read`}]:[]]:[{name:`compactX`,view:j,type:`f32`,access:`read`},...w?[{name:`compactY`,view:ce,type:`f32`,access:`read`}]:[]],{name:`parameters`,view:K,type:`u32`,access:`read`},{name:`partials`,view:de,type:`f32`,access:`read_write`}],Y=_.map(e=>Math.ceil(e.rowCount/I)),ye=(n,r=_[0])=>{let i=Y[r.chunkIndex],a=Y.slice(0,r.chunkIndex).reduce((e,t)=>e+t,0),s=r.isWhole?n.invocationCount:n.invocationCount/M*i;return te(t,{id:`${n.id}${r.idSuffix}`,operation:P,variant:n.variant,bindings:[{name:`offsets`,view:r.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:O.chunks[r.chunkIndex].effective,type:`f32`,access:`read`},...ve],workgroupSize:L,invocationCount:s,parameterValues:[r.rowStart,r.rowCount,i,a],...n.gate,declarations:`const ROWS: u32 = ${b}u;
var<private> CAPACITY: u32;
var<private> WEIGHTS_ROW_START: u32;
var<private> CHUNK_ROW_COUNT: u32;
var<private> BLOCK_BASE: u32;
const PARTIAL_BLOCK_COUNT: u32 = ${M}u;
const INVALID_POSITION: u32 = ${o}u;
const BLOCK_ROWS: u32 = ${I}u;
var<private> BLOCK_COUNT: u32;
const WORKGROUP_SIZE: u32 = ${L}u;
var<workgroup> partialSums: array<f32, ${L}>;
${e(g)}${B?``:`
${re}
${ie(ae.permutationShuffle.id,198)}
fn getPermutedPosition(position: u32, permutation: u32, count: u32) -> u32 {
  if (permutation == 0u) {
    return position;
  }
  return getSwapOrNotPermutationIndex(position, count);
}`}`,body:`CAPACITY = arrayLength(&neighbors) - neighborsOffset;
  WEIGHTS_ROW_START = getChunkParameter(36u);
  CHUNK_ROW_COUNT = getChunkParameter(37u);
  BLOCK_COUNT = getChunkParameter(38u);
  BLOCK_BASE = getChunkParameter(39u);
  let permutationWorkgroupIndex = index / WORKGROUP_SIZE;
  let permutation = ${n.permutationWGSL};
  let permutationBlockIndex = permutationWorkgroupIndex % BLOCK_COUNT;
  let count = rowPositions[rowPositionsOffset + ROWS];
  let runsPermutation = index < INVOCATION_COUNT && permutation <= readValidPermutationCount();${B?`
  // Batch slot of this workgroup's permutation inside the materialized buffers.
  let slotBase = (permutationWorkgroupIndex / BLOCK_COUNT) * ROWS;`:`
  if (runsPermutation && permutation > 0u) {
    loadSwapOrNotRounds(readSeedKey(), permutation, count, localInvocationIndex, WORKGROUP_SIZE);
  }`}
  workgroupBarrier();
  var partialSum = 0.0;
  if (runsPermutation) {
    let chunkRowStart = WEIGHTS_ROW_START;
    let rowEnd = chunkRowStart + min((permutationBlockIndex + 1u) * BLOCK_ROWS, CHUNK_ROW_COUNT);
    for (var row = chunkRowStart + permutationBlockIndex * BLOCK_ROWS + localInvocationIndex; row < rowEnd; row += WORKGROUP_SIZE) {
      let position = rowPositions[rowPositionsOffset + row];
      if (position == INVALID_POSITION) {
        continue;
      }
      // Lee's L has no focus term: it is the product of the two lags.
      let focus = ${S?`0.0`:B?w?`compactX[compactXOffset + position]`:`permutedX[permutedXOffset + slotBase + position]`:`compactX[compactXOffset + ${w?`position`:`getPermutedPosition(position, permutation, count)`}]`};
      var lag = 0.0;
      var lagY = 0.0;
      ${x(`rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION`,B?`let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
        let neighborValue = ${w?`permutedY`:`permutedX`}[${w?`permutedY`:`permutedX`}Offset + slotBase + neighborPosition];
        ${pe}`:`let neighborPosition = rowPositions[rowPositionsOffset + neighbor];
        let neighborValue = ${fe}[${fe}Offset +
          getPermutedPosition(neighborPosition, permutation, count)];
        ${pe}`,`row`,`WEIGHTS_ROW_START`)}
      ${me}
    }
  }
  partialSums[localInvocationIndex] = partialSum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (index < INVOCATION_COUNT && localInvocationIndex == 0u${U||B?` && permutation <= MAXIMUM_PERMUTATIONS`:``}) {
    partials[partialsOffset + permutation * PARTIAL_BLOCK_COUNT + BLOCK_BASE + permutationBlockIndex] = partialSums[0];
  }`})},be=e=>_.map(t=>ye({id:e.id,variant:e.variant,invocationCount:e.invocationCount,permutationWGSL:e.permutationWGSL,gate:e.getGate?.(t)},t)),xe=n=>ne(t,{id:n.id,operation:P,variant:n.variant,bindings:[{name:`partials`,view:de,type:`f32`,access:`read`},{name:`pairSums`,view:R,type:`f32`,access:`read_write`},...n.permutationWGSL?[{name:`parameters`,view:K,type:`u32`,access:`read`}]:[]],invocationCount:n.invocationCount,...n.gate,declarations:`const BLOCK_COUNT: u32 = ${M}u;${n.permutationWGSL?`\n${e(g)}`:``}`,body:n.permutationWGSL?`let permutation = ${n.permutationWGSL};
  if (permutation > readValidPermutationCount()) {
    return;
  }
  var sum = 0.0;
  for (var permutationBlockIndex = 0u; permutationBlockIndex < BLOCK_COUNT; permutationBlockIndex++) {
    sum += partials[partialsOffset + permutation * BLOCK_COUNT + permutationBlockIndex];
  }
  pairSums[pairSumsOffset + permutation] = sum;`:`var sum = 0.0;
  for (var permutationBlockIndex = 0u; permutationBlockIndex < BLOCK_COUNT; permutationBlockIndex++) {
    sum += partials[partialsOffset + index * BLOCK_COUNT + permutationBlockIndex];
  }
  pairSums[pairSumsOffset + index] = sum;`});B&&G&&z!==void 0?k.push(...this.getMaterializedBatchNodes(t,{batchState:G,batchSize:z,pairSums:R,createPairsNodes:be,createPairSumsNode:xe,blockCount:M,compactX:j,compactY:w?ce:void 0,rowPositions:A,permutedX:q,permutedY:J})):!U||!G?k.push(...be({id:`${c}-pairs`,variant:`pairs-${h}`,invocationCount:N*M*L,permutationWGSL:`permutationWorkgroupIndex / BLOCK_COUNT`}),xe({id:`${c}-pair-sums`,variant:`pair-sums`,invocationCount:N})):k.push(...this.getSequentialBatchNodes(t,{batchState:G,createPairsNodes:be,createPairSumsNode:xe,pairSums:R,blockCount:M,alternative:T,stopAfter:H}));let X=n,Z=h===`getisOrdG`?a(t,{id:c,operation:P,rows:b,compactX:j,rowPositions:A}):void 0;Z&&k.push(...Z.nodes);let Se={moran:`divideSpatialCountByValue(exactCount, totals[totalsOffset + ${X.s0}u] * totals[totalsOffset + ${X.sumSquares}u])`,geary:`divideSpatialCountByValue(max(exactCount, 1u) - 1u, 2.0 * totals[totalsOffset + ${X.s0}u] * totals[totalsOffset + ${X.sumSquares}u])`,getisOrdG:`1.0 / (multiplyBySpatialCount(multiplyBySpatialCount(totals[totalsOffset + ${X.mean}u] * totals[totalsOffset + ${X.mean}u], exactCount), max(exactCount, 1u) - 1u) - totals[totalsOffset + ${X.sumSquares}u])`,bivariateMoran:`divideSpatialCountByValue(exactCount, totals[totalsOffset + ${X.s0}u] * sqrt(totals[totalsOffset + ${X.sumSquares}u] * totals[totalsOffset + ${X.sumSquaresY}u]))`,leesL:`divideSpatialCountByValue(exactCount, totals[totalsOffset + ${X.s0Squared}u] * sqrt(totals[totalsOffset + ${X.sumSquares}u] * totals[totalsOffset + ${X.sumSquaresY}u]))`}[h],Q=W,Ce=u.output.histogram?.length??0,we=[{name:`pairSums`,view:R,type:`f32`,access:`read`},{name:`totals`,view:le,type:`f32`,access:`read`},{name:`rowPositions`,view:A,type:`u32`,access:`read`},{name:`islandCount`,view:ue,type:`u32`,access:`read`},{name:`parameters`,view:K,type:`u32`,access:`read`},{name:`results`,view:u.output.summary,type:`f32`,access:`read_write`}];u.output.referenceDistribution&&we.push({name:`referenceDistribution`,view:u.output.referenceDistribution,type:`f32`,access:`read_write`});let $=u.output.status?.invalidCount;return Z?k.push(p(t,{id:`${c}-reject-negative`,operation:P,variant:`reject-negative`,bindings:[{name:`negativeCount`,view:Z.negativeCount,type:`u32`,access:`read`},{name:`totals`,view:le,type:`f32`,access:`read_write`},...$?[{name:`invalidCount`,view:$,type:`u32`,access:`read_write`}]:[]],invocationCount:1,declarations:i,body:`if (negativeCount[negativeCountOffset] > 0u) {
    totals[totalsOffset + ${X.count}u] = getQuietNaN(5u);
  }
  ${$?`invalidCount[invalidCountOffset] = negativeCount[negativeCountOffset];`:``}`})):$&&k.push(f(t,{id:`${c}-invalid-clear`,operation:P,view:$,type:`u32`,value:`0u`,componentCount:1})),u.output.histogram&&we.push({name:`histogram`,view:u.output.histogram,type:`atomic<u32>`,access:`read_write`}),k.push(p(t,{id:`${c}-finalize`,operation:P,variant:`finalize`,bindings:we,workgroupSize:F,invocationCount:F,guardIndex:!1,declarations:`${e(g)}
${i}
const ROWS: u32 = ${b}u;
const BINS: u32 = ${Ce}u;
const WORKGROUP_SIZE: u32 = ${F}u;
var<workgroup> scratchFloat: array<f32, ${F}>;
var<workgroup> scratchCount: array<u32, ${F}>;

// Each reduction ends with a barrier so the scratch can be reused; all threads get the result.
fn reduceFloat(local: u32, value: f32, mode: u32) -> f32 {
  scratchFloat[local] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      let other = scratchFloat[local + stride];
      scratchFloat[local] = select(select(scratchFloat[local] + other, max(scratchFloat[local], other), mode == 2u), min(scratchFloat[local], other), mode == 1u);
    }
    workgroupBarrier();
  }
  let result = scratchFloat[0];
  workgroupBarrier();
  return result;
}

fn reduceCount(local: u32, value: u32) -> u32 {
  scratchCount[local] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      scratchCount[local] += scratchCount[local + stride];
    }
    workgroupBarrier();
  }
  let result = scratchCount[0];
  workgroupBarrier();
  return result;
}`,body:`let local = localInvocationIndex;
  // An invalid requested count is never clamped: no permutation is summed and the outputs are NaN.
  ${U?`// A stopped run used the first L permutations; the others the requested count.
  let stoppedAt = parameters[parametersOffset + ${V.stoppedAt}u];
  let permutations = select(readValidPermutationCount(), stoppedAt, stoppedAt != 0u);`:`let permutations = readValidPermutationCount();`}
  let count = f32(rowPositions[rowPositionsOffset + ROWS]);
  let exactCount = rowPositions[rowPositionsOffset + ROWS];
  let invalidInput = ${Z?`!isFiniteFloat(totals[totalsOffset + 0u])`:`false`};
  let scale = select(${Se}, getQuietNaN(5u), invalidInput);
  let observedSum = pairSums[pairSumsOffset];
  let observed = scale * observedSum;
  var greaterPart = 0u;
  var lesserPart = 0u;
  var totalPart = 0.0;
  var minimumPart = 3.0e38;
  var maximumPart = -3.0e38;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let pairSum = pairSums[pairSumsOffset + permutation];
    let simulated = scale * pairSum;
    greaterPart += select(0u, 1u, pairSum >= observedSum);
    lesserPart += select(0u, 1u, pairSum <= observedSum);
    totalPart += simulated;
    minimumPart = min(minimumPart, simulated);
    maximumPart = max(maximumPart, simulated);
    ${u.output.referenceDistribution?`referenceDistribution[referenceDistributionOffset + permutation - 1u] = simulated;`:``}
  }
  ${u.output.referenceDistribution?`for (var permutation = permutations + 1u + local; permutation <= MAXIMUM_PERMUTATIONS; permutation += WORKGROUP_SIZE) {
    referenceDistribution[referenceDistributionOffset + permutation - 1u] = getQuietNaN(permutation);
  }`:``}
  let greater = reduceCount(local, greaterPart);
  let lesser = reduceCount(local, lesserPart);
  let total = reduceFloat(local, totalPart, 0u);
  let minimum = reduceFloat(local, minimumPart, 1u);
  let maximum = reduceFloat(local, maximumPart, 2u);
  let mean = select(getQuietNaN(2u), total / f32(permutations), permutations > 0u);
  var squaresPart = 0.0;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let deviation = scale * pairSums[pairSumsOffset + permutation] - mean;
    squaresPart += deviation * deviation;
  }
  let squares = reduceFloat(local, squaresPart, 0u);
  let standardDeviation = select(getQuietNaN(3u), sqrt(squares / f32(permutations)), permutations > 0u);
  ${T===`folded`?`var foldedPart = 0u;
  let observedDistance = abs(observed - mean);
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    foldedPart += select(0u, 1u, abs(scale * pairSums[pairSumsOffset + permutation] - mean) >= observedDistance);
  }
  let folded = reduceCount(local, foldedPart);`:``}
  ${u.output.histogram?`for (var bin = local; bin < BINS; bin += WORKGROUP_SIZE) {
    atomicStore(&histogram[histogramOffset + bin], 0u);
  }
  storageBarrier();
  workgroupBarrier();
  let range = maximum - minimum;
  for (var permutation = 1u + local; permutation <= permutations; permutation += WORKGROUP_SIZE) {
    let simulated = scale * pairSums[pairSumsOffset + permutation];
    var bin = 0u;
    if (range > 0.0 && isFiniteFloat(range)) {
      bin = min(u32(max(floor((simulated - minimum) / range * f32(BINS)), 0.0)), BINS - 1u);
    }
    if (!invalidInput) {
      atomicAdd(&histogram[histogramOffset + bin], 1u);
    }
  }`:``}
  if (local == 0u) {
    let larger = ${oe(T)};
    let zSimulated = (observed - mean) / standardDeviation;
    let validZ = standardDeviation > 0.0 && isFiniteFloat(zSimulated);
    // Tested: a valid permutation count and an observed statistic that exists (n >= 2, not truncated).
    let tested = permutations > 0u && isFiniteFloat(observed) && !invalidInput;
    results[resultsOffset + ${Q.observed}u] = select(getQuietNaN(0u), observed, isFiniteFloat(observed) && !invalidInput);
    results[resultsOffset + ${Q.pseudoPValue}u] = select(getQuietNaN(1u), ${U?`select(f32(min(${D}u * (larger + 1u), permutations + 1u)) / f32(permutations + 1u), f32(min(${D}u * ${H}u, permutations)) / f32(permutations), stoppedAt != 0u)`:`f32(min(${D}u * (larger + 1u), permutations + 1u)) / f32(permutations + 1u)`}, tested);
    results[resultsOffset + ${Q.exceedances}u] = select(getQuietNaN(1u), f32(larger), tested);
    results[resultsOffset + ${Q.permutationCount}u] = select(getQuietNaN(1u), f32(permutations), permutations > 0u);
    results[resultsOffset + ${Q.simulatedMean}u] = select(mean, getQuietNaN(2u), invalidInput);
    results[resultsOffset + ${Q.simulatedStandardDeviation}u] = select(standardDeviation, getQuietNaN(3u), invalidInput);
    results[resultsOffset + ${Q.zSimulated}u] = select(getQuietNaN(1u), zSimulated, validZ && tested);
    results[resultsOffset + ${Q.pZSimulated}u] = select(getQuietNaN(1u), max(0.5 * getTwoSidedPValue(zSimulated), ANALYTIC_P_VALUE_FLOOR), validZ && tested);
    results[resultsOffset + ${Q.minimum}u] = select(getQuietNaN(4u), minimum, permutations > 0u && !invalidInput);
    results[resultsOffset + ${Q.maximum}u] = select(getQuietNaN(4u), maximum, permutations > 0u && !invalidInput);
    results[resultsOffset + ${Q.islandCount}u] = f32(islandCount[islandCountOffset]);
    results[resultsOffset + ${Q.analysisCount}u] = f32(rowPositions[rowPositionsOffset + ROWS]);
  }`})),k}getMaterializedBatchNodes(t,n){let{id:r,props:i}=this,{batchState:a,batchSize:o,blockCount:s,rowPositions:c,compactX:l,compactY:u}=n,{permutedX:d,permutedY:f}=n,{maximumPermutations:m,parameters:h}=i,g=i.values.length,_=V,v=Math.ceil((m+1)/o),y=Math.ceil(g/I),b=[{name:`rowPositions`,view:c,type:`u32`,access:`read`},...d?[{name:`compactX`,view:l,type:`f32`,access:`read`}]:[],...f&&u?[{name:`compactY`,view:u,type:`f32`,access:`read`}]:[],{name:`parameters`,view:a,type:`u32`,access:`read`},...d?[{name:`permutedX`,view:d,type:`f32`,access:`read_write`}]:[],...f?[{name:`permutedY`,view:f,type:`f32`,access:`read_write`}]:[]],x=n=>te(t,{id:n.id,operation:P,variant:`materialize-${i.statistic}`,bindings:b,workgroupSize:L,invocationCount:n.slotCount*y*L,...n.gate,parameterValues:[n.permutationOffset],declarations:`const ROWS: u32 = ${g}u;
const BLOCK_ROWS: u32 = ${I}u;
const BLOCK_COUNT: u32 = ${y}u;
const WORKGROUP_SIZE: u32 = ${L}u;
${e(m)}
${re}
${ie(ae.permutationShuffle.id,198)}`,body:`let permutationWorkgroupIndex = index / WORKGROUP_SIZE;
  let slot = permutationWorkgroupIndex / BLOCK_COUNT;
  let permutationBlockIndex = permutationWorkgroupIndex % BLOCK_COUNT;
  let permutation = parameters[parametersOffset + ${_.base}u] + getChunkParameter(36u) + slot;
  let count = rowPositions[rowPositionsOffset + ROWS];
  let runsPermutation = index < INVOCATION_COUNT && permutation <= readValidPermutationCount();
  if (runsPermutation && permutation > 0u) {
    loadSwapOrNotRounds(readSeedKey(), permutation, count, localInvocationIndex, WORKGROUP_SIZE);
  }
  workgroupBarrier();
  if (runsPermutation) {
    let end = min((permutationBlockIndex + 1u) * BLOCK_ROWS, count);
    for (var position = permutationBlockIndex * BLOCK_ROWS + localInvocationIndex; position < end; position += WORKGROUP_SIZE) {
      let source = select(getSwapOrNotPermutationIndex(position, count), position, permutation == 0u);${d?`
      permutedX[permutedXOffset + slot * ROWS + position] = compactX[compactXOffset + source];`:``}${f?`
      permutedY[permutedYOffset + slot * ROWS + position] = compactY[compactYOffset + source];`:``}
    }
  }`});if(i.stopAfterExceedances!==void 0)return this.getSequentialBatchNodes(t,{...n,alternative:E(r,i.alternative),stopAfter:i.stopAfterExceedances,createMaterializeNode:x});let S=[p(t,{id:`${r}-materialized-init`,operation:P,variant:`materialized-init`,bindings:[{name:`parameters`,view:h,type:`u32`,access:`read`},{name:`state`,view:a,type:`u32`,access:`read_write`}],invocationCount:1,body:`for (var word = 0u; word < 4u; word++) {
    state[stateOffset + word] = parameters[parametersOffset + word];
  }
  for (var word = ${_.base}u; word < ${_.length}u; word++) {
    state[stateOffset + word] = 0u;
  }`})];for(let e=0;e<v;e++)S.push(x({id:`${r}-batch-${e}-materialize`,slotCount:o,permutationOffset:0}),...n.createPairsNodes({id:`${r}-batch-${e}-pairs`,variant:`pairs-${i.statistic}-materialized`,invocationCount:o*s*L,permutationWGSL:`parameters[parametersOffset + ${_.base}u] + permutationWorkgroupIndex / BLOCK_COUNT`}),n.createPairSumsNode({id:`${r}-batch-${e}-pair-sums`,variant:`pair-sums-materialized`,invocationCount:o,permutationWGSL:`parameters[parametersOffset + ${_.base}u] + index`}),p(t,{id:`${r}-batch-${e}-advance`,operation:P,variant:`materialized-advance`,bindings:[{name:`state`,view:a,type:`u32`,access:`read_write`}],invocationCount:1,body:`state[stateOffset + ${_.base}u] = state[stateOffset + ${_.base}u] + ${o}u;`}));return S}getSequentialBatchNodes(t,n){let{id:r,props:i}=this,{batchState:a,pairSums:o,blockCount:s,alternative:c,stopAfter:l}=n,{maximumPermutations:u,parameters:f}=i,m=i.values.length,h=i.permutationBatchSize??H(m,u),g=Math.ceil(u/h),_=Math.ceil(h/R),v=V,y=h*s*L,b=C(i.weights),x=d(t,{id:`${r}-batch-dispatch`,operation:P,shapes:[...b.map(e=>[h*Math.ceil(e.rowCount/I)*L,L]),[h],...n.createMaterializeNode?[[h*Math.ceil(m/I)*L,L]]:[]]}),S=b.length,ee=[p(t,{id:`${r}-sequential-init`,operation:P,variant:`sequential-init`,bindings:[{name:`parameters`,view:f,type:`u32`,access:`read`},{name:`state`,view:a,type:`u32`,access:`read_write`},x.binding],invocationCount:1,declarations:e(u),body:`for (var word = 0u; word < 4u; word++) {
    state[stateOffset + word] = parameters[parametersOffset + word];
  }
  for (var word = ${v.base}u; word < ${v.length}u; word++) {
    state[stateOffset + word] = 0u;
  }
  if (isPermutationCountValid()) {
    ${x.armWGSL}
  } else {
    state[stateOffset + ${v.done}u] = 1u;
    ${x.disarmWGSL}
  }`}),...n.createMaterializeNode?[n.createMaterializeNode({id:`${r}-materialize-observed`,slotCount:1,permutationOffset:0})]:[],...n.createPairsNodes({id:`${r}-pairs-observed`,variant:`pairs-${i.statistic}-observed`,invocationCount:s*L,permutationWGSL:`0u`}),n.createPairSumsNode({id:`${r}-pair-sums-observed`,variant:`pair-sums`,invocationCount:1})],w=`const BATCH_SIZE: u32 = ${h}u;
const CHUNK_LENGTH: u32 = ${_}u;
const STOP_AFTER_EXCEEDANCES: u32 = ${l}u;
const WORKGROUP_SIZE: u32 = ${R}u;
var<workgroup> chunkGreater: array<u32, ${R}>;
var<workgroup> chunkLesser: array<u32, ${R}>;
fn getExceedanceCount(greater: u32, lesser: u32) -> u32 {
  return ${oe(c)};
}`;for(let e=0;e<g;e++){let s=x.getGate(S);ee.push(...n.createMaterializeNode?[n.createMaterializeNode({id:`${r}-batch-${e}-materialize`,slotCount:h,permutationOffset:1,gate:{...x.getGate(S+1),condition:{...x.getGate(S+1).condition,id:`${r}-batch-${e}-materialize-gate`}}})]:[],...n.createPairsNodes({id:`${r}-batch-${e}-pairs`,variant:`pairs-${i.statistic}-batch`,invocationCount:y,permutationWGSL:`parameters[parametersOffset + ${v.base}u] + 1u + permutationWorkgroupIndex / BLOCK_COUNT`,getGate:t=>{let n=x.getGate(t.chunkIndex);return{...n,condition:{...n.condition,id:`${r}-batch-${e}-pairs-gate${t.idSuffix}`}}}}),n.createPairSumsNode({id:`${r}-batch-${e}-pair-sums`,variant:`pair-sums-batch`,invocationCount:h,permutationWGSL:`parameters[parametersOffset + ${v.base}u] + 1u + index`,gate:{...s,condition:{...s.condition,id:`${r}-batch-${e}-sums-gate`}}}),p(t,{id:`${r}-batch-${e}-scan`,operation:P,variant:`sequential-scan`,bindings:[{name:`pairSums`,view:o,type:`f32`,access:`read`},{name:`state`,view:a,type:`u32`,access:`read_write`},x.binding],workgroupSize:R,invocationCount:R,guardIndex:!1,declarations:w,body:`let local = localInvocationIndex;
  let count = state[stateOffset + 2u];
  let base = state[stateOffset + ${v.base}u];
  let isRunning = state[stateOffset + ${v.done}u] == 0u;
  let observedSum = pairSums[pairSumsOffset];
  var greaterPart = 0u;
  var lesserPart = 0u;
  if (isRunning) {
    for (var item = 0u; item < CHUNK_LENGTH; item++) {
      let offset = local * CHUNK_LENGTH + item;
      let permutation = base + 1u + offset;
      if (offset < BATCH_SIZE && permutation <= count) {
        let pairSum = pairSums[pairSumsOffset + permutation];
        greaterPart += select(0u, 1u, pairSum >= observedSum);
        lesserPart += select(0u, 1u, pairSum <= observedSum);
      }
    }
  }
  chunkGreater[local] = greaterPart;
  chunkLesser[local] = lesserPart;
  workgroupBarrier();
  if (local == 0u && isRunning) {
    var greater = state[stateOffset + ${v.greater}u];
    var lesser = state[stateOffset + ${v.lesser}u];
    var stoppedAt = 0u;
    for (var chunk = 0u; chunk < WORKGROUP_SIZE && stoppedAt == 0u; chunk++) {
      let greaterAfter = greater + chunkGreater[chunk];
      let lesserAfter = lesser + chunkLesser[chunk];
      if (getExceedanceCount(greaterAfter, lesserAfter) >= STOP_AFTER_EXCEEDANCES) {
        for (var item = 0u; item < CHUNK_LENGTH; item++) {
          let offset = chunk * CHUNK_LENGTH + item;
          let permutation = base + 1u + offset;
          if (offset < BATCH_SIZE && permutation <= count) {
            let pairSum = pairSums[pairSumsOffset + permutation];
            greater += select(0u, 1u, pairSum >= observedSum);
            lesser += select(0u, 1u, pairSum <= observedSum);
            if (getExceedanceCount(greater, lesser) >= STOP_AFTER_EXCEEDANCES) {
              stoppedAt = permutation;
              break;
            }
          }
        }
      } else {
        greater = greaterAfter;
        lesser = lesserAfter;
      }
    }
    let nextBase = min(base + BATCH_SIZE, count);
    state[stateOffset + ${v.greater}u] = greater;
    state[stateOffset + ${v.lesser}u] = lesser;
    state[stateOffset + ${v.base}u] = nextBase;
    if (stoppedAt != 0u || nextBase >= count) {
      state[stateOffset + ${v.stoppedAt}u] = stoppedAt;
      state[stateOffset + ${v.done}u] = 1u;
      ${x.disarmWGSL}
    }
  }`}))}return ee}},K=[`moran`,`geary`,`getisOrdG`],ge=[`Moran I`,`Geary C`,`Getis-Ord G`],_e=999,q=28,J=k(de,`permutationCount`),ve=.45,Y=[.08,.04,.92,.41],ye=D({parameters:de,caption:`Observed statistic against shuffles of the values: more shuffles settle the p-value.`,createInputs:({random:e,world:t})=>N(e,t,500,ve),build(e,n,r){let i=ue(n.positions,6),a=e.allocate(`values`,`float32`,n.count),o=e.allocate(`permutation`,`uint32`,4),s=e.allocate(`summary`,`float32`,W.length),c=e.allocate(`histogram`,`uint32`,q),l=e.allocate(`reference`,`float32`,_e),u=e.allocate(`bins`,`float32`,q);return e.graph.add(new G({weights:j(e,i.weights),values:a.view,statistic:K[r.statistic],parameters:o.view,maximumPermutations:_e,output:{summary:s.view,histogram:c.view,referenceDistribution:l.view}})),{outputs:{valueBuffer:a,summary:s,reference:l,bins:u,positions:e.upload(`positions`,`float32x2`,n.positions)},update:e=>{a.write(M(n.field,n.noise,e.noise)),o.write(t({seed:e.seed,permutationCount:e.permutationCount},{maximumPermutations:_e}))}}},rebuildOn:[`statistic`],interaction:{kind:`time`,hint:`sweep the shuffles`,rest:[O(J,J.default),.5],resolve(e,{outputs:t,values:n,world:r,peek:i}){let a=A(J,e.unitX),o=i(t.summary),s=o?.[W.observed]??NaN,c=o?.[W.pseudoPValue],{glyphs:l}=ce({histogram:t.bins,samples:i(t.reference),sampleCount:a,observed:s,rectangle:Y,world:r,pseudoP:c});return{values:{permutationCount:a},glyphs:l,readout:`${a} shuffles · ${ge[n.statistic]} ${Number.isFinite(s)?s.toFixed(3):`…`} · p ${c===void 0?`…`:c.toFixed(3)}`}}},draw:({outputs:e,theme:t,values:n,world:r})=>[{layer:`dots`,positions:e.positions,values:e.valueBuffer,radius:4,range:[0,1]},{layer:`chart`,kind:`bars`,values:e.bins,range:[0,1],rectangle:Y,color:[...t.categories[0],.9]},{layer:`text`,...le({histogram:e.bins,rectangle:Y,world:r,statistic:ge[n.statistic],decimals:3})}]});export{ye as default};