import{r as e}from"./kernel-C3tfQOn4.js";import{o as t,t as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./statistics-kernel-node-lacKNERm.js";import{Q as i}from"./dist-BusamGlx.js";import{t as a}from"./gpu-reduction-tNgOVQ_C.js";import{t as o}from"./gpu-sort-xRm8l5yn.js";import{t as s}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as c}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{t as l}from"./spatial-statistics-collective-node-BlCi_ESE.js";import{t as u}from"./spatial-statistics-bindings-3lH8Vw4b.js";import{n as d,t as f}from"./float32-values-60_XMjYS.js";import{t as ee}from"./gpu-chunked-spatial-weights-transpose-xWSecIf6.js";import{c as p,i as m,s as h,t as g}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{o as te,r as _,s as ne}from"./spatial-weights-CqWysstN.js";import{a as re,n as v}from"./sorted-segment-sums-Dvmm6OdR.js";import{t as y}from"./spatial-statistics-writable-dispatches-BC35SiUi.js";import{E as b,T as x,f as S,p as C,v as w,x as T,y as E}from"./gpu-contributor-utils-BIwLQON5.js";import"./inference-CGICo8RY.js";import{t as D}from"./slot-grouping-CsHvMvbq.js";import{t as O}from"./status-aOwAAXek.js";function ie(n,r){let i=r.id??`global-spatial-statistics`,a=r.weights.status,o=r.output.status,s=a?.overflow||a?.candidateOverflow?e(n,`${i}-weights-incomplete`,`uint32`,1):void 0;if(!s&&!o?.overflow&&!o?.candidateOverflow)return{nodes:[]};let c=[];for(let[e,t,n]of[[`inputOverflow`,a?.overflow,`read`],[`inputCandidateOverflow`,a?.candidateOverflow,`read`],[`incomplete`,s,`read_write`],[`overflow`,o?.overflow,`read_write`],[`candidateOverflow`,o?.candidateOverflow,`read_write`]])t&&c.push({name:e,view:t,type:`u32`,access:n});return{incomplete:s,nodes:[t(n,{id:`${i}-weights-status`,operation:`GPUGlobalSpatialStatistics`,invocationCount:1,bindings:c,body:`let finalLoss = ${a?.overflow?`u32(inputOverflow[inputOverflowOffset] != 0u)`:`0u`};
  let candidateLoss = ${a?.candidateOverflow?`u32(inputCandidateOverflow[inputCandidateOverflowOffset] != 0u)`:`0u`};
  ${s?`incomplete[incompleteOffset] = finalLoss | candidateLoss;`:``}
  ${o?.overflow?`overflow[overflowOffset] = finalLoss;`:``}
  ${o?.candidateOverflow?`candidateOverflow[candidateOverflowOffset] = candidateLoss;`:``}`})]}}var ae=4096,k=4294963199,A=2147483647,j=54,oe=`${c}
${s}
fn addCompensated(pair: vec2<f32>, term: f32) -> vec2<f32> {
  let next = pair.x + term;
  return vec2<f32>(next, pair.y + getAdditionRoundingError(pair.x, term, next));
}
fn readControlFloat(word: u32) -> f32 { return bitcast<f32>(getStatisticsControl(word)); }
var<private> ROW_START: u32;
var<private> ROW_COUNT: u32;
var<private> CAPACITY: u32;
fn activeWeight(weight: f32) -> bool { return weight > 0.0 && isFiniteFloat(weight); }
`;function se(t,r,s,c){let d=r.id??`global-spatial-statistics`,p=g(r.weights),m=p.reduce((e,t)=>e+t.weights.neighbors.length,0),h=!p[0].isWhole&&(r.symmetryStrategy===`transpose`||r.symmetryStrategy!==`pairwise`&&m*4<=t.device.limits.maxStorageBufferBindingSize),_=p.reduce((e,t)=>e+t.rowCount,0),ne=Math.max(...p.map(e=>e.weights.neighbors.length)),v=Math.ceil(_/ae);u(t,d,r,{"row scratch":_*4,"column offsets":(_+1)*4,"transpose scratch":(h?m:ne)*4,"tile partials":v*8*2*4,"totals and control":256});let y=ie(t,r),b=[...y.nodes],x=(n,r,i=_)=>e(t,`${d}-${n}`,r,i),S=f(t,`${d}-values`,r.values),C=r.secondValues?f(t,`${d}-second-values`,r.secondValues):void 0;b.push(...S.nodes,...C?.nodes??[]);let w=S.view,T=C?.view,E=x(`base-states`,`uint32`),D=x(`analysis-states`,`uint32`),O=x(`centered-x`,`float32`),k=x(`centered-y`,`float32`),A=x(`row-scale`,`float32`),se=x(`temporary`,`float32`),M=x(`incoming-weight`,`float32`),N=x(`incoming-weight-error`,`float32`),P=x(`incoming-degree`,`uint32`),F=x(`incoming-lag`,`float32`),I=x(`incoming-lag-error`,`float32`),ce=x(`tile-partials`,`float32`,v*8*2),L=x(`compensated-totals`,`float32`,j),R=e(t,`${d}-control`,`uint32`,64,i.STORAGE|i.UNIFORM),z=(e,t,n=`read`,r=t.format===`uint32`?`u32`:`f32`)=>({name:e,view:t,access:n,type:r}),B=(e,r,i)=>b.push(n(t,{id:`${d}-clear-${e}`,operation:`GPUGlobalSpatialStatistics`,view:r,type:i,value:i===`f32`?`0.0`:`0u`}));B(`control`,R,`u32`),B(`totals`,L,`f32`);for(let[e,t]of Object.entries({incomingWeight:M,incomingWeightError:N,incomingLag:F,incomingLagError:I}))B(e,t,`f32`);B(`incoming-degree`,P,`u32`);let V=(e,n,r,i,a=``,o=[],s=!0)=>{let c=r.find(e=>e.view===R&&e.access===`read_write`),u=c?`fn getStatisticsControl(word: u32) -> u32 { return ${c.type===`atomic<u32>`?`atomicLoad(&counters[countersOffset + word])`:`counters[countersOffset + word]`}; }`:``;b.push(l(t,{id:`${d}-${e}`,operation:`GPUGlobalSpatialStatistics`,variant:e.replace(/-\d+/g,``),invocationCount:n,bindings:r,control:c?void 0:R,parameterValues:o,declarations:`${u}\n${oe}\n${a}`,body:`${s?`if (index >= INVOCATION_COUNT) { return; }`:``}
ROW_START = getChunkParameter(36u); ROW_COUNT = getChunkParameter(37u);
${r.some(e=>e.name===`neighbors`)?`CAPACITY = arrayLength(&neighbors) - neighborsOffset;`:``}
{ ${i} }`}))},H=e=>{let t=p[e].weights;return[z(`offsets`,t.offsets),z(`neighbors`,t.neighbors),z(`weights`,t.weights)]},U=(e,t=`rowScale[rowScaleOffset + neighbor] != 0.0`)=>`
for (var slot = min(offsets[offsetsOffset + index - ROW_START], CAPACITY); slot < min(offsets[offsetsOffset + index - ROW_START + 1u], CAPACITY); slot++) {
 let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
 if (neighbor >= ${_}u || neighbor == index || !activeWeight(weight)) { continue; }
 if (!(${t})) { continue; }
 ${e}
}`,W=(e,t,n,r,i=``)=>{let a=p[t];V(`${e}-${t}`,a.rowCount,n,`let index = index + ROW_START;\n${r}`,i,[a.rowStart,a.rowCount])},G=(e,t,n,r)=>{let i=p[t],a=e===`islands`?[0,1,9,10,24,25,26,27]:[2,3,4],o=r.replaceAll(`&counters[countersOffset]`,`&groupCounts[0u]`).replace(/&counters\[countersOffset \+ (\d+)u\]/g,`&groupCounts[$1u]`);V(`${e}-${t}`,i.rowCount,n,`
if (localInvocationIndex < 28u) { atomicStore(&groupCounts[localInvocationIndex], select(0u, 0xffffffffu, localInvocationIndex == 24u || localInvocationIndex == 26u)); }
workgroupBarrier();
if (index < INVOCATION_COUNT) { let index = index + ROW_START; ${o} }
workgroupBarrier();
if (localInvocationIndex == 0u) {
 ${a.map(e=>`atomic${e===24||e===26?`Min`:e===25||e===27?`Max`:`Add`}(&counters[countersOffset + ${e}u], atomicLoad(&groupCounts[${e}u]));`).join(`
 `)}
}`,`var<workgroup> groupCounts: array<atomic<u32>, 28>;`,[i.rowStart,i.rowCount],!1)},K=(e,t,n,r,i,a,o=``,s=[])=>{let c=Math.ceil(n/ae);V(`${e}-tiles`,c*256,[...i,z(`partials`,ce,`read_write`)],`
let tileIndex = index / 256u;
var sums: array<vec2<f32>, ${r.length}>;
for (var step = 0u; step < 16u; step++) {
 let index = ROW_START + tileIndex * 4096u + localInvocationIndex + step * 256u;
 if (index < ROW_START + ROW_COUNT) {
  var contribution: array<f32, ${r.length}>;
  ${a}
  for (var field = 0u; field < ${r.length}u; field++) { sums[field] = addCompensated(sums[field], contribution[field]); }
 }
}
for (var field = 0u; field < ${r.length}u; field++) {
 reductionPairs[localInvocationIndex] = sums[field]; workgroupBarrier();
 for (var stride = 128u; stride > 0u; stride /= 2u) {
  if (localInvocationIndex < stride) {
   var pair = addCompensated(reductionPairs[localInvocationIndex], reductionPairs[localInvocationIndex + stride].x);
   pair = addCompensated(pair, reductionPairs[localInvocationIndex + stride].y);
   reductionPairs[localInvocationIndex] = pair;
  }
  workgroupBarrier();
 }
 if (localInvocationIndex == 0u) {
  let destination = partialsOffset + (field * getChunkParameter(38u) + tileIndex) * 2u;
  partials[destination] = reductionPairs[0].x; partials[destination + 1u] = reductionPairs[0].y;
 }
 workgroupBarrier();
}`,`var<workgroup> reductionPairs: array<vec2<f32>, 256>;\n${o}`,[t,n,c,...s],!1),V(`${e}-merge`,r.length,[z(`partials`,ce),z(`totals`,L,`read_write`)],`
let columns = array<u32, ${r.length}>(${r.map(e=>`${e}u`).join(`, `)});
let destination = totalsOffset + columns[index] * 2u;
var pair = vec2<f32>(totals[destination], totals[destination + 1u]);
for (var tileIndex = 0u; tileIndex < getChunkParameter(38u); tileIndex++) {
 let source = partialsOffset + (index * getChunkParameter(38u) + tileIndex) * 2u;
 pair = addCompensated(pair, partials[source]); pair = addCompensated(pair, partials[source + 1u]);
}
totals[destination] = pair.x; totals[destination + 1u] = pair.y;`,``,[0,0,c])};V(`extrema-initialize`,1,[z(`counters`,R,`read_write`)],`counters[countersOffset + 24u] = 0xffffffffu; counters[countersOffset + 26u] = 0xffffffffu;`),V(`include`,_,[z(`values`,w),...T?[z(`secondValues`,T)]:[],...r.mask?[z(`mask`,r.mask)]:[],z(`baseStates`,E,`read_write`)],`baseStates[baseStatesOffset + index] = u32(isFiniteFloat(values[valuesOffset + index]) ${T?`&& isFiniteFloat(secondValues[secondValuesOffset + index])`:``} ${r.mask?`&& mask[maskOffset + index] != 0u`:``});`);for(let[e]of p.entries())G(`islands`,e,[...H(e),z(`baseStates`,E),z(`values`,w),...T?[z(`secondValues`,T)]:[],z(`analysisStates`,D,`read_write`),z(`counters`,R,`read_write`,`atomic<u32>`)],`
let includedBase = baseStates[baseStatesOffset + index] != 0u; var hasNeighbor = false;
if (includedBase) { ${U(`hasNeighbor = true;`,`baseStates[baseStatesOffset + neighbor] != 0u`)} }
let island = includedBase && !hasNeighbor;
let included = includedBase && ${r.islands===`keep`?`true`:`!island`};
analysisStates[analysisStatesOffset + index] = u32(included);
if (island) { atomicAdd(&counters[countersOffset + 9u], 1u); }
if (included) {
 atomicAdd(&counters[countersOffset], 1u);
 let value = values[valuesOffset + index];
 let valueBits = bitcast<u32>(value); let ordered = select(valueBits ^ 0x80000000u, ~valueBits, (valueBits & 0x80000000u) != 0u);
 atomicMin(&counters[countersOffset + 24u], ordered); atomicMax(&counters[countersOffset + 25u], ordered);
 ${T?`let secondBits = bitcast<u32>(secondValues[secondValuesOffset + index]); let secondOrdered = select(secondBits ^ 0x80000000u, ~secondBits, (secondBits & 0x80000000u) != 0u); atomicMin(&counters[countersOffset + 26u], secondOrdered); atomicMax(&counters[countersOffset + 27u], secondOrdered);`:``}
 if (value < 0.0) { atomicAdd(&counters[countersOffset + 1u], 1u); }
 if (value != 0.0) { atomicAdd(&counters[countersOffset + 10u], 1u); }
}`);for(let[e,n]of[w,T??w].entries()){V(`magnitude-${e}`,_,[z(`values`,n),z(`analysisStates`,D),z(`temporary`,se,`read_write`)],`temporary[temporaryOffset + index] = select(0.0, abs(values[valuesOffset + index]), analysisStates[analysisStatesOffset + index] != 0u);`);let r=t.createDataView(R.buffer,{format:`float32`,byteOffset:(17+e)*4,length:1});b.push(...new a({id:`${d}-maximum-${e}`,input:se,output:r,operation:`max`}).getCommandNodes(t))}V(`scales`,1,[z(`counters`,R,`read_write`)],`for (var variable = 0u; variable < 2u; variable++) {
 let magnitude = bitcast<f32>(counters[countersOffset + 17u + variable]);
 let exponent = select(0, clamp(i32((bitcast<u32>(magnitude) >> 23u) & 255u) - 126, -126, 126), magnitude > 0.0);
 counters[countersOffset + 5u + variable * 2u] = u32(127 - exponent) << 23u;
 counters[countersOffset + 6u + variable * 2u] = u32(127 + exponent) << 23u;
}`),K(`means`,0,_,[1,2],[z(`values`,w),...T?[z(`secondValues`,T)]:[],z(`analysisStates`,D)],`if (analysisStates[analysisStatesOffset + index] != 0u) { contribution[0] = values[valuesOffset + index] * readControlFloat(5u); contribution[1] = ${T?`secondValues[secondValuesOffset + index] * readControlFloat(7u)`:`0.0`}; }`);let q=`fn readTotal(column: u32) -> f32 { return totals[totalsOffset + column * 2u] + totals[totalsOffset + column * 2u + 1u]; }
fn readMean(column: u32, count: u32) -> f32 {
 let high = totals[totalsOffset + column * 2u]; let low = totals[totalsOffset + column * 2u + 1u];
 let quotient = divideBySpatialCount(high, count);
 if (count <= 16777216u) { return quotient + (fma(-quotient, f32(count), high) + low) / f32(count); }
 return quotient + divideBySpatialCount(low, count);
}`;V(`mean-control`,1,[z(`totals`,L),z(`counters`,R,`read_write`)],`let count = counters[countersOffset]; counters[countersOffset + 13u] = bitcast<u32>(readMean(1u, count)); counters[countersOffset + 15u] = bitcast<u32>(readMean(2u, count));`,q),V(`deviate`,_,[z(`values`,w),...T?[z(`secondValues`,T)]:[],z(`analysisStates`,D),z(`centeredX`,O,`read_write`),z(`centeredY`,k,`read_write`)],`let included = analysisStates[analysisStatesOffset + index] != 0u;
centeredX[centeredXOffset + index] = select(getQuietNaN(index), values[valuesOffset + index] * readControlFloat(5u) - readControlFloat(13u), included);
centeredY[centeredYOffset + index] = select(getQuietNaN(index), ${T?`secondValues[secondValuesOffset + index] * readControlFloat(7u) - readControlFloat(15u)`:`0.0`}, included);`),K(`mean-correction`,0,_,[25,26],[z(`centeredX`,O),z(`centeredY`,k)],`let x = centeredX[centeredXOffset + index]; if (isFiniteFloat(x)) { contribution[0] = x; contribution[1] = centeredY[centeredYOffset + index]; }`),V(`correction-control`,1,[z(`totals`,L),z(`counters`,R,`read_write`)],`let count = counters[countersOffset]; counters[countersOffset + 14u] = bitcast<u32>(readMean(25u, count)); counters[countersOffset + 16u] = bitcast<u32>(readMean(26u, count));`,q),V(`center`,_,[z(`centeredX`,O,`read_write`),z(`centeredY`,k,`read_write`)],`centeredX[centeredXOffset + index] = select(centeredX[centeredXOffset + index] - readControlFloat(14u), 0.0, getStatisticsControl(24u) == getStatisticsControl(25u) && isFiniteFloat(centeredX[centeredXOffset + index])); centeredY[centeredYOffset + index] = select(centeredY[centeredYOffset + index] - readControlFloat(16u), 0.0, getStatisticsControl(26u) == getStatisticsControl(27u) && isFiniteFloat(centeredY[centeredYOffset + index]));`),K(`moments`,0,_,[3,4,5,6],[z(`centeredX`,O),z(`centeredY`,k)],`let x = centeredX[centeredXOffset + index]; if (isFiniteFloat(x)) { let square = x * x; let y = centeredY[centeredYOffset + index]; contribution[0] = square; contribution[1] = square * x; contribution[2] = square * square; contribution[3] = y * y; }`);for(let[e]of p.entries())W(`row-scale`,e,[...H(e),z(`analysisStates`,D),z(`rowScale`,A,`read_write`)],`var scale = 0.0; if (analysisStates[analysisStatesOffset + index] != 0u) {
 var sum = vec2<f32>(0.0); ${U(`sum = addCompensated(sum, weight);`,`analysisStates[analysisStatesOffset + neighbor] != 0u`)}
 scale = ${te(r.weights)?`select(-1.0, 1.0 / (sum.x + sum.y), sum.x + sum.y > 0.0)`:`1.0`};
} rowScale[rowScaleOffset + index] = scale;`);let J=h?{offsets:x(`symmetry-offsets`,`uint32`,_+1),neighbors:x(`symmetry-neighbors`,`uint32`,m),weights:x(`symmetry-weights`,`float32`,m)}:void 0;J&&b.push(...new ee({id:`${d}-symmetry-transpose`,weights:r.weights,output:J}).getCommandNodes(t));let Y=J?[{rowStart:0,rowCount:_,weights:J}]:p;for(let[e,n]of p.entries()){K(`pairs-${e}`,n.rowStart,n.rowCount,[8,10,11,12,13,14],[...H(e),z(`centeredX`,O),z(`centeredY`,k),z(`values`,w),z(`rowScale`,A)],`
if (rowScale[rowScaleOffset + index] != 0.0) {
 let x = centeredX[centeredXOffset + index]; let scale = max(0.0, rowScale[rowScaleOffset + index]);
 var sums: array<vec2<f32>, 5>; var degree = 0u;
 ${U(`let scaled = weight * scale; let neighborX = centeredX[centeredXOffset + neighbor]; let difference = x - neighborX;
 sums[0] = addCompensated(sums[0], scaled);
 sums[1] = addCompensated(sums[1], scaled * x * neighborX);
 sums[2] = addCompensated(sums[2], scaled * difference * difference);
 sums[3] = addCompensated(sums[3], scaled * (values[valuesOffset + index] * readControlFloat(5u)) * (values[valuesOffset + neighbor] * readControlFloat(5u)));
 sums[4] = addCompensated(sums[4], scaled * x * centeredY[centeredYOffset + neighbor]); degree++;`)}
 for (var field = 0u; field < 5u; field++) { contribution[field] = sums[field].x + sums[field].y; }
 contribution[5] = f32(degree);
}`),G(`joins`,e,[...H(e),z(`values`,w),z(`rowScale`,A),z(`counters`,R,`read_write`,`atomic<u32>`)],`var blackBlack = 0u; var blackWhite = 0u; var whiteWhite = 0u;
if (rowScale[rowScaleOffset + index] != 0.0) { let black = values[valuesOffset + index] != 0.0;
 ${U(`let neighborBlack = values[valuesOffset + neighbor] != 0.0; if (black && neighborBlack) { blackBlack++; } else if (black != neighborBlack) { blackWhite++; } else { whiteWhite++; }`)}
} atomicAdd(&counters[countersOffset + 2u], blackBlack); atomicAdd(&counters[countersOffset + 3u], blackWhite); atomicAdd(&counters[countersOffset + 4u], whiteWhite);`);for(let[t,r]of Y.entries())K(`symmetry-${e}-${t}`,n.rowStart,n.rowCount,[9,15],[...H(e),z(`targetOffsets`,r.weights.offsets),z(`targetNeighbors`,r.weights.neighbors),z(`targetWeights`,r.weights.weights),z(`rowScale`,A)],`
if (rowScale[rowScaleOffset + index] != 0.0) {
 var symmetry = vec2<f32>(0.0); var binarySymmetry = 0u;
 ${U(`if (neighbor >= getChunkParameter(39u) && neighbor - getChunkParameter(39u) < getChunkParameter(40u)) {
  let targetRow = select(neighbor - getChunkParameter(39u), index, getChunkParameter(41u) != 0u); let targetColumn = select(index, neighbor, getChunkParameter(41u) != 0u); let targetCapacity = arrayLength(&targetNeighbors) - targetNeighborsOffset;
  var low = min(targetOffsets[targetOffsetsOffset + targetRow], targetCapacity); let end = min(targetOffsets[targetOffsetsOffset + targetRow + 1u], targetCapacity); var high = end;
  while (low < high) { let middle = low + (high - low) / 2u; if (targetNeighbors[targetNeighborsOffset + middle] < targetColumn) { low = middle + 1u; } else { high = middle; } }
  var reverse = -1.0; if (low < end && targetNeighbors[targetNeighborsOffset + low] == targetColumn) { let value = targetWeights[targetWeightsOffset + low]; if (activeWeight(value)) { reverse = value * max(0.0, rowScale[rowScaleOffset + neighbor]); } }
  let scaled = weight * max(0.0, rowScale[rowScaleOffset + index]);
  symmetry = addCompensated(symmetry, select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0)); binarySymmetry += select(1u, 2u, reverse > 0.0);
 }`)}
 contribution[0] = symmetry.x + symmetry.y; contribution[1] = f32(binarySymmetry);
}`,``,[r.rowStart,r.rowCount,Number(h)]);let r=n.weights.neighbors.length;if(r===0)continue;let i=x(`slot-rows-${e}`,`uint32`,r),a=x(`keys-${e}`,`uint32`,r),s=x(`slot-identifiers-${e}`,`uint32`,r),c=x(`column-offsets-${e}`,`uint32`,_+1);W(`slot-rows`,e,[...H(e).slice(0,2),z(`rowScale`,A),z(`slotRows`,i,`read_write`)],`let source = select(${_}u, index, rowScale[rowScaleOffset + index] != 0.0);
for (var slot = min(offsets[offsetsOffset + index - ROW_START], CAPACITY); slot < min(offsets[offsetsOffset + index - ROW_START + 1u], CAPACITY); slot++) { slotRows[slotRowsOffset + slot] = source; }`),V(`transpose-keys-${e}`,r,[...H(e),z(`rowScale`,A),z(`slotRows`,i),z(`keys`,a,`read_write`),z(`slotIdentifiers`,s,`read_write`)],`let used = index < min(offsets[offsetsOffset + ROW_COUNT], CAPACITY); var source = ${_}u; if (used) { source = slotRows[slotRowsOffset + index]; } let neighbor = neighbors[neighborsOffset + index]; var key = ${_}u;
if (source < ${_}u && neighbor < ${_}u && neighbor != source && activeWeight(weights[weightsOffset + index])) { if (rowScale[rowScaleOffset + neighbor] != 0.0) { key = neighbor; } }
keys[keysOffset + index] = key; slotIdentifiers[slotIdentifiersOffset + index] = index;`,``,[n.rowStart,n.rowCount]);let l=x(`sorted-keys-${e}`,`uint32`,r),u=x(`sorted-slots-${e}`,`uint32`,r);b.push(...new o({id:`${d}-transpose-${e}-sort`,keys:a,values:s,outputKeys:l,outputValues:u,keyBits:re(_)}).getCommandNodes(t)),V(`column-offsets-${e}`,_+1,[z(`sortedKeys`,l),z(`columnOffsets`,c,`read_write`)],`var low = 0u; var high = arrayLength(&sortedKeys) - sortedKeysOffset;
while (low < high) { let middle = low + (high - low) / 2u; if (sortedKeys[sortedKeysOffset + middle] < index) { low = middle + 1u; } else { high = middle; } }
columnOffsets[columnOffsetsOffset + index] = low;`);let f=[z(`columnOffsets`,c),z(`sortedSlots`,u),z(`slotRows`,i),z(`weights`,n.weights.weights),z(`rowScale`,A)],ee=e=>`for (var position = columnOffsets[columnOffsetsOffset + index]; position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) { let slot = sortedSlots[sortedSlotsOffset + position]; let source = slotRows[slotRowsOffset + slot]; let weight = weights[weightsOffset + slot] * max(0.0, rowScale[rowScaleOffset + source]); ${e} }`;V(`incoming-weight-${e}`,_,[...f,z(`incomingWeight`,M,`read_write`),z(`incomingWeightError`,N,`read_write`),z(`incomingDegree`,P,`read_write`)],`var pair = vec2<f32>(incomingWeight[incomingWeightOffset + index], incomingWeightError[incomingWeightErrorOffset + index]); var degree = incomingDegree[incomingDegreeOffset + index]; ${ee(`pair = addCompensated(pair, weight); degree++;`)} incomingWeight[incomingWeightOffset + index] = pair.x; incomingWeightError[incomingWeightErrorOffset + index] = pair.y; incomingDegree[incomingDegreeOffset + index] = degree;`),V(`incoming-lag-${e}`,_,[...f,z(`centeredX`,O),z(`incomingLag`,F,`read_write`),z(`incomingLagError`,I,`read_write`)],`var pair = vec2<f32>(incomingLag[incomingLagOffset + index], incomingLagError[incomingLagErrorOffset + index]); ${ee(`pair = addCompensated(pair, weight * centeredX[centeredXOffset + source]);`)} incomingLag[incomingLagOffset + index] = pair.x; incomingLagError[incomingLagErrorOffset + index] = pair.y;`)}V(`incoming-round`,_,[z(`incomingWeight`,M,`read_write`),z(`incomingWeightError`,N),z(`incomingLag`,F,`read_write`),z(`incomingLagError`,I)],`incomingWeight[incomingWeightOffset + index] += incomingWeightError[incomingWeightErrorOffset + index]; incomingLag[incomingLagOffset + index] += incomingLagError[incomingLagErrorOffset + index];`);for(let e of[!1,!0]){for(let[t,n]of p.entries())K(`${e?`degree-deviations`:`degrees`}-${t}`,n.rowStart,n.rowCount,e?[22,23,24]:[16,17,18,20,21],[...H(t),z(`rowScale`,A),z(`incomingWeight`,M),z(`incomingDegree`,P),z(`incomingLag`,F)],`if (rowScale[rowScaleOffset + index] != 0.0) {
 var outgoing = vec2<f32>(0.0); var degree = 0u; ${U(`outgoing = addCompensated(outgoing, weight * max(0.0, rowScale[rowScaleOffset + index])); degree++;`)}
 let weightedDegree = outgoing.x + outgoing.y + incomingWeight[incomingWeightOffset + index]; let binaryDegree = f32(degree + incomingDegree[incomingDegreeOffset + index]); let lag = incomingLag[incomingLagOffset + index];
 ${e?`let deviation = weightedDegree - readControlFloat(19u); let binaryDeviation = binaryDegree - readControlFloat(20u); let lagDeviation = lag - readControlFloat(21u); contribution[0] = deviation * deviation; contribution[1] = binaryDeviation * binaryDeviation; contribution[2] = lagDeviation * lagDeviation;`:`contribution[0] = weightedDegree; contribution[1] = binaryDegree; contribution[2] = lag; contribution[3] = weightedDegree * weightedDegree; contribution[4] = binaryDegree * binaryDegree;`}
}`);e||V(`degree-control`,1,[z(`totals`,L),z(`counters`,R,`read_write`)],`let count = counters[countersOffset]; for (var column = 0u; column < 3u; column++) { counters[countersOffset + 19u + column] = bitcast<u32>(readMean(16u + column, count)); }`,q)}V(`publish`,1,[z(`totals`,L,`read_write`),z(`counters`,R)],`totals[totalsOffset + 7u * 2u] = f32(counters[countersOffset + 10u]); totals[totalsOffset + 19u * 2u] = f32(counters[countersOffset + 9u]);`);let le=r.output.counts;return le&&V(`exact-counts`,1,[z(`counters`,R),z(`counts`,le,`read_write`)],`counts[countsOffset] = counters[countersOffset]; counts[countsOffset + 1u] = counters[countersOffset + 9u]; counts[countsOffset + 2u] = counters[countersOffset + 10u];`),V(`finalize-tiled`,1,[z(`totals`,L),z(`counters`,R),z(`results`,r.output.summary,`read_write`),...r.output.joinCounts?[z(`joinCounts`,r.output.joinCounts,`read_write`)]:[],...y.incomplete?[z(`overflow`,y.incomplete)]:[]],c,`${q}\n${s}`),b}var M={summary:0,moran:8,geary:16,getisOrdG:24,bivariateMoran:32,joinCount:40,length:52},N={count:0,s0:1,s1:2,s2:3,mean:4,variance:5,blackCount:6,islandCount:7},P={statistic:0,expected:1,varianceNormality:2,zNormality:3,pNormality:4,varianceRandomization:5,zRandomization:6,pRandomization:7},F=`GPUGlobalSpatialStatistics`,I=4096,ce=256,L=[`moran`,`geary`,`getisOrdG`,`bivariateMoran`,`joinCount`],R={included:0,x:1,y:2,centeredSquare:3,centeredCube:4,centeredFourth:5,centeredSquareY:6,black:7,rowWeight:8,s1:9,moranCross:10,gearySum:11,gCross:12,bivariateCross:13,binaryOutDegree:14,binaryS1:15,degree:16,binaryDegree:17,columnLag:18,island:19,degreeSquare:20,binaryDegreeSquare:21,degreeDeviation:22,binaryDegreeDeviation:23,columnLagDeviation:24,deviationX:25,deviationY:26},z=27,B={first:0,count:3},V={first:3,count:19},H={first:22,count:3},U={first:25,count:2},W={first:0,count:2},G={inverseX:5,x:6,inverseY:7,y:8},K=9,q=2**24,J=class{id;props;islands;alternative;constructor(e){let t=e.id??`global-spatial-statistics`;this.id=t,this.props=e,this.islands=ne(t,e.islands),this.alternative=Y(t,e.alternative);let n=h(t,e.weights);if(`chunks`in e.weights)for(let[n,r]of e.weights.chunks.entries())x(t,!r.weights.status?.overflow&&!r.weights.status?.candidateOverflow,`weights.chunks[${n}].weights.status loss flags must be aggregated on the outer weights.status port`);if(S(t,`weights.rowCount (GLOBAL_SPATIAL_STATISTICS_MAXIMUM_ROWS)`,n,1,k),e.mode!==void 0&&e.mode!==`tiled`&&e.mode!==`reference`)throw Error(`${t} mode must be tiled or reference`);e.mode===`reference`&&S(t,`weights.rowCount (GLOBAL_SPATIAL_STATISTICS_REFERENCE_MAXIMUM_ROWS)`,n,1,q-1),C(t,g(e.weights).length**2*8+g(e.weights).length*220+100);let r=g(e.weights).reduce((e,t)=>e+t.weights.neighbors.length,0);S(t,`weights.capacity (GLOBAL_SPATIAL_STATISTICS_MAXIMUM_SLOTS)`,r,0,A);for(let n of[`overflow`,`candidateOverflow`]){let r=e.weights.status?.[n];r&&(b(r,[`uint32`],`${t} weights.status.${n}`),S(t,`weights.status.${n} length`,r.length,1))}if(d(e.values,`${t} values`)!==n)throw Error(`${t} values needs one row per weights row`);if(e.secondValues&&d(e.secondValues,`${t} secondValues`)!==n)throw Error(`${t} secondValues needs one row per weights row`);if(e.mask&&(b(e.mask,[`uint32`],`${t} mask`),e.mask.length!==n))throw Error(`${t} mask needs one row per weights row`);if(e.statistics.length===0)throw Error(`${t} statistics must name at least one statistic`);for(let n of e.statistics)if(!L.includes(n))throw Error(`${t} unknown statistic ${n}`);if(e.statistics.includes(`bivariateMoran`)&&!e.secondValues)throw Error(`${t} bivariateMoran requires secondValues`);let{output:i}=e;O(t,i.status??{});for(let[e,n]of Object.entries(i.status??{}))n&&(b(n,[`uint32`],`${t} output.status.${e}`),S(t,`output.status.${e} length`,n.length,1,1));if(b(i.summary,[`float32`],`${t} output.summary`),i.summary.length<M.length)throw Error(`${t} output.summary needs ${M.length} values`);if(i.joinCounts&&(b(i.joinCounts,[`uint32`],`${t} output.joinCounts`),i.joinCounts.length<3))throw Error(`${t} output.joinCounts needs 3 values`);i.counts&&(b(i.counts,[`uint32`],`${t} output.counts`),S(t,`output.counts.length`,i.counts.length,3)),y(t,{summary:i.summary,joinCounts:i.joinCounts,counts:i.counts,overflow:i.status?.overflow,candidateOverflow:i.status?.candidateOverflow},[[`summary`,`joinCounts`],[`overflow`,`candidateOverflow`]]),E(t,[i.summary,i.joinCounts,i.counts,...Object.values(i.status??{})],[...fe(e.weights),e.weights.status?.overflow,e.weights.status?.candidateOverflow,e.values,e.secondValues,e.mask])}getInferenceMetadata(){return{alternative:this.alternative,permutationCount:0,multipleTesting:`none`,testedRowsDefinition:`included rows with finite values${this.islands===`exclude`?`, islands excluded`:` (islands kept, as esda)`}; one test per statistic`}}getCommandNodes(t){let{id:i,props:a}=this,{weights:o,mask:s,output:l}=a;if(a.mode!==`reference`)return se(t,a,`${Object.entries(R).map(([e,t])=>`const COLUMN_${le(e)}: u32 = ${t}u;`).join(`
`)}
${ue}
${me(this.alternative)}
fn writeBlock(offset: u32, statisticBlock: array<f32, 8>, enabled: bool) {
 for (var field = 0u; field < 8u; field++) { results[resultsOffset + offset + field] = select(getQuietNaN(field), statisticBlock[field], enabled); }
}`,he(a,!!(o.status?.overflow||o.status?.candidateOverflow)).replace(`let mean = divideBySpatialCount(readTotal(COLUMN_X), count) * inverseScaleX + divideBySpatialCount(readTotal(COLUMN_DEVIATION_X), count);`,`let mean = readControlFloat(13u) + readControlFloat(14u);`).replace(`let p = divide(blackCount, n);`,`let p = divideSpatialCounts(counters[countersOffset + 10u], count);`).replaceAll(`(blackCount - 1.0)`,`f32(counters[countersOffset + 10u] - min(counters[countersOffset + 10u], 1u))`));let d=g(o).reduce((e,t)=>e+t.rowCount,0),m=g(o).reduce((e,t)=>e+t.weights.neighbors.length,0);u(t,i,a,{matrix:z*d*4,centered:d*8,magnitudes:d*8,"column parts":d*12,"transpose offsets":(d+1)*4,"transpose slots":m*4}),T(i,t,[...fe(o),o.status?.overflow,o.status?.candidateOverflow,a.values,a.secondValues,s,l.summary,l.joinCounts,l.counts,...Object.values(l.status??{})]);let h=ie(t,a),ne=f(t,`${i}-values`,a.values),re=a.secondValues?f(t,`${i}-second-values`,a.secondValues):void 0,v=ne.view,y=re?.view,b=g(o),x=b[0].isWhole,S=b.reduce((e,t)=>e+t.rowCount,0),C=b.reduce((e,t)=>e+t.weights.neighbors.length,0),E=h.incomplete,O=te(o),ae=this.islands===`keep`,k=Math.ceil(S/I),A=e=>`const ROWS: u32 = ${S}u;
const CAPACITY: u32 = ${e?e.weights.neighbors.length:C}u;${e&&!e.isWhole?`\nconst CHUNK_ROWS: u32 = ${e.rowCount}u;\nconst CHUNK_INDEX: u32 = ${e.chunkIndex}u;\n${e.constantsWGSL}`:``}
${Object.entries(R).map(([e,t])=>`const COLUMN_${le(e)}: u32 = ${t}u;`).join(`
`)}
fn getMatrixIndex(column: u32, row: u32) -> u32 {
  return column * ROWS + row;
}
// total / count with one remainder correction. WGSL division may be 2.5 ulp off (Metal computes
// it through a reciprocal), so 24 / 12 can miss 2; the correction makes the quotient correctly
// rounded in practice, so constant values center to exactly 0 and their statistics are NaN, not
// the ratio of rounding residues.
fn getMeanQuotient(total: f32, count: f32) -> f32 {
  let quotient = total / count;
  return quotient + fma(-quotient, count, total) / count;
}
${c}`,j=A(),oe=`isFiniteFloat(centered[centeredOffset + neighbor * 2u])`;w(t.device,i,`matrix`,z*S*4,`reduce the row count`);let M=e(t,`${i}-matrix`,`float32`,z*S),N=e(t,`${i}-centered`,`float32`,S*2),P=e(t,`${i}-totals-a`,`float32`,B.count),L=e(t,`${i}-totals-b`,`float32`,V.count),q=e(t,`${i}-totals-c`,`float32`,H.count),J=e(t,`${i}-totals-d`,`float32`,U.count),Y=e(t,`${i}-magnitudes`,`float32`,W.count*S),ge=e(t,`${i}-magnitude-totals`,`float32`,W.count),_e=e(t,`${i}-base-states`,`uint32`,S),Z=e(t,`${i}-counters`,`uint32`,K),ve=e(t,`${i}-row-scale`,`float32`,S),ye=x?void 0:e(t,`${i}-column-parts`,`float32`,S*3),be=[...y?[{name:`secondValues`,view:y,type:`f32`,access:`read`}]:[],...s?[{name:`mask`,view:s,type:`u32`,access:`read`}]:[]],xe=e=>e.isWhole?`ROWS`:`WEIGHTS_ROW_START + CHUNK_ROWS`,Se=e=>e.isWhole?`index`:`(index - WEIGHTS_ROW_START)`,Ce=(e,t,n)=>e.isWhole?_(t,n):_(t,n,`index`,`WEIGHTS_ROW_START`),Q=e=>[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:e.weights.weights,type:`f32`,access:`read`}],we=b.map(e=>r(t,{id:`${i}-islands${e.idSuffix}`,operation:F,variant:`islands`,bindings:[...Q(e),{name:`baseStates`,view:_e,type:`u32`,access:`read`},{name:`counters`,view:Z,type:`atomic<u32>`,access:`read_write`},{name:`matrix`,view:M,type:`f32`,access:`read_write`},{name:`magnitudes`,view:Y,type:`f32`,access:`read_write`}],invocationCount:e.rowCount,workgroupSize:ce,guardIndex:!1,declarations:`${A(e)}
var<workgroup> groupIncluded: atomic<u32>;
var<workgroup> groupNegative: atomic<u32>;`,body:p(e,`if (index < ${xe(e)}) {
  let base = baseStates[baseStatesOffset + index] != 0u;
  var hasNeighbor = false;
  if (base) {
    ${Ce(e,`baseStates[baseStatesOffset + neighbor] != 0u`,`hasNeighbor = true;`)}
  }
  let island = base && !hasNeighbor;
  let included = base && ${ae?`true`:`!island`};
  if (included) {
    atomicAdd(&groupIncluded, 1u);
    if (matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] < 0.0) {
      atomicAdd(&groupNegative, 1u);
    }
  } else {
    // Missing rows and excluded islands leave the sums.
    matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] = 0.0;
    matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] = 0.0;
    magnitudes[magnitudesOffset + index] = 0.0;
    magnitudes[magnitudesOffset + ROWS + index] = 0.0;
  }
  matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] = select(0.0, 1.0, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ISLAND, index)] = select(0.0, 1.0, island);
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&counters[countersOffset], atomicLoad(&groupIncluded));
    atomicAdd(&counters[countersOffset + 1u], atomicLoad(&groupNegative));
  }`)})),Te=O?b.map(e=>r(t,{id:`${i}-row-scale${e.idSuffix}`,operation:F,variant:`row-scale`,bindings:[...Q(e),{name:`centered`,view:N,type:`f32`,access:`read`},{name:`rowScale`,view:ve,type:`f32`,access:`read_write`}],invocationCount:e.rowCount,declarations:A(e),body:p(e,`var sum = 0.0;
  if (isFiniteFloat(centered[centeredOffset + index * 2u])) {
    ${Ce(e,oe,`sum += weight;`)}
  }
  rowScale[rowScaleOffset + index] = select(0.0, 1.0 / sum, sum > 0.0);`)})):[],Ee=b.map(n=>{let a=n.weights.neighbors.length,o=e(t,`${i}-slot-rows${n.idSuffix}`,`uint32`,a),s=e(t,`${i}-transpose-keys${n.idSuffix}`,`uint32`,a),c=e(t,`${i}-slot-ids${n.idSuffix}`,`uint32`,a),l=e(t,`${i}-column-offsets${n.idSuffix}`,`uint32`,S+1),u=D(t,{id:`${i}-transpose${n.idSuffix}`,operation:F,slotKeys:s,slotIndices:c,columns:S,columnOffsets:l}),d=Se(n),f=n.isWhole?`ROWS`:`CHUNK_ROWS`;return{nodes:[r(t,{id:`${i}-slot-rows${n.idSuffix}`,operation:F,variant:`slot-rows`,bindings:[{name:`offsets`,view:n.weights.offsets,type:`u32`,access:`read`},{name:`centered`,view:N,type:`f32`,access:`read`},{name:`slotRows`,view:o,type:`u32`,access:`read_write`}],invocationCount:n.rowCount,declarations:A(n),body:p(n,`let source = select(ROWS, index, isFiniteFloat(centered[centeredOffset + index * 2u]));
  let begin = min(offsets[offsetsOffset + ${d}], CAPACITY);
  let end = min(offsets[offsetsOffset + ${d} + 1u], CAPACITY);
  for (var slot = begin; slot < end; slot++) {
    slotRows[slotRowsOffset + slot] = source;
  }`)}),r(t,{id:`${i}-transpose-keys${n.idSuffix}`,operation:F,variant:`transpose-keys`,bindings:[...Q(n),{name:`slotRows`,view:o,type:`u32`,access:`read`},{name:`centered`,view:N,type:`f32`,access:`read`},{name:`transposeKeys`,view:s,type:`u32`,access:`read_write`},{name:`slotIds`,view:c,type:`u32`,access:`read_write`}],invocationCount:a,declarations:A(n),body:`let used = index < min(offsets[offsetsOffset + ${f}], CAPACITY);
  let source = select(ROWS, slotRows[slotRowsOffset + index], used);
  let neighbor = neighbors[neighborsOffset + index];
  var key = ROWS;
  if (source < ROWS && neighbor < ROWS && neighbor != source &&
      weights[weightsOffset + index] > 0.0 &&
      isFiniteFloat(centered[centeredOffset + neighbor * 2u])) {
    key = neighbor;
  }
  transposeKeys[transposeKeysOffset + index] = key;
  slotIds[slotIdsOffset + index] = index;`}),...u.nodes],slotRows:o,columnOffsets:l,sortedSlots:u.sortedSlots}}),De=Ee.flatMap(e=>e.nodes),Oe=e=>{let t=!e.isWhole;return p(e,`var blackBlack = 0u;
  var blackWhite = 0u;
  var whiteWhite = 0u;
  if (index < ${xe(e)}) {
  let centeredX = centered[centeredOffset + index * 2u];
  var rowWeight = 0.0;
  var s1 = 0.0;
  var lag = 0.0;
  var gearySum = 0.0;
  var rawLag = 0.0;
  var lagY = 0.0;
  var outDegree = 0u;
  var binaryS1 = 0u;
  let x = values[valuesOffset + index];
  // General G uses the raw values in scaled units; the black test uses them unscaled.
  let inverseScaleX = bitcast<f32>(atomicLoad(&counters[countersOffset + ${G.inverseX}u]));
  if (isFiniteFloat(centeredX)) {
    let black = x != 0.0;
    let scale = getRowScale(index);
    ${Ce(e,oe,`let neighborX = centered[centeredOffset + neighbor * 2u];
      let scaled = weight * scale;
      ${t?``:`let reverse = findWeight(neighbor, index);
      s1 += select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0);
      binaryS1 += select(1u, 2u, reverse > 0.0);`}
      rowWeight += scaled;
      lag += scaled * neighborX;
      let difference = centeredX - neighborX;
      gearySum += scaled * difference * difference;
      rawLag += scaled * (values[valuesOffset + neighbor] * inverseScaleX);
      lagY += scaled * centered[centeredOffset + neighbor * 2u + 1u];
      outDegree++;
      let neighborBlack = values[valuesOffset + neighbor] != 0.0;
      if (black && neighborBlack) {
        blackBlack++;
      } else if (black != neighborBlack) {
        blackWhite++;
      } else {
        whiteWhite++;
      }`)}
  }
  let included = isFiniteFloat(centeredX);
  matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] = rowWeight;
  ${t?``:`matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;`}
  matrix[matrixOffset + getMatrixIndex(COLUMN_MORAN_CROSS, index)] = select(0.0, centeredX * lag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_GEARY_SUM, index)] = gearySum;
  matrix[matrixOffset + getMatrixIndex(COLUMN_G_CROSS, index)] = select(0.0, (x * inverseScaleX) * rawLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BIVARIATE_CROSS, index)] = select(0.0, centeredX * lagY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] = f32(outDegree);
  ${t?``:`matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);`}
  }
  atomicAdd(&groupBlackBlack, blackBlack);
  atomicAdd(&groupBlackWhite, blackWhite);
  atomicAdd(&groupWhiteWhite, whiteWhite);
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    atomicAdd(&counters[countersOffset + 2u], atomicLoad(&groupBlackBlack));
    atomicAdd(&counters[countersOffset + 3u], atomicLoad(&groupBlackWhite));
    atomicAdd(&counters[countersOffset + 4u], atomicLoad(&groupWhiteWhite));
  }`)},ke=O?[{name:`rowScale`,view:ve,type:`f32`,access:`read`}]:[],Ae=[],je=!x&&(a.symmetryStrategy===`transpose`||a.symmetryStrategy!==`pairwise`&&C*4<=t.device.limits.maxStorageBufferBindingSize),$=je?{offsets:e(t,`${i}-symmetry-offsets`,`uint32`,S+1),neighbors:e(t,`${i}-symmetry-neighbors`,`uint32`,C),weights:e(t,`${i}-symmetry-weights`,`float32`,C)}:void 0;$&&!x&&Ae.push(...new ee({id:`${i}-symmetry`,weights:o,output:$}).getCommandNodes(t));for(let e of b){let n=!e.isWhole;if(Ae.push(r(t,{id:`${i}-pairs${e.idSuffix}`,operation:F,variant:n?`pairs-chunk`:`pairs`,bindings:[...Q(e),{name:`values`,view:v,type:`f32`,access:`read`},...ke,{name:`centered`,view:N,type:`f32`,access:`read`},{name:`matrix`,view:M,type:`f32`,access:`read_write`},{name:`counters`,view:Z,type:`atomic<u32>`,access:`read_write`}],invocationCount:e.rowCount,workgroupSize:ce,guardIndex:!1,declarations:`${A(e)}
var<workgroup> groupBlackBlack: atomic<u32>;
var<workgroup> groupBlackWhite: atomic<u32>;
var<workgroup> groupWhiteWhite: atomic<u32>;
${pe(O)}
${n?``:`
fn getRowEnd(row: u32) -> u32 {
  return min(offsets[offsetsOffset + row + 1u], CAPACITY);
}

// Effective weight w'_row,column by binary search of the ascending neighbor IDs of row, or -1 when
// the slot is absent or not active (zero weight).
fn findWeight(row: u32, column: u32) -> f32 {
  var low = min(offsets[offsetsOffset + row], CAPACITY);
  let end = getRowEnd(row);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (neighbors[neighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && neighbors[neighborsOffset + low] == column) {
    let weight = weights[weightsOffset + low];
    return select(-1.0, weight * getRowScale(row), weight > 0.0);
  }
  return -1.0;
}`}`,body:Oe(e)})),n){if(je&&$)Ae.push(r(t,{id:`${i}-reverse${e.idSuffix}-transpose`,operation:F,variant:`reverse-transpose`,bindings:[...Q(e),{name:`targetOffsets`,view:$.offsets,type:`u32`,access:`read`},{name:`targetNeighbors`,view:$.neighbors,type:`u32`,access:`read`},{name:`targetWeights`,view:$.weights,type:`f32`,access:`read`},...ke,{name:`matrix`,view:M,type:`f32`,access:`read_write`}],invocationCount:e.rowCount,declarations:`${A(e)}
const TARGET_CAPACITY: u32 = ${$.neighbors.length}u;
${pe(O)}

// Effective reverse weight: the row of the transpose lists its source rows ascending. -1 when
// the slot is absent or not active (zero weight).
fn findReverseWeight(row: u32, column: u32) -> f32 {
  var low = min(targetOffsets[targetOffsetsOffset + row], TARGET_CAPACITY);
  let end = min(targetOffsets[targetOffsetsOffset + row + 1u], TARGET_CAPACITY);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (targetNeighbors[targetNeighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && targetNeighbors[targetNeighborsOffset + low] == column) {
    let weight = targetWeights[targetWeightsOffset + low];
    return select(-1.0, weight * getRowScale(column), weight > 0.0);
  }
  return -1.0;
}`,body:p(e,`var s1 = 0.0;
  var binaryS1 = 0u;
  if (matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0) {
    let scale = getRowScale(index);
    ${Ce(e,`matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, neighbor)] != 0.0`,`let scaled = weight * scale;
      let reverse = findReverseWeight(index, neighbor);
      s1 += select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0);
      binaryS1 += select(1u, 2u, reverse > 0.0);`)}
  }
  matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);`)}));else for(let n of b)Ae.push(r(t,{id:`${i}-reverse${e.idSuffix}-target${n.chunkIndex}`,operation:F,variant:`reverse-chunk`,bindings:[...Q(e),{name:`targetOffsets`,view:n.weights.offsets,type:`u32`,access:`read`},{name:`targetNeighbors`,view:n.weights.neighbors,type:`u32`,access:`read`},{name:`targetWeights`,view:n.weights.weights,type:`f32`,access:`read`},...ke,{name:`matrix`,view:M,type:`f32`,access:`read_write`}],invocationCount:e.rowCount,declarations:`${A(e)}
const TARGET_START: u32 = ${n.rowStart}u;
const TARGET_END: u32 = ${n.rowStart+n.rowCount}u;
const TARGET_CAPACITY: u32 = ${n.weights.neighbors.length}u;
const TARGET_INDEX: u32 = ${n.chunkIndex}u;
${pe(O)}

// Effective weight w'_row,column by binary search of the ascending neighbor IDs of a target chunk
// row, or -1 when the slot is absent or not active (zero weight).
fn findWeight(row: u32, column: u32) -> f32 {
  let local = row - TARGET_START;
  var low = min(targetOffsets[targetOffsetsOffset + local], TARGET_CAPACITY);
  let end = min(targetOffsets[targetOffsetsOffset + local + 1u], TARGET_CAPACITY);
  var high = end;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (targetNeighbors[targetNeighborsOffset + middle] < column) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < end && targetNeighbors[targetNeighborsOffset + low] == column) {
    let weight = targetWeights[targetWeightsOffset + low];
    return select(-1.0, weight * getRowScale(row), weight > 0.0);
  }
  return -1.0;
}`,body:p(e,`var s1 = 0.0;
  var binaryS1 = 0u;
  if (TARGET_INDEX > 0u) {
    s1 = matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)];
    binaryS1 = u32(matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)]);
  }
  if (matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0) {
    let scale = getRowScale(index);
    ${Ce(e,`matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, neighbor)] != 0.0 && neighbor >= TARGET_START && neighbor < TARGET_END`,`let scaled = weight * scale;
      let reverse = findWeight(neighbor, index);
      s1 += select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0);
      binaryS1 += select(1u, 2u, reverse > 0.0);`)}
  }
  matrix[matrixOffset + getMatrixIndex(COLUMN_S1, index)] = s1;
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_S1, index)] = f32(binaryS1);`)}))}}let Me=b.map((e,n)=>{let a=Ee[n],o=[{name:`columnOffsets`,view:a.columnOffsets,type:`u32`,access:`read`},{name:`sortedSlots`,view:a.sortedSlots,type:`u32`,access:`read`},{name:`slotRows`,view:a.slotRows,type:`u32`,access:`read`},{name:`weights`,view:e.weights.weights,type:`f32`,access:`read`},...O?[{name:`rowScale`,view:ve,type:`f32`,access:`read`}]:[],{name:`centered`,view:N,type:`f32`,access:`read`}];if(ye){let a=e=>n>0?`columnParts[columnPartsOffset + index * 3u + ${e}u]`:`0.0`;return r(t,{id:`${i}-columns${e.idSuffix}`,operation:F,variant:`columns-chunk`,bindings:[...o,{name:`columnParts`,view:ye,type:`f32`,access:`read_write`}],invocationCount:S,declarations:`${A(e)}
${pe(O)}`,body:`let begin = columnOffsets[columnOffsetsOffset + index];
  let end = columnOffsets[columnOffsetsOffset + index + 1u];
  var columnWeight = ${a(0)};
  var inDegree = ${a(1)};
  var columnLag = ${a(2)};
  for (var position = begin; position < end; position++) {
    let slot = sortedSlots[sortedSlotsOffset + position];
    let source = slotRows[slotRowsOffset + slot];
    let weight = weights[weightsOffset + slot] * getRowScale(source);
    columnWeight += weight;
    inDegree += 1.0;
    columnLag += weight * centered[centeredOffset + source * 2u];
  }
  columnParts[columnPartsOffset + index * 3u] = columnWeight;
  columnParts[columnPartsOffset + index * 3u + 1u] = inDegree;
  columnParts[columnPartsOffset + index * 3u + 2u] = columnLag;`})}return r(t,{id:`${i}-columns`,operation:F,variant:`columns`,bindings:[...o,{name:`matrix`,view:M,type:`f32`,access:`read_write`}],invocationCount:S,declarations:`${j}
${pe(O)}`,body:`let included = isFiniteFloat(centered[centeredOffset + index * 2u]);
  let begin = columnOffsets[columnOffsetsOffset + index];
  let end = columnOffsets[columnOffsetsOffset + index + 1u];
  var columnWeight = 0.0;
  var inDegree = 0u;
  var columnLag = 0.0;
  for (var position = begin; position < end; position++) {
    let slot = sortedSlots[sortedSlotsOffset + position];
    // Only active slots are keyed into columns, so every weight here is positive.
    let source = slotRows[slotRowsOffset + slot];
    let weight = weights[weightsOffset + slot] * getRowScale(source);
    columnWeight += weight;
    inDegree++;
    columnLag += weight * centered[centeredOffset + source * 2u];
  }
  ${de}`})});ye&&Me.push(r(t,{id:`${i}-columns-finalize`,operation:F,variant:`columns-finalize`,bindings:[{name:`centered`,view:N,type:`f32`,access:`read`},{name:`columnParts`,view:ye,type:`f32`,access:`read`},{name:`matrix`,view:M,type:`f32`,access:`read_write`}],invocationCount:S,declarations:j,body:`let included = isFiniteFloat(centered[centeredOffset + index * 2u]);
  let columnWeight = columnParts[columnPartsOffset + index * 3u];
  let inDegree = u32(columnParts[columnPartsOffset + index * 3u + 1u]);
  let columnLag = columnParts[columnPartsOffset + index * 3u + 2u];
  ${de}`}));let Ne=[...h.nodes,...ne.nodes,...re?.nodes??[],n(t,{id:`${i}-counters-clear`,operation:F,view:Z,type:`u32`,value:`0u`}),r(t,{id:`${i}-include`,operation:F,variant:`include`,bindings:[{name:`values`,view:v,type:`f32`,access:`read`},...be,{name:`baseStates`,view:_e,type:`u32`,access:`read_write`},{name:`matrix`,view:M,type:`f32`,access:`read_write`},{name:`magnitudes`,view:Y,type:`f32`,access:`read_write`}],invocationCount:S,declarations:j,body:`let x = values[valuesOffset + index];
  let y = ${y?`secondValues[secondValuesOffset + index]`:`0.0`};
  let included = ${s?`mask[maskOffset + index] != 0u &&`:``} isFiniteFloat(x) && isFiniteFloat(y);
  baseStates[baseStatesOffset + index] = select(0u, 1u, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] = select(0.0, x, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] = select(0.0, y, included);
  magnitudes[magnitudesOffset + index] = select(0.0, abs(x), included);
  magnitudes[magnitudesOffset + ROWS + index] = select(0.0, abs(y), included);`}),...we,...X(t,{id:`${i}-sum-a`,matrix:M,rows:S,blockCount:k,...B,totals:P}),...X(t,{id:`${i}-sum-magnitudes`,matrix:Y,rows:S,blockCount:k,...W,totals:ge}),r(t,{id:`${i}-value-scales`,operation:F,variant:`value-scales`,bindings:[{name:`magnitudeTotals`,view:ge,type:`f32`,access:`read`},{name:`counters`,view:Z,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${j}
// Binary exponent e with 2^(e - 1) <= mean magnitude < 2^e, read from the float bits (exact), or 0
// when the mean magnitude is zero or not finite. Clamped so that 2^e and 2^-e are normal floats.
fn getScaleExponent(totalMagnitude: f32, count: f32) -> i32 {
  let meanMagnitude = totalMagnitude / count;
  if (!(meanMagnitude > 0.0) || !isFiniteFloat(meanMagnitude)) {
    return 0;
  }
  let biasedExponent = i32((bitcast<u32>(meanMagnitude) >> 23u) & 0xffu);
  return clamp(biasedExponent - 126, -126, 126);
}
// Bit pattern of the float 2^exponent.
fn getPowerOfTwoBits(exponent: i32) -> u32 {
  return u32(exponent + 127) << 23u;
}`,body:`let count = f32(counters[countersOffset]);
  let exponentX = getScaleExponent(magnitudeTotals[magnitudeTotalsOffset], count);
  let exponentY = getScaleExponent(magnitudeTotals[magnitudeTotalsOffset + 1u], count);
  counters[countersOffset + ${G.inverseX}u] = getPowerOfTwoBits(-exponentX);
  counters[countersOffset + ${G.x}u] = getPowerOfTwoBits(exponentX);
  counters[countersOffset + ${G.inverseY}u] = getPowerOfTwoBits(-exponentY);
  counters[countersOffset + ${G.y}u] = getPowerOfTwoBits(exponentY);`}),r(t,{id:`${i}-deviate`,operation:F,variant:`deviate`,bindings:[{name:`totalsA`,view:P,type:`f32`,access:`read`},{name:`counters`,view:Z,type:`u32`,access:`read`},{name:`matrix`,view:M,type:`f32`,access:`read_write`}],invocationCount:S,declarations:j,body:`let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  let inverseScaleX = bitcast<f32>(counters[countersOffset + ${G.inverseX}u]);
  let inverseScaleY = bitcast<f32>(counters[countersOffset + ${G.inverseY}u]);
  let x = matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)] * inverseScaleX;
  let y = matrix[matrixOffset + getMatrixIndex(COLUMN_Y, index)] * inverseScaleY;
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_X, index)] =
    select(0.0, x - getMeanQuotient(totalsA[totalsAOffset + COLUMN_X], count) * inverseScaleX, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_Y, index)] =
    select(0.0, y - getMeanQuotient(totalsA[totalsAOffset + COLUMN_Y], count) * inverseScaleY, included);`}),...X(t,{id:`${i}-sum-d`,matrix:M,rows:S,blockCount:k,...U,totals:J}),r(t,{id:`${i}-center`,operation:F,variant:`center`,bindings:[{name:`totalsD`,view:J,type:`f32`,access:`read`},{name:`counters`,view:Z,type:`u32`,access:`read`},{name:`centered`,view:N,type:`f32`,access:`read_write`},{name:`matrix`,view:M,type:`f32`,access:`read_write`}],invocationCount:S,declarations:j,body:`let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  let x = matrix[matrixOffset + getMatrixIndex(COLUMN_X, index)];
  let centeredX = matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_X, index)] - getMeanQuotient(totalsD[totalsDOffset], count);
  let centeredY = ${y?`matrix[matrixOffset + getMatrixIndex(COLUMN_DEVIATION_Y, index)] - getMeanQuotient(totalsD[totalsDOffset + 1u], count)`:`0.0`};
  let square = centeredX * centeredX;
  centered[centeredOffset + index * 2u] = select(getQuietNaN(index), centeredX, included);
  centered[centeredOffset + index * 2u + 1u] = select(getQuietNaN(index), centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE, index)] = select(0.0, square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_CUBE, index)] = select(0.0, square * centeredX, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_FOURTH, index)] = select(0.0, square * square, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_CENTERED_SQUARE_Y, index)] = select(0.0, centeredY * centeredY, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BLACK, index)] =
    select(0.0, 1.0, included && x != 0.0);`}),...Te,...De,...Ae,...Me,...X(t,{id:`${i}-sum-b`,matrix:M,rows:S,blockCount:k,...V,totals:L}),r(t,{id:`${i}-deviations`,operation:F,variant:`deviations`,bindings:[{name:`totalsB`,view:L,type:`f32`,access:`read`},{name:`counters`,view:Z,type:`u32`,access:`read`},{name:`matrix`,view:M,type:`f32`,access:`read_write`}],invocationCount:S,declarations:`${j}
const BATCH_B_FIRST: u32 = ${V.first}u;
fn readTotalB(column: u32) -> f32 {
  return totalsB[totalsBOffset + column - BATCH_B_FIRST];
}
fn getSquaredDeviation(column: u32, row: u32, count: f32) -> f32 {
  let deviation = matrix[matrixOffset + getMatrixIndex(column, row)] - readTotalB(column) / count;
  return deviation * deviation;
}`,body:`let included = matrix[matrixOffset + getMatrixIndex(COLUMN_INCLUDED, index)] != 0.0;
  let count = f32(counters[countersOffset]);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_BINARY_DEGREE, index, count), included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG_DEVIATION, index)] =
    select(0.0, getSquaredDeviation(COLUMN_COLUMN_LAG, index, count), included);`}),...X(t,{id:`${i}-sum-c`,matrix:M,rows:S,blockCount:k,...H,totals:q}),r(t,{id:`${i}-finalize`,operation:F,variant:`finalize`,bindings:[{name:`totalsA`,view:P,type:`f32`,access:`read`},{name:`totalsB`,view:L,type:`f32`,access:`read`},{name:`totalsC`,view:q,type:`f32`,access:`read`},{name:`totalsD`,view:J,type:`f32`,access:`read`},{name:`counters`,view:Z,type:`u32`,access:`read`},{name:`results`,view:l.summary,type:`f32`,access:`read_write`},...l.joinCounts?[{name:`joinCounts`,view:l.joinCounts,type:`u32`,access:`read_write`}]:[],...E?[{name:`overflow`,view:E,type:`u32`,access:`read`}]:[]],invocationCount:1,declarations:`${j}
${ue}
const BATCH_B_FIRST: u32 = ${V.first}u;
const BATCH_C_FIRST: u32 = ${H.first}u;
const BATCH_D_FIRST: u32 = ${U.first}u;
fn readTotal(column: u32) -> f32 {
  if (column < BATCH_B_FIRST) {
    return totalsA[totalsAOffset + column];
  }
  if (column < BATCH_C_FIRST) {
    return totalsB[totalsBOffset + column - BATCH_B_FIRST];
  }
  if (column < BATCH_D_FIRST) {
    return totalsC[totalsCOffset + column - BATCH_C_FIRST];
  }
  return totalsD[totalsDOffset + column - BATCH_D_FIRST];
}
${me(this.alternative)}
fn writeBlock(offset: u32, statisticBlock: array<f32, 8>, enabled: bool) {
  for (var field = 0u; field < 8u; field++) {
    results[resultsOffset + offset + field] = select(getQuietNaN(field), statisticBlock[field], enabled);
  }
}`,body:he(a,!!E)})];return l.counts&&Ne.push(r(t,{id:`${i}-reference-counts`,operation:F,invocationCount:1,bindings:[{name:`counters`,view:Z,type:`u32`,access:`read`},{name:`totalsB`,view:L,type:`f32`,access:`read`},{name:`counts`,view:l.counts,type:`u32`,access:`read_write`}],body:`counts[countsOffset] = counters[countersOffset]; counts[countsOffset + 1u] = u32(totalsB[totalsBOffset + ${R.island-V.first}u]); counts[countsOffset + 2u] = u32(totalsB[totalsBOffset + ${R.black-V.first}u]);`})),Ne}};function Y(e,t){let n=t??`two-sided`;switch(n){case`two-sided`:case`greater`:return n;case`lesser`:return n;default:throw Error(`${e} alternative must be 'two-sided', 'greater' or 'lesser'`)}}function le(e){return e.replace(/([A-Z])/g,`_$1`).toUpperCase()}function X(t,n){let{id:i,rows:a,blockCount:o,first:s,count:c}=n,l=c*o,u=e(t,`${i}-block-offsets`,`uint32`,l+1),d=e(t,`${i}-column-offsets`,`uint32`,c+1),f=e(t,`${i}-partials`,`float32`,l);return[r(t,{id:`${i}-offsets`,operation:F,variant:`sum-offsets`,bindings:[{name:`blockOffsets`,view:u,type:`u32`,access:`read_write`},{name:`columnOffsets`,view:d,type:`u32`,access:`read_write`}],invocationCount:l+1,declarations:`const ROWS: u32 = ${a}u;
const BLOCK_ROWS: u32 = ${I}u;
const BLOCK_COUNT: u32 = ${o}u;
const FIRST_COLUMN: u32 = ${s}u;
const COLUMN_COUNT: u32 = ${c}u;`,body:`let column = index / BLOCK_COUNT;
  let block = index % BLOCK_COUNT;
  blockOffsets[blockOffsetsOffset + index] = (FIRST_COLUMN + column) * ROWS + min(block * BLOCK_ROWS, ROWS);
  if (index <= COLUMN_COUNT) {
    columnOffsets[columnOffsetsOffset + index] = index * BLOCK_COUNT;
  }`}),v(t,{id:`${i}-blocks`,operation:F,segmentCount:l,input:n.matrix,segmentOffsets:u,output:f}),v(t,{id:`${i}-columns`,operation:F,segmentCount:c,input:f,segmentOffsets:d,output:n.totals})]}var ue=`
fn getFinite(value: f32) -> f32 {
  return select(getQuietNaN(0u), value, isFiniteFloat(value));
}

// Division that never relies on the implementation-defined result of x / 0.
fn divide(numerator: f32, denominator: f32) -> f32 {
  return select(getQuietNaN(0u), getFinite(numerator / denominator), denominator != 0.0 && isFiniteFloat(denominator));
}

fn getZScore(statistic: f32, expected: f32, variance: f32) -> f32 {
  let z = (statistic - expected) / sqrt(variance);
  return select(getQuietNaN(0u), z, variance > 0.0 && isFiniteFloat(z));
}

// The moment formulas below are esda's with numerator and denominator divided by S0^2, so the
// weights enter only as the ratios r1 = S1 / S0^2 and r2 = S2 / S0^2 and the product n^3 S0^2
// (above the f32 range for about 1e7 rows with tens of neighbors) is never formed.

// esda.Moran: (expected, VI_norm, VI_rand).
fn getMoranMoments(count: u32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec3<f32> {
  let n = f32(count);
  let n2 = n * n;
  let r1 = s1 / s0 / s0;
  let r2 = s2 / s0 / s0;
  let expected = -divideSpatialCounts(1u, count - min(count, 1u));
  let varianceNormality = (n2 * r1 - n * r2 + 3.0) / (n2 - 1.0) - expected * expected;
  let a = n * ((n2 - 3.0 * n + 3.0) * r1 - n * r2 + 3.0);
  let b = kurtosis * ((n2 - n) * r1 - 2.0 * n * r2 + 6.0);
  let varianceRandomization =
    (a - b) / (f32(count - min(count, 1u)) * f32(count - min(count, 2u)) * f32(count - min(count, 3u))) - expected * expected;
  return vec3<f32>(expected, varianceNormality, varianceRandomization);
}

// esda.Geary: (VC_norm, VC_rand).
fn getGearyMoments(count: u32, s0: f32, s1: f32, s2: f32, kurtosis: f32) -> vec2<f32> {
  let n = f32(count);
  let n2 = n * n;
  let r1 = s1 / s0 / s0;
  let r2 = s2 / s0 / s0;
  let varianceNormality = ((2.0 * r1 + r2) * f32(count - min(count, 1u)) - 4.0) / (2.0 * (n + 1.0));
  let a = f32(count - min(count, 1u)) * r1 * (n2 - 3.0 * n + 3.0 - f32(count - min(count, 1u)) * kurtosis);
  let b = 0.25 * f32(count - min(count, 1u)) * r2 * (n2 + 3.0 * n - 6.0 - (n2 - n + 2.0) * kurtosis);
  let c = n2 - 3.0 - f32(count - min(count, 1u)) * f32(count - min(count, 1u)) * kurtosis;
  let varianceRandomization = (a - b + c) / (n * f32(count - min(count, 2u)) * f32(count - min(count, 3u)));
  return vec2<f32>(varianceNormality, varianceRandomization);
}

// Randomization variance of x'Wx (zero diagonal) with x = mean + z, sum z = 0:
// mean^2 Var(d'z) + 2 mean Cov(d'z, z'Wz) + Var(z'Wz), d_i = w_i. + w_.i, Vd = sum (d - mean d)^2.
fn getQuadraticFormVariance(
  count: u32, s0: f32, s1: f32, s2: f32, degreeDeviation: f32, mean: f32,
  sumSquares: f32, sumCubes: f32, sumFourths: f32
) -> f32 {
  let n = f32(count);
  // A constant input has an invariant quadratic form under every permutation.
  if (sumSquares == 0.0) { return 0.0; }
  let kurtosis = multiplyBySpatialCount(sumFourths / sumSquares, count) / sumSquares;
  let moranVariance = getMoranMoments(count, s0, s1, s2, kurtosis).z;
  let scale = divideBySpatialCount(s0 * sumSquares, count);
  let linearVariance = sumSquares * degreeDeviation / f32(count - min(count, 1u));
  let covariance = -sumCubes * degreeDeviation / (f32(count - min(count, 1u)) * f32(count - min(count, 2u)));
  return mean * mean * linearVariance + 2.0 * mean * covariance + moranVariance * scale * scale;
}
`,de=`let degree = matrix[matrixOffset + getMatrixIndex(COLUMN_ROW_WEIGHT, index)] + columnWeight;
  let binaryDegree = matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_OUT_DEGREE, index)] + f32(inDegree);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE, index)] = select(0.0, degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE, index)] = select(0.0, binaryDegree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_COLUMN_LAG, index)] = select(0.0, columnLag, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_DEGREE_SQUARE, index)] = select(0.0, degree * degree, included);
  matrix[matrixOffset + getMatrixIndex(COLUMN_BINARY_DEGREE_SQUARE, index)] =
    select(0.0, binaryDegree * binaryDegree, included);`;function fe(e){return m(e).filter((e,t)=>t%4!=3&&e!==void 0)}function pe(e){return e?`fn getRowScale(row: u32) -> f32 {
  return rowScale[rowScaleOffset + row];
}`:`fn getRowScale(row: u32) -> f32 {
  return 1.0;
}`}function me(e){let t=`0.70710678118654752`;return`const P_VALUE_FLOOR: f32 = ANALYTIC_P_VALUE_FLOOR;
fn getPValue(z: f32) -> f32 {
  return select(getQuietNaN(0u), max(${{"two-sided":`getTwoSidedPValue(z)`,greater:`min(0.5 * erfcValue(z * ${t}), 1.0)`,lesser:`min(0.5 * erfcValue(-z * ${t}), 1.0)`}[e]}, P_VALUE_FLOOR), isFiniteFloat(z));
}`}function he(e,t){let n=t=>e.statistics.includes(t)?t===`getisOrdG`?`!truncated && !hasNegative`:`!truncated`:`false`,r=M;return`let count = counters[countersOffset];
  let n = f32(count);
  let hasNegative = counters[countersOffset + 1u] > 0u;
  let truncated = ${t?`overflow[overflowOffset] != 0u`:`false`};
  // Every moment below is in scaled units (values divided by a power of two near their mean
  // magnitude); only the reported mean and variance are scaled back.
  let inverseScaleX = bitcast<f32>(counters[countersOffset + ${G.inverseX}u]);
  let scaleX = bitcast<f32>(counters[countersOffset + ${G.x}u]);
  // Mean with the second-pass correction sum(x - mean) / n.
  let mean = divideBySpatialCount(readTotal(COLUMN_X), count) * inverseScaleX + divideBySpatialCount(readTotal(COLUMN_DEVIATION_X), count);
  let sumSquares = readTotal(COLUMN_CENTERED_SQUARE);
  let sumCubes = readTotal(COLUMN_CENTERED_CUBE);
  let sumFourths = readTotal(COLUMN_CENTERED_FOURTH);
  let s0 = readTotal(COLUMN_ROW_WEIGHT);
  let s1 = readTotal(COLUMN_S1);
  let s2 = readTotal(COLUMN_DEGREE_SQUARE);
  let blackCount = readTotal(COLUMN_BLACK);
  let degreeDeviation = readTotal(COLUMN_DEGREE_DEVIATION);
  let kurtosis = divide(multiplyBySpatialCount(divide(sumFourths, sumSquares), count), sumSquares);
  let valid = n >= 4.0 && s0 > 0.0 && sumSquares > 0.0;

  results[resultsOffset + ${r.summary}u] = n;
  results[resultsOffset + ${r.summary+1}u] = select(s0, getQuietNaN(1u), truncated);
  results[resultsOffset + ${r.summary+2}u] = select(s1, getQuietNaN(2u), truncated);
  results[resultsOffset + ${r.summary+3}u] = select(s2, getQuietNaN(3u), truncated);
  results[resultsOffset + ${r.summary+4}u] = select(getFinite(mean * scaleX), getQuietNaN(4u), truncated);
  results[resultsOffset + ${r.summary+5}u] =
    select(getFinite(divideBySpatialCount(sumSquares, count) * scaleX * scaleX), getQuietNaN(5u), truncated);
  results[resultsOffset + ${r.summary+6}u] = blackCount;
  results[resultsOffset + ${r.summary+7}u] = readTotal(COLUMN_ISLAND);

  // Moran's I.
  {
    let statistic = divide(multiplyBySpatialCount(readTotal(COLUMN_MORAN_CROSS), count), s0 * sumSquares);
    let moments = getMoranMoments(count, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.z), valid);
    let zNormality = getZScore(statistic, moments.x, varianceNormality);
    let zRandomization = getZScore(statistic, moments.x, varianceRandomization);
    // E[I] = -1 / (n - 1) is undefined below two rows (n = 0 would give 1).
    let expected = select(getQuietNaN(1u), getFinite(moments.x), n >= 2.0);
    writeBlock(${r.moran}u, array<f32, 8>(statistic, expected, varianceNormality,
      zNormality, getPValue(zNormality), varianceRandomization, zRandomization,
      getPValue(zRandomization)), ${n(`moran`)});
  }
  // Geary's C.
  {
    let statistic = divide(multiplyBySpatialCount(readTotal(COLUMN_GEARY_SUM), count - min(count, 1u)), 2.0 * s0 * sumSquares);
    let moments = getGearyMoments(count, s0, s1, s2, kurtosis);
    let varianceNormality = select(getQuietNaN(1u), getFinite(moments.x), valid);
    let varianceRandomization = select(getQuietNaN(1u), getFinite(moments.y), valid);
    let zNormality = getZScore(statistic, 1.0, varianceNormality);
    let zRandomization = getZScore(statistic, 1.0, varianceRandomization);
    writeBlock(${r.geary}u, array<f32, 8>(statistic, select(getQuietNaN(1u), 1.0, n >= 2.0),
      varianceNormality, zNormality,
      getPValue(zNormality), varianceRandomization, zRandomization, getPValue(zRandomization)),
      ${n(`geary`)});
  }
  // Getis-Ord General G.
  {
    let denominator = n * f32(count - min(count, 1u)) * mean * mean - sumSquares;
    let statistic = divide(readTotal(COLUMN_G_CROSS), denominator);
    let expected = divide(s0, n * f32(count - min(count, 1u)));
    let variance = select(getQuietNaN(1u), getFinite(getQuadraticFormVariance(count, s0, s1, s2,
      degreeDeviation, mean, sumSquares, sumCubes, sumFourths) / denominator / denominator),
      n >= 4.0 && s0 > 0.0 && denominator > 0.0);
    let z = getZScore(statistic, expected, variance);
    // esda calls these VG / z_norm; retain the randomization aliases for compatibility.
    writeBlock(${r.getisOrdG}u, array<f32, 8>(statistic, expected, variance,
      z, getPValue(z), variance, z, getPValue(z)), ${n(`getisOrdG`)});
  }
  // Bivariate Moran's I, randomization of y.
  {
    let sumSquaresY = readTotal(COLUMN_CENTERED_SQUARE_Y);
    let scale = divideSpatialCountByValue(count, s0);
    let statistic = divide(scale * readTotal(COLUMN_BIVARIATE_CROSS), sqrt(sumSquares * sumSquaresY));
    let variance = select(getQuietNaN(1u), getFinite(scale * scale *
      readTotal(COLUMN_COLUMN_LAG_DEVIATION) / (f32(count - min(count, 1u)) * sumSquares)), valid);
    let z = getZScore(statistic, 0.0, variance);
    writeBlock(${r.bivariateMoran}u, array<f32, 8>(statistic, 0.0, getQuietNaN(2u),
      getQuietNaN(2u), getQuietNaN(2u), variance, z, getPValue(z)), ${n(`bivariateMoran`)});
  }
  // Join counts over the binarized weights.
  {
    let enabled = ${n(`joinCount`)};
    let blackBlack2 = counters[countersOffset + 2u];
    let blackWhite2 = counters[countersOffset + 3u];
    let whiteWhite2 = counters[countersOffset + 4u];
    ${e.output.joinCounts?`joinCounts[joinCountsOffset] = select(0u, blackBlack2, enabled);
    joinCounts[joinCountsOffset + 1u] = select(0u, blackWhite2, enabled);
    joinCounts[joinCountsOffset + 2u] = select(0u, whiteWhite2, enabled);`:``}
    let b0 = readTotal(COLUMN_BINARY_OUT_DEGREE);
    let b1 = readTotal(COLUMN_BINARY_S1);
    let b2 = readTotal(COLUMN_BINARY_DEGREE_SQUARE);
    let p = divide(blackCount, n);
    let q = 1.0 - p;
    let blackSquares = n * p * q;
    let blackCubes = blackSquares * (q - p);
    let blackFourths = blackSquares * (p * p * p + q * q * q);
    let binaryValid = n >= 4.0 && b0 > 0.0;
    // A single colour makes every join count constant: zero variance.
    let oneColour = blackCount == 0.0 || blackCount == n;
    let expectedBlackBlack = getFinite(0.5 * b0 * blackCount * (blackCount - 1.0) / (n * f32(count - min(count, 1u))));
    let varianceBlackBlack = select(getQuietNaN(3u), getFinite(0.25 * getQuadraticFormVariance(count,
      b0, b1, b2, readTotal(COLUMN_BINARY_DEGREE_DEVIATION), p, blackSquares, blackCubes,
      blackFourths)), binaryValid);
    let varianceBlackBlackOrZero = select(varianceBlackBlack, 0.0, binaryValid && oneColour);
    let expectedBlackWhite = getFinite(b0 * blackSquares / f32(count - min(count, 1u)));
    let blackKurtosis = n * (blackFourths / blackSquares) / blackSquares;
    let blackWhiteScale = b0 * blackSquares / f32(count - min(count, 1u));
    let varianceBlackWhite = select(getQuietNaN(3u), getFinite(getGearyMoments(count, b0, b1, b2,
      blackKurtosis).y * blackWhiteScale * blackWhiteScale), binaryValid && !oneColour);
    let varianceBlackWhiteOrZero = select(varianceBlackWhite, 0.0, binaryValid && oneColour);
    let blackBlack = 0.5 * f32(blackBlack2);
    let blackWhite = 0.5 * f32(blackWhite2);
    let zBlackBlack = getZScore(blackBlack, expectedBlackBlack, varianceBlackBlackOrZero);
    let zBlackWhite = getZScore(blackWhite, expectedBlackWhite, varianceBlackWhiteOrZero);
    let statisticBlock = array<f32, 12>(blackBlack, blackWhite, 0.5 * f32(whiteWhite2), 0.5 * b0,
      expectedBlackBlack, varianceBlackBlackOrZero, zBlackBlack, getPValue(zBlackBlack),
      expectedBlackWhite, varianceBlackWhiteOrZero, zBlackWhite, getPValue(zBlackWhite));
    for (var field = 0u; field < 12u; field++) {
      results[resultsOffset + ${r.joinCount}u + field] = select(getQuietNaN(field), statisticBlock[field], enabled);
    }
  }`}export{P as i,M as n,N as r,J as t};