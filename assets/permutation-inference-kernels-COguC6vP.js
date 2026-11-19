import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{t as r}from"./gpu-reduction-tNgOVQ_C.js";import{o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as o}from"./sorted-segment-sums-Dvmm6OdR.js";import{o as s,r as c}from"./spatial-weights-CqWysstN.js";import{i as l,n as u,r as d,s as f}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{t as p}from"./spatial-autocorrelation-kernels-CCrGBZuG.js";import{f as m,v as h}from"./gpu-contributor-utils-BIwLQON5.js";import{n as g,t as _}from"./float32-values-60_XMjYS.js";import{a as v}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as y}from"./graph-chunk-groups-BB8ACpQO.js";import{t as b}from"./spatial-statistics-row-dispatch-DNaskhxv.js";function x(e,t={}){let{seed:n,permutationCount:r}=e;if(!Number.isSafeInteger(n)||n<0)throw Error(`permutation seed must be a non-negative safe integer`);if(m(`getGPUPermutationParameterValues`,`permutationCount`,r,1,2**31-1),t.maximumPermutations!==void 0&&r>t.maximumPermutations)throw Error(`permutationCount exceeds maximumPermutations`);let i=e.significanceLevel??.05;if(!(i>0&&i<1))throw Error(`significanceLevel must be in (0, 1)`);let a=new Uint32Array(4);return a[0]=n%2**32,a[1]=Math.floor(n/2**32),a[2]=r,a[3]=new Uint32Array(new Float32Array([i]).buffer)[0],a}var S=4096,C=4294963199,w=4294967295,T={count:0,sumX:1,sumY:2,sumSquares:3,sumSquaresY:4,s0:5,mean:6,meanY:7,s0Squared:8};function E(e){let{id:n}=e,r=f(n,e.weights);if(m(n,`weights row count`,r,2,C),g(e.values,`${n} values`)!==r)throw Error(`${n} values length must equal the weights row count`);if(e.secondValues&&g(e.secondValues,`${n} secondValues`)!==r)throw Error(`${n} secondValues length must equal the weights row count`);if(e.mask&&(t(e.mask,`${n} mask`),e.mask.length!==r))throw Error(`${n} mask length must equal the weights row count`);e.analysisCount&&(t(e.analysisCount,`${n} output.analysisCount`),m(n,`output.analysisCount length`,e.analysisCount.length,1,1)),e.islandCount&&(t(e.islandCount,`${n} output.islandCount`),m(n,`output.islandCount length`,e.islandCount.length));let i=e.weights.status?.overflow;return i&&t(i,`${n} weights.status.overflow`),t(e.parameters,`${n} parameters`),m(n,`parameters length`,e.parameters.length,4),m(n,`maximumPermutations`,e.maximumPermutations,1,2**20),r}function D(e,t,n=[]){y(e,t.id,8),v(e,t.id,Object.fromEntries([...l(t.weights),t.weights.status?.overflow,t.values,t.secondValues,t.mask,t.parameters,t.islandCount,t.analysisCount,...n].map((e,t)=>[`binding ${t}`,e]))),h(e.device,t.id,`row positions scratch`,(t.values.length+1)*4,`reduce packed rows or request larger device bindings`)}function O(e){return`
const MAXIMUM_PERMUTATIONS: u32 = ${e}u;
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutationCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn isPermutationCountValid() -> bool {
  let requested = parameters[parametersOffset + 2u];
  return requested >= 1u && requested <= MAXIMUM_PERMUTATIONS;
}
fn readValidPermutationCount() -> u32 {
  return select(0u, parameters[parametersOffset + 2u], isPermutationCountValid());
}
fn readSignificanceLevel() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
`}var k=p;function A(t,r){D(t,r);let{id:o,operation:l,weights:f,mask:p}=r,m=_(t,`${o}-values`,r.values),h=r.secondValues?_(t,`${o}-second-values`,r.secondValues):void 0,g=m.view,v=h?.view,y=(r.islands??`exclude`)===`keep`,x=s(f),C=f.status?.overflow,T=g.length,E=Math.ceil(T/S),O=Array.from({length:8},(n,r)=>e(t,`${o}-column-${r}`,`float32`,T)),A=e=>({name:`column${e}`,view:O[e],type:`f32`,access:`read_write`}),M=e(t,`${o}-flags`,`uint32`,T),N=e(t,`${o}-analysis-flags`,`uint32`,T),P=e(t,`${o}-scanned`,`uint32`,T),F=e(t,`${o}-row-positions`,`uint32`,T+1),I=e(t,`${o}-compact-x`,`float32`,T),L=e(t,`${o}-compact-y`,`float32`,T),R=r.islandCount??e(t,`${o}-island-count`,`uint32`,1),z=u(f),B=z.length===1&&z[0].rowCount===T,V=z.map((n,r)=>({...n,effective:x?e(t,`${o}-scaled-weights${B?``:`-${r}`}`,`float32`,n.weights.neighbors.length):n.weights.weights})),H=V[0].effective,U=e(t,`${o}-totals-a`,`float32`,2),W=e(t,`${o}-totals-b`,`float32`,2),G=e(t,`${o}-totals-c`,`float32`,4),K=e(t,`${o}-totals`,`float32`,9),q=`const ROWS: u32 = ${T}u;
const INVALID_POSITION: u32 = ${w}u;
${k}`,J=(e,t)=>({idSuffix:B?``:`-${t}`,invocationCount:e.rowCount,declarations:q,rowStart:`WEIGHTS_ROW_START`,wrapBody:e=>e}),Y=(e,t=!1)=>[{name:`offsets`,view:e.weights.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:e.weights.neighbors,type:`u32`,access:`read`},{name:`weights`,view:t?e.effective:e.weights.weights,type:`f32`,access:`read`}],X=v?[{name:`secondValues`,view:v,type:`f32`,access:`read`}]:[],Z=[...m.nodes,...h?.nodes??[],i(t,{id:`${o}-flags`,operation:l,variant:`flags`,bindings:[{name:`values`,view:g,type:`f32`,access:`read`},...X,...p?[{name:`mask`,view:p,type:`u32`,access:`read`}]:[],{name:`flags`,view:M,type:`u32`,access:`read_write`}],invocationCount:T,declarations:q,body:`let x = values[valuesOffset + index];
  let y = ${v?`secondValues[secondValuesOffset + index]`:`0.0`};
  let included = ${p?`mask[maskOffset + index] != 0u &&`:``} isFiniteFloat(x) && isFiniteFloat(y);
  flags[flagsOffset + index] = select(0u, 1u, included);`}),a(t,{id:`${o}-island-clear`,operation:l,view:R,type:`u32`,value:`0u`,componentCount:1}),...V.map((e,n)=>{let r=J(e,n);return b(t,{rowBase:e.rowStart,id:`${o}-analysis${r.idSuffix}`,operation:l,variant:y?`analysis-keep`:`analysis-exclude`,bindings:[...Y(e),{name:`flags`,view:M,type:`u32`,access:`read`},{name:`analysisFlags`,view:N,type:`u32`,access:`read_write`},{name:`islandCount`,view:R,type:`atomic<u32>`,access:`read_write`},...C?[{name:`weightsOverflow`,view:C,type:`u32`,access:`read`}]:[]],invocationCount:r.invocationCount,declarations:r.declarations,body:r.wrapBody(`let included = flags[flagsOffset + index] != 0u;
  var hasNeighbor = false;
  if (included) {
    ${c(`flags[flagsOffset + neighbor] != 0u`,`hasNeighbor = true; break;`,`index`,r.rowStart)}
  }
  if (included && !hasNeighbor) {
    atomicAdd(&islandCount[islandCountOffset], 1u);
  }
  let truncated = ${C?`weightsOverflow[weightsOverflowOffset] != 0u`:`false`};
  let analyzed = included && !truncated && ${y?`true`:`hasNeighbor`};
  analysisFlags[analysisFlagsOffset + index] = select(0u, 1u, analyzed);`)})}),...d(f).map((e,n)=>i(t,{id:`${o}-incomplete-analysis-${n}`,operation:l,variant:`incomplete-input`,bindings:[{name:`loss`,view:e,type:`u32`,access:`read`},{name:`analysisFlags`,view:N,type:`u32`,access:`read_write`}],invocationCount:T,body:`if (loss[lossOffset] != 0u) { analysisFlags[analysisFlagsOffset + index] = 0u; }`})),i(t,{id:`${o}-moment-inputs`,operation:l,variant:`moment-inputs`,bindings:[{name:`values`,view:g,type:`f32`,access:`read`},...X,{name:`analysisFlags`,view:N,type:`u32`,access:`read`},A(0),A(1)],invocationCount:T,declarations:q,body:`let analyzed = analysisFlags[analysisFlagsOffset + index] != 0u;
  column0[column0Offset + index] =
    select(0.0, values[valuesOffset + index], analyzed);
  column1[column1Offset + index] =
    select(0.0, ${v?`secondValues[secondValuesOffset + index]`:`0.0`}, analyzed);`}),...j(t,{id:`${o}-sum-a`,operation:l,columns:O,rows:T,blockCount:E,first:0,count:2,totals:U}),...new n({id:`${o}-scan`,input:N,output:P,mode:`exclusive`}).getCommandNodes(t),i(t,{id:`${o}-positions`,operation:l,variant:`positions`,bindings:[{name:`analysisFlags`,view:N,type:`u32`,access:`read`},{name:`scanned`,view:P,type:`u32`,access:`read`},{name:`rowPositions`,view:F,type:`u32`,access:`read_write`}],invocationCount:T+1,declarations:q,body:`if (index < ROWS) {
    rowPositions[rowPositionsOffset + index] =
      select(INVALID_POSITION, scanned[scannedOffset + index], analysisFlags[analysisFlagsOffset + index] != 0u);
  } else {
    rowPositions[rowPositionsOffset + ROWS] =
      scanned[scannedOffset + ROWS - 1u] + analysisFlags[analysisFlagsOffset + ROWS - 1u];
  }`}),i(t,{id:`${o}-deviations`,operation:l,variant:`deviations`,bindings:[{name:`rowPositions`,view:F,type:`u32`,access:`read`},{name:`totalsA`,view:U,type:`f32`,access:`read`},A(0),A(1),A(2),A(3)],invocationCount:T,declarations:q,body:`let included = rowPositions[rowPositionsOffset + index] != INVALID_POSITION;
  let count = rowPositions[rowPositionsOffset + ROWS];
  column2[column2Offset + index] =
    select(0.0, column0[column0Offset + index] - divideBySpatialCount(totalsA[totalsAOffset], count), included);
  column3[column3Offset + index] =
    select(0.0, column1[column1Offset + index] - divideBySpatialCount(totalsA[totalsAOffset + 1u], count), included);`}),...j(t,{id:`${o}-sum-b`,operation:l,columns:O,rows:T,blockCount:E,first:2,count:2,totals:W}),...[0,1].map(e=>i(t,{id:`${o}-compact-${e}`,operation:l,variant:`compact`,bindings:[{name:`rowPositions`,view:F,type:`u32`,access:`read`},{name:`totalsB`,view:W,type:`f32`,access:`read`},{name:`rawValues`,view:O[e],type:`f32`,access:`read`},{name:`deviations`,view:O[e+2],type:`f32`,access:`read`},{name:`compact`,view:e===0?I:L,type:`f32`,access:`read_write`},{name:`squares`,view:O[e+4],type:`f32`,access:`read_write`}],invocationCount:T,declarations:`${q}\nconst VARIABLE: u32 = ${e}u;`,body:`let position = rowPositions[rowPositionsOffset + index];
  let included = position != INVALID_POSITION;
  let count = rowPositions[rowPositionsOffset + ROWS];
  let centered = deviations[deviationsOffset + index] - divideBySpatialCount(totalsB[totalsBOffset + VARIABLE], count);
  if (included) {
    compact[compactOffset + position] = ${(e===0?r.centerX:r.centerY!==!1)?`centered`:`rawValues[rawValuesOffset + index]`};
  }
  squares[squaresOffset + index] = select(0.0, centered * centered, included);`})),...V.map((e,n)=>{let r=J(e,n);return b(t,{rowBase:e.rowStart,id:`${o}-row-weights${r.idSuffix}`,operation:l,variant:x?`row-weights-standardize`:`row-weights`,bindings:[...Y(e),{name:`rowPositions`,view:F,type:`u32`,access:`read`},A(6),A(7),...x?[{name:`scaledWeights`,view:e.effective,type:`f32`,access:`read_write`}]:[]],invocationCount:r.invocationCount,declarations:r.declarations,body:r.wrapBody(`var rowWeight = 0.0;
  if (rowPositions[rowPositionsOffset + index] != INVALID_POSITION) {
    ${c(`rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION`,`rowWeight += weight;`,`index`,r.rowStart)}
  }
  // Row-standardized weights are re-standardized over the active slots: a row with any active slot sums to 1.
  column6[column6Offset + index] = ${x?`select(0.0, 1.0, rowWeight > 0.0)`:`rowWeight`};
  column7[column7Offset + index] = ${x?`select(0.0, 1.0, rowWeight > 0.0)`:`rowWeight * rowWeight`};
  ${x?`let scale = select(0.0, 1.0 / rowWeight, rowWeight > 0.0);
  let begin = min(offsets[offsetsOffset + index${r.rowStart?` - WEIGHTS_ROW_START`:``}], CAPACITY);
  let end = min(offsets[offsetsOffset + index${r.rowStart?` - WEIGHTS_ROW_START`:``} + 1u], CAPACITY);
  for (var slot = begin; slot < end; slot++) {
    scaledWeights[scaledWeightsOffset + slot] = weights[weightsOffset + slot] * scale;
  }`:``}`)})}),...j(t,{id:`${o}-sum-c`,operation:l,columns:O,rows:T,blockCount:E,first:4,count:4,totals:G}),i(t,{id:`${o}-totals`,operation:l,variant:`totals`,bindings:[{name:`rowPositions`,view:F,type:`u32`,access:`read`},{name:`totalsA`,view:U,type:`f32`,access:`read`},{name:`totalsB`,view:W,type:`f32`,access:`read`},{name:`totalsC`,view:G,type:`f32`,access:`read`},{name:`totals`,view:K,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`const ROWS: u32 = ${T}u;
${k}`,body:`let count = rowPositions[rowPositionsOffset + ROWS];
  let meanX = divideBySpatialCount(totalsA[totalsAOffset], count) + divideBySpatialCount(totalsB[totalsBOffset], count);
  let meanY = divideBySpatialCount(totalsA[totalsAOffset + 1u], count) + divideBySpatialCount(totalsB[totalsBOffset + 1u], count);
  totals[totalsOffset] = f32(count);
  totals[totalsOffset + 1u] = multiplyBySpatialCount(meanX, count);
  totals[totalsOffset + 2u] = multiplyBySpatialCount(meanY, count);
  totals[totalsOffset + 3u] = totalsC[totalsCOffset];
  totals[totalsOffset + 4u] = totalsC[totalsCOffset + 1u];
  totals[totalsOffset + 5u] = totalsC[totalsCOffset + 2u];
  totals[totalsOffset + 6u] = meanX;
  totals[totalsOffset + 7u] = meanY;
  totals[totalsOffset + 8u] = totalsC[totalsCOffset + 3u];`})];return r.analysisCount&&Z.push(i(t,{id:`${o}-analysis-count`,operation:l,invocationCount:1,bindings:[{name:`positions`,view:F,type:`u32`,access:`read`},{name:`analysisCount`,view:r.analysisCount,type:`u32`,access:`read_write`}],body:`analysisCount[analysisCountOffset] = positions[positionsOffset + ${T}u];`})),{nodes:Z,rowPositions:F,compactX:I,compactY:L,totals:K,islandCount:R,weights:H,chunks:V.map(e=>({rowStart:e.rowStart,rowCount:e.rowCount,weights:e.weights,effective:e.effective}))}}function j(e,t){return Array.from({length:t.count},(n,r)=>M(e,{...t,id:`${t.id}-${r}`,matrix:t.columns[t.first+r],first:0,count:1,totals:e.createDataView(t.totals.buffer,{format:`float32`,length:1,byteOffset:t.totals.byteOffset+r*4})})).flat()}function M(t,n){let{id:r,operation:a,rows:s,blockCount:c,first:l,count:u}=n,d=u*c,f=e(t,`${r}-block-offsets`,`uint32`,d+1),p=e(t,`${r}-column-offsets`,`uint32`,u+1),m=e(t,`${r}-partials`,`float32`,d);return[i(t,{id:`${r}-offsets`,operation:a,variant:`sum-offsets`,bindings:[{name:`blockOffsets`,view:f,type:`u32`,access:`read_write`},{name:`columnOffsets`,view:p,type:`u32`,access:`read_write`}],invocationCount:d+1,declarations:`const ROWS: u32 = ${s}u;
const BLOCK_ROWS: u32 = ${S}u;
const BLOCK_COUNT: u32 = ${c}u;
const FIRST_COLUMN: u32 = ${l}u;
const COLUMN_COUNT: u32 = ${u}u;`,body:`let column = index / BLOCK_COUNT;
  let blockIndex = index % BLOCK_COUNT;
  blockOffsets[blockOffsetsOffset + index] = (FIRST_COLUMN + column) * ROWS + min(blockIndex * BLOCK_ROWS, ROWS);
  if (index <= COLUMN_COUNT) {
    columnOffsets[columnOffsetsOffset + index] = index * BLOCK_COUNT;
  }`}),o(t,{id:`${r}-blocks`,operation:a,segmentCount:d,input:n.matrix,segmentOffsets:f,output:m}),o(t,{id:`${r}-columns`,operation:a,segmentCount:u,input:m,segmentOffsets:p,output:n.totals})]}function N(t,n){let{id:a,operation:o,rows:s}=n,c=e(t,`${a}-negative-indicators`,`uint32`,s),l=e(t,`${a}-negative-count`,`uint32`,1);return{negativeCount:l,nodes:[i(t,{id:`${a}-negative-indicators`,operation:o,variant:`negative-indicators`,bindings:[{name:`compactX`,view:n.compactX,type:`f32`,access:`read`},{name:`rowPositions`,view:n.rowPositions,type:`u32`,access:`read`},{name:`indicators`,view:c,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`const ROWS: u32 = ${s}u;`,body:`indicators[indicatorsOffset + index] = select(0u, 1u, index < rowPositions[rowPositionsOffset + ROWS] && compactX[compactXOffset + index] < 0.0);`}),...new r({id:`${a}-negative-sum`,input:c,output:l,operation:`sum`}).getCommandNodes(t)]}}export{M as a,O as c,x as d,T as i,D as l,k as n,N as o,w as r,A as s,S as t,E as u};