import{t as e}from"./regression-output-dispatches-KKVTWkHP.js";import{a as t,c as n,n as r,o as i,r as a}from"./regression-binding-plans-DG-hiroU.js";import{c as o,d as s,f as c,i as l,m as u,o as d,v as f}from"./spatial-likelihood-allocations-BGEHbqkB.js";import{a as ee}from"./graph-chunk-dispatch-zBt8k2-o.js";import{r as p}from"./kernel-C3tfQOn4.js";import{r as te}from"./wgsl-common-DfyXiYB0.js";import{o as ne}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as m,S as h,f as g,x as re,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{t as ie}from"./gpu-spatial-weights-transpose-BOTZ1Oz3.js";import{a as v,s as y}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{n as b,t as ae}from"./chunked-weights-kernels-BYYFtXmd.js";import{t as oe}from"./pairwise-sum-wgsl-iErcC5Or.js";var x=`GPUSpatialWeightsTraces`,S=64,C=32,w=1024,T=4096;function E(e,t,n){if(n&&(m(n,[`uint32`],`${e} ${t}`),n.length<1))throw Error(`${e} ${t} must hold 1 uint32`)}var D=class{id;props;rowCount;mode;probeTotal;batchProbeCount;tileRowCount;tileCount;includeGramTraces;constructor(n){n=i(n,[`id`,`weights`,`maximumPower`,`mode`,`probeCount`,`parameters`,`includeGramTraces`,`tileRowCount`,`probeBatchSize`,`output`],[`powerTraces`,`powerSums`,`powerTraceStandardErrors`,`gramTraces`,`infinityNorm`,`status`],`spatial-weights-traces`);let r=n.id??`spatial-weights-traces`;this.id=r,this.props=n;let{output:a,maximumPower:o}=n,s=y(r,n.weights);if(g(r,`maximumPower`,o,1,256),n.mode!==void 0&&n.mode!==`exact`&&n.mode!==`stochastic`)throw Error(`${r} mode must be 'exact' or 'stochastic'`);let l=n.mode??(s<=4096?`exact`:`stochastic`);if(l===`exact`&&s>16384)throw Error(`${r} exact mode supports at most ${c} rows; use mode 'stochastic'`);let d=n.probeCount??64;if(g(r,`probeCount`,d),E(r,`parameters`,n.parameters),n.tileRowCount!==void 0&&g(r,`tileRowCount`,n.tileRowCount),m(a.powerTraces,[`float32`],`${r} output.powerTraces`),a.powerTraces.length<o)throw Error(`${r} output.powerTraces must hold maximumPower float32 values`);if(a.powerTraceStandardErrors&&(m(a.powerTraceStandardErrors,[`float32`],`${r} output.powerTraceStandardErrors`),a.powerTraceStandardErrors.length<o))throw Error(`${r} output.powerTraceStandardErrors must hold maximumPower float32 values`);if(m(a.powerSums,[`float32`],`${r} output.powerSums`),a.powerSums.length<o)throw Error(`${r} output.powerSums must hold maximumPower float32 values`);let f=n.includeGramTraces??a.gramTraces!==void 0;if(f!==(a.gramTraces!==void 0))throw Error(`${r} output.gramTraces is required exactly when includeGramTraces is true`);if(a.gramTraces&&(m(a.gramTraces,[`float32`],`${r} output.gramTraces`),a.gramTraces.length<4))throw Error(`${r} output.gramTraces must hold 4 float32 values`);if(a.infinityNorm&&(m(a.infinityNorm,[`float32`],`${r} output.infinityNorm`),a.infinityNorm.length<1))throw Error(`${r} output.infinityNorm must hold one float32 value`);if(a.status?.approximation&&(m(a.status.approximation,[`uint32`],`${r} output.status.approximation`),a.status.approximation.length<1))throw Error(`${r} output.status.approximation must hold one uint32 row`);if(this.rowCount=s,this.mode=l,this.includeGramTraces=f,this.probeTotal=l===`exact`?s:d,n.probeBatchSize!==void 0&&g(r,`probeBatchSize`,n.probeBatchSize),this.batchProbeCount=Math.min(this.probeTotal,n.probeBatchSize??1/0,Math.max(1,Math.floor(u/s))),this.tileRowCount=n.tileRowCount??Math.max(C,Math.ceil(s/w)),this.tileCount=ae(n.weights,this.tileRowCount).tileCount,h(r,`tileCount`,this.tileCount,T),f&&v(n.weights))throw Error(`${r} Gram traces need whole weights`);e(this.id,{...a,status:void 0,approximation:a.status?.approximation},[{read:[`infinityNorm`],write:[`powerTraces`,`powerTraceStandardErrors`,`gramTraces`]}]),_(r,[a.powerTraces,a.powerTraceStandardErrors,a.powerSums,a.gramTraces,a.status?.approximation,a.infinityNorm],[...Object.values(t(n.weights)),n.parameters])}getCommandNodes(e){let{id:i,props:c,rowCount:u,probeTotal:m,batchProbeCount:h,tileRowCount:g,tileCount:_}=this,{weights:v,output:y,maximumPower:C}=c,w=this.mode===`stochastic`,T=this.includeGramTraces;ee(e,i,{...t(v),...y,approximation:y.status?.approximation,status:void 0,parameters:c.parameters}),l(e.device,i,a({rowCount:u,edgeCount:r(v),maximumPower:C,tileCount:_,mode:this.mode,probeCount:c.probeCount,probeBatchSize:c.probeBatchSize,includeGram:T})),re(i,e,[...Object.values(t(v)),y.powerTraces,y.powerTraceStandardErrors,y.powerSums,y.gramTraces,y.status?.approximation,y.infinityNorm,c.parameters]);let E=T?Math.max(C,2):C,D=E,se=D,ce=D+1,O=D+(T?2:0),k=d(e,i,x,m,h,c.parameters),A=s(_),j=(t,n)=>p(e,`${i}-${t}`,`float32`,n),M=`const ROWS: u32 = ${u}u;
const TILE_ROWS: u32 = ${g}u;
const TILE_COUNT: u32 = ${_}u;
const PROBE_TOTAL: u32 = ${m}u;`,N=(e,t,n)=>({name:e,view:t,type:`f32`,access:n}),P=(e,t,n)=>({name:e,view:t,type:`u32`,access:n}),F=(t,n,r,a,o,s,c)=>ne(e,{id:`${i}-${t}`,operation:x,variant:t.replace(/-\d+(-\d+)?$/,``),bindings:n,invocationCount:r,declarations:a,body:o,...s?{constants:s}:{},...c?{workgroupSize:c}:{}}),I=(e,t)=>[P(e?`${e}Offsets`:`offsets`,t.offsets,`read`),P(e?`${e}Neighbors`:`neighbors`,t.neighbors,`read`),N(e?`${e}Weights`:`weights`,t.weights,`read`)],L=(e,t,n,r)=>oe({count:r,term:`${e}[${e}Offset + ${t} + sumItem * ${n}]`}),R=[],{dispatches:z}=ae(v,g),B=v,le=T?I(``,B):[],V=y.infinityNorm??j(`infinity-norm`,1),H=j(`norm-partial`,_),U=z.map(e=>j(`scaled-weights${e.idSuffix}`,e.weights.weights.length));R.push(...z.map(e=>F(`norm-tiles${e.idSuffix}`,[...I(``,e.weights),N(`normPartial`,H,`read_write`)],e.chunkTileCount,`${M}${e.tileConstantsWGSL?`\n${e.tileConstantsWGSL}`:``}`,b(e,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] >= ROWS) {
        continue;
      }
      sum += abs(weights[weightsOffset + slot]);
    }
    largest = max(largest, sum);
  }
  normPartial[normPartialOffset + index] = largest;`),void 0,S)),F(`norm-merge`,[N(`normPartial`,H,`read`),N(`infinityNorm`,V,`read_write`)],1,M,`var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, normPartial[normPartialOffset + tile]);
  }
  // An empty or non-finite W scales by 1 (non-finite weights then propagate through the chains).
  infinityNorm[infinityNormOffset] = select(1.0, largest, largest > 0.0 && largest < 3.0e38);`),...z.map((e,t)=>F(`scale-weights${e.idSuffix}`,[N(`weights`,e.weights.weights,`read`),N(`infinityNorm`,V,`read`),N(`scaledWeights`,U[t],`read_write`)],e.weights.weights.length,``,`scaledWeights[scaledWeightsOffset + index] = weights[weightsOffset + index] / infinityNorm[infinityNormOffset];`)));let W=e=>I(``,{offsets:z[e].weights.offsets,neighbors:z[e].weights.neighbors,weights:U[e]}),G=j(`sums-partial`,C*A),K=[j(`ones-a`,u),j(`ones-b`,u)];for(let t=1;t<=C;t++){let n=K[(t-2+2)%2],r=K[(t-1)%2],i=t===1;R.push(...z.map((a,s)=>F(`${i?`sums-step-first`:`sums-step-${t}`}${a.idSuffix}`,[...W(s),...i?[]:[N(`vectorIn`,n,`read`)],N(`vectorOut`,r,`read_write`),N(`sumsPartial`,o(e,G,(t-1)*A,_),`read_write`)],a.chunkTileCount,`${M}${a.tileConstantsWGSL?`\n${a.tileConstantsWGSL}`:``}`,b(a,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var tileSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${i?`1.0`:`vectorIn[vectorInOffset + neighbor]`};
    }
    vectorOut[vectorOutOffset + row] = sum;
    tileSum += sum;
  }
  sumsPartial[sumsPartialOffset + index] = tileSum;`),void 0,S)))}R.push(F(`sums-merge`,[N(`sumsPartial`,G,`read`),N(`powerSums`,y.powerSums,`read_write`)],C,`${M}
const SUMS_STRIDE: u32 = ${A}u;`,`${L(`sumsPartial`,`index * SUMS_STRIDE`,`1u`,`TILE_COUNT`)}
  powerSums[powerSumsOffset + index] = sum;`));let q,J=T?j(`gram-partial`,_*2):void 0;T&&J&&(q={offsets:p(e,`${i}-transpose-offsets`,`uint32`,u+1),neighbors:p(e,`${i}-transpose-neighbors`,`uint32`,B.neighbors.length),weights:j(`transpose-weights`,B.neighbors.length)},R.push(...new ie({id:`${i}-transpose`,weights:{offsets:B.offsets,neighbors:B.neighbors,weights:B.weights},output:q}).getCommandNodes(e)),R.push(F(`gram-moments-tiles`,[...le,...I(`transposed`,q),N(`gramPartial`,J,`read_write`)],_,M,`let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squareSum = 0.0;
  var columnSquareSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      squareSum += weight * weight;
    }
    var columnSquares = 0.0;
    for (var slot = transposedOffsets[transposedOffsetsOffset + row]; slot < transposedOffsets[transposedOffsetsOffset + row + 1u]; slot++) {
      if (transposedNeighbors[transposedNeighborsOffset + slot] >= ROWS) {
        continue;
      }
      let weight = transposedWeights[transposedWeightsOffset + slot];
      columnSquares += weight * weight;
    }
    columnSquareSum += columnSquares * columnSquares;
  }
  gramPartial[gramPartialOffset + index * 2u] = squareSum;
  gramPartial[gramPartialOffset + index * 2u + 1u] = columnSquareSum;`,void 0,S)));let Y=k.batchCount*k.blockStride,X=j(`dots`,O*Y),Z=(t,n)=>k.getBlock(o(e,X,t*Y,Y),n),ue=w?j(`partial-power`,h*_):void 0,Q=T?j(`partial-gram`,h*_):void 0,de=[j(`scratch-a`,h*u),j(`scratch-b`,h*u)];R.push(k.node);let fe=f(w),$=(e,t,n,r)=>F(e,[N(`partial`,n,`read`),N(`dots`,Z(r,t),`read_write`),P(`batchInformation`,k.getBatchInformation(t),`read`)],h,M,`if (index >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  ${L(`partial`,`index * TILE_COUNT`,`1u`,`TILE_COUNT`)}
  dots[dotsOffset + index] = sum;`,void 0,S);for(let e=0;e<k.batchCount;e++){let t=P(`batchInformation`,k.getBatchInformation(e),`read`);for(let n=1;n<=E;n++){let r=n===1,i=T&&n===2,a=w&&n<=C,o=a||i,s=de[(n-1)%2],c=de[(n-2+2)%2],l=!w&&n<=C,u=`${r?`first`:`next`}${i?`-gram`:``}${o?`-dot`:``}${l?`-diagonal`:``}`,d=o=>[...W(o),...r?[]:[N(`vectorIn`,c,`read`)],N(`vectorOut`,s,`read_write`),...a?[N(`partialPower`,ue,`read_write`)]:[],...i?[N(`partialGram`,Q,`read_write`)]:[],...l?[N(`dots`,Z(n-1,e),`read_write`)]:[],t],f=r?`getProbeValue(probeSeed, globalProbe, neighbor)`:`vectorIn[vectorInOffset + base + neighbor]`;R.push(...z.map((t,r)=>F(`step-${u}-${e}-${n}${t.idSuffix}`,d(r),h*t.chunkTileCount,`${M}${t.tileConstantsWGSL?`\n${t.tileConstantsWGSL}`:``}
${fe}`,b(t,`let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let tile = index % TILE_COUNT;
  let globalProbe = batchInformation[batchInformationOffset] + probe;
  let probeSeed = batchInformation[batchInformationOffset + 2u];
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var powerDot = 0.0;
  var gramDot = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += weights[weightsOffset + slot] * ${f};
    }
    vectorOut[vectorOutOffset + base + row] = sum;
${l?`    if (row == globalProbe) {
      dots[dotsOffset + probe] = sum;
    }
`:``}${a?`    powerDot += getProbeValue(probeSeed, globalProbe, row) * sum;
`:``}${i?`    gramDot += vectorIn[vectorInOffset + base + row] * sum;
`:``}  }
${a?`  partialPower[partialPowerOffset + probe * TILE_COUNT + tile] = powerDot;
`:``}${i?`  partialGram[partialGramOffset + probe * TILE_COUNT + tile] = gramDot;
`:``}  _ = powerDot;
  _ = gramDot;`),void 0,S))),a&&R.push($(`probe-merge-${e}-${n}`,e,ue,n-1)),i&&R.push($(`probe-merge-pair-${e}`,e,Q,se)),i&&q&&(R.push(F(`gram-square-${e}`,[...I(`transposed`,q),N(`vectorIn`,c,`read`),N(`partialGram`,Q,`read_write`),t],h*_,M,`let probe = index / TILE_COUNT;
  if (probe >= batchInformation[batchInformationOffset + 1u]) {
    return;
  }
  let tile = index % TILE_COUNT;
  let base = probe * ROWS;
  let firstRow = tile * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var squareSum = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    for (var slot = transposedOffsets[transposedOffsetsOffset + row]; slot < transposedOffsets[transposedOffsetsOffset + row + 1u]; slot++) {
      let neighbor = transposedNeighbors[transposedNeighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      sum += transposedWeights[transposedWeightsOffset + slot] * vectorIn[vectorInOffset + base + neighbor];
    }
    squareSum += sum * sum;
  }
  partialGram[partialGramOffset + probe * TILE_COUNT + tile] = squareSum;`,void 0,S)),R.push($(`probe-merge-square-${e}`,e,Q,ce)))}}let pe=[N(`dots`,X,`read`),N(`infinityNorm`,V,`read`),...J?[N(`gramPartial`,J,`read`)]:[],N(`powerTraces`,y.powerTraces,`read_write`),...y.powerTraceStandardErrors?[N(`powerTraceStandardErrors`,y.powerTraceStandardErrors,`read_write`)]:[],...y.gramTraces?[N(`gramTraces`,y.gramTraces,`read_write`)]:[]],me=w?`1.0 / f32(PROBE_TOTAL)`:`1.0`;return R.push(F(`final-merge`,pe,O+(T?2:0),`${M}${y.powerTraceStandardErrors?`\n${te}`:``}
const POWER_COUNT: u32 = ${C}u;
const DOT_ROWS: u32 = ${O}u;
const POWER_ROWS: u32 = ${D}u;`,`if (index < DOT_ROWS) {
    ${k.getSumWGSL(`dots`,`sum`,`index * ${Y}u`)}
    let value = sum * ${me};
    if (index < POWER_COUNT) {
      powerTraces[powerTracesOffset + index] = value;${y.powerTraceStandardErrors?w?`
      // Two-pass sample variance of the per-probe dots; a single probe has no estimate (NaN).
      ${k.getCenteredSquareSumWGSL(`dots`,`sum / f32(PROBE_TOTAL)`,`squareSum`,`index * ${Y}u`)}
      let variance = squareSum / max(f32(PROBE_TOTAL) - 1.0, 1.0);
      powerTraceStandardErrors[powerTraceStandardErrorsOffset + index] = select(
        sqrt(variance / f32(PROBE_TOTAL)),
        getNaN(),
        PROBE_TOTAL < 2u
      );`:`
      powerTraceStandardErrors[powerTraceStandardErrorsOffset + index] = 0.0;`:``}
    }${T?`
    if (index >= POWER_ROWS) {
      // The chains hold W / s: tr(W'W W) = s^3 E[v1' v2] and tr((W'W)^2) = s^2 E[|W' v1|^2].
      let norm = infinityNorm[infinityNormOffset];
      gramTraces[gramTracesOffset + 1u + index - POWER_ROWS] =
        value * select(norm * norm * norm, norm * norm, index > POWER_ROWS);
    }`:``}
  }${T?`
  if (index >= DOT_ROWS) {
    let slot = index - DOT_ROWS;
    ${L(`gramPartial`,`slot`,`2u`,`TILE_COUNT`)}
    gramTraces[gramTracesOffset + select(0u, 3u, slot == 1u)] = sum;
  }`:``}`)),y.status?.approximation&&R.push(F(`approximation`,[P(`approximation`,y.status.approximation,`read_write`)],1,``,`approximation[approximationOffset] = ${w?`1u`:`0u`};`)),[...R,...n(e,i,c.weights,c.output)]}};export{E as n,D as t};