import{n as e,t}from"./gpu-geographically-weighted-regression-DmHsCpNm.js";import{r as n}from"./kernel-C3tfQOn4.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{t as i}from"./gpu-reduction-tNgOVQ_C.js";import{c as a,o}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as s,f as c,p as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{A as f,C as p,D as m,E as ee,O as h,_ as g,a as te,c as ne,g as _,h as v,j as y,k as re,l as ie,n as ae,o as oe,u as se,w as b}from"./local-weighted-least-squares-wgsl--tcHa6bd.js";import{c as ce,d as x}from"./permutation-inference-kernels-COguC6vP.js";import{t as le}from"./permutation-random-SiwCiDNI.js";import{t as ue}from"./permutation-shuffle-zu7WXW69.js";import{t as de}from"./philox-random-DfVxWNPn.js";import{n as fe}from"./spatial-regression-solve-Vf-sQz9O.js";import{s as S,v as C}from"./spatial-context-DwZEd4EZ.js";import{c as w,r as T,u as pe}from"./wgsl-common-DfyXiYB0.js";import{t as E}from"./define-run-B-brkCe4.js";import{a as D}from"./pointer-overlay-Qh8qflcA.js";import{n as O,o as k}from"./text-DYSgVV-n.js";import{Un as A}from"./entry-preview-D70TdSbK.js";import{_ as j,a as M,g as N,l as P,m as F,u as I}from"./regression-CXg5hXjb.js";import{a as L}from"./weights-BRH93xop.js";var R={permutationCount:0,locationCount:1,observedFitCount:2,failedFitCount:3,length:4},z=Object.freeze({...S,measure:`distance`}),B=`GPUGeographicallyWeightedRegressionNonstationarityTest`,V=256,H=g,U=16777216,W=4096,G=8,K=32,q=8192,J=1048576,Y=64,X=class{id;props;rowCount;coefficientCount;maximumBandwidthCount;maximumNeighborCount;tileRowCount;tileCount;indexGridSize;batchRunCount;gaussianCutoff;estimatedCommandNodeCount;constructor(e){let t=e.id??`gwr-nonstationarity-test`;this.id=t,this.props=e;let{predictorCount:n,maximumPermutations:r}=e;h(t,e.output);let i=ee(e.output);if(C(t,e.spatialContext,z),c(t,`predictorCount`,n,1,7),c(t,`maximumPermutations`,r,1,J),this.coefficientCount=n+1,this.maximumBandwidthCount=e.maximumBandwidthCount??32,c(t,`maximumBandwidthCount`,this.maximumBandwidthCount,1,32),this.maximumNeighborCount=e.maximumNeighborCount??128,c(t,`maximumNeighborCount`,this.maximumNeighborCount,1,128),s(e.positions,[`float32x2`],`${t} positions`),this.rowCount=e.positions.length,c(t,`positions.length`,this.rowCount,2,H),e.gaussianMode!==void 0&&e.gaussianMode!==`exact`&&e.gaussianMode!==`truncated`)throw Error(`${t} gaussianMode must be exact or truncated`);if(e.gaussianCutoff!==void 0&&y(t,`gaussianCutoff`,e.gaussianCutoff),e.gaussianMode===`exact`&&(c(t,`positions.length for gaussianMode exact`,this.rowCount,2,_),e.gaussianCutoff!==void 0))throw Error(`${t} gaussianMode exact requires gaussianCutoff to be omitted`);if(this.gaussianCutoff=e.gaussianCutoff??(e.gaussianMode===`truncated`||this.rowCount>1048576?8:null),e.indexGridSize===!1)this.indexGridSize=null;else if(e.indexGridSize){let[n,r]=e.indexGridSize;if(!Number.isInteger(n)||!Number.isInteger(r)||n<1||r<1||n*r>=4294967295)throw Error(`${t} indexGridSize must be two positive integers with columns * rows < 2^32 - 1`);this.indexGridSize=[n,r]}else if(this.rowCount>=1024){let e=Math.min(v,Math.max(1,Math.ceil(Math.sqrt(this.rowCount/8))));this.indexGridSize=[e,e]}else this.indexGridSize=null;if(this.indexGridSize||c(t,`positions.length without indexGridSize`,this.rowCount,2,_),s(e.predictors,[`float32`],`${t} predictors`),e.predictors.length!==this.rowCount*n)throw Error(`${t} predictors length must equal rows * predictorCount`);if(s(e.response,[`float32`],`${t} response`),e.response.length!==this.rowCount)throw Error(`${t} response length must equal the row count`);if(e.mask&&(s(e.mask,[`uint32`],`${t} mask`),e.mask.length!==this.rowCount))throw Error(`${t} mask length must equal the row count`);s(e.bandwidthParameters,[`float32`],`${t} bandwidthParameters`);let a=p(this.maximumBandwidthCount);if(e.bandwidthParameters.length<a)throw Error(`${t} bandwidthParameters must hold ${a} float32 values`);if(s(e.selectedBandwidth,[`float32`],`${t} selectedBandwidth`),e.selectedBandwidth.length<2)throw Error(`${t} selectedBandwidth must hold 2 float32 values`);if(s(e.coefficients,[`float32`],`${t} coefficients`),e.coefficients.length<this.rowCount*this.coefficientCount)throw Error(`${t} coefficients must hold rows * (predictorCount + 1) values`);if(s(e.parameters,[`uint32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must hold 4 uint32`);let o=this.coefficientCount*6;if(s(i.table,[`float32`],`${t} output.table`),i.table.length<o)throw Error(`${t} output.table must hold ${o} float32 values`);if(s(i.summary,[`float32`],`${t} output.summary`),i.summary.length<R.length)throw Error(`${t} output.summary must hold ${R.length} float32 values`);if(i.standardDeviations){s(i.standardDeviations,[`float32`],`${t} output.standardDeviations`);let e=(r+1)*this.coefficientCount;if(i.standardDeviations.length<e)throw Error(`${t} output.standardDeviations must hold ${e} float32 values`)}if(i.approximation){let n=e.output.status?.approximation?`output.status.approximation`:`output.approximation`;s(i.approximation,[`uint32`],`${t} ${n}`),c(t,`${n}.length`,i.approximation.length,1)}let u=2*this.coefficientCount+1;this.batchRunCount=Math.min(r+1,Math.max(K,Math.ceil((r+1)/q)));let m=this.batchRunCount;if(e.tileRowCount!==void 0)c(t,`tileRowCount`,e.tileRowCount),this.tileRowCount=e.tileRowCount;else{let e=this.rowCount<=1048576?r+1:m,t=Math.max(1,Math.min(W,Math.floor(U/(e*u))));this.tileRowCount=Math.max(G,Math.ceil(this.rowCount/t))}this.tileCount=Math.ceil(this.rowCount/this.tileRowCount),c(t,`tileCount * batchRunCount * partialStride for tileRowCount`,this.tileCount*m*u,1,U),this.estimatedCommandNodeCount=2080+3*Math.ceil((r+1)/this.batchRunCount),l(t,this.estimatedCommandNodeCount),f(t,[i.table,i.summary,i.standardDeviations,i.approximation],[{writes:[i.table,i.summary],reads:[i.standardDeviations]}]),d(t,[i.table,i.summary,i.standardDeviations,i.approximation],[e.positions,e.predictors,e.response,e.mask,e.bandwidthParameters,e.selectedBandwidth,e.coefficients,e.parameters])}getCommandNodes(t){let{id:a,props:s,rowCount:c,coefficientCount:l,maximumBandwidthCount:d,maximumNeighborCount:f,tileRowCount:p,tileCount:h,indexGridSize:g,batchRunCount:_,gaussianCutoff:v}=this,{predictorCount:y,maximumPermutations:b,mask:x}=s,S=ee(s.output),C=Math.ceil(h/Y);u(a,t,[s.positions,s.predictors,s.response,x,s.bandwidthParameters,s.selectedBandwidth,s.coefficients,s.parameters,S.table,S.summary,S.standardDeviations,S.approximation]),re(t.device,a,[s.positions,s.predictors,s.response,x,s.bandwidthParameters,s.selectedBandwidth,s.coefficients,s.parameters,...Object.values(S)],[[`row-valid`,c*4],[`approximation`,4],[`coefficient anchors`,l*4],[`anchor tiles and reduction scratch`,Math.ceil(c/V)*4],[`compacted training rows`,(c*(y+4)+1)*4],[`tile counts, offsets and scan scratch`,(Math.ceil(c/V)+1)*4],[`coefficient and permutation partials`,(_*h*(2*l+1)+1)*4],[`fit-counts`,(b+1)*4],[`standard-deviations`,(b+1)*l*4],...m(c,g)]);let w=y+4,T=2*l+1,E=Math.ceil(c/V),D=(e,r,i)=>n(t,`${a}-${e}`,r,Math.max(i,1)),O=D(`row-valid`,`uint32`,c),k=D(`tile-counts`,`uint32`,E+1),A=D(`tile-offsets`,`uint32`,E+1),j=D(`rows`,`float32`,c*w+1),M=g?n(t,`${a}-compact-positions`,`float32x2`,Math.max(c,1)):null,N=g&&M?oe(t,{id:a,operation:B,positions:M,rowCount:c,gridSize:g}):null,P=D(`shifts`,`float32`,l),F=D(`anchor-tiles`,`uint32`,E),I=D(`anchor-row`,`uint32`,1),L=D(`partial`,`float32`,_*h*T+1),z=D(`fit-counts`,`float32`,b+1),H=S.standardDeviations??D(`standard-deviations`,`float32`,(b+1)*l),U=(e,t,n)=>({name:e,view:t,type:n,access:`read`}),W=(e,t,n)=>({name:e,view:t,type:n,access:`read_write`}),G=N?[U(`cellOffsets`,N.cellOffsets,`u32`),U(`sortedIds`,N.sortedIds,`u32`)]:[],K=`const ROW_COUNT: u32 = ${c}u;
const BATCH_RUNS: u32 = ${_}u;
const BATCH_CURSOR: u32 = ${_*h*T}u;
const PREDICTOR_COUNT: u32 = ${y}u;
const P: u32 = ${l}u;
const ROW_STRIDE: u32 = ${w}u;
const PARTIAL_STRIDE: u32 = ${T}u;
const TILE_ROWS: u32 = ${p}u;
const TILE_COUNT: u32 = ${h}u;
const SCAN_TILE_ROWS: u32 = ${V}u;
const SCAN_TILE_COUNT: u32 = ${E}u;
const LADDER: u32 = ${d}u;
const MAXIMUM_NEIGHBORS: u32 = ${f}u;
const HEADER: u32 = 4u;
const SENTINEL: f32 = 3.0e38;
const ADAPTIVE_FACTOR: f32 = ${se};
${me}`,q=`${K}
${ce(b)}`,J=g?ne(g):``,X=c*w,Z=[o(t,{id:`${a}-validate`,operation:B,variant:`validate`,bindings:[U(`positions`,s.positions,`f32`),U(`predictors`,s.predictors,`f32`),U(`response`,s.response,`f32`),...x?[U(`rowMask`,x,`u32`)]:[],W(`rowValid`,O,`u32`)],invocationCount:c,declarations:K,body:`var isValid = true;
  ${x?`isValid = rowMask[rowMaskOffset + index] != 0u;`:``}
  if (!isFiniteBits(positions[positionsOffset + 2u * index]) ||
    !isFiniteBits(positions[positionsOffset + 2u * index + 1u]) ||
    !isFiniteBits(response[responseOffset + index])) {
    isValid = false;
  }
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + index * PREDICTOR_COUNT + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`}),o(t,{id:`${a}-tile-counts`,operation:B,variant:`tile-counts`,bindings:[U(`rowValid`,O,`u32`),W(`tileCounts`,k,`u32`)],invocationCount:E+1,declarations:K,body:`let firstRow = index * SCAN_TILE_ROWS;
  let endRow = min(firstRow + SCAN_TILE_ROWS, ROW_COUNT);
  var count = 0u;
  for (var row = firstRow; row < endRow; row++) {
    count += select(0u, 1u, rowValid[rowValidOffset + row] != 0u);
  }
  tileCounts[tileCountsOffset + index] = count;`}),...new r({id:`${a}-tile-offsets`,input:k,output:A,mode:`exclusive`}).getCommandNodes(t),o(t,{id:`${a}-compact`,operation:B,variant:`compact`,bindings:[U(`positions`,s.positions,`f32`),U(`predictors`,s.predictors,`f32`),U(`response`,s.response,`f32`),U(`rowValid`,O,`u32`),U(`tileOffsets`,A,`u32`),W(`rows`,j,`f32`)],invocationCount:c,declarations:K,body:`if (index == 0u) {
    rows[rowsOffset + ${X}u] = f32(tileOffsets[tileOffsetsOffset + SCAN_TILE_COUNT]);
  }
  if (rowValid[rowValidOffset + index] == 0u) {
    return;
  }
  let tile = index / SCAN_TILE_ROWS;
  var rank = tileOffsets[tileOffsetsOffset + tile];
  for (var row = tile * SCAN_TILE_ROWS; row < index; row++) {
    rank += select(0u, 1u, rowValid[rowValidOffset + row] != 0u);
  }
  let base = rowsOffset + rank * ROW_STRIDE;
  rows[base] = positions[positionsOffset + 2u * index];
  rows[base + 1u] = positions[positionsOffset + 2u * index + 1u];
  rows[base + 2u] = 0.0;
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    rows[base + 3u + column] = predictors[predictorsOffset + index * PREDICTOR_COUNT + column];
  }
  rows[base + 3u + PREDICTOR_COUNT] = response[responseOffset + index];`}),...N&&M?[o(t,{id:`${a}-compact-positions`,operation:B,variant:`compact-positions`,bindings:[U(`rows`,j,`f32`),W(`compactPositions`,M,`f32`)],invocationCount:c,declarations:K,body:`let count = u32(rows[rowsOffset + ${X}u]);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  let included = index < count;
  compactPositions[compactPositionsOffset + 2u * index] =
    select(invalid, rows[rowsOffset + index * ROW_STRIDE], included);
  compactPositions[compactPositionsOffset + 2u * index + 1u] =
    select(invalid, rows[rowsOffset + index * ROW_STRIDE + 1u], included);`}),...N.nodes]:[],o(t,{id:`${a}-bandwidths`,operation:B,variant:`bandwidths`,bindings:[W(`rows`,j,`f32`),U(`params`,s.bandwidthParameters,`f32`),U(`selection`,s.selectedBandwidth,`f32`),...G],invocationCount:c,declarations:`${K}
${J}
${_e()}
${g?ie():``}`,body:`let count = u32(rows[rowsOffset + ${X}u]);
  if (index >= count) {
    return;
  }
  // Slot 1 holds the selected ladder value, also for refined candidates outside the caller ladder.
  let value = selection[selectionOffset + 1u];
  var bandwidth = -1.0;
  if (params[paramsOffset + 1u] < 0.5) {
    bandwidth = select(-1.0, value, value > 0.0);
  } else if (value >= 2.0 && value <= f32(MAXIMUM_NEIGHBORS)) {
    let neighborCount = u32(floor(value + 0.5));
    var list: array<f32, MAXIMUM_NEIGHBORS>;
    ${g?`selectNeighborDistances(index, neighborCount, &list);`:`for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
      list[slot] = SENTINEL;
    }
    let origin = getPosition(index);
    for (var row = 0u; row < count; row++) {
      let delta = getPosition(row) - origin;
      let squared = dot(delta, delta);
      if (squared < list[neighborCount - 1u]) {
        var slot = neighborCount - 1u;
        loop {
          if (slot == 0u || list[slot - 1u] <= squared) {
            break;
          }
          list[slot] = list[slot - 1u];
          slot = slot - 1u;
        }
        list[slot] = squared;
      }
    }`}
    let squared = list[neighborCount - 1u];
    if (squared < SENTINEL) {
      bandwidth = sqrt(squared) * ADAPTIVE_FACTOR;
    }
  }
  rows[rowsOffset + index * ROW_STRIDE + 2u] = bandwidth;`}),o(t,{id:`${a}-anchor-tiles`,operation:B,variant:`anchor-tiles`,bindings:[U(`coefficients`,s.coefficients,`f32`),W(`anchorTiles`,F,`u32`)],invocationCount:E,declarations:`${K}\n${he}`,body:`let firstRow = index * SCAN_TILE_ROWS;
  let endRow = min(firstRow + SCAN_TILE_ROWS, ROW_COUNT);
  var firstIncludedRow = 0xffffffffu;
  for (var row = firstRow; row < endRow; row++) {
    if (isFiniteRow(row)) { firstIncludedRow = row; break; }
  }
  anchorTiles[anchorTilesOffset + index] = firstIncludedRow;`}),...new i({id:`${a}-anchor-minimum`,input:F,output:I,operation:`min`}).getCommandNodes(t),o(t,{id:`${a}-anchor`,operation:B,variant:`anchor`,bindings:[U(`coefficients`,s.coefficients,`f32`),U(`anchorRow`,I,`u32`),W(`shifts`,P,`f32`)],invocationCount:1,declarations:K,body:`let row = anchorRow[anchorRowOffset];
  for (var column = 0u; column < P; column++) {
    var value = 0.0;
    if (row < ROW_COUNT) { value = coefficients[coefficientsOffset + row * P + column]; }
    shifts[shiftsOffset + column] = value;
  }`}),o(t,{id:`${a}-observed-tiles`,operation:B,variant:`observed-tiles`,bindings:[U(`coefficients`,s.coefficients,`f32`),U(`shifts`,P,`f32`),W(`partial`,L,`f32`)],invocationCount:h,declarations:`${K}
${he}`,body:`var sums: array<f32, ${T}>;
  let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  for (var row = firstRow; row < endRow; row++) {
    if (!isFiniteRow(row)) {
      continue;
    }
    sums[0] += 1.0;
    for (var column = 0u; column < P; column++) {
      let deviation = coefficients[coefficientsOffset + row * P + column] - shifts[shiftsOffset + column];
      sums[1u + column] += deviation;
      sums[1u + P + column] += deviation * deviation;
    }
  }
  let base = partialOffset + index * PARTIAL_STRIDE;
  for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
    partial[base + slot] = sums[slot];
  }`})];Z.push(o(t,{id:`${a}-batch-reset`,operation:B,variant:`batch-reset`,bindings:[W(`partial`,L,`f32`)],invocationCount:1,declarations:K,body:`partial[partialOffset + BATCH_CURSOR] = 0.0;`}));let ve=Math.ceil((b+1)/_);for(let n=0;n<ve;n++){let r=Math.min(_,b+1-n*_);Z.push(e(t,{id:`${a}-batch-${n}-permuted-tiles`,operation:B,variant:`permuted-tiles`,bindings:[U(`rows`,j,`f32`),U(`params`,s.bandwidthParameters,`f32`),U(`parameters`,s.parameters,`u32`),U(`shifts`,P,`f32`),W(`partial`,L,`f32`),...G],workgroupSize:Y,guardIndex:!1,invocationCount:C*Y*r,declarations:`${q}
${J}
const TILE_GROUP_COUNT: u32 = ${C}u;
const PERMUTED_WORKGROUP_SIZE: u32 = ${Y}u;
${le}
${ue(de.geographicallyWeightedRegressionNonstationarityShuffle.id)}
${J?``:pe}
${fe(l)}
${ae(l)}
${te(v)}
${ge(l,!!g,v)}`,body:`let permutationWorkgroup = index / PERMUTED_WORKGROUP_SIZE;
  let localRun = permutationWorkgroup / TILE_GROUP_COUNT;
  let permutation = u32(partial[partialOffset + BATCH_CURSOR]) + localRun;
  let tile = (permutationWorkgroup % TILE_GROUP_COUNT) * PERMUTED_WORKGROUP_SIZE + localInvocationIndex;
  let count = u32(rows[rowsOffset + ${X}u]);
  let runsPermutation = localRun < BATCH_RUNS && permutation > 0u && permutation <= readPermutationCount();
  if (runsPermutation) {
    loadSwapOrNotRounds(readSeedKey(), permutation, count, localInvocationIndex, PERMUTED_WORKGROUP_SIZE);
  }
  workgroupBarrier();
  var sums: array<f32, ${T}>;
  if (runsPermutation && tile < TILE_COUNT) {
    let firstRow = tile * TILE_ROWS;
    let endRow = min(firstRow + TILE_ROWS, count);
    for (var location = firstRow; location < endRow; location++) {
      let bandwidth = rows[rowsOffset + location * ROW_STRIDE + 2u];
      if (!(bandwidth > 0.0)) {
        continue;
      }
      var estimate: array<f32, ${l}>;
      if (!fitPermuted(location, bandwidth, count, &estimate)) {
        continue;
      }
      sums[0] += 1.0;
      for (var column = 0u; column < P; column++) {
        let deviation = estimate[column] - shifts[shiftsOffset + column];
        sums[1u + column] += deviation;
        sums[1u + P + column] += deviation * deviation;
      }
    }
    let base = partialOffset + (localRun * TILE_COUNT + tile) * PARTIAL_STRIDE;
    for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
      partial[base + slot] = sums[slot];
    }
  }`}),e(t,{id:`${a}-batch-${n}-merge`,operation:B,variant:`merge`,bindings:[U(`partial`,L,`f32`),U(`parameters`,s.parameters,`u32`),W(`fitCounts`,z,`f32`),W(`standardDeviations`,H,`f32`)],invocationCount:r,workgroupSize:32,declarations:q,body:`let run = u32(partial[partialOffset + BATCH_CURSOR]) + index;
  if (run > ${b}u) { return; }
  let nan = getNaN();
  var sums: array<f32, ${T}>;
  if (run <= readPermutationCount()) {
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      let base = partialOffset + (index * TILE_COUNT + tile) * PARTIAL_STRIDE;
      for (var slot = 0u; slot < PARTIAL_STRIDE; slot++) {
        sums[slot] += partial[base + slot];
      }
    }
  } else {
    sums[0] = -1.0;
  }
  fitCounts[fitCountsOffset + run] = select(nan, sums[0], sums[0] >= 0.0);
  for (var column = 0u; column < P; column++) {
    var spread = nan;
    if (sums[0] > 0.0) {
      let mean = sums[1u + column] / sums[0];
      spread = sqrt(max(sums[1u + P + column] / sums[0] - mean * mean, 0.0));
    }
    standardDeviations[standardDeviationsOffset + run * P + column] = spread;
  }`})),Z.push(o(t,{id:`${a}-batch-${n}-advance`,operation:B,variant:`batch-advance`,bindings:[W(`partial`,L,`f32`)],invocationCount:1,declarations:K,body:`partial[partialOffset + BATCH_CURSOR] += f32(BATCH_RUNS);`}))}return Z.push(o(t,{id:`${a}-finish`,operation:B,variant:`finish`,bindings:[U(`standardDeviations`,H,`f32`),U(`fitCounts`,z,`f32`),U(`parameters`,s.parameters,`u32`),U(`rows`,j,`f32`),W(`table`,S.table,`f32`),W(`summary`,S.summary,`f32`)],invocationCount:l,declarations:q,body:`let nan = getNaN();
  let permutationCount = readPermutationCount();
  let observed = standardDeviations[standardDeviationsOffset + index];
  var greater = 0u;
  var validCount = 0.0;
  var total = 0.0;
  for (var run = 1u; run <= permutationCount; run++) {
    let value = standardDeviations[standardDeviationsOffset + run * P + index];
    if (isFiniteBits(value)) {
      validCount += 1.0;
      total += value;
      if (isFiniteBits(observed) && value >= observed) {
        greater += 1u;
      }
    }
  }
  let mean = select(nan, total / validCount, validCount > 0.0);
  var squares = 0.0;
  for (var run = 1u; run <= permutationCount; run++) {
    let value = standardDeviations[standardDeviationsOffset + run * P + index];
    if (isFiniteBits(value)) {
      squares += (value - mean) * (value - mean);
    }
  }
  let simulatedDeviation = select(nan, sqrt(squares / validCount), validCount > 0.0);
  let base = tableOffset + index * 6u;
  table[base] = observed;
  table[base + 1u] = select(nan, f32(greater + 1u) / f32(permutationCount + 1u), isFiniteBits(observed));
  table[base + 2u] = select(nan, f32(greater), isFiniteBits(observed));
  table[base + 3u] = mean;
  table[base + 4u] = simulatedDeviation;
  table[base + 5u] = select(nan, (observed - mean) / simulatedDeviation, simulatedDeviation > 0.0);
  if (index == 0u) {
    let count = rows[rowsOffset + ${X}u];
    var failed = 0.0;
    for (var run = 1u; run <= permutationCount; run++) {
      failed += count - fitCounts[fitCountsOffset + run];
    }
    summary[summaryOffset + ${R.permutationCount}u] = f32(permutationCount);
    summary[summaryOffset + ${R.locationCount}u] = count;
    summary[summaryOffset + ${R.observedFitCount}u] = fitCounts[fitCountsOffset];
    summary[summaryOffset + ${R.failedFitCount}u] = failed;
  }`})),S.approximation&&Z.push(o(t,{id:`${a}-approximation`,operation:B,variant:`approximation`,invocationCount:1,bindings:[U(`params`,s.bandwidthParameters,`f32`),W(`approximation`,S.approximation,`u32`)],body:`approximation[approximationOffset] = ${v===null?`0u`:`select(1u, 0u, params[paramsOffset] > 0.5)`};`})),Z}},me=`
${T}

${w}
`,he=`
fn isFiniteRow(row: u32) -> bool {
  for (var column = 0u; column < P; column++) {
    if (!isFiniteBits(coefficients[coefficientsOffset + row * P + column])) {
      return false;
    }
  }
  return true;
}
`;function ge(e,t,n){let r=e;return`
fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  return getLocalKernelWeight(distance, bandwidth, params[paramsOffset] > 0.5);
}

fn fitPermuted(
  location: u32,
  bandwidth: f32,
  count: u32,
  estimate: ptr<function, array<f32, ${r}>>
) -> bool {
  var a: array<f32, ${r*r}>;
  var b: array<f32, ${r}>;
  var z: array<f32, ${r}>;
  var scale: array<f32, ${r}>;
  let origin = vec2f(rows[rowsOffset + location * ROW_STRIDE], rows[rowsOffset + location * ROW_STRIDE + 1u]);
  let focalBase = rowsOffset + getSwapOrNotPermutationIndex(location, count) * ROW_STRIDE;
  z[0] = 1.0;
  ${t?`// Bisquare weights vanish beyond the bandwidth, so only the covering grid cells are visited.
  let useGrid = ${n===null?`params[paramsOffset] > 0.5`:`true`} && bandwidth > 0.0 && isFiniteBits(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, ${n===null?`bandwidth`:`select(bandwidth * ${a(n)}, bandwidth, params[paramsOffset] > 0.5)`}), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = count;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      let rowBase = rowsOffset + row * ROW_STRIDE;
      let delta = vec2f(rows[rowBase], rows[rowBase + 1u]) - origin;
      let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
      if (!(weight > 0.0)) {
        continue;
      }
      let dataBase = rowsOffset + getSwapOrNotPermutationIndex(row, count) * ROW_STRIDE;
      for (var column = 0u; column < PREDICTOR_COUNT; column++) {
        z[column + 1u] = rows[dataBase + 3u + column] - rows[focalBase + 3u + column];
      }
      let y = rows[dataBase + 3u + PREDICTOR_COUNT];
      for (var r = 0u; r < P; r++) {
        let weighted = weight * z[r];
        b[r] = b[r] + weighted * y;
        for (var c = 0u; c <= r; c++) {
          a[r * P + c] = a[r * P + c] + weighted * z[c];
        }
      }
    }
  }`:`for (var row = 0u; row < count; row++) {
    let rowBase = rowsOffset + row * ROW_STRIDE;
    let delta = vec2f(rows[rowBase], rows[rowBase + 1u]) - origin;
    let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let dataBase = rowsOffset + getSwapOrNotPermutationIndex(row, count) * ROW_STRIDE;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = rows[dataBase + 3u + column] - rows[focalBase + 3u + column];
    }
    let y = rows[dataBase + 3u + PREDICTOR_COUNT];
    for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      b[r] = b[r] + weighted * y;
      for (var c = 0u; c <= r; c++) {
        a[r * P + c] = a[r * P + c] + weighted * z[c];
      }
    }
  }`}
  if (!factorEquilibrated_${r}(&a, &scale)) {
    return false;
  }
  solveEquilibrated_${r}(&a, &scale, &b);
  var intercept = b[0];
  for (var column = 1u; column < P; column++) {
    let slope = b[column];
    (*estimate)[column] = slope;
    intercept = intercept - slope * rows[focalBase + 2u + column];
  }
  (*estimate)[0] = intercept;
  var finite = true;
  for (var column = 0u; column < P; column++) {
    finite = finite && isFiniteBits((*estimate)[column]);
  }
  return finite;
}
`}function _e(){return`
fn getPosition(row: u32) -> vec2f {
  return vec2f(rows[rowsOffset + row * ROW_STRIDE], rows[rowsOffset + row * ROW_STRIDE + 1u]);
}
`}var Z=4,ve=32,ye=[`int`,`x1`,`x2`],Q=[M+.08,.42,.97,.9],$=[M+.08,.08,.97,.34],be=E({parameters:A,caption:`Local slope (left); observed variation (bars), shuffled variation (dots), p-values (below).`,createInputs:({random:e,world:t})=>P(e,t,8,M),build(e,n){let r=n.tracts.count,i=e.upload(`positions`,`float32x2`,n.tracts.centers),a=e.upload(`predictors`,`float32`,n.predictors),o=e.allocate(`response`,`float32`,r),s=e.allocate(`bandwidth-parameters`,`float32`,p(Z)),c=e.allocate(`permutation-parameters`,`uint32`,4),l=e.allocate(`coefficients`,`float32`,r*3),u=e.allocate(`selected-bandwidth`,`float32`,2);e.graph.add(new t({positions:i.view,predictors:a.view,predictorCount:2,response:o.view,parameters:s.view,maximumBandwidthCount:Z,maximumNeighborCount:32,refinement:!1,output:{coefficients:l.view,selectedBandwidth:u.view}}));let d=e.allocate(`table`,`float32`,18);e.graph.add(new X({positions:i.view,predictors:a.view,predictorCount:2,response:o.view,bandwidthParameters:s.view,selectedBandwidth:u.view,coefficients:l.view,parameters:c.view,maximumPermutations:ve,maximumBandwidthCount:Z,maximumNeighborCount:32,output:{table:d.view,summary:e.allocate(`summary`,`float32`,R.length).view,standardDeviations:e.allocate(`standard-deviations`,`float32`,99).view}}));let f=e.upload(`marker-positions`,`float32`,Float32Array.from([.5/3,1.5/3,2.5/3]));return{outputs:{mesh:j(e,n),slope:L(e,`local-slope`,l,3,1,r),observed:L(e,`observed`,d,6,0,3),simulated:L(e,`simulated`,d,6,3,3),pseudoP:L(e,`pseudo-p`,d,6,1,3),markerPositions:f,significance:e.upload(`significance`,`float32`,new Float32Array(64).fill(.05))},update:e=>{o.write(I(n,{noise:.3,coefficients:N(n,e.drift)})),s.write(b({bandwidths:[e.bandwidth],kernel:`bisquare`},Z)),c.write(x({seed:7,permutationCount:e.permutationCount}))}}},interaction:D(A,{kind:`pad`,hint:`permutations and bandwidth`,x:`permutationCount`,y:`bandwidth`,decorate:(e,t,{outputs:n,peek:r})=>{let i=r(n.pseudoP);return{readout:`permutationCount ${e.permutationCount} · bandwidth ${e.bandwidth} · p(x1) ${i?i[1].toFixed(2):`-`}`}}}),draw:({inputs:e,outputs:t,world:n,theme:r})=>{let i=O({rectangle:k(n,Q),x:{categories:ye},y:{range:[0,1],ticks:[.5,1],title:`spatial SD (dot: shuffled)`}}),a=O({rectangle:k(n,$),x:{categories:ye},y:{range:[0,1],ticks:[1],title:`pseudo p`},references:[{value:.05,label:`0.05`}]});return[{layer:`text`,...i.grid},{layer:`text`,...a.grid},...F(e,t.mesh,t.slope,{ramp:`sequential`,range:[0,3.2]}),{layer:`chart`,kind:`bars`,values:t.observed,color:[.55,.65,.95,.9],range:[0,1],rectangle:Q},{layer:`chart`,kind:`scatter`,values:t.simulated,xValues:t.markerPositions,xRange:[0,1],range:[0,1],color:[...r.ink,.95],radius:5,rectangle:Q},{layer:`chart`,kind:`bars`,values:t.pseudoP,range:[0,1],color:[.55,.65,.95,.9],rectangle:$},{layer:`chart`,kind:`line`,values:t.significance,range:[0,1],color:[1,.55,.3,.9],radius:1,rectangle:$},{layer:`text`,...i},{layer:`text`,...a}]}});export{be as default};