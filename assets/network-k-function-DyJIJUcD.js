import{i as e,o as t,s as n,t as r}from"./network-view-utils-B1U8oNKx.js";import{t as i}from"./common-CshkX-d_.js";import{a,v as o}from"./spatial-context-DwZEd4EZ.js";import{r as s}from"./kernel-C3tfQOn4.js";import{t as c}from"./gpu-reduction-tNgOVQ_C.js";import{t as l}from"./gpu-scan-B4NNTfF9.js";import{o as u,t as d}from"./wgsl-kernel-nodes-DagJklH2.js";import{A as f,E as p,S as m,f as h,k as g,p as _,u as v,v as y,y as b}from"./gpu-contributor-utils-BIwLQON5.js";import{c as ee}from"./wgsl-common-DfyXiYB0.js";import{t as te}from"./permutation-random-SiwCiDNI.js";import{t as x}from"./gpu-network-snapping-DbjuKXIK.js";import{t as S}from"./network-accessibility-lanes-D6-5poNo.js";import{t as C}from"./network-accessibility-passes-BXCyymzh.js";import{t as w}from"./gpu-network-reachability-Coypydjh.js";import{t as T}from"./philox-random-DfVxWNPn.js";import{t as E}from"./define-run-B-brkCe4.js";import{r as D}from"./text-DYSgVV-n.js";import{p as O}from"./inputs-CJKKIRKK.js";import{i as k,p as A,t as j}from"./roads-DzovamHj.js";import{o as M}from"./display-Bu20g6o5.js";import{o as N,s as P}from"./pick-Bo3t3hjv.js";import{Sa as F}from"./entry-preview-D70TdSbK.js";function I(e,t=new Uint32Array(4)){if(t.length<4)throw Error(`GPUNetworkKFunction parameter target must hold 4 elements`);let{seed:n,activeSimulations:r}=e;if(!Number.isSafeInteger(n)||n<0)throw Error(`GPUNetworkKFunction seed must be a non-negative safe integer`);if(!Number.isSafeInteger(r)||r<0)throw Error(`GPUNetworkKFunction activeSimulations must be a non-negative integer`);return t[0]=n%2**32,t[1]=Math.floor(n/2**32),t[2]=r,t[3]=0,t}var L=Object.freeze({...a,measure:`distance`}),R=`GPUNetworkKFunction`,ne=T.networkKFunction.id,re=2**24;function ie(e){let t=Math.floor(Math.log2((2**32-1)/Math.max(e,1)));return 2**i(t,0,29)}function z(e){return Math.max(Math.ceil(e/128),1)}var B=1024;function ae(e){return v(e,4)}var oe=4294967295,V=64,H=32,U=32,se=16,W=64,G=class{id;props;eventCount;nodeCount;edgeCount;bandCount;simulationCount;rowCount;rowsPerBlock;blockCount;estimatedCommandNodeCount;spatialContext;constructor(e){this.id=e.id??`network-k-function`,this.props=e,n(this.id,e.output);let{id:t}=this;this.spatialContext=o(t,e.spatialContext,L);let{output:r}=e,{nodeCount:a,edgeCount:s}=f(t,e.topology,{weights:!0,nodePositions:!0});for(let[n,i]of[[`parameters`,e.parameters],[`output.pairCounts`,r.pairCounts],[`output.snappedEventCount`,r.snappedEventCount],[`output.status.overflow`,r.status?.overflow],[`output.status.converged`,r.status?.converged],[`output.status.iterationLimitReached`,r.status?.iterationLimitReached]])i&&p(i,[`uint32`],`${t} ${n}`);for(let[n,i]of[[`maximumSnapDistance`,e.maximumSnapDistance],[`maximumDistance`,e.maximumDistance],[`networkLength`,e.networkLength],[`output.kValues`,r.kValues],[`output.envelope`,r.envelope]])i&&p(i,[`float32`],`${t} ${n}`);if(p(e.positions,[`float32x2`],`${t} positions`),this.eventCount=e.positions.length,this.nodeCount=a,this.edgeCount=s,this.bandCount=e.bandCount??H,this.simulationCount=e.simulationCount??0,this.eventCount<2)throw Error(`${t} requires at least two events`);if(this.edgeCount<1)throw Error(`${t} requires at least one edge`);if(h(t,`bandCount`,this.bandCount,2,B),h(t,`simulationCount`,this.simulationCount,0),e.parameters.length<4)throw Error(`${t} parameters must hold 4 elements`);for(let[n,i]of[[`maximumSnapDistance`,e.maximumSnapDistance],[`maximumDistance`,e.maximumDistance],[`networkLength`,e.networkLength],[`output.snappedEventCount`,r.snappedEventCount],[`output.status.overflow`,r.status?.overflow],[`output.status.converged`,r.status?.converged],[`output.status.iterationLimitReached`,r.status?.iterationLimitReached]])if(i&&i.length!==1)throw Error(`${t} ${n} must contain exactly one row`);let c=1+this.simulationCount;for(let[e,n,i]of[[`output.kValues`,r.kValues,c*this.bandCount],[`output.pairCounts`,r.pairCounts,c*this.bandCount],[`output.envelope`,r.envelope,3*this.bandCount]])if(n&&n.length!==i)throw Error(`${t} ${e} length must be ${i}`);if(r.status?.overflow&&e.candidateCapacity===void 0)throw Error(`${t} output.status.overflow requires candidateCapacity`);if(e.candidateCapacity!==void 0&&!r.status?.overflow)throw Error(`${t} candidateCapacity requires output.status.overflow`);this.rowCount=c*this.eventCount,this.rowsPerBlock=e.rowsPerBlock??i(Math.min(U,S({rowCount:this.rowCount,nodeCount:this.nodeCount,edgeCount:this.edgeCount})),1,this.rowCount),h(t,`rowsPerBlock`,this.rowsPerBlock,1,this.rowCount),m(t,`rowsPerBlock x events`,this.rowsPerBlock*this.eventCount,4294967295),this.blockCount=Math.ceil(this.rowCount/this.rowsPerBlock);let l=Math.min(e.laneCount??this.rowsPerBlock,this.rowsPerBlock),u=this.blockCount*Math.ceil(this.rowsPerBlock/l),d=2+(e.maximumIterations??W)+Number(!!(r.status?.converged||r.status?.iterationLimitReached));this.estimatedCommandNodeCount=197+(this.simulationCount>0?5:0)+u*d+this.blockCount+Number(!!r.envelope)+2*Number(!!(r.status?.converged||r.status?.iterationLimitReached)),_(t,this.estimatedCommandNodeCount);let g=q(e);b(t,g,K(e));let v=g.filter(e=>e!==void 0).map(e=>e.buffer);if(new Set(v).size!==v.length)throw Error(`${t} outputs must use separate buffers`)}getCommandNodes(n){let{id:i,props:a,eventCount:o,nodeCount:f,edgeCount:p,bandCount:h,simulationCount:g}=this,{rowCount:_,rowsPerBlock:v,blockCount:b}=this,{output:S}=a,{offsets:T,neighbors:E,weights:D}=a.topology;t(i,n,[...K(a),...q(a)]);let O=Math.min(a.laneCount??v,v),k=Math.ceil(v/O);t(i,n,[],{seedTable:_*2*4,events:o*4,histogram:(1+g)*h*4,lengthPrefix:(r(p)+z(p))*4,blockCosts:k*r(O*f)*4,expandedSeeds:b*k*r(2*O)*4,frontier:O*f*8,batchConverged:b*k*V*4,blockInfo:b*V*4}),m(i,`expanded seed words (rows, rowsPerBlock, laneCount)`,b*Math.ceil(v/O)*r(2*O),ae(n.device));let A=[],j=1+g,M=(e,t)=>s(n,`${i}-${e}`,`uint32`,t),N=M(`seed-nodes`,2*_),P=s(n,`${i}-seed-costs`,`float32`,2*_),F=M(`snapped-edges`,o);A.push(...new x({id:`${i}-snap`,spatialContext:this.spatialContext,positions:a.positions,topology:a.topology,maximumSnapDistance:a.maximumSnapDistance,candidateCapacity:a.candidateCapacity,maximumUnindexedWork:2**53-1,spatialSort:a.spatialSort,startDirection:`both`,output:{snappedEdges:F,startNodes:e(n,N,0,2*o),startCosts:e(n,P,0,2*o),status:{overflow:S.status?.overflow}}}).getCommandNodes(n));let I=S.snappedEventCount??M(`event-count`,1),L=M(`event-flags`,o);if(A.push(u(n,{id:`${i}-event-flags`,operation:R,variant:`event-flags`,bindings:[{name:`seedNodes`,view:N,type:`u32`,access:`read`},{name:`validFlags`,view:L,type:`u32`,access:`read_write`}],invocationCount:o,declarations:`const NODE_COUNT: u32 = ${f}u;`,body:`validFlags[validFlagsOffset + index] =
    select(0u, 1u, seedNodes[seedNodesOffset + 2u * index] < NODE_COUNT);`}),...new c({id:`${i}-event-count`,input:L,output:I,operation:`sum`}).getCommandNodes(n)),g>0){let t=M(`quantized-lengths`,p),d=z(p),m=r(p),h=M(`length-prefix`,m+d),_=e(n,h,m,d),v=M(`chunk-totals`,d),y=M(`chunk-maximum`,1),b=M(`quantized-chunks`,d),x=Math.log2(ie(d)),S=s(n,`${i}-live-weights`,`float32`,p),C=s(n,`${i}-live-maximum`,`float32`,1),w=`${ee}
fn getLiveWeight(edge: u32) -> f32 {
  let weight = weights[weightsOffset + edge];
  let isLive = edge < offsets[offsetsOffset + NODE_COUNT] &&
    neighbors[neighborsOffset + edge] < NODE_COUNT && isFiniteBits(weight) && weight >= 0.0;
  return select(0.0, weight, isLive);
}`;A.push(u(n,{id:`${i}-live-weights`,operation:R,variant:`live-weights`,bindings:[{name:`offsets`,view:T,type:`u32`,access:`read`},{name:`neighbors`,view:E,type:`u32`,access:`read`},{name:`weights`,view:D,type:`f32`,access:`read`},{name:`liveWeights`,view:S,type:`f32`,access:`read_write`}],invocationCount:p,declarations:`const NODE_COUNT: u32 = ${f}u;
${w}`,body:`liveWeights[liveWeightsOffset + index] = getLiveWeight(index);`}),...new c({id:`${i}-live-maximum`,input:S,output:C,operation:`max`}).getCommandNodes(n),u(n,{id:`${i}-quantize`,operation:R,variant:`quantize`,bindings:[{name:`liveWeights`,view:S,type:`f32`,access:`read`},{name:`liveMaximum`,view:C,type:`f32`,access:`read`},{name:`quantized`,view:t,type:`u32`,access:`read_write`}],invocationCount:p,body:`let maximum = liveMaximum[liveMaximumOffset];
  var value = 0u;
  if (maximum > 0.0) {
    let scaled = min(liveWeights[liveWeightsOffset + index] / maximum, 1.0) * ${re}.0;
    let whole = floor(scaled);
    value = u32(whole) + select(0u, 1u, scaled - whole >= 0.5);
  }
  quantized[quantizedOffset + index] = value;`}),u(n,{id:`${i}-chunk-prefix`,operation:R,variant:`chunk-prefix`,bindings:[{name:`quantized`,view:t,type:`u32`,access:`read`},{name:`prefix`,view:h,type:`u32`,access:`read_write`},{name:`chunkTotals`,view:v,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const EDGE_COUNT: u32 = ${p}u;
const CHUNK_EDGE_COUNT: u32 = 128u;`,body:`let firstEdge = index * CHUNK_EDGE_COUNT;
  let endEdge = min(firstEdge + CHUNK_EDGE_COUNT, EDGE_COUNT);
  var running = 0u;
  for (var edge = firstEdge; edge < endEdge; edge++) {
    running += quantized[quantizedOffset + edge];
    prefix[prefixOffset + edge] = running;
  }
  chunkTotals[chunkTotalsOffset + index] = running;`}),...new c({id:`${i}-chunk-maximum`,input:v,output:y,operation:`max`}).getCommandNodes(n),u(n,{id:`${i}-chunk-quantize`,operation:R,variant:`chunk-quantize`,bindings:[{name:`chunkTotals`,view:v,type:`u32`,access:`read`},{name:`chunkMaximum`,view:y,type:`u32`,access:`read`},{name:`quantizedChunks`,view:b,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const CHUNK_RANGE_BITS: u32 = ${x}u;`,body:`let maximumBits = 32u - countLeadingZeros(chunkMaximum[chunkMaximumOffset]);
  let total = chunkTotals[chunkTotalsOffset + index];
  var value = total;
  if (maximumBits > CHUNK_RANGE_BITS) {
    let shift = min(maximumBits - CHUNK_RANGE_BITS, 31u);
    value = (total >> shift) + ((total >> (shift - 1u)) & 1u);
  }
  quantizedChunks[quantizedChunksOffset + index] = value;`}),...new l({id:`${i}-length-scan`,input:b,output:_,mode:`inclusive`}).getCommandNodes(n),u(n,{id:`${i}-simulate`,operation:R,variant:`simulate`,bindings:[{name:`prefix`,view:h,type:`u32`,access:`read`},{name:`offsets`,view:T,type:`u32`,access:`read`},{name:`neighbors`,view:E,type:`u32`,access:`read`},{name:`weights`,view:D,type:`f32`,access:`read`},{name:`validCount`,view:I,type:`u32`,access:`read`},{name:`parameters`,view:a.parameters,type:`u32`,access:`read`},{name:`outNodes`,view:N,type:`u32`,access:`read_write`},{name:`outCosts`,view:P,type:`f32`,access:`read_write`}],invocationCount:g*o,declarations:`const NODE_COUNT: u32 = ${f}u;
const EDGE_COUNT: u32 = ${p}u;
const EVENT_COUNT: u32 = ${o}u;
const SIMULATION_COUNT: u32 = ${g}u;
const CHUNK_EDGE_COUNT: u32 = 128u;
const CHUNK_COUNT: u32 = ${d}u;
const CHUNK_PREFIX_BASE: u32 = ${m}u;
const NONE: u32 = ${oe}u;
const STREAM_TAG: u32 = ${ne}u;
${te}`,body:`let pattern = 1u + index / EVENT_COUNT;
  let eventIndex = index - (pattern - 1u) * EVENT_COUNT;
  let slot = 2u * (pattern * EVENT_COUNT + eventIndex);
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  let total = prefix[prefixOffset + CHUNK_PREFIX_BASE + CHUNK_COUNT - 1u];
  var firstNode = NONE;
  var secondNode = NONE;
  var firstCost = -1.0;
  var secondCost = -1.0;
  if (pattern <= activeCount && eventIndex < validCount[validCountOffset] && total > 0u) {
    var stream = createPhiloxStream(
      vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]),
      eventIndex,
      pattern,
      STREAM_TAG
    );
    // Level 1: a chunk, by the prefix sum of the quantized chunk totals. The first chunk whose
    // prefix exceeds the draw has a positive quantized total, so its own total is positive too.
    let chunkDraw = nextPhiloxBelow(&stream, total);
    var chunkLow = 0u;
    var chunkHigh = CHUNK_COUNT;
    while (chunkLow < chunkHigh) {
      let middle = (chunkLow + chunkHigh) / 2u;
      if (prefix[prefixOffset + CHUNK_PREFIX_BASE + middle] > chunkDraw) {
        chunkHigh = middle;
      } else {
        chunkLow = middle + 1u;
      }
    }
    // Level 2: an edge of the chunk, by the chunk's own prefix sum of 2^24-unit edge lengths.
    let chunkFirstEdge = chunkLow * CHUNK_EDGE_COUNT;
    let chunkEndEdge = min(chunkFirstEdge + CHUNK_EDGE_COUNT, EDGE_COUNT);
    let draw = nextPhiloxBelow(&stream, prefix[prefixOffset + chunkEndEdge - 1u]);
    let fractionBits = nextPhiloxUint32(&stream);
    var low = chunkFirstEdge;
    var high = chunkEndEdge;
    while (low < high) {
      let middle = (low + high) / 2u;
      if (prefix[prefixOffset + middle] > draw) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    let edge = low;
    var nodeLow = 0u;
    var nodeHigh = NODE_COUNT;
    while (nodeLow < nodeHigh) {
      let middle = (nodeLow + nodeHigh) / 2u;
      if (offsets[offsetsOffset + middle + 1u] > edge) {
        nodeHigh = middle;
      } else {
        nodeLow = middle + 1u;
      }
    }
    let fraction = f32(fractionBits >> 8u) / 16777216.0;
    let weight = weights[weightsOffset + edge];
    firstNode = nodeLow;
    secondNode = neighbors[neighborsOffset + edge];
    firstCost = fraction * weight;
    secondCost = (1.0 - fraction) * weight;
  }
  outNodes[outNodesOffset + slot] = firstNode;
  outNodes[outNodesOffset + slot + 1u] = secondNode;
  outCosts[outCostsOffset + slot] = firstCost;
  outCosts[outCostsOffset + slot + 1u] = secondCost;`}))}let B=M(`histogram`,j*h);A.push(d(n,{id:`${i}-clear`,operation:R,view:B,type:`u32`,value:`0u`}));let H=Math.min(a.laneCount??v,v);if(H*f>=4294967295)throw Error(`${i} laneCount * nodeCount must fit in uint32`);let U=Math.ceil(v/H),W=b*U,G=r(H*f),J=r(2*H);y(n.device,i,`block costs`,U*G*4,`lower rowsPerBlock`);let Y=s(n,`${i}-costs`,`float32`,U*G),X=M(`expanded-seeds`,W*J),Z=s(n,`${i}-expanded-seed-costs`,`float32`,W*J);A.push(u(n,{id:`${i}-expand-seeds`,operation:R,variant:`expand-seeds`,bindings:[{name:`seedNodes`,view:N,type:`u32`,access:`read`},{name:`seedCosts`,view:P,type:`f32`,access:`read`},{name:`expandedSeeds`,view:X,type:`u32`,access:`read_write`},{name:`expandedSeedCosts`,view:Z,type:`f32`,access:`read_write`}],invocationCount:W*2*H,declarations:`const NODE_COUNT: u32 = ${f}u;
const ROW_COUNT: u32 = ${_}u;
const ROWS_PER_BLOCK: u32 = ${v}u;
const LANE_COUNT: u32 = ${H}u;
const BATCHES_PER_BLOCK: u32 = ${U}u;
const SEED_STRIDE: u32 = ${J}u;`,body:`let batch = index / (2u * LANE_COUNT);
  let slot = index - batch * 2u * LANE_COUNT;
  let lane = slot / 2u;
  let blockIndex = batch / BATCHES_PER_BLOCK;
  let rowInBlock = (batch - blockIndex * BATCHES_PER_BLOCK) * LANE_COUNT + lane;
  let row = blockIndex * ROWS_PER_BLOCK + rowInBlock;
  var node = ${C}u;
  var cost = -1.0;
  if (rowInBlock < ROWS_PER_BLOCK && row < ROW_COUNT) {
    let seedNode = seedNodes[seedNodesOffset + 2u * row + (slot & 1u)];
    if (seedNode < NODE_COUNT) {
      node = lane * NODE_COUNT + seedNode;
      cost = seedCosts[seedCostsOffset + 2u * row + (slot & 1u)];
    }
  }
  expandedSeeds[expandedSeedsOffset + batch * SEED_STRIDE + slot] = node;
  expandedSeedCosts[expandedSeedCostsOffset + batch * SEED_STRIDE + slot] = cost;`}));let Q=M(`block-info`,b*V);A.push(u(n,{id:`${i}-block-info`,operation:R,variant:`block-info`,bindings:[{name:`blockInfo`,view:Q,type:`u32`,access:`read_write`}],invocationCount:b,declarations:`const ROW_COUNT: u32 = ${_}u;
const ROWS_PER_BLOCK: u32 = ${v}u;
const SLOT_WORDS: u32 = ${V}u;`,body:`let firstRow = index * ROWS_PER_BLOCK;
  blockInfo[blockInfoOffset + index * SLOT_WORDS] = firstRow;
  blockInfo[blockInfoOffset + index * SLOT_WORDS + 1u] = min(ROWS_PER_BLOCK, ROW_COUNT - firstRow);`}));let $=S.status?.converged||S.status?.iterationLimitReached?M(`batch-converged`,W*V):void 0;for(let t=0;t<b;t++){for(let r=0;r<U;r++){let o=t*U+r;A.push(...new w({id:`${i}-block-${t}-batch-${r}`,topology:{offsets:T,neighbors:E,weights:D},laneCount:H,startNodes:e(n,X,o*J,2*H),startCosts:e(n,Z,o*J,2*H),costLimit:a.maximumDistance,maximumIterations:a.maximumIterations,localIterations:a.localIterations??se,output:{costs:e(n,Y,r*G,H*f),status:{converged:$?e(n,$,o*V,1):void 0}}}).getCommandNodes(n))}A.push(u(n,{id:`${i}-block-${t}-count`,operation:R,variant:`count`,bindings:[{name:`costs`,view:Y,type:`f32`,access:`read`},{name:`seedNodes`,view:N,type:`u32`,access:`read`},{name:`seedCosts`,view:P,type:`f32`,access:`read`},{name:`maximumDistanceIn`,view:a.maximumDistance,type:`f32`,access:`read`},{name:`histogram`,view:B,type:`atomic<u32>`,access:`read_write`},{name:`blockInfo`,view:e(n,Q,t*V,2),type:`u32`,access:`read`}],invocationCount:v*o,guardIndex:!1,declarations:`const NODE_COUNT: u32 = ${f}u;
const EVENT_COUNT: u32 = ${o}u;
const BAND_COUNT: u32 = ${h}u;
const LANE_COUNT: u32 = ${H}u;
const BATCH_STRIDE: u32 = ${G}u;
const WORKGROUP_SIZE: u32 = 64u;
// Per-workgroup histogram of the pattern that owns the workgroup's first invocation. A workgroup
// spans 64 consecutive (row, column) pairs, so nearly all of them share one pattern and one
// global atomic per pair collapses into one local atomic and at most BAND_COUNT global adds.
var<workgroup> localHistogram: array<atomic<u32>, ${h}>;
fn isFiniteCost(cost: f32) -> bool {
  return bitcast<u32>(cost) < 0x7f800000u;
}
fn getThreshold(maximumDistance: f32, band: u32) -> f32 {
  return maximumDistance * f32(band) / f32(BAND_COUNT - 1u);
}
// The block's first row, read from its info slot (rule 12: the block index is data).
var<private> blockFirstRow: u32;
fn getPattern(index: u32) -> u32 {
  return (blockFirstRow + index / EVENT_COUNT) / EVENT_COUNT;
}
// Start of a block row's costs: batch (localRow / LANE_COUNT) at BATCH_STRIDE, then its lane.
fn getRowCostBase(localRow: u32) -> u32 {
  let batch = localRow / LANE_COUNT;
  return costsOffset + batch * BATCH_STRIDE + (localRow - batch * LANE_COUNT) * NODE_COUNT;
}
// Distance band of the unordered pair of row and column events, or BAND_COUNT when the pair is
// not counted.
fn classifyPair(index: u32) -> u32 {
  let localRow = index / EVENT_COUNT;
  let columnEvent = index - localRow * EVENT_COUNT;
  let row = blockFirstRow + localRow;
  let pattern = row / EVENT_COUNT;
  let eventIndex = row - pattern * EVENT_COUNT;
  let maximumDistance = maximumDistanceIn[maximumDistanceInOffset];
  if (columnEvent <= eventIndex || !(maximumDistance > 0.0)) {
    return BAND_COUNT;
  }
  let rowSlot = 2u * row;
  let columnSlot = 2u * (pattern * EVENT_COUNT + columnEvent);
  let rowFirst = seedNodes[seedNodesOffset + rowSlot];
  let rowSecond = seedNodes[seedNodesOffset + rowSlot + 1u];
  let columnFirst = seedNodes[seedNodesOffset + columnSlot];
  let columnSecond = seedNodes[seedNodesOffset + columnSlot + 1u];
  if (rowFirst >= NODE_COUNT || rowSecond >= NODE_COUNT ||
      columnFirst >= NODE_COUNT || columnSecond >= NODE_COUNT) {
    return BAND_COUNT;
  }
  let rowFirstCost = seedCosts[seedCostsOffset + rowSlot];
  let columnFirstCost = seedCosts[seedCostsOffset + columnSlot];
  let columnSecondCost = seedCosts[seedCostsOffset + columnSlot + 1u];
  var best = 0.0;
  var found = false;
  let viaFirst = costs[getRowCostBase(localRow) + columnFirst];
  if (isFiniteCost(viaFirst)) {
    best = viaFirst + columnFirstCost;
    found = true;
  }
  let viaSecond = costs[getRowCostBase(localRow) + columnSecond];
  if (isFiniteCost(viaSecond)) {
    let candidate = viaSecond + columnSecondCost;
    if (!found || candidate < best) {
      best = candidate;
      found = true;
    }
  }
  var sameEdge = -1.0;
  if (rowFirst != rowSecond) {
    if (rowFirst == columnFirst && rowSecond == columnSecond) {
      sameEdge = abs(rowFirstCost - columnFirstCost);
    } else if (rowFirst == columnSecond && rowSecond == columnFirst) {
      sameEdge = abs(rowFirstCost - columnSecondCost);
    }
  }
  if (sameEdge >= 0.0 && (!found || sameEdge < best)) {
    best = sameEdge;
    found = true;
  }
  if (!found || !(best < maximumDistance)) {
    return BAND_COUNT;
  }
  var band = u32(max(floor(best / (maximumDistance / f32(BAND_COUNT - 1u))), 0.0)) + 1u;
  band = min(band, BAND_COUNT);
  while (band > 0u && best < getThreshold(maximumDistance, band - 1u)) {
    band--;
  }
  while (band < BAND_COUNT && !(best < getThreshold(maximumDistance, band))) {
    band++;
  }
  return band;
}`,workgroupSize:64,body:`blockFirstRow = blockInfo[blockInfoOffset];
  let liveCount = blockInfo[blockInfoOffset + 1u] * EVENT_COUNT;
  let firstIndex = index - localInvocationIndex;
  let ownerPattern = getPattern(min(firstIndex, liveCount - 1u));
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    atomicStore(&localHistogram[bin], 0u);
  }
  workgroupBarrier();
  if (index < liveCount) {
    let band = classifyPair(index);
    if (band < BAND_COUNT) {
      let pattern = getPattern(index);
      if (pattern == ownerPattern) {
        atomicAdd(&localHistogram[band], 1u);
      } else {
        atomicAdd(&histogram[histogramOffset + pattern * BAND_COUNT + band], 1u);
      }
    }
  }
  workgroupBarrier();
  for (var bin = localInvocationIndex; bin < BAND_COUNT; bin += WORKGROUP_SIZE) {
    let binCount = atomicLoad(&localHistogram[bin]);
    if (binCount > 0u) {
      atomicAdd(&histogram[histogramOffset + ownerPattern * BAND_COUNT + bin], binCount);
    }
  }`}))}let ce=[{name:`histogram`,view:B,type:`u32`,access:`read`},{name:`networkLength`,view:a.networkLength,type:`f32`,access:`read`},{name:`validCount`,view:I,type:`u32`,access:`read`},{name:`parameters`,view:a.parameters,type:`u32`,access:`read`},{name:`kValues`,view:S.kValues,type:`f32`,access:`read_write`}];if(S.pairCounts&&ce.push({name:`pairCounts`,view:S.pairCounts,type:`u32`,access:`read_write`}),A.push(u(n,{id:`${i}-finalize`,operation:R,variant:`finalize`,bindings:ce,invocationCount:j*h,declarations:`const BAND_COUNT: u32 = ${h}u;
const SIMULATION_COUNT: u32 = ${g}u;`,body:`let pattern = index / BAND_COUNT;
  let band = index - pattern * BAND_COUNT;
  let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var cumulative = 0u;
  if (pattern <= activeCount) {
    for (var bin = 0u; bin <= band; bin++) {
      cumulative += histogram[histogramOffset + pattern * BAND_COUNT + bin];
    }
  }
  let count = f32(validCount[validCountOffset]);
  var k = 0.0;
  if (count >= 2.0) {
    k = 2.0 * f32(cumulative) * networkLength[networkLengthOffset] / (count * count);
  }
  kValues[kValuesOffset + index] = k;
  ${S.pairCounts?`pairCounts[pairCountsOffset + index] = cumulative;`:``}`})),S.envelope&&A.push(u(n,{id:`${i}-envelope`,operation:R,variant:`envelope`,bindings:[{name:`kValues`,view:S.kValues,type:`f32`,access:`read`},{name:`parameters`,view:a.parameters,type:`u32`,access:`read`},{name:`envelope`,view:S.envelope,type:`f32`,access:`read_write`}],invocationCount:h,declarations:`const BAND_COUNT: u32 = ${h}u;
const SIMULATION_COUNT: u32 = ${g}u;`,body:`let activeCount = min(parameters[parametersOffset + 2u], SIMULATION_COUNT);
  var lowest = 0.0;
  var highest = 0.0;
  var sum = 0.0;
  for (var pattern = 1u; pattern <= activeCount; pattern++) {
    let value = kValues[kValuesOffset + pattern * BAND_COUNT + index];
    lowest = select(min(lowest, value), value, pattern == 1u);
    highest = select(max(highest, value), value, pattern == 1u);
    sum += value;
  }
  envelope[envelopeOffset + index] = lowest;
  envelope[envelopeOffset + BAND_COUNT + index] = select(0.0, sum / f32(activeCount), activeCount > 0u);
  envelope[envelopeOffset + 2u * BAND_COUNT + index] = highest;`})),$){let e=[...S.status?.converged?[{name:`converged`,view:S.status.converged}]:[],...S.status?.iterationLimitReached?[{name:`iterationLimitReached`,view:S.status.iterationLimitReached}]:[]];A.push(u(n,{id:`${i}-converged-initialize`,operation:R,variant:`converged-initialize`,invocationCount:1,bindings:e.map(e=>({...e,type:`u32`,access:`read_write`})),body:`${S.status?.converged?`converged[convergedOffset] = 1u;`:``}
          ${S.status?.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = 0u;`:``}`}),u(n,{id:`${i}-converged`,operation:R,variant:`converged`,invocationCount:W,bindings:[{name:`batchConverged`,view:$,type:`u32`,access:`read`},...e.map(e=>({...e,type:`atomic<u32>`,access:`read_write`}))],body:`let done = batchConverged[batchConvergedOffset + index * ${V}u];
          ${S.status?.converged?`atomicMin(&converged[convergedOffset], done);`:``}
          ${S.status?.iterationLimitReached?`atomicMax(&iterationLimitReached[iterationLimitReachedOffset], 1u - done);`:``}`}))}return A}};function K(e){return[e.positions,...g(e.topology),e.maximumSnapDistance,e.maximumDistance,e.networkLength,e.parameters]}function q(e){let{output:t}=e;return[t.kValues,t.pairCounts,t.envelope,t.snappedEventCount,t.status?.overflow,t.status?.converged,t.status?.iterationLimitReached]}var J=[.56,.14,.97,.88],Y=90,X=48,Z=19,Q=E({parameters:F,caption:`Observed K(d) against the envelope of random patterns on the same network.`,createInputs:({random:e,world:t})=>{let n=k(e,t,8),r=0;for(let e=0;e<n.slotCount;e++)r+=n.lengths[e]/2;return{network:n,world:t,inset:j(t),networkLength:r,uniform:Float32Array.from({length:180},(n,r)=>r%2==0?e()*t.width:e()),hotspots:Float32Array.from({length:180},(n,r)=>r%2==0?t.width*(.25+.5*(Math.floor(r/2)%2))+(e()-.5)*.18:.5+(e()-.5)*.5),events:new Float32Array(180)}},build(e,t){let{network:n}=t,{topology:r}=O(e,n),i=e.upload(`display-nodes`,`float32x2`,t.inset.transform(n.nodes)),a=e.allocate(`event-dots`,`float32x2`,Y),o=e.upload(`node-positions`,`float32x2`,n.nodes),s=e.allocate(`events`,`float32x2`,Y),c=e.allocate(`maximum-distance`,`float32`,1),l=e.allocate(`network-length`,`float32`,1),u=e.allocate(`parameters`,`uint32`,4),d=e.allocate(`k-values`,`float32`,960),f=e.allocate(`envelope`,`float32`,144),p=e.allocate(`snapped`,`uint32`,1),m=e.allocate(`converged`,`uint32`,1);return e.graph.add(new G({positions:s.view,topology:{...r,nodePositions:o.view},maximumDistance:c.view,networkLength:l.view,parameters:u.view,bandCount:X,simulationCount:Z,maximumIterations:48,output:{kValues:d.view,envelope:f.view,snappedEventCount:p.view,status:{converged:m.view}}})),{outputs:{nodes:i,edges:e.upload(`road-edges`,`uint32x2`,n.edgePairs),eventDots:a,observed:M(e,{id:`observed`,source:d,offset:0,length:X}),lower:M(e,{id:`lower`,source:f,offset:0,length:X}),upper:M(e,{id:`upper`,source:f,offset:96,length:X})},update:e=>{for(let n=0;n<t.events.length;n++)t.events[n]=t.uniform[n]*(1-e.clustering)+t.hotspots[n]*e.clustering;s.write(t.events),a.write(t.inset.transform(t.events)),c.write(Float32Array.of(e.maximumDistance)),l.write(Float32Array.of(t.networkLength)),u.write(I({seed:4306,activeSimulations:Math.round(e.activeSimulations)}))}}},interaction:{kind:`level`,hint:`read K at a distance`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=N(J,i,e.x,e.y),s={u:e.unitX<J[0]?e.unitX/J[0]:o.u},c=Math.min(47,Math.round(s.u*47)),l=a(n.observed),u=a(n.lower),d=a(n.upper),f=(c+1)/X*r.maximumDistance,p=l&&u&&d?`K ${l[c].toFixed(1)} · envelope ${u[c].toFixed(1)}-${d[c].toFixed(1)}`:``,m=0;for(let e=1;e<Y;e++)Math.hypot(t.events[e*2]-i.width/2,t.events[e*2+1]-.5)<Math.hypot(t.events[m*2]-i.width/2,t.events[m*2+1]-.5)&&(m=e);return{glyphs:[{shape:`curtain`,x:P(J,i,c/47)},{shape:`ring`,at:t.inset.toMap(t.events[m*2],t.events[m*2+1]),radius:f*t.inset.scale}],readout:`d ${f.toFixed(2)} · ${p}`}}},draw:({inputs:e,outputs:t})=>{let n=[0,e.networkLength*1.05];return[A(t),{layer:`dots`,positions:t.eventDots,radius:4,color:[1,.78,.25,1]},{layer:`chart`,kind:`line`,rectangle:J,values:t.upper,range:n,color:[.5,.7,1,.6],radius:1.5},{layer:`chart`,kind:`line`,rectangle:J,values:t.lower,range:n,color:[.5,.7,1,.6],radius:1.5},{layer:`chart`,kind:`line`,rectangle:J,values:t.observed,range:n,color:[1,.78,.25,1],radius:3},{layer:`text`,...D({corner:`top-left`,entries:[{label:`observed K(r)`,swatch:[1,.78,.25]},{label:`random-event envelope`,swatch:[.5,.7,1]},{label:`events`,swatch:[1,.78,.25]}]})}]}});export{Q as default};