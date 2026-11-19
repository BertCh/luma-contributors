import{a as e,i as t,n,r,t as i}from"./raster-sampling-utils-wF9rn6Mp.js";import{n as a}from"./raster-storage-preflight-CIxFc-0g.js";import{a as o,g as s,i as c,o as l,r as u,s as d}from"./raster-algebra-input-5ZxymrFJ.js";import{t as f}from"./common-CshkX-d_.js";import{r as p}from"./kernel-C3tfQOn4.js";import{t as m}from"./gpu-scan-B4NNTfF9.js";import{o as h,t as g}from"./wgsl-kernel-nodes-DagJklH2.js";import{S as _,s as v,u as y,x as ee,y as b}from"./gpu-contributor-utils-BIwLQON5.js";import{r as x,u as S}from"./wgsl-common-DfyXiYB0.js";import{t as C}from"./define-run-B-brkCe4.js";import{n as w,t as T}from"./elevation-DSYnxAKv.js";import{c as E}from"./pointer-overlay-Qh8qflcA.js";import{Fi as D}from"./entry-preview-D70TdSbK.js";import{r as O,t as k}from"./bands-CVDIl0aE.js";function A(e){return Math.max(1,Math.ceil(e/256))}var j=`
var<workgroup> scanScratch: array<ScanState, 256>;
fn scanWorkgroup(lane: u32, value: ScanState) -> ScanState {
  scanScratch[lane] = value;
  workgroupBarrier();
  for (var spanLength = 1u; spanLength < 256u; spanLength = spanLength << 1u) {
    var leftValue = getIdentityScanState();
    if (lane >= spanLength) {
      leftValue = scanScratch[lane - spanLength];
    }
    workgroupBarrier();
    if (lane >= spanLength) {
      scanScratch[lane] = combineScanStates(leftValue, scanScratch[lane]);
    }
    workgroupBarrier();
  }
  return scanScratch[lane];
}
`;function M(e){let{writesAggregates:t,loadsItems:n}=e;return`
struct ScanState {
  flag: u32,
  sum: f32
}
fn getIdentityScanState() -> ScanState {
  return ScanState(0u, 0.0);
}
// Segmented sum: a flagged right operand starts a new segment and replaces the running sum.
fn combineScanStates(left: ScanState, right: ScanState) -> ScanState {
  var result = right;
  if (right.flag == 0u) {
    result.sum = left.sum + right.sum;
  }
  result.flag = left.flag | right.flag;
  return result;
}
fn loadAggregate(blockIndex: u32) -> ScanState {
  let base = aggregatesOffset + 2u * blockIndex;
  return ScanState(aggregates[base], bitcast<f32>(aggregates[base + 1u]));
}
${t?`fn storeAggregate(blockIndex: u32, value: ScanState) {
  let base = aggregatesOffset + 2u * blockIndex;
  aggregates[base] = value.flag;
  aggregates[base + 1u] = bitcast<u32>(value.sum);
}`:``}
${n?`// Number of lines whose first vertex is at or before the vertex (lines are sorted by offset).
fn countLinesStartedBy(vertex: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (getVertexBegin(middle) <= vertex) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}
// Edge length from the previous vertex; a line's first vertex (or one outside every line) starts
// a segment with distance 0.
fn getVertexScanState(vertex: u32) -> ScanState {
  let startedCount = countLinesStartedBy(vertex);
  if (startedCount == 0u || getVertexBegin(startedCount - 1u) == vertex) {
    return ScanState(1u, 0.0);
  }
  let currentX = linePositions[linePositionsOffset + 2u * vertex];
  let currentY = linePositions[linePositionsOffset + 2u * vertex + 1u];
  let deltaX = currentX - linePositions[linePositionsOffset + 2u * vertex - 2u];
  let deltaY = currentY - linePositions[linePositionsOffset + 2u * vertex - 1u];
  return ScanState(0u, sqrt(deltaX * deltaX + deltaY * deltaY));
}`:``}
${j}`}function N(e){let{writesAggregates:t,loadsItems:n}=e;return`
struct ScanState {
  flag: u32,
  hasValue: u32,
  firstValue: f32,
  lastValue: f32,
  gain: f32,
  loss: f32,
  minimum: f32,
  maximum: f32
}
fn getIdentityScanState() -> ScanState {
  return ScanState(0u, 0u, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}
// Segmented profile monoid: a flagged right operand starts a new line and replaces the running
// state. The junction delta is the difference of the last finite value on the left and the first
// finite value on the right, exactly the consecutive finite pair of a serial walk.
fn combineScanStates(left: ScanState, right: ScanState) -> ScanState {
  var result = right;
  if (right.flag == 0u) {
    if (left.hasValue == 0u) {
      result = right;
    } else if (right.hasValue == 0u) {
      result = left;
    } else {
      let delta = right.firstValue - left.lastValue;
      result.firstValue = left.firstValue;
      result.gain = (left.gain + max(delta, 0.0)) + right.gain;
      result.loss = (left.loss + max(-delta, 0.0)) + right.loss;
      result.minimum = min(left.minimum, right.minimum);
      result.maximum = max(left.maximum, right.maximum);
    }
  }
  result.flag = left.flag | right.flag;
  return result;
}
fn loadAggregate(blockIndex: u32) -> ScanState {
  let base = aggregatesOffset + 8u * blockIndex;
  return ScanState(
    aggregates[base],
    aggregates[base + 1u],
    bitcast<f32>(aggregates[base + 2u]),
    bitcast<f32>(aggregates[base + 3u]),
    bitcast<f32>(aggregates[base + 4u]),
    bitcast<f32>(aggregates[base + 5u]),
    bitcast<f32>(aggregates[base + 6u]),
    bitcast<f32>(aggregates[base + 7u])
  );
}
${t?`fn storeAggregate(blockIndex: u32, value: ScanState) {
  let base = aggregatesOffset + 8u * blockIndex;
  aggregates[base] = value.flag;
  aggregates[base + 1u] = value.hasValue;
  aggregates[base + 2u] = bitcast<u32>(value.firstValue);
  aggregates[base + 3u] = bitcast<u32>(value.lastValue);
  aggregates[base + 4u] = bitcast<u32>(value.gain);
  aggregates[base + 5u] = bitcast<u32>(value.loss);
  aggregates[base + 6u] = bitcast<u32>(value.minimum);
  aggregates[base + 7u] = bitcast<u32>(value.maximum);
}`:``}
${n?`// Line of an emitted sample: the last line whose first sample row is at or before it.
fn findSampleLine(sample: u32) -> u32 {
  var low = 0u;
  var high = LINE_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (sampleOffsets[sampleOffsetsOffset + middle] <= sample) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low - 1u;
}
fn getSampleScanState(sample: u32, line: u32) -> ScanState {
  var result = getIdentityScanState();
  if (sampleOffsets[sampleOffsetsOffset + line] == sample) {
    result.flag = 1u;
  }
  let value = sampleValues[sampleValuesOffset + sample];
  if (isFiniteValue(value)) {
    result.hasValue = 1u;
    result.firstValue = value;
    result.lastValue = value;
    result.minimum = value;
    result.maximum = value;
  }
  return result;
}
fn getEmittedSampleCount() -> u32 {
  return min(sampleOffsets[sampleOffsetsOffset + LINE_COUNT], SAMPLE_CAPACITY);
}`:``}
${j}`}function P(e){return`let lane = localInvocationIndex;
  ${e}
  let inclusive = scanWorkgroup(lane, itemState);
  if (lane == 255u) {
    storeAggregate(workgroupIndex, inclusive);
  }`}function F(){return`let lane = localInvocationIndex;
  let chunk = (BLOCK_COUNT + 255u) / 256u;
  let chunkBegin = min(lane * chunk, BLOCK_COUNT);
  let chunkEnd = min(chunkBegin + chunk, BLOCK_COUNT);
  var chunkTotal = getIdentityScanState();
  for (var chunkBlock = chunkBegin; chunkBlock < chunkEnd; chunkBlock++) {
    chunkTotal = combineScanStates(chunkTotal, loadAggregate(chunkBlock));
  }
  let inclusive = scanWorkgroup(lane, chunkTotal);
  var running = getIdentityScanState();
  if (lane > 0u) {
    running = scanScratch[lane - 1u];
  }
  for (var chunkBlock = chunkBegin; chunkBlock < chunkEnd; chunkBlock++) {
    let aggregate = loadAggregate(chunkBlock);
    storeAggregate(chunkBlock, running);
    running = combineScanStates(running, aggregate);
  }
  _ = inclusive;`}var I=`GPURasterProfile`,L=1<<24;function R(e){return y(e,16)}var z=class{id;props;lineCount;sampleCapacity;constructor(e){this.id=e.id??`raster-profile`,this.props=e;let{id:n}=this,{output:a}=e,o=r(n,e.grid,e.noDataValue);if(l(n,`values`,e.values,e.grid,o),e.mask&&t(n,`mask`,e.mask,`uint32`,o),t(n,`positions`,e.positions,`float32x2`,1),t(n,`lineOffsets`,e.lineOffsets,`uint32`,2),t(n,`parameters`,e.parameters,`float32`,12),!a?.status?.count||!a.status.overflow)throw Error(`${n} needs output.status.count and output.status.overflow`);this.lineCount=e.lineOffsets.length-1;let c=[a.samplePositions,a.sampleDistances,a.sampleValues,a.sampleLineIds,a.sampleCumulativeGain,a.sampleCumulativeLoss].filter(e=>e!==void 0),u=e.sampleCapacity??c[0]?.length;if(u===void 0)throw Error(`${n} needs sampleCapacity when no sample column is provided`);if(!Number.isSafeInteger(u)||u<1)throw Error(`${n} sampleCapacity must be a positive integer`);this.sampleCapacity=u;for(let[e,r]of[[`samplePositions`,`float32x2`],[`sampleDistances`,`float32`],[`sampleValues`,`float32`],[`sampleLineIds`,`uint32`],[`sampleCumulativeGain`,`float32`],[`sampleCumulativeLoss`,`float32`]])a[e]&&t(n,`output.${e}`,a[e],r,u);for(let e of[`count`,`overflow`,`requiredCount`])a.status[e]&&t(n,`output.status.${e}`,a.status[e],`uint32`,1);a.lineSampleOffsets&&t(n,`output.lineSampleOffsets`,a.lineSampleOffsets,`uint32`,this.lineCount+1);for(let e of[`lineLengths`,`lineGains`,`lineLosses`,`lineMinimums`,`lineMaximums`])a[e]&&t(n,`output.${e}`,a[e],`float32`,this.lineCount);let{status:d,...f}=a;s(n,Object.fromEntries([...Object.entries(f).map(([e,t])=>[`output.${e}`,t]),...Object.entries(d).map(([e,t])=>[`output.status.${e}`,t])])),b(n,Object.fromEntries([...Object.entries(f).map(([e,t])=>[`output.${e}`,t]),...Object.entries(d).map(([e,t])=>[`output.status.${e}`,t])]),{...i(e.values),mask:e.mask,positions:e.positions,lineOffsets:e.lineOffsets,parameters:e.parameters})}getCommandNodes(e){_(this.id,`line count (lineOffsets.length - 1)`,this.lineCount,R(e.device));let t=this.props;d(e,this.id,[t.values],t.grid),a(e,this.id,[t.mask,t.positions,t.lineOffsets,t.parameters,...v(t.output)],{"vertex distances":t.positions.length*4,"sample counts and offsets":(this.lineCount+1)*4,"sample positions":t.output.samplePositions?0:this.sampleCapacity*8,"sample values":t.output.sampleValues?0:this.sampleCapacity*4,"vertex scan states":A(t.positions.length)*2*4,"sample scan states":A(this.sampleCapacity)*8*4,"line summaries":t.output.lineGains||t.output.lineLosses||t.output.lineMinimums||t.output.lineMaximums?this.lineCount*16:0,"truncation status":4});let{id:r,props:i,lineCount:s,sampleCapacity:l}=this,{output:y}=i;ee(r,e,[...c(i.values),i.mask,i.positions,i.lineOffsets,i.parameters,...Object.values({...y,status:void 0}),...Object.values(y.status)]);let b=i.positions.length,C=f(Math.floor(4294967295/(s+1)),1,L),w=u(e,`${r}-source`,i.values,i.grid),T=[...w.nodes],E=p(e,`${r}-vertex-distances`,`float32`,b),D=p(e,`${r}-sample-counts`,`uint32`,s+1),O=p(e,`${r}-sample-offsets`,`uint32`,s+1),k=p(e,`${r}-truncated`,`uint32`,1),j=y.samplePositions??p(e,`${r}-sample-positions`,`float32x2`,l),z=y.sampleValues??p(e,`${r}-sample-values`,`float32`,l),B=`const LINE_COUNT: u32 = ${s}u;
const VERTEX_COUNT: u32 = ${b}u;
const SAMPLE_CAPACITY: u32 = ${l}u;
const MAXIMUM_PER_LINE: u32 = ${C}u;
fn getVertexBegin(line: u32) -> u32 {
  return min(lineOffsets[lineOffsetsOffset + line], VERTEX_COUNT);
}
fn getVertexEnd(line: u32, vertexBegin: u32) -> u32 {
  return max(min(lineOffsets[lineOffsetsOffset + line + 1u], VERTEX_COUNT), vertexBegin);
}`,V=A(b),H=p(e,`${r}-vertex-aggregates`,`uint32`,V*2),U=e=>[{name:`linePositions`,view:i.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`},{name:`aggregates`,view:H,type:`u32`,access:e}];T.push(g(e,{id:`${r}-truncated-fill`,operation:I,view:k,type:`u32`,value:`0u`}),h(e,{id:`${r}-measure`,operation:I,variant:`measure`,bindings:U(`read_write`),invocationCount:V*256,guardIndex:!1,workgroupSize:256,declarations:`${B}
${M({writesAggregates:!0,loadsItems:!0})}`,body:P(`var itemState = getIdentityScanState();
  if (index < VERTEX_COUNT) {
    itemState = getVertexScanState(index);
  }`)}),h(e,{id:`${r}-measure-carries`,operation:I,variant:`measure-carries`,bindings:[U(`read_write`)[2]],invocationCount:256,guardIndex:!1,workgroupSize:256,declarations:`const BLOCK_COUNT: u32 = ${V}u;
${M({writesAggregates:!0,loadsItems:!1})}`,body:F()}),h(e,{id:`${r}-measure-distances`,operation:I,variant:`measure-distances`,bindings:[...U(`read`),{name:`vertexDistances`,view:E,type:`f32`,access:`read_write`}],invocationCount:V*256,guardIndex:!1,workgroupSize:256,declarations:`${B}
${M({writesAggregates:!1,loadsItems:!0})}`,body:`var itemState = getIdentityScanState();
  if (index < VERTEX_COUNT) {
    itemState = getVertexScanState(index);
  }
  let prefix = scanWorkgroup(localInvocationIndex, itemState);
  let result = combineScanStates(loadAggregate(workgroupIndex), prefix);
  if (index < VERTEX_COUNT) {
    vertexDistances[vertexDistancesOffset + index] = result.sum;
  }`}),h(e,{id:`${r}-sample-counts`,operation:I,variant:`sample-counts`,bindings:[{name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`},{name:`params`,view:i.parameters,type:`f32`,access:`read`},{name:`vertexDistances`,view:E,type:`f32`,access:`read`},{name:`sampleCounts`,view:D,type:`u32`,access:`read_write`},{name:`truncated`,view:k,type:`atomic<u32>`,access:`read_write`}],invocationCount:s+1,declarations:B,body:`if (index >= LINE_COUNT) {
    sampleCounts[sampleCountsOffset + index] = 0u;
    return;
  }
  let vertexBegin = getVertexBegin(index);
  let vertexEnd = getVertexEnd(index, vertexBegin);
  var count = 0u;
  if (vertexEnd > vertexBegin) {
    let distance = vertexDistances[vertexDistancesOffset + vertexEnd - 1u];
    count = 1u;
    if (distance > 0.0) {
      let spacing = params[paramsOffset + 10u];
      var segments = u32(min(ceil(distance / spacing), f32(MAXIMUM_PER_LINE)));
      if (segments > 0u && f32(segments - 1u) * spacing >= distance) {
        segments -= 1u;
      }
      if (f32(segments) * spacing < distance) {
        segments += 1u;
      }
      count = segments + 1u;
      if (count > MAXIMUM_PER_LINE) {
        count = MAXIMUM_PER_LINE;
        atomicMax(&truncated[truncatedOffset], 1u);
      }
    }
  }
  sampleCounts[sampleCountsOffset + index] = count;`})),T.push(...new m({id:`${r}-scan`,input:D,output:O}).getCommandNodes(e));let W=[{name:`linePositions`,view:i.positions,type:`f32`,access:`read`},{name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`},{name:`vertexDistances`,view:E,type:`f32`,access:`read`},{name:`sampleOffsets`,view:O,type:`u32`,access:`read`},{name:`params`,view:i.parameters,type:`f32`,access:`read`},{name:`positionsOut`,view:j,type:`f32`,access:`read_write`}];y.sampleDistances&&W.push({name:`distancesOut`,view:y.sampleDistances,type:`f32`,access:`read_write`}),y.sampleLineIds&&W.push({name:`lineIdsOut`,view:y.sampleLineIds,type:`u32`,access:`read_write`}),T.push(h(e,{id:`${r}-locate`,operation:I,variant:`locate`,bindings:W,invocationCount:l,declarations:`${B}
${x}`,body:`let total = sampleOffsets[sampleOffsetsOffset + LINE_COUNT];
  if (index >= total) {
    positionsOut[positionsOutOffset + 2u * index] = getNaN();
    positionsOut[positionsOutOffset + 2u * index + 1u] = getNaN();
    ${y.sampleDistances?`distancesOut[distancesOutOffset + index] = getNaN();`:``}
    ${y.sampleLineIds?`lineIdsOut[lineIdsOutOffset + index] = 0xffffffffu;`:``}
    return;
  }
  var low = 0u;
  var high = LINE_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (sampleOffsets[sampleOffsetsOffset + middle] <= index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let line = low - 1u;
  let vertexBegin = getVertexBegin(line);
  let vertexEnd = getVertexEnd(line, vertexBegin);
  let firstSample = sampleOffsets[sampleOffsetsOffset + line];
  let sampleCount = sampleOffsets[sampleOffsetsOffset + line + 1u] - firstSample;
  let localSample = index - firstSample;
  let lastVertex = vertexEnd - 1u;
  let length = vertexDistances[vertexDistancesOffset + lastVertex];
  var distance = length;
  var positionX = linePositions[linePositionsOffset + 2u * lastVertex];
  var positionY = linePositions[linePositionsOffset + 2u * lastVertex + 1u];
  if (localSample + 1u < sampleCount) {
    distance = min(f32(localSample) * params[paramsOffset + 10u], length);
    var searchLow = vertexBegin + 1u;
    var searchHigh = vertexEnd;
    loop {
      if (searchLow >= searchHigh) {
        break;
      }
      let middle = (searchLow + searchHigh) / 2u;
      if (vertexDistances[vertexDistancesOffset + middle] <= distance) {
        searchLow = middle + 1u;
      } else {
        searchHigh = middle;
      }
    }
    let segmentStart = searchLow - 1u;
    if (segmentStart < lastVertex) {
      let startDistance = vertexDistances[vertexDistancesOffset + segmentStart];
      let endDistance = vertexDistances[vertexDistancesOffset + segmentStart + 1u];
      let t = (distance - startDistance) / (endDistance - startDistance);
      let startX = linePositions[linePositionsOffset + 2u * segmentStart];
      let startY = linePositions[linePositionsOffset + 2u * segmentStart + 1u];
      let endX = linePositions[linePositionsOffset + 2u * segmentStart + 2u];
      let endY = linePositions[linePositionsOffset + 2u * segmentStart + 3u];
      positionX = startX + (endX - startX) * t;
      positionY = startY + (endY - startY) * t;
    }
  }
  positionsOut[positionsOutOffset + 2u * index] = positionX;
  positionsOut[positionsOutOffset + 2u * index + 1u] = positionY;
  ${y.sampleDistances?`distancesOut[distancesOutOffset + index] = distance;`:``}
  ${y.sampleLineIds?`lineIdsOut[lineIdsOutOffset + index] = line;`:``}`}));let G=[{name:`raster`,view:w.values,type:`f32`,access:`read`}];i.mask&&G.push({name:`validity`,view:i.mask,type:`u32`,access:`read`}),G.push({name:`samplePositions`,view:j,type:`f32`,access:`read`},{name:`params`,view:i.parameters,type:`f32`,access:`read`},{name:`valuesOut`,view:z,type:`f32`,access:`read_write`}),T.push(h(e,{id:`${r}-sample`,operation:I,variant:`sample`,bindings:G,invocationCount:l,declarations:n({width:i.grid.width,height:i.grid.height,noDataValue:o(i.values)?void 0:i.noDataValue,hasValidity:!!i.mask}),body:`valuesOut[valuesOutOffset + index] = sampleRaster(
    samplePositions[samplePositionsOffset + 2u * index],
    samplePositions[samplePositionsOffset + 2u * index + 1u]
  );`}));let K=A(l),te=p(e,`${r}-sample-aggregates`,`uint32`,K*8),q=!!(y.lineGains||y.lineLosses||y.lineMinimums||y.lineMaximums),J=q?p(e,`${r}-line-summaries`,`float32`,4*s):void 0,Y=`const LINE_COUNT: u32 = ${s}u;
const SAMPLE_CAPACITY: u32 = ${l}u;
${x}
${S}`,X=e=>[{name:`sampleOffsets`,view:O,type:`u32`,access:`read`},{name:`sampleValues`,view:z,type:`f32`,access:`read`},{name:`aggregates`,view:te,type:`u32`,access:e}],Z=`var itemState = getIdentityScanState();
  var line = 0u;
  let isActive = index < getEmittedSampleCount();
  if (isActive) {
    line = findSampleLine(index);
    itemState = getSampleScanState(index, line);
  }`,Q=X(`read`),ne=[[`cumulativeGainOut`,y.sampleCumulativeGain],[`cumulativeLossOut`,y.sampleCumulativeLoss],[`lineSummaries`,J]];for(let[e,t]of ne)t&&Q.push({name:e,view:t,type:`f32`,access:`read_write`});if((q||y.sampleCumulativeGain||y.sampleCumulativeLoss)&&T.push(h(e,{id:`${r}-cumulative-blocks`,operation:I,variant:`cumulative-blocks`,bindings:X(`read_write`),invocationCount:K*256,guardIndex:!1,workgroupSize:256,declarations:`${Y}
${N({writesAggregates:!0,loadsItems:!0})}`,body:P(Z)}),h(e,{id:`${r}-cumulative-carries`,operation:I,variant:`cumulative-carries`,bindings:[X(`read_write`)[2]],invocationCount:256,guardIndex:!1,workgroupSize:256,declarations:`const BLOCK_COUNT: u32 = ${K}u;
${N({writesAggregates:!0,loadsItems:!1})}`,body:F()}),h(e,{id:`${r}-cumulative`,operation:I,variant:`cumulative`,bindings:Q,invocationCount:K*256,guardIndex:!1,workgroupSize:256,declarations:`${Y}
${N({writesAggregates:!1,loadsItems:!0})}`,body:`${Z}
  let prefix = scanWorkgroup(localInvocationIndex, itemState);
  let result = combineScanStates(loadAggregate(workgroupIndex), prefix);
  if (isActive) {
    ${y.sampleCumulativeGain?`cumulativeGainOut[cumulativeGainOutOffset + index] = result.gain;`:``}
    ${y.sampleCumulativeLoss?`cumulativeLossOut[cumulativeLossOutOffset + index] = result.loss;`:``}
    ${q?`// The last emitted sample of a line holds the line totals.
    if (min(sampleOffsets[sampleOffsetsOffset + line + 1u], SAMPLE_CAPACITY) == index + 1u) {
      let summaryBase = lineSummariesOffset + 4u * line;
      lineSummaries[summaryBase] = result.gain;
      lineSummaries[summaryBase + 1u] = result.loss;
      lineSummaries[summaryBase + 2u] = select(getNaN(), result.minimum, result.hasValue != 0u);
      lineSummaries[summaryBase + 3u] = select(getNaN(), result.maximum, result.hasValue != 0u);
    }`:``}
  }`})),J){let t=[{name:`sampleOffsets`,view:O,type:`u32`,access:`read`},{name:`lineSummaries`,view:J,type:`f32`,access:`read`}];for(let[e,n]of[[`lineGainsOut`,y.lineGains],[`lineLossesOut`,y.lineLosses],[`lineMinimumsOut`,y.lineMinimums],[`lineMaximumsOut`,y.lineMaximums]])n&&t.push({name:e,view:n,type:`f32`,access:`read_write`});T.push(h(e,{id:`${r}-summaries`,operation:I,variant:`summaries`,bindings:t,invocationCount:s,declarations:`const SAMPLE_CAPACITY: u32 = ${l}u;
${x}`,body:`let firstSample = min(sampleOffsets[sampleOffsetsOffset + index], SAMPLE_CAPACITY);
  let endSample = min(sampleOffsets[sampleOffsetsOffset + index + 1u], SAMPLE_CAPACITY);
  let hasSamples = endSample > firstSample;
  let summaryBase = lineSummariesOffset + 4u * index;
  ${y.lineGains?`lineGainsOut[lineGainsOutOffset + index] = select(0.0, lineSummaries[summaryBase], hasSamples);`:``}
  ${y.lineLosses?`lineLossesOut[lineLossesOutOffset + index] = select(0.0, lineSummaries[summaryBase + 1u], hasSamples);`:``}
  ${y.lineMinimums?`lineMinimumsOut[lineMinimumsOutOffset + index] = select(getNaN(), lineSummaries[summaryBase + 2u], hasSamples);`:``}
  ${y.lineMaximums?`lineMaximumsOut[lineMaximumsOutOffset + index] = select(getNaN(), lineSummaries[summaryBase + 3u], hasSamples);`:``}`}))}if(y.lineSampleOffsets||y.lineLengths){let t=[{name:`sampleOffsets`,view:O,type:`u32`,access:`read`},{name:`lineOffsets`,view:i.lineOffsets,type:`u32`,access:`read`},{name:`vertexDistances`,view:E,type:`f32`,access:`read`}];y.lineSampleOffsets&&t.push({name:`lineSampleOffsetsOut`,view:y.lineSampleOffsets,type:`u32`,access:`read_write`}),y.lineLengths&&t.push({name:`lineLengthsOut`,view:y.lineLengths,type:`f32`,access:`read_write`}),T.push(h(e,{id:`${r}-finish-lines`,operation:I,variant:`finish-lines`,bindings:t,invocationCount:s+1,declarations:B,body:`${y.lineSampleOffsets?`lineSampleOffsetsOut[lineSampleOffsetsOutOffset + index] = min(sampleOffsets[sampleOffsetsOffset + index], SAMPLE_CAPACITY);`:``}
  if (index < LINE_COUNT) {
    let vertexBegin = getVertexBegin(index);
    let vertexEnd = getVertexEnd(index, vertexBegin);
    var length = 0.0;
    if (vertexEnd > vertexBegin) {
      length = vertexDistances[vertexDistancesOffset + vertexEnd - 1u];
    }
    ${y.lineLengths?`lineLengthsOut[lineLengthsOutOffset + index] = length;`:``}
  }`}))}let $=[{name:`sampleOffsets`,view:O,type:`u32`,access:`read`},{name:`truncated`,view:k,type:`u32`,access:`read`},{name:`countOut`,view:y.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:y.status.overflow,type:`u32`,access:`read_write`}];return y.status.requiredCount&&$.push({name:`totalOut`,view:y.status.requiredCount,type:`u32`,access:`read_write`}),T.push(h(e,{id:`${r}-finish-counts`,operation:I,variant:`finish-counts`,bindings:$,invocationCount:1,declarations:`const LINE_COUNT: u32 = ${s}u;
const SAMPLE_CAPACITY: u32 = ${l}u;`,body:`let total = sampleOffsets[sampleOffsetsOffset + LINE_COUNT];
  countOut[countOutOffset] = min(total, SAMPLE_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, total > SAMPLE_CAPACITY || truncated[truncatedOffset] != 0u);
  ${y.status.requiredCount?`totalOut[totalOutOffset] = total;`:``}`})),T}},B=[`nearest`,`bilinear`,`bicubic`],V=1500,H=[.05,.04,.5,.3],U=2.2,W=C({parameters:D,rebuildOn:[`gridWidth`],caption:`A raster and three lines in; samples along each line and their profile out.`,createInputs:({world:e})=>({world:e,anchor:[.04*e.width,.5],end:[.96*e.width,.55]}),build(t,n,r){let{grid:i,columns:a,rows:o}=O(n.world,r.gridWidth),s=t.upload(`raster`,`float32`,T(w(a,o).elevation)),c=t.allocate(`line-positions`,`float32x2`,2),l=t.upload(`line-offsets`,`uint32`,Uint32Array.of(0,2)),u=t.allocate(`parameters`,`float32`,12),d=t.allocate(`count`,`uint32`,1),f=t.allocate(`overflow`,`uint32`,1),p=t.allocate(`required-count`,`uint32`,1),m=t.allocate(`sample-positions`,`float32x2`,V),h=t.allocate(`sample-distances`,`float32`,V),g=t.allocate(`sample-values`,`float32`,V);return t.graph.add(new z({grid:i,values:s.view,positions:c.view,lineOffsets:l.view,sampleCapacity:V,parameters:u.view,output:{status:{count:d.view,overflow:f.view,requiredCount:p.view},samplePositions:m.view,sampleDistances:h.view,sampleValues:g.view}})),{outputs:{raster:s,samplePositions:m,sampleDistances:h,sampleValues:g,count:d,columns:a,rows:o},update:t=>{c.write(Float32Array.from([...n.anchor,...n.end])),u.write(e({spacing:t.spacing,method:B[Math.round(t.method)]},i))}}},interaction:{kind:`drag`,hint:`drag the profile line`,rest:[.96,.45],resolve:(e,{inputs:t,outputs:n,world:r,peek:i})=>{t.end=[e.x,e.y];let a=i(n.count),o=i(n.sampleValues),s=0;if(a&&o){let e=1/0,t=-1/0;for(let n=0;n<a[0];n++)e=Math.min(e,o[n]),t=Math.max(t,o[n]);s=t-e}let c=Math.hypot(t.end[0]-t.anchor[0],t.end[1]-t.anchor[1]);return{rerun:!0,glyphs:[{shape:`line`,from:t.anchor,to:t.end},{shape:`crosshair`,at:t.end},{shape:`box`,minimum:[r.width*H[0],r.height*H[1]],maximum:[r.width*H[2],r.height*H[3]]}],readout:`length ${E(c)} | ${a?a[0]:`-`} samples | relief ${E(s)}`}}},draw:({outputs:e})=>[k(e,e.raster,{ramp:`sequential`,range:[0,1],alpha:.8}),{layer:`dots`,positions:e.samplePositions,values:e.sampleValues,count:V,radius:4,range:[0,1],ramp:`diverging`},{layer:`chart`,kind:`scatter`,values:e.sampleValues,xValues:e.sampleDistances,xRange:[0,U],range:[0,1],color:[1,1,1,.95],radius:3,rectangle:H}]});export{W as default};