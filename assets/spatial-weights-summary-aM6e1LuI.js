import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./statistics-kernel-node-lacKNERm.js";import{s as r}from"./graph-chunk-dispatch-zBt8k2-o.js";import{n as i}from"./spatial-weights-parameters-DDN52fQU.js";import{E as a,f as o,g as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{t as d}from"./slot-grouping-CsHvMvbq.js";import{r as f}from"./spatial-weights-CqWysstN.js";import{i as p,s as m,t as h}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{t as g}from"./define-run-B-brkCe4.js";import{f as _,r as v}from"./inputs-CJKKIRKK.js";import{o as y}from"./text-DYSgVV-n.js";import{n as b,t as x}from"./pick-Bo3t3hjv.js";import{vn as S}from"./entry-preview-D70TdSbK.js";import{t as C}from"./labels-BymUYkjU.js";import{C as w,m as T,n as E,r as D,v as O}from"./weights-BRH93xop.js";var k=`GPUSpatialWeightsSummary`,A=4096,j=256,M={statistics:{s0:0,s1:1,s2:2,length:3},counts:{slots:0,asymmetricSlots:1,isolates:2,minimumCardinality:3,maximumCardinality:4,emptyRows:5,length:6}},N=class{id;props;toleranceValues;constructor(e){let t=e.id??`spatial-weights-summary`;this.id=t,this.props=e;let n=m(t,e.weights);if(a(e.output.statistics,[`float32`],`${t} statistics`),e.output.statistics.length<M.statistics.length)throw Error(`${t} statistics must hold at least 3 values`);if(a(e.output.counts,[`uint32`],`${t} counts`),e.output.counts.length<M.counts.length)throw Error(`${t} counts must hold at least 6 values`);if(e.output.cardinality&&(a(e.output.cardinality,[`uint32`],`${t} cardinality`),e.output.cardinality.length!==n))throw Error(`${t} cardinality length must equal the weights row count`);let r=e.symmetryTolerance??0;typeof r==`number`?(s(t,`symmetryTolerance`,r,0),this.toleranceValues=i({symmetryTolerance:r})):(a(r,[`float32`],`${t} symmetryTolerance`),o(t,`symmetryTolerance.length`,r.length,1)),u(t,[e.output.statistics,e.output.counts,e.output.cardinality],[...p(e.weights),typeof e.symmetryTolerance==`object`?e.symmetryTolerance:void 0])}getCommandNodes(i){let{id:a,props:o}=this,{statistics:s,counts:u}=o.output,m=h(o.weights),g=!m[0].isWhole,_=m[0].weights;l(a,i,[...p(o.weights),s,u,o.output.cardinality,typeof o.symmetryTolerance==`object`?o.symmetryTolerance:void 0]),typeof o.symmetryTolerance==`object`&&c(i.device,a,`symmetryTolerance binding`,t(o.symmetryTolerance).size,`use a smaller packed symmetryTolerance view`);let v=m.reduce((e,t)=>e+t.rowCount,0),y=_.neighbors.length;c(i.device,a,`symmetry state`,(v+1)*4,`use fewer weights rows`);let b=r(i,`${a}-parameters`,this.toleranceValues??o.symmetryTolerance),x=b.parameters,S=(t,n,r)=>e(i,`${a}-${t}`,n,r),C=S(`column-sums`,`float32`,v),w=S(`row-sums`,`float32`,v),T=S(`s1-partials`,`float32`,v),E=S(`s2-partials`,`float32`,v),D=S(`symmetry-state`,`uint32`,v+1),O=i.createDataView(D.buffer,{format:`uint32`,length:v,byteOffset:4}),M=S(`isolates`,`uint32`,v),N=o.output.cardinality??S(`cardinality`,`uint32`,v),P=S(`chunk-slot-counts`,`uint32`,m.length),F=`const ROWS: u32 = ${v}u;`,I=g?this.getChunkedRowNodes(i,{dispatches:m,rows:v,tolerance:x,symmetryState:D,view:S,columnSums:C,rowSums:w,s1Partials:T,s2Partials:E,asymmetric:O,isolates:M,cardinality:N,chunkSlotCounts:P}):(()=>{let e=S(`slot-keys`,`uint32`,y),t=S(`slot-indices`,`uint32`,y),r=S(`column-offsets`,`uint32`,v+1),o=d(i,{id:`${a}-columns`,operation:k,slotKeys:e,slotIndices:t,columns:v,columnOffsets:r});return[n(i,{id:`${a}-slot-keys`,operation:k,variant:`slot-keys`,bindings:[{name:`offsets`,view:_.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:_.neighbors,type:`u32`,access:`read`},{name:`slotKeys`,view:e,type:`u32`,access:`read_write`},{name:`slotIndices`,view:t,type:`u32`,access:`read_write`}],invocationCount:y,declarations:F,body:`let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + ROWS] && neighbor < ROWS;
  slotKeys[slotKeysOffset + index] = select(ROWS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`}),...o.nodes,n(i,{id:`${a}-column-sums`,operation:k,variant:`column-sums`,bindings:[{name:`columnOffsets`,view:r,type:`u32`,access:`read`},{name:`sortedSlots`,view:o.sortedSlots,type:`u32`,access:`read`},{name:`weights`,view:_.weights,type:`f32`,access:`read`},{name:`columnSums`,view:C,type:`f32`,access:`read_write`}],invocationCount:v,body:`var sum = 0.0;
  for (var position = columnOffsets[columnOffsetsOffset + index];
      position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) {
    sum += weights[weightsOffset + sortedSlots[sortedSlotsOffset + position]];
  }
  columnSums[columnSumsOffset + index] = sum;`}),n(i,{id:`${a}-rows`,operation:k,variant:`rows`,bindings:[{name:`offsets`,view:_.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:_.neighbors,type:`u32`,access:`read`},{name:`weights`,view:_.weights,type:`f32`,access:`read`},{name:`rowSums`,view:w,type:`f32`,access:`read_write`},{name:`s1Partials`,view:T,type:`f32`,access:`read_write`},{name:`asymmetric`,view:O,type:`u32`,access:`read_write`},{name:`cardinality`,view:N,type:`u32`,access:`read_write`},{name:`tolerance`,view:x,type:`f32`,access:`read`}],invocationCount:v,declarations:`${F}
const CAPACITY: u32 = ${y}u;
const MISSING: f32 = -1.0;

// Weight w_row,column by binary search of the ascending neighbor IDs of row, or MISSING.
fn findWeight(row: u32, column: u32) -> f32 {
  var low = offsets[offsetsOffset + row];
  let end = offsets[offsetsOffset + row + 1u];
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
    return weights[weightsOffset + low];
  }
  return MISSING;
}`,body:`var sum = 0.0;
  var s1 = 0.0;
  var unmatched = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    sum += weight;
    var reverse = MISSING;
    if (neighbor < ROWS) {
      reverse = findWeight(neighbor, index);
    }
    if (reverse == MISSING) {
      s1 += 2.0 * weight * weight;
      unmatched++;
    } else {
      s1 += (weight + reverse) * (weight + reverse);
      if (abs(weight - reverse) > tolerance[toleranceOffset]) {
        unmatched++;
      }
    }
  }
  // Cardinality counts active slots only (the shared neighbor definition).
  var activeCount = 0u;
  ${f(`true`,`activeCount++;`)}
  cardinality[cardinalityOffset + index] = activeCount;
  rowSums[rowSumsOffset + index] = sum;
  s1Partials[s1PartialsOffset + index] = s1;
  asymmetric[asymmetricOffset + index] = unmatched;`}),n(i,{id:`${a}-rows-final`,operation:k,variant:`rows-final`,bindings:[{name:`offsets`,view:_.offsets,type:`u32`,access:`read`},{name:`rowSums`,view:w,type:`f32`,access:`read`},{name:`columnSums`,view:C,type:`f32`,access:`read`},{name:`s2Partials`,view:E,type:`f32`,access:`read_write`},{name:`cardinality`,view:N,type:`u32`,access:`read`},{name:`isolates`,view:M,type:`u32`,access:`read_write`}],invocationCount:v,body:`let count = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];
  let total = rowSums[rowSumsOffset + index] + columnSums[columnSumsOffset + index];
  s2Partials[s2PartialsOffset + index] = total * total;
  // 0: has an active slot, 1: isolate that lists slots, 2: empty row (no slot at all).
  let isolate = cardinality[cardinalityOffset + index] == 0u;
  isolates[isolatesOffset + index] = select(0u, select(1u, 2u, count == 0u), isolate);`})]})(),L=Math.ceil(v/A),R=S(`float-partials`,`float32`,3*L),z=S(`count-partials`,`uint32`,5*L),B=[...b.nodes,...I];return B.push(n(i,{id:`${a}-blocks`,operation:k,variant:`blocks`,bindings:[{name:`rowSums`,view:w,type:`f32`,access:`read`},{name:`s1Partials`,view:T,type:`f32`,access:`read`},{name:`s2Partials`,view:E,type:`f32`,access:`read`},{name:`asymmetric`,view:O,type:`u32`,access:`read`},{name:`isolates`,view:M,type:`u32`,access:`read`},{name:`cardinality`,view:N,type:`u32`,access:`read`},{name:`floatPartials`,view:R,type:`f32`,access:`read_write`},{name:`countPartials`,view:z,type:`u32`,access:`read_write`}],workgroupSize:j,invocationCount:L*j,guardIndex:!1,declarations:`${F}
const BLOCK_ROWS: u32 = ${A}u;
const BLOCK_COUNT: u32 = ${L}u;
const WORKGROUP_SIZE: u32 = ${j}u;
var<workgroup> partialS0: array<f32, ${j}>;
var<workgroup> partialS1: array<f32, ${j}>;
var<workgroup> partialS2: array<f32, ${j}>;
var<workgroup> partialAsymmetric: array<u32, ${j}>;
var<workgroup> partialIsolates: array<u32, ${j}>;
var<workgroup> partialEmpty: array<u32, ${j}>;
var<workgroup> partialMinimum: array<u32, ${j}>;
var<workgroup> partialMaximum: array<u32, ${j}>;`,body:`let block = index / WORKGROUP_SIZE;
  let local = localInvocationIndex;
  let isInRange = index < INVOCATION_COUNT;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var emptyCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROWS);
    for (var row = block * BLOCK_ROWS + local; row < end; row += WORKGROUP_SIZE) {
      s0 += rowSums[rowSumsOffset + row];
      s1 += s1Partials[s1PartialsOffset + row];
      s2 += s2Partials[s2PartialsOffset + row];
      asymmetricCount += asymmetric[asymmetricOffset + row];
      let isolateCode = isolates[isolatesOffset + row];
      isolateCount += select(0u, 1u, isolateCode != 0u);
      emptyCount += select(0u, 1u, isolateCode == 2u);
      let degree = cardinality[cardinalityOffset + row];
      minimum = min(minimum, degree);
      maximum = max(maximum, degree);
    }
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialEmpty[local] = emptyCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialEmpty[local] += partialEmpty[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (isInRange && local == 0u) {
    floatPartials[floatPartialsOffset + block] = partialS0[0];
    floatPartials[floatPartialsOffset + BLOCK_COUNT + block] = partialS1[0];
    floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block] = partialS2[0];
    countPartials[countPartialsOffset + block] = partialAsymmetric[0];
    countPartials[countPartialsOffset + BLOCK_COUNT + block] = partialIsolates[0];
    countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block] = partialMinimum[0];
    countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block] = partialMaximum[0];
    countPartials[countPartialsOffset + 4u * BLOCK_COUNT + block] = partialEmpty[0];
  }`}),n(i,{id:`${a}-totals`,operation:k,variant:`totals`,bindings:[g?{name:`chunkSlotCounts`,view:P,type:`u32`,access:`read`}:{name:`offsets`,view:_.offsets,type:`u32`,access:`read`},{name:`floatPartials`,view:R,type:`f32`,access:`read`},{name:`countPartials`,view:z,type:`u32`,access:`read`},{name:`statistics`,view:s,type:`f32`,access:`read_write`},{name:`counts`,view:u,type:`u32`,access:`read_write`}],workgroupSize:j,invocationCount:j,guardIndex:!1,declarations:`${F}
const BLOCK_COUNT: u32 = ${L}u;
const WORKGROUP_SIZE: u32 = ${j}u;
var<workgroup> partialS0: array<f32, ${j}>;
var<workgroup> partialS1: array<f32, ${j}>;
var<workgroup> partialS2: array<f32, ${j}>;
var<workgroup> partialAsymmetric: array<u32, ${j}>;
var<workgroup> partialIsolates: array<u32, ${j}>;
var<workgroup> partialEmpty: array<u32, ${j}>;
var<workgroup> partialMinimum: array<u32, ${j}>;
var<workgroup> partialMaximum: array<u32, ${j}>;`,body:`let local = localInvocationIndex;
  var s0 = 0.0;
  var s1 = 0.0;
  var s2 = 0.0;
  var asymmetricCount = 0u;
  var isolateCount = 0u;
  var emptyCount = 0u;
  var minimum = 0xffffffffu;
  var maximum = 0u;
  for (var block = local; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    s0 += floatPartials[floatPartialsOffset + block];
    s1 += floatPartials[floatPartialsOffset + BLOCK_COUNT + block];
    s2 += floatPartials[floatPartialsOffset + 2u * BLOCK_COUNT + block];
    asymmetricCount += countPartials[countPartialsOffset + block];
    isolateCount += countPartials[countPartialsOffset + BLOCK_COUNT + block];
    minimum = min(minimum, countPartials[countPartialsOffset + 2u * BLOCK_COUNT + block]);
    maximum = max(maximum, countPartials[countPartialsOffset + 3u * BLOCK_COUNT + block]);
    emptyCount += countPartials[countPartialsOffset + 4u * BLOCK_COUNT + block];
  }
  partialS0[local] = s0;
  partialS1[local] = s1;
  partialS2[local] = s2;
  partialAsymmetric[local] = asymmetricCount;
  partialIsolates[local] = isolateCount;
  partialEmpty[local] = emptyCount;
  partialMinimum[local] = minimum;
  partialMaximum[local] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      partialS0[local] += partialS0[local + stride];
      partialS1[local] += partialS1[local + stride];
      partialS2[local] += partialS2[local + stride];
      partialAsymmetric[local] += partialAsymmetric[local + stride];
      partialIsolates[local] += partialIsolates[local + stride];
      partialEmpty[local] += partialEmpty[local + stride];
      partialMinimum[local] = min(partialMinimum[local], partialMinimum[local + stride]);
      partialMaximum[local] = max(partialMaximum[local], partialMaximum[local + stride]);
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    statistics[statisticsOffset] = partialS0[0];
    statistics[statisticsOffset + 1u] = 0.5 * partialS1[0];
    statistics[statisticsOffset + 2u] = partialS2[0];
    ${g?`var slotTotal = 0u;
    for (var chunk = 0u; chunk < ${m.length}u; chunk++) {
      slotTotal += chunkSlotCounts[chunkSlotCountsOffset + chunk];
    }
    counts[countsOffset] = slotTotal;`:`counts[countsOffset] = offsets[offsetsOffset + ROWS];`}
    counts[countsOffset + 1u] = partialAsymmetric[0];
    counts[countsOffset + 2u] = partialIsolates[0];
    counts[countsOffset + 3u] = partialMinimum[0];
    counts[countsOffset + 4u] = partialMaximum[0];
    counts[countsOffset + 5u] = partialEmpty[0];
  }`})),B}getChunkedRowNodes(e,t){let{id:r}=this,{dispatches:i,rows:a,tolerance:o,symmetryState:s,view:c}=t,{columnSums:l,rowSums:u,s1Partials:p,s2Partials:m}=t,{asymmetric:h,isolates:g,cardinality:_,chunkSlotCounts:v}=t,y=[n(e,{id:`${r}-stage-tolerance`,operation:k,variant:`stage-tolerance`,invocationCount:1,bindings:[{name:`tolerance`,view:o,type:`f32`,access:`read`},{name:`symmetryState`,view:s,type:`u32`,access:`read_write`}],body:`symmetryState[symmetryStateOffset] = bitcast<u32>(tolerance[toleranceOffset]);`})],b=[];for(let t of i){let i=t.weights,o=i.neighbors.length,s=c(`slot-keys${t.idSuffix}`,`uint32`,o),x=c(`slot-indices${t.idSuffix}`,`uint32`,o),S=c(`column-offsets${t.idSuffix}`,`uint32`,a+1),C=d(e,{id:`${r}-columns${t.idSuffix}`,operation:k,slotKeys:s,slotIndices:x,columns:a,columnOffsets:S}),w=`const ROWS: u32 = ${a}u;
const CHUNK_ROWS: u32 = ${t.rowCount}u;
const CHUNK_INDEX: u32 = ${t.chunkIndex}u;
${t.constantsWGSL}`;b.push(n(e,{id:`${r}-rows-final${t.idSuffix}`,operation:k,variant:`rows-final-chunk`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`rowSums`,view:u,type:`f32`,access:`read`},{name:`columnSums`,view:l,type:`f32`,access:`read`},{name:`s2Partials`,view:m,type:`f32`,access:`read_write`},{name:`cardinality`,view:_,type:`u32`,access:`read`},{name:`isolates`,view:g,type:`u32`,access:`read_write`}],invocationCount:t.rowCount,declarations:w,body:`let globalRow = index + WEIGHTS_ROW_START;
  let count = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];
  let total = rowSums[rowSumsOffset + globalRow] + columnSums[columnSumsOffset + globalRow];
  s2Partials[s2PartialsOffset + globalRow] = total * total;
  let isolate = cardinality[cardinalityOffset + globalRow] == 0u;
  isolates[isolatesOffset + globalRow] = select(0u, select(1u, 2u, count == 0u), isolate);`})),y.push(n(e,{id:`${r}-slot-keys${t.idSuffix}`,operation:k,variant:`slot-keys`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`slotKeys`,view:s,type:`u32`,access:`read_write`},{name:`slotIndices`,view:x,type:`u32`,access:`read_write`}],invocationCount:o,declarations:w,body:`let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + CHUNK_ROWS] && neighbor < ROWS;
  slotKeys[slotKeysOffset + index] = select(ROWS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`}),...C.nodes,n(e,{id:`${r}-column-sums${t.idSuffix}`,operation:k,variant:`column-sums-chunk`,bindings:[{name:`columnOffsets`,view:S,type:`u32`,access:`read`},{name:`sortedSlots`,view:C.sortedSlots,type:`u32`,access:`read`},{name:`weights`,view:i.weights,type:`f32`,access:`read`},{name:`columnSums`,view:l,type:`f32`,access:`read_write`}],invocationCount:a,declarations:`const CHUNK_INDEX: u32 = ${t.chunkIndex}u;`,body:`var sum = 0.0;
  if (CHUNK_INDEX > 0u) {
    sum = columnSums[columnSumsOffset + index];
  }
  for (var position = columnOffsets[columnOffsetsOffset + index];
      position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) {
    sum += weights[weightsOffset + sortedSlots[sortedSlotsOffset + position]];
  }
  columnSums[columnSumsOffset + index] = sum;`}),n(e,{id:`${r}-rows${t.idSuffix}`,operation:k,variant:`rows-chunk`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:i.neighbors,type:`u32`,access:`read`},{name:`weights`,view:i.weights,type:`f32`,access:`read`},{name:`rowSums`,view:u,type:`f32`,access:`read_write`},{name:`s1Partials`,view:p,type:`f32`,access:`read_write`},{name:`asymmetric`,view:h,type:`u32`,access:`read_write`},{name:`cardinality`,view:_,type:`u32`,access:`read_write`}],invocationCount:t.rowCount,declarations:`${w}
const CAPACITY: u32 = ${o}u;`,body:`let globalRow = index + WEIGHTS_ROW_START;
  var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    sum += weights[weightsOffset + slot];
  }
  var activeCount = 0u;
  ${f(`true`,`activeCount++;`,`globalRow`,`WEIGHTS_ROW_START`)}
  cardinality[cardinalityOffset + globalRow] = activeCount;
  rowSums[rowSumsOffset + globalRow] = sum;
  s1Partials[s1PartialsOffset + globalRow] = 0.0;
  asymmetric[asymmetricOffset + globalRow] = 0u;`}),n(e,{id:`${r}-slot-count${t.idSuffix}`,operation:k,variant:`slot-count-chunk`,bindings:[{name:`offsets`,view:i.offsets,type:`u32`,access:`read`},{name:`chunkSlotCounts`,view:v,type:`u32`,access:`read_write`}],invocationCount:1,declarations:w,body:`chunkSlotCounts[chunkSlotCountsOffset + CHUNK_INDEX] = offsets[offsetsOffset + CHUNK_ROWS];`}))}y.push(...b);for(let t of i)for(let o of i){let c=o.chunkIndex===i.length-1,l=t.weights,u=o.weights;y.push(n(e,{id:`${r}-reverse-${t.chunkIndex}-${o.chunkIndex}`,operation:k,variant:`reverse-chunk`,bindings:[{name:`offsets`,view:l.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:l.neighbors,type:`u32`,access:`read`},{name:`weights`,view:l.weights,type:`f32`,access:`read`},{name:`targetOffsets`,view:u.offsets,type:`u32`,access:`read`},{name:`targetNeighbors`,view:u.neighbors,type:`u32`,access:`read`},{name:`targetWeights`,view:u.weights,type:`f32`,access:`read`},{name:`s1Partials`,view:p,type:`f32`,access:`read_write`},{name:`symmetryState`,view:s,type:`u32`,access:`read_write`}],invocationCount:t.rowCount,declarations:`const ROWS: u32 = ${a}u;
const ROW_START: u32 = ${t.rowStart}u;
const TARGET_START: u32 = ${o.rowStart}u;
const TARGET_END: u32 = ${o.rowStart+o.rowCount}u;
const IS_LAST_TARGET: bool = ${c};
const MISSING: f32 = -1.0;

fn findWeight(row: u32, column: u32) -> f32 {
  var low = targetOffsets[targetOffsetsOffset + row];
  let end = targetOffsets[targetOffsetsOffset + row + 1u];
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
    return targetWeights[targetWeightsOffset + low];
  }
  return MISSING;
}`,body:`let globalRow = index + ROW_START;
  var s1 = s1Partials[s1PartialsOffset + globalRow];
  var unmatched = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < TARGET_START || (neighbor >= TARGET_END && !IS_LAST_TARGET)) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    var reverse = MISSING;
    if (neighbor < ROWS) {
      reverse = findWeight(neighbor - TARGET_START, globalRow);
    }
    if (reverse == MISSING) {
      s1 += 2.0 * weight * weight;
      unmatched++;
    } else {
      s1 += (weight + reverse) * (weight + reverse);
      if (abs(weight - reverse) > bitcast<f32>(symmetryState[symmetryStateOffset])) {
        unmatched++;
      }
    }
  }
  s1Partials[s1PartialsOffset + globalRow] = s1;
  symmetryState[symmetryStateOffset + 1u + globalRow] += unmatched;`}))}return y}},P=[.64,.14,.96,.86],F=[`slots`,`asymmetric`,`isolates`,`min degree`,`max degree`,`empty rows`];function I(e,t){let n=Math.sin(e*12.9898)*43758.5453;return 1+(n-Math.floor(n)-.5)*t}var L=g({parameters:S,caption:`Cardinality per point (left) and the summary counts (right, log(1 + count)).`,createInputs:({random:e,world:t})=>{let n={width:t.width*.55,height:t.height},r=O(_(e,n,200,.1),n);return{...r,csr:v(r.positions,.09),index:b(r.positions),focus:{row:-1}}},build(e,t,n){let r=e.upload(`positions`,`float32x2`,t.positions),{csr:i}=t,a=w(e,`band`,i),o=e.allocate(`statistics`,`float32`,M.statistics.length),s=e.allocate(`counts`,`uint32`,M.counts.length),c=e.allocate(`cardinality`,`uint32`,t.count);return e.graph.add(new N({weights:a.input,symmetryTolerance:n.symmetryTolerance,output:{statistics:o.view,counts:s.view,cardinality:c.view}})),{outputs:{positions:r,counts:s,loggedCounts:D(e,`summary`,s),cardinality:c,links:E(e,`band`,a,r)},update:e=>a.weights.write(Float32Array.from({length:a.capacity},(t,n)=>I(n,e.noise)))}},rebuildOn:[`symmetryTolerance`],async afterRun({outputs:e}){let t=await e.counts.buffer.readAsync();return new Uint32Array(t.slice().buffer)},interaction:{kind:`probe`,hint:`hover a point`,rest:[.25,.5],resolve(e,{inputs:t,values:n}){let r=x(t.index,e.x,e.y,t.focus.row);if(t.focus.row=r,r<0)return{};let{csr:i}=t,a=[t.positions[r*2],t.positions[r*2+1]],o=[{shape:`ring`,at:a,radius:.012}],s=0;for(let e=i.offsets[r];e<i.offsets[r+1];e++){let c=i.neighbors[e],l=i.offsets[c];for(;l<i.offsets[c+1]&&i.neighbors[l]!==r;)l++;let u=Math.abs(I(e,n.noise)-I(l,n.noise))>n.symmetryTolerance;u&&s++;let[d]=T(t.positions,[c]);o.push({shape:`line`,from:a,to:d,dashed:!u},{shape:`dot`,at:d})}return{focus:r,glyphs:o,readout:`cardinality ${i.offsets[r+1]-i.offsets[r]} · ${s} asymmetric at tolerance ${n.symmetryTolerance}`}}},draw:({outputs:e,extra:t,world:n})=>[{layer:`lines`,context:!0,segments:e.links},{layer:`dots`,positions:e.positions,values:e.cardinality,ramp:`sequential`,radius:3.6},{layer:`chart`,kind:`bars`,values:e.loggedCounts,color:[.55,.65,.95,.9],rectangle:P},{layer:`text`,...C(y(n,P),F.map((e,n)=>t?`${e} ${t[n]}`:e))}]});export{L as default};