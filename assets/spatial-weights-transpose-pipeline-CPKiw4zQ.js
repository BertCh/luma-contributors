import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-reduction-tNgOVQ_C.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{t as i}from"./gpu-sort-xRm8l5yn.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as o}from"./graph-chunk-dispatch-zBt8k2-o.js";import{E as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{t as d}from"./binary-search-wgsl-CMmaMYIt.js";import{a as f}from"./sorted-segment-sums-Dvmm6OdR.js";import{i as p,s as m,t as h}from"./gpu-chunked-spatial-weights-CfTf1Nzu.js";import{c as g,t as _}from"./spatial-weights-CqWysstN.js";function v(e,t,n){let{weights:r,output:i}=n,a=m(e,r),o=h(r),c=n.columnCount??a;if(!Number.isInteger(c)||c<1)throw Error(`${e} columnCount must be a positive integer`);if(g(e,i,`output`)!==c)throw Error(`${e} output.offsets must hold columnCount + 1 entries`);let l=o.reduce((e,t)=>e+t.weights.neighbors.length,0);if(i.neighbors.length<l)throw Error(`${e} output capacity must be >= the input capacity`);if(o[0].isWhole&&o[0].weights.distances&&!i.distances)throw Error(`${e} output.distances is required with weights.distances`);if(i.asymmetricSlots){if(s(i.asymmetricSlots,[`uint32`],`${e} asymmetricSlots`),i.asymmetricSlots.length!==1)throw Error(`${e} asymmetricSlots must hold one uint32 row`);if(a!==c)throw Error(`${e} asymmetricSlots requires square weights`)}i.distances&&s(i.distances,[`float32`],`${e} output.distances`);let d=r.information?.transform??`unknown`,f=_(i,{spatialContext:r.spatialContext,information:{rowCount:c,transform:d===`row`||d===`variance`?`unknown`:d,symmetric:a===c?r.information?.symmetric??`unknown`:`unknown`,hasSelfWeights:r.information?.hasSelfWeights,provenance:{...r.information?.provenance,source:t,criterion:`transpose`}},status:r.status});return u(e,[i.offsets,i.neighbors,i.weights,i.distances,i.asymmetricSlots],[typeof n.symmetryTolerance==`object`?n.symmetryTolerance:void 0,...p(r),...Object.values(r.status??{}),...o.flatMap(e=>Object.values(e.weights.status??{}))]),{rowCount:a,columnCount:c,outputWeights:f}}function y(s,u,m,g,_,v,y){let{weights:x,output:S}=g,{rowCount:C,columnCount:w}=_,T=y?Object.values(y.status):[],E=[...p(x),S.offsets,S.neighbors,S.weights,S.distances,S.asymmetricSlots,...T,typeof g.symmetryTolerance==`object`?g.symmetryTolerance:void 0];for(let e of E)e&&c(s.device,u,`weights or output binding`,t(e).size,`use smaller input or output row chunks`);S.asymmetricSlots&&c(s.device,u,`symmetry rows`,C*4,`use chunked output for a chunked symmetry reduction`),l(u,s,[...p(x),S.offsets,S.neighbors,S.weights,S.distances,S.asymmetricSlots,...T,typeof g.symmetryTolerance==`object`?g.symmetryTolerance:void 0]);let D=h(x),O=!!S.distances&&D.every(e=>e.weights.distances),k=`0xffffffffu`,A=e(s,`${u}-column-counts`,`uint32`,w),j=e(s,`${u}-column-starts`,`uint32`,w),M=e(s,`${u}-column-fill`,`uint32`,w),N=`const COLUMNS: u32 = ${w}u;
const COLUMN_START: u32 = ${y?.rowStart??0}u;
const OUTPUT_CAPACITY: u32 = ${S.neighbors.length}u;`,P=[a(s,{id:`${u}-clear-columns`,operation:m,variant:`clear-columns`,bindings:[{name:`columnCounts`,view:A,type:`u32`,access:`read_write`},{name:`columnFill`,view:M,type:`u32`,access:`read_write`}],invocationCount:w,body:`columnCounts[columnCountsOffset + index] = 0u;
  columnFill[columnFillOffset + index] = 0u;`}),a(s,{id:`${u}-clear-output`,operation:m,variant:O?`clear-output-distances`:`clear-output`,bindings:[{name:`outputNeighbors`,view:S.neighbors,type:`u32`,access:`read_write`},{name:`outputWeights`,view:S.weights,type:`f32`,access:`read_write`},...O?[{name:`outputDistances`,view:S.distances,type:`f32`,access:`read_write`}]:[]],invocationCount:S.neighbors.length,body:`outputNeighbors[outputNeighborsOffset + index] = 0u;
  outputWeights[outputWeightsOffset + index] = 0.0;
  ${O?`outputDistances[outputDistancesOffset + index] = 0.0;`:``}`})],F=D.map(t=>{let n=`${u}${t.idSuffix}`,r=t.weights,o=r.neighbors.length,c=e(s,`${n}-slot-keys`,`uint32`,o),l=e(s,`${n}-slot-indices`,`uint32`,o),d=e(s,`${n}-sorted-keys`,`uint32`,o),p=e(s,`${n}-sorted-slots`,`uint32`,o),h=`${N}
const CHUNK_ROWS: u32 = ${t.rowCount}u;`;return P.push(a(s,{id:`${n}-slot-keys`,operation:m,variant:`slot-keys`,bindings:[{name:`offsets`,view:r.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:r.neighbors,type:`u32`,access:`read`},{name:`slotKeys`,view:c,type:`u32`,access:`read_write`},{name:`slotIndices`,view:l,type:`u32`,access:`read_write`}],invocationCount:o,declarations:h,body:`let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + CHUNK_ROWS] && neighbor >= COLUMN_START && neighbor - COLUMN_START < COLUMNS;
  slotKeys[slotKeysOffset + index] = select(COLUMNS, neighbor - COLUMN_START, used);
  slotIndices[slotIndicesOffset + index] = index;`}),...new i({id:`${n}-sort`,keys:c,values:l,outputKeys:d,outputValues:p,keyBits:f(w)}).getCommandNodes(s),b(s,m,n,`count`,d,o,A)),{dispatch:t,chunkId:n,capacity:o,sortedKeys:d,sortedSlots:p}});if(P.push(...new r({id:`${u}-scan`,input:A,output:j,mode:`exclusive`}).getCommandNodes(s),a(s,{id:`${u}-offsets`,operation:m,variant:`offsets`,bindings:[{name:`columnStarts`,view:j,type:`u32`,access:`read`},{name:`columnCounts`,view:A,type:`u32`,access:`read`},{name:`outputOffsets`,view:S.offsets,type:`u32`,access:`read_write`}],invocationCount:w+1,declarations:N,body:`if (index < COLUMNS) {
    outputOffsets[outputOffsetsOffset + index] = min(columnStarts[columnStartsOffset + index], OUTPUT_CAPACITY);
  } else {
    outputOffsets[outputOffsetsOffset + COLUMNS] =
      min(columnStarts[columnStartsOffset + COLUMNS - 1u] + columnCounts[columnCountsOffset + COLUMNS - 1u], OUTPUT_CAPACITY);
  }`})),y){let e=Object.entries(y.status).filter(([e])=>e===`count`||e===`requiredCount`||e===`overflow`);P.push(a(s,{id:`${u}-status`,operation:m,variant:`status`,bindings:[{name:`columnStarts`,view:j,type:`u32`,access:`read`},{name:`columnCounts`,view:A,type:`u32`,access:`read`},...e.map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],invocationCount:1,declarations:N,body:`let required = columnStarts[columnStartsOffset + COLUMNS - 1u] + columnCounts[columnCountsOffset + COLUMNS - 1u];
  ${y.status.count?`count[countOffset] = min(required, OUTPUT_CAPACITY);`:``}
  ${y.status.requiredCount?`requiredCount[requiredCountOffset] = required;`:``}
  ${y.status.overflow?`overflow[overflowOffset] = u32(required > OUTPUT_CAPACITY);`:``}`}))}for(let{dispatch:t,chunkId:n,capacity:r,sortedKeys:i,sortedSlots:o}of F){let c=t.weights,l=e(s,`${n}-destinations`,`uint32`,r);P.push(a(s,{id:`${n}-destinations`,operation:m,variant:`destinations`,bindings:[{name:`sortedKeys`,view:i,type:`u32`,access:`read`},{name:`columnStarts`,view:j,type:`u32`,access:`read`},{name:`columnFill`,view:M,type:`u32`,access:`read`},{name:`destinations`,view:l,type:`u32`,access:`read_write`}],invocationCount:r,declarations:`${N}
${d(`sortedKeys`)}`,body:`let key = sortedKeys[sortedKeysOffset + index];
  var destination = ${k};
  if (key < COLUMNS) {
    let low = lowerBoundSortedKeys(0u, index, key);
    let candidate = columnStarts[columnStartsOffset + key] + columnFill[columnFillOffset + key] + (index - low);
    if (candidate < OUTPUT_CAPACITY) {
      destination = candidate;
    }
  }
  destinations[destinationsOffset + index] = destination;`}),a(s,{id:`${n}-scatter`,operation:m,variant:O?`scatter-distances`:`scatter`,bindings:[{name:`offsets`,view:c.offsets,type:`u32`,access:`read`},{name:`weights`,view:c.weights,type:`f32`,access:`read`},{name:`sortedSlots`,view:o,type:`u32`,access:`read`},{name:`destinations`,view:l,type:`u32`,access:`read`},{name:`outputNeighbors`,view:S.neighbors,type:`u32`,access:`read_write`},{name:`outputWeights`,view:S.weights,type:`f32`,access:`read_write`},...O?[{name:`distances`,view:c.distances,type:`f32`,access:`read`},{name:`outputDistances`,view:S.distances,type:`f32`,access:`read_write`}]:[]],invocationCount:r,declarations:`const CHUNK_ROWS: u32 = ${t.rowCount}u;
const CHUNK_ROW_START: u32 = ${t.rowStart}u;
${d(`offsets`)}
fn findRow(slot: u32) -> u32 {
  return upperBoundOffsets(0u, CHUNK_ROWS + 1u, slot) - 1u;
}`,body:`let destination = destinations[destinationsOffset + index];
  if (destination != ${k}) {
    let slot = sortedSlots[sortedSlotsOffset + index];
    outputNeighbors[outputNeighborsOffset + destination] = findRow(slot) + CHUNK_ROW_START;
    outputWeights[outputWeightsOffset + destination] = weights[weightsOffset + slot];
    ${O?`outputDistances[outputDistancesOffset + destination] = distances[distancesOffset + slot];`:``}
  }`}),b(s,m,n,`fill`,i,r,M))}if(S.asymmetricSlots){let t=o(s,`${u}-symmetry-parameters`,v);P.push(...t.nodes);let r=e(s,`${u}-row-mismatches`,`uint32`,C);for(let e of D){let n=e.weights;P.push(a(s,{id:`${u}-symmetry-rows${e.idSuffix}`,operation:m,variant:`symmetry-rows`,bindings:[{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`neighbors`,view:n.neighbors,type:`u32`,access:`read`},{name:`weights`,view:n.weights,type:`f32`,access:`read`},{name:`transposeOffsets`,view:S.offsets,type:`u32`,access:`read`},{name:`transposeNeighbors`,view:S.neighbors,type:`u32`,access:`read`},{name:`transposeWeights`,view:S.weights,type:`f32`,access:`read`},{name:`rowMismatches`,view:r,type:`u32`,access:`read_write`},{name:`tolerance`,view:t.parameters,type:`f32`,access:`read`}],invocationCount:e.rowCount,declarations:`${N}
const CHUNK_ROW_START: u32 = ${e.rowStart}u;
const NONE: u32 = 0xffffffffu;`,body:`let row = index + CHUNK_ROW_START;
  var first = offsets[offsetsOffset + index];
  let firstEnd = offsets[offsetsOffset + index + 1u];
  var second = transposeOffsets[transposeOffsetsOffset + row];
  let secondEnd = transposeOffsets[transposeOffsetsOffset + row + 1u];
  var mismatches = 0u;
  loop {
    while (first < firstEnd && neighbors[neighborsOffset + first] >= COLUMNS) {
      first++;
    }
    if (first >= firstEnd && second >= secondEnd) {
      break;
    }
    let firstColumn = select(NONE, neighbors[neighborsOffset + first], first < firstEnd);
    let secondColumn = select(
      NONE,
      transposeNeighbors[transposeNeighborsOffset + second],
      second < secondEnd
    );
    if (firstColumn == secondColumn) {
      if (abs(weights[weightsOffset + first] - transposeWeights[transposeWeightsOffset + second]) > tolerance[toleranceOffset]) {
        mismatches++;
      }
      first++;
      second++;
    } else if (firstColumn < secondColumn) {
      mismatches++;
      first++;
    } else {
      mismatches++;
      second++;
    }
  }
  rowMismatches[rowMismatchesOffset + row] = mismatches;`}))}P.push(...new n({id:`${u}-symmetry-total`,input:r,output:S.asymmetricSlots,operation:`sum`}).getCommandNodes(s))}return P}function b(e,t,n,r,i,o,s){return a(e,{id:`${n}-${r}-columns`,operation:t,variant:`column-counts`,bindings:[{name:`sortedKeys`,view:i,type:`u32`,access:`read`},{name:`columnTarget`,view:s,type:`u32`,access:`read_write`}],invocationCount:s.length,declarations:d(`sortedKeys`),body:`let low = lowerBoundSortedKeys(0u, ${o}u, index);
  let upper = upperBoundSortedKeys(low, ${o}u, index);
  columnTarget[columnTargetOffset + index] += upper - low;`})}export{v as n,y as t};