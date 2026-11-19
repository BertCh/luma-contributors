import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-scan-B4NNTfF9.js";import{t as r}from"./gpu-sort-xRm8l5yn.js";import{t as i}from"./gpu-uint32-gather-C7ioZBaz.js";import{o as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as o}from"./gpu-contributor-utils-BIwLQON5.js";import{t as s}from"./cell-keys-Dl1tIQ_Z.js";import{o as c}from"./cell-validation-D7P5BmPo.js";import{n as l,r as u}from"./cell-key-wgsl-Cdgu1snt.js";import{t as d}from"./construction-snapshot-DdkX9Pc0.js";function f(e){return d(e,[`count`,`overflow`,`candidateOverflow`,`requiredCount`])}function p(e){let t=d(e,[`cells`,`counts`,`sums`,`sumValues`,`minimums`,`maximums`,`status`]);return Object.freeze({...t,status:f(t.status)})}function m(e){return[e.cells,e.counts,e.sums,e.sumValues,e.minimums,e.maximums,e.status.count,e.status.overflow,e.status.candidateOverflow,e.status.requiredCount]}function h(e){return e.cells.byteOffset/e.cells.rowByteLength}function g(e,n,r){o(r.cells,[`uint32x2`],`${e} ${n}.cells`);let i=r.cells.length;if(i<1)throw Error(`${e} ${n}.cells must hold at least one row`);t(r.counts,`${e} ${n}.counts`),r.sums&&o(r.sums,[`uint32x2`],`${e} ${n}.sums`);for(let t of[`sumValues`,`minimums`,`maximums`]){let i=r[t];i&&o(i,[`float32`],`${e} ${n}.${t}`)}for(let t of[`counts`,`sums`,`sumValues`,`minimums`,`maximums`]){let a=r[t];if(a&&a.length!==i)throw Error(`${e} ${n}.${t} must have the same length as ${n}.cells`)}for(let i of[`count`,`overflow`,`candidateOverflow`,`requiredCount`]){let a=r.status[i];if(a&&(t(a,`${e} ${n}.status.${i}`),a.length<1))throw Error(`${e} ${n}.status.${i} must contain one uint32 row`)}}var _=256;function v(e){return e.width+1>32}function y(e,t,n){return v(e)?`(${t} >> (CELL_WIDTH - 32u)) != 0u`:`(${n} >> CELL_WIDTH) != 0u`}function b(t,o){let{id:d,operation:f,layout:p,output:m,source:h}=o,g=o.keyLow.length,b=m.cells.length,x=v(p),S=(n,r)=>e(t,`${d}-${n}`,`uint32`,r),C=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read`}),w=(e,t,n=`u32`)=>({name:e,view:t,type:n,access:`read_write`}),T=[],E=e=>{let n=c(t,e,{ROW_COUNT:g,CAPACITY:b,CELL_WIDTH:p.width,CELL_LOW_BIT:p.lowBit,CELL_HEADER:p.headerHigh,CELL_RESOLUTION:p.resolution});return T.push(...n.uploads),n.node},D=o.keyLow,O=o.keyHigh,k;if(!o.sorted){if(!o.rowIds)throw Error(`${d} needs row IDs to sort`);let e=x?32:p.width+1,n=S(`sorted-low-first`,g),a=S(`sorted-rows-first`,g);if(T.push(...new r({id:`${d}-sort-low`,keys:o.keyLow,values:o.rowIds,outputKeys:n,outputValues:a,keyBits:e}).getCommandNodes(t)),x&&o.keyHigh){let e=S(`high-by-low-order`,g);T.push(...new i({id:`${d}-gather-high`,source:o.keyHigh,indices:a,output:e}).getCommandNodes(t)),O=S(`sorted-high`,g),k=S(`permutation`,g),T.push(...new r({id:`${d}-sort-high`,keys:e,values:a,outputKeys:O,outputValues:k,keyBits:p.width+1-32}).getCommandNodes(t)),D=S(`sorted-low`,g),T.push(...new i({id:`${d}-gather-low`,source:o.keyLow,indices:k,output:D}).getCommandNodes(t))}else D=n,k=a}let A=()=>[C(`sortedLow`,D),...x&&O?[C(`sortedHigh`,O)]:[]],j=e=>x?`vec2u(sortedHigh[sortedHighOffset + ${e}], sortedLow[sortedLowOffset + ${e}])`:`vec2u(0u, sortedLow[sortedLowOffset + ${e}])`,M=S(`heads`,g),N=S(`valid-count`,1);T.push(E({id:`${d}-heads`,operation:f,variant:`heads`,bindings:[...A(),w(`heads`,M),w(`validCount`,N)],invocationCount:g,declarations:`fn isInvalidKey(key: vec2u) -> bool {
  return ${y(p,`key.x`,`key.y`)};
}`,body:`let key = ${j(`index`)};
  let isValid = !isInvalidKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${j(`index - 1u`)});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isInvalidKey(${j(`index + 1u`)}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`}));let P=S(`cell-indices`,g);T.push(...new n({id:`${d}-scan`,input:M,output:P,mode:`exclusive`}).getCommandNodes(t));let F=S(`offsets`,b+1),I=S(`cell-total`,1),L=S(`row-cells`,g);T.push(E({id:`${d}-offsets`,operation:f,variant:`offsets`,bindings:[C(`heads`,M),C(`cellIndices`,P),C(`validCount`,N),w(`offsets`,F),w(`cellTotal`,I),w(`rowCells`,L)],invocationCount:g,body:`let isHead = heads[headsOffset + index] != 0u;
  let cell = cellIndices[cellIndicesOffset + index] + select(0u, 1u, isHead) - 1u;
  let isValid = index < validCount[validCountOffset];
  rowCells[rowCellsOffset + index] = select(0xffffffffu, cell, isValid && cell < CAPACITY);
  if (isHead && cell <= CAPACITY) {
    offsets[offsetsOffset + cell] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let total = cellIndices[cellIndicesOffset + last] + heads[headsOffset + last];
    cellTotal[cellTotalOffset] = total;
    if (total <= CAPACITY) {
      offsets[offsetsOffset + total] = validCount[validCountOffset];
    }
  }`}));let R=`fn getCellCount() -> u32 {
  return min(cellTotal[cellTotalOffset], CAPACITY);
}`,z=h.kind===`cells`?h.table:void 0;T.push(E({id:`${d}-cells`,operation:f,variant:`cells`,bindings:[C(`offsets`,F),C(`cellTotal`,I),...A(),w(`cellsOut`,m.cells),w(`countsOut`,m.counts)],invocationCount:b,declarations:`${R}
${s}`,body:`if (index >= getCellCount()) {
    cellsOut[cellsOutOffset + 2u * index] = 0xffffffffu;
    cellsOut[cellsOutOffset + 2u * index + 1u] = 0xffffffffu;
    countsOut[countsOutOffset + index] = 0u;
    return;
  }
  let begin = offsets[offsetsOffset + index];
  let key = cellGetKey(${j(`begin`)}, CELL_HEADER, CELL_RESOLUTION, CELL_LOW_BIT);
  cellsOut[cellsOutOffset + 2u * index] = key.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = key.x;
  // Roll-ups sum the child counts in the accumulation pass.
  countsOut[countsOutOffset + index] = ${z?`0u`:`offsets[offsetsOffset + index + 1u] - begin`};`}));let B=!!(m.sums||m.sumValues),V=!!(m.minimums||m.maximums),H=h.kind===`rows`?h.values:void 0;if((B||V)&&h.kind===`rows`&&!H)throw Error(`${d} needs values to compute sums and extremes`);if(B&&z&&!z.sums)throw Error(`${d} source table needs sums to roll up sums`);if(z&&(m.minimums&&!z.minimums||m.maximums&&!z.maximums))throw Error(`${d} source table needs the extremes it rolls up`);let U=B?m.sums??e(t,`${d}-sums-scratch`,`uint32x2`,b):void 0,W=m.minimums?S(`minimum-keys`,b):void 0,G=m.maximums?S(`maximum-keys`,b):void 0,K=k?`permutation[permutationOffset + index]`:`index`,q=()=>H?[...k?[C(`permutation`,k)]:[],C(`values`,H,`f32`)]:[];if((U||W||G)&&T.push(E({id:`${d}-clear`,operation:f,variant:`clear`,bindings:[...U?[w(`sums`,U)]:[],...W?[w(`minimumKeys`,W)]:[],...G?[w(`maximumKeys`,G)]:[]],invocationCount:b,body:`${U?`sums[sumsOffset + 2u * index] = 0u;
  sums[sumsOffset + 2u * index + 1u] = 0u;`:``}
  ${W?`minimumKeys[minimumKeysOffset + index] = 0xffffffffu;`:``}
  ${G?`maximumKeys[maximumKeysOffset + index] = 0u;`:``}`})),U||z){let e=H?`cellScaleValue(values[valuesOffset + ${K}])`:`vec2u(childSums[childSumsOffset + 2u * index + 1u], childSums[childSumsOffset + 2u * index])`;T.push(E({id:`${d}-accumulate-sums`,operation:f,variant:`accumulate-sums`,bindings:[C(`rowCells`,L),...q(),...z&&U?[C(`childSums`,z.sums)]:[],...z?[C(`childCounts`,z.counts),{name:`countsOut`,view:m.counts,type:`atomic<u32>`,access:`read_write`}]:[],...U?[{name:`sums`,view:U,type:`atomic<u32>`,access:`read_write`}]:[]],workgroupSize:_,invocationCount:g,guardIndex:!1,declarations:`${u(o.sumScale)}
const TILE_SIZE: u32 = ${_}u;
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> tileCells: array<u32, ${_}>;
${U?`var<workgroup> tileHigh: array<u32, ${_}>;\nvar<workgroup> tileLow: array<u32, ${_}>;`:``}
${z?`var<workgroup> tileCounts: array<u32, ${_}>;`:``}`,body:`let lane = localInvocationIndex;
  var cell = NO_CELL;
  ${U?`var value = vec2u(0u, 0u);`:``}
  ${z?`var childCount = 0u;`:``}
  if (index < INVOCATION_COUNT) {
    cell = rowCells[rowCellsOffset + index];
    if (cell != NO_CELL) {
      ${U?`value = ${e};`:``}
      ${z?`childCount = childCounts[childCountsOffset + index];`:``}
    }
  }
  tileCells[lane] = cell;
  ${U?`tileHigh[lane] = value.x;
  tileLow[lane] = value.y;`:``}
  ${z?`tileCounts[lane] = childCount;`:``}
  workgroupBarrier();
  for (var stride = 1u; stride < TILE_SIZE; stride = stride << 1u) {
    ${U?`var addValue = vec2u(0u, 0u);`:``}
    ${z?`var addCount = 0u;`:``}
    if (lane >= stride && tileCells[lane - stride] == cell) {
      ${U?`addValue = vec2u(tileHigh[lane - stride], tileLow[lane - stride]);`:``}
      ${z?`addCount = tileCounts[lane - stride];`:``}
    }
    workgroupBarrier();
    ${U?`value = cellAddI64(value, addValue);
    tileHigh[lane] = value.x;
    tileLow[lane] = value.y;`:``}
    ${z?`childCount += addCount;
    tileCounts[lane] = childCount;`:``}
    workgroupBarrier();
  }
  let isRunEnd = lane == TILE_SIZE - 1u || tileCells[lane + 1u] != cell;
  if (cell == NO_CELL || !isRunEnd) {
    return;
  }
  ${z?`atomicAdd(&countsOut[countsOutOffset + cell], childCount);`:``}
  ${U?`let previous = atomicAdd(&sums[sumsOffset + 2u * cell], value.y);
  let high = value.x + select(0u, 1u, previous + value.y < previous);
  if (high != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * cell + 1u], high);
  }`:``}`}))}if(W||G){let e=H?`values[valuesOffset + ${K}]`:`childMinimums[childMinimumsOffset + index]`,t=H?`values[valuesOffset + ${K}]`:`childMaximums[childMaximumsOffset + index]`;T.push(E({id:`${d}-accumulate-extremes`,operation:f,variant:`accumulate-extremes`,bindings:[C(`rowCells`,L),...q(),...z&&W?[C(`childMinimums`,z.minimums,`f32`)]:[],...z&&G?[C(`childMaximums`,z.maximums,`f32`)]:[],...W?[{name:`minimumKeys`,view:W,type:`atomic<u32>`,access:`read_write`}]:[],...G?[{name:`maximumKeys`,view:G,type:`atomic<u32>`,access:`read_write`}]:[]],workgroupSize:_,invocationCount:g,guardIndex:!1,declarations:`${l}
const TILE_SIZE: u32 = ${_}u;
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> tileCells: array<u32, ${_}>;
${W?`var<workgroup> tileMinimums: array<u32, ${_}>;`:``}
${G?`var<workgroup> tileMaximums: array<u32, ${_}>;`:``}`,body:`let lane = localInvocationIndex;
  var cell = NO_CELL;
  ${W?`var minimumKey = 0xffffffffu;`:``}
  ${G?`var maximumKey = 0u;`:``}
  if (index < INVOCATION_COUNT) {
    cell = rowCells[rowCellsOffset + index];
    if (cell != NO_CELL) {
      ${W?`minimumKey = getOrderedKey(${e});`:``}
      ${G?`maximumKey = getOrderedKey(${t});`:``}
    }
  }
  tileCells[lane] = cell;
  ${W?`tileMinimums[lane] = minimumKey;`:``}
  ${G?`tileMaximums[lane] = maximumKey;`:``}
  workgroupBarrier();
  for (var stride = 1u; stride < TILE_SIZE; stride = stride << 1u) {
    ${W?`var otherMinimum = 0xffffffffu;`:``}
    ${G?`var otherMaximum = 0u;`:``}
    if (lane >= stride && tileCells[lane - stride] == cell) {
      ${W?`otherMinimum = tileMinimums[lane - stride];`:``}
      ${G?`otherMaximum = tileMaximums[lane - stride];`:``}
    }
    workgroupBarrier();
    ${W?`minimumKey = min(minimumKey, otherMinimum);
    tileMinimums[lane] = minimumKey;`:``}
    ${G?`maximumKey = max(maximumKey, otherMaximum);
    tileMaximums[lane] = maximumKey;`:``}
    workgroupBarrier();
  }
  let isRunEnd = lane == TILE_SIZE - 1u || tileCells[lane + 1u] != cell;
  if (cell == NO_CELL || !isRunEnd) {
    return;
  }
  ${W?`atomicMin(&minimumKeys[minimumKeysOffset + cell], minimumKey);`:``}
  ${G?`atomicMax(&maximumKeys[maximumKeysOffset + cell], maximumKey);`:``}`}))}(m.sumValues||W||G)&&T.push(E({id:`${d}-finish`,operation:f,variant:`finish`,bindings:[C(`cellTotal`,I),...m.sumValues&&U?[C(`sums`,U)]:[],...W?[C(`minimumKeys`,W)]:[],...G?[C(`maximumKeys`,G)]:[],...m.sumValues?[w(`sumValuesOut`,m.sumValues,`f32`)]:[],...m.minimums&&W?[w(`minimumsOut`,m.minimums,`f32`)]:[],...m.maximums&&G?[w(`maximumsOut`,m.maximums,`f32`)]:[]],invocationCount:b,declarations:`${R}
${l}
${u(o.sumScale)}`,body:`let isOccupied = index < getCellCount();
  let nan = getNaN();
  ${m.sumValues?`let total = vec2u(sums[sumsOffset + 2u * index + 1u], sums[sumsOffset + 2u * index]);
  sumValuesOut[sumValuesOutOffset + index] = cellI64ToF32(total) / SUM_SCALE;`:``}
  ${m.minimums&&W?`minimumsOut[minimumsOutOffset + index] = select(nan, decodeOrderedKey(minimumKeys[minimumKeysOffset + index]), isOccupied);`:``}
  ${m.maximums&&G?`maximumsOut[maximumsOutOffset + index] = select(nan, decodeOrderedKey(maximumKeys[maximumKeysOffset + index]), isOccupied);`:``}`}));let J=o.candidateSources??[],Y=o.extraCounts??[];return T.push(E({id:`${d}-publish`,operation:f,variant:`publish`,invocationCount:1,bindings:[C(`cellTotal`,I),w(`countOut`,m.status.count),w(`overflowOut`,m.status.overflow),...m.status.requiredCount?[w(`totalOut`,m.status.requiredCount)]:[],...Y.map((e,t)=>w(`extraOut${t}`,e))],body:`let total = cellTotal[cellTotalOffset];
  let count = min(total, CAPACITY);
  countOut[countOutOffset] = count;
  overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
  ${m.status.requiredCount?`totalOut[totalOutOffset] = total;`:``}
  ${Y.map((e,t)=>`extraOut${t}[extraOut${t}Offset] = count;`).join(`
  `)}`})),m.status.candidateOverflow&&T.push(a(t,{id:`${d}-candidate-status`,operation:f,variant:`candidate-status`,invocationCount:1,bindings:[...J.map((e,t)=>C(`candidateIn${t}`,e)),w(`candidateOverflowOut`,m.status.candidateOverflow)],body:`candidateOverflowOut[candidateOverflowOutOffset] = select(0u, 1u, false${J.map((e,t)=>` || candidateIn${t}[candidateIn${t}Offset] != 0u`).join(``)});`})),T}export{p as a,v as i,b as n,f as o,m as r,g as s,h as t};