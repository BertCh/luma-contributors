import{o as e,s as t}from"./cell-index-wgsl-Cw7syfdA.js";import{n}from"./scan-total-C4QddMmi.js";import{r}from"./kernel-C3tfQOn4.js";import{t as i}from"./gpu-scan-B4NNTfF9.js";import{t as a}from"./gpu-sort-xRm8l5yn.js";import{o,t as s}from"./wgsl-kernel-nodes-DagJklH2.js";import{E as c,T as l,f as u,v as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{r as m,t as h}from"./cell-keys-Dl1tIQ_Z.js";import{a as g}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as _}from"./cell-dispatch-validation-PIkQyAK_.js";import{i as v,n as y}from"./cell-table-OQfP6eNV.js";import{t as b}from"./gpu-cell-topology-BbtEmE_O.js";import{t as x}from"./define-run-B-brkCe4.js";import{a as S}from"./point-observations-CeqP-hIo.js";import{n as C,o as w,r as T,s as E}from"./cells-Di9l-3nc.js";import{a as D,o as O}from"./kernels-IDOsx_br.js";import{Cl as k}from"./entry-preview-D70TdSbK.js";function A(e,t){let{id:n,layout:s}=t,c=t.keyLow.length,l=t=>r(e,`${n}-${t}`,`uint32`,c),u=[],d=t.keyLow,f=t.keyHigh;if(!t.sorted){let r=l(`sorted-low`),i=l(`paired-high`);d=l(`ordered-low`),f=l(`ordered-high`),u.push(...new a({id:`${n}-sort-low`,keys:t.keyLow,values:t.keyHigh,outputKeys:r,outputValues:i}).getCommandNodes(e),...new a({id:`${n}-sort-high`,keys:i,values:r,outputKeys:f,outputValues:d}).getCommandNodes(e))}let p=l(`flags`),m=l(`positions`);return u.push(o(e,{id:`${n}-boundaries`,operation:`GPUCellCompaction`,variant:`s2-boundaries`,invocationCount:c,bindings:[{name:`keyLow`,view:d,type:`u32`,access:`read`},{name:`keyHigh`,view:f,type:`u32`,access:`read`},{name:`flags`,view:p,type:`u32`,access:`read_write`}],declarations:h,body:`let key = vec2u(keyHigh[keyHighOffset + index], keyLow[keyLowOffset + index]);
  let valid = all(cellShiftRight(key, ${s.width}u) == vec2u(0u));
  var boundary = index == 0u;
  if (index > 0u) {
    boundary = any(key != vec2u(keyHigh[keyHighOffset + index - 1u], keyLow[keyLowOffset + index - 1u]));
  }
  flags[flagsOffset + index] = select(0u, 1u, valid && boundary);`}),...new i({id:`${n}-scan`,input:p,output:m,mode:`exclusive`}).getCommandNodes(e),o(e,{id:`${n}-emit`,operation:`GPUCellCompaction`,variant:`s2-unique`,invocationCount:c,bindings:[{name:`keyLow`,view:d,type:`u32`,access:`read`},{name:`keyHigh`,view:f,type:`u32`,access:`read`},{name:`flags`,view:p,type:`u32`,access:`read`},{name:`positions`,view:m,type:`u32`,access:`read`},{name:`cells`,view:t.cells,type:`u32`,access:`read_write`},{name:`count`,view:t.count,type:`u32`,access:`read_write`}],declarations:h,body:`if (index == 0u) {
    count[countOffset] = positions[positionsOffset + ${c-1}u] + flags[flagsOffset + ${c-1}u];
  }
  if (flags[flagsOffset + index] != 0u) {
    let compact = vec2u(keyHigh[keyHighOffset + index], keyLow[keyLowOffset + index]);
    let key = cellShiftLeft(compact, ${s.lowBit}u) | cellShiftLeft(vec2u(0u, 1u), ${s.lowBit-1}u);
    let slot = positions[positionsOffset + index];
    cells[cellsOffset + 2u * slot] = key.y;
    cells[cellsOffset + 2u * slot + 1u] = key.x;
  }`})),u}var j=`GPUCellCompaction`,M=2147483647,N=15,P=11;function F(){let e=[1],t=[1];for(let n=1;n<=P;n++)e.push(e[n-1]*7),t.push(t[n-1]*7-1);return{hexagon:e,pentagon:t}}function I(e,n){let r=F(),i=(e,t)=>`const ${e} = array<u32, ${t.length}>(${t.map(e=>`${e}u`).join(`, `)});`;return`
${e===`quadbin`?``:t.source}
${h}
const BITS_PER_LEVEL: u32 = ${e===`h3`?3:2}u;
const HEADER_HIGH: u32 = ${e===`s2`?0:m(e,0).headerHigh}u;
${e===`h3`?i(`H3_HEXAGON_SIZES`,r.hexagon):``}
${e===`h3`?i(`H3_PENTAGON_SIZES`,r.pentagon):``}

fn getLowBit(resolution: u32) -> u32 {
  return ${e===`s2`?`61u - 2u * resolution`:e===`quadbin`?`52u - 2u * resolution`:`45u - 3u * resolution`};
}

fn getInputKey(words: vec2u) -> vec2u {
  return ${n===`high-low`?`words`:`words.yx`};
}

fn getCanonicalKey(compact: vec2u, resolution: u32) -> vec2u {
  ${e===`s2`?`return cellShiftLeft(compact, getLowBit(resolution)) | cellShiftLeft(vec2u(0u, 1u), getLowBit(resolution) - 1u);`:`return cellGetKey(compact, HEADER_HIGH, resolution, getLowBit(resolution));`}
}

fn getKeyResolution(key: vec2u) -> u32 {
  return ${e===`s2`?`dggs_s2_get_level(key)`:e===`quadbin`?`(key.x >> 20u) & 0x1fu`:`dggs_h3_get_resolution(key)`};
}

fn isValidKey(key: vec2u) -> bool {
  return ${e===`s2`?`dggs_s2_is_valid_cell_id(key)`:e===`quadbin`?`cellIsValidQuadbin(key)`:`dggs_h3_is_valid_cell_id(key)`};
}

/** Number of descendants of a cell with this compact key, \`depth\` levels down; saturates. */
fn getDescendantCount(compact: vec2u, resolution: u32, depth: u32) -> u32 {
  ${e===`h3`?`if (depth > ${P}u) {
    return 0xffffffffu;
  }
  let pathBits = 3u * resolution;
  let baseCell = cellShiftRight(compact, pathBits).y;
  let isPentagon = dggs_h3_is_base_cell_pentagon(baseCell) &&
    all((compact & cellMaskLow(pathBits)) == vec2u(0u));
  return select(H3_HEXAGON_SIZES[depth], H3_PENTAGON_SIZES[depth], isPentagon);`:`return select(0xffffffffu, 1u << (2u * depth), depth < 16u);`}
}
`}var L=class{id;props;layout;constructor(t){this.id=t.id??`cell-compaction`,this.props=t;let{id:n}=this,{operation:r,output:i}=t;if(l(n,[`quadbin`,`h3`,`s2`].includes(t.family),`family must be 'quadbin', 'h3' or 's2'`),e(n,t.family,r.resolution),this.layout=t.family===`s2`?{family:`s2`,resolution:r.resolution,lowBit:61-2*r.resolution,width:3+2*r.resolution,bitsPerLevel:2,headerHigh:0}:m(t.family,r.resolution),r.type===`compact`){let e=r.minimumResolution??0;u(n,`minimumResolution`,e,0,r.resolution)}else if(r.type===`uncompact`){let e=t.family===`h3`?P:N,i=r.maximumDepth??8;u(n,`maximumDepth`,i,0,e)}else throw Error(`${n} operation.type must be 'compact' or 'uncompact'`);if(t.wordOrder&&t.wordOrder!==`little-endian`&&t.wordOrder!==`high-low`)throw Error(`${n} wordOrder must be 'little-endian' or 'high-low'`);if(c(t.cells,[`uint32x2`],`${n} cells`),u(n,`cells.length`,t.cells.length,1,M),t.mask&&(c(t.mask,[`uint32`],`${n} mask`),t.mask.length!==t.cells.length))throw Error(`${n} mask must have the same length as cells`);for(let[e,r]of[[`count`,t.count],[`output.status.count`,i.status.count],[`output.status.overflow`,i.status.overflow],[`output.status.requiredCount`,i.status.requiredCount],[`output.status.droppedCount`,i.status.droppedCount]])if(r&&(c(r,[`uint32`],`${n} ${e}`),r.length<1))throw Error(`${n} ${e} must contain one uint32 row`);c(i.cells,[`uint32x2`],`${n} output.cells`),u(n,`output.cells.length`,i.cells.length,1,M),p(n,[i.cells,i.status.count,i.status.overflow,i.status.requiredCount,i.status.droppedCount],[t.cells,t.mask,t.count]),_(n,{cells:i.cells,count:i.status.count,overflow:i.status.overflow,requiredCount:i.status.requiredCount,droppedCount:i.status.droppedCount},[{cells:i.cells,count:i.status.count,overflow:i.status.overflow,requiredCount:i.status.requiredCount}])}getCommandNodes(e){let{props:t,id:n}=this;return g(e,n,{cells:t.cells,mask:t.mask,inputCount:t.count,outputCells:t.output.cells,...t.output.status}),d(e.device,n,`cell scratch`,t.cells.length*8,`reduce cells.length or use a device with larger bindings`),u(n,`cell scratch byteLength`,t.cells.length*8,0,e.device.limits.maxBufferSize),t.operation.type===`compact`&&t.family!==`s2`&&(u(n,`cell table parameter byteLength`,128,0,e.device.limits.maxBufferSize),d(e.device,n,`cell table parameters`,128,`use a device with larger bindings`)),f(this.id,e,[t.cells,t.mask,t.count,t.output.cells,t.output.status.count,t.output.status.overflow,t.output.status.requiredCount,t.output.status.droppedCount]),t.operation.type===`compact`?this.getCompactNodes(e,t.operation):this.getUncompactNodes(e,t.operation)}getInputBindings(){let{props:e}=this,t=[{name:`cells`,view:e.cells,type:`u32`,access:`read`}];return e.mask&&t.push({name:`mask`,view:e.mask,type:`u32`,access:`read`}),e.count&&t.push({name:`rowCount`,view:e.count,type:`u32`,access:`read`}),{bindings:t,rowCountWGSL:e.count?`rowCount[rowCountOffset]`:`${e.cells.length}u`,maskWGSL:e.mask?`mask[maskOffset + index] != 0u`:`true`}}getCompactNodes(e,t){let{id:n,props:c,layout:l}=this,{output:u}=c,d=c.cells.length,f=t.minimumResolution??0,p=t.resolution-f,m=t.sorted??!1,h=l.family===`s2`||v(l),g=(t,i=d)=>r(e,`${n}-${t}`,`uint32`,i),_=[],b=this.getInputBindings(),x=I(c.family,c.wordOrder??`little-endian`);u.status.droppedCount&&_.push(s(e,{id:`${n}-clear-dropped`,operation:j,view:u.status.droppedCount,type:`u32`,value:`0u`,componentCount:1}));let S=g(`key-low`),C=h?g(`key-high`):void 0,w=m||c.family===`s2`?void 0:g(`row-ids`);_.push(o(e,{id:`${n}-keys`,operation:j,variant:`compact-keys`,bindings:[...b.bindings,{name:`keyLow`,view:S,type:`u32`,access:`read_write`},...C?[{name:`keyHigh`,view:C,type:`u32`,access:`read_write`}]:[],...w?[{name:`rowIds`,view:w,type:`u32`,access:`read_write`}]:[],...u.status.droppedCount?[{name:`droppedCount`,view:u.status.droppedCount,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:d,declarations:`const RESOLUTION: u32 = ${t.resolution}u;
${x}`,body:`var compact = cellShiftLeft(vec2u(0u, 1u), ${l.width}u);
  if (index < ${b.rowCountWGSL}) {
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]));
    if (${b.maskWGSL} && isValidKey(key) && getKeyResolution(key) == RESOLUTION) {
      compact = cellGetCompactKey(key, ${l.lowBit}u, ${l.width}u);
    } else {
      ${u.status.droppedCount?`atomicAdd(&droppedCount[droppedCountOffset], 1u);`:``}
    }
  }
  keyLow[keyLowOffset + index] = compact.y;
  ${C?`keyHigh[keyHighOffset + index] = compact.x;`:``}
  ${w?`rowIds[rowIdsOffset + index] = index;`:``}`}));let T=r(e,`${n}-unique-table`,`uint32x2`,d),E=g(`unique-count`,1);_.push(...l.family===`s2`?A(e,{id:`${n}-table`,layout:l,keyLow:S,keyHigh:C,sorted:m,cells:T,count:E}):y(e,{id:`${n}-table`,operation:j,layout:l,keyLow:S,keyHigh:C,rowIds:w,sorted:m,source:{kind:`rows`},sumScale:65536,output:{cells:T,counts:g(`unique-counts`),status:{count:E,overflow:g(`unique-overflow`,1)}}}));let D=g(`flags`),O=g(`depths`);_.push(o(e,{id:`${n}-analyze`,operation:j,variant:`analyze`,bindings:[{name:`cells`,view:T,type:`u32`,access:`read`},{name:`uniqueCount`,view:E,type:`u32`,access:`read`},{name:`flags`,view:D,type:`u32`,access:`read_write`},{name:`depths`,view:O,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const RESOLUTION: u32 = ${t.resolution}u;
const MAXIMUM_DEPTH: u32 = ${p}u;
${x}

fn getCompactAt(row: u32) -> vec2u {
  return cellGetCompactKey(
    vec2u(cells[cellsOffset + 2u * row + 1u], cells[cellsOffset + 2u * row]), ${l.lowBit}u, ${l.width}u);
}

fn getAncestor(compact: vec2u, depth: u32) -> vec2u {
  return cellShiftRight(compact, BITS_PER_LEVEL * depth);
}

fn isLess(a: vec2u, b: vec2u) -> bool {
  return a.x < b.x || (a.x == b.x && a.y < b.y);
}

/** First row whose ancestor is >= (or, for upper bounds, > ) \`ancestor\`. */
fn findBound(ancestor: vec2u, depth: u32, rowCount: u32, isUpper: bool) -> u32 {
  var low = 0u;
  var high = rowCount;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) >> 1u;
    let value = getAncestor(getCompactAt(middle), depth);
    let goRight = select(isLess(value, ancestor), !isLess(ancestor, value), isUpper);
    if (goRight) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,body:`let rowCount = uniqueCount[uniqueCountOffset];
  if (index >= rowCount) {
    flags[flagsOffset + index] = 0u;
    depths[depthsOffset + index] = 0u;
    return;
  }
  let compact = getCompactAt(index);
  var bestDepth = 0u;
  var bestStart = index;
  // Completeness is monotone: a complete ancestor implies complete descendants.
  for (var depth = 1u; depth <= MAXIMUM_DEPTH; depth++) {
    let ancestor = getAncestor(compact, depth);
    let lower = findBound(ancestor, depth, rowCount, false);
    let upper = findBound(ancestor, depth, rowCount, true);
    if (upper - lower != getDescendantCount(ancestor, RESOLUTION - depth, depth)) {
      break;
    }
    bestDepth = depth;
    bestStart = lower;
  }
  flags[flagsOffset + index] = select(0u, 1u, bestStart == index);
  depths[depthsOffset + index] = bestDepth;`}));let k=g(`positions`);_.push(...new i({id:`${n}-scan`,input:D,output:k,mode:`exclusive`}).getCommandNodes(e));let M=r(e,`${n}-emitted`,`uint32x2`,d),N=g(`sort-keys`),P=g(`sort-ids`),F=g(`total`,1);_.push(o(e,{id:`${n}-scatter`,operation:j,variant:`scatter`,bindings:[{name:`cells`,view:T,type:`u32`,access:`read`},{name:`flags`,view:D,type:`u32`,access:`read`},{name:`depths`,view:O,type:`u32`,access:`read`},{name:`positions`,view:k,type:`u32`,access:`read`},{name:`emitted`,view:M,type:`u32`,access:`read_write`},{name:`sortKeys`,view:N,type:`u32`,access:`read_write`},{name:`sortIds`,view:P,type:`u32`,access:`read_write`},{name:`total`,view:F,type:`u32`,access:`read_write`}],invocationCount:d,declarations:`const RESOLUTION: u32 = ${t.resolution}u;
const ROW_COUNT: u32 = ${d}u;
${x}`,body:`let emitCount = positions[positionsOffset + ROW_COUNT - 1u] + flags[flagsOffset + ROW_COUNT - 1u];
  if (index == 0u) {
    total[totalOffset] = emitCount;
  }
  if (index >= emitCount) {
    // Padding rows sort last (resolutions are below 31).
    sortKeys[sortKeysOffset + index] = 31u;
    sortIds[sortIdsOffset + index] = index;
  }
  if (flags[flagsOffset + index] != 0u) {
    let depth = depths[depthsOffset + index];
    let compact = cellGetCompactKey(
      vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]), ${l.lowBit}u, ${l.width}u);
    let resolution = RESOLUTION - depth;
    let key = getCanonicalKey(cellShiftRight(compact, BITS_PER_LEVEL * depth), resolution);
    let slot = positions[positionsOffset + index];
    emitted[emittedOffset + 2u * slot] = key.y;
    emitted[emittedOffset + 2u * slot + 1u] = key.x;
    sortKeys[sortKeysOffset + slot] = resolution;
    sortIds[sortIdsOffset + slot] = slot;
  }`}));let L=c.family===`s2`?P:g(`ordered-ids`);return c.family!==`s2`&&_.push(...new a({id:`${n}-sort-resolution`,keys:N,values:P,outputKeys:g(`ordered-keys`),outputValues:L,keyBits:5}).getCommandNodes(e)),_.push(o(e,{id:`${n}-emit`,operation:j,variant:`emit`,bindings:[{name:`total`,view:F,type:`u32`,access:`read`},{name:`orderedIds`,view:L,type:`u32`,access:`read`},{name:`emitted`,view:M,type:`u32`,access:`read`},{name:`cellsOut`,view:u.cells,type:`u32`,access:`read_write`},{name:`countOut`,view:u.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:u.status.overflow,type:`u32`,access:`read_write`},...u.status.requiredCount?[{name:`totalOut`,view:u.status.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:u.cells.length,declarations:`const CAPACITY: u32 = ${u.cells.length}u;`,body:`let totalCells = total[totalOffset];
  let count = min(totalCells, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, totalCells > CAPACITY);
    ${u.status.requiredCount?`totalOut[totalOutOffset] = totalCells;`:``}
  }
  var low = 0u;
  var high = 0u;
  if (index < count) {
    let slot = orderedIds[orderedIdsOffset + index];
    low = emitted[emittedOffset + 2u * slot];
    high = emitted[emittedOffset + 2u * slot + 1u];
  }
  cellsOut[cellsOutOffset + 2u * index] = low;
  cellsOut[cellsOutOffset + 2u * index + 1u] = high;`})),_}getUncompactNodes(e,t){let{id:i,props:a}=this,{output:c}=a,l=a.cells.length,u=t.maximumDepth??8,d=(t,n=l)=>r(e,`${i}-${t}`,`uint32`,n),f=[],p=this.getInputBindings(),m=I(a.family,a.wordOrder??`little-endian`),h=`const RESOLUTION: u32 = ${t.resolution}u;
const MAXIMUM_DEPTH: u32 = ${u}u;
${m}

fn getCompact(key: vec2u, resolution: u32) -> vec2u {
  let lowBit = getLowBit(resolution);
  return cellGetCompactKey(key, lowBit, ${a.family===`s2`?64:52}u - lowBit);
}`;c.status.droppedCount&&f.push(s(e,{id:`${i}-clear-dropped`,operation:j,view:c.status.droppedCount,type:`u32`,value:`0u`,componentCount:1}));let g=d(`counts`);f.push(o(e,{id:`${i}-count`,operation:j,variant:`uncompact-count`,bindings:[...p.bindings,{name:`counts`,view:g,type:`u32`,access:`read_write`},...c.status.droppedCount?[{name:`droppedCount`,view:c.status.droppedCount,type:`atomic<u32>`,access:`read_write`}]:[]],invocationCount:l,declarations:h,body:`var descendants = 0u;
  if (index < ${p.rowCountWGSL}) {
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]));
    var keep = false;
    if (${p.maskWGSL} && isValidKey(key)) {
      let resolution = getKeyResolution(key);
      if (resolution <= RESOLUTION && RESOLUTION - resolution <= MAXIMUM_DEPTH) {
        keep = true;
        descendants = getDescendantCount(getCompact(key, resolution), resolution, RESOLUTION - resolution);
      }
    }
    if (!keep) {
      ${c.status.droppedCount?`atomicAdd(&droppedCount[droppedCountOffset], 1u);`:``}
    }
  }
  counts[countsOffset + index] = descendants;`}));let _=d(`offsets`),v=d(`scan-total`,1);return f.push(...n(e,{id:`${i}-scan`,operation:j,counts:g,offsets:_,rowCount:l,total:v})),f.push(o(e,{id:`${i}-expand`,operation:j,variant:`uncompact-expand`,bindings:[{name:`cells`,view:a.cells,type:`u32`,access:`read`},{name:`scanTotal`,view:v,type:`u32`,access:`read`},{name:`offsets`,view:_,type:`u32`,access:`read`},{name:`cellsOut`,view:c.cells,type:`u32`,access:`read_write`},{name:`countOut`,view:c.status.count,type:`u32`,access:`read_write`},{name:`overflowOut`,view:c.status.overflow,type:`u32`,access:`read_write`},...c.status.requiredCount?[{name:`totalOut`,view:c.status.requiredCount,type:`u32`,access:`read_write`}]:[]],invocationCount:c.cells.length,declarations:`const CAPACITY: u32 = ${c.cells.length}u;
const ROW_COUNT: u32 = ${l}u;
${h}

/** The \`rank\`-th (ascending) descendant path \`depth\` levels below \`compact\`. */
fn getDescendant(compact: vec2u, resolution: u32, depth: u32, rankIn: u32) -> vec2u {
  ${a.family===`h3`?`var rank = rankIn;
  var digits = vec2u(0u);
  var isChain = dggs_h3_is_base_cell_pentagon(cellShiftRight(compact, 3u * resolution).y) &&
    all((compact & cellMaskLow(3u * resolution)) == vec2u(0u));
  for (var step = 0u; step < depth; step++) {
    let remaining = depth - 1u - step;
    var digit = 0u;
    if (isChain) {
      // Pentagon: digit 0 keeps the pentagon, digit 1 is deleted, digits 2..6 are hexagons.
      if (rank >= H3_PENTAGON_SIZES[remaining]) {
        rank = rank - H3_PENTAGON_SIZES[remaining];
        digit = 2u + rank / H3_HEXAGON_SIZES[remaining];
        rank = rank % H3_HEXAGON_SIZES[remaining];
        isChain = false;
      }
    } else {
      digit = rank / H3_HEXAGON_SIZES[remaining];
      rank = rank % H3_HEXAGON_SIZES[remaining];
    }
    digits = cellShiftLeft(digits, 3u) | vec2u(0u, digit);
  }
  return cellShiftLeft(compact, 3u * depth) | digits;`:`return cellShiftLeft(compact, 2u * depth) | vec2u(0u, rankIn);`}
}`,body:`let totalCells = scanTotal[scanTotalOffset];
  let count = min(totalCells, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = count;
    overflowOut[overflowOutOffset] = select(0u, 1u, totalCells > CAPACITY);
    ${c.status.requiredCount?`totalOut[totalOutOffset] = totalCells;`:``}
  }
  var result = vec2u(0u);
  if (index < count) {
    // Last row whose offset is <= index; zero-count rows share the next row's offset.
    var low = 0u;
    var high = ROW_COUNT;
    loop {
      if (low >= high) {
        break;
      }
      let middle = (low + high) >> 1u;
      if (offsets[offsetsOffset + middle] <= index) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    let row = low - 1u;
    let key = getInputKey(vec2u(cells[cellsOffset + 2u * row], cells[cellsOffset + 2u * row + 1u]));
    let resolution = getKeyResolution(key);
    let depth = RESOLUTION - resolution;
    let compact = getDescendant(getCompact(key, resolution), resolution, depth, index - offsets[offsetsOffset + row]);
    result = getCanonicalKey(compact, RESOLUTION);
  }
  cellsOut[cellsOutOffset + 2u * index] = result.y;
  cellsOut[cellsOutOffset + 2u * index + 1u] = result.x;`})),f}},R=[140.6,37.1,142.6,38.3],z=7,B=343,V=.1,H=Math.cos((R[1]+R[3])/2*Math.PI/180);function U(e){let t=Math.sin(e*12.9898+4.1)*43758.5453;return t-Math.floor(t)}var W=x({parameters:k,geographicWindow:R,caption:`Compact a masked H3 disk around the M7.3 Honshu earthquake in March 2022.`,createInputs:()=>({center:Float32Array.from(S()),pointer:[(R[0]+R[2])/2,(R[1]+R[3])/2]}),build(e,t,n){let r=e.upload(`center`,`float32x2`,t.center),i=T(e,`h3`,4,r),a=e.allocate(`disk`,`uint32x2`,z);e.graph.add(new b({id:`disk-topology`,family:`h3`,operation:{type:`disk`,k:1},cells:i.view,output:{cells:a.view}}));let o=e.allocate(`children`,`uint32x2`,B);e.graph.add(new b({id:`children-topology`,family:`h3`,operation:{type:`children`,resolution:6,inputResolution:4},cells:a.view,output:{cells:o.view}}));let s=C(e,`h3`,o,`input-geometry`),c=e.upload(`row-hashes`,`float32`,Float32Array.from({length:B},(e,t)=>U(t))),l=e.allocate(`mask-control`,`float32`,4),u=e.allocate(`mask`,`uint32`,B);O(e,{id:`mask-eraser`,invocationCount:B,bindings:[{name:`centers`,buffer:s.centers,access:`read`},{name:`hashes`,buffer:c,access:`read`},{name:`control`,buffer:l,access:`read`},{name:`mask`,buffer:u,access:`read_write`}],body:`let base = centersOffset + index * 2u;
  let offset = vec2<f32>(
    (centers[base] - control[controlOffset + 1u]) * ${H},
    centers[base + 1u] - control[controlOffset + 2u]
  );
  let isErased = length(offset) < control[controlOffset + 3u];
  let isKept = hashes[hashesOffset + index] < control[controlOffset];
  mask[maskOffset + index] = select(0u, 1u, isKept && !isErased);`});let d=e.allocate(`compacted`,`uint32x2`,B),f=e.allocate(`count`,`uint32`,1),p=e.allocate(`overflow`,`uint32`,1);e.graph.add(new L({family:`h3`,operation:{type:`compact`,resolution:6,minimumResolution:n.minimumResolution},cells:o.view,mask:u.view,output:{cells:d.view,status:{count:f.view,overflow:p.view}}}));let m=C(e,`h3`,d,`output-geometry`),h=w(B,10,6),g=E(B,10,6);return{outputs:{inputBoundaries:s.boundaries,boundaries:m.boundaries,mask:u,count:f,indices:e.upload(`fill-indices`,`uint32`,h.indices),owners:e.upload(`fill-owners`,`uint32`,h.owners),triangleCount:h.triangleCount,wireframe:e.upload(`wireframe`,`uint32`,g),wireframeCount:g.length/3,resolutions:D(e,`compacted-resolution`,d,20)},update:e=>l.write(Float32Array.of(e.keep,t.pointer[0],t.pointer[1],V))}},rebuildOn:[`minimumResolution`],interaction:{kind:`window`,hint:`erase cells`,rest:[.5,.5],resolve(e,{inputs:t,outputs:n,peek:r}){t.pointer=[e.x,e.y];let i=r(n.mask),a=r(n.count),o=0;if(i)for(let e of i)e===0&&o++;return{rerun:!0,glyphs:[{shape:`ring`,at:t.pointer,radius:V,dashed:!0}],readout:`${o} of ${B} inputs masked · ${a?a[0]:`…`} cells out`}}},draw:({outputs:e})=>[{layer:`triangles`,context:!0,positions:e.inputBoundaries,triangles:e.wireframe,count:e.wireframeCount,radius:.8,alpha:.5},{layer:`polygons`,positions:e.boundaries,indices:e.indices,owners:e.owners,values:e.resolutions,triangleCount:e.triangleCount,ramp:`categorical`,alpha:.8},{layer:`triangles`,positions:e.boundaries,triangles:e.wireframe,count:e.wireframeCount,radius:1.6}]});export{W as default};