import{r as e,u as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-group-aggregation-CEYb6GJw.js";import{t as r}from"./gpu-scan-B4NNTfF9.js";import{t as i}from"./gpu-sort-xRm8l5yn.js";import{o as a,t as o}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as s}from"./point-neighbor-grid-index-D0wS1y1I.js";import{E as c,f as l,u}from"./gpu-contributor-utils-BIwLQON5.js";import{a as d,n as f}from"./sorted-segment-sums-Dvmm6OdR.js";import{t as p}from"./exact-orientation-wgsl-B_jI-HEc.js";import{t as m}from"./exact-geometry-wgsl-y7TUnqOh.js";import{a as h,n as g,o as _}from"./study-window-wgsl-B3mOUanG.js";import{a as v,u as y}from"./wgsl-common-DfyXiYB0.js";function b(e){return Math.min(u(e,12),2**24-1)}function x(e,t,n){l(e,`positions length`,n,1,b(t))}var S=4096;function C(e,n){c(n.positions,[`float32x2`],`${e} positions`),c(n.parameters,[`float32`],`${e} parameters`);let r=n.positions.length;if(l(e,`positions length`,r,1,2**24-1),n.parameters.length<n.parameterLength)throw Error(`${e} parameters must hold ${n.parameterLength} float32 values`);let[i,a]=n.gridSize;if(!Number.isInteger(i)||!Number.isInteger(a)||i<1||a<1||i*a>=2**24)throw Error(`${e} gridSize must be two positive integers with columns * rows < 2^24`);if(n.values&&(c(n.values,[`float32`],`${e} values`),n.values.length!==r))throw Error(`${e} values length must equal positions length`);if(n.mask&&(t(n.mask,`${e} mask`),n.mask.length!==r))throw Error(`${e} mask length must equal positions length`);return r}var w=`
${y}
// A runtime operand keeps the quiet-NaN bit pattern out of constant evaluation.
${v}

fn getOrderedFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn getOrderedFloatValue(key: u32) -> f32 {
  return bitcast<f32>(select(~key, key & 0x7fffffffu, (key & 0x80000000u) != 0u));
}
`;function T(e,t=!1){return`
const COLUMNS: u32 = ${e[0]}u;
const ROWS: u32 = ${e[1]}u;
const CELL_COUNT: u32 = ${e[0]*e[1]}u;
const CELL_MARGIN: f32 = 1.0009765625;

struct Lattice {
  valid: bool,
  minimumX: f32,
  minimumY: f32,
  maximumX: f32,
  maximumY: f32,
  cellWidth: f32,
  cellHeight: f32,
  columns: u32,
  rows: u32,
  maximumDistance: f32,
  radiusSquared: f32
}

fn readParameter(slot: u32) -> f32 {
  return parameters[parametersOffset + slot];
}

fn readLattice() -> Lattice {
  var lattice: Lattice;
  lattice.cellWidth = 1.0;
  lattice.cellHeight = 1.0;
  lattice.columns = 1u;
  lattice.rows = 1u;
  lattice.radiusSquared = 0.0;
  let minimumX = readParameter(0u);
  let minimumY = readParameter(1u);
  let maximumX = readParameter(2u);
  let maximumY = readParameter(3u);
  let radius = readParameter(4u);
  let width = maximumX - minimumX;
  let height = maximumY - minimumY;
  let radiusSquared = radius * radius;
  let cellRadius = radius * CELL_MARGIN;
  lattice.valid =
    isFiniteValue(minimumX) && isFiniteValue(minimumY) && isFiniteValue(maximumX) &&
    ${t?`isFiniteValue(maximumY) && isFiniteValue(radius) &&`:`isFiniteValue(maximumY) && isFiniteValue(radius) && isFiniteValue(width) &&
    isFiniteValue(height) && isFiniteValue(radiusSquared) && isFiniteValue(cellRadius) &&`}
    radius > 0.0 && width >= 0.0 && height >= 0.0;
  if (!lattice.valid) {
    return lattice;
  }
  lattice.minimumX = minimumX;
  lattice.minimumY = minimumY;
  lattice.maximumX = maximumX;
  lattice.maximumY = maximumY;${t?`
  lattice.maximumDistance = radius;
  lattice.radiusSquared = radiusSquared;
  // Keep all candidates when a finite-input lattice cannot represent its extent or radius.
  // Single-cell lookups below avoid subtracting the overflowing bounds again.
  if (!isFiniteValue(width) || !isFiniteValue(height) || !isFiniteValue(radiusSquared)) {
    return lattice;
  }`:``}
  lattice.cellWidth = max(width / f32(COLUMNS), cellRadius);
  lattice.cellHeight = max(height / f32(ROWS), cellRadius);
  lattice.columns = min(COLUMNS - 1u, u32(floor(width / lattice.cellWidth))) + 1u;
  lattice.rows = min(ROWS - 1u, u32(floor(height / lattice.cellHeight))) + 1u;
${t?``:`  lattice.maximumDistance = radius;
  lattice.radiusSquared = radiusSquared;
`}  return lattice;
}

fn isPointValid(lattice: Lattice, x: f32, y: f32) -> bool {
  return lattice.valid && isFiniteValue(x) && isFiniteValue(y) &&
    x >= lattice.minimumX && x <= lattice.maximumX &&
    y >= lattice.minimumY && y <= lattice.maximumY;
}

fn getCellColumn(lattice: Lattice, x: f32) -> u32 {
${t?`  if (lattice.columns == 1u) { return 0u; }
`:``}  return min(u32(floor((x - lattice.minimumX) / lattice.cellWidth)), lattice.columns - 1u);
}

fn getCellRow(lattice: Lattice, y: f32) -> u32 {
${t?`  if (lattice.rows == 1u) { return 0u; }
`:``}  return min(u32(floor((y - lattice.minimumY) / lattice.cellHeight)), lattice.rows - 1u);
}

${w}
`}function E(t,n){let r=e(t,`${n.id}-window-parameters`,`float32`,n.parameterLength);return{parameters:r,nodes:[a(t,{id:`${n.id}-window-parameters`,operation:n.operation,variant:`window-parameters`,bindings:[{name:`sourceParameters`,view:n.parameters,type:`u32`,access:`read`},g(n.window),{name:`parameters`,view:r,type:`u32`,access:`read_write`}],invocationCount:n.parameterLength,declarations:`${p}
${m}
${h(_(n.window))}`,body:`var value = sourceParameters[sourceParametersOffset + index];
  if (index < 4u) {
    value = select(0x7fc00000u, bitcast<u32>(studyWindowSummaryValue(index)), isStudyWindowUsable());
  }
  parameters[parametersOffset + index] = value;`})]}}function D(t,o){let{id:c,operation:l,positions:u,values:f,parameters:v,gridSize:y,mask:b,studyWindow:x}=o,C=u.length,E=y[0]*y[1],D=T(y,o.legacyLattice),O=[],A=e(t,`${c}-cell-keys`,`uint32`,C),j=e(t,`${c}-row-ids`,`uint32`,C),M=o.legacyIndex?void 0:s(t,{id:c,cellKeys:A,sourceRows:j,cellCount:E,rowCount:C}),N=M?.cellOffsets??e(t,`${c}-cell-offsets`,`uint32`,E+1),P=M?.sortedRows??e(t,`${c}-sorted-rows`,`uint32`,C),F=f?e(t,`${c}-value-terms`,`float32`,C):void 0;if(O.push(a(t,{id:`${c}-cell-keys`,operation:l,variant:`cell-keys`,bindings:[...x?[g(x)]:[],{name:`positions`,view:u,type:`f32`,access:`read`},{name:`parameters`,view:v,type:`f32`,access:`read`},...f?[{name:`values`,view:f,type:`f32`,access:`read`}]:[],...b?[{name:`mask`,view:b,type:`u32`,access:`read`}]:[],{name:`cellKeys`,view:A,type:`u32`,access:`read_write`},{name:`rowIds`,view:j,type:`u32`,access:`read_write`},...F?[{name:`valueTerms`,view:F,type:`f32`,access:`read_write`}]:[]],invocationCount:C,declarations:x?`${D}
${p}
${m}
${h(_(x))}`:D,body:`let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${b?`mask[maskOffset + index] != 0u`:`true`};
  ${f?`let value = values[valuesOffset + index];`:``}
  let valid = included && ${f?`isFiniteValue(value) &&`:``} isPointValid(lattice, x, y)${x?` && isInsideStudyWindow(vec2f(x, y))`:``};
  var key = CELL_COUNT;
  if (valid) {
    key = getCellRow(lattice, y) * lattice.columns + getCellColumn(lattice, x);
  }
  cellKeys[cellKeysOffset + index] = key;
  rowIds[rowIdsOffset + index] = index;
  ${f?`valueTerms[valueTermsOffset + index] = select(0.0, value, valid);`:``}`}),...M?.nodes??(()=>{let o=e(t,`${c}-cell-counts`,`uint32`,E),s=e(t,`${c}-sorted-keys`,`uint32`,C);return[...new n({id:`${c}-cell-counts`,keys:A,output:o}).getCommandNodes(t),...new r({id:`${c}-cell-scan`,input:o,output:N,mode:`exclusive`}).getCommandNodes(t),a(t,{id:`${c}-cell-total`,operation:l,variant:`cell-total`,bindings:[{name:`counts`,view:o,type:`u32`,access:`read`},{name:`cellOffsets`,view:N,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`const LAST_CELL: u32 = ${E-1}u;`,body:`cellOffsets[cellOffsetsOffset + LAST_CELL + 1u] =
    cellOffsets[cellOffsetsOffset + LAST_CELL] + counts[countsOffset + LAST_CELL];`}),...new i({id:`${c}-cell-sort`,keys:A,values:j,outputKeys:s,outputValues:P,keyBits:d(E)}).getCommandNodes(t)]})()),!f||!F)return{nodes:O,sortedRows:P,cellOffsets:N,cellKeys:A};let I=Math.ceil(C/S),L=e(t,`${c}-block-offsets`,`uint32`,I+1),R=e(t,`${c}-total-offsets`,`uint32`,2),z=e(t,`${c}-value-partials`,`float32`,I),B=e(t,`${c}-value-total`,`float32`,1),V=e(t,`${c}-square-terms`,`float32`,C),H=e(t,`${c}-square-partials`,`float32`,I),U=e(t,`${c}-square-total`,`float32`,1),W=e(t,`${c}-extremes`,`uint32`,2),G=e(t,`${c}-statistics`,`float32`,C+6),K=`const MOMENTS: u32 = ${C}u;
const CELL_COUNT: u32 = ${E}u;`;return O.push(a(t,{id:`${c}-block-offsets`,operation:l,variant:`block-offsets`,bindings:[{name:`blockOffsets`,view:L,type:`u32`,access:`read_write`},{name:`totalOffsets`,view:R,type:`u32`,access:`read_write`},{name:`extremes`,view:W,type:`u32`,access:`read_write`}],invocationCount:I+1,declarations:`const ROW_COUNT: u32 = ${C}u;
const BLOCK_ROWS: u32 = ${S}u;
const BLOCK_COUNT: u32 = ${I}u;`,body:`blockOffsets[blockOffsetsOffset + index] = min(index * BLOCK_ROWS, ROW_COUNT);
  if (index == 0u) {
    totalOffsets[totalOffsetsOffset] = 0u;
    totalOffsets[totalOffsetsOffset + 1u] = BLOCK_COUNT;
    // Ordered keys: 0xffffffff is above every finite key, 0 below every finite key.
    extremes[extremesOffset] = 0xffffffffu;
    extremes[extremesOffset + 1u] = 0u;
  }`}),...k(t,{id:`${c}-value-sum`,operation:l,input:F,partials:z,output:B,blockOffsets:L,totalOffsets:R}),a(t,{id:`${c}-moments-mean`,operation:l,variant:`moments-mean`,bindings:[{name:`cellOffsets`,view:N,type:`u32`,access:`read`},{name:`valueTotal`,view:B,type:`f32`,access:`read`},{name:`statistics`,view:G,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${K}
${w}`,body:`let count = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  statistics[statisticsOffset + MOMENTS] = count;
  statistics[statisticsOffset + MOMENTS + 1u] =
    select(getQuietNaN(index), valueTotal[valueTotalOffset] / count, count >= 1.0);`}),a(t,{id:`${c}-center`,operation:l,variant:`center`,bindings:[{name:`values`,view:f,type:`f32`,access:`read`},{name:`cellKeys`,view:A,type:`u32`,access:`read`},{name:`statistics`,view:G,type:`f32`,access:`read_write`},{name:`squareTerms`,view:V,type:`f32`,access:`read_write`},{name:`extremes`,view:W,type:`atomic<u32>`,access:`read_write`}],invocationCount:C,declarations:`${K}
${w}`,body:`let valid = cellKeys[cellKeysOffset + index] < CELL_COUNT;
  let value = values[valuesOffset + index];
  let centered = value - statistics[statisticsOffset + MOMENTS + 1u];
  // Excluded rows store quiet NaN, which every later kernel reads as "not included".
  statistics[statisticsOffset + index] = select(getQuietNaN(index), centered, valid);
  squareTerms[squareTermsOffset + index] = select(0.0, centered * centered, valid);
  if (valid) {
    let key = getOrderedFloatKey(value);
    atomicMin(&extremes[extremesOffset], key);
    atomicMax(&extremes[extremesOffset + 1u], key);
  }`}),...k(t,{id:`${c}-square-sum`,operation:l,input:V,partials:H,output:U,blockOffsets:L,totalOffsets:R}),a(t,{id:`${c}-moments`,operation:l,variant:`moments`,bindings:[{name:`squareTotal`,view:U,type:`f32`,access:`read`},{name:`extremes`,view:W,type:`u32`,access:`read`},{name:`statistics`,view:G,type:`f32`,access:`read_write`}],invocationCount:1,declarations:`${K}
${w}`,body:`let count = statistics[statisticsOffset + MOMENTS];
  let mean = statistics[statisticsOffset + MOMENTS + 1u];
  let hasRows = count >= 1.0;
  let minimum = select(getQuietNaN(index), getOrderedFloatValue(extremes[extremesOffset]), hasRows);
  let maximum = select(getQuietNaN(index), getOrderedFloatValue(extremes[extremesOffset + 1u]), hasRows);
  statistics[statisticsOffset + MOMENTS + 2u] = select(getQuietNaN(index), squareTotal[squareTotalOffset], hasRows);
  statistics[statisticsOffset + MOMENTS + 3u] = minimum;
  statistics[statisticsOffset + MOMENTS + 4u] = maximum;
  statistics[statisticsOffset + MOMENTS + 5u] = max(maximum - mean, mean - minimum);`})),{nodes:O,sortedRows:P,cellOffsets:N,cellKeys:A,statistics:G}}function O(t,n){let{id:r,operation:i,positions:o,sortedRows:s,cellOffsets:c,gridSize:l}=n,u=o.length,d=e(t,`${r}-sorted-points`,`uint32`,u*3);return{sortedPoints:d,nodes:[a(t,{id:`${r}-sorted-points`,operation:i,variant:`sorted-points`,bindings:[{name:`positions`,view:o,type:`f32`,access:`read`},{name:`sortedRows`,view:s,type:`u32`,access:`read`},{name:`cellOffsets`,view:c,type:`u32`,access:`read`},{name:`sortedPoints`,view:d,type:`u32`,access:`read_write`}],invocationCount:u,declarations:`const CELL_COUNT: u32 = ${l[0]*l[1]}u;`,body:`if (index >= cellOffsets[cellOffsetsOffset + CELL_COUNT]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  sortedPoints[sortedPointsOffset + index * 3u] = bitcast<u32>(positions[positionsOffset + row * 2u]);
  sortedPoints[sortedPointsOffset + index * 3u + 1u] = bitcast<u32>(positions[positionsOffset + row * 2u + 1u]);
  sortedPoints[sortedPointsOffset + index * 3u + 2u] = row;`})]}}function k(e,t){return[f(e,{id:`${t.id}-blocks`,operation:t.operation,segmentCount:t.partials.length,input:t.input,segmentOffsets:t.blockOffsets,output:t.partials}),f(e,{id:`${t.id}-total`,operation:t.operation,segmentCount:1,input:t.partials,segmentOffsets:t.totalOffsets,output:t.output})]}function A(t,n){let{id:r,operation:i,input:o,output:s}=n,c=o.length,l=Math.max(1,Math.ceil(c/S)),u=e(t,`${r}-block-offsets`,`uint32`,l+1),d=e(t,`${r}-total-offsets`,`uint32`,2),f=e(t,`${r}-partials`,`float32`,l);return[a(t,{id:`${r}-block-offsets`,operation:i,variant:`block-offsets`,bindings:[{name:`blockOffsets`,view:u,type:`u32`,access:`read_write`},{name:`totalOffsets`,view:d,type:`u32`,access:`read_write`}],invocationCount:l+1,declarations:`const ROW_COUNT: u32 = ${c}u;
const BLOCK_ROWS: u32 = ${S}u;
const BLOCK_COUNT: u32 = ${l}u;`,body:`blockOffsets[blockOffsetsOffset + index] = min(index * BLOCK_ROWS, ROW_COUNT);
  if (index == 0u) {
    totalOffsets[totalOffsetsOffset] = 0u;
    totalOffsets[totalOffsetsOffset + 1u] = BLOCK_COUNT;
  }`}),...k(t,{id:`${r}-sum`,operation:i,input:o,partials:f,output:s,blockOffsets:u,totalOffsets:d})]}function j(e,t,n,r){return o(e,{id:t,operation:n,view:r,type:`u32`,value:`0u`})}export{T as a,E as c,D as i,C as l,j as n,O as o,b as r,A as s,w as t,x as u};