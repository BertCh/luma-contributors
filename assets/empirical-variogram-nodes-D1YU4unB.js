import{r as e}from"./kernel-C3tfQOn4.js";import{x as t}from"./gpu-contributor-utils-BIwLQON5.js";import{o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as r,r as i}from"./pair-histogram-COfc4qPq.js";import{a,i as o,t as s,u as c}from"./pair-statistics-grid-L8bKgCyF.js";import{t as l}from"./pair-statistics-fixed-point-unit-DoZ07ePl.js";import{c as u,r as d}from"./wgsl-common-DfyXiYB0.js";import{r as f}from"./pair-statistics-cell-neighbors-dAomEkm0.js";function p(t,r,i,o){let s=i.positions.length,c=i.lagCount*i.temporalLagCount,l=e(t,`${r}-legacy-rows`,`float32`,s*3),p=e(t,`${r}-legacy-sums`,`float32`,s*c),h=e(t,`${r}-legacy-counts`,`uint32`,s*c),g=e(t,`${r}-legacy-total-sums`,`float32`,c),_=e(t,`${r}-legacy-total-counts`,`uint32`,c*2);return[n(t,{id:`${r}-legacy-rows`,operation:`GPUCrossVariogram`,variant:`legacy-rows`,invocationCount:s,bindings:[{name:`values`,view:i.values,type:`f32`,access:`read`},...i.secondValues?[{name:`secondValues`,view:i.secondValues,type:`f32`,access:`read`}]:[],...i.times?[{name:`times`,view:i.times,type:`f32`,access:`read`}]:[],...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[],{name:`rowTable`,view:l,type:`f32`,access:`read_write`}],declarations:d,body:`rowTable[rowTableOffset + index * 3u] = ${i.mask?`select(getNaN(), values[valuesOffset + index], mask[maskOffset + index] != 0u)`:`values[valuesOffset + index]`};
rowTable[rowTableOffset + index * 3u + 1u] = ${i.secondValues?`secondValues[secondValuesOffset + index]`:`values[valuesOffset + index]`};
rowTable[rowTableOffset + index * 3u + 2u] = ${i.times?`times[timesOffset + index]`:`0.0`};`}),n(t,{id:`${r}-legacy-pairs`,operation:`GPUCrossVariogram`,variant:`legacy-pairs`,invocationCount:s,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`rowTable`,view:l,type:`f32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`cellOffsets`,view:o.cellOffsets,type:`u32`,access:`read`},{name:`sortedRows`,view:o.sortedRows,type:`u32`,access:`read`},{name:`sums`,view:p,type:`f32`,access:`read_write`},{name:`counts`,view:h,type:`u32`,access:`read_write`}],declarations:`${a(i.gridSize,!0)}\n${u}
const BINS: u32 = ${c}u;`,body:`var localSums: array<f32, BINS>;
var localCounts: array<u32, BINS>;
let lattice = readLattice();
let focus = index;
let focusX = positions[positionsOffset + 2u * focus];
let focusY = positions[positionsOffset + 2u * focus + 1u];
var focusKey = CELL_COUNT;
if (isPointValid(lattice, focusX, focusY) && isFiniteBits(rowTable[rowTableOffset + focus * 3u])) {
  focusKey = getCellRow(lattice, focusY) * lattice.columns + getCellColumn(lattice, focusX);
}
let maximumDistance = parameters[parametersOffset + 4u];
let maximumTime = parameters[parametersOffset + 5u];
if (lattice.valid && focusKey < CELL_COUNT) ${m(`focusKey`,`if (neighbor <= focus) { continue; }
let delta = vec2f(positions[positionsOffset + 2u * focus] - positions[positionsOffset + 2u * neighbor], positions[positionsOffset + 2u * focus + 1u] - positions[positionsOffset + 2u * neighbor + 1u]);
let distance = length(delta);
let time = abs(rowTable[rowTableOffset + focus * 3u + 2u] - rowTable[rowTableOffset + neighbor * 3u + 2u]);
if (!isFiniteBits(distance) || distance > maximumDistance || time > maximumTime) { continue; }
let spatialBin = min(u32(distance / maximumDistance * ${i.lagCount}.0), ${i.lagCount-1}u);
let temporalBin = min(u32(time / maximumTime * ${i.temporalLagCount}.0), ${i.temporalLagCount-1}u);
let bin = spatialBin * ${i.temporalLagCount}u + temporalBin;
let difference = rowTable[rowTableOffset + focus * 3u] - rowTable[rowTableOffset + neighbor * 3u];
let secondDifference = rowTable[rowTableOffset + focus * 3u + 1u] - rowTable[rowTableOffset + neighbor * 3u + 1u];
localSums[bin] += 0.5 * difference * secondDifference;
localCounts[bin] += 1u;`)}
for (var bin = 0u; bin < BINS; bin++) {
  sums[sumsOffset + bin * ${s}u + focus] = localSums[bin];
  counts[countsOffset + bin * ${s}u + focus] = localCounts[bin];
}`}),...f(t,{id:`${r}-legacy-sums`,operation:`GPUCrossVariogram`,segmentCount:c,rowCount:s,input:p,output:g}),...f(t,{id:`${r}-legacy-counts`,operation:`GPUCrossVariogram`,segmentCount:c,rowCount:s,input:h,output:_,detectOverflow:!0}),n(t,{id:`${r}-legacy-finish`,operation:`GPUCrossVariogram`,variant:`legacy-finish`,invocationCount:c,bindings:[{name:`totalSums`,view:g,type:`f32`,access:`read`},{name:`totalCounts`,view:_,type:`u32`,access:`read`},{name:`semivariances`,view:i.output.semivariances,type:`f32`,access:`read_write`},{name:`pairCounts`,view:i.output.pairCounts,type:`u32`,access:`read_write`},{name:`status`,view:i.output.status,type:`u32`,access:`read_write`}],declarations:`${d}\n${u}`,body:`let count = totalCounts[totalCountsOffset + index];
let overflow = totalCounts[totalCountsOffset + ${c}u + index] != 0u;
let value = totalSums[totalSumsOffset + index] / f32(count);
semivariances[semivariancesOffset + index] = select(getNaN(), value, count > 0u && isFiniteBits(value) && !overflow);
pairCounts[pairCountsOffset + index] = count;
status[statusOffset + index] = select(select(select(2u, 0u, isFiniteBits(value)), 1u, count == 0u), 3u, overflow);`})]}function m(e,t){return`{
  // Literal initializers avoid the GPU back-end's constant-loop integer-array store loss.
  var cursorSlots = array<u32, 9>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  var endSlots = array<u32, 9>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
  var activeCellCount = 0u;
  let focusColumn = i32(${e} % lattice.columns);
  let focusRow = i32(${e} / lattice.columns);
  let lastCellRow = min(focusRow + 1, i32(lattice.rows) - 1);
  let lastCellColumn = min(focusColumn + 1, i32(lattice.columns) - 1);
  for (var cellRow = max(focusRow - 1, 0); cellRow <= lastCellRow; cellRow++) {
    for (var cellColumn = max(focusColumn - 1, 0); cellColumn <= lastCellColumn; cellColumn++) {
      let cell = u32(cellRow) * lattice.columns + u32(cellColumn);
      let begin = cellOffsets[cellOffsetsOffset + cell];
      let end = cellOffsets[cellOffsetsOffset + cell + 1u];
      if (begin < end) {
        cursorSlots[activeCellCount] = begin;
        endSlots[activeCellCount] = end;
        activeCellCount++;
      }
    }
  }
  loop {
    var neighbor = 0xffffffffu;
    var selectedCell = 0xffffffffu;
    for (var cellIndex = 0u; cellIndex < activeCellCount; cellIndex++) {
      let cursor = cursorSlots[cellIndex];
      if (cursor < endSlots[cellIndex]) {
        let candidate = sortedRows[sortedRowsOffset + cursor];
        if (candidate < neighbor) {
          neighbor = candidate;
          selectedCell = cellIndex;
        }
      }
    }
    if (selectedCell == 0xffffffffu) { break; }
    cursorSlots[selectedCell]++;
    ${t}
  }
}`}var h=`GPUSpaceTimeVariogram`,g=16777216;function _(a,u,d,f=!1){let{positions:m,values:_,secondValues:v,times:y,mask:b,parameters:x,gridSize:S,output:C}=d;if(c(u,a.device,m.length),t(u,a,[m,_,v,y,b,x,...Object.values(C)]),f){let e=o(a,{id:u,operation:h,positions:m,parameters:x,gridSize:S,legacyLattice:!0,mask:b});return[...e.nodes,...p(a,u,d,e)]}let w=m.length,T=(d.lagCount+1)*(d.temporalLagCount+1),E=d.temporalLagCount+1,D=o(a,{id:u,operation:h,positions:m,parameters:x,gridSize:S,values:_,mask:b}),O=[...D.nodes],k=l(a,{id:u,operation:h,values:_,mask:b,statistics:D.statistics});O.push(...k.nodes);let A,j;if(v){let e=o(a,{id:`${u}-second`,operation:h,positions:m,parameters:x,gridSize:S,values:v,mask:b}),t=l(a,{id:`${u}-second`,operation:h,values:v,mask:b,statistics:e.statistics});O.push(...e.nodes,...t.nodes),A=e.statistics,j=t.unitVariance}let M=e(a,`${u}-scale`,`float32`,1);O.push(n(a,{id:`${u}-scale`,operation:h,variant:`scale`,invocationCount:1,bindings:[{name:`firstUnit`,view:k.unitVariance,type:`f32`,access:`read`},...j?[{name:`secondUnit`,view:j,type:`f32`,access:`read`}]:[],{name:`scale`,view:M,type:`f32`,access:`read_write`}],body:`let first = firstUnit[firstUnitOffset];
  let second = ${j?`secondUnit[secondUnitOffset]`:`first`};
  scale[scaleOffset] = select(0.0, ${g}.0 / (sqrt(first) * sqrt(second)), first > 0.0 && second > 0.0);`}));let N=e(a,`${u}-row-table`,`float32`,w*3);O.push(n(a,{id:`${u}-row-table`,operation:h,variant:`row-table`,invocationCount:w,bindings:[{name:`firstStatistics`,view:D.statistics,type:`f32`,access:`read`},...A?[{name:`secondStatistics`,view:A,type:`f32`,access:`read`}]:[],...y?[{name:`times`,view:y,type:`f32`,access:`read`}]:[],{name:`rowTable`,view:N,type:`f32`,access:`read_write`}],declarations:s,body:`let first = firstStatistics[firstStatisticsOffset + index];
  let second = ${A?`secondStatistics[secondStatisticsOffset + index]`:`first`};
  let time = ${y?`times[timesOffset + index]`:`0.0`};
  let usable = isFiniteValue(first) && isFiniteValue(second) && isFiniteValue(time);
  rowTable[rowTableOffset + index * 3u] = select(getQuietNaN(index), first, usable);
  rowTable[rowTableOffset + index * 3u + 1u] = second;
  rowTable[rowTableOffset + index * 3u + 2u] = time;`}));let P=`
const LAG_COUNT: u32 = ${d.lagCount}u;
const TEMPORAL_LAG_COUNT: u32 = ${d.temporalLagCount}u;
const TEMPORAL_CLASSES: u32 = ${E}u;
const LARGE_TERM_UNIT: f32 = 4096.0;`,F=r(a,{id:`${u}-histogram`,operation:h,positions:m,parameters:x,gridSize:S,sortedRows:D.sortedRows,cellOffsets:D.cellOffsets,slotCount:T,channelCount:5,pairOrder:`unordered`,extraBindings:[{name:`rowTable`,view:N,type:`f32`,access:`read`},{name:`scaleView`,view:M,type:`f32`,access:`read`}],declarations:P,focusPrologue:`let focusBase = rowTableOffset + focus * 3u;
    let focusFirst = rowTable[focusBase];
    let focusSecond = rowTable[focusBase + 1u];
    let focusTime = rowTable[focusBase + 2u];
    let maximumTime = readParameter(5u);
    let scale = scaleView[scaleViewOffset];
    let focusUsable = isFiniteValue(focusFirst) && isFiniteValue(maximumTime) && maximumTime > 0.0;`,pairAction:`let otherBase = rowTableOffset + neighbor * 3u;
            let otherFirst = rowTable[otherBase];
            if (!focusUsable || !isFiniteValue(otherFirst)) { continue; }
            let timeLag = abs(rowTable[otherBase + 2u] - focusTime);
            if (timeLag > maximumTime) { continue; }
            let spatialClass = select(1u + min(u32(pairDistance / lattice.maximumDistance * f32(LAG_COUNT)), LAG_COUNT - 1u), 0u, deltaX == 0.0 && deltaY == 0.0);
            let temporalClass = select(1u + min(u32(timeLag / maximumTime * f32(TEMPORAL_LAG_COUNT)), TEMPORAL_LAG_COUNT - 1u), 0u, timeLag == 0.0);
            let bin = spatialClass * TEMPORAL_CLASSES + temporalClass;
            let term = 0.5 * (rowTable[otherBase] - focusFirst) * (rowTable[otherBase + 1u] - focusSecond) * scale;
            accumulate(bin, 0u, 1u);
            accumulateScaled(bin, 1u, 3u, term);
            accumulateScaled(bin, 2u, 4u, -term);`});return O.push(...F.nodes),O.push(n(a,{id:`${u}-finish`,operation:h,variant:`finish`,invocationCount:T,bindings:[{name:`accumulators`,view:F.accumulators,type:`u32`,access:`read`},{name:`scaleView`,view:M,type:`f32`,access:`read`},{name:`semivariances`,view:C.semivariances,type:`f32`,access:`read_write`},{name:`pairCounts`,view:C.pairCounts,type:`u32`,access:`read_write`},{name:`status`,view:C.status,type:`u32`,access:`read_write`}],declarations:`${s}
${P}
${i(`accumulators`,5,``,T)}`,body:`let scale = scaleView[scaleViewOffset];
  let count = readPairAccumulator(index, 0u);
  let positive = readPairAccumulator(index, 1u) + readPairAccumulator(index, 3u) * LARGE_TERM_UNIT;
  let negative = readPairAccumulator(index, 2u) + readPairAccumulator(index, 4u) * LARGE_TERM_UNIT;
  // A zero scale means a constant column: every term is 0.
  let value = select(0.0, (positive - negative) / scale / count, scale > 0.0);
  let finite = isFiniteValue(value);
  semivariances[semivariancesOffset + index] = select(getQuietNaN(index), value, count > 0.0 && finite);
  pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);
  status[statusOffset + index] = select(select(select(2u, 0u, finite), 3u, readPairOverflow()), 1u, count == 0.0);`})),O}export{_ as t};