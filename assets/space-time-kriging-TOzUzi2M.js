import{a as e,r as t}from"./prediction-covariance-DfKlqgpX.js";import{c as n,r,s as i}from"./kernel-C3tfQOn4.js";import{n as a,t as o}from"./temporal-context-xddVhAJD.js";import{s,v as c}from"./spatial-context-DwZEd4EZ.js";import{E as l,O as u,T as d,c as f,f as p,p as m,v as h,x as g,y as _}from"./gpu-contributor-utils-BIwLQON5.js";import{a as v,s as y}from"./graph-chunk-dispatch-zBt8k2-o.js";import{i as b,t as x}from"./graph-chunk-groups-BB8ACpQO.js";import{o as S}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as C}from"./gpu-prediction-utils-DjZFSzGW.js";import{r as w,u as T}from"./wgsl-common-DfyXiYB0.js";import{t as ee}from"./equilibrated-solve-wgsl-f1D0OhNH.js";import{t as E}from"./gpu-reduction-tNgOVQ_C.js";import{n as D}from"./point-neighbor-grid-index-D0wS1y1I.js";import{t as te}from"./define-run-B-brkCe4.js";import{c as O}from"./pointer-overlay-Qh8qflcA.js";import{r as ne}from"./rain-BPTWkQeX.js";import{a as k,n as re,o as A,t as j}from"./surface-CdnX4cOV.js";import{Yt as M}from"./entry-preview-D70TdSbK.js";function N(e,t){let n=new Map;for(let[e,r]of Object.entries(t))for(let[t,i]of f(r).entries()){if(i.length===0)continue;let r=i.buffer.defaultBuffer?u(i.buffer.defaultBuffer):i.buffer,a=n.get(r)??[];a.push({name:`output.${e} chunk ${t}`,offset:i.byteOffset,end:i.byteOffset+i.length*i.rowByteLength}),n.set(r,a)}for(let t of n.values()){t.sort((e,t)=>e.offset-t.offset);for(let n=1;n<t.length;n++){let r=t[n-1],i=t[n];d(e,r.end<=i.offset,`${r.name} and ${i.name} byte ranges must not overlap`)}}}var P=32,ie=3,ae=1e-5,oe=2**-62,F=`${C({observations:4})}
fn isUsableObservation(row: u32) -> bool {
  let observation = read_observations(row);
  return observation.w > 0.0 && isFiniteValue(observation.x) && isFiniteValue(observation.y) && observation.y >= 0.0;
}`;function I(e){return`
const GRID_SIDE: u32 = ${e}u;
const EXTENT_OFFSET: u32 = ${P}u;
const CELL_SLACK: f32 = ${ie}.0;
const RELATIVE_SLACK: f32 = ${ae};
const MINIMUM_PRUNABLE_DISTANCE: f32 = ${oe};
fn getAxisMinimum(axis: u32) -> f32 { return params[paramsOffset + EXTENT_OFFSET + axis * 2u]; }
fn getAxisWidth(axis: u32) -> f32 { return params[paramsOffset + EXTENT_OFFSET + axis * 2u + 1u] - getAxisMinimum(axis); }
fn getAxisScale(axis: u32) -> f32 { return select(1.0, params[paramsOffset + 26u], axis == 2u); }
fn getAxisLastCell(axis: u32) -> u32 {
  let width = getAxisWidth(axis);
  return select(0u, GRID_SIDE - 1u, isFiniteValue(width) && width > 0.0 && getAxisScale(axis) > 0.0);
}
fn getAxisCell(value: f32, axis: u32) -> u32 {
  if (getAxisLastCell(axis) == 0u) { return 0u; }
  // Scaling the numerator and denominator by searchAnisotropy cancels. Keep the raw ratio
  // to avoid overflow/underflow; bounds and pruning below use the scaled metric.
  let coordinate = (value - getAxisMinimum(axis)) / getAxisWidth(axis);
  return u32(floor(clamp(coordinate, 0.0, 1.0) * f32(GRID_SIDE - 1u)));
}
fn getAxisSlack(value: f32, axis: u32) -> f32 {
  return CELL_SLACK * getAxisWidth(axis) / f32(GRID_SIDE - 1u) + RELATIVE_SLACK * max(abs(value), max(abs(getAxisMinimum(axis)), abs(getAxisMinimum(axis) + getAxisWidth(axis))));
}
fn getAxisGap(value: f32, axis: u32, firstCell: u32, lastCell: u32) -> f32 {
  if (getAxisLastCell(axis) == 0u) { return 0.0; }
  let minimum = getAxisMinimum(axis);
  let extent = getAxisWidth(axis) / f32(GRID_SIDE - 1u);
  let lower = minimum + f32(firstCell) * extent;
  let upper = minimum + f32(lastCell + 1u) * extent;
  return max(0.0, max(lower - value, value - upper) - getAxisSlack(value, axis)) * getAxisScale(axis);
}
fn getOutsideGap(value: f32, axis: u32, firstCell: u32, lastCell: u32) -> f32 {
  let extent = getAxisWidth(axis) / f32(GRID_SIDE - 1u);
  var gap = 3.402823466e38;
  if (firstCell > 0u) { gap = min(gap, value - (getAxisMinimum(axis) + f32(firstCell) * extent)); }
  if (lastCell < getAxisLastCell(axis)) { gap = min(gap, getAxisMinimum(axis) + f32(lastCell + 1u) * extent - value); }
  if (firstCell == 0u && lastCell == getAxisLastCell(axis)) { return 3.402823466e38; }
  return max(0.0, gap - getAxisSlack(value, axis)) * getAxisScale(axis);
}`}function se(e,t){let n=t===`octant`?`var searchThreshold = 0.0;
      for (var octant = 0u; octant < 8u; octant++) {
        if (octantCounts[octant] < NEIGHBORS_PER_OCTANT) { searchThreshold = 3.402823466e38; break; }
        searchThreshold = max(searchThreshold, candidateDistance[octant * NEIGHBORS_PER_OCTANT + NEIGHBORS_PER_OCTANT - 1u]);
      }`:`var searchThreshold = 3.402823466e38;
      if (candidateCount == CANDIDATE_CAPACITY) { searchThreshold = candidateDistance[candidateCount - 1u]; }`;return`
let queryPoint = vec3<f32>(queryPosition, queryTime);
let centerCell = vec3<i32>(i32(getAxisCell(queryPoint.x, 0u)), i32(getAxisCell(queryPoint.y, 1u)), i32(getAxisCell(queryPoint.z, 2u)));
let lastCell = vec3<i32>(i32(getAxisLastCell(0u)), i32(getAxisLastCell(1u)), i32(getAxisLastCell(2u)));
let lastRing = max(max(max(centerCell.x, lastCell.x - centerCell.x), max(centerCell.y, lastCell.y - centerCell.y)), max(centerCell.z, lastCell.z - centerCell.z));
for (var ring = 0; ring <= lastRing; ring++) {
  let firstCell = max(centerCell - vec3<i32>(ring), vec3<i32>(0));
  let finalCell = min(centerCell + vec3<i32>(ring), lastCell);
  ${n}
  for (var cellTime = firstCell.z; cellTime <= finalCell.z; cellTime++) {
    for (var cellRow = firstCell.y; cellRow <= finalCell.y; cellRow++) {
      let edgePlane = abs(cellTime - centerCell.z) == ring || abs(cellRow - centerCell.y) == ring;
      for (var side = 0u; side < 2u; side++) {
        var firstColumn = firstCell.x;
        var finalColumn = finalCell.x;
        if (edgePlane) {
          if (side == 1u) { continue; }
        } else {
          let column = select(centerCell.x - ring, centerCell.x + ring, side == 1u);
          if (column < firstCell.x || column > finalCell.x) { continue; }
          firstColumn = column;
          finalColumn = column;
        }
        for (var cellColumn = firstColumn; cellColumn <= finalColumn; cellColumn++) {
        let gapX = getAxisGap(queryPoint.x, 0u, u32(cellColumn), u32(cellColumn));
        let gapY = getAxisGap(queryPoint.y, 1u, u32(cellRow), u32(cellRow));
        let gapTime = getAxisGap(queryPoint.z, 2u, u32(cellTime), u32(cellTime));
        if (max(max(gapX, gapY), gapTime) > max(searchThreshold * (1.0 + RELATIVE_SLACK), MINIMUM_PRUNABLE_DISTANCE)) { continue; }
        if (max(gapX, gapY) > max(searchRadius, MINIMUM_PRUNABLE_DISTANCE) || gapTime > max(maximumTimeLag * searchAnisotropy, MINIMUM_PRUNABLE_DISTANCE)) { continue; }
        let cell = u32(cellColumn) + GRID_SIDE * (u32(cellRow) + GRID_SIDE * u32(cellTime));
        let firstSlot = cellOffsets[cellOffsetsOffset + cell];
        let finalSlot = cellOffsets[cellOffsetsOffset + cell + 1u];
        for (var indexSlot = firstSlot; indexSlot < finalSlot; indexSlot++) {
          let sample = sortedRows[sortedRowsOffset + indexSlot];
          ${e}
        }
        }
      }
    }
  }
  ${n.replaceAll(`searchThreshold`,`finalThreshold`)}
  let outsideGap = min(min(getOutsideGap(queryPoint.x, 0u, u32(firstCell.x), u32(finalCell.x)), getOutsideGap(queryPoint.y, 1u, u32(firstCell.y), u32(finalCell.y))), getOutsideGap(queryPoint.z, 2u, u32(firstCell.z), u32(finalCell.z)));
  if (outsideGap > max(finalThreshold * (1.0 + RELATIVE_SLACK), MINIMUM_PRUNABLE_DISTANCE)) { break; }
}`}function ce(e,t,n,i,a,o){let s=n.length,c=[],l=r(e,`${t}-cell-keys`,`uint32`,s);for(let o=0;o<3;o++){let u=r(e,`${t}-axis-${o}`,`float32`,s),d=r(e,`${t}-extent-${o}`,`float32`,2);c.push(S(e,{id:`${t}-extract-${o}`,operation:`GPUSpaceTimeKriging`,bindings:[{name:`geometry`,view:n,type:`f32`,access:`read`},{name:`observations`,view:i,type:`f32`,access:`read`},{name:`coordinates`,view:u,type:`f32`,access:`read_write`},{name:`validity`,view:l,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`${T}\n${F}`,body:`let point = vec3<f32>(geometry[index * 3u], geometry[index * 3u + 1u], geometry[index * 3u + 2u]);
let usable = isUsableObservation(index) && isFiniteValue(point.x) && isFiniteValue(point.y) && isFiniteValue(point.z);
coordinates[index] = select(0.0, point[${o}u], usable);
validity[index] = select(0u, 1u, usable);`})),c.push(...new E({id:`${t}-reduce-${o}`,input:u,mask:l,output:d,operation:`extent`}).getCommandNodes(e)),c.push(S(e,{id:`${t}-publish-extent-${o}`,operation:`GPUSpaceTimeKriging`,invocationCount:2,bindings:[{name:`extent`,view:d,type:`f32`,access:`read`},{name:`params`,view:a,type:`f32`,access:`read_write`}],body:`params[${P+o*2}u + index] = extent[index];`}))}let u=r(e,`${t}-source-rows`,`uint32`,s);c.push(S(e,{id:`${t}-keys`,operation:`GPUSpaceTimeKriging`,bindings:[{name:`geometry`,view:n,type:`f32`,access:`read`},{name:`observations`,view:i,type:`f32`,access:`read`},{name:`params`,view:a,type:`f32`,access:`read`},{name:`cellKeys`,view:l,type:`u32`,access:`read_write`},{name:`sourceRows`,view:u,type:`u32`,access:`read_write`}],invocationCount:s,declarations:`${T}\n${F}\n${I(o)}`,body:`let point = vec3<f32>(geometry[index * 3u], geometry[index * 3u + 1u], geometry[index * 3u + 2u]);
sourceRows[index] = index;
var key = GRID_SIDE * GRID_SIDE * GRID_SIDE;
if (isUsableObservation(index) && isFiniteValue(point.x) && isFiniteValue(point.y) && isFiniteValue(point.z)) {
  key = getAxisCell(point.x, 0u) + GRID_SIDE * (getAxisCell(point.y, 1u) + GRID_SIDE * getAxisCell(point.z, 2u));
}
cellKeys[index] = key;`}));let d=D(e,{id:`${t}-grid`,rowCount:s,cellKeys:l,sourceRows:u,cellCount:o**3});return c.push(...d.nodes),{nodes:c,cellOffsets:d.cellOffsets,sortedRows:d.sortedRows}}function le(t,n,r,i,a,o,s,c,l,u){let{props:d,neighborCount:f,neighborSearch:p,neighborsPerOctant:m,stagedCandidateCount:h,gridSide:g}=n,_=!!u,v=+(d.mode===`ordinary`),y=p===`octant`?8*m:p===`staged`?h:f,b=[{name:`geometry`,view:a,type:`f32`,access:`read`},{name:`observations`,view:d.observations,type:`f32`,access:`read`},{name:`queryGeometry`,view:o,type:`f32`,access:`read`},{name:`params`,view:s,type:`f32`,access:`read`},...u?[{name:`cellOffsets`,view:u.cellOffsets,type:`u32`,access:`read`},{name:`sortedRows`,view:u.sortedRows,type:`u32`,access:`read`}]:[],{name:`results`,view:c,type:`f32`,access:`read_write`},{name:`statuses`,view:l,type:`u32`,access:`read_write`}],S=`  let observation = read_observations(sample);
  let position = read_positions(sample);
  let sampleTime = geometry[sample * 3u + 2u];
  if (!(observation.w > 0.0) || !isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y) || !isFiniteValue(sampleTime)) { continue; }
  let spatialLag = distance(position, queryPosition);
  let timeLag = abs(sampleTime - queryTime);
  if (!(spatialLag <= searchRadius) || !(timeLag <= maximumTimeLag)) { continue; }
  let scaledTime = searchAnisotropy * timeLag;
  let metric = sqrt(spatialLag * spatialLag + scaledTime * scaledTime);
  ${p===`octant`?`let offset = position - queryPosition;
  let octant = select(0u, 1u, offset.x < 0.0) + select(0u, 2u, offset.y < 0.0) + select(0u, 4u, sampleTime < queryTime);
  octantCounts[octant] = insertCandidate(octant * NEIGHBORS_PER_OCTANT, NEIGHBORS_PER_OCTANT, octantCounts[octant], sample, metric);`:`candidateCount = insertCandidate(0u, CANDIDATE_CAPACITY, candidateCount, sample, metric);`}`,E=p===`octant`?`var cursors: array<u32, 8>;
for (var slot = 0u; slot < NEIGHBOR_COUNT; slot++) {
  var chosen = 8u;
  for (var octant = 0u; octant < 8u; octant++) {
    if (cursors[octant] >= octantCounts[octant]) { continue; }
    if (chosen == 8u) { chosen = octant; continue; }
    let current = octant * NEIGHBORS_PER_OCTANT + cursors[octant];
    let best = chosen * NEIGHBORS_PER_OCTANT + cursors[chosen];
    if (candidateDistance[current] < candidateDistance[best] || (candidateDistance[current] == candidateDistance[best] && candidateRow[current] < candidateRow[best])) { chosen = octant; }
  }
  if (chosen == 8u) { break; }
  neighbors[retained] = candidateRow[chosen * NEIGHBORS_PER_OCTANT + cursors[chosen]];
  cursors[chosen] += 1u;
  retained += 1u;
}`:p===`staged`?`var selectedCovariance: array<f32, ${f}>;
for (var slot = 0u; slot < candidateCount; slot++) {
  let sample = candidateRow[slot];
  let position = read_positions(sample);
  let sampleTime = geometry[sample * 3u + 2u];
  let covarianceValue = spaceTimeCovariance(distance(position, queryPosition), abs(sampleTime - queryTime), all(position == queryPosition), sampleTime == queryTime);
  if (!isFiniteValue(covarianceValue)) { return; }
  var insertion = retained;
  for (var other = 0u; other < retained; other++) { if (covarianceValue > selectedCovariance[other]) { insertion = other; break; } }
  if (insertion >= NEIGHBOR_COUNT) { continue; }
  var shift = min(retained, NEIGHBOR_COUNT - 1u);
  loop { if (shift <= insertion) { break; } selectedCovariance[shift] = selectedCovariance[shift - 1u]; neighbors[shift] = neighbors[shift - 1u]; shift--; }
  selectedCovariance[insertion] = covarianceValue; neighbors[insertion] = sample; retained = min(retained + 1u, NEIGHBOR_COUNT);
}`:`for (var slot = 0u; slot < candidateCount; slot++) { neighbors[slot] = candidateRow[slot]; retained += 1u; }`;return x(t,{id:r,operation:`GPUSpaceTimeKriging`,variant:`${d.mode}-${p}`,bindings:b,invocationCount:i,workgroupSize:32,declarations:`${C({observations:4},[`results`])}
${w}
${T}
${ee(f+v)}
${e()}
const NEIGHBOR_COUNT: u32 = ${f}u;
const DRIFT_COUNT: u32 = ${v}u;
fn read_positions(row: u32) -> vec2<f32> { return vec2<f32>(geometry[row * 3u], geometry[row * 3u + 1u]); }
fn read_queries(row: u32) -> vec2<f32> { return vec2<f32>(queryGeometry[row * 3u], queryGeometry[row * 3u + 1u]); }
${_?I(g):``}
const CANDIDATE_CAPACITY: u32 = ${y}u;
${p===`octant`?`const NEIGHBORS_PER_OCTANT: u32 = ${m}u;`:``}
var<private> candidateDistance: array<f32, ${y}>;
var<private> candidateRow: array<u32, ${y}>;
fn insertCandidate(base: u32, capacity: u32, count: u32, row: u32, metric: f32) -> u32 {
  var insertion = count;
  for (var slot = 0u; slot < count; slot++) { if (metric < candidateDistance[base + slot] || (metric == candidateDistance[base + slot] && row < candidateRow[base + slot])) { insertion = slot; break; } }
  if (insertion >= capacity) { return count; }
  var shift = min(count, capacity - 1u);
  loop { if (shift <= insertion) { break; } candidateDistance[base + shift] = candidateDistance[base + shift - 1u]; candidateRow[base + shift] = candidateRow[base + shift - 1u]; shift--; }
  candidateDistance[base + insertion] = metric; candidateRow[base + insertion] = row;
  return min(count + 1u, capacity);
}`,body:`
if (index >= arrayLength(&queryGeometry) / 3u) { return; }
write_results(index, vec4<f32>(getNaN(), getNaN(), 0.0, 0.0));
statuses[statusesOffset + index] = 1u;
if (!loadSpaceTimeCovariance()) { return; }
let covarianceScale = params[paramsOffset];
let searchRadius = params[paramsOffset + 24u];
let maximumTimeLag = params[paramsOffset + 25u];
let searchAnisotropy = params[paramsOffset + 26u];
let queryPosition = read_queries(index);
let queryTime = queryGeometry[index * 3u + 2u];
if (!isFiniteValue(queryPosition.x) || !isFiniteValue(queryPosition.y) || !isFiniteValue(queryTime)) { return; }
var candidateCount = 0u;
${p===`octant`?`var octantCounts: array<u32, 8>;`:``}
${_?se(S,p):`for (var sample = 0u; sample < arrayLength(&geometry) / 3u; sample++) { ${S} }`}
var neighbors: array<u32, ${f}>;
var retained = 0u;
${E}
if (retained == 0u || retained < DRIFT_COUNT) { return; }
var crossValues: array<f32, ${f}>;
for (var row = 0u; row < retained; row++) {
  let position = read_positions(neighbors[row]);
  let observation = read_observations(neighbors[row]);
  let sampleTime = geometry[neighbors[row] * 3u + 2u];
  for (var column = 0u; column < retained; column++) {
    let otherPosition = read_positions(neighbors[column]);
    let otherTime = geometry[neighbors[column] * 3u + 2u];
    matrix[row * SYSTEM_CAPACITY + column] = spaceTimeCovariance(distance(position, otherPosition), abs(sampleTime - otherTime), all(position == otherPosition), sampleTime == otherTime) / covarianceScale;
  }
  matrix[row * SYSTEM_CAPACITY + row] += observation.y / covarianceScale;
  let crossValue = spaceTimeCovariance(distance(position, queryPosition), abs(sampleTime - queryTime), all(position == queryPosition), sampleTime == queryTime);
  crossValues[row] = crossValue; right[row] = crossValue / covarianceScale;
  for (var column = 0u; column < DRIFT_COUNT; column++) {
    matrix[row * SYSTEM_CAPACITY + retained + column] = 1.0;
    matrix[(retained + column) * SYSTEM_CAPACITY + row] = 1.0;
  }
}
for (var column = 0u; column < DRIFT_COUNT; column++) {
  right[retained + column] = 1.0;
  for (var other = 0u; other < DRIFT_COUNT; other++) { matrix[(retained + column) * SYSTEM_CAPACITY + retained + other] = 0.0; }
}
statuses[statusesOffset + index] = 2u;
if (!solveSystem(retained + DRIFT_COUNT)) { results[resultsOffset + index * 4u + 2u] = pivotRatio; return; }
let mean = params[paramsOffset + 27u];
var prediction = ${v?`0.0`:`mean`};
var variance = spaceTimeCovariance(0.0, 0.0, true, true);
let varianceScale = max(variance, 1e-20);
for (var row = 0u; row < retained; row++) {
  let observation = read_observations(neighbors[row]);
  prediction += solution[row] * (observation.x ${v?``:`- mean`});
  variance -= solution[row] * crossValues[row];
}
for (var column = 0u; column < DRIFT_COUNT; column++) { variance -= solution[retained + column] * covarianceScale; }
if (!isFiniteValue(prediction) || !isFiniteValue(variance) || variance < -1e-4 * varianceScale) { return; }
write_results(index, vec4<f32>(prediction, max(variance, 0.0), pivotRatio, f32(retained)));
statuses[statusesOffset + index] = 0u;`})}var L=32,R=1048576,z=1073741808,B=64,V=8,H=2,U=128,ue=16,de=512,W=64,fe=5,pe=1073741808,me=class{id;props;neighborCount;neighborSearch;neighborsPerOctant;stagedCandidateCount;queryCount;temporalContext;sampleIndex;gridSide;estimatedCommandNodeCount;queryChunks;constructor(e){this.props=e,this.id=e.id??`space-time-kriging`;let{id:t}=this;c(t,e.spatialContext,{...s,measure:`distance`});let n={encodings:[`float32-relative`]};this.temporalContext=a(t,e.temporalContext,n),e.queryTemporalContext&&(a(t,e.queryTemporalContext,n),o(t,e.temporalContext,e.queryTemporalContext)),this.neighborCount=e.neighborCount??16,this.neighborSearch=e.neighborSearch??`metric`,this.neighborsPerOctant=e.neighborsPerOctant??Math.ceil(this.neighborCount/8),this.stagedCandidateCount=e.stagedCandidateCount??2*this.neighborCount,this.queryCount=e.queries.length,d(t,[`auto`,`brute-force`,`grid`].includes(e.sampleIndex??`auto`),`sampleIndex must be 'auto', 'brute-force' or 'grid'`),this.sampleIndex=e.sampleIndex===`grid`||(e.sampleIndex??`auto`)===`auto`&&e.positions.length>4096?`grid`:`brute-force`,p(t,`positions.length`,e.positions.length,1,this.sampleIndex===`grid`?z:R),p(t,`neighborCount`,this.neighborCount,1,L),p(t,`queryCount`,this.queryCount,0),p(t,`neighborsPerOctant`,this.neighborsPerOctant,1,V),p(t,`stagedCandidateCount`,this.stagedCandidateCount,this.neighborCount,B),d(t,[`metric`,`octant`,`staged`].includes(this.neighborSearch),`neighborSearch must be 'metric', 'octant' or 'staged'`),d(t,[`simple`,`ordinary`].includes(e.mode),`mode must be 'simple' or 'ordinary'`),d(t,e.times.length===e.positions.length,`times.length must equal positions.length`),d(t,e.observations.length===e.positions.length,`observations.length must equal positions.length`),p(t,`parameters.length`,e.parameters.length,28),l(e.positions,[`float32x2`],`${t} positions`),l(e.times,[`float32`],`${t} times`),l(e.observations,[`float32x4`],`${t} observations`),l(e.parameters,[`float32`],`${t} parameters`);let r=f(e.queryTimes),i=f(e.output.values),u=f(e.output.status),h=f(e.queries);d(t,r.length===h.length&&i.length===h.length&&u.length===h.length,`queryTimes and output chunk counts must match queries`),this.queryChunks=h.map((e,n)=>{let a={queries:e,queryTimes:r[n],values:i[n],status:u[n]};return p(t,`queries chunk ${n} length`,e.length,0,pe),d(t,a.queryTimes.length===e.length,`queryTimes chunk lengths must match queries`),d(t,a.values.length>=e.length&&a.status.length>=e.length,`output chunk capacities must cover queries`),d(t,n===h.length-1||a.values.length===e.length&&a.status.length===e.length,`non-final output chunk lengths must match queries; spare capacity is allowed only in the final chunk`),l(a.queries,[`float32x2`],`${t} queries`),l(a.queryTimes,[`float32`],`${t} queryTimes`),l(a.values,[`float32x4`],`${t} output.values`),l(a.status,[`uint32`],`${t} output.status`),a}),d(t,h.reduce((e,t)=>e+t.length,0)===this.queryCount,`queries.length must equal its chunk total`),_(t,[e.output.values,e.output.status],[e.positions,e.times,e.observations,e.queries,e.queryTimes,e.parameters]),N(t,e.output),this.gridSide=Math.max(H,Math.min(U,Math.ceil(Math.cbrt(e.positions.length/ue))));let g=this.queryChunks.filter(e=>e.queries.length>0).length;this.estimatedCommandNodeCount=g===0?0:4*g+3+(this.sampleIndex===`grid`?de:0),m(t,this.estimatedCommandNodeCount)}getCommandNodes(e){let{id:t,props:a}=this;g(t,e,[a.positions,a.times,a.observations,a.parameters,a.queries,a.queryTimes,a.output.values,a.output.status]),v(e,t,{positions:a.positions,times:a.times,observations:a.observations,parameters:a.parameters});let o=this.queryChunks.filter(e=>e.queries.length>0);for(let n of o)v(e,t,n);let s=Math.max(0,...o.map(e=>e.queries.length)),c=a.positions.length,l=this.gridSide**3,u=[[`sample geometry`,c*12],[`query geometry`,s*12],[`query values`,s*16],[`query status`,s*4],[`private parameters`,160],[`chunk parameter slot`,20],...this.sampleIndex===`grid`?[[`axis coordinates, keys, sorted keys and row IDs`,c*4],[`small bitonic sort scratch`,c<=256?2**Math.ceil(Math.log2(c))*4:0],[`extent reduction partials`,Math.ceil(c/256)*8],[`cell counts and offsets`,(l+1)*4],[`radix sort histograms`,Math.ceil(c/256)*16*4]]:[]];for(let[n,r]of u)h(e.device,t,String(n),Math.max(Number(r),4),`reduce sample count or query chunk length`);if(p(t,`chunk parameter buffer byteLength`,o.length*W*4,0,e.device.limits.maxBufferSize),d(t,e.device.limits.maxStorageBuffersPerShaderStage>=(this.sampleIndex===`grid`?8:6),`sampleIndex ${this.sampleIndex} requires at least ${this.sampleIndex===`grid`?8:6} storage bindings`),b(e,t,this.sampleIndex===`grid`?8:6,32),o.length===0)return[];let f=r(e,`${t}-geometry`,`float32x3`,c),m=r(e,`${t}-parameters`,`float32`,40),_=[S(e,{id:`${t}-pack-samples`,operation:`GPUSpaceTimeKriging`,invocationCount:c,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`times`,view:a.times,type:`f32`,access:`read`},{name:`geometry`,view:f,type:`f32`,access:`read_write`}],body:`geometry[index * 3u] = positions[positionsOffset + index * 2u]; geometry[index * 3u + 1u] = positions[positionsOffset + index * 2u + 1u]; geometry[index * 3u + 2u] = times[timesOffset + index];`}),S(e,{id:`${t}-pack-parameters`,operation:`GPUSpaceTimeKriging`,invocationCount:28,bindings:[{name:`sourceParameters`,view:a.parameters,type:`f32`,access:`read`},{name:`params`,view:m,type:`f32`,access:`read_write`}],body:`params[index] = sourceParameters[sourceParametersOffset + index];`})],C=this.sampleIndex===`grid`?ce(e,t,f,a.observations,m,this.gridSide):void 0;C&&_.push(...C.nodes);let w=new Uint32Array(o.length*W);for(let[e,t]of o.entries())w.set([t.queries.length,n(t.queries),n(t.queryTimes),n(t.values),n(t.status)],e*W);let T=y(e,`${t}-chunks`,w);_.push(...T.nodes);for(let[n,a]of o.entries()){let o=`${t}-chunk-${n}`,s=e.createDataView(T.parameters.buffer,{format:`uint32`,length:fe,byteOffset:n*W*4}),c=r(e,`${o}-geometry`,`float32x3`,a.queries.length),l=r(e,`${o}-values`,`float32x4`,a.queries.length),u=r(e,`${o}-status`,`uint32`,a.queries.length),d=(t,n)=>{let r=i(t);return e.createDataView(t.buffer,{format:n,byteOffset:r.offset,length:r.size/4})};_.push(x(e,{id:`${o}-pack`,operation:`GPUSpaceTimeKriging`,invocationCount:a.queries.length,workgroupSize:32,bindings:[{name:`queries`,view:d(a.queries,`float32`),type:`f32`,access:`read`},{name:`queryTimes`,view:d(a.queryTimes,`float32`),type:`f32`,access:`read`},{name:`chunkParameters`,view:s,type:`u32`,access:`read`},{name:`queryGeometry`,view:c,type:`f32`,access:`read_write`}],body:`if (index >= chunkParameters[0u]) { return; }
queryGeometry[index * 3u] = queries[chunkParameters[1u] + index * 2u];
queryGeometry[index * 3u + 1u] = queries[chunkParameters[1u] + index * 2u + 1u];
queryGeometry[index * 3u + 2u] = queryTimes[chunkParameters[2u] + index];`})),_.push(le(e,this,`${o}-predict`,a.queries.length,f,c,m,l,u,C));for(let[t,n,r,i,c]of[[`values`,l,a.values,4,3],[`status`,u,a.status,1,4]]){let l=t===`values`?`f32`:`u32`;_.push(x(e,{id:`${o}-publish-${t}`,operation:`GPUSpaceTimeKriging`,invocationCount:a.queries.length,workgroupSize:32,bindings:[{name:`sourceValues`,view:n,type:l,access:`read`},{name:`chunkParameters`,view:s,type:`u32`,access:`read`},{name:`outputValues`,view:d(r,t===`values`?`float32`:`uint32`),type:l,access:`read_write`}],body:`if (index >= chunkParameters[0u]) { return; }
for (var component = 0u; component < ${i}u; component++) { outputValues[chunkParameters[${c}u] + index * ${i}u + component] = sourceValues[index * ${i}u + component]; }`}))}}return _}},G=300,he=16,K=44,q=40,J=32,Y=20,X=86400,ge=.1,_e=.05,ve=.5/X,ye={unit:`second`,epoch:0n,encoding:`float32-relative`,axis:{kind:`irregular`}},Z={parameter:M[0]},Q={parameter:M[1],logarithmic:!0},$=.95;function be(e){return{type:`separable`,sill:1,spatial:{model:`exponential`,nugget:.1,sill:.9,range:e.spatialRange},temporal:{model:`exponential`,nugget:.050000000000000044,sill:$,range:e.temporalRange*X}}}function xe({random:e,world:t}){let n=ne(e,t,G),r={...n.panel,y0:.5},i={x0:n.panel.x0,y0:.08,x1:n.panel.x0+.36*n.mapWidth,y1:.44},a={...n,panel:r},o={x0:0,y0:0,x1:t.width,y1:1},s={x0:0,y0:0,x1:a.mapWidth,y1:1},c=new Float32Array(3520);for(let e=0;e<1760;e++)c[e*2]=(e%K+.5)*(a.mapWidth/K),c[e*2+1]=(Math.floor(e/K)+.5)/q;let l=new Float32Array(G*2*4),u=new Float32Array(G*4),d=new Float32Array(G*2),f=[];for(let e=0;e<G;e++)for(let t=0;t<2;t++){let n=t===0?a.previousResiduals[e]:a.residuals[e],r=t*G+e;l.set([n,_e,0,+!!Number.isFinite(n)],r*4),u.set([a.positions[e*2],a.positions[e*2+1]],r*2),d[r]=t*X,Number.isFinite(n)&&f.push(Math.abs(n))}f.sort((e,t)=>e-t);let p=j(a.panel,J,Y),m=j(i,K,q),h=new Float32Array(640);for(let e=0;e<h.length;e++){let t=Z.parameter.min+(e%J+.5)/J*(Z.parameter.max-Z.parameter.min),n=(Math.floor(e/J)+.5)/Y,{min:r,max:i}=Q.parameter,a=r*(i/r)**+n,o=Math.min(Math.abs(t),Math.abs(t-1));h[e]=o===0?1:$*Math.exp(-3*o/a)}return{...a,mapMesh:j(s,K,q),varianceMesh:m,padRectangle:o,varianceRectangle:i,panelMesh:p,correlation:h,queries:c,observations:l,samplePositions:u,times:d,amplitude:f[Math.floor(f.length*.7)]??1}}function Se(e,n){let r=1760,i=e.allocate(`query-times`,`float32`,r),a=e.allocate(`parameters`,`float32`,28),o=e.allocate(`predictions`,`float32x4`,r);return e.graph.add(new me({temporalContext:ye,positions:e.upload(`sample-positions`,`float32x2`,n.samplePositions).view,times:e.upload(`sample-times`,`float32`,n.times).view,observations:e.upload(`observations`,`float32x4`,n.observations).view,queries:e.upload(`queries`,`float32x2`,n.queries).view,queryTimes:i.view,mode:`simple`,parameters:a.view,neighborCount:he,neighborSearch:`metric`,output:{values:o.view,status:e.allocate(`status`,`uint32`,r).view}})),{outputs:{predictions:o,stationPositions:e.upload(`station-positions`,`float32x2`,n.positions),previousValues:e.upload(`previous-residuals`,`float32`,n.previousResiduals),currentValues:e.upload(`residuals`,`float32`,n.residuals),mapPositions:e.upload(`map-positions`,`float32x2`,n.mapMesh.positions),mapIndices:e.upload(`map-indices`,`uint32`,n.mapMesh.indices),mapOwners:e.upload(`map-owners`,`uint32`,n.mapMesh.owners),panelPositions:e.upload(`panel-positions`,`float32x2`,n.panelMesh.positions),panelIndices:e.upload(`panel-indices`,`uint32`,n.panelMesh.indices),variancePositions:e.upload(`variance-positions`,`float32x2`,n.varianceMesh.positions),varianceIndices:e.upload(`variance-indices`,`uint32`,n.varianceMesh.indices),varianceOwners:e.upload(`variance-owners`,`uint32`,n.varianceMesh.owners),panelOwners:e.upload(`panel-owners`,`uint32`,n.panelMesh.owners),correlation:e.upload(`correlation`,`float32`,n.correlation)},update:e=>{i.write(new Float32Array(r).fill(e.queryTime*X)),a.write(t({covariance:be(e),searchAnisotropy:ve,mean:0}))}}}var Ce=te({parameters:M,caption:`Space-time kriging of Helene log rainfall residuals (Carolinas, two days) at a query time; right: temporal correlation over (queryTime, temporalRange).`,createInputs:xe,build:Se,async afterRun({outputs:e,upload:t}){let n=await e.predictions.buffer.readAsync(),r=new Float32Array(n.buffer,n.byteOffset,7040);return{prediction:t(`prediction`,`float32`,Float32Array.from({length:1760},(e,t)=>r[t*4])),variance:t(`variance`,`float32`,Float32Array.from({length:1760},(e,t)=>r[t*4+1]))}},interaction:re({hint:`query time and temporal range`,rectangle:({inputs:e})=>e.padRectangle,x:Z,y:Q,decorate(e,{inputs:t}){let{panel:n}=t,r=e=>{let t=A(n,Z,Q,e,Q.parameter.min);return{shape:`line`,from:t,to:[t[0],n.y1],dashed:!0}},i=e.queryTime<.5?0:1;return{glyphs:[k(n),k(t.varianceRectangle),r(0),r(1),{shape:`crosshair`,at:A(n,Z,Q,e.queryTime,e.temporalRange)}],spotlight:null,readout:`queryTime ${O(e.queryTime,2)} d · range ${O(e.temporalRange,1)} d · day ${i}`}}}),draw:({inputs:e,outputs:t,values:n,extra:r})=>[{layer:`polygons`,context:!0,positions:t.panelPositions,indices:t.panelIndices,owners:t.panelOwners,values:t.correlation,triangleCount:e.panelMesh.triangleCount,ramp:`sequential`,range:[0,1]},...r?[{layer:`polygons`,positions:t.mapPositions,indices:t.mapIndices,owners:t.mapOwners,values:r.prediction,triangleCount:e.mapMesh.triangleCount,ramp:`diverging`,range:[-e.amplitude,e.amplitude]},{layer:`polygons`,positions:t.variancePositions,indices:t.varianceIndices,owners:t.varianceOwners,values:r.variance,triangleCount:e.varianceMesh.triangleCount,ramp:`sequential`,range:[ge,1]}]:[],{layer:`dots`,context:!0,positions:t.stationPositions,values:n.queryTime<.5?t.previousValues:t.currentValues,ramp:`diverging`,range:[-e.amplitude,e.amplitude],radius:4.5}]});export{Ce as default};