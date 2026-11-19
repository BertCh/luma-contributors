import{a as e,o as t}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as n}from"./kernel-C3tfQOn4.js";import{D as r,E as i,T as a,f as o,p as s,v as c,x as l,y as u}from"./gpu-contributor-utils-BIwLQON5.js";import{v as d}from"./spatial-context-DwZEd4EZ.js";import{a as f}from"./graph-chunk-dispatch-zBt8k2-o.js";import{c as p,r as m,u as h}from"./wgsl-common-DfyXiYB0.js";import{n as g}from"./temporal-context-xddVhAJD.js";import{t as _}from"./status-aOwAAXek.js";import{n as v}from"./point-neighbor-grid-index-D0wS1y1I.js";import{t as y}from"./bounds-reduction-6MyKJx-7.js";import{n as b}from"./spatial-regression-solve-Vf-sQz9O.js";import{n as x}from"./grouped-sum-reduction-Bfd_PmGa.js";import{t as S}from"./regression-output-dispatches-KKVTWkHP.js";import{S as C,T as w,a as T,n as E}from"./local-weighted-least-squares-wgsl--tcHa6bd.js";import{r as D,t as O}from"./gwr-golden-section-nodes-BFRSkOqC.js";import{u as k}from"./inputs-CJKKIRKK.js";function A(e,t,n){if(a(e,!n||!!t,`output.status requires bandwidthSearch`),!t)return 0;i(t.parameters,[`float32`],`${e} bandwidthSearch.parameters`),a(e,t.parameters.length>=6,`bandwidthSearch.parameters must have at least 6 values`);let r=t.maximumIterations??8;return o(e,`bandwidthSearch.maximumIterations`,r,1,32),n&&_(e,n),r}function j(e,t){return D(e,D(e,t+2)+1)+1}function M(e){return`
fn getAdaptiveBandwidth(focal: u32, requested: f32, timeScale: f32) -> f32 {
  if (!isFiniteBits(requested) || requested < 1.0 || requested != round(requested)
      || requested > 128.0) { return getNaN(); }
  let count = u32(requested);
  var distances: array<f32, 128>;
  for (var neighbor = 0u; neighbor < count; neighbor++) { distances[neighbor] = 3.402823466e+38; }
  let origin = pointsOffset + focal * 3u;
  var included = 0u;
  for (var row = 0u; row < ${e}u; row++) {
    let base = pointsOffset + row * 3u;
    if (!isFiniteBits(points[base])) { continue; }
    included++;
    let deltaX = points[base] - points[origin];
    let deltaY = points[base + 1u] - points[origin + 1u];
    let deltaTime = points[base + 2u] - points[origin + 2u];
    let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
    var position = count;
    for (var neighbor = 0u; neighbor < count; neighbor++) {
      if (distance < distances[neighbor]) { position = neighbor; break; }
    }
    if (position < count) {
      for (var neighbor = count - 1u; neighbor > position; neighbor--) { distances[neighbor] = distances[neighbor - 1u]; }
      distances[position] = distance;
    }
  }
  if (included < count) { return getNaN(); }
  // The mgtwr/mgwr inflation keeps the kth distance inside the bisquare support. All-coincident
  // neighbourhoods have zero bandwidth and are singular, consistently with fixed bandwidth zero.
  return distances[count - 1u] * 1.0000001;
}`}function N(r,i){let a=i.search.maximumIterations??8,o=(e,t,n,o)=>O(r,{id:t,bounds:i.search.parameters,axis:e,integer:e===0&&i.adaptive,maximumValue:e===0&&i.adaptive?128:void 0,maximumIterations:a,parent:n,evaluate:o}),s=o(1,`${i.id}-time-scale`,void 0,(a,s,c,l)=>{let u=o(0,`${l}-bandwidth`,c,(o,s,c,l)=>{let u=n(r,`${l}-parameters`,`float32`,4),d=e(r,{id:`${l}-gate`,operation:`GTWRBandwidthSearch`,shapes:[[i.rows],[1]]}),f=i.evaluate(u,d,l);return[t(r,{id:`${l}-parameters`,operation:`GTWRBandwidthSearch`,invocationCount:1,bindings:[{name:`bandwidth`,view:o,type:`f32`,access:`read`},{name:`timeScale`,view:a,type:`f32`,access:`read`},{name:`enabled`,view:c,type:`f32`,access:`read`},{name:`parameters`,view:u,type:`f32`,access:`read_write`},d.binding],body:`parameters[parametersOffset] = bandwidth[bandwidthOffset];
  parameters[parametersOffset + 1u] = timeScale[timeScaleOffset];
  parameters[parametersOffset + 2u] = 0.0;
  parameters[parametersOffset + 3u] = 0.0;
  if (enabled[enabledOffset + 5u] > 0.5) { ${d.armWGSL} }
  else { ${d.disarmWGSL} }`}),...f.nodes,t(r,{id:`${l}-objective`,operation:`GTWRBandwidthSearch`,invocationCount:1,...d.getGate(1),bindings:[{name:`summary`,view:f.summary,type:`f32`,access:`read`},{name:`score`,view:s,type:`f32`,access:`read_write`}],declarations:m,body:`score[scoreOffset] = select(getNaN(), summary[summaryOffset + 2u],
    summary[summaryOffset + 6u] == 0.0 && summary[summaryOffset + 4u] > 0.0);
  score[scoreOffset + 1u] = 0.0;
  score[scoreOffset + 2u] = 1.0;
  score[scoreOffset + 3u] = 0.0;`})]});return[...u.nodes,t(r,{id:`${l}-profile`,operation:`GTWRBandwidthSearch`,invocationCount:1,bindings:[{name:`result`,view:u.result,type:`f32`,access:`read`},{name:`score`,view:s,type:`f32`,access:`read_write`}],body:`score[scoreOffset] = result[resultOffset + 1u];
  score[scoreOffset + 1u] = result[resultOffset];
  score[scoreOffset + 2u] = result[resultOffset + 3u];
  score[scoreOffset + 3u] = result[resultOffset + 6u];`})]}),c=Object.entries(i.status??{}).filter(e=>!!e[1]);return[...s.nodes,t(r,{id:`${i.id}-selected-parameters`,operation:`GTWRBandwidthSearch`,invocationCount:1,bindings:[{name:`result`,view:s.result,type:`f32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read_write`},...c.map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],body:`parameters[parametersOffset + ${i.parameterOffset??0}u] = result[resultOffset + 2u];
  parameters[parametersOffset + ${(i.parameterOffset??0)+1}u] = result[resultOffset];
  ${c.map(([e])=>{let t=e===`iterationCount`?`u32(result[resultOffset + 4u])`:e===`converged`?`u32(result[resultOffset + 3u])`:`u32(result[resultOffset + 6u])`;return`${e}[${e}Offset] = ${i.accumulateStatus?e===`iterationCount`?`${e}[${e}Offset] + ${t}`:`${e===`converged`?`min`:`max`}(${e}[${e}Offset], ${t})`:t};`}).join(`
  `)}`})]}function ee(e,r){let{id:i,rows:a,gate:o}=r,s=n(e,`${i}-fit`,`float32`,a*3),c=n(e,`${i}-summary`,`float32`,12),l=e=>r.component===0?`1.0`:`predictors[predictorsOffset + ${e} * ${r.predictorCount}u + ${r.component-1}u]`;return{summary:c,nodes:[t(e,{id:`${i}-fit`,operation:`MGTWRComponentBandwidthSearch`,invocationCount:a,...o.getGate(0),bindings:[{name:`points`,view:r.points,type:`f32`,access:`read`},{name:`predictors`,view:r.predictors,type:`f32`,access:`read`},{name:`partial`,view:r.partial,type:`f32`,access:`read`},{name:`parameters`,view:r.parameters,type:`f32`,access:`read`},{name:`scratch`,view:s,type:`f32`,access:`read_write`}],declarations:`${m}\n${p}\n${T()}\n${r.adaptive?M(a):``}`,body:`let base = scratchOffset + index * 3u;
  scratch[base] = getNaN();
  scratch[base + 1u] = getNaN();
  scratch[base + 2u] = select(0.0, 1.0, isFiniteBits(points[pointsOffset + index * 3u]));
  if (scratch[base + 2u] == 0.0) { return; }
  let timeScale = parameters[parametersOffset + 1u];
  let bandwidth = ${r.adaptive?`getAdaptiveBandwidth(index, parameters[parametersOffset], timeScale)`:`parameters[parametersOffset]`};
  if (!(bandwidth > 0.0 && isFiniteBits(bandwidth) && timeScale >= 0.0 && isFiniteBits(timeScale))) { return; }
  var numerator = 0.0;
  var denominator = 0.0;
  for (var row = 0u; row < ${a}u; row++) {
    let point = pointsOffset + row * 3u;
    let origin = pointsOffset + index * 3u;
    if (!isFiniteBits(points[point])) { continue; }
    let deltaX = points[point] - points[origin];
    let deltaY = points[point + 1u] - points[origin + 1u];
    let deltaTime = points[point + 2u] - points[origin + 2u];
    let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
    let weight = getLocalKernelWeight(distance, bandwidth, ${r.bisquare?`true`:`false`});
    let design = ${l(`row`)};
    numerator += weight * design * partial[partialOffset + row];
    denominator += weight * design * design;
  }
  let ownDesign = ${l(`index`)};
  if (denominator > 0.0 && isFiniteBits(numerator) && isFiniteBits(denominator)) {
    // Divide before multiplying by the focal design, as in the final component smoother.
    // ownDesign * numerator can overflow even when the coefficient and fitted value are finite.
    let coefficient = numerator / denominator;
    scratch[base] = ownDesign * coefficient;
    scratch[base + 1u] = ownDesign * ownDesign / denominator;
  }`}),t(e,{id:`${i}-summary`,operation:`MGTWRComponentBandwidthSearch`,invocationCount:1,...o.getGate(1),bindings:[{name:`scratch`,view:s,type:`f32`,access:`read`},{name:`partial`,view:r.partial,type:`f32`,access:`read`},{name:`summary`,view:c,type:`f32`,access:`read_write`}],declarations:`${m}\n${p}`,body:`var count = 0.0;
  var singular = 0.0;
  var residualSquares = 0.0;
  var trace = 0.0;
  for (var row = 0u; row < ${a}u; row++) {
    let base = scratchOffset + row * 3u;
    if (scratch[base + 2u] == 0.0) { continue; }
    if (!isFiniteBits(scratch[base]) || !isFiniteBits(scratch[base + 1u])) { singular += 1.0; continue; }
    count += 1.0;
    let residual = partial[partialOffset + row] - scratch[base];
    residualSquares += residual * residual;
    trace += scratch[base + 1u];
  }
  let denominator = count - trace - 2.0;
  // Work in log space: 2 pi * RSS can overflow even when RSS and 2 pi * RSS / n are finite.
  let logVariance = log(residualSquares) - log(count);
  let logLikelihood = -0.5 * count * (logVariance + 1.8378770664093453 + 1.0);
  for (var field = 0u; field < 12u; field++) { summary[summaryOffset + field] = getNaN(); }
  summary[summaryOffset + 2u] = select(getNaN(), -2.0 * logLikelihood + 2.0 * count * (trace + 1.0) / denominator, count > 0.0 && denominator > 0.0);
  summary[summaryOffset + 4u] = count;
  summary[summaryOffset + 6u] = singular;`})]}}var P=10842021724855044e-35,F=11754943508222875e-54;function I(){return`
// Distance to a not-yet-visited slab, padded by a whole cell for float bin/edge rounding.
fn getUnvisitedAxisGap(origin: f32, lower: f32, upper: f32, firstCell: i32, lastCell: i32) -> f32 {
  let width = (upper - lower) / f32(GRID_DIMENSION);
  if (!isFiniteBits(width) || !(width > 0.0)) { return 0.0; }
  var gap = 3.402823466e+38;
  if (firstCell > 0) {
    gap = min(gap, origin - (lower + f32(firstCell) * width));
  }
  if (lastCell < i32(GRID_DIMENSION) - 1) {
    gap = min(gap, lower + f32(lastCell + 1) * width - origin);
  }
  return max(0.0, gap - width - abs(origin) * 1.0e-6);
}

fn getAdaptiveBandwidth(focal: u32, requested: f32, timeScale: f32) -> f32 {
  if (!isFiniteBits(requested) || requested < 1.0 || requested != round(requested)
      || requested > 128.0 || !isFiniteBits(timeScale) || timeScale < 0.0) {
    return getNaN();
  }
  let count = u32(requested);
  if (count > cellOffsets[cellOffsetsOffset + GRID_DIMENSION * GRID_DIMENSION * GRID_DIMENSION]) { return getNaN(); }
  let originBase = recordsOffset + focal * RECORD_WORDS;
  let originX = records[originBase];
  if (!isFiniteBits(originX)) { return getNaN(); }
  let originY = records[originBase + 1u];
  let originTime = records[originBase + 2u];
  let minimumX = spatialBounds[spatialBoundsOffset];
  let maximumX = spatialBounds[spatialBoundsOffset + 2u];
  let minimumY = spatialBounds[spatialBoundsOffset + 1u];
  let maximumY = spatialBounds[spatialBoundsOffset + 3u];
  let minimumTime = temporalBounds[temporalBoundsOffset];
  let maximumTime = temporalBounds[temporalBoundsOffset + 2u];
  let centerColumn = i32(getGTWRBinIndex(originX, minimumX, maximumX));
  let centerRow = i32(getGTWRBinIndex(originY, minimumY, maximumY));
  var centerTime = 0;
  var lastTime = 0;
  if (timeScale > 0.0 && maximumTime > minimumTime) {
    centerTime = i32(getGTWRBinIndex(originTime, minimumTime, maximumTime));
    lastTime = i32(GRID_DIMENSION) - 1;
  }
  let lastColumnCell = select(0, i32(GRID_DIMENSION) - 1, maximumX > minimumX);
  let lastRowCell = select(0, i32(GRID_DIMENSION) - 1, maximumY > minimumY);
  let lastRing = max(max(lastColumnCell, lastRowCell), lastTime);
  var distances: array<f32, 128>;
  for (var neighbor = 0u; neighbor < count; neighbor++) { distances[neighbor] = 3.402823466e+38; }
  for (var ring = 0; ring <= lastRing; ring++) {
    let firstColumn = max(0, centerColumn - ring);
    let lastColumn = min(lastColumnCell, centerColumn + ring);
    let firstRow = max(0, centerRow - ring);
    let lastRow = min(lastRowCell, centerRow + ring);
    let firstTimeCell = max(0, centerTime - ring);
    let lastTimeCell = min(lastTime, centerTime + ring);
    for (var timeCell = firstTimeCell; timeCell <= lastTimeCell; timeCell++) {
      let timeEdge = timeScale > 0.0 && abs(timeCell - centerTime) == ring;
      for (var gridRow = firstRow; gridRow <= lastRow; gridRow++) {
        let fullColumns = timeEdge || abs(gridRow - centerRow) == ring;
        for (var side = 0; side < 2; side++) {
          var columnLow = firstColumn;
          var columnHigh = lastColumn;
          if (fullColumns) {
            if (side == 1) { continue; }
          } else {
            let column = select(centerColumn + ring, centerColumn - ring, side == 0);
            if (column < 0 || column > lastColumnCell) { continue; }
            columnLow = column;
            columnHigh = column;
          }
          let cellBase = (u32(timeCell) * GRID_DIMENSION + u32(gridRow)) * GRID_DIMENSION;
          let firstSlot = cellOffsets[cellOffsetsOffset + cellBase + u32(columnLow)];
          let endSlot = cellOffsets[cellOffsetsOffset + cellBase + u32(columnHigh) + 1u];
          for (var slot = firstSlot; slot < endSlot; slot++) {
            let row = sortedRows[sortedRowsOffset + slot];
            let base = recordsOffset + row * RECORD_WORDS;
            let deltaX = records[base] - originX;
            let deltaY = records[base + 1u] - originY;
            let deltaTime = records[base + 2u] - originTime;
            let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
            if (distance < distances[count - 1u]) {
              var position = count - 1u;
              loop {
                if (position == 0u || distances[position - 1u] <= distance) { break; }
                distances[position] = distances[position - 1u];
                position--;
              }
              distances[position] = distance;
            }
          }
        }
      }
    }
    var gap = 3.402823466e+38;
    if (firstColumn > 0 || lastColumn < lastColumnCell) {
      gap = min(gap, getUnvisitedAxisGap(originX, minimumX, maximumX, firstColumn, lastColumn));
    }
    if (firstRow > 0 || lastRow < lastRowCell) {
      gap = min(gap, getUnvisitedAxisGap(originY, minimumY, maximumY, firstRow, lastRow));
    }
    if (firstTimeCell > 0 || lastTimeCell < lastTime) {
      let rawTimeGap = getUnvisitedAxisGap(originTime, minimumTime, maximumTime, firstTimeCell, lastTimeCell);
      var timeGap = 0.0;
      if (rawTimeGap > max(${P}, ${F} / timeScale)) {
        timeGap = rawTimeGap * sqrt(timeScale);
      }
      gap = min(gap, timeGap);
    }
    // Never stop on a distance that could be affected by subnormal squared-distance flushing.
    gap = max(0.0, gap - ${2*P});
    if (distances[count - 1u] < gap) { break; }
  }
  return distances[count - 1u] * 1.0000001;
}`}function L(e,t,r){let i=r.response.length,a=`GPUGeographicallyTemporallyWeightedRegression`,{output:o}=r,s=Array.from({length:4},(r,i)=>n(e,`${t}-statistic-${i}`,`float32`,1)),c=n(e,`${t}-included-count`,`uint32`,1),l=n(e,`${t}-response-anchor`,`float32`,4),u=n(e,`${t}-total-squares`,`float32`,1);return[...y(e,{id:`${t}-response-anchor`,operation:a,variant:`response-anchor`,itemCount:i,output:l,bindings:[{name:`response`,view:r.response,type:`f32`,access:`read`},{name:`localStatus`,view:o.localStatus,type:`u32`,access:`read`}],declarations:`fn readItem(row: u32) -> vec4f {
  return vec4f(response[responseOffset + row], 0.0,
    select(0.0, 1.0, localStatus[localStatusOffset + row] == 0u), 0.0);
}`}),...x(e,{id:`${t}-statistics`,operation:a,rows:i,groupCount:1,bindings:[{name:`response`,view:r.response,type:`f32`,access:`read`},{name:`responseAnchor`,view:l,type:`f32`,access:`read`},{name:`fitted`,view:o.fitted,type:`f32`,access:`read`},{name:`hatDiagonal`,view:o.hatDiagonal,type:`f32`,access:`read`},{name:`localStatus`,view:o.localStatus,type:`u32`,access:`read`}],outputs:s,countOutput:c,rowSnippet:`let code = localStatus[localStatusOffset + row];
  if (code != 2u) { group = 0u; }
  if (code == 0u) {
    let observed = response[responseOffset + row];
    let residual = observed - fitted[fittedOffset + row];
    value0 = residual * residual;
    value1 = hatDiagonal[hatDiagonalOffset + row];
    // Center before reducing: every constant response gives exactly zero, even when 1/ROWS
    // is inexact. Half scaling keeps subtraction finite for opposite finite extrema.
    value2 = (observed * 0.5 - responseAnchor[responseAnchorOffset] * 0.5) / f32(ROWS);
    value3 = 1.0;
  }`}),...x(e,{id:`${t}-variance`,operation:a,rows:i,groupCount:1,bindings:[{name:`response`,view:r.response,type:`f32`,access:`read`},{name:`localStatus`,view:o.localStatus,type:`u32`,access:`read`},{name:`scaledDeviation`,view:s[2],type:`f32`,access:`read`},{name:`responseAnchor`,view:l,type:`f32`,access:`read`},{name:`observationCount`,view:s[3],type:`f32`,access:`read`}],outputs:[u],rowSnippet:`if (localStatus[localStatusOffset + row] == 0u) {
    group = 0u;
    let meanDeviation = scaledDeviation[scaledDeviationOffset] / (observationCount[observationCountOffset] / f32(ROWS));
    let deviation = 2.0 * ((response[responseOffset + row] * 0.5 - responseAnchor[responseAnchorOffset] * 0.5) - meanDeviation);
    value0 = deviation * deviation;
  }`,foldSuffix:`summary`,finish:{bindings:[{name:`residualSquares`,view:s[0],type:`f32`,access:`read`},{name:`hatTrace`,view:s[1],type:`f32`,access:`read`},{name:`observationCount`,view:s[3],type:`f32`,access:`read`},{name:`includedCount`,view:c,type:`u32`,access:`read`},{name:`summary`,view:o.summary,type:`f32`,access:`read_write`}],declarations:m,body:`let observations = observationCount[observationCountOffset];
  let included = f32(includedCount[includedCountOffset]);
  let residualSum = residualSquares[residualSquaresOffset];
  let trace = hatTrace[hatTraceOffset];
  let logVariance = log(residualSum) - log(observations);
  let logLikelihood = -0.5 * observations * (logVariance + 1.8378770664093453 + 1.0);
  let correctedDenominator = observations - trace - 2.0;
  let corrected = -2.0 * logLikelihood + 2.0 * observations * (trace + 1.0) / correctedDenominator;
  for (var slot = 0u; slot < 12u; slot++) { summary[summaryOffset + slot] = getNaN(); }
  summary[summaryOffset] = residualSum;
  summary[summaryOffset + 1u] = trace;
  summary[summaryOffset + 2u] = select(getNaN(), corrected, observations > 0.0 && correctedDenominator > 0.0);
  summary[summaryOffset + 3u] = select(getNaN(), 1.0 - residualSum / sum0, sum0 > 0.0);
  summary[summaryOffset + 4u] = observations;
  summary[summaryOffset + 6u] = included - observations;
  summary[summaryOffset + 7u] = select(getNaN(), residualSum / (observations - trace), observations > trace);
  summary[summaryOffset + 10u] = select(getNaN(), logLikelihood, observations > 0.0);
  summary[summaryOffset + 11u] = included;`}})]}var R=8,z=4;function B(e){return Math.max(1,Math.ceil(Math.cbrt(e/R)))}function V(e,t,n,r){let i=B(n),a=Math.min(256,Math.ceil(n/4096)),o=Math.min(1024,Math.ceil(n/256));for(let[s,l]of[[`packed input and fit records`,n*(r+z)*4],[`grid cell offsets`,(i**3+1)*4],[`grid keys and source rows`,n*4],[`grid radix histogram and scan scratch`,Math.ceil(n/256)*16*4],[`bounds`,16],[`bounds partials`,a>1?a*16:0],[`summary partials`,o*4*4]])c(e,t,s,l,`reduce rows or predictorCount`)}function H(e,r,i){let a=i.response.length,o=i.predictorCount,s=o+1,c=o+z,l=B(a),u=l**3,d=`GPUGeographicallyTemporallyWeightedRegression`,f=n(e,`${r}-records`,`float32`,a*c),g=n(e,`${r}-fits`,`float32`,a*c),_=n(e,`${r}-spatial-bounds`,`float32`,4),x=n(e,`${r}-temporal-bounds`,`float32`,4),S=n(e,`${r}-cell-keys`,`uint32`,a),C=n(e,`${r}-source-rows`,`uint32`,a),D=`const RECORD_WORDS: u32 = ${c}u;
const GRID_DIMENSION: u32 = ${l}u;
${p}
${m}
${w}
// Opposite finite extrema can overflow their difference; scale only the bin mapping in that case.
fn getGTWRBinIndex(value: f32, lower: f32, upper: f32) -> u32 {
  if (!isFiniteBits(upper - lower)) {
    return getBinIndex(value * 0.5, lower * 0.5, upper * 0.5, GRID_DIMENSION);
  }
  return getBinIndex(value, lower, upper, GRID_DIMENSION);
}`,O=[t(e,{id:`${r}-pack-records`,operation:d,invocationCount:a,bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`times`,view:i.times,type:`f32`,access:`read`},{name:`predictors`,view:i.predictors,type:`f32`,access:`read`},{name:`response`,view:i.response,type:`f32`,access:`read`},...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[],{name:`records`,view:f,type:`f32`,access:`read_write`}],declarations:D,body:`let base = recordsOffset + index * RECORD_WORDS;
  let positionX = positions[positionsOffset + 2u * index];
  let positionY = positions[positionsOffset + 2u * index + 1u];
  let time = times[timesOffset + index];
  let observed = response[responseOffset + index];
  var included = ${i.mask?`mask[maskOffset + index] != 0u`:`true`};
  included = included && isFiniteBits(positionX) && isFiniteBits(positionY) && isFiniteBits(time) && isFiniteBits(observed);
  for (var column = 0u; column < ${o}u; column++) {
    let predictor = predictors[predictorsOffset + index * ${o}u + column];
    included = included && isFiniteBits(predictor);
    records[base + 4u + column] = predictor;
  }
  records[base] = select(getNaN(), positionX, included);
  records[base + 1u] = positionY;
  records[base + 2u] = time;
  records[base + 3u] = observed;`})];for(let[t,n,i,o]of[[`spatial`,_,0,1],[`temporal`,x,2,2]])O.push(...y(e,{id:`${r}-${t}`,operation:d,variant:`${t}-extent`,itemCount:a,output:n,bindings:[{name:`records`,view:f,type:`f32`,access:`read`}],declarations:`${D}
fn readItem(row: u32) -> vec4f {
  let base = recordsOffset + row * RECORD_WORDS;
  let included = isFiniteBits(records[base]);
  return vec4f(records[base + ${i}u], records[base + ${o}u], select(0.0, 1.0, included), 0.0);
}`}));O.push(t(e,{id:`${r}-cell-keys`,operation:d,invocationCount:a,bindings:[{name:`records`,view:f,type:`f32`,access:`read`},{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`spatialBounds`,view:_,type:`f32`,access:`read`},{name:`temporalBounds`,view:x,type:`f32`,access:`read`},{name:`cellKeys`,view:S,type:`u32`,access:`read_write`},{name:`sourceRows`,view:C,type:`u32`,access:`read_write`}],declarations:D,body:`let base = recordsOffset + index * RECORD_WORDS;
  var key = ${u}u;
  if (isFiniteBits(records[base])) {
    let column = getGTWRBinIndex(records[base], spatialBounds[spatialBoundsOffset], spatialBounds[spatialBoundsOffset + 2u]);
    let gridRow = getGTWRBinIndex(records[base + 1u], spatialBounds[spatialBoundsOffset + 1u], spatialBounds[spatialBoundsOffset + 3u]);
    var timeCell = 0u;
    if (parameters[parametersOffset + 1u] > 0.0) {
      timeCell = getGTWRBinIndex(records[base + 2u], temporalBounds[temporalBoundsOffset], temporalBounds[temporalBoundsOffset + 2u]);
    }
    key = (timeCell * GRID_DIMENSION + gridRow) * GRID_DIMENSION + column;
  }
  cellKeys[cellKeysOffset + index] = key;
  sourceRows[sourceRowsOffset + index] = index;`}));let k=v(e,{id:`${r}-grid`,rowCount:a,cellKeys:S,sourceRows:C,cellCount:u});O.push(...k.nodes),O.push(t(e,{id:`${r}-fit`,operation:d,variant:`indexed-local-fit`,invocationCount:a,bindings:[{name:`parameters`,view:i.parameters,type:`f32`,access:`read`},{name:`records`,view:f,type:`f32`,access:`read`},{name:`cellOffsets`,view:k.cellOffsets,type:`u32`,access:`read`},{name:`sortedRows`,view:k.sortedRows,type:`u32`,access:`read`},{name:`spatialBounds`,view:_,type:`f32`,access:`read`},{name:`temporalBounds`,view:x,type:`f32`,access:`read`},{name:`fits`,view:g,type:`f32`,access:`read_write`}],declarations:`${D}
${h}
${b(s,s)}
${E(s)}
${T(i.gaussianCutoff??null)}
${i.adaptive?I():``}
// Inclusive padded cell range. Exact original metric/weight tests reject false positives.
fn getCellRange(origin: f32, reach: f32, lower: f32, upper: f32) -> vec2u {
  if (!isFiniteBits(reach)) { return vec2u(0u, GRID_DIMENSION - 1u); }
  let padded = reach * 1.0001 + abs(origin) * 2.4e-7;
  return vec2u(getGTWRBinIndex(clamp(origin - padded, lower, upper), lower, upper),
    getGTWRBinIndex(clamp(origin + padded, lower, upper), lower, upper));
}`,body:U(o,i.kernel===`bisquare`,i.gaussianCutoff!==void 0,i.adaptive??!1)}));let{output:A}=i;return O.push(t(e,{id:`${r}-publish-fit`,operation:d,invocationCount:a,bindings:[{name:`fits`,view:g,type:`f32`,access:`read`},{name:`coefficients`,view:A.coefficients,type:`f32`,access:`read_write`},{name:`fitted`,view:A.fitted,type:`f32`,access:`read_write`},{name:`hatDiagonal`,view:A.hatDiagonal,type:`f32`,access:`read_write`},{name:`localStatus`,view:A.localStatus,type:`u32`,access:`read_write`}],body:`let base = fitsOffset + index * ${c}u;
  for (var column = 0u; column < ${s}u; column++) {
    coefficients[coefficientsOffset + index * ${s}u + column] = fits[base + column];
  }
  fitted[fittedOffset + index] = fits[base + ${s}u];
  hatDiagonal[hatDiagonalOffset + index] = fits[base + ${s+1}u];
  localStatus[localStatusOffset + index] = u32(fits[base + ${s+2}u]);`})),A.residuals&&O.push(t(e,{id:`${r}-residuals`,operation:d,invocationCount:a,bindings:[{name:`response`,view:i.response,type:`f32`,access:`read`},{name:`fitted`,view:A.fitted,type:`f32`,access:`read`},{name:`residuals`,view:A.residuals,type:`f32`,access:`read_write`}],body:`residuals[residualsOffset + index] = response[responseOffset + index] - fitted[fittedOffset + index];`})),O.push(...L(e,r,i)),O}function U(e,t,n,r){let i=e+1;return`let timeScale = parameters[parametersOffset + 1u];
  let bandwidth = ${r?`getAdaptiveBandwidth(index, parameters[parametersOffset], timeScale)`:`parameters[parametersOffset]`};
  let originX = records[recordsOffset + RECORD_WORDS * index];
  var code = 1u;
  var coefficient: array<f32, ${i}>;
  for (var column = 0u; column < ${i}u; column++) {
    coefficient[column] = getNaN();
  }
  var fittedValue = getNaN();
  var hat = getNaN();
  if (!isFiniteBits(originX)) {
    code = 2u;
  } else if (bandwidth > 0.0 && isFiniteBits(bandwidth) && timeScale >= 0.0 && isFiniteBits(timeScale)) {
    let originY = records[recordsOffset + RECORD_WORDS * index + 1u];
    let originTime = records[recordsOffset + RECORD_WORDS * index + 2u];
    let originBase = recordsOffset + index * RECORD_WORDS + 4u;
    var normalMatrix: array<f32, ${i*i}>;
    var normalResponse: array<f32, ${i}>;
    var design: array<f32, ${i}>;
    for (var entry = 0u; entry < ${i*i}u; entry++) { normalMatrix[entry] = 0.0; }
    for (var column = 0u; column < ${i}u; column++) { normalResponse[column] = 0.0; }
    design[0] = 1.0;
    let radius = max(${t||!n?`bandwidth`:`bandwidth * GAUSSIAN_CUTOFF`}, ${P});
    let columnRange = getCellRange(originX, radius, spatialBounds[spatialBoundsOffset], spatialBounds[spatialBoundsOffset + 2u]);
    let rowRange = getCellRange(originY, radius, spatialBounds[spatialBoundsOffset + 1u], spatialBounds[spatialBoundsOffset + 3u]);
    var timeRange = vec2u(0u);
    if (timeScale > 0.0) {
      // Account for either multiplication order of tau * dt * dt and subnormal flushing.
      let timeReach = max(max(radius / sqrt(timeScale), ${P}), ${F} / timeScale);
      timeRange = getCellRange(originTime, timeReach, temporalBounds[temporalBoundsOffset], temporalBounds[temporalBoundsOffset + 2u]);
    }
    for (var timeCell = timeRange.x; timeCell <= timeRange.y; timeCell++) {
      for (var gridRow = rowRange.x; gridRow <= rowRange.y; gridRow++) {
        let cellBase = (timeCell * GRID_DIMENSION + gridRow) * GRID_DIMENSION;
        let firstSlot = cellOffsets[cellOffsetsOffset + cellBase + columnRange.x];
        let endSlot = cellOffsets[cellOffsetsOffset + cellBase + columnRange.y + 1u];
        for (var slot = firstSlot; slot < endSlot; slot++) {
          let row = sortedRows[sortedRowsOffset + slot];
          let rowX = records[recordsOffset + RECORD_WORDS * row];
          if (!isFiniteBits(rowX)) { continue; }
          let deltaX = rowX - originX;
          let deltaY = records[recordsOffset + RECORD_WORDS * row + 1u] - originY;
          let deltaTime = records[recordsOffset + RECORD_WORDS * row + 2u] - originTime;
          let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
          let weight = getLocalKernelWeight(distance, bandwidth, ${t?`true`:`false`});
          if (!(weight > 0.0)) { continue; }
          let rowBase = recordsOffset + row * RECORD_WORDS + 4u;
          for (var column = 0u; column < ${e}u; column++) {
            design[column + 1u] = records[rowBase + column] - records[originBase + column];
          }
          let observed = records[recordsOffset + row * RECORD_WORDS + 3u];
          for (var firstColumn = 0u; firstColumn < ${i}u; firstColumn++) {
            let weighted = weight * design[firstColumn];
            normalResponse[firstColumn] = normalResponse[firstColumn] + weighted * observed;
            for (var secondColumn = 0u; secondColumn <= firstColumn; secondColumn++) {
              normalMatrix[firstColumn * ${i}u + secondColumn] = normalMatrix[firstColumn * ${i}u + secondColumn] + weighted * design[secondColumn];
            }
          }
        }
      }
    }
    var scale: array<f32, ${i}>;
    if (factorEquilibrated_${i}(&normalMatrix, &scale)) {
      let inverse00 = inverseDiagonalEquilibrated_${i}(&normalMatrix, &scale, 0u);
      // normalResponse is X'Wy of the centered design; the solve is in place and unscales the coefficients.
      solveEquilibrated_${i}(&normalMatrix, &scale, &normalResponse);
      var finite = isFiniteBits(inverse00) && inverse00 >= 0.0;
      for (var column = 0u; column < ${i}u; column++) {
        finite = finite && isFiniteBits(normalResponse[column]);
      }
      if (finite) {
        code = 0u;
        // The centered intercept is the fitted value; uncentering it gives the raw intercept.
        fittedValue = normalResponse[0];
        hat = inverse00;
        var intercept = normalResponse[0];
        for (var column = 0u; column < ${e}u; column++) {
          coefficient[column + 1u] = normalResponse[column + 1u];
          intercept = intercept - normalResponse[column + 1u] * records[originBase + column];
        }
        coefficient[0] = intercept;
      }
    }
  }
  for (var column = 0u; column < ${i}u; column++) {
    fits[fitsOffset + index * RECORD_WORDS + column] = coefficient[column];
  }
  fits[fitsOffset + index * RECORD_WORDS + ${i}u] = fittedValue;
  fits[fitsOffset + index * RECORD_WORDS + ${i+1}u] = hat;
  fits[fitsOffset + index * RECORD_WORDS + ${i+2}u] = f32(code);`}var W=32768,G=16777216;function K(e,t=new Float32Array(4)){let n=Math.fround(e.bandwidth),r=Math.fround(e.timeScale);if(!(Number.isFinite(n)&&n>0))throw Error(`getGPUGTWRParameterValues bandwidth must be a positive finite number`);if(!(Number.isFinite(r)&&r>=0))throw Error(`getGPUGTWRParameterValues timeScale must be a non-negative finite number`);if(t.length<4)throw Error(`getGPUGTWRParameterValues target needs 4 elements`);return t.set([n,r,0,0]),t}var q=`GPUGeographicallyTemporallyWeightedRegression`,J=C,Y=class e{props;id;rows;estimatedCommandNodeCount;searchStructure;constructor(e){this.props=e,this.id=e.id??`geographically-temporally-weighted-regression`;let{id:t}=this,n=e.predictorCount,c=A(t,e.bandwidthSearch,e.output.status);a(t,e.kernel===void 0||[`gaussian`,`bisquare`].includes(e.kernel),`kernel must be gaussian or bisquare`),a(t,e.searchStructure===void 0||[`grid`,`brute-force`].includes(e.searchStructure),`searchStructure must be grid or brute-force`),e.gaussianCutoff!==void 0&&a(t,Number.isFinite(Math.fround(e.gaussianCutoff))&&Math.fround(e.gaussianCutoff)>0,`gaussianCutoff must be positive and finite in float32`);let l=e.kernel===`bisquare`||e.gaussianCutoff!==void 0;if(this.searchStructure=e.searchStructure??(l&&!e.bandwidthSearch?`grid`:`brute-force`),a(t,this.searchStructure!==`grid`||l&&!e.bandwidthSearch,`searchStructure grid requires a compact kernel and no bandwidthSearch`),this.estimatedCommandNodeCount=this.searchStructure===`grid`?256:3+Number(!!e.output.residuals)+(c?j(c,2)+Number(!!e.output.selectedParameters):0),s(t,this.estimatedCommandNodeCount),d(t,e.spatialContext,{coordinateSpaces:[`planar`],metrics:[`native`],measure:`distance`}),o(t,`predictorCount`,n,1,11),g(t,e.temporalContext,{encodings:[`float32-relative`]}),i(e.response,[`float32`],`${t} response`),this.rows=e.response.length,o(t,`rows`,this.rows,2,this.searchStructure===`grid`?G:W),i(e.positions,[`float32x2`],`${t} positions`),r(t,`positions`,e.positions,this.rows),i(e.times,[`float32`],`${t} times`),r(t,`times`,e.times,this.rows),i(e.predictors,[`float32`],`${t} predictors`),r(t,`predictors`,e.predictors,this.rows*n),e.mask&&(i(e.mask,[`uint32`],`${t} mask`),r(t,`mask`,e.mask,this.rows)),i(e.parameters,[`float32`],`${t} parameters`),e.parameters.length<4)throw Error(`${t} parameters must have at least 4 values`);let{output:f}=e;for(let[e,a,o,s]of[[`coefficients`,f.coefficients,`float32`,this.rows*(n+1)],[`localStatus`,f.localStatus,`uint32`,this.rows],[`fitted`,f.fitted,`float32`,this.rows],[`hatDiagonal`,f.hatDiagonal,`float32`,this.rows],[`summary`,f.summary,`float32`,12]])i(a,[o],`${t} output.${e}`),r(t,`output.${e}`,a,s);f.residuals&&(i(f.residuals,[`float32`],`${t} output.residuals`),r(t,`output.residuals`,f.residuals,this.rows)),f.selectedParameters&&(i(f.selectedParameters,[`float32`],`${t} output.selectedParameters`),a(t,!!e.bandwidthSearch,`output.selectedParameters requires bandwidthSearch`),a(t,f.selectedParameters.length>=2,`output.selectedParameters must have at least 2 values`)),u(t,this.getOutputViews(),this.getInputViews()),S(t,{...f,status:void 0,...Object.fromEntries(Object.entries(f.status??{}).map(([e,t])=>[`status.${e}`,t]))},[[`coefficients`,`fitted`,`hatDiagonal`,`localStatus`],{read:[`fitted`],write:[`residuals`]},...this.searchStructure===`brute-force`?[{read:[`fitted`,`hatDiagonal`,`localStatus`],write:[`summary`]}]:[],Object.keys(f.status??{}).map(e=>`status.${e}`)])}getInputViews(){let{props:e}=this;return[e.positions,e.times,e.predictors,e.response,e.mask,e.bandwidthSearch?void 0:e.parameters,e.bandwidthSearch?.parameters].filter(e=>e!==void 0)}getOutputViews(){let{status:e,...t}=this.props.output;return[...Object.values(t),...Object.values(e??{})].filter(e=>!!e)}getCommandNodes(e){return l(this.id,e,[...this.getInputViews(),...this.getOutputViews()]),f(e,this.id,{...Object.fromEntries(this.getInputViews().map((e,t)=>[`input[${t}]`,e])),...Object.fromEntries(this.getOutputViews().map((e,t)=>[`output[${t}]`,e]))}),c(e.device,this.id,`space-time and coefficient scratch`,this.rows*Math.max(3,this.props.predictorCount+1)*4,`reduce rows or predictorCount`),this.searchStructure===`grid`?(V(e.device,this.id,this.rows,this.props.predictorCount),H(e,this.id,this.props)):this.createNodes(e)}createNodes(r,i,a){let{id:o,props:s,rows:u}=this,{output:d}=s,f=s.predictorCount,g=f+1;if(l(o,r,[...this.getInputViews(),...this.getOutputViews()]),c(r.device,o,`space-time and coefficient scratch`,u*Math.max(3,g)*4,`reduce rows or predictorCount`),s.bandwidthSearch){let i=X(r,o,s,u),a=n(r,`${o}-selected`,`float32`,4),c=N(r,{id:o,rows:u,adaptive:s.adaptive??!1,search:s.bandwidthSearch,parameters:a,status:d.status,evaluate:(t,a,o)=>{let c={coefficients:n(r,`${o}-coefficients`,`float32`,u*g),localStatus:n(r,`${o}-status`,`uint32`,u),fitted:n(r,`${o}-fitted`,`float32`,u),hatDiagonal:n(r,`${o}-hat`,`float32`,u),summary:n(r,`${o}-summary`,`float32`,12)};return{nodes:new e({...s,id:o,bandwidthSearch:void 0,searchStructure:`brute-force`,parameters:t,output:c}).createNodes(r,a,i.points),summary:c.summary}}}),{status:l,selectedParameters:f,...p}=d,m=new e({...s,id:`${o}-selected-fit`,bandwidthSearch:void 0,searchStructure:`brute-force`,parameters:a,output:p});return f&&c.push(t(r,{id:`${o}-publish-selected`,operation:q,invocationCount:2,bindings:[{name:`selected`,view:a,type:`f32`,access:`read`},{name:`output`,view:f,type:`f32`,access:`read_write`}],body:`output[outputOffset + index] = selected[selectedOffset + index];`})),[...i.nodes,...c,...m.createNodes(r,void 0,i.points)]}let _=a?{points:a,nodes:[]}:X(r,o,s,u,i),v=_.points;return[..._.nodes,t(r,{id:`${o}-fit`,operation:q,variant:`local-fit`,invocationCount:u,...i?.getGate(0),bindings:[{name:`parameters`,view:s.parameters,type:`f32`,access:`read`},{name:`points`,view:v,type:`f32`,access:`read`},{name:`predictors`,view:s.predictors,type:`f32`,access:`read`},{name:`response`,view:s.response,type:`f32`,access:`read`},{name:`coefficients`,view:d.coefficients,type:`f32`,access:`read_write`},{name:`fitted`,view:d.fitted,type:`f32`,access:`read_write`},{name:`hatDiagonal`,view:d.hatDiagonal,type:`f32`,access:`read_write`},{name:`localStatus`,view:d.localStatus,type:`u32`,access:`read_write`}],declarations:`${p}
${h}
${m}
${b(g,g)}
${E(g)}
${T(s.gaussianCutoff??null)}
${s.adaptive?M(u):``}`,body:`let timeScale = parameters[parametersOffset + 1u];
  let bandwidth = ${s.adaptive?`getAdaptiveBandwidth(index, parameters[parametersOffset], timeScale)`:`parameters[parametersOffset]`};
  let originX = points[pointsOffset + 3u * index];
  var code = ${J.SINGULAR}u;
  var coefficient: array<f32, ${g}>;
  for (var column = 0u; column < ${g}u; column++) {
    coefficient[column] = getNaN();
  }
  var fittedValue = getNaN();
  var hat = getNaN();
  if (!isFiniteBits(originX)) {
    code = ${J.EXCLUDED}u;
  } else if (bandwidth > 0.0 && isFiniteBits(bandwidth) && timeScale >= 0.0 && isFiniteBits(timeScale)) {
    let originY = points[pointsOffset + 3u * index + 1u];
    let originTime = points[pointsOffset + 3u * index + 2u];
    let originBase = predictorsOffset + index * ${f}u;
    var a: array<f32, ${g*g}>;
    var b: array<f32, ${g}>;
    var z: array<f32, ${g}>;
    for (var entry = 0u; entry < ${g*g}u; entry++) { a[entry] = 0.0; }
    for (var column = 0u; column < ${g}u; column++) { b[column] = 0.0; }
    z[0] = 1.0;
    for (var row = 0u; row < ${u}u; row++) {
      let rowX = points[pointsOffset + 3u * row];
      if (!isFiniteBits(rowX)) { continue; }
      let deltaX = rowX - originX;
      let deltaY = points[pointsOffset + 3u * row + 1u] - originY;
      let deltaTime = points[pointsOffset + 3u * row + 2u] - originTime;
      let distance = sqrt(deltaX * deltaX + deltaY * deltaY + timeScale * deltaTime * deltaTime);
      let weight = getLocalKernelWeight(distance, bandwidth, ${s.kernel===`bisquare`?`true`:`false`});
      if (!(weight > 0.0)) { continue; }
      let rowBase = predictorsOffset + row * ${f}u;
      for (var column = 0u; column < ${f}u; column++) {
        z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
      }
      let observed = response[responseOffset + row];
      for (var r = 0u; r < ${g}u; r++) {
        let weighted = weight * z[r];
        b[r] = b[r] + weighted * observed;
        for (var c = 0u; c <= r; c++) {
          a[r * ${g}u + c] = a[r * ${g}u + c] + weighted * z[c];
        }
      }
    }
    var scale: array<f32, ${g}>;
    if (factorEquilibrated_${g}(&a, &scale)) {
      let inverse00 = inverseDiagonalEquilibrated_${g}(&a, &scale, 0u);
      // b is X'Wy of the centered design; the solve is in place and unscales the coefficients.
      solveEquilibrated_${g}(&a, &scale, &b);
      var finite = isFiniteBits(inverse00) && inverse00 >= 0.0;
      for (var column = 0u; column < ${g}u; column++) {
        finite = finite && isFiniteBits(b[column]);
      }
      if (finite) {
        code = ${J.OK}u;
        // The centered intercept is the fitted value; uncentering it gives the raw intercept.
        fittedValue = b[0];
        hat = inverse00;
        var intercept = b[0];
        for (var column = 0u; column < ${f}u; column++) {
          coefficient[column + 1u] = b[column + 1u];
          intercept = intercept - b[column + 1u] * predictors[originBase + column];
        }
        coefficient[0] = intercept;
      }
    }
  }
  for (var column = 0u; column < ${g}u; column++) {
    coefficients[coefficientsOffset + index * ${g}u + column] = coefficient[column];
  }
  fitted[fittedOffset + index] = fittedValue;
  hatDiagonal[hatDiagonalOffset + index] = hat;
  localStatus[localStatusOffset + index] = code;`}),...d.residuals?[t(r,{id:`${o}-residuals`,operation:q,variant:`residuals`,invocationCount:u,...i?.getGate(0),bindings:[{name:`response`,view:s.response,type:`f32`,access:`read`},{name:`fitted`,view:d.fitted,type:`f32`,access:`read`},{name:`residuals`,view:d.residuals,type:`f32`,access:`read_write`}],body:`residuals[residualsOffset + index] = response[responseOffset + index] - fitted[fittedOffset + index];`})]:[],t(r,{id:`${o}-summary`,operation:q,variant:`summary`,invocationCount:1,...i?.getGate(1),bindings:[{name:`response`,view:s.response,type:`f32`,access:`read`},{name:`fitted`,view:d.fitted,type:`f32`,access:`read`},{name:`hatDiagonal`,view:d.hatDiagonal,type:`f32`,access:`read`},{name:`localStatus`,view:d.localStatus,type:`u32`,access:`read`},{name:`summary`,view:d.summary,type:`f32`,access:`read_write`}],declarations:m,body:`var count = 0.0;
  var singular = 0.0;
  var included = 0.0;
  var mean = 0.0;
  var residualSquares = 0.0;
  var trace = 0.0;
  for (var row = 0u; row < ${u}u; row++) {
    let code = localStatus[localStatusOffset + row];
    if (code == ${J.EXCLUDED}u) { continue; }
    included += 1.0;
    if (code != ${J.OK}u) { singular += 1.0; continue; }
    count += 1.0;
    let observed = response[responseOffset + row];
    mean += (observed - mean) / count;
    let residual = observed - fitted[fittedOffset + row];
    residualSquares += residual * residual;
    trace += hatDiagonal[hatDiagonalOffset + row];
  }
  var totalSquares = 0.0;
  for (var row = 0u; row < ${u}u; row++) {
    if (localStatus[localStatusOffset + row] == ${J.OK}u) {
      let deviation = response[responseOffset + row] - mean;
      totalSquares += deviation * deviation;
    }
  }
  // Work in log space: 2 pi * RSS can overflow even when RSS and 2 pi * RSS / n are finite.
  let logVariance = log(residualSquares) - log(count);
  let logLikelihood = -0.5 * count * (logVariance + 1.8378770664093453 + 1.0);
  // The AICc of GPUGeographicallyWeightedRegression (mgwr): -2 llf + 2 n (tr + 1) / (n - tr - 2),
  // defined only while n - tr - 2 > 0. mgtwr's own aicc is 2 (tr + 1) (n - 1) / (n - tr - 1) in
  // the penalty and is not published here.
  let correctedDenominator = count - trace - 2.0;
  let corrected = -2.0 * logLikelihood + 2.0 * count * (trace + 1.0) / correctedDenominator;
  for (var slot = 0u; slot < 12u; slot++) {
    summary[summaryOffset + slot] = getNaN();
  }
  summary[summaryOffset] = residualSquares;
  summary[summaryOffset + 1u] = trace;
  summary[summaryOffset + 2u] = select(getNaN(), corrected, count > 0.0 && correctedDenominator > 0.0);
  summary[summaryOffset + 3u] = select(getNaN(), 1.0 - residualSquares / totalSquares, totalSquares > 0.0);
  summary[summaryOffset + 4u] = count;
  summary[summaryOffset + 6u] = singular;
  summary[summaryOffset + 7u] = select(getNaN(), residualSquares / (count - trace), count > trace);
  summary[summaryOffset + 10u] = select(getNaN(), logLikelihood, count > 0.0);
  summary[summaryOffset + 11u] = included;`})]}};function X(e,r,i,a,o){let s=i.predictorCount,c=n(e,`${r}-points`,`float32`,a*3);return{nodes:[t(e,{id:`${r}-points`,operation:q,variant:`usable-rows`,invocationCount:a,...o?.getGate(0),bindings:[{name:`positions`,view:i.positions,type:`f32`,access:`read`},{name:`times`,view:i.times,type:`f32`,access:`read`},{name:`predictors`,view:i.predictors,type:`f32`,access:`read`},{name:`response`,view:i.response,type:`f32`,access:`read`},...i.mask?[{name:`mask`,view:i.mask,type:`u32`,access:`read`}]:[],{name:`points`,view:c,type:`f32`,access:`read_write`}],declarations:`${p}\n${m}`,body:`var valid = ${i.mask?`mask[maskOffset + index] != 0u`:`true`};
  let x = positions[positionsOffset + 2u * index];
  let y = positions[positionsOffset + 2u * index + 1u];
  let time = times[timesOffset + index];
  valid = valid && isFiniteBits(x) && isFiniteBits(y) && isFiniteBits(time) && isFiniteBits(response[responseOffset + index]);
  for (var column = 0u; column < ${s}u; column++) {
    valid = valid && isFiniteBits(predictors[predictorsOffset + index * ${s}u + column]);
  }
  points[pointsOffset + 3u * index] = select(getNaN(), x, valid);
  points[pointsOffset + 3u * index + 1u] = y;
  points[pointsOffset + 3u * index + 2u] = time;`})],points:c}}var Z={unit:`second`,epoch:0n,encoding:`float32-relative`,axis:{kind:`irregular`}},Q=10,$=.5;function te(e){return 1/(1+Math.exp(-e))}function ne(e,t){return .3+2.8*te(-(e-(.2+.2*t))/.1)}function re(e,t){return .8+.8*e+.15*t}function ie(e){let t=0;for(let n=0;n<6;n++)t+=e();return(t-3)/Math.sqrt(.5)}function ae(e,t,n){let{tileColumns:r,tileRows:i,periodsFillGrid:a}=n,o=t.width/r,s=n.bottomMargin??0,c=(t.height-s)/i,l=k(e,{width:o,height:c},Q),u=l.count,d=u*4,f=new Float32Array(d*2),p=new Float32Array(d),m=new Float32Array(d),h=new Float32Array(d),g=new Float32Array(d),_=new Float32Array(d);for(let t=0;t<4;t++)for(let n=0;n<u;n++){let r=t*u+n,i=l.centers[n*2],a=l.centers[n*2+1];f.set([i,a],r*2),p[r]=t,m[r]=l.values[n],g[r]=re(a/c,t),_[r]=ne(i/o,t),h[r]=g[r]+_[r]*m[r]+$*ie(e)}let v=[],y=[],b=a?1:i,x=a?r*i:r;for(let e=0;e<b;e++){let n=new Float32Array(x*l.vertices.length),i=new Uint32Array(x*l.triangleIndices.length),s=new Uint32Array(x*l.triangleOwners.length),d=new Float32Array(x*l.outline.length),f=l.vertices.length/2;for(let p=0;p<x;p++){let m=p,h=a?Math.floor(p/r):e,g=p%r*o,_=t.height-(h+1)*c;y.push([g+o/2,_+c/2]);for(let e=0;e<l.vertices.length;e+=2)n[p*l.vertices.length+e]=l.vertices[e]+g,n[p*l.vertices.length+e+1]=l.vertices[e+1]+_;for(let e=0;e<l.triangleIndices.length;e++)i[p*l.triangleIndices.length+e]=l.triangleIndices[e]+p*f;for(let e=0;e<l.triangleOwners.length;e++)s[p*l.triangleOwners.length+e]=l.triangleOwners[e]+m*u;for(let e=0;e<l.outline.length;e+=4){let t=p*l.outline.length+e;d[t]=l.outline[e]+g,d[t+1]=l.outline[e+1]+_,d[t+2]=l.outline[e+2]+g,d[t+3]=l.outline[e+3]+_}}v.push({vertices:n,indices:i,owners:s,triangleCount:x*l.triangleOwners.length,outline:d,outlineCount:x*l.outlineCount})}return{tracts:l,tractCount:u,rowCount:d,tileColumns:r,tileRows:i,tileWidth:o,tileHeight:c,positions:f,times:p,predictors:m,response:h,trueIntercept:g,trueSlope:_,meshes:v,tileCenters:y}}function oe(e,t){return{positions:e.upload(`positions`,`float32x2`,t.positions),times:e.upload(`times`,`float32`,t.times),predictors:e.upload(`predictors`,`float32`,t.predictors),response:e.upload(`response`,`float32`,t.response)}}function se(e,t,n){let r=t.meshes[n];return{vertices:e.upload(`tile-vertices-${n}`,`float32x2`,r.vertices),indices:e.upload(`tile-indices-${n}`,`uint32`,r.indices),owners:e.upload(`tile-owners-${n}`,`uint32`,r.owners),outline:e.upload(`tile-outline-${n}`,`float32x4`,r.outline)}}function ce(e,t,n,r,i){return[{layer:`polygons`,positions:n.vertices,indices:n.indices,owners:n.owners,values:r,triangleCount:e.meshes[t].triangleCount,ramp:`sequential`,range:i,bands:10},{layer:`lines`,context:!0,segments:n.outline,count:e.meshes[t].outlineCount,alpha:.5}]}export{oe as a,X as c,ee as d,M as f,se as i,K as l,A as m,ae as n,Y as o,j as p,ce as r,W as s,Z as t,N as u};