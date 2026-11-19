import{r as e}from"./kernel-C3tfQOn4.js";import{s as t}from"./raster-grid-VB0AM_GY.js";import{a as n}from"./wgsl-common-DfyXiYB0.js";import{c as r,o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{n as a}from"./terrain-illumination-utils-CoEB4UZd.js";import{a as o,i as s,n as c,r as l}from"./terrain-summed-area-table-BJldJnOP.js";function u(e){let t=Math.floor(e),n=e-t;return n<.5?t:n>.5?t+1:t%2==0?t:t+1}function d(t,n,r){return{anchors:e(t,`${n}-mean-anchors`,`float32`,r),rowSums:e(t,`${n}-mean-row-sums`,`float32`,r),rowCounts:e(t,`${n}-mean-row-counts`,`uint32`,r)}}function f(e,t){let n=b(t),{width:r,height:o,radius:s,scratch:c,groundRadius:l}=t,u=r*o,d=`const WIDTH: u32 = ${r}u;
const HEIGHT: u32 = ${o}u;
const RADIUS: i32 = ${s};${n}`,{radiusSlot:f}=t,p=f?`let radiusWord = radiusSettings[radiusSettingsOffset + ${f.index}u];
  let radius = select(RADIUS, clamp(i32(radiusWord), 1, RADIUS), radiusWord >= 1.0);
  `:``,m=f||l?`radius`:`RADIUS`,h=l?`-${l.mode}`:f?`-radius`:``,g=f?[{name:`radiusSettings`,view:f.settings,type:`f32`,access:`read`}]:[],_=i(e,{id:`${t.id}-rows`,operation:t.operation,variant:`mean-rows${h}`,bindings:[{name:`values`,view:t.values,type:`f32`,access:`read`},{name:`validity`,view:t.validity,type:`u32`,access:`read`},{name:`anchors`,view:c.anchors,type:`f32`,access:`read_write`},{name:`rowSums`,view:c.rowSums,type:`f32`,access:`read_write`},{name:`rowCounts`,view:c.rowCounts,type:`u32`,access:`read_write`},...g],invocationCount:u,declarations:d,body:`let column = i32(index % WIDTH);
  let rowStart = index - u32(column);
  ${l?`let radius = getReliefPixelRadius(index / WIDTH).x;`:p}
  var anchor = 0.0;
  if (validity[validityOffset + index] != 0u) {
    anchor = values[valuesOffset + index];
  } else {
    for (var offset = -${m}; offset <= ${m}; offset++) {
      let sampleIndex = rowStart + u32(clamp(column + offset, 0, i32(WIDTH) - 1));
      if (validity[validityOffset + sampleIndex] != 0u) {
        anchor = values[valuesOffset + sampleIndex];
        break;
      }
    }
  }
  var rowSum = 0.0;
  var rowCount = 0u;
  for (var offset = -${m}; offset <= ${m}; offset++) {
    let sampleIndex = rowStart + u32(clamp(column + offset, 0, i32(WIDTH) - 1));
    if (validity[validityOffset + sampleIndex] != 0u) {
      rowSum += values[valuesOffset + sampleIndex] - anchor;
      rowCount += 1u;
    }
  }
  anchors[anchorsOffset + index] = anchor;
  rowSums[rowSumsOffset + index] = rowSum;
  rowCounts[rowCountsOffset + index] = rowCount;`}),v=[...g,{name:`values`,view:t.values,type:`f32`,access:`read`},{name:`validity`,view:t.validity,type:`u32`,access:`read`},{name:`anchors`,view:c.anchors,type:`f32`,access:`read`},{name:`rowSums`,view:c.rowSums,type:`f32`,access:`read`},{name:`rowCounts`,view:c.rowCounts,type:`u32`,access:`read`},{name:`deviation`,view:t.deviation,type:`f32`,access:`read_write`}],y=t.scale&&f&&t.scale.settings===f.settings?`radiusSettings`:`scaleSettings`;return t.scale&&y===`scaleSettings`&&v.push({name:`scaleSettings`,view:t.scale.settings,type:`f32`,access:`read`}),t.validityOutput&&v.push({name:`validityOutput`,view:t.validityOutput,type:`u32`,access:`read_write`}),[_,i(e,{id:`${t.id}-columns`,operation:t.operation,variant:`mean-columns${h}`,bindings:v,invocationCount:u,declarations:`${d}\n${a}`,body:`let column = index % WIDTH;
  let row = i32(index / WIDTH);
  ${l?`let radius = getReliefPixelRadius(u32(row)).y;`:p}
  let isValid = validity[validityOffset + index] != 0u;
  var deviationValue = getQuietNaN(index);
  if (isValid) {
    let center = values[valuesOffset + index];
    var total = 0.0;
    var count = 0u;
    for (var offset = -${m}; offset <= ${m}; offset++) {
      let sampleIndex = u32(clamp(row + offset, 0, i32(HEIGHT) - 1)) * WIDTH + column;
      let rowCount = rowCounts[rowCountsOffset + sampleIndex];
      total += rowSums[rowSumsOffset + sampleIndex] +
        f32(rowCount) * (anchors[anchorsOffset + sampleIndex] - center);
      count += rowCount;
    }
    deviationValue = -total / f32(count) ${t.scale?`* ${y}[${y}Offset + ${t.scale.index}u]`:``};
  }
  deviation[deviationOffset + index] = deviationValue;
  ${t.validityOutput?`validityOutput[validityOutputOffset + index] = select(0u, 1u, isValid);`:``}`})]}var p=2**-16,m=23170;function h(e,t){return t<=o(e)}function g(e,t){if(!(t>0)||!Number.isFinite(t)||2**Math.round(Math.log2(t))!==t)throw Error(`${e} quantum must be a positive power of two`)}function _(e,t,n,r){if(h(t,n)&&r>23170)throw Error(`${e} radius must be at most ${m}`)}function v(e,t){let{id:n,width:r,height:i,band:a}=t,o=r*i,c=a.storage.values;if(!h(e.device,o)){let t=d(e,n,o),s=a.validity;return{method:`separable`,nodes:[],getFilterNodes:n=>f(e,{...n,width:r,height:i,values:c,validity:s,scratch:t})}}let l=s(e,{id:n,width:r,height:i,elevation:a,quantum:t.quantum,includeSquares:!1,originRelative:!0});return{method:`summed-area`,nodes:l.nodes,getFilterNodes:n=>y(e,{...n,width:r,height:i,values:c,table:l,quantum:t.quantum})}}function y(e,t){let a=b(t),{width:o,height:s,radius:u,radiusSlot:d,groundRadius:f}=t,p=f?`radiusX`:`radius`,m=f?`radiusY`:`radius`,h=[...t.table.anchor?[{name:`anchor`,view:t.table.anchor,type:`f32`,access:`read`}]:[],{name:`lowTable`,view:t.table.lowTable,type:`u32`,access:`read`},{name:`highTable`,view:t.table.highTable,type:`u32`,access:`read`},{name:`values`,view:t.values,type:`f32`,access:`read`},{name:`deviation`,view:t.deviation,type:`f32`,access:`read_write`}];d&&h.push({name:`radiusSettings`,view:d.settings,type:`f32`,access:`read`});let g=t.scale&&d&&t.scale.settings===d.settings?`radiusSettings`:`scaleSettings`;t.scale&&g===`scaleSettings`&&h.push({name:`scaleSettings`,view:t.scale.settings,type:`f32`,access:`read`}),t.validityOutput&&h.push({name:`validityOutput`,view:t.validityOutput,type:`u32`,access:`read_write`});let _=d?`let radiusWord = radiusSettings[radiusSettingsOffset + ${d.index}u];
  let radius = select(RADIUS, clamp(i32(radiusWord), 1, RADIUS), radiusWord >= 1.0);`:`let radius = RADIUS;`;return[i(e,{id:`${t.id}-deviation`,operation:t.operation,variant:`mean-summed-area${f?`-${f.mode}`:d?`-radius`:``}`,bindings:h,invocationCount:o*s,declarations:`const WIDTH: u32 = ${o}u;
const HEIGHT: u32 = ${s}u;
const PIXEL_COUNT: u32 = ${o*s}u;
const RADIUS: i32 = ${u};${a}
const QUANTUM: f32 = ${r(t.quantum)};
const INVERSE_QUANTUM: f32 = ${r(1/t.quantum)};
${l}
${c}
// Runtime operand so the NaN is not constant-folded away.
${n}// total + weight * strip, modulo 2^64 for the sum.
fn addWeighted(total: TerrainBoxSums, strip: TerrainBoxSums, weight: u32) -> TerrainBoxSums {
  return TerrainBoxSums(
    add64(total.sum, multiply64By32(strip.sum, weight)),
    total.count + strip.count * weight
  );
}`,body:`let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  ${f?`let pixelRadius = getReliefPixelRadius(u32(row));
  let radiusX = pixelRadius.x;
  let radiusY = pixelRadius.y;`:_}
  // A single-cell box returns the centre's quantized elevation and validity exactly.
  let centre = readBoxSums(column, column, row, row);
  let isValid = centre.count == 1u;
  var deviationValue = getQuietNaN(index);
  if (isValid) {
    let firstColumn = max(column - ${p}, 0i);
    let lastColumn = min(column + ${p}, i32(WIDTH) - 1i);
    let firstRow = max(row - ${m}, 0i);
    let lastRow = min(row + ${m}, i32(HEIGHT) - 1i);
    let overhangLeft = u32(max(${p} - column, 0i));
    let overhangRight = u32(max(column + ${p} - (i32(WIDTH) - 1i), 0i));
    let overhangTop = u32(max(${m} - row, 0i));
    let overhangBottom = u32(max(row + ${m} - (i32(HEIGHT) - 1i), 0i));
    var total = readBoxSums(firstColumn, lastColumn, firstRow, lastRow);
    if (overhangLeft > 0u) {
      total = addWeighted(total, readBoxSums(0i, 0i, firstRow, lastRow), overhangLeft);
    }
    if (overhangRight > 0u) {
      total = addWeighted(total, readBoxSums(i32(WIDTH) - 1i, i32(WIDTH) - 1i, firstRow, lastRow), overhangRight);
    }
    if (overhangTop > 0u) {
      total = addWeighted(total, readBoxSums(firstColumn, lastColumn, 0i, 0i), overhangTop);
      if (overhangLeft > 0u) {
        total = addWeighted(total, readBoxSums(0i, 0i, 0i, 0i), overhangLeft * overhangTop);
      }
      if (overhangRight > 0u) {
        total = addWeighted(total, readBoxSums(i32(WIDTH) - 1i, i32(WIDTH) - 1i, 0i, 0i), overhangRight * overhangTop);
      }
    }
    if (overhangBottom > 0u) {
      total = addWeighted(total, readBoxSums(firstColumn, lastColumn, i32(HEIGHT) - 1i, i32(HEIGHT) - 1i), overhangBottom);
      if (overhangLeft > 0u) {
        total = addWeighted(total, readBoxSums(0i, 0i, i32(HEIGHT) - 1i, i32(HEIGHT) - 1i), overhangLeft * overhangBottom);
      }
      if (overhangRight > 0u) {
        total = addWeighted(total, readBoxSums(i32(WIDTH) - 1i, i32(WIDTH) - 1i, i32(HEIGHT) - 1i, i32(HEIGHT) - 1i), overhangRight * overhangBottom);
      }
    }
    // D = N * q_c - S, exact modulo 2^64 and |D| < 2^63.
    let quantized = bitcast<i32>(centre.sum.x);
    var centreTotal = multiply64By32(vec2<u32>(total.count, 0u), u32(abs(quantized)));
    if (quantized < 0i) {
      centreTotal = negate64(centreTotal);
    }
    let difference = subtract64(centreTotal, total.sum);
    // The centre's own quantization residual is exact: scaling by a power of two does not round.
    let residual = ((values[valuesOffset + index]${t.table.anchor?` - anchor[anchorOffset]`:``}) * INVERSE_QUANTUM - f32(quantized)) * QUANTUM;
    deviationValue = residual + signed64ToFloat(difference) / f32(total.count) * QUANTUM;
    ${t.scale?`deviationValue = deviationValue * ${g}[${g}Offset + ${t.scale.index}u];`:``}
  }
  deviation[deviationOffset + index] = deviationValue;
  ${t.validityOutput?`validityOutput[validityOutputOffset + index] = select(0u, 1u, isValid);`:``}`})]}function b(e){let{groundRadius:n,radiusSlot:i}=e;if(!n)return``;if(i)throw Error(`groundRadius cannot use radiusSlot`);let{cellSize:a,northEdge:o,southEdge:s,radius:c,maximumRadius:l,mode:u}=n;return`
const NORTH_EDGE: f32 = ${r(o)};
const SOUTH_EDGE: f32 = ${r(s)};
const GROUND_RADIUS: f32 = ${r(c)};
const MAXIMUM_RADIUS_X: i32 = ${l[0]};
const MAXIMUM_RADIUS_Y: i32 = ${l[1]};
fn getReliefGroundCell(row: u32) -> vec2<f32> {
  let cellSize = vec2<f32>(${r(a[0])}, ${r(a[1])});
  let edge = mix(NORTH_EDGE, SOUTH_EDGE, (f32(row) + 0.5) / f32(HEIGHT));
  ${t(u)}
  return cellSize * groundScale;
}
fn getReliefPixelRadius(row: u32) -> vec2<i32> {
  let rounded = vec2<i32>(round(vec2<f32>(GROUND_RADIUS) / getReliefGroundCell(row)));
  return clamp(rounded, vec2<i32>(1), vec2<i32>(MAXIMUM_RADIUS_X, MAXIMUM_RADIUS_Y));
}`}export{g as a,h as i,v as n,_ as o,u as r,p as t};