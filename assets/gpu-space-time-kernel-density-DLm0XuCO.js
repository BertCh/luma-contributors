import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-reduction-tNgOVQ_C.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{E as r,S as i,T as a,f as o,g as s,o as c,u as l,x as u,y as d}from"./gpu-contributor-utils-BIwLQON5.js";import{c as f,o as p}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as m,v as h}from"./spatial-context-DwZEd4EZ.js";import{i as g,n as _,u as v}from"./wgsl-common-DfyXiYB0.js";import{u as y}from"./raster-band-utils-BcTxc8_Y.js";import{n as b}from"./gpu-inverse-distance-weighting-KIb4SzJG.js";import{t as x}from"./grid-cell-row-order-DUF05y4e.js";import{n as S}from"./point-neighbor-grid-index-D0wS1y1I.js";import{i as C}from"./space-time-bands-Lbr2KxkP.js";import{i as w,n as T,r as E}from"./gpu-kernel-density-ZcuIBWnT.js";import{_ as D,b as O,h as k,m as A,v as j,x as M,y as N}from"./density-scratch-D0AtmvQz.js";function P(e,t,n,r=new Float32Array(14)){let i=`getGPUSpaceTimeKernelDensityParameterValues`;if(r.length<14)throw Error(`${i} target must hold 14 elements`);if(e.window===`study-window`)throw Error(`${i} does not support a polygon study window`);if(!Number.isFinite(e.temporalBandwidth)||e.temporalBandwidth<=0)throw Error(`${i} temporalBandwidth must be positive and finite`);if(!Number.isFinite(e.firstSliceTime))throw Error(`${i} firstSliceTime must be finite`);if(!Number.isFinite(e.sliceInterval)||e.sliceInterval<=0)throw Error(`${i} sliceInterval must be positive and finite`);o(i,`sliceCount`,n,1,2**53-1);let a=e.timeWindow??[e.firstSliceTime,e.firstSliceTime+(n-1)*e.sliceInterval],s=e.timeWindow===void 0&&n===1;if(a.length!==2||!a.every(Number.isFinite)||!(a[0]<a[1]||s))throw Error(`${i} timeWindow must be finite [start, end] with start < end`);return T(e,t,r),r[9]=e.temporalBandwidth,r[10]=e.firstSliceTime,r[11]=e.sliceInterval,r[12]=a[0],r[13]=a[1],r}function F(e){if(e===`gaussian`){let e=f(1/Math.sqrt(2*Math.PI));return`
fn getTemporalKernelValue(offset: f32, bandwidth: f32) -> f32 {
  let s = offset / bandwidth;
  let t = s * s;
  if (!(t < ${f(25)})) { return 0.0; }
  return ${e} * exp(-0.5 * t) / bandwidth;
}`}return`
fn getTemporalKernelValue(offset: f32, bandwidth: f32) -> f32 {
  let s = offset / bandwidth;
  let t = s * s;
  if (!(t < 1.0)) { return 0.0; }
  let complement = 1.0 - t;
  return ${f(e===`epanechnikov`?.75:.9375)} * ${e===`epanechnikov`?`complement`:`complement * complement`} / bandwidth;
}`}function I(e,t){return e===`gaussian`?`
${t?_:``}
fn getTemporalUpperTail(z: f32) -> f32 { return 0.5 * erfcValue(z * 0.70710678); }
fn getTemporalNormalMass(low: f32, high: f32) -> f32 {
  if (low > 0.0) { return getTemporalUpperTail(low) - getTemporalUpperTail(high); }
  if (high < 0.0) { return getTemporalUpperTail(-high) - getTemporalUpperTail(-low); }
  return 1.0 - getTemporalUpperTail(-low) - getTemporalUpperTail(high);
}
fn getTemporalWindowMass(time: f32, windowStart: f32, windowEnd: f32, bandwidth: f32) -> f32 {
  return getTemporalNormalMass((windowStart - time) / bandwidth, (windowEnd - time) / bandwidth);
}`:`
fn getTemporalCumulative(s: f32) -> f32 {
  let c = clamp(s, -1.0, 1.0);
  return ${e===`epanechnikov`?`0.5 + 0.75 * (c - c * c * c / 3.0)`:`0.5 + 0.9375 * (c - 2.0 * c * c * c / 3.0 + c * c * c * c * c / 5.0)`};
}
fn getTemporalWindowMass(time: f32, windowStart: f32, windowEnd: f32, bandwidth: f32) -> f32 {
  return getTemporalCumulative((windowEnd - time) / bandwidth) -
    getTemporalCumulative((windowStart - time) / bandwidth);
}`}var L=`GPUSpaceTimeKernelDensity`,R=Object.freeze({...m,measure:`area`}),z=4294967295,B=class{id;props;spatialContext;temporalContext;spatialKernel;temporalKernel;spatialEdgeCorrection;temporalEdgeCorrection;normalization;voxelCount;constructor(e){this.id=e.id??`space-time-kernel-density`,this.props=e;let{id:t}=this;this.spatialContext=h(t,e.spatialContext,R),this.temporalContext=C(t,e.temporalContext),this.spatialKernel=e.spatialKernel===`bisquare`?`quartic`:e.spatialKernel??`gaussian`,this.temporalKernel=e.temporalKernel??`gaussian`,this.spatialEdgeCorrection=e.spatialEdgeCorrection??`uniform`,this.temporalEdgeCorrection=e.temporalEdgeCorrection??`uniform`,this.normalization=e.normalization??`intensity`,a(t,[`gaussian`,`quartic`,`epanechnikov`].includes(this.spatialKernel),`spatialKernel must be gaussian, quartic (alias bisquare), or epanechnikov`),a(t,[`gaussian`,`quartic`,`epanechnikov`].includes(this.temporalKernel),`temporalKernel must be gaussian, quartic, or epanechnikov`),a(t,[`none`,`uniform`].includes(this.spatialEdgeCorrection)&&[`none`,`uniform`].includes(this.temporalEdgeCorrection),`edge corrections must be 'none' or 'uniform'`),a(t,[`intensity`,`probability`].includes(this.normalization),`normalization must be intensity or probability`);let n=y(t,e.grid);a(t,e.grid.affine[1]===0&&e.grid.affine[3]===0,`needs an axis-aligned grid`),o(t,`sliceCount`,e.sliceCount,1,A),this.voxelCount=n*e.sliceCount;for(let[n,r]of[[`indexGridSize[0]`,e.indexGridSize[0]],[`indexGridSize[1]`,e.indexGridSize[1]]])o(t,n,r,1,A);for(let[n,r]of e.indexBounds.entries())s(t,`indexBounds[${n}]`,r);let[i,c,l,u]=e.indexBounds;a(t,e.indexBounds.length===4&&e.indexBounds.every(Number.isFinite)&&i<=l&&c<=u,`indexBounds must be finite and ordered`),r(e.positions,[`float32x2`],`${t} positions`),r(e.times,[`float32`],`${t} times`),o(t,`positions length`,e.positions.length,1,k),a(t,e.times.length===e.positions.length,`times length must equal positions length`),e.weights&&(r(e.weights,[`float32`],`${t} weights`),a(t,e.weights.length===e.positions.length,`weights length must equal positions length`)),r(e.parameters,[`float32`],`${t} parameters`),a(t,e.parameters.length>=14,`parameters must hold 14 float32 values`);let{output:f}=e;r(f.values,[`float32`],`${t} output.values`),a(t,f.values.length>=this.voxelCount,`output.values must hold grid.width * grid.height * sliceCount rows`);for(let[e,n,i]of[[`output.weightSum`,f.weightSum,`float32`],[`output.status.invalidCount`,f.status?.invalidCount,`uint32`]])n&&(r(n,[i],`${t} ${e}`),a(t,n.length===1,`${e} must contain exactly one row`));let p=[f.values,f.weightSum,f.status?.invalidCount].filter(e=>e!==void 0);M(t,{values:f.values,weightSum:f.weightSum,invalidCount:f.status?.invalidCount}),d(t,p,[e.positions,e.times,e.weights,e.parameters])}getCommandNodes(r){V(r,this);let{id:i,props:a,spatialKernel:o,temporalKernel:s,normalization:c}=this,{grid:l,indexBounds:u,indexGridSize:d,sliceCount:m}=a,h=D(r,i,L,{values:a.output.values,weightSum:a.output.weightSum,invalidCount:a.output.status?.invalidCount},{values:this.voxelCount}),_={...h.views,status:{invalidCount:h.views.invalidCount}},y=a.positions.length,{width:C,height:T}=l,O=this.spatialEdgeCorrection===`uniform`,k=this.temporalEdgeCorrection===`uniform`,A=`const DOMAIN_MIN: vec2f = vec2f(${f(u[0])}, ${f(u[1])});
const DOMAIN_MAX: vec2f = vec2f(${f(u[2])}, ${f(u[3])});`,j=(t,n)=>e(r,`${i}-${t}`,`uint32`,Math.max(n,1)),M=(t,n)=>e(r,`${i}-${t}`,`float32`,Math.max(n,1)),N=[],P=j(`keys`,y),R=j(`row-ids`,y),B=M(`included-weights`,y),H=_.status?.invalidCount?j(`excluded`,y):void 0;N.push(p(r,{id:`${i}-row-keys`,operation:L,variant:`row-keys-${this.spatialEdgeCorrection}-${this.temporalEdgeCorrection}`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`times`,view:a.times,type:`f32`,access:`read`},...a.weights?[{name:`inputWeights`,view:a.weights,type:`f32`,access:`read`}]:[],{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`keys`,view:P,type:`u32`,access:`read_write`},{name:`rowIds`,view:R,type:`u32`,access:`read_write`},{name:`includedWeights`,view:B,type:`f32`,access:`read_write`},...H?[{name:`excluded`,view:H,type:`u32`,access:`read_write`}]:[]],invocationCount:y,declarations:`${A}
${v}
${g}`,body:`let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let time = times[timesOffset + index];
  let weight = ${a.weights?`inputWeights[inputWeightsOffset + index]`:`1.0`};
  var included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(time) && isFiniteValue(weight) && weight >= 0.0;
  ${O?`let relative = position - gridStart;
  included = included && all(relative >= windowMin) && all(relative <= windowMax);`:``}
  ${k?`included = included && time >= params[paramsOffset + 12u] && time <= params[paramsOffset + 13u];`:``}
  keys[keysOffset + index] = select(${z}u, getOrderedKey(time), included);
  rowIds[rowIdsOffset + index] = index;
  includedWeights[includedWeightsOffset + index] = select(0.0, weight, included);
  ${H?`excluded[excludedOffset + index] = select(1u, 0u, included);`:``}`}));let U=j(`sorted-keys`,y),W=j(`time-order`,y);N.push(...new n({id:`${i}-time-sort`,keys:P,values:R,outputKeys:U,outputValues:W,keyBits:32}).getCommandNodes(r));let G=e(r,`${i}-sorted-positions`,`float32x2`,y);N.push(p(r,{id:`${i}-sorted-positions`,operation:L,variant:`sorted-positions`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`timeOrder`,view:W,type:`u32`,access:`read`},{name:`sortedKeys`,view:U,type:`u32`,access:`read`},{name:`sortedPositions`,view:G,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`fn getNaNBits() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`,body:`let row = timeOrder[timeOrderOffset + index];
  let kept = sortedKeys[sortedKeysOffset + index] != ${z}u;
  let nan = getNaNBits();
  sortedPositions[sortedPositionsOffset + 2u * index] = select(nan, positions[positionsOffset + 2u * row], kept);
  sortedPositions[sortedPositionsOffset + 2u * index + 1u] = select(nan, positions[positionsOffset + 2u * row + 1u], kept);`}));let K=S(r,{id:i,indexId:`${i}-index`,positions:G,gridSize:d,bounds:u,rowCount:y}),{cellOffsets:q,sortedRows:J,gridCount:Y}=K;N.push(...K.nodes);let X=x(r,{id:`${i}-cell-row-order`,operation:L,cellOffsets:q,objectIds:J,indexCount:Y,rowCount:y});N.push(...X.nodes);let Z=M(`slot-points`,4*y);N.push(p(r,{id:`${i}-slot-points`,operation:L,variant:`slot-points`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`times`,view:a.times,type:`f32`,access:`read`},{name:`includedWeights`,view:B,type:`f32`,access:`read`},{name:`timeOrder`,view:W,type:`u32`,access:`read`},{name:`sortedRows`,view:X.sortedRows,type:`u32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`slotPoints`,view:Z,type:`f32`,access:`read_write`}],invocationCount:y,body:`let row = timeOrder[timeOrderOffset + sortedRows[sortedRowsOffset + index]];
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let position = vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
  let relative = position - gridStart;
  slotPoints[slotPointsOffset + 4u * index] = relative.x;
  slotPoints[slotPointsOffset + 4u * index + 1u] = relative.y;
  slotPoints[slotPointsOffset + 4u * index + 2u] = times[timesOffset + row];
  slotPoints[slotPointsOffset + 4u * index + 3u] = includedWeights[includedWeightsOffset + row];`}));let Q=c===`probability`||_.weightSum?_.weightSum??M(`weight-sum-scratch`,1):void 0;Q&&N.push(...new t({id:`${i}-weight-sum`,input:B,output:Q,operation:`sum`}).getCommandNodes(r)),H&&_.status?.invalidCount&&N.push(...new t({id:`${i}-invalid-count`,input:H,output:_.status.invalidCount,operation:`sum`}).getCommandNodes(r));let $=o===`gaussian`?5:1,ee=s===`gaussian`?5:1,te=!(O&&o===`gaussian`);return N.push(p(r,{id:`${i}-gather`,operation:L,variant:`gather-${o}-${s}-${this.spatialEdgeCorrection}-${this.temporalEdgeCorrection}-${c}`,bindings:[{name:`slotPoints`,view:Z,type:`f32`,access:`read`},{name:`cellOffsets`,view:q,type:`u32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},...c===`probability`&&Q?[{name:`weightSum`,view:Q,type:`f32`,access:`read`}]:[],{name:`valuesOut`,view:_.values,type:`f32`,access:`read_write`}],invocationCount:this.voxelCount,workgroupSize:64,declarations:`const WIDTH: u32 = ${C}u;
const HEIGHT: u32 = ${T}u;
const SLICES: u32 = ${m}u;
const INDEX_WIDTH: u32 = ${d[0]}u;
const INDEX_HEIGHT: u32 = ${d[1]}u;
const SPATIAL_FACTOR: f32 = ${f($)};
const TEMPORAL_FACTOR: f32 = ${f(ee)};
${A}
${b}
${E(o)}
${F(s)}
${O?w(o):``}
${k?I(s,te):``}`,body:`let slice = index % SLICES;
  let cell = index / SLICES;
  let column = cell % WIDTH;
  let rasterRow = cell / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = params[paramsOffset + 4u];
  let temporalBandwidth = params[paramsOffset + 9u];
  let sliceTime = params[paramsOffset + 10u] + f32(slice) * params[paramsOffset + 11u];
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let radius = SPATIAL_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  let temporalRadius = TEMPORAL_FACTOR * temporalBandwidth;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  var valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(radius) &&
    isFiniteValue(temporalBandwidth) && temporalBandwidth > 0.0 && isFiniteValue(temporalRadius) &&
    isFiniteValue(sliceTime) && all(gridEnd != gridStart);
  var total = 0.0;
  if (valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    let timeLow = sliceTime - temporalRadius;
    let timeHigh = sliceTime + temporalRadius;
    for (var indexRow = rowLow; indexRow <= rowHigh; indexRow++) {
      for (var indexColumn = columnLow; indexColumn <= columnHigh; indexColumn++) {
        let indexCell = indexRow * INDEX_WIDTH + indexColumn;
        let end = cellOffsets[cellOffsetsOffset + indexCell + 1u];
        // First slot of the run whose time is at or after timeLow.
        var low = cellOffsets[cellOffsetsOffset + indexCell];
        var high = end;
        while (low < high) {
          let middle = low + (high - low) / 2u;
          if (slotPoints[slotPointsOffset + 4u * middle + 2u] < timeLow) {
            low = middle + 1u;
          } else {
            high = middle;
          }
        }
        for (var slot = low; slot < end; slot++) {
          let pointTime = slotPoints[slotPointsOffset + 4u * slot + 2u];
          if (pointTime > timeHigh) {
            break;
          }
          let delta = vec2f(slotPoints[slotPointsOffset + 4u * slot], slotPoints[slotPointsOffset + 4u * slot + 1u]) - cellOffset;
          let distanceSquared = dot(delta, delta);
          if (!(distanceSquared < supportSquared)) {
            continue;
          }
          total += slotPoints[slotPointsOffset + 4u * slot + 3u] * getKernelValue(distanceSquared, bandwidth) *
            getTemporalKernelValue(sliceTime - pointTime, temporalBandwidth);
        }
      }
    }
  }
  var result = total;
  ${O?`if (valid) {
    if (!(all(cellOffset >= windowMin) && all(cellOffset <= windowMax))) {
      valid = false;
    } else {
      let mass = getWindowMass(cellOffset, windowMin, windowMax, bandwidth);
      valid = mass > 1e-6;
      result = result / max(mass, 1e-6);
    }
  }`:``}
  ${k?`if (valid) {
    let windowStart = params[paramsOffset + 12u];
    let windowEnd = params[paramsOffset + 13u];
    if (!(sliceTime >= windowStart && sliceTime <= windowEnd)) {
      valid = false;
    } else {
      let mass = getTemporalWindowMass(sliceTime, windowStart, windowEnd, temporalBandwidth);
      valid = mass > 1e-6;
      result = result / max(mass, 1e-6);
    }
  }`:``}
  ${c===`probability`?`if (valid) {
    let totalWeight = weightSum[weightSumOffset];
    valid = totalWeight > 0.0;
    result = result / max(totalWeight, 1e-30);
  }`:``}
  valuesOut[valuesOutOffset + index] = select(getNaN(), result, valid);`})),[...N,...h.publishNodes]}};function V(e,t){let{id:n,props:r,voxelCount:a}=t,{output:o,indexGridSize:s}=r,d=r.positions.length;u(n,e,[r.positions,r.times,r.weights,r.parameters,o.values,o.weightSum,o.status?.invalidCount]),i(n,`indexGridSize cells`,s[0]*s[1],c(e.device)),i(n,`voxels (grid.width * grid.height * sliceCount)`,a,l(e.device,4)),i(n,`positions length`,d,l(e.device,16)),O(e,n,[[`positions`,r.positions],[`times`,r.times],[`weights`,r.weights],[`parameters`,r.parameters],[`output.values`,o.values],[`output.weightSum`,o.weightSum],[`output.status.invalidCount`,o.status?.invalidCount]],{dispatchMetadata:!1}),j(e,n,d,s[0]*s[1]),N(e,n,[[`cell-ordered point table scratch`,d*16]])}export{V as n,P as r,B as t};