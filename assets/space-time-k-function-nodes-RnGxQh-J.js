import{r as e}from"./kernel-C3tfQOn4.js";import{t}from"./gpu-sort-xRm8l5yn.js";import{S as n,u as r}from"./gpu-contributor-utils-BIwLQON5.js";import{o as i,t as a}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as o,t as s}from"./wgsl-common-DfyXiYB0.js";import{n as c,r as l}from"./pair-histogram-COfc4qPq.js";import{i as u,r as d,t as f}from"./pair-statistics-grid-L8bKgCyF.js";var p=65536,m=`
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;

// Angle of the arc of a circle of radius d (centered at distance h from a side line) outside it.
fn getSideAngle(h: f32, d: f32) -> f32 {
  if (d > h) {
    // A point on the side line: a quarter turn. h can be -0.0 (a coordinate of -0.0 against a
    // bound of 0), and atan2(y, -0.0) is -pi / 2 on some adapters.
    if (h == 0.0) {
      return HALF_PI;
    }
    return atan2(sqrt((d - h) * (d + h)), h);
  }
  return 0.0;
}

fn getArcOverlap(first: f32, second: f32) -> f32 {
  return max(0.0, first + second - HALF_PI);
}

fn getSpaceTimeSpatialWeight(left: f32, bottom: f32, right: f32, top: f32, d: f32) -> f32 {
  let a = getSideAngle(left, d);
  let b = getSideAngle(bottom, d);
  let c = getSideAngle(right, d);
  let e = getSideAngle(top, d);
  let overlap = getArcOverlap(a, b) + getArcOverlap(b, c) + getArcOverlap(c, e) + getArcOverlap(e, a);
  let inside = 1.0 - (a + b + c + e) / PI + overlap / (2.0 * PI);
  return min(1.0 / max(inside, 1.0 / WEIGHT_CAP), WEIGHT_CAP);
}

fn getSpaceTimeTemporalWeight(lag: f32, radius: f32) -> f32 {
  return select(2.0, 1.0, lag <= radius);
}
`,h=`
fn getSpaceTimeSplancsTemporalWeight(lag: f32, radius: f32) -> f32 {
  return select(1.0, 2.0, radius <= lag);
}

fn getSpaceTimeSplancsClass(value: f32, maximum: f32, count: u32) -> u32 {
  if (!(value < maximum)) {
    return count;
  }
  // A first guess from the scaled value, then one step either way by the product test.
  var classIndex = min(u32(max(ceil(value / maximum * f32(count)) - 1.0, 0.0)), count - 1u);
  let scaled = value * f32(count);
  if (classIndex > 0u && scaled <= maximum * f32(classIndex)) {
    classIndex = classIndex - 1u;
  } else if (classIndex + 1u < count && scaled > maximum * f32(classIndex + 1u)) {
    classIndex = classIndex + 1u;
  }
  return classIndex;
}
`,g=16;function _(e){return Math.min(d(e),r(e,g))}function v(e,t,r){n(e,`positions length`,r,_(t))}function y(e,t=new Float32Array(11)){let n=`getGPUSpaceTimeKFunctionParameterValues`;if(t.length<11)throw Error(`${n} target must hold 11 elements`);let{bounds:r,timeWindow:i,maximumDistance:a,maximumTimeLag:o}=e;if(r.length!==4||!r.every(Number.isFinite))throw Error(`${n} bounds must be four finite numbers [minX, minY, maxX, maxY]`);if(!(r[0]<r[2]&&r[1]<r[3]))throw Error(`${n} bounds must have positive area`);if(i.length!==2||!i.every(Number.isFinite)||!(i[0]<i[1]))throw Error(`${n} timeWindow must be finite [start, end] with start < end`);for(let[e,t]of[[`maximumDistance`,a],[`maximumTimeLag`,o]])if(!Number.isFinite(t)||t<=0)throw Error(`${n} ${e} must be positive and finite`);let s=e.spatialBandwidth??0,c=e.temporalBandwidth??0;for(let[e,t]of[[`spatialBandwidth`,s],[`temporalBandwidth`,c]])if(!Number.isFinite(t)||t<0)throw Error(`${n} ${e} must be non-negative and finite`);return t.set([...r,a+s,a,o,...i,s,c]),t}var b=4096;function x(e){let{distanceClassCount:t,timeClassCount:n}=e;return`
const DISTANCE_CLASS_COUNT: u32 = ${t}u;
const TIME_CLASS_COUNT: u32 = ${n}u;
const BINS: u32 = ${t*n}u;
const WEIGHT_SCALE: f32 = ${p}.0;
const WEIGHT_CAP: f32 = 100.0;
const LARGE_UNIT: f32 = ${b}.0;
const EDGE_ISOTROPIC: bool = ${e.edgeCorrection===`isotropic`};
const INHOMOGENEOUS: bool = ${!!e.intensities};
`}function S(n,r){let{id:s,operation:d,positions:p,times:g,parameters:_,gridSize:v}=r,y=p.length,b=r.distanceClassCount,S=r.timeClassCount,w=b*S,T=v[0]*v[1],E=x(r),D=!!r.spatialK,O=!!r.temporalK,k=!!r.pairCorrelation,A=!!r.local,j=r.tieRule===`splancs`,M=j?`let splancsDistanceClass = getSpaceTimeSplancsClass(pairDistance, maximumDistance, DISTANCE_CLASS_COUNT);
            let splancsLagClass = getSpaceTimeSplancsClass(timeLag, maximumTimeLag, TIME_CLASS_COUNT);
            let distanceClass = min(splancsDistanceClass, DISTANCE_CLASS_COUNT - 1u);
            let lagClass = min(splancsLagClass, TIME_CLASS_COUNT - 1u);
            let inDistance = splancsDistanceClass < DISTANCE_CLASS_COUNT;
            let inLag = splancsLagClass < TIME_CLASS_COUNT;`:`let scaledDistance = pairDistance / maximumDistance * f32(DISTANCE_CLASS_COUNT);
            let scaledLag = timeLag / maximumTimeLag * f32(TIME_CLASS_COUNT);
            let distanceClass = min(u32(max(ceil(scaledDistance) - 1.0, 0.0)), DISTANCE_CLASS_COUNT - 1u);
            let lagClass = min(u32(max(ceil(scaledLag) - 1.0, 0.0)), TIME_CLASS_COUNT - 1u);
            let inDistance = scaledDistance <= f32(DISTANCE_CLASS_COUNT);
            let inLag = scaledLag <= f32(TIME_CLASS_COUNT);`,N=[],P=e(n,`${s}-effective-mask`,`uint32`,y);N.push(i(n,{id:`${s}-effective-mask`,operation:d,variant:`effective-mask-${r.mask?`mask`:`all`}-${r.intensities?`intensity`:`homogeneous`}`,bindings:[{name:`times`,view:g,type:`f32`,access:`read`},{name:`parameters`,view:_,type:`f32`,access:`read`},...r.mask?[{name:`mask`,view:r.mask,type:`u32`,access:`read`}]:[],...r.intensities?[{name:`intensities`,view:r.intensities,type:`f32`,access:`read`}]:[],{name:`effectiveMask`,view:P,type:`u32`,access:`read_write`}],invocationCount:y,declarations:f,body:`let timeStart = parameters[parametersOffset + 7u];
  let timeEnd = parameters[parametersOffset + 8u];
  let time = times[timesOffset + index];
  var keep = isFiniteValue(timeStart) && isFiniteValue(timeEnd) && timeStart < timeEnd &&
    isFiniteValue(time) && time >= timeStart && time <= timeEnd;
  ${r.mask?`keep = keep && mask[maskOffset + index] != 0u;`:``}
  ${r.intensities?`let intensity = intensities[intensitiesOffset + index];
  keep = keep && isFiniteValue(intensity) && intensity > 0.0;`:``}
  effectiveMask[effectiveMaskOffset + index] = select(0u, 1u, keep);`}));let F=u(n,{id:s,operation:d,positions:p,parameters:_,gridSize:v,mask:P});N.push(...F.nodes);let I=e(n,`${s}-row-table`,`float32`,4*y);N.push(i(n,{id:`${s}-row-table`,operation:d,variant:`row-table-${r.intensities?`intensity`:`homogeneous`}`,bindings:[{name:`times`,view:g,type:`f32`,access:`read`},{name:`parameters`,view:_,type:`f32`,access:`read`},{name:`cellOffsets`,view:F.cellOffsets,type:`u32`,access:`read`},...r.intensities?[{name:`intensities`,view:r.intensities,type:`f32`,access:`read`}]:[],{name:`rowTable`,view:I,type:`f32`,access:`read_write`}],invocationCount:y,declarations:`const CELL_COUNT: u32 = ${T}u;`,body:`let timeStart = parameters[parametersOffset + 7u];
  let timeEnd = parameters[parametersOffset + 8u];
  let time = times[timesOffset + index];
  ${r.intensities?`// lambda' = lambda / (n / (|W| |T|)), so 1 / lambda' = lambdaBar / lambda.
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let lambdaBar = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]) / (area * (timeEnd - timeStart));
  let inverse = lambdaBar / intensities[intensitiesOffset + index];`:`let inverse = 1.0;`}
  rowTable[rowTableOffset + 4u * index] = time;
  rowTable[rowTableOffset + 4u * index + 1u] = inverse;
  rowTable[rowTableOffset + 4u * index + 2u] = min(time - timeStart, timeEnd - time);
  rowTable[rowTableOffset + 4u * index + 3u] = 0.0;`})),r.local&&N.push(i(n,{id:`${s}-local-fill`,operation:d,variant:`local-fill`,bindings:[{name:`local`,view:r.local,type:`f32`,access:`read_write`}],invocationCount:r.local.length,declarations:f,body:`local[localOffset + index] = getQuietNaN(index);`}));let L=w+(D?b:0),R=k?5:3,z=r.pairCorrelationKernel??`box`,B=e=>`
fn get${e}PairKernel(offset: f32, bandwidth: f32) -> f32 {
  // The normalised kernel times its bandwidth, so the accumulated term stays of order one.
  let scaled = offset / bandwidth;
  if (!(abs(scaled) <= 1.0)) { return 0.0; }
  return ${z===`box`?`0.5`:`0.75 * (1.0 - scaled * scaled)`};
}`,V=c(n,{id:`${s}-histogram`,operation:d,positions:p,parameters:_,gridSize:v,sortedRows:F.sortedRows,cellOffsets:F.cellOffsets,slotCount:L,channelCount:R,pairOrder:`ordered`,extraBindings:[{name:`rowTable`,view:I,type:`f32`,access:`read`},...r.local?[{name:`local`,view:r.local,type:`f32`,access:`read_write`}]:[]],declarations:`${E}
${m}${j?h:``}
const HAS_MARGINAL: bool = ${D};
const HAS_PAIR_CORRELATION: bool = ${k};
const HAS_LOCAL: bool = ${A};
${B(`Spatial`)}
${B(`Temporal`)}`,focusPrologue:`let focusBase = rowTableOffset + focus * 4u;
    let focusTime = rowTable[focusBase];
    let focusInverse = rowTable[focusBase + 1u];
    let focusRadius = rowTable[focusBase + 2u];
    let maximumDistance = readParameter(5u);
    let maximumTimeLag = readParameter(6u);
    let spatialBandwidth = readParameter(9u);
    let temporalBandwidth = readParameter(10u);
    let focusUsable = isFiniteValue(maximumDistance) && maximumDistance > 0.0 &&
      isFiniteValue(maximumTimeLag) && maximumTimeLag > 0.0;
    let left = x - lattice.minimumX;
    let bottom = y - lattice.minimumY;
    let right = lattice.maximumX - x;
    let top = lattice.maximumY - y;
    var localBins: array<f32, ${A?w:1}>;`,pairAction:`if (!focusUsable) { continue; }
            let otherBase = rowTableOffset + neighbor * 4u;
            let timeLag = abs(rowTable[otherBase] - focusTime);
            let spatialWeight = select(1.0, getSpaceTimeSpatialWeight(left, bottom, right, top, pairDistance), EDGE_ISOTROPIC);
            let temporalWeight = select(1.0, ${j?`getSpaceTimeSplancsTemporalWeight`:`getSpaceTimeTemporalWeight`}(timeLag, focusRadius), EDGE_ISOTROPIC);
            let pairIntensity = focusInverse * rowTable[otherBase + 1u];
            ${M}
            if (HAS_MARGINAL && inDistance) {
              let marginalSlot = BINS + distanceClass;
              accumulate(marginalSlot, 0u, 1u);
              accumulateScaled(marginalSlot, 1u, 2u, spatialWeight * WEIGHT_SCALE);
            }
            if (inDistance && inLag) {
              let slot = distanceClass * TIME_CLASS_COUNT + lagClass;
              let weight = spatialWeight * temporalWeight * pairIntensity;
              accumulate(slot, 0u, 1u);
              accumulateScaled(slot, 1u, 2u, weight * WEIGHT_SCALE);
              if (HAS_LOCAL) {
                localBins[slot] += weight;
              }
            }
            if (HAS_PAIR_CORRELATION && spatialBandwidth > 0.0 && temporalBandwidth > 0.0 &&
                pairDistance <= maximumDistance + spatialBandwidth &&
                timeLag <= maximumTimeLag + temporalBandwidth) {
              // Grid distance a has radius maximumDistance * (a + 1) / count.
              let firstDistance = max(i32(ceil((pairDistance - spatialBandwidth) / maximumDistance * f32(DISTANCE_CLASS_COUNT))) - 1, 0);
              let lastDistance = min(i32(floor((pairDistance + spatialBandwidth) / maximumDistance * f32(DISTANCE_CLASS_COUNT))) - 1, i32(DISTANCE_CLASS_COUNT) - 1);
              let firstLag = max(i32(ceil((timeLag - temporalBandwidth) / maximumTimeLag * f32(TIME_CLASS_COUNT))) - 1, 0);
              let lastLag = min(i32(floor((timeLag + temporalBandwidth) / maximumTimeLag * f32(TIME_CLASS_COUNT))) - 1, i32(TIME_CLASS_COUNT) - 1);
              let pairWeight = spatialWeight * temporalWeight * pairIntensity;
              for (var distanceIndex = firstDistance; distanceIndex <= lastDistance; distanceIndex++) {
                let spatialTerm = getSpatialPairKernel(maximumDistance * f32(distanceIndex + 1) / f32(DISTANCE_CLASS_COUNT) - pairDistance, spatialBandwidth);
                for (var lagIndex = firstLag; lagIndex <= lastLag; lagIndex++) {
                  let temporalTerm = getTemporalPairKernel(maximumTimeLag * f32(lagIndex + 1) / f32(TIME_CLASS_COUNT) - timeLag, temporalBandwidth);
                  accumulateScaled(u32(distanceIndex) * TIME_CLASS_COUNT + u32(lagIndex), 3u, 4u, pairWeight * spatialTerm * temporalTerm * WEIGHT_SCALE);
                }
              }
            }`,focusEpilogue:A?`if (HAS_LOCAL) {
      // 2D inclusive prefix of the local sums, in place.
      for (var distanceIndex = 0u; distanceIndex < DISTANCE_CLASS_COUNT; distanceIndex++) {
        for (var lagIndex = 0u; lagIndex < TIME_CLASS_COUNT; lagIndex++) {
          var value = localBins[distanceIndex * TIME_CLASS_COUNT + lagIndex];
          if (distanceIndex > 0u) { value += localBins[(distanceIndex - 1u) * TIME_CLASS_COUNT + lagIndex]; }
          if (lagIndex > 0u) { value += localBins[distanceIndex * TIME_CLASS_COUNT + lagIndex - 1u]; }
          if (distanceIndex > 0u && lagIndex > 0u) { value -= localBins[(distanceIndex - 1u) * TIME_CLASS_COUNT + lagIndex - 1u]; }
          localBins[distanceIndex * TIME_CLASS_COUNT + lagIndex] = value;
        }
      }
      // K_i = n * s * (sum over j), the scale s of the global estimate, so mean_i K_i = K.
      let includedCount = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
      let area = (lattice.maximumX - lattice.minimumX) * (lattice.maximumY - lattice.minimumY);
      let duration = readParameter(8u) - readParameter(7u);
      let scale = area * duration / select(includedCount * (includedCount - 1.0), includedCount * includedCount, INHOMOGENEOUS);
      for (var bin = 0u; bin < BINS; bin++) {
        local[localOffset + focus * BINS + bin] = includedCount * scale * localBins[bin];
      }
    }`:void 0});N.push(...V.nodes);let H;if(O){H=e(n,`${s}-temporal-accumulators`,`uint32`,2*S);let c=e(n,`${s}-time-keys`,`uint32`,y),l=e(n,`${s}-time-row-ids`,`uint32`,y),u=e(n,`${s}-sorted-time-keys`,`uint32`,y),f=e(n,`${s}-sorted-time-rows`,`uint32`,y);N.push(i(n,{id:`${s}-time-keys`,operation:d,variant:`time-keys`,bindings:[{name:`times`,view:g,type:`f32`,access:`read`},{name:`cellKeys`,view:F.cellKeys,type:`u32`,access:`read`},{name:`timeKeys`,view:c,type:`u32`,access:`read_write`},{name:`timeRowIds`,view:l,type:`u32`,access:`read_write`}],invocationCount:y,declarations:`${o}
const CELL_COUNT: u32 = ${T}u;`,body:`// Rows the pair index excluded sort last.
  let included = cellKeys[cellKeysOffset + index] < CELL_COUNT;
  timeKeys[timeKeysOffset + index] = select(0xffffffffu, getOrderedKey(times[timesOffset + index]), included);
  timeRowIds[timeRowIdsOffset + index] = index;`}),...new t({id:`${s}-time-sort`,keys:c,values:l,outputKeys:u,outputValues:f,keyBits:32}).getCommandNodes(n),a(n,{id:`${s}-temporal-clear`,operation:d,view:H,type:`u32`,value:`0u`}),C(n,{id:s,operation:d,sortedKeys:u,cellOffsets:F.cellOffsets,parameters:_,accumulators:H,timeClassCount:S,cellCount:T,edgeCorrection:r.edgeCorrection,tieRule:r.tieRule}))}let U=l(`accumulators`,R,``,L),W=[{name:`accumulators`,view:V.accumulators,type:`u32`,access:`read`},{name:`cellOffsets`,view:F.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:_,type:`f32`,access:`read`},{name:`k`,view:r.k,type:`f32`,access:`read_write`},...r.spatialK?[{name:`spatialK`,view:r.spatialK,type:`f32`,access:`read_write`}]:[],...H&&r.temporalK?[{name:`temporalAccumulators`,view:H,type:`u32`,access:`read`},{name:`temporalK`,view:r.temporalK,type:`f32`,access:`read_write`}]:[],...r.pairCounts?[{name:`pairCounts`,view:r.pairCounts,type:`u32`,access:`read_write`}]:[]],G=`let includedCount = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let duration = parameters[parametersOffset + 8u] - parameters[parametersOffset + 7u];
  let defined = includedCount >= 2.0 && isFiniteValue(area) && area > 0.0 &&
    isFiniteValue(duration) && duration > 0.0 &&
    parameters[parametersOffset + 5u] > 0.0 && parameters[parametersOffset + 6u] > 0.0;
  let pairNormalizer = select(includedCount * (includedCount - 1.0), includedCount * includedCount, INHOMOGENEOUS);`;return N.push(i(n,{id:`${s}-finish`,operation:d,variant:`finish-${r.edgeCorrection}-${!!r.intensities}-${D}-${O}-${!!r.pairCounts}-${k}`,bindings:W,invocationCount:w,declarations:`${f}
${E}
const CELL_COUNT: u32 = ${T}u;
const HAS_MARGINAL: bool = ${D};
${U}`,body:`${G}
  let distanceIndex = index / TIME_CLASS_COUNT;
  let lagIndex = index % TIME_CLASS_COUNT;
  var sum = 0.0;
  for (var firstIndex = 0u; firstIndex <= distanceIndex; firstIndex++) {
    for (var secondIndex = 0u; secondIndex <= lagIndex; secondIndex++) {
      let slot = firstIndex * TIME_CLASS_COUNT + secondIndex;
      sum += readPairAccumulator(slot, 1u) + readPairAccumulator(slot, 2u) * LARGE_UNIT;
    }
  }
  k[kOffset + index] = select(getQuietNaN(index), area * duration / pairNormalizer * sum / WEIGHT_SCALE, defined);
  ${r.spatialK?`if (index < DISTANCE_CLASS_COUNT) {
    var spatialSum = 0.0;
    for (var firstIndex = 0u; firstIndex <= index; firstIndex++) {
      spatialSum += readPairAccumulator(BINS + firstIndex, 1u) + readPairAccumulator(BINS + firstIndex, 2u) * LARGE_UNIT;
    }
    spatialK[spatialKOffset + index] = select(getQuietNaN(index), area / pairNormalizer * spatialSum / WEIGHT_SCALE, defined);
  }`:``}
  ${H&&r.temporalK?`if (index < TIME_CLASS_COUNT) {
    let word = index * 2u;
    let temporalSum = f32(temporalAccumulators[temporalAccumulatorsOffset + word + 1u]) * 4294967296.0 + f32(temporalAccumulators[temporalAccumulatorsOffset + word]);
    temporalK[temporalKOffset + index] = select(getQuietNaN(index), duration / pairNormalizer * temporalSum, defined);
  }`:``}
  ${r.pairCounts?`pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);`:``}`})),r.d&&r.spatialK&&r.temporalK&&N.push(i(n,{id:`${s}-d`,operation:d,variant:`diggle-d`,bindings:[{name:`k`,view:r.k,type:`f32`,access:`read`},{name:`spatialK`,view:r.spatialK,type:`f32`,access:`read`},{name:`temporalK`,view:r.temporalK,type:`f32`,access:`read`},{name:`d`,view:r.d,type:`f32`,access:`read_write`}],invocationCount:w,body:`d[dOffset + index] = k[kOffset + index] -
    spatialK[spatialKOffset + index / ${S}u] * temporalK[temporalKOffset + index % ${S}u];`})),r.pairCorrelation&&N.push(i(n,{id:`${s}-pair-correlation`,operation:d,variant:`pair-correlation`,bindings:[{name:`accumulators`,view:V.accumulators,type:`u32`,access:`read`},{name:`cellOffsets`,view:F.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:_,type:`f32`,access:`read`},{name:`pairCorrelation`,view:r.pairCorrelation,type:`f32`,access:`read_write`}],invocationCount:w,declarations:`${f}
${E}
const CELL_COUNT: u32 = ${T}u;
${U}`,body:`${G}
  let spatialBandwidth = parameters[parametersOffset + 9u];
  let temporalBandwidth = parameters[parametersOffset + 10u];
  let distanceIndex = index / TIME_CLASS_COUNT;
  let radius = parameters[parametersOffset + 5u] * f32(distanceIndex + 1u) / f32(DISTANCE_CLASS_COUNT);
  let sum = readPairAccumulator(index, 3u) + readPairAccumulator(index, 4u) * LARGE_UNIT;
  // g = s / (4 pi u) * sum ks kt w / (lambda lambda'), with the kernels' bandwidths undone.
  let value = area * duration / pairNormalizer * sum / WEIGHT_SCALE / (4.0 * 3.14159265358979 * radius * spatialBandwidth * temporalBandwidth);
  pairCorrelation[pairCorrelationOffset + index] = select(getQuietNaN(index), value, defined && spatialBandwidth > 0.0 && temporalBandwidth > 0.0);`})),r.status&&N.push(i(n,{id:`${s}-status`,operation:d,variant:`status`,bindings:[{name:`accumulators`,view:V.accumulators,type:`u32`,access:`read`},{name:`cellOffsets`,view:F.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:_,type:`f32`,access:`read`},{name:`status`,view:r.status,type:`u32`,access:`read_write`}],invocationCount:1,declarations:`${f}
${E}
const CELL_COUNT: u32 = ${T}u;
${U}`,body:`${G}
  // 0 valid, 1 undefined (fewer than two included rows or an invalid window or maximum),
  // 3 the exact integer accumulation overflowed.
  status[statusOffset] = select(select(0u, 3u, readPairOverflow()), 1u, !defined);`})),{nodes:N,inputs:F,rowTable:I,effectiveMask:P}}function C(e,t){let n=t.timeClassCount,r=t.tieRule===`splancs`;return i(e,{id:`${t.id}-temporal-marginal`,operation:t.operation,variant:`temporal-marginal-${t.edgeCorrection}${r?`-splancs`:``}`,bindings:[{name:`sortedKeys`,view:t.sortedKeys,type:`u32`,access:`read`},{name:`cellOffsets`,view:t.cellOffsets,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:`f32`,access:`read`},{name:`accumulators`,view:t.accumulators,type:`atomic<u32>`,access:`read_write`}],invocationCount:t.sortedKeys.length,workgroupSize:64,guardIndex:!1,declarations:`${s}
const CELL_COUNT: u32 = ${t.cellCount}u;
const TIME_CLASS_COUNT: u32 = ${n}u;
const EDGE_ISOTROPIC: bool = ${t.edgeCorrection===`isotropic`};
// One workgroup holds 64 rows, each adding at most 2 n < 2^25 to a class, so a u32 never wraps.
var<workgroup> localCounts: array<atomic<u32>, ${n}>;

fn isInside(lag: f32, limit: f32, radius: f32, useRadius: bool, maximumTimeLag: f32) -> bool {
  return ${r?`lag * f32(TIME_CLASS_COUNT) <= maximumTimeLag * limit && lag < maximumTimeLag && (!useRadius || lag < radius)`:`lag / maximumTimeLag * f32(TIME_CLASS_COUNT) <= limit && (!useRadius || lag <= radius)`};
}

// Number of sorted partners j != i inside the lag, for the event at the given time whose first
// equal key is at index first.
fn countPartners(time: f32, first: u32, count: u32, limit: f32, radius: f32, useRadius: bool, maximumTimeLag: f32) -> u32 {${r?`
  // No lag is below a radius of zero, not even the event's own lag 0, which the count relies on.
  if (useRadius && !(radius > 0.0)) {
    return 0u;
  }`:``}
  // Partners at or after the event: the first index failing the test.
  var low = first;
  var high = count;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isInside(decodeOrderedKey(sortedKeys[sortedKeysOffset + middle]) - time, limit, radius, useRadius, maximumTimeLag)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let after = low - first;
  // Partners before the event: the first index passing the test.
  low = 0u;
  high = first;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (isInside(time - decodeOrderedKey(sortedKeys[sortedKeysOffset + middle]), limit, radius, useRadius, maximumTimeLag)) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  // The event itself is among the partners at or after it.
  return after + (first - low) - 1u;
}`,body:`for (var word = localInvocationIndex; word < TIME_CLASS_COUNT; word += 64u) {
    atomicStore(&localCounts[word], 0u);
  }
  workgroupBarrier();
  let includedCount = cellOffsets[cellOffsetsOffset + CELL_COUNT];
  let maximumTimeLag = parameters[parametersOffset + 6u];
  if (index < includedCount && maximumTimeLag > 0.0) {
    let key = sortedKeys[sortedKeysOffset + index];
    let time = decodeOrderedKey(key);
    let radius = min(time - parameters[parametersOffset + 7u], parameters[parametersOffset + 8u] - time);
    // First index holding this key.
    var first = 0u;
    var end = index;
    while (first < end) {
      let middle = first + (end - first) / 2u;
      if (sortedKeys[sortedKeysOffset + middle] < key) {
        first = middle + 1u;
      } else {
        end = middle;
      }
    }
    for (var timeClass = 0u; timeClass < TIME_CLASS_COUNT; timeClass++) {
      let limit = f32(timeClass + 1u);
      var weighted = countPartners(time, first, includedCount, limit, radius, false, maximumTimeLag);
      if (EDGE_ISOTROPIC) {
        weighted = 2u * weighted - countPartners(time, first, includedCount, limit, radius, true, maximumTimeLag);
      }
      atomicAdd(&localCounts[timeClass], weighted);
    }
  }
  workgroupBarrier();
  for (var word = localInvocationIndex; word < TIME_CLASS_COUNT; word += 64u) {
    let low = atomicLoad(&localCounts[word]);
    if (low != 0u) {
      // Cumulative in the class index: the per-class counts already are, so add them as they are.
      let previous = atomicAdd(&accumulators[accumulatorsOffset + word * 2u], low);
      if (previous > 0xffffffffu - low) {
        atomicAdd(&accumulators[accumulatorsOffset + word * 2u + 1u], 1u);
      }
    }
  }`})}export{h as a,m as i,y as n,p as o,v as r,S as t};