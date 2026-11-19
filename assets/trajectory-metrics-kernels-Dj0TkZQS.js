import{n as e}from"./sorted-segment-sums-Dvmm6OdR.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{s as r}from"./wgsl-common-DfyXiYB0.js";import{t as i}from"./time-words-BFl8KCxq.js";import{t as a}from"./geodesic-wgsl-I0CYDKcz.js";var o=`GPUTrajectoryMetrics`,s=8;function c(e){return e.timestamps.format===`uint32x2`?`words`:e.timestampsLow?`double-single`:`float32`}function l(e){let t=[{name:`timestamps`,view:e.timestamps,type:e.timestamps.format===`uint32x2`?`u32`:`f32`,access:`read`}];return e.timestampsLow&&t.push({name:`timestampsLow`,view:e.timestampsLow,type:`f32`,access:`read`}),t}function u(e){switch(e){case`words`:return`${i}
fn getTimeWords(row: u32) -> vec2<u32> {
  return vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
}

fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return timeWordsDifferenceToF32(getTimeWords(a), getTimeWords(b));
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  return timeWordsCompare(getTimeWords(a), getTimeWords(b)) >= 0i;
}
`;case`double-single`:return`
fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return (timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b]) +
    (timestampsLow[timestampsLowOffset + a] - timestampsLow[timestampsLowOffset + b]);
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  let aHigh = timestamps[timestampsOffset + a];
  let bHigh = timestamps[timestampsOffset + b];
  return aHigh > bHigh ||
    (aHigh == bHigh && timestampsLow[timestampsLowOffset + a] >= timestampsLow[timestampsLowOffset + b]);
}
`;default:return`
fn timeDifferenceRows(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}

fn isTimeNonDecreasing(a: u32, b: u32) -> bool {
  return timestamps[timestampsOffset + a] >= timestamps[timestampsOffset + b];
}
`}}function d(e){return`
const TRACK_COUNT: u32 = ${e}u;
const NO_TRACK: u32 = 0xffffffffu;

fn findTrack(row: u32) -> u32 {
  var low = 0u;
  var high = TRACK_COUNT;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high) / 2u;
    if (trackOffsets[trackOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || row >= trackOffsets[trackOffsetsOffset + TRACK_COUNT]) {
    return NO_TRACK;
  }
  return low - 1u;
}`}function f(e=!0,n){let r=`${n===void 0?`
fn getStepDistance(row: u32) -> f32 {
  let deltaX = positions[positionsOffset + 2u * row] - positions[positionsOffset + 2u * (row - 1u)];
  let deltaY = positions[positionsOffset + 2u * row + 1u] - positions[positionsOffset + 2u * (row - 1u) + 1u];
  return sqrt(deltaX * deltaX + deltaY * deltaY);
}
`:`${a}
fn getStepPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getStepDistance(row: u32) -> f32 {
  return geodesicHaversineDistance(getStepPosition(row - 1u), getStepPosition(row), ${t(n)});
}
`}

fn getStepDuration(row: u32) -> f32 {
  return timeDifferenceRows(row, row - 1u);
}
`;return e?`${r}
fn isSlowStep(row: u32) -> bool {
  let stepDuration = getStepDuration(row);
  let stepDistance = getStepDistance(row);
  return isTimeNonDecreasing(row, row - 1u) &&
    (stepDistance == 0.0 || stepDistance < parameters[parametersOffset] * stepDuration);
}`:r}function p(e,t){let n=!!(t.runStartFlags&&t.runEndFlags&&t.parameters);if(3+ +!!t.timestampsLow+(n?3:0)+ +!!t.stepDistances+ +!!t.stepSpeeds<=s||!n)return[m(e,t)];let{runStartFlags:r,runEndFlags:i,parameters:a,...o}=t,c=[];return(t.stepDistances||t.stepSpeeds)&&c.push(m(e,o)),c.push(m(e,{id:`${t.id}-runs`,positions:t.positions,timestamps:t.timestamps,timestampsLow:t.timestampsLow,trackOffsets:t.trackOffsets,parameters:a,geodesicRadius:t.geodesicRadius,runStartFlags:r,runEndFlags:i})),c}function m(e,t){let r=t.trackOffsets.length-1,i=!!(t.runStartFlags&&t.runEndFlags&&t.parameters),a=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...l(t),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}];return i&&a.push({name:`parameters`,view:t.parameters,type:`f32`,access:`read`}),t.stepDistances&&a.push({name:`stepDistances`,view:t.stepDistances,type:`f32`,access:`read_write`}),t.stepSpeeds&&a.push({name:`stepSpeeds`,view:t.stepSpeeds,type:`f32`,access:`read_write`}),i&&a.push({name:`runStartFlags`,view:t.runStartFlags,type:`u32`,access:`read_write`},{name:`runEndFlags`,view:t.runEndFlags,type:`u32`,access:`read_write`}),n(e,{id:t.id,operation:o,variant:`steps`,bindings:a,invocationCount:t.positions.length,declarations:`${d(r)}
${u(c(t))}
${f(i,t.geodesicRadius)}`,body:`
  var stepDistance = 0.0;
  var stepSpeed = 0.0;
  var runStart = 0u;
  var runEnd = 0u;
  let track = findTrack(index);
  if (track != NO_TRACK) {
    let trackStart = trackOffsets[trackOffsetsOffset + track];
    let trackEnd = trackOffsets[trackOffsetsOffset + track + 1u];
    if (index > trackStart) {
      stepDistance = getStepDistance(index);
      let stepDuration = getStepDuration(index);
      if (stepDuration > 0.0) {
        stepSpeed = stepDistance / stepDuration;
      }
      ${i?`if (isSlowStep(index)) {
        let previousSlow = index - 1u > trackStart && isSlowStep(index - 1u);
        let nextSlow = index + 1u < trackEnd && isSlowStep(index + 1u);
        runStart = select(1u, 0u, previousSlow);
        runEnd = select(1u, 0u, nextSlow);
      }`:``}
    }
  }
  ${t.stepDistances?`stepDistances[stepDistancesOffset + index] = stepDistance;`:``}
  ${t.stepSpeeds?`stepSpeeds[stepSpeedsOffset + index] = stepSpeed;`:``}
  ${i?`runStartFlags[runStartFlagsOffset + index] = runStart;
  runEndFlags[runEndFlagsOffset + index] = runEnd;`:``}`})}function h(e,t){let i=t.trackOffsets.length-1,a=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},...l(t),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}];for(let[e,n]of[[`stepSpeeds`,t.stepSpeeds],[`stepHeadings`,t.stepHeadings],[`stepAccelerations`,t.stepAccelerations]])n&&a.push({name:e,view:n,type:`f32`,access:`read_write`});return n(e,{id:t.id,operation:o,variant:`step-columns`,bindings:a,invocationCount:t.positions.length,declarations:`${d(i)}
${u(c(t))}
${f(!1,t.geodesicRadius)}
${r}

fn getStepSpeed(row: u32) -> f32 {
  let duration = getStepDuration(row);
  if (duration > 0.0) {
    return getStepDistance(row) / duration;
  }
  return 0.0;
}`,body:`
  var speed = 0.0;
  var heading = 0.0;
  var acceleration = 0.0;
  let track = findTrack(index);
  if (track != NO_TRACK) {
    let trackStart = trackOffsets[trackOffsetsOffset + track];
    if (index > trackStart) {
      speed = getStepSpeed(index);
      ${t.geodesicRadius===void 0?`let deltaX = positions[positionsOffset + 2u * index] - positions[positionsOffset + 2u * (index - 1u)];
      let deltaY = positions[positionsOffset + 2u * index + 1u] - positions[positionsOffset + 2u * (index - 1u) + 1u];
      if (deltaX != 0.0 || deltaY != 0.0) {
        // deltaX is -0.0 for a due north or south step from x = +0 to x = -0 (FX-17).
        heading = guardedAtan2(deltaY, deltaX);
      }`:`if (getStepDistance(index) > 0.0) {
        // The planar convention (counterclockwise from east) applied to the initial bearing
        // (clockwise from north): atan2(cos(bearing), sin(bearing)) = pi / 2 - bearing, wrapped.
        let bearing = geodesicInitialBearingDegrees(getStepPosition(index - 1u), getStepPosition(index)) * GEODESIC_DEGREES_TO_RADIANS;
        // sin(bearing) is -0.0 for a bearing of -0.0 (due north with a longitude step of -0.0).
        heading = guardedAtan2(cos(bearing), sin(bearing));
      }`}
      let duration = getStepDuration(index);
      if (index - 1u > trackStart && duration > 0.0) {
        acceleration = (speed - getStepSpeed(index - 1u)) / duration;
      }
    }
  }
  ${t.stepSpeeds?`stepSpeeds[stepSpeedsOffset + index] = speed;`:``}
  ${t.stepHeadings?`stepHeadings[stepHeadingsOffset + index] = heading;`:``}
  ${t.stepAccelerations?`stepAccelerations[stepAccelerationsOffset + index] = acceleration;`:``}`})}function g(e,t){let r=t.trackOffsets.length-1,i=[...l(t),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}];return t.averageSpeeds&&t.trackLengths&&i.push({name:`trackLengths`,view:t.trackLengths,type:`f32`,access:`read`}),t.trackDurations&&i.push({name:`trackDurations`,view:t.trackDurations,type:`f32`,access:`read_write`}),t.averageSpeeds&&i.push({name:`averageSpeeds`,view:t.averageSpeeds,type:`f32`,access:`read_write`}),n(e,{id:t.id,operation:o,variant:`finalize`,bindings:i,invocationCount:r,declarations:u(c(t)),body:`
  let trackStart = trackOffsets[trackOffsetsOffset + index];
  let trackEnd = trackOffsets[trackOffsetsOffset + index + 1u];
  var trackDuration = 0.0;
  if (trackEnd >= trackStart + 2u) {
    trackDuration = timeDifferenceRows(trackEnd - 1u, trackStart);
  }
  ${t.trackDurations?`trackDurations[trackDurationsOffset + index] = trackDuration;`:``}
  ${t.averageSpeeds?`var averageSpeed = 0.0;
  if (trackDuration > 0.0) {
    averageSpeed = trackLengths[trackLengthsOffset + index] / trackDuration;
  }
  averageSpeeds[averageSpeedsOffset + index] = averageSpeed;`:``}`})}function _(e,t){return n(e,{id:t.id,operation:o,variant:`row-ids`,bindings:[{name:`rowIds`,view:t.rowIds,type:`u32`,access:`read_write`},{name:`runIndices`,view:t.runIndices,type:`u32`,access:`read_write`}],invocationCount:Math.max(t.rowCount,t.runCapacity),declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const RUN_CAPACITY: u32 = ${t.runCapacity}u;`,body:`
  if (index < ROW_COUNT) {
    rowIds[rowIdsOffset + index] = index;
  }
  if (index < RUN_CAPACITY) {
    runIndices[runIndicesOffset + index] = index;
  }`})}function v(e,t){let r=l(t),i=u(c(t)),a=[{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`},{name:`runCountIn`,view:t.runCount,type:`u32`,access:`read`}],f=[{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}],p=[{name:`qualifyFlags`,view:t.qualifyFlags,type:`u32`,access:`read_write`},{name:`runTracks`,view:t.runTracks,type:`u32`,access:`read_write`}],m=[{name:`parameters`,view:t.parameters,type:`f32`,access:`read`}],h=`
  var flag = 0u;
  if (index < runCountIn[runCountInOffset]) {
    let firstStepRow = runStarts[runStartsOffset + index];
    let lastStepRow = runEnds[runEndsOffset + index];
    if (timeDifferenceRows(lastStepRow, firstStepRow - 1u) >= parameters[parametersOffset + 1u]) {
      flag = 1u;
    }
  }`,g={operation:o,invocationCount:t.runCapacity},_=d(t.trackOffsets.length-1),v=[...a,...r,...m,...f,...p];return v.length<=s?[n(e,{...g,id:t.id,variant:`qualify`,bindings:v,declarations:`${_}\n${i}`,body:`${h}
  var track = NO_TRACK;
  if (flag == 1u) {
    track = findTrack(runStarts[runStartsOffset + index]);
  }
  qualifyFlags[qualifyFlagsOffset + index] = flag;
  runTracks[runTracksOffset + index] = track;`})]:[n(e,{...g,id:t.id,variant:`qualify`,bindings:[...a,...r,...m,p[0]],declarations:i,body:`${h}
  qualifyFlags[qualifyFlagsOffset + index] = flag;`}),n(e,{...g,id:`${t.id}-tracks`,variant:`qualify-tracks`,bindings:[a[0],a[2],...f,{name:`qualifyFlags`,view:t.qualifyFlags,type:`u32`,access:`read`},p[1]],declarations:_,body:`
  var track = NO_TRACK;
  if (index < runCountIn[runCountInOffset] && qualifyFlags[qualifyFlagsOffset + index] == 1u) {
    track = findTrack(runStarts[runStartsOffset + index]);
  }
  runTracks[runTracksOffset + index] = track;`})]}function y(e,t){return n(e,{id:t.id,operation:o,variant:`centroid-offsets`,bindings:[{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`},{name:`runCountIn`,view:t.runCount,type:`u32`,access:`read`},{name:`centroidOffsets`,view:t.centroidOffsets,type:`u32`,access:`read_write`}],invocationCount:t.runCapacity,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const RUN_CAPACITY: u32 = ${t.runCapacity}u;`,body:`
  var segmentStart = ROW_COUNT;
  var segmentEnd = ROW_COUNT;
  if (index < runCountIn[runCountInOffset]) {
    segmentStart = runStarts[runStartsOffset + index] - 1u;
    segmentEnd = runEnds[runEndsOffset + index] + 1u;
  }
  centroidOffsets[centroidOffsetsOffset + 2u * index] = segmentStart;
  centroidOffsets[centroidOffsetsOffset + 2u * index + 1u] = segmentEnd;
  if (index == 0u) {
    centroidOffsets[centroidOffsetsOffset + 2u * RUN_CAPACITY] = ROW_COUNT;
  }`})}function b(e,t){return n(e,{id:t.id,operation:o,variant:`split-positions`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`xs`,view:t.xs,type:`f32`,access:`read_write`},{name:`ys`,view:t.ys,type:`f32`,access:`read_write`}],invocationCount:t.positions.length,body:`
  xs[xsOffset + index] = positions[positionsOffset + 2u * index];
  ys[ysOffset + index] = positions[positionsOffset + 2u * index + 1u];`})}function x(e,t){let r=[],i=[{name:`stopTotalIn`,view:t.stopTotal,type:`u32`,access:`read`},{name:`qualifiedRuns`,view:t.qualifiedRuns,type:`u32`,access:`read`},{name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`},{name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`}],a=`const CAPACITY: u32 = ${t.capacity}u;`,s=`
  let count = min(stopTotalIn[stopTotalInOffset], CAPACITY);
  let isStop = index < count;
  var run = 0u;
  if (isStop) {
    run = qualifiedRuns[qualifiedRunsOffset + index];
  }`,d=[{name:`stopTotalIn`,view:t.stopTotal,type:`u32`,access:`read`},{name:`qualifiedRuns`,view:t.qualifiedRuns,type:`u32`,access:`read`},{name:`runTracks`,view:t.runTracks,type:`u32`,access:`read`},{name:`idsOut`,view:t.ids,type:`u32`,access:`read_write`}];return t.startRows&&d.push({name:`runStarts`,view:t.runStarts,type:`u32`,access:`read`}),t.endRows&&d.push({name:`runEnds`,view:t.runEnds,type:`u32`,access:`read`}),t.startRows&&d.push({name:`startRowsOut`,view:t.startRows,type:`u32`,access:`read_write`}),t.endRows&&d.push({name:`endRowsOut`,view:t.endRows,type:`u32`,access:`read_write`}),r.push(n(e,{id:t.id,operation:o,variant:`stop-gather`,bindings:d,invocationCount:t.capacity,declarations:a,body:`${s}
  idsOut[idsOutOffset + index] = select(0xffffffffu, runTracks[runTracksOffset + run], isStop);
  ${t.startRows?`startRowsOut[startRowsOutOffset + index] = select(0xffffffffu, runStarts[runStartsOffset + run] - 1u, isStop);`:``}
  ${t.endRows?`endRowsOut[endRowsOutOffset + index] = select(0xffffffffu, runEnds[runEndsOffset + run], isStop);`:``}`})),t.durations&&r.push(n(e,{id:`${t.id}-durations`,operation:o,variant:`stop-durations`,bindings:[...i,...l(t),{name:`durationsOut`,view:t.durations,type:`f32`,access:`read_write`}],invocationCount:t.capacity,declarations:`${a}
${u(c(t))}`,body:`${s}
  var stopDuration = 0.0;
  if (isStop) {
    stopDuration = timeDifferenceRows(runEnds[runEndsOffset + run], runStarts[runStartsOffset + run] - 1u);
  }
  durationsOut[durationsOutOffset + index] = stopDuration;`})),t.centroids&&t.sumsX&&t.sumsY&&r.push(n(e,{id:`${t.id}-centroids`,operation:o,variant:`stop-centroids`,bindings:[...i,{name:`sumsX`,view:t.sumsX,type:`f32`,access:`read`},{name:`sumsY`,view:t.sumsY,type:`f32`,access:`read`},{name:`centroidsOut`,view:t.centroids,type:`f32`,access:`read_write`}],invocationCount:t.capacity,declarations:a,body:`${s}
  var centroid = vec2<f32>(0.0, 0.0);
  if (isStop) {
    let rowCount = f32(runEnds[runEndsOffset + run] - runStarts[runStartsOffset + run] + 2u);
    centroid = vec2<f32>(sumsX[sumsXOffset + 2u * run], sumsY[sumsYOffset + 2u * run]) / rowCount;
  }
  centroidsOut[centroidsOutOffset + 2u * index] = centroid.x;
  centroidsOut[centroidsOutOffset + 2u * index + 1u] = centroid.y;`})),r}function S(t,n){return[e(t,{id:n.id,operation:o,segmentCount:n.output.length,input:n.input,segmentOffsets:n.segmentOffsets,output:n.output,reduction:n.operation})]}export{b as a,x as c,c as d,u as f,_ as i,S as l,g as n,h as o,v as r,p as s,y as t,l as u};