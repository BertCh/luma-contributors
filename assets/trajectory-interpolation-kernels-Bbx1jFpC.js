import{s as e}from"./wgsl-common-DfyXiYB0.js";import{c as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{t as r}from"./time-words-BFl8KCxq.js";import{t as i}from"./geodesic-wgsl-I0CYDKcz.js";import{t as a}from"./addition-rounding-error-wgsl-DVwPAqKC.js";import{t as o}from"./trajectory-common-DFp8lX7h.js";var s={empty:0,active:1,beforeStart:2,afterEnd:3,gap:4};function c(e,t=new Float32Array(4)){if(o(t,4,`Trajectory playhead`),typeof e.playhead==`bigint`)throw Error(`Float32 playheads must be numbers, not bigints`);let n=l(e);if(!Number.isFinite(Math.fround(e.playhead)))throw Error(`Trajectory playhead must be finite`);return t.set([e.playhead,n,0,0]),t}function l(e){let t=e.maximumGap??0;if(!Number.isFinite(Math.fround(t))||t<0)throw Error(`Playhead maximumGap must be finite and non-negative`);return t}var u=`GPUTrajectoryPlayhead`,d=`GPUTrajectoryResample`,f=`
const NO_ROW: u32 = 0xffffffffu;
const STATUS_EMPTY: u32 = ${s.empty}u;
const STATUS_ACTIVE: u32 = ${s.active}u;
const STATUS_BEFORE_START: u32 = ${s.beforeStart}u;
const STATUS_AFTER_END: u32 = ${s.afterEnd}u;
const STATUS_GAP: u32 = ${s.gap}u;
`;function p(n){if(n===void 0)return`
fn interpolatePosition(a: vec2<f32>, b: vec2<f32>, fraction: f32) -> vec2<f32> {
  return select(a + (b - a) * fraction, b, fraction >= 1.0);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return length(b - a);
}

${e}
fn getSegmentHeading(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let delta = b - a;
  // delta.x is -0.0 for a due north or south step from x = +0 to x = -0.
  return select(0.0, guardedAtan2(delta.y, delta.x), delta.x != 0.0 || delta.y != 0.0);
}
`;let r=t(n);return`${i}
fn interpolatePosition(a: vec2<f32>, b: vec2<f32>, fraction: f32) -> vec2<f32> {
  if (fraction >= 1.0) {
    // b shifted by whole turns to continue the longitude of a (exactly b without a shift).
    return vec2<f32>(b.x - 360.0 * round((b.x - a.x) / 360.0), b.y);
  }
  if (fraction <= 0.0) {
    return a;
  }
  return geodesicInterpolate(a, b, geodesicCentralAngle(a, b), fraction);
}

fn getSegmentLength(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return geodesicHaversineDistance(a, b, ${r});
}

fn getSegmentHeading(a: vec2<f32>, b: vec2<f32>) -> f32 {
  // Planar convention (counterclockwise from east) of the initial bearing: pi / 2 - bearing, wrapped.
  let bearing = geodesicInitialBearingDegrees(a, b) * GEODESIC_DEGREES_TO_RADIANS;
  // sin(bearing) is -0.0 for a bearing of -0.0 (due north with a longitude step of -0.0).
  return select(0.0, guardedAtan2(cos(bearing), sin(bearing)), geodesicCentralAngle(a, b) > 0.0);
}
${e}
`}function m(e){return{name:`timestamps`,view:e,type:e.format===`uint32x2`?`u32`:`f32`,access:`read`}}function h(e){return e?`${r}
fn getTimeWords(row: u32) -> vec2<u32> {
  return vec2<u32>(timestamps[timestampsOffset + 2u * row], timestamps[timestampsOffset + 2u * row + 1u]);
}

fn getRowTimeDifference(a: u32, b: u32) -> f32 {
  return timeWordsDifferenceToF32(getTimeWords(a), getTimeWords(b));
}
`:`
fn getRowTimeDifference(a: u32, b: u32) -> f32 {
  return timestamps[timestampsOffset + a] - timestamps[timestampsOffset + b];
}
`}function g(e){return e?`
fn getPlayheadWords() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}

fn getPlayheadFraction() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 2u]);
}

fn getPlayheadMaxGap() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}

fn isRowAfterPlayhead(row: u32) -> bool {
  return !isTimeWordsAtLeast(getPlayheadWords(), getPlayheadFraction(), getTimeWords(row), 0.0);
}

fn isRowBeforePlayhead(row: u32) -> bool {
  return !isTimeWordsAtLeast(getTimeWords(row), 0.0, getPlayheadWords(), getPlayheadFraction());
}

fn getPlayheadElapsed(row: u32) -> f32 {
  return timeWordsDifference(getPlayheadWords(), getPlayheadFraction(), getTimeWords(row), 0.0);
}
`:`
fn getPlayheadMaxGap() -> f32 {
  return parameters[parametersOffset + 1u];
}

fn isRowAfterPlayhead(row: u32) -> bool {
  return timestamps[timestampsOffset + row] > parameters[parametersOffset];
}

fn isRowBeforePlayhead(row: u32) -> bool {
  return timestamps[timestampsOffset + row] < parameters[parametersOffset];
}

fn getPlayheadElapsed(row: u32) -> f32 {
  return parameters[parametersOffset] - timestamps[timestampsOffset + row];
}
`}function _(e){return e?`
fn getClockOffset(firstRow: u32) -> f32 {
  let start = vec2<u32>(clockParameters[clockParametersOffset], clockParameters[clockParametersOffset + 1u]);
  return timeWordsDifference(start, bitcast<f32>(clockParameters[clockParametersOffset + 2u]), getTimeWords(firstRow), 0.0);
}

fn getClockStep() -> f32 {
  return bitcast<f32>(clockParameters[clockParametersOffset + 3u]);
}
`:`
fn getClockOffset(firstRow: u32) -> f32 {
  return clockParameters[clockParametersOffset] - timestamps[timestampsOffset + firstRow];
}

fn getClockStep() -> f32 {
  return clockParameters[clockParametersOffset + 1u];
}
`}var v=`
  let trackStart = min(trackOffsets[trackOffsetsOffset + track], ROW_COUNT);
  let trackEnd = clamp(trackOffsets[trackOffsetsOffset + track + 1u], trackStart, ROW_COUNT);`;function y(e,t){let r=t.timestamps.format===`uint32x2`;return n(e,{id:t.id,operation:u,variant:r?`bracket-words`:`bracket`,bindings:[m(t.timestamps),{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`parameters`,view:t.parameters,type:r?`u32`:`f32`,access:`read`},{name:`bracket`,view:t.bracket,type:`u32`,access:`read_write`}],invocationCount:t.trackCount,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const BRACKET_STRIDE: u32 = 5u;
${f}
${h(r)}
${g(r)}`,body:`let track = index;
  ${v}
  var status = STATUS_EMPTY;
  var segmentRow = NO_ROW;
  var nextRow = NO_ROW;
  var fraction = 0.0;
  var duration = 0.0;
  if (trackEnd > trackStart) {
    let lastRow = trackEnd - 1u;
    let lastSegmentRow = select(trackStart, lastRow - 1u, lastRow > trackStart);
    if (isRowAfterPlayhead(trackStart)) {
      status = STATUS_BEFORE_START;
      segmentRow = trackStart;
      nextRow = min(trackStart + 1u, lastRow);
    } else if (isRowBeforePlayhead(lastRow)) {
      status = STATUS_AFTER_END;
      segmentRow = lastSegmentRow;
      nextRow = lastRow;
      fraction = 1.0;
    } else {
      // t[trackStart] <= playhead <= t[lastRow]: first row in (trackStart, trackEnd) after it.
      var low = trackStart + 1u;
      var high = trackEnd;
      loop {
        if (low >= high) {
          break;
        }
        let middle = (low + high) / 2u;
        if (isRowAfterPlayhead(middle)) {
          high = middle;
        } else {
          low = middle + 1u;
        }
      }
      status = STATUS_ACTIVE;
      if (low == trackEnd) {
        // The playhead equals the last sample time.
        segmentRow = lastSegmentRow;
        nextRow = lastRow;
        fraction = 1.0;
      } else {
        // t[low - 1] <= playhead < t[low], so the interval is strictly positive.
        segmentRow = low - 1u;
        nextRow = low;
        let interval = getRowTimeDifference(nextRow, segmentRow);
        fraction = clamp(getPlayheadElapsed(segmentRow) / interval, 0.0, 1.0);
        let maxGap = getPlayheadMaxGap();
        if (maxGap > 0.0 && interval > maxGap && isRowBeforePlayhead(segmentRow)) {
          status = STATUS_GAP;
          fraction = 0.0;
        }
      }
    }
    duration = getRowTimeDifference(nextRow, segmentRow);
  }
  let base = bracketOffset + BRACKET_STRIDE * track;
  bracket[base] = segmentRow;
  bracket[base + 1u] = nextRow;
  bracket[base + 2u] = status;
  bracket[base + 3u] = bitcast<u32>(fraction);
  bracket[base + 4u] = bitcast<u32>(duration);`})}function b(e,t){let r=[{name:`bracket`,view:t.bracket,type:`u32`,access:`read`},{name:`positions`,view:t.positions,type:`f32`,access:`read`}],i=(e,t,n)=>{t&&r.push({name:e,view:t,type:`f32`,access:n})},a=!!(t.elevations&&t.currentElevations);return i(`elevations`,a?t.elevations:void 0,`read`),i(`currentPositions`,t.currentPositions,`read_write`),i(`currentElevations`,a?t.currentElevations:void 0,`read_write`),i(`headings`,t.headings,`read_write`),i(`speeds`,t.speeds,`read_write`),n(e,{id:t.id,operation:u,variant:`geometry`,bindings:r,invocationCount:t.trackCount,declarations:`const BRACKET_STRIDE: u32 = 5u;
${f}
${p(t.geodesicRadius)}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`let base = bracketOffset + BRACKET_STRIDE * index;
  let segmentRow = bracket[base];
  let nextRow = bracket[base + 1u];
  let status = bracket[base + 2u];
  let fraction = bitcast<f32>(bracket[base + 3u]);
  let duration = bitcast<f32>(bracket[base + 4u]);
  var position = vec2<f32>(0.0, 0.0);
  var elevation = 0.0;
  var heading = 0.0;
  var speed = 0.0;
  if (status != STATUS_EMPTY) {
    let startPosition = getPosition(segmentRow);
    let endPosition = getPosition(nextRow);
    position = interpolatePosition(startPosition, endPosition, fraction);
    ${a?`let startElevation = elevations[elevationsOffset + segmentRow];
    let endElevation = elevations[elevationsOffset + nextRow];
    elevation = select(
      startElevation + (endElevation - startElevation) * fraction,
      endElevation,
      fraction >= 1.0
    );`:``}
    heading = getSegmentHeading(startPosition, endPosition);
    if (status == STATUS_ACTIVE && duration > 0.0) {
      speed = getSegmentLength(startPosition, endPosition) / duration;
    }
  }
  ${t.currentPositions?`currentPositions[currentPositionsOffset + 2u * index] = position.x;
  currentPositions[currentPositionsOffset + 2u * index + 1u] = position.y;`:``}
  ${a?`currentElevations[currentElevationsOffset + index] = elevation;`:``}
  ${t.headings?`headings[headingsOffset + index] = heading;`:``}
  ${t.speeds?`speeds[speedsOffset + index] = speed;`:``}
  _ = elevation;`})}function x(e,t){let r=[{name:`bracket`,view:t.bracket,type:`u32`,access:`read`}],i=[];for(let[e,n,a,o]of[[`status`,t.status,`u32`,`trackStatus`],[`segmentRows`,t.segmentRows,`u32`,`trackSegmentRow`],[`segmentFractions`,t.segmentFractions,`f32`,`trackFraction`],[`activeFlags`,t.activeFlags,`u32`,`select(0u, 1u, trackStatus == STATUS_ACTIVE)`],[`trackIndices`,t.trackIndices,`u32`,`index`]])n&&(r.push({name:e,view:n,type:a,access:`read_write`}),i.push(`${e}[${e}Offset + index] = ${o};`));return n(e,{id:t.id,operation:u,variant:`columns`,bindings:r,invocationCount:t.trackCount,declarations:`const BRACKET_STRIDE: u32 = 5u;
${f}`,body:`let base = bracketOffset + BRACKET_STRIDE * index;
  let trackSegmentRow = bracket[base];
  let trackStatus = bracket[base + 2u];
  let trackFraction = bitcast<f32>(bracket[base + 3u]);
  _ = trackSegmentRow;
  _ = trackFraction;
  _ = trackStatus;
  ${i.join(`
  `)}`})}function S(e,t){return n(e,{id:t.id,operation:d,variant:`arc-length`,bindings:[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`cumulativeLengths`,view:t.cumulativeLengths,type:`f32`,access:`read_write`}],invocationCount:t.trackCount,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
${a}
${p(t.geodesicRadius)}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`let track = index;
  ${v}
  if (trackEnd > trackStart) {
    // Neumaier-compensated running sum, as the line family's prefix (line-segmentize-kernels.ts):
    // a plain f32 sum of 86,400 equal steps of 13.9 ends 1,051 short of 1,200,960. The correction
    // is read off the integer significands; the float form is removed by fast-math compilers.
    var sum = 0.0;
    var compensation = 0.0;
    cumulativeLengths[cumulativeLengthsOffset + trackStart] = 0.0;
    for (var row = trackStart + 1u; row < trackEnd; row++) {
      let segmentLength = getSegmentLength(getPosition(row - 1u), getPosition(row));
      let nextSum = sum + segmentLength;
      compensation = compensation + getAdditionRoundingError(sum, segmentLength, nextSum);
      sum = nextSum;
      cumulativeLengths[cumulativeLengthsOffset + row] = sum + compensation;
    }
  }`})}function C(e,t){return n(e,{id:t.id,operation:d,variant:`clock-times`,bindings:[{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`},{name:`clockParameters`,view:t.clock,type:t.isWordMode?`u32`:`f32`,access:`read`},{name:`sampleTimes`,view:t.sampleTimes,type:`f32`,access:`read_write`}],invocationCount:t.trackCount*t.sampleCount,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const SAMPLE_COUNT: u32 = ${t.sampleCount}u;
fn getClockStep() -> f32 {
  return ${t.isWordMode?`bitcast<f32>(clockParameters[clockParametersOffset + 3u])`:`clockParameters[clockParametersOffset + 1u]`};
}`,body:`let track = index / SAMPLE_COUNT;
  let sampleIndex = index - track * SAMPLE_COUNT;
  ${v}
  sampleTimes[sampleTimesOffset + index] = select(0.0, f32(sampleIndex) * getClockStep(), trackEnd > trackStart);`})}function w(e,t){let r=!!t.cumulativeLengths,i=!!t.clock,a=t.timestamps,o=a?.format===`uint32x2`,s=!!(t.elevations&&t.sampleElevations),c=[{name:`positions`,view:t.positions,type:`f32`,access:`read`},{name:`trackOffsets`,view:t.trackOffsets,type:`u32`,access:`read`}];a&&c.push(m(a)),t.clock&&c.push({name:`clockParameters`,view:t.clock,type:o?`u32`:`f32`,access:`read`}),t.maximumGap&&c.push({name:`maximumGap`,view:t.maximumGap,type:`f32`,access:`read`}),t.cumulativeLengths&&c.push({name:`cumulativeLengths`,view:t.cumulativeLengths,type:`f32`,access:`read`}),s&&t.elevations&&t.sampleElevations&&c.push({name:`elevations`,view:t.elevations,type:`f32`,access:`read`},{name:`sampleElevations`,view:t.sampleElevations,type:`f32`,access:`read_write`}),c.push({name:`samples`,view:t.samples,type:`f32`,access:`read_write`}),t.sampleTimes&&c.push({name:`sampleTimes`,view:t.sampleTimes,type:`f32`,access:`read_write`});let l=r?`fn getProgress(row: u32, firstRow: u32) -> f32 {
  _ = firstRow;
  return cumulativeLengths[cumulativeLengthsOffset + row];
}`:`fn getProgress(row: u32, firstRow: u32) -> f32 {
  return getRowTimeDifference(row, firstRow);
}`,u=t.sampleTimes?i?`time = f32(sampleIndex) * clockStep;`:r?`let startTime = getRowTimeDifference(row0, trackStart);
    let endTime = getRowTimeDifference(row1, trackStart);
    time = select(startTime + (endTime - startTime) * fraction, endTime, fraction >= 1.0);`:`time = targetProgress;`:``;return n(e,{id:t.id,operation:d,variant:i?`${o?`clock-words`:`clock`}${t.maximumGap?`-gap`:``}`:r?`arc-length`:t.maximumGap?`time-gap`:`time`,bindings:c,invocationCount:t.trackCount*t.sampleCount,declarations:`const ROW_COUNT: u32 = ${t.rowCount}u;
const SAMPLE_COUNT: u32 = ${t.sampleCount}u;
${a?h(o):``}
${i?_(o):``}
${p(t.geodesicRadius)}
${l}
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`,body:`let track = index / SAMPLE_COUNT;
  let sampleIndex = index - track * SAMPLE_COUNT;
  ${v}
  var position = vec2<f32>(0.0, 0.0);
  var elevation = 0.0;
  var time = 0.0;
  ${i?`let clockStep = getClockStep();
  var isAbsent = trackEnd <= trackStart;`:t.maximumGap?`var isAbsent = false;`:``}
  if (trackEnd > trackStart) {
    let lastRow = trackEnd - 1u;
    let total = getProgress(lastRow, trackStart);
    var targetProgress = 0.0;
    ${i?`targetProgress = getClockOffset(trackStart) + f32(sampleIndex) * clockStep;
    isAbsent = !(targetProgress >= 0.0 && targetProgress <= total);`:``}
    if (${i?`false`:`SAMPLE_COUNT > 1u`}) {
      targetProgress = select(
        total * (f32(sampleIndex) / f32(SAMPLE_COUNT - 1u)),
        total,
        sampleIndex == SAMPLE_COUNT - 1u
      );
    }
    var low = trackStart + 1u;
    var high = trackEnd;
    loop {
      if (low >= high) {
        break;
      }
      let middle = (low + high) / 2u;
      if (getProgress(middle, trackStart) > targetProgress) {
        high = middle;
      } else {
        low = middle + 1u;
      }
    }
    // Past the last row: the end of the last segment, so a great-circle track keeps its longitude
    // continuous across the antimeridian (planar: exactly the last row).
    var row0 = select(lastRow, lastRow - 1u, lastRow > trackStart);
    var row1 = lastRow;
    var fraction = 1.0;
    if (low < trackEnd) {
      row0 = low - 1u;
      row1 = low;
      let startProgress = getProgress(row0, trackStart);
      fraction = clamp(
        (targetProgress - startProgress) / (getProgress(row1, trackStart) - startProgress),
        0.0,
        1.0
      );
      ${t.maximumGap?`// Strictly inside a bracketing interval longer than the limit: absent (the fixes are kept).
      let gapLimit = maximumGap[maximumGapOffset];
      if (gapLimit > 0.0 && getRowTimeDifference(row1, row0) > gapLimit && targetProgress > startProgress) {
        isAbsent = true;
      }`:``}
    }
    let startPosition = getPosition(row0);
    let endPosition = getPosition(row1);
    position = interpolatePosition(startPosition, endPosition, fraction);
    ${s?`let startElevation = elevations[elevationsOffset + row0];
    let endElevation = elevations[elevationsOffset + row1];
    elevation = select(
      startElevation + (endElevation - startElevation) * fraction,
      endElevation,
      fraction >= 1.0
    );`:``}
    ${u}
  }
  ${i||t.maximumGap?`if (isAbsent) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    position = vec2<f32>(bitcast<f32>(0x7fc00000u | (index & 0u)));
    elevation = position.x;
  }`:``}
  samples[samplesOffset + 2u * index] = position.x;
  samples[samplesOffset + 2u * index + 1u] = position.y;
  ${s?`sampleElevations[sampleElevationsOffset + index] = elevation;`:``}
  ${t.sampleTimes?`sampleTimes[sampleTimesOffset + index] = time;`:``}
  _ = elevation;
  _ = time;`})}export{b as a,x as i,y as n,w as o,C as r,c as s,S as t};