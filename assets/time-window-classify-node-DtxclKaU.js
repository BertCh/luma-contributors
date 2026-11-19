import{l as e}from"./gpu-command-graph-BdzdmV6u.js";import{g as t,o as n}from"./wgsl-kernel-nodes-DagJklH2.js";import{c as r}from"./gpu-contributor-utils-BIwLQON5.js";import{t as i}from"./time-words-BFl8KCxq.js";function a(e,n=new Float32Array(8)){if(n.length<8)throw Error(`Time window target must hold 8 elements`);let r=e.startFadeDuration??0,i=e.endFadeDuration??0;for(let t of[e.start,e.end,r,i])if(!Number.isFinite(t))throw Error(`Time window values must be finite`);if(r<0||i<0)throw Error(`Time window fade durations must be non-negative`);let[a,o]=t(e.start),[s,c]=t(e.end);return n.set([a,o,s,c,r,i,0,0]),n}var o=`// Exact double-single a >= b. Backends with fast-math may reassociate the sum above and drop the
// low parts, so acceptance compares the high parts first and uses the low parts only on ties.
fn isTimeAtLeast(aHigh: f32, aLow: f32, bHigh: f32, bLow: f32) -> bool {
  return aHigh > bHigh || (aHigh == bHigh && aLow >= bLow);
}`;function s(t,i){let a=[],s=i.timestamps.format===`uint32x2`;c(i,s);let u=i.timestamps instanceof e,d=r(i.timestamps),f=(e,t)=>e?r(e)[t]:void 0;for(let[e,r]of d.entries()){if(r.length===0)continue;let c=f(i.timestampsLow,e),d=f(i.endTimestamps,e),p=f(i.endTimestampsLow,e),m=f(i.fadeWeights,e),h=f(i.clipFractions,e);if(s){a.push(l(t,i,e,u,r,f));continue}let g=[{name:`params`,view:i.window,type:`f32`,access:`read`},{name:`startHigh`,view:r,type:`f32`,access:`read`}];c&&g.push({name:`startLow`,view:c,type:`f32`,access:`read`}),d&&g.push({name:`endHigh`,view:d,type:`f32`,access:`read`}),p&&g.push({name:`endLow`,view:p,type:`f32`,access:`read`}),g.push({name:`outputMask`,view:f(i.mask,e),type:`u32`,access:`read_write`}),m&&g.push({name:`outputWeights`,view:m,type:`f32`,access:`read_write`}),h&&g.push({name:`outputClip`,view:h,type:`f32`,access:`read_write`});let _=p?`endLow[endLowOffset + index]`:d?`0.0`:`rowStartLow`;a.push(n(t,{id:u?`${i.id}-chunk-${e}`:i.id,operation:`GPUTimeWindowFilter`,variant:`classify`,bindings:g,invocationCount:r.length,declarations:`
// Double-single difference (aHigh + aLow) - (bHigh + bLow). High parts are subtracted first so
// nearby large epoch values cancel exactly before the small low parts are added. The result is
// used for fade and clip ratios, where rounding is harmless.
fn timeDifference(aHigh: f32, aLow: f32, bHigh: f32, bLow: f32) -> f32 {
  return (aHigh - bHigh) + (aLow - bLow);
}

${o}`,body:`
  let windowStartHigh = params[paramsOffset + 0u];
  let windowStartLow = params[paramsOffset + 1u];
  let windowEndHigh = params[paramsOffset + 2u];
  let windowEndLow = params[paramsOffset + 3u];
  let startFade = params[paramsOffset + 4u];
  let endFade = params[paramsOffset + 5u];

  let rowStartHigh = startHigh[startHighOffset + index];
  let rowStartLow = ${c?`startLow[startLowOffset + index]`:`0.0`};
  let rowEndHigh = ${d?`endHigh[endHighOffset + index]`:`rowStartHigh`};
  let rowEndLow = ${_};

  // Ordered comparisons are false for NaN, so NaN rows are rejected. A reversed window
  // (end < start) accepts nothing, interval rows included (B-13).
  let endAfterWindowStart = timeDifference(rowEndHigh, rowEndLow, windowStartHigh, windowStartLow);
  let startBeforeWindowEnd = timeDifference(windowEndHigh, windowEndLow, rowStartHigh, rowStartLow);
  let accepted =
    isTimeAtLeast(windowEndHigh, windowEndLow, windowStartHigh, windowStartLow) &&
    isTimeAtLeast(rowEndHigh, rowEndLow, windowStartHigh, windowStartLow) &&
    isTimeAtLeast(windowEndHigh, windowEndLow, rowStartHigh, rowStartLow);
  outputMask[outputMaskOffset + index] = select(0u, 1u, accepted);
  ${m?`var startRamp = 1.0;
  if (startFade > 0.0) { startRamp = clamp(endAfterWindowStart / startFade, 0.0, 1.0); }
  var endRamp = 1.0;
  if (endFade > 0.0) { endRamp = clamp(startBeforeWindowEnd / endFade, 0.0, 1.0); }
  outputWeights[outputWeightsOffset + index] = select(0.0, min(startRamp, endRamp), accepted);`:``}
  ${h?`let duration = timeDifference(rowEndHigh, rowEndLow, rowStartHigh, rowStartLow);
  var clipStart = 0.0;
  var clipEnd = 1.0;
  if (duration > 0.0) {
    clipStart = clamp(timeDifference(windowStartHigh, windowStartLow, rowStartHigh, rowStartLow) / duration, 0.0, 1.0);
    clipEnd = clamp(startBeforeWindowEnd / duration, 0.0, 1.0);
  }
  outputClip[outputClipOffset + 2u * index] = select(0.0, clipStart, accepted);
  outputClip[outputClipOffset + 2u * index + 1u] = select(0.0, clipEnd, accepted);`:``}`}))}return a}function c(e,t){let{id:n}=e;if(e.endTimestamps&&e.endTimestamps.format===`uint32x2`!==t)throw Error(`${n} endTimestamps must have the same format as timestamps`);if(t){if(e.timestampsLow||e.endTimestampsLow)throw Error(`${n} timestampsLow and endTimestampsLow require float32 timestamps`);if(e.window.format!==`uint32`)throw Error(`${n} uint32x2 word timestamps require a packed uint32 word window`);if(e.window.length<8)throw Error(`${n} word window must hold 8 uint32 values`)}else{if(e.window.format!==`float32`)throw Error(`${n} float32 timestamps require a float32 window`);if(e.window.length<8)throw Error(`${n} window must hold 8 float32 values`)}}function l(e,t,r,a,o,s){let c=s(t.endTimestamps,r),l=s(t.fadeWeights,r),u=s(t.clipFractions,r),d=[{name:`params`,view:t.window,type:`u32`,access:`read`},{name:`startWords`,view:o,type:`u32`,access:`read`}];return c&&d.push({name:`endWords`,view:c,type:`u32`,access:`read`}),d.push({name:`outputMask`,view:s(t.mask,r),type:`u32`,access:`read_write`}),l&&d.push({name:`outputWeights`,view:l,type:`f32`,access:`read_write`}),u&&d.push({name:`outputClip`,view:u,type:`f32`,access:`read_write`}),n(e,{id:a?`${t.id}-chunk-${r}`:t.id,operation:`GPUTimeWindowFilter`,variant:`classify-words`,bindings:d,invocationCount:o.length,declarations:i,body:`
  let windowStart = vec2<u32>(params[paramsOffset + 0u], params[paramsOffset + 1u]);
  let windowEnd = vec2<u32>(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let windowStartFraction = bitcast<f32>(params[paramsOffset + 4u]);
  let windowEndFraction = bitcast<f32>(params[paramsOffset + 5u]);
  let startFade = bitcast<f32>(params[paramsOffset + 6u]);
  let endFade = bitcast<f32>(params[paramsOffset + 7u]);

  let rowStart = vec2<u32>(
    startWords[startWordsOffset + 2u * index],
    startWords[startWordsOffset + 2u * index + 1u]
  );
  ${c?`let rowEnd = vec2<u32>(
    endWords[endWordsOffset + 2u * index],
    endWords[endWordsOffset + 2u * index + 1u]
  );`:`let rowEnd = rowStart;`}

  let endAfterWindowStart = timeWordsDifference(rowEnd, 0.0, windowStart, windowStartFraction);
  let startBeforeWindowEnd = timeWordsDifference(windowEnd, windowEndFraction, rowStart, 0.0);
  // A reversed window (end < start) accepts nothing, interval rows included (B-13).
  let accepted =
    isTimeWordsAtLeast(windowEnd, windowEndFraction, windowStart, windowStartFraction) &&
    isTimeWordsAtLeast(rowEnd, 0.0, windowStart, windowStartFraction) &&
    isTimeWordsAtLeast(windowEnd, windowEndFraction, rowStart, 0.0);
  outputMask[outputMaskOffset + index] = select(0u, 1u, accepted);
  ${l?`var startRamp = 1.0;
  if (startFade > 0.0) { startRamp = clamp(endAfterWindowStart / startFade, 0.0, 1.0); }
  var endRamp = 1.0;
  if (endFade > 0.0) { endRamp = clamp(startBeforeWindowEnd / endFade, 0.0, 1.0); }
  outputWeights[outputWeightsOffset + index] = select(0.0, min(startRamp, endRamp), accepted);`:``}
  ${u?`let duration = timeWordsDifference(rowEnd, 0.0, rowStart, 0.0);
  var clipStart = 0.0;
  var clipEnd = 1.0;
  if (duration > 0.0) {
    clipStart = clamp(timeWordsDifference(windowStart, windowStartFraction, rowStart, 0.0) / duration, 0.0, 1.0);
    clipEnd = clamp(startBeforeWindowEnd / duration, 0.0, 1.0);
  }
  outputClip[outputClipOffset + 2u * index] = select(0.0, clipStart, accepted);
  outputClip[outputClipOffset + 2u * index + 1u] = select(0.0, clipEnd, accepted);`:``}`})}export{s as n,a as r,o as t};