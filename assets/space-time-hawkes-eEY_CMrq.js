import{r as e,s as t}from"./kernel-C3tfQOn4.js";import{t as n}from"./gpu-sort-xRm8l5yn.js";import{s as r,v as i}from"./spatial-context-DwZEd4EZ.js";import{E as a,S as o,T as s,f as c,p as l,u,v as d,x as f,y as p}from"./gpu-contributor-utils-BIwLQON5.js";import{a as m,n as h}from"./trajectory-binding-validation-C89oJIAZ.js";import{i as g,u as _}from"./wgsl-common-DfyXiYB0.js";import{t as v}from"./scaled-quotient-wgsl-UDyhQag6.js";import{a as y,c as b,o as x}from"./wgsl-kernel-nodes-DagJklH2.js";import{i as S,n as C,r as w,t as T}from"./regression-row-tiles-7Xsw3Utq.js";import{i as E}from"./space-time-bands-Lbr2KxkP.js";import{t as D}from"./define-run-B-brkCe4.js";import{i as O,o as k}from"./events-CU2v4wkX.js";import{n as A,t as j}from"./panel-BovyVgLI.js";import{sn as M}from"./entry-preview-D70TdSbK.js";function N(e,t=new Float32Array(9)){let n=`getGPUSpaceTimeHawkesParameterValues`;if(t.length<9)throw Error(`${n} target must hold 9 elements`);let r=e.tolerance??1e-4,i=[[`windowArea`,e.windowArea],[`windowDuration`,e.windowDuration],[`maximumLag`,e.maximumLag],[`maximumDistance`,e.maximumDistance],[`tolerance`,r],[`initialBackground`,e.initialBackground],[`initialBranchingRatio`,e.initialBranchingRatio],[`initialDecayRate`,e.initialDecayRate],[`initialSpatialStandardDeviation`,e.initialSpatialStandardDeviation]];for(let[e,t]of i)if(!(Math.fround(t)>0)||!Number.isFinite(Math.fround(t)))throw Error(`${n} ${e} must be positive and finite`);return t.set(i.map(([,e])=>e)),t}var P=`
fn getHawkesBackgroundCompensator(background: f32, area: f32, duration: f32) -> f32 {
  let rate = frexp(background);
  let windowArea = frexp(area);
  let windowDuration = frexp(duration);
  let product = frexp(rate.fract * windowArea.fract * windowDuration.fract);
  return ldexp(product.fract, rate.exp + windowArea.exp + windowDuration.exp + product.exp);
}
`,F=11754944e-45,I=1e-30,L=2**-126,R={background:0,branchingRatio:1,decayRate:2,spatialVariance:3,sums:4,done:12,iterationCount:13,converged:14,iterationLimitReached:15,logLikelihood:16};function z(e){let t=R;return`${v}
const MAXIMUM_PARENTS: u32 = ${e}u;
const INTENSITY_FLOOR: f32 = ${b(F)};
const BRANCHING_RATIO_FLOOR: f32 = ${b(I)};
struct EventStatistics {
  background: f32,
  triggered: f32,
  lag: f32,
  distanceSquared: f32,
  logIntensity: f32,
  cut: f32
}
fn evaluateEvent(row: u32) -> EventStatistics {
  var result = EventStatistics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  let time = events[eventsOffset + 4u * row + 2u];
  let position = vec2f(events[eventsOffset + 4u * row], events[eventsOffset + 4u * row + 1u]);
  let background = state[stateOffset + ${t.background}u];
  let branchingRatio = state[stateOffset + ${t.branchingRatio}u];
  let decayRate = state[stateOffset + ${t.decayRate}u];
  let variance = state[stateOffset + ${t.spatialVariance}u];
  let logBackground = log(max(background, INTENSITY_FLOOR));
  let logProductivity = log(max(branchingRatio, BRANCHING_RATIO_FLOOR)) + log(decayRate) - log(6.2831853) - log(variance);
  let maximumDistance = params[paramsOffset + 3u];
  let maximumDistanceSquared = maximumDistance * maximumDistance;
  // First sorted row whose time is within the maximum lag of this event.
  let earliest = time - params[paramsOffset + 2u];
  var low = 0u;
  var high = row;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (events[eventsOffset + 4u * middle + 2u] < earliest) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let firstCandidate = low;
  // Simultaneous events cannot trigger each other. Locate the strict upper end before
  // applying the capacity so a timestamp tie never displaces genuinely earlier parents.
  high = row;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    if (events[eventsOffset + 4u * middle + 2u] < time) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  let parentEnd = low;
  low = firstCandidate;
  if (parentEnd - low > MAXIMUM_PARENTS) {
    low = parentEnd - MAXIMUM_PARENTS;
    result.cut = 1.0;
  }
  var maximumLog = logBackground;
  var backgroundWeight = 1.0;
  var parentSum = 0.0;
  var lagSum = 0.0;
  var distanceSum = 0.0;
  for (var parent = low; parent < parentEnd; parent++) {
    let lag = time - events[eventsOffset + 4u * parent + 2u];
    let delta = position - vec2f(events[eventsOffset + 4u * parent], events[eventsOffset + 4u * parent + 1u]);
    let distanceSquared = dot(delta, delta);
    if (!(lag > 0.0) || distanceSquared > maximumDistanceSquared) {
      continue;
    }
    let term = logProductivity - decayRate * lag - getScaledQuotient(distanceSquared, variance, 1.0, -1);
    if (term > maximumLog) {
      let scale = exp(maximumLog - term);
      backgroundWeight = backgroundWeight * scale;
      parentSum = parentSum * scale + 1.0;
      lagSum = lagSum * scale + lag;
      distanceSum = distanceSum * scale + distanceSquared;
      maximumLog = term;
    } else {
      let weight = exp(term - maximumLog);
      parentSum += weight;
      lagSum += weight * lag;
      distanceSum += weight * distanceSquared;
    }
  }
  let sum = backgroundWeight + parentSum;
  result.background = backgroundWeight / sum;
  result.triggered = parentSum / sum;
  result.lag = lagSum / sum;
  result.distanceSquared = distanceSum / sum;
  result.logIntensity = maximumLog + log(sum);
  return result;
}`}var B=`GPUSpaceTimeHawkes`,V=4294967295,H=1e3,U=4096,W=class{id;props;spatialContext;temporalContext;maximumIterations;maximumParents;estimatedCommandNodeCount;constructor(e){this.id=e.id??`space-time-hawkes`,this.props=e;let{id:t}=this;this.spatialContext=i(t,e.spatialContext,{...r,measure:`area`}),this.temporalContext=E(t,e.temporalContext),this.maximumIterations=e.maximumIterations??100,this.maximumParents=e.maximumParents??256,c(t,`maximumIterations`,this.maximumIterations,1,H),c(t,`maximumParents`,this.maximumParents,1,U),a(e.positions,[`float32x2`],`${t} positions`),a(e.times,[`float32`],`${t} times`),a(e.parameters,[`float32`],`${t} parameters`),s(t,e.positions.length>=2,`needs at least two events`),o(t,`positions length`,e.positions.length,T),s(t,e.times.length===e.positions.length,`times length must equal positions length`),s(t,e.parameters.length>=9,`parameters must hold 9 float32 values`);let{output:n}=e;a(n.estimates,[`float32`],`${t} output.estimates`),s(t,n.estimates.length>=4,`output.estimates must hold 4 rows`),n.logLikelihood&&(a(n.logLikelihood,[`float32`],`${t} output.logLikelihood`),s(t,n.logLikelihood.length===1,`output.logLikelihood must contain exactly one row`)),n.fittedLogLikelihood&&(a(n.fittedLogLikelihood,[`float32`],`${t} output.fittedLogLikelihood`),s(t,n.fittedLogLikelihood.length===1,`output.fittedLogLikelihood must contain exactly one row`)),n.backgroundProbability&&(a(n.backgroundProbability,[`float32`],`${t} output.backgroundProbability`),s(t,n.backgroundProbability.length>=e.positions.length,`output.backgroundProbability must hold one row per event`));for(let[e,r]of Object.entries(n.status??{}))r&&(a(r,[`uint32`],`${t} output.status.${e}`),s(t,r.length===1,`output.status.${e} must contain exactly one row`));let u=this.getOutputViews();m(t,[{estimates:n.estimates,logLikelihood:n.logLikelihood,fittedLogLikelihood:n.fittedLogLikelihood},{backgroundProbability:n.backgroundProbability},{...n.status}]),p(t,u,[e.positions,e.times,e.parameters]),this.estimatedCommandNodeCount=70+3*this.maximumIterations+(n.fittedLogLikelihood?2:0),l(t,this.estimatedCommandNodeCount)}getOutputViews(){let{output:e}=this.props;return[e.estimates,e.logLikelihood,e.fittedLogLikelihood,e.backgroundProbability,...Object.values(e.status??{})].filter(e=>e!==void 0)}getCommandNodes(r){let{id:i,props:a,maximumIterations:s,maximumParents:c}=this,{output:l}=a,p=a.positions.length;f(i,r,[a.positions,a.times,a.parameters,...this.getOutputViews()]),o(i,`positions length`,p,u(r.device,16));let m=C(p);d(r.device,i,`state`,68,`use a device with storage bindings`);for(let[e,n]of Object.entries({positions:a.positions,times:a.times,parameters:a.parameters,"output.estimates":l.estimates,"output.logLikelihood":l.logLikelihood,"output.backgroundProbability":l.backgroundProbability,...Object.fromEntries(Object.entries(l.status??{}).map(([e,t])=>[`output.status.${e}`,t]))}))n&&d(r.device,i,e,t(n).size,`reduce the view binding range`);h(r,i,a);for(let[e,t]of[[`event table`,p*16],[`sort rows`,p*4],[`sort histograms`,16*Math.ceil(p/256)*4],[`event partials`,m.tileCount*8*4],[`iteration dispatch`,24]])d(r.device,i,e,t,`reduce positions length`);let T=R,E=e(r,`${i}-state`,`float32`,17),D=[],O=e(r,`${i}-keys`,`uint32`,p),k=e(r,`${i}-row-ids`,`uint32`,p);D.push(x(r,{id:`${i}-row-keys`,operation:B,variant:`row-keys`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`times`,view:a.times,type:`f32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`keys`,view:O,type:`u32`,access:`read_write`},{name:`rowIds`,view:k,type:`u32`,access:`read_write`}],invocationCount:p,declarations:`${_}\n${g}`,body:`let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let time = times[timesOffset + index];
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    isFiniteValue(time) && time >= 0.0 && time <= params[paramsOffset + 1u];
  keys[keysOffset + index] = select(${V}u, getOrderedKey(time), included);
  rowIds[rowIdsOffset + index] = index;`}));let A=e(r,`${i}-sorted-keys`,`uint32`,p),j=e(r,`${i}-time-order`,`uint32`,p);D.push(...new n({id:`${i}-time-sort`,keys:O,values:k,outputKeys:A,outputValues:j,keyBits:32}).getCommandNodes(r));let M=e(r,`${i}-events`,`float32`,4*p);D.push(x(r,{id:`${i}-sorted-events`,operation:B,variant:`sorted-events`,bindings:[{name:`positions`,view:a.positions,type:`f32`,access:`read`},{name:`times`,view:a.times,type:`f32`,access:`read`},{name:`timeOrder`,view:j,type:`u32`,access:`read`},{name:`sortedKeys`,view:A,type:`u32`,access:`read`},{name:`events`,view:M,type:`f32`,access:`read_write`}],invocationCount:p,declarations:`fn getNaNBits() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`,body:`let row = timeOrder[timeOrderOffset + index];
  let kept = sortedKeys[sortedKeysOffset + index] != ${V}u;
  let nan = getNaNBits();
  events[eventsOffset + 4u * index] = select(nan, positions[positionsOffset + 2u * row], kept);
  events[eventsOffset + 4u * index + 1u] = select(nan, positions[positionsOffset + 2u * row + 1u], kept);
  events[eventsOffset + 4u * index + 2u] = select(nan, times[timesOffset + row], kept);
  events[eventsOffset + 4u * index + 3u] = 0.0;`}));let N=e(r,`${i}-partials`,`float32`,m.tileCount*8),[H,U]=w(m,8),W=y(r,{id:`${i}-iteration-dispatch`,operation:B,shapes:[H,U]}),K=Object.entries(T).map(([e,t])=>`const S_${e.replace(/[A-Z]/g,e=>`_${e}`).toUpperCase()}: u32 = ${t}u;`).join(`
`);D.push(x(r,{id:`${i}-initialize`,operation:B,variant:`initialize`,invocationCount:1,bindings:[{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`state`,view:E,type:`f32`,access:`read_write`},W.binding],declarations:`${K}\n${_}\n${G()}
const MINIMUM_SPATIAL_VARIANCE: f32 = ${b(L)};`,body:`let area = params[paramsOffset];
  let duration = params[paramsOffset + 1u];
  let lag = params[paramsOffset + 2u];
  let distance = params[paramsOffset + 3u];
  let tolerance = params[paramsOffset + 4u];
  let background = params[paramsOffset + 5u];
  let branchingRatio = params[paramsOffset + 6u];
  let decayRate = params[paramsOffset + 7u];
  let deviation = params[paramsOffset + 8u];
  let variance = deviation * deviation;
  let valid = area > 0.0 && duration > 0.0 && lag > 0.0 && distance > 0.0 && tolerance > 0.0 &&
    background > 0.0 && branchingRatio > 0.0 && decayRate > 0.0 && deviation > 0.0 &&
    isFiniteValue(area) && isFiniteValue(duration) && isFiniteValue(lag) && isFiniteValue(distance) &&
    isFiniteValue(tolerance) && isFiniteValue(background) && isFiniteValue(branchingRatio) && isFiniteValue(decayRate) && isFiniteValue(deviation) &&
    isFiniteValue(variance) && variance >= MINIMUM_SPATIAL_VARIANCE;
  let nan = getNaN();
  state[stateOffset + S_BACKGROUND] = select(nan, background, valid);
  state[stateOffset + S_BRANCHING_RATIO] = select(nan, branchingRatio, valid);
  state[stateOffset + S_DECAY_RATE] = select(nan, decayRate, valid);
  state[stateOffset + S_SPATIAL_VARIANCE] = select(nan, variance, valid);
  state[stateOffset + S_DONE] = select(1.0, 0.0, valid);
  state[stateOffset + S_ITERATION_COUNT] = 0.0;
  state[stateOffset + S_CONVERGED] = 0.0;
  state[stateOffset + S_ITERATION_LIMIT_REACHED] = 0.0;
  state[stateOffset + S_LOG_LIKELIHOOD] = nan;
  // Reused compiled graphs must not publish statistics from a previous valid fit.
  ${Array.from({length:8},(e,t)=>`state[stateOffset + S_SUMS + ${t}u] = 0.0;`).join(`
  `)}
  ${W.disarmWGSL}
  if (valid) {
    ${W.armWGSL}
  }`}));let q=z(c),J=[{name:`events`,view:M,type:`f32`,access:`read`},{name:`state`,view:E,type:`f32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`}],Y=T.sums;for(let e=0;e<s;e++)D.push(...S(r,{id:`${i}-iteration-${e}-sums`,operation:B,variant:`event-sums`,rowCount:p,layout:m,sumCount:8,bindings:J,declarations:`${K}\n${_}\n${q}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let time = events[eventsOffset + 4u * row + 2u];
  if (!isFiniteValue(time)) {
    return;
  }
  let statistics = evaluateEvent(row);
  let decayRate = state[stateOffset + S_DECAY_RATE];
  let exponent = decayRate * (params[paramsOffset + 1u] - time);
  // 1 - exp(-x), by its series for small x where the difference would cancel.
  let windowMass = select(1.0 - exp(-exponent), exponent - 0.5 * exponent * exponent, exponent < 1e-3);
  (*terms)[0] += statistics.background;
  (*terms)[1] += statistics.triggered;
  (*terms)[2] += statistics.lag;
  (*terms)[3] += statistics.distanceSquared;
  (*terms)[4] += statistics.logIntensity;
  (*terms)[5] += statistics.cut;
  (*terms)[6] += 1.0;
  (*terms)[7] += windowMass;
}`,partial:N,output:E,outputSlot:Y,tileGate:W.getGate(0),mergeGate:W.getGate(1)})),D.push(x(r,{id:`${i}-iteration-${e}-update`,operation:B,variant:`update`,invocationCount:1,bindings:[{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`state`,view:E,type:`f32`,access:`read_write`},W.binding],declarations:`${K}\n${G()}
const INTENSITY_FLOOR: f32 = ${b(F)};
const BRANCHING_RATIO_FLOOR: f32 = ${b(I)};
const MAXIMUM_ITERATIONS: f32 = ${s}.0;
${v}
${P}
fn sumAt(index: u32) -> f32 { return state[stateOffset + S_SUMS + index]; }
fn relativeChange(next: f32, previous: f32) -> f32 {
  return getScaledQuotient(abs(next - previous), max(abs(previous), INTENSITY_FLOOR), 1.0, 0);
}`,body:`if (state[stateOffset + S_DONE] != 0.0) {
    return;
  }
  let iterations = state[stateOffset + S_ITERATION_COUNT] + 1.0;
  state[stateOffset + S_ITERATION_COUNT] = iterations;
  let area = params[paramsOffset];
  let duration = params[paramsOffset + 1u];
  let tolerance = params[paramsOffset + 4u];
  let background = state[stateOffset + S_BACKGROUND];
  let branchingRatio = state[stateOffset + S_BRANCHING_RATIO];
  let decayRate = state[stateOffset + S_DECAY_RATE];
  let variance = state[stateOffset + S_SPATIAL_VARIANCE];
  let backgroundSum = sumAt(0u);
  let triggeredSum = sumAt(1u);
  let lagSum = sumAt(2u);
  let distanceSum = sumAt(3u);
  let eventCount = sumAt(6u);
  let windowMass = sumAt(7u);
  state[stateOffset + S_LOG_LIKELIHOOD] = sumAt(4u) - (getHawkesBackgroundCompensator(background, area, duration) + branchingRatio * windowMass);
  if (!(eventCount > 0.0) || !(windowMass > 0.0)) {
    let nan = getNaN();
    state[stateOffset + S_BACKGROUND] = nan;
    state[stateOffset + S_BRANCHING_RATIO] = nan;
    state[stateOffset + S_DECAY_RATE] = nan;
    state[stateOffset + S_SPATIAL_VARIANCE] = nan;
    state[stateOffset + S_DONE] = 1.0;
    ${W.disarmWGSL}
    return;
  }
  let nextBackground = getScaledQuotient(backgroundSum, area, duration, 0);
  let nextBranchingRatio = max(getScaledQuotient(triggeredSum, windowMass, 1.0, 0), BRANCHING_RATIO_FLOOR);
  var nextDecayRate = decayRate;
  var nextVariance = variance;
  if (triggeredSum > 1e-6 && lagSum > 0.0 && distanceSum > 0.0) {
    nextDecayRate = getScaledQuotient(triggeredSum, lagSum, 1.0, 0);
    nextVariance = getScaledQuotient(distanceSum, triggeredSum, 1.0, -1);
  }
  let change = max(
    max(relativeChange(nextBackground, background), relativeChange(nextBranchingRatio, branchingRatio)),
    max(relativeChange(nextDecayRate, decayRate), relativeChange(nextVariance, variance))
  );
  state[stateOffset + S_BACKGROUND] = nextBackground;
  state[stateOffset + S_BRANCHING_RATIO] = nextBranchingRatio;
  state[stateOffset + S_DECAY_RATE] = nextDecayRate;
  state[stateOffset + S_SPATIAL_VARIANCE] = nextVariance;
  let converged = change <= tolerance;
  if (converged || iterations >= MAXIMUM_ITERATIONS) {
    state[stateOffset + S_DONE] = 1.0;
    state[stateOffset + S_CONVERGED] = select(0.0, 1.0, converged);
    state[stateOffset + S_ITERATION_LIMIT_REACHED] = select(1.0, 0.0, converged);
    ${W.disarmWGSL}
  }`}));let X;l.fittedLogLikelihood?(X=e(r,`${i}-fitted-sums`,`float32`,2),D.push(...S(r,{id:`${i}-fitted-sums`,operation:B,variant:`fitted-sums`,rowCount:p,layout:m,sumCount:2,partial:N,output:X,bindings:[...J,...l.backgroundProbability?[{name:`timeOrder`,view:j,type:`u32`,access:`read`},{name:`backgroundProbability`,view:l.backgroundProbability,type:`f32`,access:`read_write`}]:[]],declarations:`${K}\n${_}\n${G()}\n${q}
fn accumulateRow(row: u32, terms: ptr<function, array<f32, TERM_COUNT>>) {
  let time = events[eventsOffset + 4u * row + 2u];
  let hasEstimates = isFiniteValue(state[stateOffset + S_BACKGROUND]) &&
    isFiniteValue(state[stateOffset + S_BRANCHING_RATIO]) &&
    isFiniteValue(state[stateOffset + S_DECAY_RATE]) &&
    isFiniteValue(state[stateOffset + S_SPATIAL_VARIANCE]);
  var probability = getNaN();
  if (hasEstimates && isFiniteValue(time)) {
    let statistics = evaluateEvent(row);
    probability = statistics.background;
    let exponent = state[stateOffset + S_DECAY_RATE] * (params[paramsOffset + 1u] - time);
    let windowMass = select(1.0 - exp(-exponent), exponent - 0.5 * exponent * exponent, exponent < 1e-3);
    (*terms)[0] += statistics.logIntensity;
    (*terms)[1] += windowMass;
  }
  ${l.backgroundProbability?`backgroundProbability[backgroundProbabilityOffset + timeOrder[timeOrderOffset + row]] = probability;`:``}
}`}))):l.backgroundProbability&&D.push(x(r,{id:`${i}-responsibilities`,operation:B,variant:`responsibilities`,bindings:[{name:`events`,view:M,type:`f32`,access:`read`},{name:`state`,view:E,type:`f32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`timeOrder`,view:j,type:`u32`,access:`read`},{name:`backgroundProbability`,view:l.backgroundProbability,type:`f32`,access:`read_write`}],invocationCount:p,declarations:`${K}\n${_}\n${G()}\n${q}`,body:`var probability = getNaN();
  // A parentless event returns backgroundWeight / sum = 1 even for NaN estimates.
  // Check the stored bit patterns before evaluating; NaN arithmetic cannot mark invalid fits.
  let hasEstimates = isFiniteValue(state[stateOffset + S_BACKGROUND]) &&
    isFiniteValue(state[stateOffset + S_BRANCHING_RATIO]) &&
    isFiniteValue(state[stateOffset + S_DECAY_RATE]) &&
    isFiniteValue(state[stateOffset + S_SPATIAL_VARIANCE]);
  if (hasEstimates && isFiniteValue(events[eventsOffset + 4u * index + 2u])) {
    probability = evaluateEvent(index).background;
  }
  backgroundProbability[backgroundProbabilityOffset + timeOrder[timeOrderOffset + index]] = probability;`})),D.push(x(r,{id:`${i}-publish-estimates`,operation:B,variant:`publish-estimates`,invocationCount:1,bindings:[{name:`state`,view:E,type:`f32`,access:`read`},{name:`estimates`,view:l.estimates,type:`f32`,access:`read_write`},...l.logLikelihood?[{name:`logLikelihood`,view:l.logLikelihood,type:`f32`,access:`read_write`}]:[],...l.fittedLogLikelihood&&X?[{name:`fittedSums`,view:X,type:`f32`,access:`read`},{name:`params`,view:a.parameters,type:`f32`,access:`read`},{name:`fittedLogLikelihood`,view:l.fittedLogLikelihood,type:`f32`,access:`read_write`}]:[]],declarations:`${K}\n${_}\n${G()}\n${P}`,body:`estimates[estimatesOffset] = state[stateOffset + S_BACKGROUND];
  estimates[estimatesOffset + 1u] = state[stateOffset + S_BRANCHING_RATIO];
  estimates[estimatesOffset + 2u] = state[stateOffset + S_DECAY_RATE];
  estimates[estimatesOffset + 3u] = sqrt(state[stateOffset + S_SPATIAL_VARIANCE]);
  ${l.logLikelihood?`logLikelihood[logLikelihoodOffset] = state[stateOffset + S_LOG_LIKELIHOOD];`:``}
  ${l.fittedLogLikelihood?`
  let valid = isFiniteValue(state[stateOffset + S_BACKGROUND]) &&
    isFiniteValue(state[stateOffset + S_BRANCHING_RATIO]) &&
    isFiniteValue(state[stateOffset + S_DECAY_RATE]) &&
    isFiniteValue(state[stateOffset + S_SPATIAL_VARIANCE]);
  fittedLogLikelihood[fittedLogLikelihoodOffset] = select(getNaN(),
    fittedSums[fittedSumsOffset] - (getHawkesBackgroundCompensator(state[stateOffset + S_BACKGROUND], params[paramsOffset], params[paramsOffset + 1u]) +
    state[stateOffset + S_BRANCHING_RATIO] * fittedSums[fittedSumsOffset + 1u]), valid);`:``}`}));let Z=l.status??{},Q=[[`converged`,Z.converged],[`iterationLimitReached`,Z.iterationLimitReached],[`iterationCount`,Z.iterationCount],[`candidateOverflow`,Z.candidateOverflow],[`invalidCount`,Z.invalidCount],[`approximation`,Z.approximation]].filter(([,e])=>e!==void 0);return Q.length>0&&D.push(x(r,{id:`${i}-publish-status`,operation:B,variant:`publish-status`,invocationCount:1,bindings:[{name:`state`,view:E,type:`f32`,access:`read`},...Q.map(([e,t])=>({name:e,view:t,type:`u32`,access:`read_write`}))],declarations:K,body:`${Z.converged?`converged[convergedOffset] = u32(state[stateOffset + S_CONVERGED]);`:``}
  ${Z.iterationLimitReached?`iterationLimitReached[iterationLimitReachedOffset] = u32(state[stateOffset + S_ITERATION_LIMIT_REACHED]);`:``}
  ${Z.iterationCount?`iterationCount[iterationCountOffset] = u32(state[stateOffset + S_ITERATION_COUNT]);`:``}
  ${Z.candidateOverflow?`candidateOverflow[candidateOverflowOffset] = u32(state[stateOffset + S_SUMS + 5u]);`:``}
  ${Z.invalidCount?`invalidCount[invalidCountOffset] = ${p}u - min(u32(max(state[stateOffset + S_SUMS + 6u], 0.0)), ${p}u);`:``}
  ${Z.approximation?`approximation[approximationOffset] = 1u;`:``}`})),D}};function G(){return`fn getNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`}var K=352,q=4,J=K/q,Y=[.06,.04,.94,.18],X=.22,Z=.08;function Q(e,t,n,r,i,a){let o=e.days[n]-e.days[t];if(!(o>0)||o>i)return 0;let s=(e.positions[n*2]-e.positions[t*2])**2+(e.positions[n*2+1]-e.positions[t*2+1])**2;if(s>a**2)return 0;let c=r[1],l=r[2],u=r[3];return c*l*Math.exp(-l*o)/(2*Math.PI*u*u)*Math.exp(-s/(2*u*u))}var $=D({parameters:M,caption:`USGS M4+ earthquakes, Türkiye 2023: color is P(background) (blue triggered, red background); strip is events per 4 days with the fitted rate. One constant background, no magnitude term (not ETAS).`,createInputs:({world:e})=>{let t=O([0,X,e.width,e.height]),n=new Float32Array(J);for(let e of t.days)n[Math.min(87,Math.floor(e/q))]++;return{...t,worldWidth:e.width,binCounts:n,maximumBinCount:Math.max(...n),intensityCache:{key:``,values:new Float64Array}}},build(e,t,n){let r=e.upload(`positions`,`float32x2`,t.positions),i=e.allocate(`parameters`,`float32`,9),a=e.allocate(`estimates`,`float32`,4),o=e.allocate(`background`,`float32`,t.count);return e.graph.add(new W({positions:r.view,times:e.upload(`times`,`float32`,t.days).view,parameters:i.view,maximumIterations:n.maximumIterations,maximumParents:256,output:{estimates:a.view,backgroundProbability:o.view}})),{outputs:{positions:r,background:o,estimates:a,panel:A(e,Y,{width:t.worldWidth,height:1}),bins:e.upload(`bins`,`float32`,t.binCounts)},update:e=>{i.write(N({windowArea:t.windowArea,windowDuration:K,maximumLag:e.maximumLag,maximumDistance:e.maximumDistance/t.kilometersPerUnit,initialBackground:.5*t.count/(t.windowArea*K),initialBranchingRatio:.5,initialDecayRate:.5,initialSpatialStandardDeviation:20/t.kilometersPerUnit}))}}},rebuildOn:[`maximumIterations`],buildCache:5,interaction:{kind:`probe`,hint:`hover an event`,rest:[.3,.45],resolve(e,{inputs:t,outputs:n,values:r,world:i,peek:a}){let o=a(n.estimates),s=o?`branchingRatio ${k(o[1])} · 1/decayRate ${k(1/o[2],1)} d · σ ${k(o[3]*t.kilometersPerUnit)} km`:`…`,c=o&&o.every(Number.isFinite)?ee(t,o,i.width):[],l=t.index.nearest(e.x,e.y);if(t.focus.row=l,l<0||!o||!o.every(Number.isFinite))return{glyphs:c,readout:s};let u=r.maximumDistance/t.kilometersPerUnit,d=(e,n)=>Q(t,e,n,o,r.maximumLag,u),f=`${Array.from(o).join()},${r.maximumLag},${u}`;t.intensityCache.key!==f&&(t.intensityCache.key=f,t.intensityCache.values=Float64Array.from({length:t.count},(e,n)=>{let r=o[0];for(let e=0;e<t.count;e++)r+=d(e,n);return r}));let p=t.intensityCache.values,m=e=>p[e],h=[t.positions[l*2],t.positions[l*2+1]],g=[...c,{shape:`crosshair`,at:h,size:12},{shape:`ring`,at:h,radius:2*o[3],dashed:!0}],_=0;for(let e=0;e<t.count;e++){let n=d(l,e);if(n===0)continue;let r=n/m(e);_+=r,r>=Z&&g.push({shape:`arrow`,from:h,to:[t.positions[e*2],t.positions[e*2+1]]})}let v=m(l),y=-1,b=o[0]/v;for(let e=0;e<t.count;e++){let t=d(e,l)/v;t>b&&(b=t,y=e)}let x=a(n.background)?.[l],S=`no parent`;return y>=0&&(S=`parent ${(t.days[l]-t.days[y]).toFixed(1)} d earlier, ${(Math.hypot(t.positions[l*2]-t.positions[y*2],t.positions[l*2+1]-t.positions[y*2+1])*t.kilometersPerUnit).toFixed(0)} km`,g.push({shape:`line`,from:[t.positions[y*2],t.positions[y*2+1]],to:h,dashed:!0})),{focus:l,spotlight:{at:h,radius:.14},glyphs:g,readout:`P(background) ${k(x)} · ${S} · M${t.magnitudes[l].toFixed(1)} · offspring ${_.toFixed(1)} · ${s}`}}},draw:({inputs:e,outputs:t,theme:n})=>[{layer:`dots`,positions:t.positions,values:t.background,ramp:`diverging`,range:[0,1],radius:4},j(t.panel,n),{layer:`chart`,kind:`bars`,values:t.bins,range:[0,e.maximumBinCount],rectangle:Y,color:[...n.categories[0],.9]}]});function ee(e,t,n){let[r,i,a,o]=Y,s=[];for(let c=0;c<J;c++){let l=c*q,u=l+q,d=t[0]*e.windowArea*q;for(let n of e.days){if(n>=u)continue;let e=Math.max(l,n);d+=t[1]*(Math.exp(-t[2]*(e-n))-Math.exp(-t[2]*(u-n)))}let f=(r+(c+.5)/J*(a-r))*n,p=i+Math.min(1,d/e.maximumBinCount)*(o-i);s.push(f,p)}return[{shape:`polyline`,points:s}]}export{$ as default};