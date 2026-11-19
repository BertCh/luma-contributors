import{a as e,n as t,r as n}from"./trajectory-binding-validation-C89oJIAZ.js";import{c as r,o as i}from"./wgsl-kernel-nodes-DagJklH2.js";import{r as a}from"./wgsl-common-DfyXiYB0.js";import{t as o}from"./time-words-BFl8KCxq.js";import{n as s,o as c,r as l}from"./trajectory-common-DFp8lX7h.js";import{s as u,v as d}from"./spatial-context-DwZEd4EZ.js";import{D as f,E as p,T as m,x as h,y as g}from"./gpu-contributor-utils-BIwLQON5.js";import{t as _}from"./define-run-B-brkCe4.js";import{a as v}from"./pointer-overlay-Qh8qflcA.js";import{t as y}from"./fleet-Cm7PfbI5.js";import{ir as b}from"./entry-preview-D70TdSbK.js";function x(e){let t=e.timeUnitsPerSecond??1,n=t===1?`elapsed`:`elapsed / ${r(t)}`;return e.times.format===`uint32x2`?{bindingType:`u32`,source:`${o}
fn getTimeWords(row: u32) -> vec2<u32> { return vec2<u32>(times[timesOffset + 2u * row], times[timesOffset + 2u * row + 1u]); }
fn isRowTimeFinite(row: u32) -> bool { _ = row; return true; }
fn rowElapsed(row: u32, previous: u32) -> f32 { return timeWordsDifferenceToF32(getTimeWords(row), getTimeWords(previous)); }`,deltaExpression:n}:{bindingType:`f32`,source:`fn isRowTimeFinite(row: u32) -> bool { return finite(times[timesOffset + row]); }
fn rowElapsed(row: u32, previous: u32) -> f32 { return times[timesOffset + row] - times[timesOffset + previous]; }`,deltaExpression:n}}function S(e,t,n){let r=`${a}
const IDENTITY = mat4x4<f32>(vec4<f32>(1.0,0.0,0.0,0.0),vec4<f32>(0.0,1.0,0.0,0.0),vec4<f32>(0.0,0.0,1.0,0.0),vec4<f32>(0.0,0.0,0.0,1.0));
const OBSERVATION = mat4x2<f32>(vec2<f32>(1.0,0.0),vec2<f32>(0.0,1.0),vec2<f32>(0.0),vec2<f32>(0.0));
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn transition(delta: f32) -> mat4x4<f32> { var result = IDENTITY; result[2][0] = delta; result[3][1] = delta; return result; }
fn covariance2(offset: u32) -> mat2x2<f32> { return mat2x2<f32>(parameters[parametersOffset + offset],parameters[parametersOffset + offset + 1u],parameters[parametersOffset + offset + 1u],parameters[parametersOffset + offset + 2u]); }
fn processCovariance(delta: f32) -> mat4x4<f32> {
  let noise = covariance2(0u); let first = delta * delta * delta / 3.0; let second = delta * delta / 2.0;
  return mat4x4<f32>(vec4<f32>(noise[0] * first, noise[0] * second),vec4<f32>(noise[1] * first, noise[1] * second),vec4<f32>(noise[0] * second, noise[0] * delta),vec4<f32>(noise[1] * second, noise[1] * delta));
}
fn symmetric(value: mat4x4<f32>) -> mat4x4<f32> { return (value + transpose(value)) * 0.5; }
fn covarianceScale(value: mat2x2<f32>) -> f32 { return max(max(abs(value[0][0]), abs(value[0][1])), max(abs(value[1][0]), abs(value[1][1]))); }
fn normalizedCovariance(value: mat2x2<f32>, scale: f32) -> mat2x2<f32> { return mat2x2<f32>(value[0] / scale, value[1] / scale); }
fn validCovariance(value: mat2x2<f32>, positive: bool) -> bool {
  if (!finite(value[0][0]) || !finite(value[0][1]) || !finite(value[1][1]) || value[0][0] < 0.0 || value[1][1] < 0.0) { return false; }
  let scale = covarianceScale(value); if (scale == 0.0) { return !positive; }
  let normalized = normalizedCovariance(value, scale);
  let determinant = normalized[0][0] * normalized[1][1] - normalized[0][1] * normalized[1][0];
  return select(determinant >= 0.0, determinant > 0.0, positive);
}
fn inverse2(value: mat2x2<f32>) -> mat2x2<f32> {
  let scale = covarianceScale(value); let normalized = normalizedCovariance(value, scale);
  let determinant = normalized[0][0] * normalized[1][1] - normalized[0][1] * normalized[1][0];
  return mat2x2<f32>(normalized[1][1],-normalized[0][1],-normalized[1][0],normalized[0][0]) * (1.0 / determinant) * (1.0 / scale);
}
struct InverseResult { value: mat4x4<f32>, valid: bool }
fn inverse4(value: mat4x4<f32>) -> InverseResult {
  var left = transpose(value); var right = IDENTITY;
  for (var column = 0u; column < 4u; column++) {
    var pivot = column;
    for (var row = column + 1u; row < 4u; row++) { if (abs(left[row][column]) > abs(left[pivot][column])) { pivot = row; } }
    let divisor = left[pivot][column]; if (!finite(divisor) || divisor == 0.0) { return InverseResult(IDENTITY, false); }
    let temporaryLeft = left[column]; left[column] = left[pivot]; left[pivot] = temporaryLeft;
    let temporaryRight = right[column]; right[column] = right[pivot]; right[pivot] = temporaryRight;
    left[column] *= 1.0 / divisor; right[column] *= 1.0 / divisor;
    for (var row = 0u; row < 4u; row++) { if (row != column) { let factor = left[row][column]; left[row] -= left[column] * factor; right[row] -= right[column] * factor; } }
  }
  return InverseResult(transpose(right), true);
}`,o=(e,t,n)=>{let r=t?`mat4x4<f32>`:`vec4<f32>`,i=t?16:4,a=e[0].toUpperCase()+e.slice(1);return`fn load${a}(row: u32) -> ${r} { let base = ${e}Offset + row * ${i}u; return ${r}(${Array.from({length:i},(t,n)=>`${e}[base + ${n}u]`).join(`,`)}); }
${n?`fn store${a}(row: u32, value: ${r}) { let base = ${e}Offset + row * ${i}u; ${Array.from({length:i},(n,r)=>`${e}[base + ${r}u] = value[${t?`${Math.floor(r/4)}][${r%4}`:r}];`).join(` `)} }`:``}`},c=x(n),l=c.bindingType===`u32`?`-words`:``,u=[i(e,{id:`${t}-filter`,operation:`GPUTrajectoryKalman`,variant:`full-covariance-filter${l}`,invocationCount:n.offsets.length-1,workgroupSize:32,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`times`,view:n.times,type:c.bindingType,access:`read`},{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`state`,view:n.output.filteredState,type:`f32`,access:`read_write`},{name:`covariance`,view:n.output.filteredCovariance,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],declarations:r+c.source+o(`state`,!1,!0)+o(`covariance`,!0,!0),body:`let start = min(offsets[offsetsOffset + index], ${n.positions.length}u); let end = min(offsets[offsetsOffset + index + 1u], ${n.positions.length}u);
let noise = covariance2(0u); let measurement = covariance2(3u); let initialVelocity = covariance2(6u); let maximumGap = ${s(`parameters[parametersOffset + 9u]`)};
let validParameters = validCovariance(noise, false) && validCovariance(measurement, true) && validCovariance(initialVelocity, false) && maximumGap > 0.0;
var initialized = false; var estimate = vec4<f32>(0.0); var uncertainty = IDENTITY; var previousRow = start;
for (var row = start; row < end; row++) {
  let missing = getNaN(); storeState(row, vec4<f32>(missing)); storeCovariance(row, mat4x4<f32>(vec4<f32>(missing),vec4<f32>(missing),vec4<f32>(missing),vec4<f32>(missing)));
  let elapsed = rowElapsed(row, previousRow); let delta = ${c.deltaExpression}; let position = vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); let measured = finite(position.x) && finite(position.y);
  if (!validParameters || !isRowTimeFinite(row) || (initialized && elapsed < 0.0)) { status[statusOffset + row] = 4u; initialized = false; continue; }
  if (initialized && elapsed > maximumGap) { initialized = false; }
  var flags = 0u;
  if (!initialized) {
    if (!measured) { status[statusOffset + row] = 6u; continue; }
    estimate = vec4<f32>(position,0.0,0.0);
    uncertainty = mat4x4<f32>(vec4<f32>(measurement[0],0.0,0.0),vec4<f32>(measurement[1],0.0,0.0),vec4<f32>(0.0,0.0,initialVelocity[0]),vec4<f32>(0.0,0.0,initialVelocity[1]));
    initialized = true; flags = 1u;
  } else {
    let prediction = transition(delta); estimate = prediction * estimate; uncertainty = prediction * uncertainty * transpose(prediction) + processCovariance(delta);
    if (measured) {
      let innovation = OBSERVATION * uncertainty * transpose(OBSERVATION) + measurement;
      if (!validCovariance(innovation, true)) { status[statusOffset + row] = 4u; initialized = false; continue; }
      let gain = uncertainty * transpose(OBSERVATION) * inverse2(innovation);
      estimate += gain * (position - estimate.xy);
      let correction = IDENTITY - gain * OBSERVATION;
      uncertainty = symmetric(correction * uncertainty * transpose(correction) + gain * measurement * transpose(gain));
    } else { flags = 2u; }
  }
  var valid = true;
  for (var component = 0u; component < 4u; component++) { valid = valid && finite(estimate[component]); for (var column = 0u; column < 4u; column++) { valid = valid && finite(uncertainty[column][component]); } }
  if (!valid) { status[statusOffset + row] = 4u; initialized = false; continue; }
  previousRow = row; storeState(row, estimate); storeCovariance(row, uncertainty); status[statusOffset + row] = flags;
}`})];return n.output.smoothedState&&n.output.smoothedCovariance&&u.push(i(e,{id:`${t}-smooth`,operation:`GPUTrajectoryKalman`,variant:`full-covariance-smooth${l}`,invocationCount:n.offsets.length-1,workgroupSize:32,bindings:[{name:`times`,view:n.times,type:c.bindingType,access:`read`},{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`state`,view:n.output.filteredState,type:`f32`,access:`read`},{name:`covariance`,view:n.output.filteredCovariance,type:`f32`,access:`read`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`},{name:`smoothedState`,view:n.output.smoothedState,type:`f32`,access:`read_write`},{name:`smoothedCovariance`,view:n.output.smoothedCovariance,type:`f32`,access:`read_write`}],declarations:r+c.source+o(`state`,!1,!1)+o(`covariance`,!0,!1)+o(`smoothedState`,!1,!0)+o(`smoothedCovariance`,!0,!0),body:`let start = min(offsets[offsetsOffset + index], ${n.positions.length}u); let end = min(offsets[offsetsOffset + index + 1u], ${n.positions.length}u);
var row = end;
loop {
  if (row <= start) { break; } row--;
  let estimate = loadState(row); let uncertainty = loadCovariance(row); storeSmoothedState(row, estimate); storeSmoothedCovariance(row, uncertainty);
  if (row + 1u >= end || (status[statusOffset + row] & 4u) != 0u || (status[statusOffset + row + 1u] & 5u) != 0u) { continue; }
  let elapsed = rowElapsed(row + 1u, row); let delta = ${c.deltaExpression}; let prediction = transition(delta); let predicted = prediction * uncertainty * transpose(prediction) + processCovariance(delta); let inverse = inverse4(predicted);
  if (!inverse.valid) { status[statusOffset + row] |= 8u; continue; }
  let gain = uncertainty * transpose(prediction) * inverse.value;
  let correctedState = estimate + gain * (loadSmoothedState(row + 1u) - prediction * estimate);
  let correctedCovariance = symmetric(uncertainty + gain * (loadSmoothedCovariance(row + 1u) - predicted) * transpose(gain));
  var valid = true;
  for (var component = 0u; component < 4u; component++) { valid = valid && finite(correctedState[component]); for (var column = 0u; column < 4u; column++) { valid = valid && finite(correctedCovariance[column][component]); } }
  if (!valid) { status[statusOffset + row] |= 8u; continue; }
  storeSmoothedState(row, correctedState); storeSmoothedCovariance(row, correctedCovariance);
}`})),u}var C=Object.freeze({...u,measure:`distance`}),w={second:1,millisecond:1e3,microsecond:1e6,nanosecond:1e9},T=class{id;props;spatialContext;constructor(t){this.id=t.id??`trajectory-kalman`,this.spatialContext=d(this.id,t.spatialContext,C);let{times:r,offsets:i}=l(this.id,t),a=c(this.id,r,t.temporalContext),o=a&&w[a.unit];m(this.id,t.timeUnitsPerSecond===void 0||(o??t.timeUnitsPerSecond)===t.timeUnitsPerSecond,`timeUnitsPerSecond must equal the ticks per second of the temporalContext unit`),m(this.id,a!==void 0||t.timeUnitsPerSecond===void 0||r.format===`uint32x2`,`timeUnitsPerSecond needs a temporalContext or uint32x2 times`),this.props={...t,times:r,offsets:i,timeUnitsPerSecond:o??t.timeUnitsPerSecond};let{props:s}=this;p(s.positions,[`float32x2`],this.id),p(s.times,[`float32`,`uint32x2`],this.id),m(this.id,s.timeUnitsPerSecond===void 0||Number.isFinite(Math.fround(s.timeUnitsPerSecond))&&Math.fround(s.timeUnitsPerSecond)>0,`timeUnitsPerSecond must be a positive finite number`),p(s.offsets,[`uint32`],this.id),p(s.parameters,[`float32`],this.id),f(this.id,`times`,s.times,s.positions.length),m(this.id,s.offsets.length>=1,`offsets must have at least 1 row`),f(this.id,`parameters`,s.parameters,s.covarianceMode===`full`?10:4);for(let e of[s.output.filteredState,s.output.filteredCovariance,s.output.smoothedState,s.output.smoothedCovariance])e&&(p(e,[`float32x4`],this.id),f(this.id,`output state or covariance`,e,s.positions.length*(s.covarianceMode===`full`&&(e===s.output.filteredCovariance||e===s.output.smoothedCovariance)?4:1)));p(s.output.status,[`uint32`],this.id),f(this.id,`output.status`,s.output.status,s.positions.length),m(this.id,!!s.output.smoothedState==!!s.output.smoothedCovariance,`output.smoothedState and output.smoothedCovariance must be given together`),n(this.id,this.props.output),e(this.id,[{filteredState:s.output.filteredState,filteredCovariance:s.output.filteredCovariance,status:s.output.status},{smoothedState:s.output.smoothedState,smoothedCovariance:s.output.smoothedCovariance,status:s.output.status}]),s.output.smoothedState&&g(this.id,[s.output.status,s.output.smoothedState,s.output.smoothedCovariance],[s.output.filteredState,s.output.filteredCovariance]),g(this.id,Object.values(s.output),[s.positions,s.times,s.offsets,s.parameters])}getCommandNodes(e){t(e,this.id,this.props);let{props:n,id:r}=this;if(h(r,e,[n.positions,n.times,n.offsets,n.parameters,...Object.values(n.output)]),n.covarianceMode===`full`)return S(e,r,n);let o=x(n),c=o.bindingType===`u32`?`-words`:``,l=`${a}
const OBSERVATION_COUNT: u32 = ${n.positions.length}u;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn transition(delta: f32) -> mat2x2<f32> { return mat2x2<f32>(1.0, 0.0, delta, 1.0); }
fn processCovariance(delta: f32, noise: f32) -> mat2x2<f32> {
  return mat2x2<f32>(delta * delta * delta / 3.0, delta * delta / 2.0, delta * delta / 2.0, delta) * noise;
}
fn unpackCovariance(packed: vec4<f32>) -> mat2x2<f32> { return mat2x2<f32>(packed.xy, packed.zw); }
fn packCovariance(matrix: mat2x2<f32>) -> vec4<f32> { return vec4<f32>(matrix[0][0], (matrix[0][1] + matrix[1][0]) * 0.5, (matrix[0][1] + matrix[1][0]) * 0.5, matrix[1][1]); }
${o.source}`,u=[i(e,{id:`${r}-filter`,operation:`GPUTrajectoryKalman`,variant:`filter${c}`,invocationCount:n.offsets.length-1,workgroupSize:32,bindings:[{name:`positions`,view:n.positions,type:`f32`,access:`read`},{name:`times`,view:n.times,type:o.bindingType,access:`read`},{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`state`,view:n.output.filteredState,type:`f32`,access:`read_write`},{name:`covariance`,view:n.output.filteredCovariance,type:`f32`,access:`read_write`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`}],declarations:l+E(!1),body:`let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT);
let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let noise = parameters[parametersOffset];
let measurementVariance = parameters[parametersOffset + 1u];
let velocityVariance = parameters[parametersOffset + 2u];
let maximumGap = ${s(`parameters[parametersOffset + 3u]`)};
let parametersValid = finite(noise) && noise >= 0.0 && finite(measurementVariance) && measurementVariance > 0.0 && finite(velocityVariance) && velocityVariance >= 0.0 && maximumGap > 0.0;
let invalidValue = getNaN();
var initialized = false;
var estimate = vec4<f32>(0.0);
var uncertainty = mat2x2<f32>();
var previousObservation = start;
for (var observation = start; observation < end; observation++) {
  let position = vec2<f32>(positions[positionsOffset + 2u * observation], positions[positionsOffset + 2u * observation + 1u]);
  let measured = finite(position.x) && finite(position.y);
  let elapsed = rowElapsed(observation, previousObservation);
  let delta = ${o.deltaExpression};
  var flags = 0u;
  storeState(observation, vec4<f32>(invalidValue));
  storeCovariance(observation, vec4<f32>(invalidValue));
  if (!parametersValid || !isRowTimeFinite(observation) || (initialized && elapsed < 0.0)) {
    status[statusOffset + observation] = 4u;
    initialized = false;
    continue;
  }
  if (initialized && elapsed > maximumGap) { initialized = false; }
  if (!initialized) {
    if (!measured) { status[statusOffset + observation] = 6u; continue; }
    estimate = vec4<f32>(position, 0.0, 0.0);
    uncertainty = mat2x2<f32>(measurementVariance, 0.0, 0.0, velocityVariance);
    initialized = true;
    flags = 1u;
  } else {
    let prediction = transition(delta);
    estimate = vec4<f32>(estimate.xy + delta * estimate.zw, estimate.zw);
    uncertainty = prediction * uncertainty * transpose(prediction) + processCovariance(delta, noise);
    if (measured) {
      let innovationVariance = uncertainty[0][0] + measurementVariance;
      if (!finite(innovationVariance) || !(innovationVariance > 0.0)) { status[statusOffset + observation] = 4u; initialized = false; continue; }
      let gain = vec2<f32>(uncertainty[0][0], uncertainty[0][1]) / innovationVariance;
      let residual = position - estimate.xy;
      estimate = vec4<f32>(estimate.xy + gain.x * residual, estimate.zw + gain.y * residual);
      // Joseph form preserves positive semidefiniteness under f32 rounding.
      let correction = mat2x2<f32>(1.0 - gain.x, -gain.y, 0.0, 1.0);
      uncertainty = correction * uncertainty * transpose(correction) + mat2x2<f32>(gain * gain.x, gain * gain.y) * measurementVariance;
    } else { flags = 2u; }
  }
  previousObservation = observation;
  let packed = packCovariance(uncertainty);
  if (!finite(estimate.x) || !finite(estimate.y) || !finite(estimate.z) || !finite(estimate.w) || !finite(packed.x) || !finite(packed.y) || !finite(packed.w)) {
    status[statusOffset + observation] = 4u; initialized = false; continue;
  }
  storeState(observation, estimate);
  storeCovariance(observation, packed);
  status[statusOffset + observation] = flags;
}`})];return n.output.smoothedState&&n.output.smoothedCovariance&&u.push(i(e,{id:`${r}-smooth`,operation:`GPUTrajectoryKalman`,variant:`smooth${c}`,invocationCount:n.offsets.length-1,workgroupSize:32,bindings:[{name:`times`,view:n.times,type:o.bindingType,access:`read`},{name:`offsets`,view:n.offsets,type:`u32`,access:`read`},{name:`parameters`,view:n.parameters,type:`f32`,access:`read`},{name:`state`,view:n.output.filteredState,type:`f32`,access:`read`},{name:`covariance`,view:n.output.filteredCovariance,type:`f32`,access:`read`},{name:`status`,view:n.output.status,type:`u32`,access:`read_write`},{name:`smoothedState`,view:n.output.smoothedState,type:`f32`,access:`read_write`},{name:`smoothedCovariance`,view:n.output.smoothedCovariance,type:`f32`,access:`read_write`}],declarations:l+E(!0),body:`let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT);
let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let noise = parameters[parametersOffset];
var observation = end;
loop {
  if (observation <= start) { break; }
  observation--;
  let estimate = loadState(observation);
  let uncertainty = unpackCovariance(loadCovariance(observation));
  storeSmoothedState(observation, estimate);
  storeSmoothedCovariance(observation, packCovariance(uncertainty));
  if (observation + 1u >= end || (status[statusOffset + observation] & 4u) != 0u || (status[statusOffset + observation + 1u] & 5u) != 0u) { continue; }
  let elapsed = rowElapsed(observation + 1u, observation);
  let delta = ${o.deltaExpression};
  let prediction = transition(delta);
  let predictedCovariance = prediction * uncertainty * transpose(prediction) + processCovariance(delta, noise);
  let determinant = predictedCovariance[0][0] * predictedCovariance[1][1] - predictedCovariance[0][1] * predictedCovariance[1][0];
  // Singular deterministic velocity covariance has no inverse; its forward estimate remains valid.
  if (!(determinant > 0.0) || !finite(determinant)) { status[statusOffset + observation] |= 8u; continue; }
  let inverse = mat2x2<f32>(predictedCovariance[1][1], -predictedCovariance[0][1], -predictedCovariance[1][0], predictedCovariance[0][0]) * (1.0 / determinant);
  let gain = uncertainty * transpose(prediction) * inverse;
  let nextEstimate = loadSmoothedState(observation + 1u);
  let residual = nextEstimate - vec4<f32>(estimate.xy + delta * estimate.zw, estimate.zw);
  let correctionX = gain * vec2<f32>(residual.x, residual.z);
  let correctionY = gain * vec2<f32>(residual.y, residual.w);
  storeSmoothedState(observation, estimate + vec4<f32>(correctionX.x, correctionY.x, correctionX.y, correctionY.y));
  let nextCovariance = unpackCovariance(loadSmoothedCovariance(observation + 1u));
  storeSmoothedCovariance(observation, packCovariance(uncertainty + gain * (nextCovariance - predictedCovariance) * transpose(gain)));
}`})),u}};function E(e){return[`state`,`covariance`,...e?[`smoothedState`,`smoothedCovariance`]:[]].map(t=>{let n=t[0].toUpperCase()+t.slice(1),r=`fn load${n}(row: u32) -> vec4<f32> { let offset = ${t}Offset + 4u * row; return vec4<f32>(${t}[offset], ${t}[offset + 1u], ${t}[offset + 2u], ${t}[offset + 3u]); }`;return e&&(t===`state`||t===`covariance`)?r:`${r}\nfn store${n}(row: u32, value: vec4<f32>) { let offset = ${t}Offset + 4u * row; ${t}[offset] = value.x; ${t}[offset + 1u] = value.y; ${t}[offset + 2u] = value.z; ${t}[offset + 3u] = value.w; }`}).join(`
`)}function D(e){let t=[];for(let n=0;n<e.length-1;n++)for(let r=e[n];r<e[n+1]-1;r++)t.push(r*2,(r+1)*2);return Uint32Array.from(t)}var O=_({parameters:b,caption:`Noisy fixes, forward Kalman estimate and smoothed track.`,createInputs:({random:e,world:t})=>{let n=y(t,{trackCount:8,minimumFixes:26,maximumFixes:36}),r=Float32Array.from({length:n.rowCount*2},()=>e()+e()-1);return{...n,unitNoise:r}},build(e,t){let n=e.allocate(`parameters`,`float32`,4),r=e.allocate(`noisy`,`float32x2`,t.rowCount),i=e.allocate(`filtered`,`float32x4`,t.rowCount),a=e.allocate(`filtered-covariance`,`float32x4`,t.rowCount),o=e.allocate(`smoothed`,`float32x4`,t.rowCount),s=e.allocate(`smoothed-covariance`,`float32x4`,t.rowCount),c=e.allocate(`status`,`uint32`,t.rowCount);return e.graph.add(new T({positions:r.view,times:e.upload(`times`,`float32`,t.timestamps).view,offsets:e.upload(`offsets`,`uint32`,t.trackOffsets).view,parameters:n.view,output:{filteredState:i.view,filteredCovariance:a.view,status:c.view,smoothedState:o.view,smoothedCovariance:s.view}})),{outputs:{noisy:r,filteredState:i,smoothedState:o,truth:e.upload(`truth`,`float32x4`,t.segments),edges:e.upload(`edges`,`uint32x2`,D(t.trackOffsets))},update:e=>{r.write(Float32Array.from(t.positions,(n,r)=>n+t.unitNoise[r]*e.noise)),n.write(Float32Array.of(e.accelerationDensity*1e-9,e.measurementVariance*1e-6,1e-5,1e3))}}},interaction:v(b,{kind:`pad`,hint:`model vs fixes`,x:{parameter:`accelerationDensity`,scale:`logarithmic`},y:{parameter:`measurementVariance`,scale:`logarithmic`},decorate:(e,t)=>({glyphs:[{shape:`crosshair`,at:[t.x,t.y]}],readout:`q ${e.accelerationDensity}e-9 · r ${e.measurementVariance}e-6`})}),draw:({inputs:e,outputs:t})=>[{layer:`lines`,context:!0,segments:t.truth,count:e.segmentCount},{layer:`dots`,context:!0,positions:t.noisy,radius:3.5},{layer:`segments`,nodes:t.filteredState,edges:t.edges,color:[.45,.75,1,.45],radius:3.5},{layer:`segments`,nodes:t.smoothedState,edges:t.edges,color:[1,.78,.25,1],radius:1.8}]});export{O as default};