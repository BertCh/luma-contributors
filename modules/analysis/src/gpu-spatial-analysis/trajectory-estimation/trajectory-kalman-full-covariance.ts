// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../../dataframe-analysis/time-window-filter/time-words';
import {getEffectiveMaximumGapWGSL} from '../trajectory-analysis/trajectory-common';
import type {GPUTrajectoryKalmanResolvedProps} from './gpu-trajectory-kalman';

/** Filter/smoother time helpers: the `times` binding type, WGSL and the `delta` scaling. @internal */
export function getKalmanTimeSource(props: GPUTrajectoryKalmanResolvedProps): {
  bindingType: 'f32' | 'u32';
  source: string;
  deltaExpression: string;
} {
  const timeUnitsPerSecond = props.timeUnitsPerSecond ?? 1;
  const deltaExpression =
    timeUnitsPerSecond === 1 ? 'elapsed' : `elapsed / ${getWGSLFloatLiteral(timeUnitsPerSecond)}`;
  if (props.times.format !== 'uint32x2') {
    return {
      bindingType: 'f32',
      source: `fn isRowTimeFinite(row: u32) -> bool { return finite(times[timesOffset + row]); }
fn rowElapsed(row: u32, previous: u32) -> f32 { return times[timesOffset + row] - times[timesOffset + previous]; }`,
      deltaExpression
    };
  }
  return {
    bindingType: 'u32',
    source: `${TIME_WORDS_WGSL}
fn getTimeWords(row: u32) -> vec2<u32> { return vec2<u32>(times[timesOffset + 2u * row], times[timesOffset + 2u * row + 1u]); }
fn isRowTimeFinite(row: u32) -> bool { _ = row; return true; }
fn rowElapsed(row: u32, previous: u32) -> f32 { return timeWordsDifferenceToF32(getTimeWords(row), getTimeWords(previous)); }`,
    deltaExpression
  };
}

/** Full correlated-axis covariance, with Joseph updates and pivoted backward covariance solves. */
export function createFullCovarianceKalmanNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUTrajectoryKalmanResolvedProps
): GPUCommandNode<Parameters>[] {
  const declarations = `${GET_NAN_WGSL}
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
}`;
  const accessors = (name: string, matrix: boolean, writable: boolean) => {
    const type = matrix ? 'mat4x4<f32>' : 'vec4<f32>';
    const width = matrix ? 16 : 4;
    const capitalized = name[0].toUpperCase() + name.slice(1);
    return `fn load${capitalized}(row: u32) -> ${type} { let base = ${name}Offset + row * ${width}u; return ${type}(${Array.from({length: width}, (_, component) => `${name}[base + ${component}u]`).join(',')}); }
${writable ? `fn store${capitalized}(row: u32, value: ${type}) { let base = ${name}Offset + row * ${width}u; ${Array.from({length: width}, (_, component) => `${name}[base + ${component}u] = value[${matrix ? `${Math.floor(component / 4)}][${component % 4}` : component}];`).join(' ')} }` : ''}`;
  };
  const time = getKalmanTimeSource(props);
  const wordSuffix = time.bindingType === 'u32' ? '-words' : '';
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-filter`,
      operation: 'GPUTrajectoryKalman',
      variant: `full-covariance-filter${wordSuffix}`,
      invocationCount: props.offsets.length - 1,
      workgroupSize: 32,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'times', view: props.times, type: time.bindingType, access: 'read'},
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'state', view: props.output.filteredState, type: 'f32', access: 'read_write'},
        {
          name: 'covariance',
          view: props.output.filteredCovariance,
          type: 'f32',
          access: 'read_write'
        },
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations:
        declarations +
        time.source +
        accessors('state', false, true) +
        accessors('covariance', true, true),
      body: `let start = min(offsets[offsetsOffset + index], ${props.positions.length}u); let end = min(offsets[offsetsOffset + index + 1u], ${props.positions.length}u);
let noise = covariance2(0u); let measurement = covariance2(3u); let initialVelocity = covariance2(6u); let maximumGap = ${getEffectiveMaximumGapWGSL('parameters[parametersOffset + 9u]')};
let validParameters = validCovariance(noise, false) && validCovariance(measurement, true) && validCovariance(initialVelocity, false) && maximumGap > 0.0;
var initialized = false; var estimate = vec4<f32>(0.0); var uncertainty = IDENTITY; var previousRow = start;
for (var row = start; row < end; row++) {
  let missing = getNaN(); storeState(row, vec4<f32>(missing)); storeCovariance(row, mat4x4<f32>(vec4<f32>(missing),vec4<f32>(missing),vec4<f32>(missing),vec4<f32>(missing)));
  let elapsed = rowElapsed(row, previousRow); let delta = ${time.deltaExpression}; let position = vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); let measured = finite(position.x) && finite(position.y);
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
}`
    })
  ];
  if (props.output.smoothedState && props.output.smoothedCovariance)
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-smooth`,
        operation: 'GPUTrajectoryKalman',
        variant: `full-covariance-smooth${wordSuffix}`,
        invocationCount: props.offsets.length - 1,
        workgroupSize: 32,
        bindings: [
          {name: 'times', view: props.times, type: time.bindingType, access: 'read'},
          {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'state', view: props.output.filteredState, type: 'f32', access: 'read'},
          {name: 'covariance', view: props.output.filteredCovariance, type: 'f32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {
            name: 'smoothedState',
            view: props.output.smoothedState,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'smoothedCovariance',
            view: props.output.smoothedCovariance,
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations:
          declarations +
          time.source +
          accessors('state', false, false) +
          accessors('covariance', true, false) +
          accessors('smoothedState', false, true) +
          accessors('smoothedCovariance', true, true),
        body: `let start = min(offsets[offsetsOffset + index], ${props.positions.length}u); let end = min(offsets[offsetsOffset + index + 1u], ${props.positions.length}u);
var row = end;
loop {
  if (row <= start) { break; } row--;
  let estimate = loadState(row); let uncertainty = loadCovariance(row); storeSmoothedState(row, estimate); storeSmoothedCovariance(row, uncertainty);
  if (row + 1u >= end || (status[statusOffset + row] & 4u) != 0u || (status[statusOffset + row + 1u] & 5u) != 0u) { continue; }
  let elapsed = rowElapsed(row + 1u, row); let delta = ${time.deltaExpression}; let prediction = transition(delta); let predicted = prediction * uncertainty * transpose(prediction) + processCovariance(delta); let inverse = inverse4(predicted);
  if (!inverse.valid) { status[statusOffset + row] |= 8u; continue; }
  let gain = uncertainty * transpose(prediction) * inverse.value;
  let correctedState = estimate + gain * (loadSmoothedState(row + 1u) - prediction * estimate);
  let correctedCovariance = symmetric(uncertainty + gain * (loadSmoothedCovariance(row + 1u) - predicted) * transpose(gain));
  var valid = true;
  for (var component = 0u; component < 4u; component++) { valid = valid && finite(correctedState[component]); for (var column = 0u; column < 4u; column++) { valid = valid && finite(correctedCovariance[column][component]); } }
  if (!valid) { status[statusOffset + row] |= 8u; continue; }
  storeSmoothedState(row, correctedState); storeSmoothedCovariance(row, correctedCovariance);
}`
      })
    );
  return nodes;
}
