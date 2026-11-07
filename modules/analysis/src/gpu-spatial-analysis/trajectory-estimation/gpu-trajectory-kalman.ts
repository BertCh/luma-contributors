// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createFullCovarianceKalmanNodes} from './trajectory-kalman-full-covariance';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Per-observation flags. Reset marks the beginning of a new independently estimated segment. */
export const GPU_TRAJECTORY_KALMAN_STATUS = {
  reset: 1,
  predictionOnly: 2,
  invalid: 4,
  smoothingSingular: 8
} as const;

export type GPUTrajectoryKalmanProps = {
  id?: string;
  /** Projected planar observations. NaN/infinite coordinates mean missing measurement. */
  positions: GraphDataView<'float32x2'>;
  /** Relative time in seconds, finite and nondecreasing inside each track. */
  times: GraphDataView<'float32'>;
  offsets: GraphDataView<'uint32'>;
  /** Full covariance uses [qxx,qxy,qyy,rxx,rxy,ryy,vxx,vxy,vyy,maximumGap]. */
  covarianceMode?: 'isotropic' | 'full';
  /** Isotropic: acceleration spectral density, measurement variance, initial velocity variance, maximum gap. */
  parameters: GraphDataView<'float32'>;
  output: {
    /** [x,y,vx,vy], always the forward filtered estimate. */
    filteredState: GraphDataView<'float32x4'>;
    /** Isotropic: one [positionVariance,covariance,covariance,velocityVariance] row per observation. Full: four column-major rows per observation. */
    filteredCovariance: GraphDataView<'float32x4'>;
    status: GraphDataView<'uint32'>;
    smoothedState?: GraphDataView<'float32x4'>;
    smoothedCovariance?: GraphDataView<'float32x4'>;
  };
};

/** Constant-velocity Kalman filter and optional Rauch–Tung–Striebel smoother, parallel across tracks. */
export class GPUTrajectoryKalman implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryKalmanProps;
  constructor(props: GPUTrajectoryKalmanProps) {
    this.id = props.id ?? 'trajectory-kalman';
    this.props = props;
    validatePackedView(props.positions, ['float32x2'], this.id);
    validatePackedView(props.times, ['float32'], this.id);
    validatePackedView(props.offsets, ['uint32'], this.id);
    validatePackedView(props.parameters, ['float32'], this.id);
    assert(
      props.times.length === props.positions.length &&
        props.offsets.length >= 1 &&
        props.parameters.length === (props.covarianceMode === 'full' ? 10 : 4)
    );
    for (const view of [
      props.output.filteredState,
      props.output.filteredCovariance,
      props.output.smoothedState,
      props.output.smoothedCovariance
    ]) {
      if (view) {
        validatePackedView(view, ['float32x4'], this.id);
        assert(
          view.length ===
            props.positions.length *
              (props.covarianceMode === 'full' &&
              (view === props.output.filteredCovariance || view === props.output.smoothedCovariance)
                ? 4
                : 1)
        );
      }
    }
    validatePackedView(props.output.status, ['uint32'], this.id);
    assert(props.output.status.length === props.positions.length);
    assert(Boolean(props.output.smoothedState) === Boolean(props.output.smoothedCovariance));
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.times,
      props.offsets,
      props.parameters
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.times,
      props.offsets,
      props.parameters,
      ...Object.values(props.output)
    ]);
    if (props.covarianceMode === 'full') return createFullCovarianceKalmanNodes(graph, id, props);
    const declarations = `${GET_NAN_WGSL}
const OBSERVATION_COUNT: u32 = ${props.positions.length}u;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn transition(delta: f32) -> mat2x2<f32> { return mat2x2<f32>(1.0, 0.0, delta, 1.0); }
fn processCovariance(delta: f32, noise: f32) -> mat2x2<f32> {
  return mat2x2<f32>(delta * delta * delta / 3.0, delta * delta / 2.0, delta * delta / 2.0, delta) * noise;
}
fn unpackCovariance(packed: vec4<f32>) -> mat2x2<f32> { return mat2x2<f32>(packed.xy, packed.zw); }
fn packCovariance(matrix: mat2x2<f32>) -> vec4<f32> { return vec4<f32>(matrix[0][0], (matrix[0][1] + matrix[1][0]) * 0.5, (matrix[0][1] + matrix[1][0]) * 0.5, matrix[1][1]); }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-filter`,
        operation: 'GPUTrajectoryKalman',
        variant: 'filter',
        invocationCount: props.offsets.length - 1,
        workgroupSize: 32,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'times', view: props.times, type: 'f32', access: 'read'},
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
        declarations: declarations + getStateAccessors(false),
        body: `let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT);
let end = min(offsets[offsetsOffset + index + 1u], OBSERVATION_COUNT);
let noise = parameters[parametersOffset];
let measurementVariance = parameters[parametersOffset + 1u];
let velocityVariance = parameters[parametersOffset + 2u];
let maximumGap = parameters[parametersOffset + 3u];
let parametersValid = finite(noise) && noise >= 0.0 && finite(measurementVariance) && measurementVariance > 0.0 && finite(velocityVariance) && velocityVariance >= 0.0 && maximumGap > 0.0;
let invalidValue = getNaN();
var initialized = false;
var estimate = vec4<f32>(0.0);
var uncertainty = mat2x2<f32>();
var previousTime = 0.0;
for (var observation = start; observation < end; observation++) {
  let time = times[timesOffset + observation];
  let position = vec2<f32>(positions[positionsOffset + 2u * observation], positions[positionsOffset + 2u * observation + 1u]);
  let measured = finite(position.x) && finite(position.y);
  let delta = time - previousTime;
  var flags = 0u;
  storeState(observation, vec4<f32>(invalidValue));
  storeCovariance(observation, vec4<f32>(invalidValue));
  if (!parametersValid || !finite(time) || (initialized && delta < 0.0)) {
    status[statusOffset + observation] = 4u;
    initialized = false;
    continue;
  }
  if (initialized && delta > maximumGap) { initialized = false; }
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
  previousTime = time;
  let packed = packCovariance(uncertainty);
  if (!finite(estimate.x) || !finite(estimate.y) || !finite(estimate.z) || !finite(estimate.w) || !finite(packed.x) || !finite(packed.y) || !finite(packed.w)) {
    status[statusOffset + observation] = 4u; initialized = false; continue;
  }
  storeState(observation, estimate);
  storeCovariance(observation, packed);
  status[statusOffset + observation] = flags;
}`
      })
    ];
    if (props.output.smoothedState && props.output.smoothedCovariance) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-smooth`,
          operation: 'GPUTrajectoryKalman',
          variant: 'smooth',
          invocationCount: props.offsets.length - 1,
          workgroupSize: 32,
          bindings: [
            {name: 'times', view: props.times, type: 'f32', access: 'read'},
            {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
            {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
            {name: 'state', view: props.output.filteredState, type: 'f32', access: 'read'},
            {
              name: 'covariance',
              view: props.output.filteredCovariance,
              type: 'f32',
              access: 'read'
            },
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
          declarations: declarations + getStateAccessors(true),
          body: `let start = min(offsets[offsetsOffset + index], OBSERVATION_COUNT);
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
  let delta = times[timesOffset + observation + 1u] - times[timesOffset + observation];
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
}`
        })
      );
    }
    return nodes;
  }
}

function getStateAccessors(smoothing: boolean): string {
  return ['state', 'covariance', ...(smoothing ? ['smoothedState', 'smoothedCovariance'] : [])]
    .map(name => {
      const capitalized = name[0].toUpperCase() + name.slice(1);
      const load = `fn load${capitalized}(row: u32) -> vec4<f32> { let offset = ${name}Offset + 4u * row; return vec4<f32>(${name}[offset], ${name}[offset + 1u], ${name}[offset + 2u], ${name}[offset + 3u]); }`;
      if (smoothing && (name === 'state' || name === 'covariance')) return load;
      return `${load}\nfn store${capitalized}(row: u32, value: vec4<f32>) { let offset = ${name}Offset + 4u * row; ${name}[offset] = value.x; ${name}[offset + 1u] = value.y; ${name}[offset + 2u] = value.z; ${name}[offset + 3u] = value.w; }`;
    })
    .join('\n');
}
