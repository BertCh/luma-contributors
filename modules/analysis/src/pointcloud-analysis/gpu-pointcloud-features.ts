// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateViewLength
} from '../analysis-core/kernels/gpu-contributor-utils';
import {getSymmetricEigenWGSL} from '../analysis-core/kernels/symmetric-eigen-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY} from './pointcloud-limits';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';

// Eight sweeps cover 24 three-axis pair rotations, the previous rotation budget.
const POINT_CLOUD_FEATURES_MAXIMUM_EIGEN_SWEEPS = 8;

/** Float32 rows: viewpoint X/Y/Z and an enabled flag. */
export const GPU_POINT_CLOUD_FEATURES_PARAMETER_LENGTH = 4;

export type GPUPointCloudFeaturesSettings = {
  /** Orient normals toward this point; omitted chooses nonnegative z, then y, then x. */
  viewpoint?: readonly [number, number, number];
};

/** Pack per-frame orientation for a caller-owned GPUParameterBuffer. */
export function getGPUPointCloudFeaturesParameterValues(
  settings: GPUPointCloudFeaturesSettings = {}
): Float32Array {
  settings.viewpoint?.forEach((value, axis) =>
    validateFiniteNumber('gpu-pointcloud-features', `viewpoint[${axis}]`, value)
  );
  return new Float32Array([...(settings.viewpoint ?? [0, 0, 0]), settings.viewpoint ? 1 : 0]);
}

export type GPUPointCloudFeaturesProps = {
  /** Per-frame orientation; overrides viewpoint. Packed by getGPUPointCloudFeaturesParameterValues. */
  parameters?: GraphDataView<'float32'>;
  id?: string;
  positions: GraphDataView<'float32x3'>;
  indices: GraphDataView<'uint32'>;
  counts: GraphDataView<'uint32'>;
  capacity: number;
  /** Normal orientation toward this point; default chooses nonnegative z (then y, then x). */
  viewpoint?: readonly [number, number, number];
  normals: GraphDataView<'float32x3'>;
  /** Descending eigenvalues of population covariance. */
  eigenvalues: GraphDataView<'float32x3'>;
  /** Optional packed population covariance: xx, xy, xz, yy, yz, zz per row. */
  covariance?: GraphDataView<'float32'>;
  /** Linearity, planarity, scattering. */
  shapes: GraphDataView<'float32x3'>;
  /**
   * Per-row bit set, zero when clean. Bit 1 (`GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE`): fewer
   * than three neighbors, or collinear or coincident neighbors (middle eigenvalue at most 1e-6 of the
   * largest, or largest at most 1e-30); normals and shapes are zero. Bit 2 (`..._TRUNCATED`): `counts` exceeded `capacity`, so only the nearest
   * `capacity` neighbors were used. Bit 4 (`..._NOT_CONVERGED`): the Jacobi eigensolve did
   * not reach its tolerance within eight cyclic sweeps (24 pair rotations); the outputs are the last iterate. Bits above 4 are zero.
   */
  status: GraphDataView<'uint32'>;
};

/** `GPUPointCloudFeatures` status bit 1: fewer than three neighbors or a degenerate covariance. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE = 1;
/** `GPUPointCloudFeatures` status bit 2: the neighborhood was truncated to `capacity`. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED = 2;
/** `GPUPointCloudFeatures` status bit 4: the eigensolve did not converge. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED = 4;

/**
 * Centered 3D covariance and the shared symmetric Jacobi eigensolve on GPU neighborhoods.
 * Per-frame: input contents and parameters (orientation). Compile-time: capacity, view shapes and
 * optional outputs. Without parameters, viewpoint retains its constructor-time behavior.
 */
export class GPUPointCloudFeatures implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUPointCloudFeaturesProps) {
    this.id = props.id ?? 'gpu-pointcloud-features';
    validateCapacity(
      this.id,
      'capacity',
      props.capacity,
      1,
      GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY
    );
    validateViewLength(this.id, 'indices', props.indices, props.counts.length * props.capacity);
    validateViewLength(this.id, 'normals', props.normals, props.counts.length);
    validateViewLength(this.id, 'eigenvalues', props.eigenvalues, props.counts.length);
    validateViewLength(this.id, 'shapes', props.shapes, props.counts.length);
    validateViewLength(this.id, 'status', props.status, props.counts.length);
    if (props.covariance) {
      validateViewLength(this.id, 'covariance', props.covariance, props.counts.length * 6);
    }
    if (props.parameters) {
      validateViewLength(
        this.id,
        'parameters',
        props.parameters,
        GPU_POINT_CLOUD_FEATURES_PARAMETER_LENGTH
      );
    }
    props.viewpoint?.forEach((value, axis) =>
      validateFiniteNumber(this.id, `viewpoint[${axis}]`, value)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      positions,
      indices,
      counts,
      normals,
      eigenvalues,
      shapes,
      status,
      capacity,
      viewpoint,
      parameters,
      covariance
    } = this.props;
    const nodes = getPointCloudKernelNodes(graph, {
      id: this.id,
      length: counts.length,
      inputs: {positions, indices, counts},
      outputs: {
        normals,
        eigenvalues,
        shapes,
        status,
        ...(covariance ? {covariances: covariance} : {})
      },
      declarations: IS_FINITE_VALUE_WGSL + getSymmetricEigenWGSL(3),
      body: `
let count = min(readCounts(index), ${capacity}u);
var flags = select(0u, ${GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED}u, readCounts(index) > ${capacity}u);
${covariance ? 'for (var component = 0u; component < 6u; component++) { writeCovariances(index * 6u + component, 0.0); }' : ''}
writeNormals(index, vec3f(0.0)); writeEigenvalues(index, vec3f(0.0)); writeShapes(index, vec3f(0.0));
if (count < 3u) { writeStatus(index, flags | ${GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE}u); return; }
let origin = readPositions(readIndices(index * ${capacity}u));
var mean = vec3f(0.0);
for (var slot = 0u; slot < count; slot++) { mean += readPositions(readIndices(index * ${capacity}u + slot)) - origin; }
mean /= f32(count);
var covariance = mat3x3f(vec3f(0.0), vec3f(0.0), vec3f(0.0));
for (var slot = 0u; slot < count; slot++) {
  let difference = readPositions(readIndices(index * ${capacity}u + slot)) - origin - mean;
  covariance += mat3x3f(difference * difference.x, difference * difference.y, difference * difference.z) * (1.0 / f32(count));
}
${covariance ? `writeCovariances(index * 6u, covariance[0][0]); writeCovariances(index * 6u + 1u, covariance[0][1]); writeCovariances(index * 6u + 2u, covariance[0][2]); writeCovariances(index * 6u + 3u, covariance[1][1]); writeCovariances(index * 6u + 4u, covariance[1][2]); writeCovariances(index * 6u + 5u, covariance[2][2]);` : ''}
// The shared solver consumes a row-major lower triangle and returns ascending eigenpairs.
var matrix: array<f32, 9>;
for (var row = 0u; row < 3u; row++) {
  for (var column = 0u; column < 3u; column++) { matrix[row * 3u + column] = covariance[column][row]; }
}
var vectors: array<f32, 9>;
let decomposition = getSymmetricEigen_3(&matrix, &vectors, 3u, ${POINT_CLOUD_FEATURES_MAXIMUM_EIGEN_SWEEPS}u, 1e-6);
if (decomposition.status != 0u) { flags |= ${GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED}u; }
let values = vec3f(max(decomposition.values[0], 0.0), max(decomposition.values[1], 0.0), max(decomposition.values[2], 0.0));
var normal = safeNormal(vec3f(vectors[0], vectors[3], vectors[6]));
${parameters ? '' : viewpoint ? `if (dot(normal, vec3f(${viewpoint.map(Math.fround).join(',')}) - (origin+mean)) < 0.0) { normal = -normal; }` : 'if (normal.z < 0.0 || (normal.z == 0.0 && normal.y < 0.0) || (normal.z == 0.0 && normal.y == 0.0 && normal.x < 0.0)) { normal = -normal; }'}
let largest = max(max(values.x,values.y),values.z);
let least = min(min(values.x,values.y),values.z);
let middle = values.y;
writeEigenvalues(index, vec3f(largest, middle, least));
if (!(middle > largest * 1e-6) || !(largest > 1e-30)) { flags |= ${GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE}u; }
else { writeNormals(index, normal); writeShapes(index, vec3f((largest-middle)/largest, (middle-least)/largest, least/largest)); }
writeStatus(index, flags);`
    });
    if (parameters) {
      // Covariance output already uses all eight baseline storage bindings. Keep orientation in a
      // separate pass so parameters work with every output configuration on a baseline device.
      nodes.push(
        ...getPointCloudKernelNodes(graph, {
          id: `${this.id}-orient`,
          length: counts.length,
          inputs: {positions, indices, counts, parameters},
          outputs: {normals},
          body: `var normal = readNormals(index);
if (dot(normal, normal) < 0.5) { return; }
if (readParameters(3u) == 0.0) {
  if (normal.z < 0.0 || (normal.z == 0.0 && normal.y < 0.0) || (normal.z == 0.0 && normal.y == 0.0 && normal.x < 0.0)) { normal = -normal; }
} else {
  let count = min(readCounts(index), ${capacity}u);
  let origin = readPositions(readIndices(index * ${capacity}u));
  var mean = vec3f(0.0);
  for (var slot = 0u; slot < count; slot++) { mean += readPositions(readIndices(index * ${capacity}u + slot)) - origin; }
  mean /= f32(count);
  let viewpoint = vec3f(readParameters(0u), readParameters(1u), readParameters(2u));
  if (dot(normal, viewpoint - (origin + mean)) < 0.0) { normal = -normal; }
}
writeNormals(index, normal);`
        })
      );
    }
    return nodes;
  }
}
