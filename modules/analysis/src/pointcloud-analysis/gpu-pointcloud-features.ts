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
import {GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY} from './pointcloud-limits';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';

export type GPUPointCloudFeaturesProps = {
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
   * not reach its tolerance within 24 rotations (one pivot each); the outputs are the last iterate. Bits above 4 are zero.
   */
  status: GraphDataView<'uint32'>;
};

/** `GPUPointCloudFeatures` status bit 1: fewer than three neighbors or a degenerate covariance. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE = 1;
/** `GPUPointCloudFeatures` status bit 2: the neighborhood was truncated to `capacity`. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED = 2;
/** `GPUPointCloudFeatures` status bit 4: the eigensolve did not converge. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED = 4;

/** Centered 3D covariance and bounded symmetric Jacobi eigensolve on GPU neighborhoods. */
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
      covariance
    } = this.props;
    return getPointCloudKernelNodes(graph, {
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
var vectors = mat3x3f(vec3f(1.0,0.0,0.0), vec3f(0.0,1.0,0.0), vec3f(0.0,0.0,1.0));
var converged = false;
for (var iteration = 0u; iteration < 24u; iteration++) {
  var first = 0u; var second = 1u;
  if (abs(covariance[0][2]) > abs(covariance[first][second])) { first = 0u; second = 2u; }
  if (abs(covariance[1][2]) > abs(covariance[first][second])) { first = 1u; second = 2u; }
  let scale = max(max(abs(covariance[0][0]), abs(covariance[1][1])), abs(covariance[2][2]));
  if (abs(covariance[first][second]) <= max(scale * 1e-6, 1e-30)) { converged = true; break; }
  let angle = 0.5 * atan2(2.0 * covariance[first][second], covariance[second][second] - covariance[first][first]);
  let cosine = cos(angle); let sine = sin(angle);
  var rotation = mat3x3f(vec3f(1.0,0.0,0.0), vec3f(0.0,1.0,0.0), vec3f(0.0,0.0,1.0));
  rotation[first][first] = cosine; rotation[second][second] = cosine;
  rotation[first][second] = -sine; rotation[second][first] = sine;
  covariance = transpose(rotation) * covariance * rotation;
  vectors = vectors * rotation;
}
if (!converged) { flags |= ${GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED}u; }
var values = vec3f(max(covariance[0][0],0.0), max(covariance[1][1],0.0), max(covariance[2][2],0.0));
var smallest = 0u;
for (var axis = 1u; axis < 3u; axis++) { if (values[axis] < values[smallest]) { smallest = axis; } }
var normal = safeNormal(vectors[smallest]);
${viewpoint ? `if (dot(normal, vec3f(${viewpoint.map(Math.fround).join(',')}) - (origin+mean)) < 0.0) { normal = -normal; }` : 'if (normal.z < 0.0 || (normal.z == 0.0 && normal.y < 0.0) || (normal.z == 0.0 && normal.y == 0.0 && normal.x < 0.0)) { normal = -normal; }'}
let largest = max(max(values.x,values.y),values.z);
let least = min(min(values.x,values.y),values.z);
let middle = values.x + values.y + values.z - largest - least;
writeEigenvalues(index, vec3f(largest, middle, least));
if (!(middle > largest * 1e-6) || !(largest > 1e-30)) { flags |= ${GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE}u; }
else { writeNormals(index, normal); writeShapes(index, vec3f((largest-middle)/largest, (middle-least)/largest, least/largest)); }
writeStatus(index, flags);`
    });
  }
}
