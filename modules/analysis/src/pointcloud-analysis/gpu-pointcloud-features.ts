// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getSpatialOperationNodes} from '@luma.gl/graph';

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
  /** Bit 1 insufficient/degenerate, bit 2 truncated, bit 4 eigensolve did not converge. */
  status: GraphDataView<'uint32'>;
};

/** Centered 3D covariance and bounded symmetric Jacobi eigensolve on GPU neighborhoods. */
export class GPUPointCloudFeatures {
  readonly id: string;
  constructor(readonly props: GPUPointCloudFeaturesProps) {
    this.id = props.id ?? 'gpu-pointcloud-features';
    assert(Number.isInteger(props.capacity) && props.capacity > 0 && props.capacity <= 64);
    assert(props.indices.length === props.counts.length * props.capacity);
    for (const view of [props.counts, props.normals, props.eigenvalues, props.shapes, props.status])
      assert(view.length === props.counts.length);
    assert(!props.covariance || props.covariance.length === props.counts.length * 6);
    assert(!props.viewpoint || props.viewpoint.every(value => Number.isFinite(Math.fround(value))));
  }
  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
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
    return getSpatialOperationNodes(graph, {
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
var flags = select(0u, 2u, readCounts(index) > ${capacity}u);
${covariance ? 'for (var component = 0u; component < 6u; component++) { writeCovariances(index * 6u + component, 0.0); }' : ''}
writeNormals(index, vec3f(0.0)); writeEigenvalues(index, vec3f(0.0)); writeShapes(index, vec3f(0.0));
if (count < 3u) { writeStatus(index, flags | 1u); return; }
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
if (!converged) { flags |= 4u; }
var values = vec3f(max(covariance[0][0],0.0), max(covariance[1][1],0.0), max(covariance[2][2],0.0));
var smallest = 0u;
for (var axis = 1u; axis < 3u; axis++) { if (values[axis] < values[smallest]) { smallest = axis; } }
var normal = safeNormal(vectors[smallest]);
${viewpoint ? `if (dot(normal, vec3f(${viewpoint.map(Math.fround).join(',')}) - (origin+mean)) < 0.0) { normal = -normal; }` : 'if (normal.z < 0.0 || (normal.z == 0.0 && normal.y < 0.0) || (normal.z == 0.0 && normal.y == 0.0 && normal.x < 0.0)) { normal = -normal; }'}
let largest = max(max(values.x,values.y),values.z);
let least = min(min(values.x,values.y),values.z);
let middle = values.x + values.y + values.z - largest - least;
writeEigenvalues(index, vec3f(largest, middle, least));
if (!(middle > largest * 1e-6) || !(largest > 1e-30)) { flags |= 1u; }
else { writeNormals(index, normal); writeShapes(index, vec3f((largest-middle)/largest, (middle-least)/largest, least/largest)); }
writeStatus(index, flags);`
    });
  }
}
