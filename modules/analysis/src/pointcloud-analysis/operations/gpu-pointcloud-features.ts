// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPointCloudReadbackLaneReader} from './pointcloud-readback';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  validateCommandNodeBudget,
  validateGraphPorts,
  validateSinglePackedView,
  validateRequirement,
  validateCapacity,
  validateFiniteNumber,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getSymmetricEigenWGSL} from '../../analysis-core/kernels/symmetric-eigen-wgsl';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY} from './pointcloud-limits';
import {
  getPointCloudChunkKernelNodes,
  addPointCloudStridedFields,
  validatePointCloudChunkViews
} from './pointcloud-chunk-kernel-nodes';
import {
  planGraphChunkDispatch,
  validateGraphChunkDispatchBindings,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {inferPointCloudNeighborCapacity} from './pointcloud-neighbor-capacity';

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

export type GPUPointCloudFeaturesOutput = {
  normals: GraphDataView<'float32x3'> | GraphVectorView<'float32x3'>;
  /** Descending eigenvalues of population covariance. */
  eigenvalues: GraphDataView<'float32x3'> | GraphVectorView<'float32x3'>;
  /** Optional packed population covariance: xx, xy, xz, yy, yz, zz per row. */
  covariance?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Linearity, planarity, scattering. */
  shapes: GraphDataView<'float32x3'> | GraphVectorView<'float32x3'>;
  /**
   * Per-row bit set, zero when clean. Bit 1 (`GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE`): fewer
   * than three neighbors, or collinear or coincident neighbors (middle eigenvalue at most 1e-6 of the
   * largest, or largest at most 1e-30); normals and shapes are zero. Bit 2 (`..._TRUNCATED`): `counts` exceeded `capacity`, or supplied neighborhoodStatus reports truncation,
   * so only the retained nearest neighbors were used. Bit 4 (`..._NOT_CONVERGED`): the Jacobi eigensolve did
   * not reach its tolerance within eight cyclic sweeps (24 pair rotations); the outputs are the last iterate.
   * Bit 8 (`..._INCOMPLETE_SUPPORT`): supplied neighborhoodStatus reports invalid or unsupported
   * support. Features describe the available population and are qualified by this bit. Bits above 8 are zero.
   */
  status: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

export type GPUPointCloudFeaturesProps = {
  /** Per-frame orientation; overrides viewpoint. Packed by getGPUPointCloudFeaturesParameterValues. */
  parameters?: GraphDataView<'float32'>;
  id?: string;
  positions: GraphDataView<'float32x3'>;
  indices: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  counts: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Optional neighborhood flags aligned with query rows, including chunk topology. */
  neighborhoodStatus?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** 1–64 slots per query; inferred from exact indices/counts lengths for nonempty inputs. Zero query rows require an explicit capacity. */
  capacity?: number;
  /** Normal orientation toward this point; default chooses nonnegative z (then y, then x). */
  viewpoint?: readonly [number, number, number];
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudFeaturesOutput;
};

/** `GPUPointCloudFeatures` status bit 1: fewer than three neighbors or a degenerate covariance. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE = 1;
/** `GPUPointCloudFeatures` status bit 2: the neighborhood was truncated to `capacity`. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED = 2;
/** `GPUPointCloudFeatures` status bit 4: the eigensolve did not converge. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED = 4;
/** Supplied neighborhood support is invalid or extends outside certified bounds. */
export const GPU_POINT_CLOUD_FEATURES_STATUS_INCOMPLETE_SUPPORT = 8;

/** Decode one query's feature status; pass a one-word subview for a later query.
 * `notConverged` is the observed eigensolve flag. A degenerate row does not certify an attempted
 * solve, so `converged` requires both nondegeneracy and a clear failure flag. `supportComplete`
 * describes neighborhood support; `featureComplete` additionally requires a usable converged fit.
 */
export function decodeGPUPointCloudFeaturesStatus(values: Uint32Array | Uint8Array | ArrayBuffer): {
  degenerate: boolean;
  truncated: boolean;
  converged: boolean;
  notConverged: boolean;
  incompleteSupport: boolean;
  supportComplete: boolean;
  featureComplete: boolean;
} {
  const flags = getPointCloudReadbackLaneReader(
    'decodeGPUPointCloudFeaturesStatus',
    'status',
    values,
    'uint32',
    1
  )(0);
  const supportComplete =
    (flags &
      (GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED |
        GPU_POINT_CLOUD_FEATURES_STATUS_INCOMPLETE_SUPPORT)) ===
    0;
  return {
    degenerate: (flags & GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE) !== 0,
    truncated: (flags & GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED) !== 0,
    converged:
      (flags &
        (GPU_POINT_CLOUD_FEATURES_STATUS_DEGENERATE |
          GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED)) ===
      0,
    notConverged: (flags & GPU_POINT_CLOUD_FEATURES_STATUS_NOT_CONVERGED) !== 0,
    incompleteSupport: (flags & GPU_POINT_CLOUD_FEATURES_STATUS_INCOMPLETE_SUPPORT) !== 0,
    supportComplete,
    featureComplete: flags === 0
  };
}

/**
 * Centered 3D covariance and the shared symmetric Jacobi eigensolve on GPU neighborhoods.
 * Per-frame: input contents and parameters (orientation). Compile-time: capacity, view shapes and
 * optional outputs. Counts, fixed-stride indices, normals, eigenvalues, shapes, covariance and status
 * accept aligned chunks; positions remain a resident packed cloud and indices are global rows.
 * Capacity is 1–64 retained slots per query (GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY), across all routes.
 * Without parameters, viewpoint retains its constructor-time behavior.
 */
export class GPUPointCloudFeatures implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  readonly props: GPUPointCloudFeaturesProps & {capacity: number};
  private readonly chunkPlan: GraphChunkDispatchPlan;
  constructor(props: GPUPointCloudFeaturesProps) {
    this.id = props.id ?? 'gpu-pointcloud-features';
    validateGraphPorts(this.id, props, {
      positions: {required: true, viewType: 'data', formats: ['float32x3']},
      indices: {required: true, viewType: 'either', formats: ['uint32']},
      counts: {required: true, viewType: 'either', formats: ['uint32']},
      neighborhoodStatus: {required: false, viewType: 'either', formats: ['uint32']},
      parameters: {required: false, viewType: 'data', formats: ['float32']},
      'output.normals': {required: true, viewType: 'either', formats: ['float32x3']},
      'output.eigenvalues': {required: true, viewType: 'either', formats: ['float32x3']},
      'output.shapes': {required: true, viewType: 'either', formats: ['float32x3']},
      'output.covariance': {required: false, viewType: 'either', formats: ['float32']},
      'output.status': {required: true, viewType: 'either', formats: ['uint32']}
    });
    props = snapshotConstructionFields(props, [
      'parameters',
      'id',
      'positions',
      'indices',
      'counts',
      'neighborhoodStatus',
      'capacity',
      'viewpoint',
      'output'
    ]);
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    const capacity =
      props.capacity ??
      inferPointCloudNeighborCapacity(this.id, props.counts.length, props.indices.length);
    this.props = Object.freeze({
      ...props,
      capacity,
      output: snapshotConstructionFields(props.output, [
        'normals',
        'eigenvalues',
        'covariance',
        'shapes',
        'status'
      ])
    });
    props = this.props;
    validateCapacity(this.id, 'capacity', capacity, 1, GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY);
    validateViewLength(this.id, 'indices', props.indices, props.counts.length * capacity);
    if (props.neighborhoodStatus)
      validateViewLength(
        this.id,
        'neighborhoodStatus',
        props.neighborhoodStatus,
        props.counts.length
      );
    validateViewLength(this.id, 'normals', props.output.normals, props.counts.length);
    validateViewLength(this.id, 'eigenvalues', props.output.eigenvalues, props.counts.length);
    validateViewLength(this.id, 'shapes', props.output.shapes, props.counts.length);
    validateViewLength(this.id, 'status', props.output.status, props.counts.length);
    if (props.output.covariance) {
      validateViewLength(this.id, 'covariance', props.output.covariance, props.counts.length * 6);
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
    validateSinglePackedView(props.positions, ['float32x3'], `${this.id} positions`);
    if (props.parameters)
      validateSinglePackedView(props.parameters, ['float32'], `${this.id} parameters`);
    this.chunkPlan = addPointCloudStridedFields(
      this.id,
      planGraphChunkDispatch(this.id, {
        counts: props.counts,
        ...(props.neighborhoodStatus ? {neighborhoodStatus: props.neighborhoodStatus} : {}),
        normals: props.output.normals,
        eigenvalues: props.output.eigenvalues,
        shapes: props.output.shapes,
        status: props.output.status
      }),
      {
        indices: {view: props.indices, stride: capacity},
        covariances: {view: props.output.covariance, stride: 6}
      }
    );
    validatePointCloudChunkViews(
      this.id,
      this.chunkPlan,
      ['counts', 'indices', 'neighborhoodStatus'],
      ['normals', 'eigenvalues', 'shapes', 'status', 'covariances'],
      {positions: props.positions, parameters: props.parameters},
      [
        {
          inputNames: ['indices', 'counts'],
          outputNames: ['normals', 'eigenvalues', 'status', 'covariances'],
          inputs: {positions: props.positions}
        },
        {inputNames: ['eigenvalues', 'status'], outputNames: ['shapes']},
        ...(props.neighborhoodStatus
          ? [{inputNames: ['neighborhoodStatus'], outputNames: ['status']}]
          : []),
        ...(props.parameters
          ? [
              {
                inputNames: ['indices', 'counts'],
                outputNames: ['normals'],
                inputs: {positions: props.positions, parameters: props.parameters}
              }
            ]
          : [])
      ]
    );
    this.estimatedCommandNodeCount =
      ((props.parameters ? 3 : 2) + (props.neighborhoodStatus ? 1 : 0)) *
      (1 + this.chunkPlan.nonemptyChunkCount);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      positions,
      counts,
      capacity,
      viewpoint,
      parameters,
      output: {covariance}
    } = this.props;
    if (!this.chunkPlan.nonemptyChunkCount) return [];
    validateGraphChunkDispatchBindings(graph, this.id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, this.id, {positions, parameters});
    const rowViews = this.chunkPlan.chunks[0].views;
    const nodes = getPointCloudChunkKernelNodes(graph, {
      id: this.id,
      length: counts.length,
      plan: this.chunkPlan,
      rowStrides: {indices: capacity, covariances: 6},
      inputs: {
        positions,
        indices: rowViews['indices'] as GraphDataView<'uint32'>,
        counts: rowViews['counts'] as GraphDataView<'uint32'>
      },
      outputs: {
        normals: rowViews['normals'] as GraphDataView<'float32x3'>,
        eigenvalues: rowViews['eigenvalues'] as GraphDataView<'float32x3'>,
        status: rowViews['status'] as GraphDataView<'uint32'>,
        ...(covariance ? {covariances: rowViews['covariances'] as GraphDataView<'float32'>} : {})
      },
      declarations: IS_FINITE_VALUE_WGSL + getSymmetricEigenWGSL(3),
      body: `
let count = min(readCounts(index), ${capacity}u);
var flags = select(0u, ${GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED}u, readCounts(index) > ${capacity}u);
${covariance ? 'for (var component = 0u; component < 6u; component++) { writeCovariances(index * 6u + component, 0.0); }' : ''}
writeNormals(index, vec3f(0.0)); writeEigenvalues(index, vec3f(0.0)); 
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
else { writeNormals(index, normal); }
writeStatus(index, flags);`
    });
    nodes.push(
      ...getPointCloudChunkKernelNodes(graph, {
        id: `${this.id}-shapes`,
        length: counts.length,
        plan: this.chunkPlan,
        inputs: {
          eigenvalues: rowViews['eigenvalues'] as GraphDataView<'float32x3'>,
          status: rowViews['status'] as GraphDataView<'uint32'>
        },
        outputs: {shapes: rowViews['shapes'] as GraphDataView<'float32x3'>},
        body: `var shape = vec3f(0.0);
if ((readStatus(index) & 1u) == 0u) {
  let values = readEigenvalues(index);
  shape = vec3f((values.x-values.y)/values.x, (values.y-values.z)/values.x, values.z/values.x);
}
writeShapes(index, shape);`
      })
    );
    if (parameters) {
      // Covariance output already uses all eight baseline storage bindings. Keep orientation in a
      // separate pass so parameters work with every output configuration on a baseline device.
      nodes.push(
        ...getPointCloudChunkKernelNodes(graph, {
          id: `${this.id}-orient`,
          length: counts.length,
          plan: this.chunkPlan,
          rowStrides: {indices: capacity},
          inputs: {
            positions,
            indices: rowViews['indices'] as GraphDataView<'uint32'>,
            counts: rowViews['counts'] as GraphDataView<'uint32'>,
            parameters
          },
          outputs: {normals: rowViews['normals'] as GraphDataView<'float32x3'>},
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
    if (this.props.neighborhoodStatus)
      nodes.push(
        ...getPointCloudChunkKernelNodes(graph, {
          id: `${this.id}-support`,
          length: counts.length,
          plan: this.chunkPlan,
          inputs: {neighborhoodStatus: rowViews['neighborhoodStatus'] as GraphDataView<'uint32'>},
          outputs: {status: rowViews['status'] as GraphDataView<'uint32'>},
          body: `let upstream = readNeighborhoodStatus(index);
var flags = readStatus(index);
if ((upstream & 5u) != 0u) { flags |= ${GPU_POINT_CLOUD_FEATURES_STATUS_INCOMPLETE_SUPPORT}u; }
if ((upstream & 2u) != 0u) { flags |= ${GPU_POINT_CLOUD_FEATURES_STATUS_TRUNCATED}u; }
writeStatus(index, flags);`
        })
      );
    return nodes;
  }
}
