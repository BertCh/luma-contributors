// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUPointCloudFeatures} from './gpu-pointcloud-features';
import {GPUPointCloudNeighborhoods} from './gpu-pointcloud-neighborhoods';
import {getPointCloudKernelNodes, validatePointCloudPackedViews} from './pointcloud-kernel-nodes';
import {
  GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY,
  GPU_POINT_CLOUD_MAXIMUM_NORMAL_SCALE_COUNT,
  GPU_POINT_CLOUD_NORMAL_SCALE_INDEX_NODE_ALLOWANCE,
  GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK
} from './pointcloud-limits';
import {
  validatePointCloudSearchGrid,
  validatePointCloudGridIndexBindings,
  type PointCloudGrid
} from './pointcloud-grid-index';

/** Float32 rows: minimumSupport, viewpoint X/Y/Z, and viewpoint-enabled flag. */
export const GPU_POINT_CLOUD_NORMAL_SCALE_PARAMETER_LENGTH = 5;

import {
  getPointCloudRuntimeKernelNodes,
  validatePointCloudDispatchParameterBindings
} from './pointcloud-chunk-kernel-nodes';
import {
  GRAPH_CHUNK_MAXIMUM_ROW_COUNT,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {validatePointCloudGridParameters} from './pointcloud-grid-parameters';

export type GPUPointCloudNormalScaleSettings = {
  /** Minimum other points for a candidate plane, 3–64, default 3. Above capacity no scale qualifies. */
  minimumSupport?: number;
  /** Orient toward this viewpoint; omitted uses positive z, then y, then x. */
  viewpoint?: readonly [number, number, number];
};

/** Pack per-frame eligibility and orientation; write into a caller-owned GPUParameterBuffer. */
export function getGPUPointCloudNormalScaleParameterValues(
  settings: GPUPointCloudNormalScaleSettings = {}
): Float32Array {
  const identifier = 'gpu-pointcloud-normal-scale';
  validateCapacity(
    identifier,
    'minimumSupport',
    settings.minimumSupport ?? 3,
    3,
    GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY
  );
  settings.viewpoint?.forEach((value, axis) =>
    validateFiniteNumber(identifier, `viewpoint[${axis}]`, value)
  );
  return new Float32Array([
    settings.minimumSupport ?? 3,
    ...(settings.viewpoint ?? [0, 0, 0]),
    settings.viewpoint ? 1 : 0
  ]);
}

export type GPUPointCloudNormalScaleOutput = {
  normals: GraphDataView<'float32x3'>;
  /** Selected radius, zero if no usable scale. */
  normalScales: GraphDataView<'float32'>;
  /** Smallest population covariance eigenvalue (squared plane roughness). */
  normalVariances: GraphDataView<'float32'>;
  /** Per-row bits: 1 no usable scale; 2 candidateOverflow at any scale; 4 eigensolve exhaustion at any scale.
   * A row with bit 1 clear publishes a normal, even when another scale was rejected. Other outputs
   * are zero when bit 1 is set. Nonfinite, out-of-range and degenerate queries have no usable scale. */
  status: GraphDataView<'uint32'>;
};

export type GPUPointCloudNormalScaleProps = {
  /** Runtime origin/cellSize for the resident search grid; requires grid. */
  gridParameters?: GraphDataView<'float32'>;
  id?: string;
  positions: GraphDataView<'float32x3'>;
  /** Owned core rows into positions; omitted selects every position. The query itself is excluded. */
  queryRows?: GraphDataView<'uint32'>;
  spatialContext?: GPUSpatialContext;
  /** Strictly increasing spherical radii, in native Cartesian coordinate units (not diameters). */
  radii: readonly number[];
  /** Maximum support retained at the largest radius; overflowing scales are rejected. */
  capacity: number;
  /** Per-frame eligibility/orientation, packed by getGPUPointCloudNormalScaleParameterValues. */
  parameters: GraphDataView<'float32'>;
  /** Index must cover the full intended support; points outside the half-open grid are excluded. */
  grid?: PointCloudGrid;
  /** Caller-owned result views; their contents are written by the graph. */
  output: GPUPointCloudNormalScaleOutput;
};

/**
 * Select the minimum-roughness normal over a bounded set of spherical scales.
 *
 * Composes exact nearest neighborhoods and covariance features; the smallest population eigenvalue
 * measures squared orthogonal plane roughness. Exact ties choose the smaller radius. Only complete,
 * nondegenerate, converged supports participate. A truncated largest-radius list can still certify a
 * smaller radius when its boundary precedes the last retained neighbor. There is no host readback.
 *
 * Compile-time: view shapes, radii (search bounds), capacity and grid dimensions.
 * Per-frame: positions, queryRows, parameters and gridParameters contents. Coordinates
 * are local Cartesian 3D in a planar/native spatialContext; recenter large absolute coordinates.
 * The grid must contain every intended neighbor; no missing halo can be inferred from absent rows.
 */
export class GPUPointCloudNormalScale implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPUPointCloudNormalScaleProps) {
    props = snapshotConstructionFields(props, [
      'gridParameters',
      'id',
      'positions',
      'queryRows',
      'spatialContext',
      'radii',
      'capacity',
      'parameters',
      'grid',
      'output'
    ]);
    this.id = props.id ?? 'gpu-pointcloud-normal-scale';
    validateRequirement(
      this.id,
      props.output !== undefined && props.output !== null,
      'output must be given'
    );
    this.props = props = Object.freeze({
      ...props,
      output: snapshotConstructionFields(props.output, [
        'normals',
        'normalScales',
        'normalVariances',
        'status'
      ])
    });
    validatePointCloudGridParameters(this.id, props.gridParameters);
    validateRequirement(
      this.id,
      !props.gridParameters || Boolean(props.grid),
      'gridParameters requires grid'
    );
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      coordinateSpaces: ['planar'],
      metrics: ['native'],
      measure: 'distance'
    });
    validateCapacity(
      this.id,
      'radii.length',
      props.radii.length,
      1,
      GPU_POINT_CLOUD_MAXIMUM_NORMAL_SCALE_COUNT
    );
    props.radii.forEach((radius, index) => {
      validateRequirement(
        this.id,
        radius > 0 &&
          Math.fround(radius ** 2) > 0 &&
          Number.isFinite(Math.fround(radius ** 2)) &&
          (index === 0 || Math.fround(radius ** 2) > Math.fround(props.radii[index - 1] ** 2)),
        'radii must be strictly increasing with positive finite distinct float32 squares'
      );
    });
    validateCapacity(
      this.id,
      'capacity',
      props.capacity,
      3,
      GPU_POINT_CLOUD_MAXIMUM_NEIGHBOR_CAPACITY
    );
    validateViewLength(
      this.id,
      'parameters',
      props.parameters,
      GPU_POINT_CLOUD_NORMAL_SCALE_PARAMETER_LENGTH
    );
    const rows = props.queryRows?.length ?? props.positions.length;
    if (props.gridParameters)
      validateCapacity(
        this.id,
        'runtime-grid packed neighbor words',
        rows * (2 * props.capacity + 1),
        0,
        GRAPH_CHUNK_MAXIMUM_ROW_COUNT
      );
    for (const name of ['normals', 'normalScales', 'normalVariances', 'status'] as const) {
      validateViewLength(this.id, name, props.output[name], rows);
    }
    validatePointCloudPackedViews(
      this.id,
      {
        positions: props.positions,
        queryRows: props.queryRows,
        parameters: props.parameters,
        gridParameters: props.gridParameters
      },
      {
        normals: props.output.normals,
        normalScales: props.output.normalScales,
        normalVariances: props.output.normalVariances,
        status: props.output.status
      },
      [
        {
          outputs: {
            normals: props.output.normals,
            normalScales: props.output.normalScales,
            normalVariances: props.output.normalVariances,
            status: props.output.status
          }
        },
        {
          inputs: {
            positions: props.positions,
            queryRows: props.queryRows,
            parameters: props.parameters,
            normalScales: props.output.normalScales
          },
          outputs: {normals: props.output.normals}
        }
      ]
    );
    if (props.grid) {
      validatePointCloudSearchGrid(this.id, 'grid', props.grid);
      validateRequirement(
        this.id,
        Number.isFinite(Math.fround(props.radii.at(-1)! / props.grid.cellSize)),
        'radii divided by grid.cellSize must be finite in float32'
      );
    } else {
      validateRequirement(
        this.id,
        rows * props.positions.length <= GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK,
        `queries x positions must be at most ${GPU_POINT_CLOUD_MAXIMUM_UNINDEXED_WORK} without grid; pass grid`
      );
    }
    this.estimatedCommandNodeCount =
      3 +
      Number(Boolean(props.gridParameters)) +
      8 * props.radii.length +
      (props.grid ? GPU_POINT_CLOUD_NORMAL_SCALE_INDEX_NODE_ALLOWANCE : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      positions,
      queryRows,
      capacity,
      output: {normals, normalScales, normalVariances, status}
    } = this.props;
    if (!normals.length) return [];
    validateGraphChunkBindingRanges(graph, this.id, {
      positions,
      queryRows,
      normals,
      normalScales,
      normalVariances,
      status,
      parameters: this.props.parameters,
      gridParameters: this.props.gridParameters
    });
    if (this.props.grid)
      validatePointCloudGridIndexBindings(
        graph,
        this.id,
        positions,
        this.props.grid.dimensions,
        this.props.gridParameters
      );
    validatePointCloudDispatchParameterBindings(graph, this.id, 12, 1);
    const rows = normals.length;
    // Check the largest row-proportional scratch before constructing any nodes.
    validateGPUStorageBindingByteLength(
      graph.device,
      this.id,
      'rows x capacity scratch',
      rows * capacity * 4,
      'reduce queryRows or capacity'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      this.id,
      'feature scratch',
      rows * 12,
      'reduce queryRows'
    );
    if (this.props.gridParameters)
      validateGPUStorageBindingByteLength(
        graph.device,
        this.id,
        'runtime-grid neighborhood scratch',
        rows * (2 * capacity + 1) * 4,
        'reduce queryRows or capacity'
      );
    const indices = createTransientView(graph, `${this.id}-indices`, 'uint32', rows * capacity);
    const distances = createTransientView(
      graph,
      `${this.id}-distances`,
      'float32',
      rows * capacity
    );
    const counts = createTransientView(graph, `${this.id}-counts`, 'uint32', rows);
    const scaleCounts = createTransientView(graph, `${this.id}-scale-counts`, 'uint32', rows);
    const candidateNormals = createTransientView(
      graph,
      `${this.id}-candidate-normals`,
      'float32x3',
      rows
    );
    const eigenvalues = createTransientView(graph, `${this.id}-eigenvalues`, 'float32x3', rows);
    const shapes = createTransientView(graph, `${this.id}-shapes`, 'float32x3', rows);
    const featureStatus = createTransientView(graph, `${this.id}-feature-status`, 'uint32', rows);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUPointCloudNeighborhoods({
        id: `${this.id}-neighborhoods`,
        positions,
        queryRows,
        capacity,
        radius: this.props.radii.at(-1)!,
        grid: this.props.grid,
        gridParameters: this.props.gridParameters,
        output: {indices, squaredDistances: distances, counts}
      }).getCommandNodes(graph)
    ];
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-initialize`,
        length: rows,
        inputs: {},
        outputs: {normals, normalScales, normalVariances, status},
        body: 'writeNormals(index, vec3f(0.0)); writeNormalScales(index, 0.0); writeNormalVariances(index, 0.0); writeStatus(index, 1u);'
      })
    );
    for (const [scaleIndex, radius] of this.props.radii.entries()) {
      nodes.push(
        ...getPointCloudRuntimeKernelNodes(graph, {
          id: `${this.id}-support-${scaleIndex}`,
          length: rows,
          inputs: {distances, counts, parameters: this.props.parameters},
          outputs: {scaleCounts},
          runtimeWords: [new Uint32Array(new Float32Array([radius ** 2]).buffer)[0]],
          body: `var support = 0u;
let retained = min(readCounts(index), ${capacity}u);
for (var slot = 0u; slot < retained; slot++) {
  if (readDistances(index * ${capacity}u + slot) <= bitcast<f32>(readDispatchParameter(0u))) { support++; }
}
// If the boundary reaches the last retained point, omitted ties might also belong to this scale.
if (readCounts(index) > ${capacity}u && support == ${capacity}u) { support = ${capacity + 1}u; }
if (support <= ${capacity}u && f32(support) < readParameters(0u)) { support = 0u; }
writeScaleCounts(index, support);`
        })
      );
      nodes.push(
        ...new GPUPointCloudFeatures({
          id: `${this.id}-features-${scaleIndex}`,
          positions,
          indices,
          counts: scaleCounts,
          capacity,
          output: {normals: candidateNormals, eigenvalues, shapes, status: featureStatus}
        }).getCommandNodes(graph)
      );
      nodes.push(
        ...getPointCloudRuntimeKernelNodes(graph, {
          id: `${this.id}-select-${scaleIndex}`,
          length: rows,
          inputs: {candidateNormals, eigenvalues, featureStatus},
          outputs: {normals, normalScales, normalVariances, status},
          runtimeWords: [new Uint32Array(new Float32Array([radius]).buffer)[0]],
          body: `let flags = readFeatureStatus(index);
var resultStatus = readStatus(index) | (flags & 6u);
let variance = readEigenvalues(index).z;
if (flags == 0u &&
    variance >= 0.0 && variance <= 3.402823e38 &&
    ((resultStatus & 1u) != 0u || variance < readNormalVariances(index))) {
  writeNormals(index, readCandidateNormals(index));
  writeNormalScales(index, bitcast<f32>(readDispatchParameter(0u)));
  writeNormalVariances(index, variance);
  resultStatus &= ~1u;
}
writeStatus(index, resultStatus);`
        })
      );
    }
    nodes.push(
      ...getPointCloudKernelNodes(graph, {
        id: `${this.id}-orient`,
        length: rows,
        inputs: {
          positions,
          normalScales,
          parameters: this.props.parameters,
          ...(queryRows ? {queryRows} : {})
        },
        outputs: {normals},
        body: `if (readParameters(4u) == 0.0 || readNormalScales(index) == 0.0) { return; }
let core = readPositions(${queryRows ? 'readQueryRows(index)' : 'index'});
let viewpoint = vec3f(readParameters(1u), readParameters(2u), readParameters(3u));
let normal = readNormals(index);
if (dot(normal, viewpoint - core) < 0.0) { writeNormals(index, -normal); }`
      })
    );
    return nodes;
  }
}
