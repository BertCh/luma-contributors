// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {snapshotSinglePartGeometryPort} from '../../analysis-core/contracts/ports';

import {addPreparedRecipe} from './prepared-recipe';

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPURecipeStatusPort,
  GPUStatusScalar,
  GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {GPUGroupConcaveHull, type GPUConcaveHullProps} from '../concave-hull/gpu-concave-hull';
import {getDelaunayMaximumTriangleCount} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPUGeometryMeasures} from '../geometry-measures/index';
import {GPUGroupConvexHull, GPUGroupGeometry} from '../group-geometry/index';
import {GPUSpatialClustering} from '../spatial-clustering/index';
import {
  addRecipeCountNodes,
  assertRecipe,
  createIdentityOffsetsNode,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

const ID = 'GPUClusterAndOutlineRecipe';

/** Planar only: the group summaries and convex hulls are planar operators. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/** Properties for {@link addClusterAndOutlineRecipe}. */
export type GPUClusterAndOutlineRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    concaveHull?: GPUConcaveHullProps['output'];
    labels?: GraphDataView<'uint32'>;
    /** Optional per-point core flags from the same clustering dispatch. */
    coreFlags?: GraphDataView<'uint32'>;
    clusterCount?: GraphDataView<'uint32'>;
    hullPositions?: GraphDataView<'float32x2'>;
    hullOffsets?: GraphDataView<'uint32'>;
    areas?: GraphDataView<'float32'>;
    perimeters?: GraphDataView<'float32'>;
    centroids?: GraphDataView<'float32x2'>;
    /** Mirrors `result.status`. */
    status?: {
      clusterOverflow?: GPUStatusScalar;
      hullOverflow?: GPUStatusScalar;
      /** One-row count of hulls rejected by `GPUGeometryMeasures` (stage `hull-measures`). */
      invalidCount?: GPUStatusScalar;
    };
  },
  {
    clusterIds?: GraphDataView<'uint32'>;
    clusterBoundCount?: GraphDataView<'uint32'>;
    counts?: GraphDataView<'uint32'>;
    bounds?: GraphDataView<'float32x4'>;
    meanCenters?: GraphDataView<'float32x2'>;
    hullVertexIndices?: GraphDataView<'uint32'>;
    hullCounts?: GraphDataView<'uint32'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'cluster-outline-recipe'`. */
  id?: string;
  /** Alpha outlines preserve holes/disconnected pieces as oriented boundary edges. */
  outline?: {kind: 'convex'} | {kind: 'alpha'; maximumCircumradius: number};
  /**
   * Points in the coordinates of `spatialContext`; optional `sourceIds` become the cluster
   * provenance IDs of `GPUSpatialClustering`.
   */
  points: GPUPackedGeometryPort<GPUPointGeometryPort>;
  /**
   * Canonical coordinate and metric contract of `points`, used by the clustering and by hull
   * measurement. Omitted means native planar; `epsilon` uses the context's length unit. Planar
   * only, because the group summaries and convex hulls are planar: project longitude/latitude
   * points first (`createGPULocalFrame` azimuthal-equidistant frame). A lambert-equal-area frame
   * throws (clustering measures distance). The resolved context is returned as
   * `result.spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /** `getGPUSpatialClusteringParameterValues` view: bounds, epsilon, minimum points. */
  clusteringParameters: GraphDataView<'float32'>;
  /** Maximum neighbor-search lattice of the clustering. */
  gridSize: readonly [number, number];
  /** `getGPUGeographicDistributionParameterValues` view for the per-cluster summaries. */
  geometryParameters: GraphDataView<'float32'>;
  /**
   * Compile-time cluster capacity: clusters with a label at or above it are dropped from the
   * summaries and hulls (their labels are still written).
   */
  maximumClusterCount: number;
  /** Largest hull emitted per cluster (at least 3); larger hulls are dropped and flagged. */
  maximumVerticesPerHull: number;
  /** Capacity of the shared hull vertex list. */
  hullCapacity: number;
};

/** Named outputs of {@link addClusterAndOutlineRecipe}. */
export type GPUClusterAndOutlineRecipeOutputs = Pick<
  GPUClusterAndOutlineRecipeViews,
  | 'labels'
  | 'clusterCount'
  | 'hullPositions'
  | 'hullOffsets'
  | 'areas'
  | 'perimeters'
  | 'centroids'
  | 'concaveHull'
>;

/** Ports between cluster-and-outline stages. */
export type GPUClusterAndOutlineRecipeIntermediates = Pick<
  GPUClusterAndOutlineRecipeViews,
  'counts' | 'bounds' | 'meanCenters' | 'hullVertexIndices' | 'hullCounts'
>;

/** Cluster-and-outline outputs, intermediates and top-level convenience fields. */
export type GPUClusterAndOutlineRecipeResult = GPURecipeResult<
  GPUClusterAndOutlineRecipeOutputs,
  GPUClusterAndOutlineRecipeIntermediates
> &
  GPUClusterAndOutlineRecipeViews;

type GPUClusterAndOutlineRecipeViews = {
  /** Resolved coordinate and metric context shared by all spatial stages. */
  spatialContext: GPUSpatialContext;
  /** Cluster capacity: the row count of every per-cluster output. */
  maximumClusterCount: number;
  outlineType: 'convex' | 'alpha';
  /** Present for alpha mode; legacy single-ring hull views are empty in that mode. */
  concaveHull?: GPUConcaveHullProps['output'];
  labels: GraphDataView<'uint32'>;
  clusterCount: GraphDataView<'uint32'>;
  counts: GraphDataView<'uint32'>;
  bounds: GraphDataView<'float32x4'>;
  meanCenters: GraphDataView<'float32x2'>;
  hullVertexIndices: GraphDataView<'uint32'>;
  hullPositions: GraphDataView<'float32x2'>;
  hullOffsets: GraphDataView<'uint32'>;
  hullCounts: GraphDataView<'uint32'>;
  areas: GraphDataView<'float32'>;
  perimeters: GraphDataView<'float32'>;
  centroids: GraphDataView<'float32x2'>;
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row flag: 1 when the cluster summary list was incomplete. Stage `clustering`. */
    clusterOverflow: GPUStatusScalar;
    /** One-row flag: 1 when the shared hull vertex list overflowed. Stage `convex-hull`. */
    hullOverflow: GPUStatusScalar;
    /**
     * One-row count of hull polygons the measurement rejected (geographic hulls enclosing a pole
     * or spanning 180 degrees of longitude); their area and centroid are NaN. Stage
     * `hull-measures`.
     */
    invalidCount: GPUStatusScalar;
  } & GPURecipeStatusPort;
};

/**
 * Cluster and outline recipe: DBSCAN clusters, per-cluster summaries, convex hulls and their
 * area, perimeter and centroid.
 *
 * Chain: `GPUSpatialClustering` -> `GPUGroupGeometry` (counts, bounds, mean centers) and
 * `GPUGroupConvexHull` -> `GPUGeometryMeasures` with each hull as one single-ring polygon feature
 * (the hull's `offsets` is the ring offset list; feature and polygon offsets are the identity). With `minimumPoints = 1` the
 * clusters are PostGIS `ST_ClusterWithin` groups.
 *
 * Status: `status.stages` reports `clustering`, `convex-hull` and `hull-measures` (`status.invalidCount`
 * of rejected hull polygons).
 */
export function addClusterAndOutlineRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUClusterAndOutlineRecipeProps
): GPUClusterAndOutlineRecipeResult {
  return addPreparedRecipe(graph, props, declareClusterAndOutlineRecipe);
}

function declareClusterAndOutlineRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUClusterAndOutlineRecipeProps
): GPUClusterAndOutlineRecipeResult {
  const id = props.id ?? 'cluster-outline-recipe';
  const spatialContext = resolveGPUSpatialContext(
    id,
    props.spatialContext,
    SPATIAL_CONTEXT_SUPPORT
  );
  props = {...props, points: snapshotSinglePartGeometryPort(id, 'points', props.points)};
  const positions = props.points.positions;
  const pointCount = positions.length;
  const groupCount = props.maximumClusterCount;
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  assertRecipe(ID, groupCount >= 1, 'maximumClusterCount must be positive');
  const builder = new RecipeBuilder(graph);
  const labels = getOrCreateView(graph, `${id}-labels`, 'uint32', pointCount, outputs.labels);
  const clusterCount = getOrCreateView(
    graph,
    `${id}-cluster-count`,
    'uint32',
    1,
    outputs.clusterCount
  );
  const clusterIds = getOrCreateView(
    graph,
    `${id}-cluster-ids`,
    'uint32',
    groupCount,
    scratch.clusterIds
  );
  const clusterBoundCount = getOrCreateView(
    graph,
    `${id}-cluster-bound-count`,
    'uint32',
    1,
    scratch.clusterBoundCount
  );
  const clusterOverflow = getOrCreateView(
    graph,
    `${id}-cluster-overflow`,
    'uint32',
    1,
    outputs.status?.clusterOverflow
  );
  builder.add(
    new GPUSpatialClustering({
      id: `${id}-clustering`,
      spatialContext,
      positions,
      sourceIds: props.points.sourceIds,
      parameters: props.clusteringParameters,
      gridSize: props.gridSize,
      output: {
        labels,
        coreFlags: outputs.coreFlags,
        clusterCount,
        clusters: {
          ids: clusterIds,
          status: {count: clusterBoundCount, overflow: clusterOverflow}
        }
      }
    })
  );

  const counts = getOrCreateView(graph, `${id}-counts`, 'uint32', groupCount, scratch.counts);
  const bounds = getOrCreateView(graph, `${id}-bounds`, 'float32x4', groupCount, scratch.bounds);
  const meanCenters = getOrCreateView(
    graph,
    `${id}-mean-centers`,
    'float32x2',
    groupCount,
    scratch.meanCenters
  );
  builder.add(
    new GPUGroupGeometry({
      id: `${id}-group-geometry`,
      spatialContext,
      positions,
      labels,
      groupCount,
      parameters: props.geometryParameters,
      output: {counts, bounds, meanCenters}
    })
  );

  const hullVertexIndices = getOrCreateView(
    graph,
    `${id}-hull-vertex-indices`,
    'uint32',
    props.hullCapacity,
    scratch.hullVertexIndices
  );
  const hullPositions = getOrCreateView(
    graph,
    `${id}-hull-positions`,
    'float32x2',
    props.hullCapacity,
    outputs.hullPositions
  );
  const hullOffsets = getOrCreateView(
    graph,
    `${id}-hull-offsets`,
    'uint32',
    groupCount + 1,
    outputs.hullOffsets
  );
  const hullCounts = getOrCreateView(
    graph,
    `${id}-hull-counts`,
    'uint32',
    groupCount,
    scratch.hullCounts
  );
  const hullOverflow = getOrCreateView(
    graph,
    `${id}-hull-overflow`,
    'uint32',
    1,
    outputs.status?.hullOverflow
  );
  if (props.outline?.kind !== 'alpha')
    builder.add(
      new GPUGroupConvexHull({
        id: `${id}-convex-hull`,
        spatialContext,
        positions,
        labels,
        groupCount,
        maximumVerticesPerGroup: props.maximumVerticesPerHull,
        vertexCapacity: props.hullCapacity,
        output: {
          vertexIndices: hullVertexIndices,
          vertexPositions: hullPositions,
          offsets: hullOffsets,
          counts: hullCounts,
          status: {overflow: hullOverflow}
        }
      })
    );

  const areas = getOrCreateView(graph, `${id}-areas`, 'float32', groupCount, outputs.areas);
  const perimeters = getOrCreateView(
    graph,
    `${id}-perimeters`,
    'float32',
    groupCount,
    outputs.perimeters
  );
  const centroids = getOrCreateView(
    graph,
    `${id}-centroids`,
    'float32x2',
    groupCount,
    outputs.centroids
  );
  const invalidCount = getOrCreateView(
    graph,
    `${id}-invalid-count`,
    'uint32',
    1,
    outputs.status?.invalidCount
  );
  // Each hull is one feature with one polygon of one ring: identity feature and polygon offsets.
  let concaveHull: GPUConcaveHullProps['output'] | undefined;
  if (props.outline?.kind !== 'alpha') {
    const identityOffsets = createIdentityOffsetsNode(
      graph,
      `${id}-hull-identity`,
      groupCount,
      builder
    );
    builder.add(
      new GPUGeometryMeasures({
        spatialContext,
        id: `${id}-hull-measures`,
        geometryType: 'polygons',
        positions: hullPositions,
        featureOffsets: identityOffsets,
        polygonOffsets: identityOffsets,
        ringOffsets: hullOffsets,
        output: {areas, lengths: perimeters, centroids, status: {invalidCount}}
      })
    );
  } else {
    const triangleCapacity = getDelaunayMaximumTriangleCount(pointCount);
    const supplied = outputs.concaveHull;
    concaveHull = {
      triangles: getOrCreateView(
        graph,
        `${id}-alpha-triangles`,
        'uint32x3',
        groupCount * triangleCapacity,
        supplied?.triangles
      ),
      triangleValidity: getOrCreateView(
        graph,
        `${id}-alpha-triangle-validity`,
        'uint32',
        groupCount * triangleCapacity,
        supplied?.triangleValidity
      ),
      boundaryEdges: getOrCreateView(
        graph,
        `${id}-alpha-edges`,
        'uint32x2',
        groupCount * triangleCapacity * 3,
        supplied?.boundaryEdges
      ),
      edgeValidity: getOrCreateView(
        graph,
        `${id}-alpha-edge-validity`,
        'uint32',
        groupCount * triangleCapacity * 3,
        supplied?.edgeValidity
      ),
      status: {
        converged: getOrCreateView(
          graph,
          `${id}-alpha-converged`,
          'uint32',
          groupCount,
          supplied?.status?.converged
        )
      }
    };
    builder.add(
      new GPUGroupConcaveHull({
        id: `${id}-alpha`,
        positions,
        groupIds: labels,
        groupCount,
        spatialContext,
        maximumCircumradius: props.outline.maximumCircumradius,
        output: concaveHull
      })
    );
    builder.addGlue(
      'alpha-measures',
      createWGSLKernelNode(graph, {
        id: `${id}-alpha-measures`,
        operation: ID,
        invocationCount: groupCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'triangles', view: concaveHull.triangles, type: 'u32', access: 'read'},
          {name: 'valid', view: concaveHull.triangleValidity, type: 'u32', access: 'read'},
          {name: 'edgeValid', view: concaveHull.edgeValidity, type: 'u32', access: 'read'},
          {name: 'complete', view: concaveHull.status.converged, type: 'u32', access: 'read'},
          {name: 'areas', view: areas, type: 'f32', access: 'read_write'},
          {name: 'perimeters', view: perimeters, type: 'f32', access: 'read_write'},
          {name: 'centroids', view: centroids, type: 'f32', access: 'read_write'}
        ],
        declarations: `fn position(vertex: u32) -> vec2<f32> {return vec2<f32>(positions[positionsOffset + vertex*2u], positions[positionsOffset + vertex*2u + 1u]);}`,
        body: `var area = 0.0; var perimeter = 0.0; var centroid = vec2<f32>(0.0);
for (var slot = 0u; slot < ${triangleCapacity}u; slot++) {
 let triangle = index * ${triangleCapacity}u + slot;
 if (valid[validOffset+triangle] == 0u) {continue;}
 let first = position(triangles[trianglesOffset+triangle*3u]); let second = position(triangles[trianglesOffset+triangle*3u+1u]); let third = position(triangles[trianglesOffset+triangle*3u+2u]);
 let firstEdge = second-first; let secondEdge = third-first; let triangleArea = abs(firstEdge.x*secondEdge.y-firstEdge.y*secondEdge.x)*0.5;
 area += triangleArea; centroid += triangleArea*(first+second+third)/3.0;
 if (edgeValid[edgeValidOffset+triangle*3u] != 0u) {perimeter += distance(first,second);}
 if (edgeValid[edgeValidOffset+triangle*3u+1u] != 0u) {perimeter += distance(second,third);}
 if (edgeValid[edgeValidOffset+triangle*3u+2u] != 0u) {perimeter += distance(third,first);}
}
let invalid = bitcast<f32>(0x7fc00000u | (index & 0u)); let completeValue = complete[completeOffset+index] != 0u;
areas[areasOffset+index] = select(invalid,area,completeValue); perimeters[perimetersOffset+index] = select(invalid,perimeter,completeValue);
centroids[centroidsOffset+index*2u] = select(invalid,centroid.x/max(area,1.0e-30),completeValue && area>0.0); centroids[centroidsOffset+index*2u+1u] = select(invalid,centroid.y/max(area,1.0e-30),completeValue && area>0.0);`
      })
    );
    addRecipeCountNodes(builder, {
      id: `${id}-alpha-status`,
      operation: ID,
      rowCount: groupCount,
      bindings: [
        {name: 'complete', view: concaveHull.status.converged, type: 'u32', access: 'read'},
        {name: 'hullOverflow', view: hullOverflow, type: 'u32', access: 'read_write'},
        {name: 'offsets', view: hullOffsets, type: 'u32', access: 'read_write'},
        {name: 'counts', view: hullCounts, type: 'u32', access: 'read_write'}
      ],
      prelude: `offsets[offsetsOffset+index] = 0u; counts[countsOffset+index] = 0u;
if (index == 0u) { offsets[offsetsOffset+${groupCount}u] = 0u; hullOverflow[hullOverflowOffset] = 0u; }`,
      counts: [
        {
          name: 'invalid',
          output: invalidCount,
          value: 'u32(complete[completeOffset+index] == 0u)'
        }
      ]
    });
  }

  return {
    contributors: builder.contributors,
    outlineType: props.outline?.kind ?? 'convex',
    concaveHull,
    spatialContext,
    maximumClusterCount: groupCount,
    labels,
    clusterCount,
    counts,
    bounds,
    meanCenters,
    hullVertexIndices,
    hullPositions,
    hullOffsets,
    hullCounts,
    areas,
    perimeters,
    centroids,
    outputs: {
      labels,
      clusterCount,
      hullPositions,
      hullOffsets,
      areas,
      perimeters,
      centroids,
      ...(concaveHull ? {concaveHull} : {})
    },
    intermediates: {counts, bounds, meanCenters, hullVertexIndices, hullCounts},
    status: {
      clusterOverflow,
      hullOverflow,
      invalidCount,
      stages: [
        {
          stage: 'clustering',
          status: {count: clusterBoundCount, requiredCount: clusterCount, overflow: clusterOverflow}
        },
        {
          stage: concaveHull ? 'alpha-hull' : 'convex-hull',
          status: {overflow: hullOverflow, invalidCount}
        },
        {stage: 'hull-measures', status: {invalidCount}}
      ]
    }
  };
}
