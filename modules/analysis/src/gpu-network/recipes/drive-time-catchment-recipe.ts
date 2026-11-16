// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {GPUGroupStatistics} from '../../dataframe-analysis/group-statistics/index';
import {GPUNetworkSnapping} from '../network-accessibility/index';
import {createNetworkEdgeSourceNodes} from '../network-accessibility/network-edge-sources';
import {GPUNetworkServiceAreas} from '../network-analysis/index';
import {GPUNetworkIsochrones} from '../network-isochrones/index';
import type {
  GPUNetworkIsochronesCellOutline,
  GPUNetworkIsochronesRaster
} from '../network-isochrones/index';
import {
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/network-topology';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport,
  getGPUSphereRadius
} from '../../analysis-core/contracts/spatial-context';
import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPURecipeStageStatusPort,
  GPURecipeStatusPort,
  GPUUncertaintyStatusPort
} from '../../analysis-core/contracts/index';
import {GPUPointInPolygonJoin} from '../../gpu-spatial-analysis/spatial-join/index';
import type {GPUSegmentRingAssemblyOutput} from '../../analysis-core/ring-assembly/index';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from '../../gpu-spatial-analysis/recipes/recipe-utils';

const ID = 'GPUDriveTimeCatchmentRecipe';

/**
 * Spatial contexts of {@link addDriveTimeCatchmentRecipe}: native planar coordinates (for example a
 * `createGPULocalFrame` frame in meters), or longitude/latitude degrees with the `'none'` or
 * `'great-circle'` metric. Both snappings measure distance in the context's length unit
 * (meters by default, or kilometers on longitude/latitude). Lambert equal-area frames are rejected.
 */
export const GPU_DRIVE_TIME_CATCHMENT_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport =
  Object.freeze({
    coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
    metrics: Object.freeze(['none', 'native', 'great-circle'] as const),
    measure: 'distance'
  });

/** Band key written for demand points that are unsnapped or beyond every break. */
export const GPU_DRIVE_TIME_CATCHMENT_NO_BAND = 0xffffffff;

/**
 * Join of the demand points against the isochrone rings (`cellOutline.rings.output.polygons`).
 * The polygon layout has GPU-written offsets over fixed-length views, so the join's compile-time
 * topology is the ring capacity; unused polygons and rings are empty.
 */
export type GPUDriveTimeIsochroneJoinOptions = {
  /** Maximum `(demand point, isochrone)` bounding-box candidates per encoding. */
  candidateCapacity: number;
};

/**
 * Optional isochrone polygons of the facility costs: raster isobands (triangles per band), or the
 * outline of the cells holding reached nodes with optional closed rings. At least one is required.
 */
export type GPUDriveTimeIsochroneOptions = {
  /** Band edges in cost units; its length is the maximum break count. */
  costBreaks: GraphDataView<'float32'>;
  /** `getGPUNetworkIsochronesParameterValues` view (12 elements). */
  parameters: GraphDataView<'float32'>;
  /** Min-mode raster size and caller-owned isoband outputs. Source paths are provided by the recipe. */
  raster?: GPUNetworkIsochronesRaster;
  /**
   * Cell-outline path (positions must be longitude/latitude). Set `cellOutline.rings` for isochrone
   * rings (shells, holes, shell assignment) of the nodes with cost at most `cellCostLimit`.
   */
  cellOutline?: GPUNetworkIsochronesCellOutline;
  /** Join demand against the cell isochrone rings; needs `cellOutline.rings.output.polygons`. */
  joinDemand?: GPUDriveTimeIsochroneJoinOptions;
};

/** Properties for {@link addDriveTimeCatchmentRecipe}. */
export type GPUDriveTimeCatchmentRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    assignments?: GraphDataView<'uint32'>;
    nodeCosts?: GraphDataView<'float32'>;
    /** Mirrors `result.status`. */
    status?: {serviceAreaConverged?: GraphDataView<'uint32'>};
    demandTimes?: GraphDataView<'float32'>;
    demandBands?: GraphDataView<'uint32'>;
    bands?: {
      keys?: GraphDataView<'uint32'>;
      counts?: GraphDataView<'uint32'>;
      status?: Partial<GPUCardinalityPort & GPUOverflowStatusPort>;
      sumValues?: GraphDataView<'float32'>;
      means?: GraphDataView<'float32'>;
    };
    isochroneDemand?: {
      pointFeatureIds?: GraphDataView<'uint32'>;
      status?: Pick<GPUOverflowStatusPort, 'overflow'> & GPUUncertaintyStatusPort;
    };
    snapping?: {
      status?: {
        facilitiesOverflow?: GraphDataView<'uint32'>;
        demandOverflow?: GraphDataView<'uint32'>;
      };
    };
  },
  {
    startNodes?: GraphDataView<'uint32'>;
    startCosts?: GraphDataView<'float32'>;
    facilityEdges?: GraphDataView<'uint32'>;
    demandEdges?: GraphDataView<'uint32'>;
    demandSourceCosts?: GraphDataView<'float32'>;
    demandTargetCosts?: GraphDataView<'float32'>;
    bandSumWords?: GraphDataView<'uint32x2'>;
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'drive-time-catchment'`. */
  id?: string;
  /**
   * Coordinate space of the facility, demand and node positions. Omitted means native planar:
   * snapping compares Euclidean distances in the coordinate units (meters for a
   * `createGPULocalFrame` frame), and `maximumSnapDistance` is in those units. With
   * `coordinateSpace: 'longitude-latitude'` (metric `'none'` or `'great-circle'`; the cell-outline
   * isochrone path needs degrees) both snappings are great-circle: edges are ranked by great-circle
   * distance in the context's length unit (meters by default, or kilometers), as is
   * `maximumSnapDistance`, with no east-west bias at any latitude
   * (`sphereRadius` of a great-circle context is honored, else the mean Earth radius). Travel
   * costs (`topology.weights`) are independent of positions: the snapped fraction splits the edge
   * cost, so drive times stay in the units of the weights. Other geographic metrics (`'rhumb'`,
   * `'ellipsoidal'`) throw. Local planar frames must be azimuthal-equidistant, not Lambert
   * equal-area. The resolved context is returned as
   * the returned `spatialContext`.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Road network. `weights` are the travel cost per edge (also the cost used to split snapped
   * edges) and `nodePositions` are planar in the coordinates of the facility and demand points.
   */
  topology: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  /** Facility locations, snapped to the nearest edge. */
  facilities: GraphDataView<'float32x2'>;
  /** Demand points (households, customers), snapped to the nearest edge. */
  demand: GraphDataView<'float32x2'>;
  /** Optional per-demand-point value summarised per drive-time band. */
  demandValues?: GraphDataView<'float32'>;
  /**
   * Optional one-row per-frame maximum snap distance in the context's length unit
   * (needed with `candidateCapacity`).
   * On longitude/latitude this distance arrives as a GPU view, so no short-distance warning is possible;
   * below about 17 m use createGPULocalFrame (see Precision of longitude/latitude positions).
   */
  maximumSnapDistance?: GraphDataView<'float32'>;
  /**
   * Compile-time BVH candidate capacity of the snapping; omit for the exact all-edges scan, which
   * tests every point against every edge with no index and raises the snapping's unindexed-work
   * limit, so it becomes slow once points times edges reaches about 2^27 (a frame). When
   * set, each snapping reports candidate overflow (the returned `snapping`, stages `snap-facilities` and
   * `snap-demand`): an overflow means some facilities or demand may be snapped to a farther edge.
   * Great-circle searches ignore the capacity; kilometer contexts scan directly and report no overflow.
   */
  candidateCapacity?: number;
  /** Optional one-row cost cutoff of the service-area search. */
  costLimit?: GraphDataView<'float32'>;
  /** Compile-time iteration bound of each service-area phase. */
  maximumIterations?: number;
  /**
   * Ascending drive-time upper bounds, one per band. Band `k` is the first with
   * `time <= costBreaks[k]`, so it holds `(costBreaks[k - 1], costBreaks[k]]` and a time exactly
   * on a break is within it, the rule of the `GPUNetworkReachability` and `GPUNetworkIsochrones`
   * bands. Slower, unreached or unsnapped demand gets no band. Per-frame values.
   */
  costBreaks: GraphDataView<'float32'>;
  /** Rows of the band table; defaults to `costBreaks.length`. */
  bandCapacity?: number;
  /** Optional isochrone polygons of the same costs. */
  isochrones?: GPUDriveTimeIsochroneOptions;
};

/**
 * Status stages of the caller-owned isochrone outputs. The views belong to the caller (the
 * isochrone contributor requires them together with their geometry), so only those present are
 * reported.
 */
function getIsochroneStatusStages(
  isochrones: GPUDriveTimeIsochroneOptions | undefined
): GPURecipeStageStatusPort[] {
  const stages: GPURecipeStageStatusPort[] = [];
  const raster = isochrones?.raster;
  const output = raster?.output;
  const rasterStatus = output?.status;
  if (rasterStatus?.count && rasterStatus.overflow) {
    stages.push({
      stage: 'isochrone-bands',
      status: {
        count: rasterStatus.count,
        overflow: rasterStatus.overflow,
        requiredCount: rasterStatus.requiredCount
      }
    });
  }
  if (rasterStatus?.edgeCount && rasterStatus.edgeOverflow) {
    stages.push({
      stage: 'isochrone-band-edges',
      status: {
        count: rasterStatus.edgeCount,
        overflow: rasterStatus.edgeOverflow,
        requiredCount: rasterStatus.edgeRequiredCount
      }
    });
  }
  const bandRings = raster?.rings?.output;
  if (bandRings) {
    stages.push({
      stage: 'isochrone-band-rings',
      status: {
        count: bandRings.status.count,
        overflow: bandRings.status.overflow,
        requiredCount: bandRings.status.requiredCount
      }
    });
  }
  const cellOutline = isochrones?.cellOutline;
  if (cellOutline?.table) {
    stages.push({
      stage: 'isochrone-cells',
      status: {
        count: cellOutline.table.status.count,
        overflow: cellOutline.table.status.overflow,
        requiredCount: cellOutline.table.status.requiredCount
      }
    });
  }
  if (cellOutline) {
    stages.push({
      stage: 'isochrone-outline',
      status: {
        count: cellOutline.output.status.count,
        overflow: cellOutline.output.status.overflow,
        requiredCount: cellOutline.output.status.requiredCount
      }
    });
  }
  const cellRings = cellOutline?.rings?.output;
  if (cellRings) {
    stages.push({
      stage: 'isochrone-rings',
      status: {
        count: cellRings.status.count,
        overflow: cellRings.status.overflow,
        requiredCount: cellRings.status.requiredCount
      }
    });
  }
  return stages;
}

/**
 * Adds the kernel that turns snapped edge rows into roads: per point the two end nodes of its
 * edge (source, then target; both `0xffffffff` when unsnapped) and the cost of driving the whole
 * road source -> target, then target -> source. A direction costs the weight of the cheapest
 * passable edge going that way, and -1 when the topology has no such edge (a one-way edge).
 * `keys` names the road whatever edge row the point snapped to: the smallest edge row between the
 * two end nodes in either direction, or the edge capacity for an unsnapped point.
 */
function addSnappedRoadNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    snappedEdges: GraphDataView<'uint32'>;
    edgeSources: GraphDataView<'uint32'>;
    topology: GPUNetworkTopology;
    nodeCount: number;
  }
): {
  nodes: GraphDataView<'uint32x2'>;
  costs: GraphDataView<'float32x2'>;
  keys: GraphDataView<'uint32'>;
} {
  const {id, snappedEdges, edgeSources, topology, nodeCount} = props;
  const nodes = createTransientView(graph, `${id}-nodes`, 'uint32x2', snappedEdges.length);
  const costs = createTransientView(graph, `${id}-costs`, 'float32x2', snappedEdges.length);
  const keys = createTransientView(graph, `${id}-keys`, 'uint32', snappedEdges.length);
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'snapped-roads',
      bindings: [
        {name: 'snappedEdges', view: snappedEdges, type: 'u32', access: 'read'},
        {name: 'edgeSources', view: edgeSources, type: 'u32', access: 'read'},
        {name: 'offsets', view: topology.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: topology.neighbors, type: 'u32', access: 'read'},
        {name: 'weights', view: topology.weights, type: 'f32', access: 'read'},
        {name: 'roads', view: nodes, type: 'u32', access: 'read_write'},
        {name: 'roadCosts', view: costs, type: 'f32', access: 'read_write'},
        {name: 'roadKeys', view: keys, type: 'u32', access: 'read_write'}
      ],
      invocationCount: snappedEdges.length,
      declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${topology.neighbors.length}u;
const NONE: u32 = 0xffffffffu;
const IMPASSABLE: f32 = 3.0e38;
fn getRoadCost(fromNode: u32, toNode: u32) -> f32 {
  var roadCost = -1.0;
  let end = min(offsets[offsetsOffset + fromNode + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + fromNode]; edge < end; edge++) {
    let weight = weights[weightsOffset + edge];
    if (neighbors[neighborsOffset + edge] == toNode && weight >= 0.0 && weight < IMPASSABLE && (roadCost < 0.0 || weight < roadCost)) {
      roadCost = weight;
    }
  }
  return roadCost;
}
// Smallest edge row from the first node to the second, passable or not, or the edge capacity.
fn getFirstRoadEdge(fromNode: u32, toNode: u32) -> u32 {
  let end = min(offsets[offsetsOffset + fromNode + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + fromNode]; edge < end; edge++) {
    if (neighbors[neighborsOffset + edge] == toNode) {
      return edge;
    }
  }
  return EDGE_COUNT;
}`,
      body: `let edge = snappedEdges[snappedEdgesOffset + index];
  var sourceNode = NONE;
  var targetNode = NONE;
  if (edge < EDGE_COUNT) {
    sourceNode = edgeSources[edgeSourcesOffset + edge];
    targetNode = neighbors[neighborsOffset + edge];
  }
  var forwardCost = -1.0;
  var reverseCost = -1.0;
  var roadKey = EDGE_COUNT;
  if (sourceNode < NODE_COUNT && targetNode < NODE_COUNT) {
    forwardCost = getRoadCost(sourceNode, targetNode);
    reverseCost = getRoadCost(targetNode, sourceNode);
    roadKey = min(getFirstRoadEdge(sourceNode, targetNode), getFirstRoadEdge(targetNode, sourceNode));
  } else {
    sourceNode = NONE;
    targetNode = NONE;
  }
  roads[roadsOffset + 2u * index] = sourceNode;
  roads[roadsOffset + 2u * index + 1u] = targetNode;
  roadCosts[roadCostsOffset + 2u * index] = forwardCost;
  roadCosts[roadCostsOffset + 2u * index + 1u] = reverseCost;
  roadKeys[roadKeysOffset + index] = roadKey;`
    })
  );
  return {nodes, costs, keys};
}

/** Named outputs of {@link addDriveTimeCatchmentRecipe}. */
export type GPUDriveTimeCatchmentRecipeOutput = GPURecipeResult & {
  /** Resolved coordinate context of the positions. */
  spatialContext: GPUSpatialContext;
  /**
   * Facility start nodes (`2 * facilityCount` rows, two edge endpoints per facility: the edge
   * source, then the edge target). A row is `0xffffffff` when the facility is unsnapped or no
   * edge lets it drive there and the facility is not coincident with that endpoint.
   */
  startNodes: GraphDataView<'uint32'>;
  /** Cost from each facility to its start nodes, aligned with `startNodes`; -1 on a missing row. */
  startCosts: GraphDataView<'float32'>;
  assignments: GraphDataView<'uint32'>;
  nodeCosts: GraphDataView<'float32'>;
  /**
   * Drive time of each demand point. `+Infinity` when the point is unsnapped or no facility
   * reaches it: an unreachable cost, the value `nodeCosts` holds for an unreached node, not a
   * missing value. Such a point gets `GPU_DRIVE_TIME_CATCHMENT_NO_BAND`.
   */
  demandTimes: GraphDataView<'float32'>;
  demandBands: GraphDataView<'uint32'>;
  /** Status words next to the per-stage `stages`. */
  status: {
    /** One-row flag: 1 when the service-area search converged (stage `service-areas`). */
    serviceAreaConverged: GraphDataView<'uint32'>;
  } & GPURecipeStatusPort;
  bands: {
    keys: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32'>;
    /** One-row status words of stage `band-statistics`; `requiredCount` is the unclamped number of bands present. */
    status: GPUCardinalityPort & GPUOverflowStatusPort;
    sumValues?: GraphDataView<'float32'>;
    means?: GraphDataView<'float32'>;
  };
  /** Isochrone ring outputs (the caller-owned `isochrones.cellOutline.rings.output`), if requested. */
  isochroneRings?: GPUSegmentRingAssemblyOutput;
  /** Join of demand against the isochrone rings, if `isochrones.joinDemand` was set. */
  isochroneDemand?: {
    pointFeatureIds: GraphDataView<'uint32'>;
    /** One-row status words of stage `isochrone-demand`. */
    status: {
      overflow: GraphDataView<'uint32'>;
      uncertainCount: GraphDataView<'uint32'>;
    };
  };
  /**
   * BVH candidate overflow flags of the facility and demand snapping, if `candidateCapacity` was
   * set: 1 when a candidate search dropped edges (the snap may not be the nearest edge).
   */
  snapping?: {
    status: {
      facilitiesOverflow: GraphDataView<'uint32'>;
      demandOverflow: GraphDataView<'uint32'>;
    };
  };
};

/**
 * Drive-time catchment recipe: which demand is within each drive-time band of the facilities.
 *
 * Chain: `GPUNetworkSnapping` (facilities) -> start-row adapter -> `GPUNetworkServiceAreas`
 * (multi-source costs) -> optional `GPUNetworkIsochrones` (isoband triangles, or cell-outline
 * rings) -> `GPUNetworkSnapping` (demand) -> adapter kernels (demand time, then the band) ->
 * `GPUGroupStatistics` keyed by band.
 *
 * Roads and direction: a snapped point lies at fraction `f` of the straight road between the two
 * end nodes of its edge. Driving that road from one end to the other costs the weight of the
 * cheapest passable edge going that way, and is not possible when the topology has no such edge.
 * So a facility starts at the edge target with cost `(1 - f) * w(source -> target)` and at the
 * edge source with cost `f * w(target -> source)`, and a one-way edge gives one interior start row; an endpoint coincident with the facility
 * is always seeded at zero cost. The
 * demand time is the smallest of `cost[source] + f * w(source -> target)`,
 * `cost[target] + (1 - f) * w(target -> source)`, and, for every facility snapped to the same
 * road, the way straight along it: `(f - f_facility) * w(source -> target)` when the demand is
 * ahead of the facility and `(f_facility - f) * w(target -> source)` when it is behind. On a
 * two-way road of one cost that last term is `abs(f - f_facility) * w`. The facilities are
 * sorted by road (the published `GPUSort`, keyed by the smallest edge row between the two end
 * nodes) and each demand point binary-searches its own road, so it is compared only with the
 * facilities snapped to that road: no demand point scans every facility, and the facility count
 * needs no cap. A demand point coincident with an endpoint uses that node cost without
 * traversing a direction, even if the opposite road direction is unavailable.
 *
 * Bands are classified in network space (exact at the snapped edge, no polygon approximation):
 * isoband output has a GPU-written, data-dependent triangle count that `GPUPointInPolygonJoin`
 * cannot take as polygons. The cell path produces closed rings (`GPUSegmentRingAssembly`) whose
 * polygon layout does fit the join: its offsets are GPU-written over fixed-length views, so
 * `isochrones.joinDemand` joins the demand points against the isochrone at `cellCostLimit` in the
 * coordinates of the facilities and demand (longitude/latitude). That polygon is the union of
 * cells holding reached nodes, a coarser answer than the network-space bands. The ring edges are
 * axis-aligned cell edges in degrees, so the join evaluates containment in the degree plane exactly
 * and is constructed with the native planar context whatever `spatialContext` says.
 *
 * Spatial context: see `GPUDriveTimeCatchmentRecipeProps.spatialContext`. Costs are in the units
 * of `topology.weights`; distances (`maximumSnapDistance`) are in the context's length unit.
 *
 * Status: `status.stages` reports `snap-facilities` and `snap-demand` (candidate overflow, with
 * `candidateCapacity`), `service-areas` (convergence), the isochrone stages for the caller-owned
 * isochrone outputs that carry status (`isochrone-bands`, `isochrone-band-edges`,
 * `isochrone-band-rings` on the raster path; `isochrone-cells`, `isochrone-outline`,
 * `isochrone-rings` on the cell path), `band-statistics` (count, overflow, required count) and,
 * with a demand join, `isochrone-demand`. The service-area `assignments` are handed to
 * `GPUNetworkIsochrones`, so `raster.output.pixelFacilities`, `triangleFacilities` and
 * `cellOutline.byFacility` work through the recipe; facility rows there are start rows
 * (`2 * facility + endpoint`).
 */
export function addDriveTimeCatchmentRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUDriveTimeCatchmentRecipeProps
): GPUDriveTimeCatchmentRecipeOutput {
  const id = props.id ?? 'drive-time-catchment';
  const spatialContext = resolveGPUSpatialContext(
    id,
    props.spatialContext,
    GPU_DRIVE_TIME_CATCHMENT_SPATIAL_CONTEXT_SUPPORT
  );
  const builder = new RecipeBuilder(graph);
  const {topology} = props;
  const outputs = props.outputs ?? {};
  const scratch = props.scratch ?? {};
  const {nodeCount} = validateGPUNetworkTopology(id, topology, {
    weights: true,
    nodePositions: true
  });
  const facilityCount = props.facilities.length;
  const demandCount = props.demand.length;
  const bandCount = props.costBreaks.length;
  assertRecipe(ID, bandCount >= 1, 'costBreaks must not be empty');

  // Edge source node per CSR edge, shared by the demand-time adapter (instead of a binary search).
  // The shared gpu-network nodes write every row, so rows past offsets[nodeCount] are never stale.
  const edgeCount = topology.neighbors.length;
  const edgeSources = createTransientView(
    graph,
    `${id}-edge-sources`,
    'uint32',
    Math.max(edgeCount, 1)
  );
  for (const node of createNetworkEdgeSourceNodes<Parameters>(graph, {
    id: `${id}-edge-sources-kernel`,
    operation: 'GPURecipeDriveTimeCatchment',
    offsets: topology.offsets,
    edgeSources,
    nodeCount
  })) {
    graph.add(node);
  }

  // Longitude/latitude positions snap great-circle in the context's length unit for either metric.
  const snappingContext: GPUSpatialContext =
    spatialContext.coordinateSpace === 'longitude-latitude'
      ? {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: getGPUSphereRadius(spatialContext),
          ...(spatialContext.units ? {units: spatialContext.units} : {})
        }
      : spatialContext;
  const providedSnapping = outputs.snapping ?? {};
  const snapping: GPUDriveTimeCatchmentRecipeOutput['snapping'] =
    props.candidateCapacity !== undefined
      ? {
          status: {
            facilitiesOverflow: getOrCreateView(
              graph,
              `${id}-snap-facilities-overflow`,
              'uint32',
              1,
              providedSnapping.status?.facilitiesOverflow
            ),
            demandOverflow: getOrCreateView(
              graph,
              `${id}-snap-demand-overflow`,
              'uint32',
              1,
              providedSnapping.status?.demandOverflow
            )
          }
        }
      : undefined;

  const startNodes = getOrCreateView(
    graph,
    `${id}-start-nodes`,
    'uint32',
    2 * facilityCount,
    scratch.startNodes
  );
  const startCosts = getOrCreateView(
    graph,
    `${id}-start-costs`,
    'float32',
    2 * facilityCount,
    scratch.startCosts
  );
  const facilityEdges = getOrCreateView(
    graph,
    `${id}-facility-edges`,
    'uint32',
    facilityCount,
    scratch.facilityEdges
  );
  const facilityFractions = createTransientView(
    graph,
    `${id}-facility-fractions`,
    'float32',
    facilityCount
  );
  builder.add(
    new GPUNetworkSnapping({
      id: `${id}-snap-facilities`,
      spatialContext: snappingContext,
      positions: props.facilities,
      topology,
      maximumSnapDistance: props.maximumSnapDistance,
      candidateCapacity: props.candidateCapacity,
      // The exact scan has no size limit here, as before the snapping limit existed.
      maximumUnindexedWork: Number.MAX_SAFE_INTEGER,
      output: {
        status: {overflow: snapping?.status.facilitiesOverflow},
        snappedEdges: facilityEdges,
        snapFractions: facilityFractions
      }
    })
  );

  // Adapter: the start rows of each facility, honouring edge direction. A facility reaches the
  // target of its road by driving source -> target and the source by driving target -> source,
  // each at the cost of an edge that goes that way; a direction with no edge gets no start row unless the facility is already at that endpoint.
  // The roads of the facilities are kept so the demand adapter can find facilities on its road.
  const facilityRoads = addSnappedRoadNodes(graph, {
    id: `${id}-facility-roads`,
    snappedEdges: facilityEdges,
    edgeSources,
    topology,
    nodeCount
  });
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-facility-starts-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'facility-starts',
      bindings: [
        {name: 'roads', view: facilityRoads.nodes, type: 'u32', access: 'read'},
        {name: 'roadCosts', view: facilityRoads.costs, type: 'f32', access: 'read'},
        {name: 'fractions', view: facilityFractions, type: 'f32', access: 'read'},
        {name: 'startNodes', view: startNodes, type: 'u32', access: 'read_write'},
        {name: 'startCosts', view: startCosts, type: 'f32', access: 'read_write'}
      ],
      invocationCount: facilityCount,
      declarations: 'const NONE: u32 = 0xffffffffu;',
      body: `let fraction = fractions[fractionsOffset + index];
  let forwardCost = roadCosts[roadCostsOffset + 2u * index];
  let reverseCost = roadCosts[roadCostsOffset + 2u * index + 1u];
  let useSource = reverseCost >= 0.0 || fraction == 0.0;
  let useTarget = forwardCost >= 0.0 || fraction == 1.0;
  startNodes[startNodesOffset + 2u * index] = select(NONE, roads[roadsOffset + 2u * index], useSource);
  startCosts[startCostsOffset + 2u * index] = select(-1.0, fraction * max(reverseCost, 0.0), useSource);
  startNodes[startNodesOffset + 2u * index + 1u] = select(NONE, roads[roadsOffset + 2u * index + 1u], useTarget);
  startCosts[startCostsOffset + 2u * index + 1u] = select(-1.0, (1.0 - fraction) * max(forwardCost, 0.0), useTarget);`
    })
  );
  // Facilities sorted by road key, so a demand point finds the facilities on its road by a
  // binary search instead of a scan of every facility (contract sections 2 and 3: candidates
  // through a sort plus search). The sorted payload is the facility's fraction along its edge
  // with the sign bit set when that edge runs from the higher node ID to the lower, which lets
  // the demand adapter turn it into a fraction along its own edge without a second lookup.
  const facilityOrientedFractions = createTransientView(
    graph,
    `${id}-facility-oriented-fractions`,
    'uint32',
    facilityCount
  );
  const sortedFacilityKeys = createTransientView(
    graph,
    `${id}-sorted-facility-keys`,
    'uint32',
    facilityCount
  );
  const sortedFacilityFractions = createTransientView(
    graph,
    `${id}-sorted-facility-fractions`,
    'uint32',
    facilityCount
  );
  graph.add([
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-facility-oriented-fractions-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'facility-oriented-fractions',
      bindings: [
        {name: 'roads', view: facilityRoads.nodes, type: 'u32', access: 'read'},
        {name: 'fractions', view: facilityFractions, type: 'f32', access: 'read'},
        {name: 'oriented', view: facilityOrientedFractions, type: 'u32', access: 'read_write'}
      ],
      invocationCount: facilityCount,
      body: `let isReversed = roads[roadsOffset + 2u * index] > roads[roadsOffset + 2u * index + 1u];
  let fractionBits = bitcast<u32>(fractions[fractionsOffset + index]) & 0x7fffffffu;
  oriented[orientedOffset + index] = fractionBits | select(0u, 0x80000000u, isReversed);`
    }),
    ...new GPUSort({
      id: `${id}-facility-road-sort`,
      keys: facilityRoads.keys,
      values: facilityOrientedFractions,
      outputKeys: sortedFacilityKeys,
      outputValues: sortedFacilityFractions,
      keyBits: getSortKeyBits(edgeCount)
    }).getCommandNodes(graph)
  ]);

  const assignments = getOrCreateView(
    graph,
    `${id}-assignments`,
    'uint32',
    nodeCount,
    outputs.assignments
  );
  const nodeCosts = getOrCreateView(
    graph,
    `${id}-node-costs`,
    'float32',
    nodeCount,
    outputs.nodeCosts
  );
  const serviceAreaConverged = getOrCreateView(
    graph,
    `${id}-service-area-converged`,
    'uint32',
    1,
    outputs.status?.serviceAreaConverged
  );
  builder.add(
    new GPUNetworkServiceAreas({
      id: `${id}-service-areas`,
      topology,
      facilities: startNodes,
      facilityCosts: startCosts,
      costLimit: props.costLimit,
      maximumIterations: props.maximumIterations,
      output: {assignments, costs: nodeCosts, status: {converged: serviceAreaConverged}}
    })
  );

  const isochrones = props.isochrones;
  if (isochrones) {
    assertRecipe(
      ID,
      Boolean(isochrones.raster || isochrones.cellOutline),
      'isochrones needs raster or cellOutline'
    );
    assertRecipe(
      ID,
      !isochrones.joinDemand || Boolean(isochrones.cellOutline?.rings?.output.polygons),
      'isochrones.joinDemand needs cellOutline.rings.output.polygons'
    );
    let raster = isochrones.raster;
    if (raster) {
      validateRequirement(
        ID,
        (raster.mode ?? 'min') === 'min',
        'isochrones.raster.mode must be min to include direct travel from snapped facilities'
      );
      validateRequirement(
        ID,
        !raster.sourcePaths,
        'isochrones.raster.sourcePaths is owned by the drive-time facility adapter'
      );
      // Each facility contributes a zero-cost snapped origin and its two directed endpoint
      // paths. Reuse the isochrone splatter, so direct on-road travel reaches the raster even
      // when both original endpoint costs exceed an early band. No second routing search.
      const pathPositions = createTransientView(
        graph,
        `${id}-source-path-positions`,
        'float32x2',
        3 * facilityCount
      );
      const pathCosts = createTransientView(
        graph,
        `${id}-source-path-costs`,
        'float32',
        3 * facilityCount
      );
      const pathAssignments = createTransientView(
        graph,
        `${id}-source-path-assignments`,
        'uint32',
        3 * facilityCount
      );
      const pathOffsets = createTransientView(
        graph,
        `${id}-source-path-offsets`,
        'uint32',
        3 * facilityCount + 1
      );
      const pathNeighbors = createTransientView(
        graph,
        `${id}-source-path-neighbors`,
        'uint32',
        2 * facilityCount
      );
      const pathWeights = createTransientView(
        graph,
        `${id}-source-path-weights`,
        'float32',
        2 * facilityCount
      );
      graph.add(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-source-path-geometry`,
          operation: 'GPURecipeDriveTimeCatchment',
          invocationCount: facilityCount,
          bindings: [
            {name: 'roads', view: facilityRoads.nodes, type: 'u32', access: 'read'},
            {name: 'fractions', view: facilityFractions, type: 'f32', access: 'read'},
            {name: 'positions', view: topology.nodePositions, type: 'f32', access: 'read'},
            {name: 'startNodes', view: startNodes, type: 'u32', access: 'read'},
            {name: 'pathPositions', view: pathPositions, type: 'f32', access: 'read_write'},
            {name: 'pathCosts', view: pathCosts, type: 'f32', access: 'read_write'},
            {name: 'pathAssignments', view: pathAssignments, type: 'u32', access: 'read_write'}
          ],
          declarations: `const NODE_COUNT: u32 = ${nodeCount}u;`,
          body: `let sourceNode = roads[roadsOffset + 2u * index];
  let targetNode = roads[roadsOffset + 2u * index + 1u];
  let valid = sourceNode < NODE_COUNT && targetNode < NODE_COUNT;
  var startPoint = vec2f(0.0);
  var endPoint = vec2f(0.0);
  if (valid) {
    startPoint = vec2f(positions[positionsOffset + 2u * sourceNode], positions[positionsOffset + 2u * sourceNode + 1u]);
    endPoint = vec2f(positions[positionsOffset + 2u * targetNode], positions[positionsOffset + 2u * targetNode + 1u]);
  }
  let point = mix(startPoint, endPoint, vec2f(fractions[fractionsOffset + index]));
  let offset = pathPositionsOffset + 6u * index;
  pathPositions[offset] = point.x;
  pathPositions[offset + 1u] = point.y;
  pathPositions[offset + 2u] = startPoint.x;
  pathPositions[offset + 3u] = startPoint.y;
  pathPositions[offset + 4u] = endPoint.x;
  pathPositions[offset + 5u] = endPoint.y;
  // A runtime bitcast preserves infinity without an invalid WGSL constant expression.
  var infinityBits = 0x7f800000u;
  let infinity = bitcast<f32>(infinityBits);
  pathCosts[pathCostsOffset + 3u * index] = select(infinity, 0.0, valid);
  pathCosts[pathCostsOffset + 3u * index + 1u] = infinity;
  pathCosts[pathCostsOffset + 3u * index + 2u] = infinity;
  let assignment = select(2u * index + 1u, 2u * index, startNodes[startNodesOffset + 2u * index] != 0xffffffffu);
  pathAssignments[pathAssignmentsOffset + 3u * index] = assignment;
  pathAssignments[pathAssignmentsOffset + 3u * index + 1u] = assignment;
  pathAssignments[pathAssignmentsOffset + 3u * index + 2u] = assignment;`
        })
      );
      graph.add(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-source-path-topology`,
          operation: 'GPURecipeDriveTimeCatchment',
          invocationCount: facilityCount,
          bindings: [
            {name: 'roadCosts', view: facilityRoads.costs, type: 'f32', access: 'read'},
            {name: 'fractions', view: facilityFractions, type: 'f32', access: 'read'},
            {name: 'offsets', view: pathOffsets, type: 'u32', access: 'read_write'},
            {name: 'neighbors', view: pathNeighbors, type: 'u32', access: 'read_write'},
            {name: 'weights', view: pathWeights, type: 'f32', access: 'read_write'}
          ],
          declarations: `const FACILITY_COUNT: u32 = ${facilityCount}u;`,
          body: `let fraction = fractions[fractionsOffset + index];
  let forwardCost = roadCosts[roadCostsOffset + 2u * index];
  let reverseCost = roadCosts[roadCostsOffset + 2u * index + 1u];
  offsets[offsetsOffset + 3u * index] = 2u * index;
  offsets[offsetsOffset + 3u * index + 1u] = 2u * index + 2u;
  offsets[offsetsOffset + 3u * index + 2u] = 2u * index + 2u;
  if (index + 1u == FACILITY_COUNT) { offsets[offsetsOffset + 3u * FACILITY_COUNT] = 2u * FACILITY_COUNT; }
  neighbors[neighborsOffset + 2u * index] = 3u * index + 1u;
  neighbors[neighborsOffset + 2u * index + 1u] = 3u * index + 2u;
  weights[weightsOffset + 2u * index] = select(-1.0, fraction * max(reverseCost, 0.0), reverseCost >= 0.0 || fraction == 0.0);
  weights[weightsOffset + 2u * index + 1u] = select(-1.0, (1.0 - fraction) * max(forwardCost, 0.0), forwardCost >= 0.0 || fraction == 1.0);`
        })
      );
      raster = {
        ...raster,
        sourcePaths: {
          topology: {
            offsets: pathOffsets,
            neighbors: pathNeighbors,
            weights: pathWeights,
            nodePositions: pathPositions
          },
          costs: pathCosts,
          assignments: pathAssignments
        }
      };
    }
    builder.add(
      new GPUNetworkIsochrones({
        id: `${id}-isochrones`,
        topology,
        costs: nodeCosts,
        // Input from the service areas: enables the facility-labelled outputs
        // (`raster.output.pixelFacilities`, `triangleFacilities`, `cellOutline.byFacility`). The
        // facility row is the start row, `2 * facility + endpoint`.
        assignments,
        costBreaks: isochrones.costBreaks,
        parameters: isochrones.parameters,
        raster,
        cellOutline: isochrones.cellOutline
      })
    );
  }
  let isochroneDemand: GPUDriveTimeCatchmentRecipeOutput['isochroneDemand'];
  const polygons = isochrones?.cellOutline?.rings?.output.polygons;
  if (isochrones?.joinDemand && polygons) {
    const join = isochrones.joinDemand;
    const provided = outputs.isochroneDemand ?? {};
    isochroneDemand = {
      pointFeatureIds: getOrCreateView(
        graph,
        `${id}-isochrone-demand`,
        'uint32',
        demandCount,
        provided.pointFeatureIds
      ),
      status: {
        overflow: getOrCreateView(
          graph,
          `${id}-isochrone-join-overflow`,
          'uint32',
          1,
          provided.status?.overflow
        ),
        uncertainCount: getOrCreateView(
          graph,
          `${id}-isochrone-join-uncertain`,
          'uint32',
          1,
          provided.status?.uncertainCount
        )
      }
    };
    builder.add(
      new GPUPointInPolygonJoin({
        id: `${id}-isochrone-join`,
        points: {kind: 'points', positions: props.demand},
        polygons: {
          kind: 'polygons',
          positions: polygons.positions,
          featureOffsets: polygons.featureOffsets,
          polygonOffsets: polygons.polygonOffsets,
          ringOffsets: polygons.ringOffsets
        },
        candidateCapacity: join.candidateCapacity,
        output: {
          pointFeatureIds: isochroneDemand.pointFeatureIds,
          status: isochroneDemand.status
        }
      })
    );
  }

  const snappedEdges = getOrCreateView(
    graph,
    `${id}-demand-edges`,
    'uint32',
    demandCount,
    scratch.demandEdges
  );
  const demandFractions = createTransientView(
    graph,
    `${id}-demand-fractions`,
    'float32',
    demandCount
  );
  builder.add(
    new GPUNetworkSnapping({
      id: `${id}-snap-demand`,
      spatialContext: snappingContext,
      positions: props.demand,
      topology,
      maximumSnapDistance: props.maximumSnapDistance,
      candidateCapacity: props.candidateCapacity,
      // The exact scan has no size limit here, as before the snapping limit existed.
      maximumUnindexedWork: Number.MAX_SAFE_INTEGER,
      output: {
        status: {overflow: snapping?.status.demandOverflow},
        snappedEdges,
        snapFractions: demandFractions,
        // The demand adapter splits the road cost itself; these two are written for a caller
        // that supplied them.
        sourceCosts: scratch.demandSourceCosts,
        targetCosts: scratch.demandTargetCosts
      }
    })
  );

  // Adapter: demand drive time from the snapped road, the node costs and the facilities on the
  // same road. Demand is reached from the road's source driving source -> target, from its target
  // driving target -> source, or straight along the road from a facility snapped to it.
  const demandTimes = getOrCreateView(
    graph,
    `${id}-demand-times`,
    'float32',
    demandCount,
    outputs.demandTimes
  );
  const demandRoads = addSnappedRoadNodes(graph, {
    id: `${id}-demand-roads`,
    snappedEdges,
    edgeSources,
    topology,
    nodeCount
  });
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-demand-times-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'demand-time',
      bindings: [
        {name: 'roads', view: demandRoads.nodes, type: 'u32', access: 'read'},
        {name: 'roadCosts', view: demandRoads.costs, type: 'f32', access: 'read'},
        {name: 'roadKeys', view: demandRoads.keys, type: 'u32', access: 'read'},
        {name: 'fractions', view: demandFractions, type: 'f32', access: 'read'},
        {name: 'nodeCosts', view: nodeCosts, type: 'f32', access: 'read'},
        {name: 'sortedFacilityKeys', view: sortedFacilityKeys, type: 'u32', access: 'read'},
        {
          name: 'sortedFacilityFractions',
          view: sortedFacilityFractions,
          type: 'u32',
          access: 'read'
        },
        {name: 'times', view: demandTimes, type: 'f32', access: 'read_write'}
      ],
      invocationCount: demandCount,
      declarations: `const NODE_COUNT: u32 = ${nodeCount}u;
const FACILITY_COUNT: u32 = ${facilityCount}u;
const UNREACHED: f32 = 3.0e38;
${getBinarySearchWGSL('sortedFacilityKeys')}`,
      body: `let sourceNode = roads[roadsOffset + 2u * index];
  let targetNode = roads[roadsOffset + 2u * index + 1u];
  var best = UNREACHED;
  if (sourceNode < NODE_COUNT && targetNode < NODE_COUNT) {
    let fraction = fractions[fractionsOffset + index];
    let forwardCost = roadCosts[roadCostsOffset + 2u * index];
    let reverseCost = roadCosts[roadCostsOffset + 2u * index + 1u];
    if (forwardCost >= 0.0 || fraction == 0.0) {
      let viaSource = nodeCosts[nodeCostsOffset + sourceNode] + fraction * max(forwardCost, 0.0);
      if (viaSource < best) {
        best = viaSource;
      }
    }
    if (reverseCost >= 0.0 || fraction == 1.0) {
      let viaTarget = nodeCosts[nodeCostsOffset + targetNode] + (1.0 - fraction) * max(reverseCost, 0.0);
      if (viaTarget < best) {
        best = viaTarget;
      }
    }
    // The facilities on this road are one run of the sorted keys: a bounded neighbourhood of the
    // demand point (the facilities that share its road), not a scan of every facility.
    let roadKey = roadKeys[roadKeysOffset + index];
    let isReversed = sourceNode > targetNode;
    for (
      var position = lowerBoundSortedFacilityKeys(0u, FACILITY_COUNT, roadKey);
      position < FACILITY_COUNT && sortedFacilityKeys[sortedFacilityKeysOffset + position] == roadKey;
      position++
    ) {
      // Fraction of the facility along this demand's edge, then the signed way from it.
      let facilityBits = sortedFacilityFractions[sortedFacilityFractionsOffset + position];
      let facilityFraction = bitcast<f32>(facilityBits & 0x7fffffffu);
      let sameDirection = ((facilityBits & 0x80000000u) != 0u) == isReversed;
      let ahead = fraction - select(1.0 - facilityFraction, facilityFraction, sameDirection);
      if (ahead >= 0.0 && forwardCost >= 0.0 && ahead * forwardCost < best) {
        best = ahead * forwardCost;
      }
      if (ahead <= 0.0 && reverseCost >= 0.0 && -ahead * reverseCost < best) {
        best = -ahead * reverseCost;
      }
    }
  }
  if (best < UNREACHED) {
    times[timesOffset + index] = best;
  } else {
    var infinityBits = 0x7f800000u;
    times[timesOffset + index] = bitcast<f32>(infinityBits);
  }`
    })
  );

  const demandBands = getOrCreateView(
    graph,
    `${id}-demand-bands`,
    'uint32',
    demandCount,
    outputs.demandBands
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-demand-bands-kernel`,
      operation: 'GPURecipeDriveTimeCatchment',
      variant: 'band',
      bindings: [
        {name: 'times', view: demandTimes, type: 'f32', access: 'read'},
        {name: 'bandBreaks', view: props.costBreaks, type: 'f32', access: 'read'},
        {name: 'bands', view: demandBands, type: 'u32', access: 'read_write'}
      ],
      invocationCount: demandCount,
      declarations: `const BAND_COUNT: u32 = ${bandCount}u;`,
      body: `let time = times[timesOffset + index];
  var band = 0xffffffffu;
  if (time < 3.0e38) {
    for (var k = 0u; k < BAND_COUNT; k++) {
      if (time <= bandBreaks[bandBreaksOffset + k]) {
        band = k;
        break;
      }
    }
  }
  bands[bandsOffset + index] = band;`
    })
  );

  const capacity = props.bandCapacity ?? bandCount;
  const provided = outputs.bands ?? {};
  const bands = {
    keys: getOrCreateView(graph, `${id}-band-keys`, 'uint32', capacity, provided.keys),
    counts: getOrCreateView(graph, `${id}-band-counts`, 'uint32', capacity, provided.counts),
    status: {
      count: getOrCreateView(graph, `${id}-band-count`, 'uint32', 1, provided.status?.count),
      overflow: getOrCreateView(
        graph,
        `${id}-band-overflow`,
        'uint32',
        1,
        provided.status?.overflow
      ),
      requiredCount: getOrCreateView(
        graph,
        `${id}-band-required-count`,
        'uint32',
        1,
        provided.status?.requiredCount
      )
    },
    sumValues: props.demandValues
      ? getOrCreateView(graph, `${id}-band-sums`, 'float32', capacity, provided.sumValues)
      : undefined,
    means: props.demandValues
      ? getOrCreateView(graph, `${id}-band-means`, 'float32', capacity, provided.means)
      : undefined
  };
  builder.add(
    new GPUGroupStatistics({
      id: `${id}-band-statistics`,
      keys: demandBands,
      columns: props.demandValues
        ? [
            {
              values: props.demandValues,
              statistics: ['sum', 'mean'],
              output: {
                sums: getOrCreateView(
                  graph,
                  `${id}-band-sum-words`,
                  'uint32x2',
                  capacity,
                  scratch.bandSumWords
                ),
                sumValues: bands.sumValues,
                means: bands.means
              }
            }
          ]
        : [],
      output: {
        keys: bands.keys,
        counts: bands.counts,
        status: bands.status
      }
    })
  );

  return {
    spatialContext,
    contributors: builder.contributors,
    startNodes,
    startCosts,
    assignments,
    nodeCosts,
    demandTimes,
    demandBands,
    bands,
    isochroneRings: isochrones?.cellOutline?.rings?.output,
    isochroneDemand,
    snapping,
    outputs: {
      assignments,
      nodeCosts,
      status: {serviceAreaConverged},
      demandTimes,
      demandBands,
      bands,
      ...(isochroneDemand ? {isochroneDemand} : {}),
      ...(snapping ? {snapping} : {})
    },
    intermediates: {startNodes, startCosts},
    status: {
      serviceAreaConverged,
      stages: [
        ...(snapping
          ? [
              {stage: 'snap-facilities', status: {overflow: snapping.status.facilitiesOverflow}},
              {stage: 'snap-demand', status: {overflow: snapping.status.demandOverflow}}
            ]
          : []),
        {stage: 'service-areas', status: {converged: serviceAreaConverged}},
        ...getIsochroneStatusStages(isochrones),
        {
          stage: 'band-statistics',
          status: bands.status
        },
        ...(isochroneDemand
          ? [
              {
                stage: 'isochrone-demand',
                status: isochroneDemand.status
              }
            ]
          : [])
      ]
    }
  };
}
