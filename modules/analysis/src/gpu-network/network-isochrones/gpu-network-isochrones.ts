// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement,
  validateMaximumCount,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  isGraphDataView,
  isGraphVectorView,
  getGraphViewChunks
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUNetworkTopology,
  type GPUNetworkTopology
} from '../../analysis-core/contracts/network-topology';
import {
  GPUIsobandRings,
  GPUIsobands,
  getGPUIsobandRingsMaximumEdgeCapacity,
  getGPUIsobandRingsMaximumRingCapacity,
  GPU_ISOBANDS_PARAMETER_LENGTH
} from '../../raster-analysis/isolines/index';
import type {GPUIsobandRingsOutput, GPUIsobandsOutput} from '../../raster-analysis/isolines/index';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUCellAggregation} from '../../gpu-spatial-analysis/cell-aggregation/index';
import type {GPUCellTable} from '../../gpu-spatial-analysis/cell-aggregation/index';
import {GPUPointToCell} from '../../gpu-spatial-analysis/cell-indexing/index';
import {GPUCellSetOutline} from '../../gpu-spatial-analysis/cell-set-outline/index';
import type {
  GPUCellSetOutlineOutput,
  GPUCellSetOutlineRings
} from '../../gpu-spatial-analysis/cell-set-outline/index';
import {
  resolveGPUSpatialContext,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {GPUNetworkServiceAreas} from '../network-analysis/gpu-network-service-areas';
import {GPUNetworkReachability} from '../network-reachability/gpu-network-reachability';
import {
  getNetworkPhysicalBuffer,
  validateNetworkViewBindings,
  validateNetworkWritableViews
} from '../network-view-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} from './network-isochrones-parameters';
import {
  createIsochronesBandParametersNode,
  createIsochronesCellFacilityNodes,
  createIsochronesDecodeNode,
  createIsochronesInclusiveBreaksNode,
  createIsochronesNodeMaskNode,
  createIsochronesSplatNode,
  createIsochronesTriangleFacilityNodes,
  getIsochronesEncodedInitialValue,
  GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES,
  type GPUNetworkIsochronesMode
} from './network-isochrones-passes';

const DEFAULT_ID = 'network-isochrones';
const DEFAULT_MAXIMUM_BUFFER_PIXELS = 6;
const MAXIMUM_BUFFER_PIXELS_LIMIT = 16;
const DEFAULT_MAXIMUM_SAMPLES_PER_EDGE = 64;
const MAXIMUM_SAMPLES_PER_EDGE_LIMIT = 1024;
const DEFAULT_UNREACHED_COST = 1e30;
/** The raster path is planar; the cell path takes longitude/latitude. */
const ISOCHRONES_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  measure: 'distance'
});

/** Raster path of {@link GPUNetworkIsochrones}: splat edge costs to a raster, then isobands. */
export type GPUNetworkIsochronesRaster = {
  /** Raster width in pixels, at least 2. */
  width: number;
  /** Raster height in pixels, at least 2. */
  height: number;
  /**
   * `'min'` (default) keeps the smallest cost that lands on a pixel, the usual isochrone: a pixel
   * is as close as the closest street sample. `'max'` keeps the largest, a conservative surface
   * where a pixel is within a cost only when every street sample touching it is.
   *
   * In `'max'` mode the cost of a street sample is its network cost, the cheaper of the two
   * directions of its road: `min(cost[u] + t * weight, cost[v] + (1 - t) * reverseWeight)` when a
   * slot `v -> u` exists and `v` is reached, the forward term alone on a one-way road. The maximum
   * is then taken over the samples and roads within the buffer of the pixel, so a pixel on the
   * start node holds the cost of the costliest street sample within the radius (the cost of
   * travelling up to one buffer radius, plus the walk), not 0. A slot
   * `v -> u` is taken as the other direction of the same road; among parallel slots the costliest
   * is used, which is exact when every parallel road lists both directions with one weight.
   */
  mode?: GPUNetworkIsochronesMode;
  /**
   * Compile-time cap in pixels of the half-window that one edge sample marks per axis, 1 to 16.
   * Defaults to 6. A buffer radius wider than this many pixels is truncated.
   */
  maximumBufferPixels?: number;
  /**
   * Compile-time cap on the samples taken along one edge, 2 to 1024. Defaults to 64. An edge longer
   * than `maximumSamplesPerEdge * bufferRadius / 2` leaves gaps, so raise it or use a larger buffer.
   */
  maximumSamplesPerEdge?: number;
  /**
   * Additional directed paths from snapped sources to their road endpoints. Min raster mode only.
   * Uses the same positions/extent units and splat budgets as the main network. Costs are the
   * starting cost at each path node, not a second routing search. This retains direct source-road
   * travel that cannot be reconstructed from the main network's endpoint costs alone.
   */
  sourcePaths?: {
    topology: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
    costs: GraphDataView<'float32'>;
    /** Required with pixelFacilities; uses the main network's facility/start-row identity. */
    assignments?: GraphDataView<'uint32'>;
  };

  /**
   * Value of pixels that no edge sample reached. Defaults to 1e30, a finite number far above any
   * break, so unreached space lands in the last band. Pixels next to the network interpolate
   * toward it, which insets the outer contour by up to one pixel.
   */
  unreachedCost?: number;
  /**
   * Caller-owned outputs of {@link GPUIsobands} (band classes or triangles), plus the optional
   * `values` raster. Band `k` holds cost in `(costBreaks[k - 1], costBreaks[k]]`: `GPUIsobands`
   * reads each break moved to the next float32 above it, so a cost exactly on a break is within it.
   */
  output: GPUIsobandsOutput & {
    /** Optional `width * height` float32 cost raster, row 0 at `minY`. */
    values?: GraphDataView<'float32'>;
    /**
     * Optional `width * height` uint32 facility row (index into `startNodes`) of the edge sample that
     * set each pixel's cost, `0xffffffff` for unreached pixels. Needs `assignments`, mode `'min'`
     * and at most 255 start nodes. The facility shares the pixel word with the cost, which then keeps
     * its top 24 bits (relative precision 2^-15, rounded down); ties go to the lowest facility row.
     */
    pixelFacilities?: GraphDataView<'uint32'>;
    /**
     * Optional facility row per isoband triangle (length `triangleBands.length`), taken from the
     * labelled raster sample nearest to the triangle centroid in world units across the full
     * raster, breaking distance ties by smaller facility row. A spatial hierarchy prunes empty
     * regions and regions beyond the nearest known distance. Writes
     * `0xffffffff` when the raster has no labels or past `count`. Needs `pixelFacilities` and the triangle output.
     */
    triangleFacilities?: GraphDataView<'uint32'>;
  };
  /**
   * Optional closed band rings (`GPUIsobandRings`): shells and holes per cost band as GeoArrow
   * offsets, in the raster's units. `rings.output.ringGroups` receives the band of each ring. Bands
   * of several facilities that touch merge into one ring; use the cell path with `byFacility` for
   * rings that never mix facilities.
   */
  rings?: {
    /** Compile-time capacity of the intermediate boundary edge list, see `GPUIsobandRings`. */
    edgeCapacity: number;
    /** Vertex matching distance in raster units, see `GPUIsobandRings`. */
    vertexTolerance?: number;
    /** Caller-owned ring outputs. */
    output: GPUIsobandRingsOutput;
  };
};

/** Cell-outline path of {@link GPUNetworkIsochrones}: outline the cells that contain reached nodes. */
export type GPUNetworkIsochronesCellOutline = {
  /** Cell family of the outline. */
  family: 'h3' | 'quadbin';
  /** Cell resolution: Quadbin 0 to 26, H3 0 to 15. */
  resolution: number;
  /**
   * Optional caller-owned table that receives the reached cells (sorted, distinct). Provide it to
   * read its `count`, `overflow` and optional `candidateOverflow`. The full status is forwarded
   * to the outline: cell loss sets output.status.candidateOverflow, or suppresses the outline
   * count when that view is omitted. Nested rings are suppressed until complete input returns.
   * Defaults to a graph transient of `tableCapacity` rows.
   */
  table?: GPUCellTable;
  /** Row capacity of the transient table when `table` is omitted. Defaults to the node count. */
  tableCapacity?: number;
  /**
   * Label every reached cell with the facility (row of `startNodes`) of its cheapest reached node and
   * outline each facility separately: the outline gets the labels as `groups`, so borders between
   * facilities are emitted from both sides, `output.groups` carries the facility of each segment
   * and rings never mix facilities (give `rings.output.ringGroups` to read each ring's facility).
   * Needs `assignments`. Cells are labelled by their lowest-cost node, ties to the lowest facility
   * row. Quadbin cells are keyed through `GPUPointToCell`, like H3.
   */
  byFacility?: boolean;
  /**
   * Optional caller-owned facility row per table row (`table.cells.length` rows, `0xffffffff` past
   * the cell count). Needs `byFacility`. Defaults to a graph transient.
   */
  cellFacilities?: GraphDataView<'uint32'>;
  /** Caller-owned boundary segments of the reached cells. */
  output: GPUCellSetOutlineOutput;
  /**
   * Optional ring assembly of the boundary segments (`GPUSegmentRingAssembly`): closed isochrone
   * rings as GeoArrow-style offsets plus positions, shells and holes classified, holes assigned to
   * shells. Rings are in longitude/latitude degrees. One run covers one cost limit
   * (`cellCostLimit`); several bands need one run each.
   */
  rings?: GPUCellSetOutlineRings;
};

/** GPU-written views of {@link GPUNetworkIsochrones}. */
export type GPUNetworkIsochronesOutput = {
  /** Status words of the search. */
  status?: {
    /**
     * Optional one-row flag written by the search (with `startNodes`): 1 when it reached a fixpoint,
     * 0 when `maximumIterations` (or `maximumLabelIterations`) stopped it first. A 0 means `costs`
     * are truncated: nodes past the last round stay unreached or too costly, so every isochrone
     * polygon is smaller than the true one. As with `GPUNetworkReachability` and
     * `GPUNetworkServiceAreas`, omitting it accepts that truncation unchecked; pass it whenever the
     * network diameter is not known to fit `maximumIterations`. Not accepted without `startNodes`
     * (the caller's costs carry their own status).
     */
    converged?: GraphDataView<'uint32'>;
    /**
     * Optional one-row complement of converged: 1 when the search exhausted a cost or label
     * round budget, otherwise 0. May be requested alone; requires startNodes. Search incumbents
     * and their derived geometry are still published on a budget exit.
     */
    iterationLimitReached?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkIsochrones}.
 *
 * Per-frame (no recompile): the contents of the topology, costs or start nodes, `costBreaks` and
 * `parameters`. Compile-time: node, edge, pixel and break capacities, a numeric `startCount`,
 * raster and cell options.
 *
 * `costs` and `assignments` stay top-level because each is an input when `startNodes` is absent and
 * an output the contributor writes when `startNodes` is given. The search convergence flag is
 * `output.status.converged`.
 */
export type GPUNetworkIsochronesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-isochrones'`. */
  id?: string;
  /**
   * Directed network with node positions. `neighbors` indices `>= nodeCount` are ignored; negative
   * or NaN `weights` are skipped. The raster path uses `nodePositions` in the units of the extent
   * (projected meters give an isotropic buffer; longitude/latitude degrees would give an
   * anisotropic one). The cell path needs longitude/latitude degrees.
   */
  topology: GPUNetworkTopology & {nodePositions: GraphDataView<'float32x2'>};
  /**
   * Optional coordinates of `topology.nodePositions`. The raster path is planar (pixel grid,
   * `bufferRadius` and `walkCostPerUnit` are Euclidean in the units of the extent), so a
   * longitude/latitude context with `raster` throws: project to a planar frame first. The cell
   * path indexes longitude/latitude degrees, so a planar context with `cellOutline` throws.
   * Omitted resolves to native planar for `raster`, or longitude/latitude with metric `'none'`
   * for `cellOutline` alone. With both paths and no context, the raster default applies and
   * cell-outline positions remain unchecked for compatibility. Distance buffers accept an
   * azimuthal-equidistant local frame and reject a Lambert equal-area frame.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Per-node travel cost. An input (for example `GPUNetworkServiceAreasOutput.costs`) unless
   * `startNodes` is given, in which case this contributor runs `GPUNetworkReachability` and writes
   * it. Unreached nodes are `+Infinity`. Its length defines the node count.
   */
  costs: GraphDataView<'float32'>;
  /**
   * Optional start nodes (facilities) that begin a multi-source search writing `costs`. Pass
   * `output.status.converged` to detect a search cut short by `maximumIterations`.
   */
  startNodes?: GraphDataView<'uint32'>;
  /**
   * Optional per-node facility row (index into `startNodes`) of the nearest facility,
   * `GPU_NETWORK_REACHABILITY_NONE` where none reaches it. Like `costs` it is an input (for
   * example `GPUNetworkServiceAreasOutput.assignments`) unless `startNodes` is given, in which case
   * this contributor writes it: the search is then a `GPUNetworkServiceAreas` run
   * (nearest-facility allocation, ties to the lowest row) rather than a plain multi-source
   * `GPUNetworkReachability`, and `costs` is the same minimum cost either way. It enables the
   * facility-labelled outputs (`raster.output.pixelFacilities`, `triangleFacilities`,
   * `cellOutline.byFacility`). Length `costs.length`.
   */
  assignments?: GraphDataView<'uint32'>;
  /** Compile-time label-phase rounds of the allocation (with `assignments`), see `GPUNetworkServiceAreas`. */
  maximumLabelIterations?: number;
  /** Optional per-start initial cost. */
  startCosts?: GraphDataView<'float32'>;
  /**
   * Optional active start count. A number is a compile-time constant in `[0, startNodes.length]`; a
   * one-row view is read per frame. Defaults to every row.
   */
  startCount?: number | GraphDataView<'uint32'>;
  /** Optional one-row search cost cutoff. */
  costLimit?: GraphDataView<'float32'>;
  /** Compile-time relaxation rounds of the search (with `startNodes`). Defaults to 64. */
  maximumIterations?: number;
  /**
   * Compile-time hops one workgroup chains per round of the search (with `startNodes`), forwarded
   * to `GPUNetworkReachability` (default 32) or `GPUNetworkServiceAreas` (default 16). A round
   * covers about 0.7 of `localIterations` hops in practice, so a network whose longest shortest
   * path has `h` hops needs about `ceil(h / (0.7 * localIterations)) + 4` rounds: size
   * `maximumIterations` with `recommendReachabilityIterations`.
   */
  localIterations?: number;
  /**
   * Optional GPU-written status of the search (with `startNodes`). Not accepted without
   * `startNodes` (the caller's costs carry their own status).
   */
  output?: GPUNetworkIsochronesOutput;
  /**
   * Ascending cost thresholds (isochrone times); the length is the maximum break count. The first
   * `breakCount` entries of `parameters` are active.
   */
  costBreaks: GraphDataView<'float32'>;
  /** Per-frame float32 parameters, written by `getGPUNetworkIsochronesParameterValues` (12 elements). */
  parameters: GraphDataView<'float32'>;
  /** Raster path. At least one of `raster` and `cellOutline` is required. */
  raster?: GPUNetworkIsochronesRaster;
  /** Cell-outline path. */
  cellOutline?: GPUNetworkIsochronesCellOutline;
};

/**
 * Network isochrone polygons: edge-interpolated travel costs splatted to a raster and contoured
 * into bands, or the outline of the cells that contain reached nodes.
 *
 * The contributor composes existing parts. Node costs come from the caller (any producer such as
 * `GPUNetworkServiceAreas`, whose `costs` are the multi-source minimum used here) or from a
 * `GPUNetworkReachability` search over `startNodes`, whose optional `output.status.converged` flag reports a search
 * cut short by `maximumIterations` (truncated costs, too-small polygons). A splat kernel then samples every edge at a
 * bounded number of points, interpolates the cost linearly from the source node along the edge
 * weight, and writes the minimum (or maximum) cost to the pixels within the walking buffer using
 * integer atomics on order-preserving float bits. Results do not depend on thread order. In
 * `'max'` mode a sample's cost is first reduced to the cheaper of the two directions of its road,
 * so the maximum never compares a road with its own reverse direction. The raster feeds
 * `GPUIsobands`, so band boundaries are the isochrone polygons as triangles per band. Bands are
 * upper-inclusive, `(a, b]`.
 *
 * The raster path is planar in the units of `nodePositions`; the pixel grid has resolution
 * `(maxX - minX) / width`, so the polygons are accurate to about a pixel and thinner than a pixel
 * features need a buffer. Costs of pixels in the buffer are the sample cost plus
 * `walkCostPerUnit` times the distance, which gives the standard "street cost plus walk" isochrone.
 * Edges are straight segments between node positions, so curved street geometry is approximated.
 * An explicit `spatialContext` checks both paths at construction; omitted defaults follow
 * `GPUNetworkIsochronesProps.spatialContext`.
 *
 * The cell path outlines cells holding at least one reached node (cost at or below
 * `cellCostLimit`) with `GPUCellSetOutline`; `cellOutline.rings` chains the outline into closed
 * isochrone rings with shells and holes. It samples nodes only, so on sparse networks long
 * edges leave their cells out; use the raster path for smooth polygons.
 *
 * Nothing here compiles, submits or reads back.
 */
export class GPUNetworkIsochrones implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkIsochronesProps;
  /** Number of network nodes, `costs.length`. */
  readonly nodeCount: number;
  /** Resolved raster accumulation mode, when the raster path is on. */
  readonly mode: GPUNetworkIsochronesMode;
  /** Resolved context: native planar for raster, longitude/latitude for cell outlines alone. */
  readonly spatialContext: GPUSpatialContext;
  private readonly cellTableCapacity: number;

  constructor(props: GPUNetworkIsochronesProps) {
    // Contexts are resolved public identities; only option/port maps are owned snapshots.
    // Uploaded topologies may also carry ownership metadata (buffers, devices, graphs). Only
    // declared topology ports belong in this snapshot; resource owners can contain cycles.
    props = snapshotIsochroneOptions(props, ISOCHRONE_OPTION_FIELDS);
    this.id = props.id ?? DEFAULT_ID;
    this.props = props;
    const {id} = this;
    const identity = {id, contributor: 'GPUNetworkIsochrones'};
    const {raster, cellOutline} = props;
    if (!raster && !cellOutline) {
      throw new Error(`${id} needs raster or cellOutline`);
    }
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext ?? (raster ? undefined : GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT),
      ISOCHRONES_SPATIAL_CONTEXT_SUPPORT
    );
    const coordinateSpace = this.spatialContext.coordinateSpace;
    if (raster && coordinateSpace === 'longitude-latitude') {
      throw new Error(`${id} raster needs planar positions; project to a planar frame first`);
    }
    // Legacy combined outputs without a context leave cell-outline positions unchecked.
    if (cellOutline && coordinateSpace === 'planar' && props.spatialContext) {
      throw new Error(`${id} cellOutline needs longitude-latitude positions`);
    }
    for (const [name, view] of [
      ['costs', props.costs],
      ['costBreaks', props.costBreaks],
      ['parameters', props.parameters],
      ['startNodes', props.startNodes],
      ['assignments', props.assignments]
    ] as const) {
      if (isGraphVectorView(view)) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const {nodeCount: networkNodeCount, edgeCount} = validateGPUNetworkTopology(
      id,
      props.topology,
      {weights: true, nodePositions: true}
    );
    validateSinglePackedView(props.costs, ['float32'], `${id} costs`);
    validateSinglePackedView(props.costBreaks, ['float32'], `${id} costBreaks`);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    this.nodeCount = props.costs.length;
    if (this.nodeCount < 1) {
      throw new Error(`${id} costs must contain at least one node`);
    }
    if (networkNodeCount !== this.nodeCount) {
      throw new Error(`${id} topology must contain one node per costs row`);
    }
    if (edgeCount < 1) {
      throw new Error(`${id} needs at least one edge`);
    }
    if (props.costBreaks.length < 1) {
      throw new Error(`${id} costBreaks must contain at least one row`);
    }
    if (props.parameters.length < GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH} float32 rows`
      );
    }
    if (!props.startNodes) {
      for (const name of [
        'startCosts',
        'startCount',
        'costLimit',
        'output',
        'localIterations'
      ] as const) {
        if (props[name] !== undefined) {
          throw new Error(`${id} ${name} needs startNodes`);
        }
      }
    } else {
      validateSinglePackedView(props.startNodes, ['uint32'], `${id} startNodes`);
      if (props.startCosts) {
        validateSinglePackedView(props.startCosts, ['float32'], identity, 'startCosts');
        validateRequirement(
          id,
          props.startCosts.length === props.startNodes.length,
          `startCosts length must equal startNodes.length (${props.startNodes.length}); received ${props.startCosts.length}`
        );
      }
      if (props.costLimit) {
        validateSinglePackedView(props.costLimit, ['float32'], identity, 'costLimit');
        validateRequirement(
          id,
          props.costLimit.length === 1,
          `costLimit must contain exactly one row; received ${props.costLimit.length}`
        );
      }
      if (typeof props.startCount === 'object') {
        validateSinglePackedView(props.startCount, ['uint32'], identity, 'startCount');
        validateRequirement(
          id,
          props.startCount.length === 1,
          `startCount must contain exactly one row; received ${props.startCount.length}`
        );
      }
      if (typeof props.startCount === 'number') {
        validateCapacity(id, 'startCount', props.startCount, 0, props.startNodes.length);
      }
    }
    for (const [name, view] of Object.entries(props.output?.status ?? {})) {
      if (!view) continue;
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      validateRequirement(
        id,
        view.length === 1,
        `output.status.${name} must contain exactly one row`
      );
    }
    if (props.assignments) {
      validateSinglePackedView(props.assignments, ['uint32'], `${id} assignments`);
      if (props.assignments.length !== this.nodeCount) {
        throw new Error(`${id} assignments length must equal the node count`);
      }
    }
    if (props.maximumLabelIterations !== undefined && !(props.assignments && props.startNodes)) {
      throw new Error(`${id} maximumLabelIterations needs startNodes and assignments`);
    }
    this.mode = raster?.mode ?? 'min';
    if (this.mode !== 'min' && this.mode !== 'max') {
      throw new Error(`${id} raster.mode must be 'min' or 'max'`);
    }
    if (raster) {
      for (const [name, value, minimum, maximum] of [
        ['raster.width', raster.width, 2, 0x7fffffff],
        ['raster.height', raster.height, 2, 0x7fffffff],
        [
          'raster.maximumBufferPixels',
          raster.maximumBufferPixels ?? DEFAULT_MAXIMUM_BUFFER_PIXELS,
          1,
          MAXIMUM_BUFFER_PIXELS_LIMIT
        ],
        [
          'raster.maximumSamplesPerEdge',
          raster.maximumSamplesPerEdge ?? DEFAULT_MAXIMUM_SAMPLES_PER_EDGE,
          2,
          MAXIMUM_SAMPLES_PER_EDGE_LIMIT
        ]
      ] as const) {
        validateCapacity(id, name, value, minimum, maximum);
      }
      if (raster.sourcePaths) {
        const paths = raster.sourcePaths;
        validateRequirement(
          id,
          this.mode === 'min',
          "raster.sourcePaths requires raster.mode 'min'"
        );
        const {nodeCount: pathNodeCount} = validateGPUNetworkTopology(id, paths.topology, {
          weights: true,
          nodePositions: true
        });
        validateSinglePackedView(paths.costs, ['float32'], `${id} raster.sourcePaths.costs`);
        validateRequirement(
          id,
          paths.costs.length === pathNodeCount,
          'raster.sourcePaths.costs must contain one row per path node'
        );
        if (paths.assignments) {
          validateSinglePackedView(
            paths.assignments,
            ['uint32'],
            `${id} raster.sourcePaths.assignments`
          );
          validateRequirement(
            id,
            paths.assignments.length === pathNodeCount,
            'raster.sourcePaths.assignments must contain one row per path node'
          );
        }
        validateRequirement(
          id,
          !raster.output.pixelFacilities || Boolean(paths.assignments),
          'raster.sourcePaths.assignments is required with pixelFacilities'
        );
      }
      const unreached = raster.unreachedCost ?? DEFAULT_UNREACHED_COST;
      if (!Number.isFinite(unreached) || !(unreached > 0) || unreached >= 3.0e38) {
        throw new Error(`${id} raster.unreachedCost must be a positive finite number below 3e38`);
      }
      const pixelCount = raster.width * raster.height;
      if (!Number.isSafeInteger(pixelCount) || pixelCount > 0x7fffffff) {
        throw new Error(`${id} raster width * height must fit in 31 bits`);
      }
      if (raster.output.pixelFacilities) {
        if (!props.assignments) {
          throw new Error(`${id} raster.output.pixelFacilities needs assignments`);
        }
        if (this.mode !== 'min') {
          throw new Error(`${id} raster.output.pixelFacilities needs raster.mode 'min'`);
        }
        if (
          props.startNodes &&
          props.startNodes.length > GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES
        ) {
          throw new Error(
            `${id} raster.output.pixelFacilities supports at most ${GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES} startNodes`
          );
        }
        validateSinglePackedView(
          raster.output.pixelFacilities,
          ['uint32'],
          `${id} raster.output.pixelFacilities`
        );
        if (raster.output.pixelFacilities.length < pixelCount) {
          throw new Error(`${id} raster.output.pixelFacilities must hold width * height rows`);
        }
      }
      if (raster.output.triangleFacilities) {
        if (!raster.output.pixelFacilities || !raster.output.triangles) {
          throw new Error(
            `${id} raster.output.triangleFacilities needs pixelFacilities and the triangle output`
          );
        }
        validateSinglePackedView(
          raster.output.triangleFacilities,
          ['uint32'],
          `${id} raster.output.triangleFacilities`
        );
        if (raster.output.triangleFacilities.length !== raster.output.triangleBands!.length) {
          throw new Error(
            `${id} raster.output.triangleFacilities length must equal triangleBands length`
          );
        }
      }
      if (raster.rings) {
        validateCapacity(id, 'raster.rings.edgeCapacity', raster.rings.edgeCapacity);
      }
      if (raster.output.values) {
        validateSinglePackedView(raster.output.values, ['float32'], `${id} raster.output.values`);
        if (raster.output.values.length < pixelCount) {
          throw new Error(`${id} raster.output.values must hold width * height rows`);
        }
      }
    }
    this.cellTableCapacity =
      cellOutline?.table?.cells.length ?? cellOutline?.tableCapacity ?? this.nodeCount;
    if (cellOutline) {
      for (const name of ['sums', 'sumValues', 'minimums', 'maximums'] as const) {
        validateRequirement(
          id,
          !cellOutline.table?.[name],
          `cellOutline.table.${name} has no values input; use GPUCellAggregation with values for reductions`
        );
      }
      if (cellOutline.family !== 'h3' && cellOutline.family !== 'quadbin') {
        throw new Error(`${id} cellOutline.family must be 'h3' or 'quadbin'`);
      }
      if (cellOutline.byFacility && !props.assignments) {
        throw new Error(`${id} cellOutline.byFacility needs assignments`);
      }
      if (cellOutline.cellFacilities && !cellOutline.byFacility) {
        throw new Error(`${id} cellOutline.cellFacilities needs cellOutline.byFacility`);
      }
      if (cellOutline.byFacility && !cellOutline.output.groups) {
        throw new Error(`${id} cellOutline.byFacility needs cellOutline.output.groups`);
      }
      if (!cellOutline.table && cellOutline.tableCapacity !== undefined) {
        validateCapacity(id, 'cellOutline.tableCapacity', cellOutline.tableCapacity);
      }
      validateCapacity(id, 'cellOutline table row capacity', this.cellTableCapacity);
      validateRequirement(
        id,
        Boolean(cellOutline.table) || this.cellTableCapacity >= this.nodeCount,
        `cellOutline.tableCapacity must be at least the node count (${this.nodeCount}) when table is omitted, because transient cell loss cannot be reported`
      );
      if (cellOutline.cellFacilities) {
        validateSinglePackedView(
          cellOutline.cellFacilities,
          ['uint32'],
          `${id} cellOutline.cellFacilities`
        );
        validateRequirement(
          id,
          cellOutline.cellFacilities.length === this.cellTableCapacity,
          `cellOutline.cellFacilities must contain exactly ${this.cellTableCapacity} table rows`
        );
      }
    }
    const inputs = collectNamedIsochroneViews({
      topology: props.topology,
      raster: {sourcePaths: raster?.sourcePaths},
      costBreaks: props.costBreaks,
      parameters: props.parameters,
      startNodes: props.startNodes,
      startCosts: props.startCosts,
      startCount: typeof props.startCount === 'object' ? props.startCount : undefined,
      costLimit: props.costLimit,
      ...(!props.startNodes ? {costs: props.costs, assignments: props.assignments} : {})
    });
    const geometryOutputs = collectNamedIsochroneViews({
      ...(props.startNodes ? {costs: props.costs, assignments: props.assignments} : {}),
      raster: {output: raster?.output, rings: {output: raster?.rings?.output}},
      cellOutline: {
        table: cellOutline?.table,
        cellFacilities: cellOutline?.cellFacilities,
        output: cellOutline?.output,
        rings: {output: cellOutline?.rings?.output}
      }
    });
    const searchStatus = collectNamedIsochroneViews({output: props.output});
    const previousStatus: Record<string, GraphDataView> = {};
    for (const [name, view] of Object.entries(searchStatus)) {
      validateIsochronePhysicalSeparation(
        id,
        {[name]: view},
        previousStatus,
        'output.status views must use separate buffers'
      );
      previousStatus[name] = view;
    }
    validateIsochronePhysicalSeparation(
      id,
      searchStatus,
      {...inputs, ...geometryOutputs},
      'output.status views must not share buffers with inputs or geometry outputs'
    );
    validateIsochronePhysicalSeparation(
      id,
      geometryOutputs,
      inputs,
      'outputs must not share buffers with inputs'
    );
    // Compare logical ranges across the whole composition, including outputs written in
    // different passes. Each child contributor validates the bindings it uses together.
    validateNetworkWritableViews(id, {...geometryOutputs, ...searchStatus}, [
      ...(raster
        ? [
            {
              outputs: {
                'raster.output.values': raster.output.values,
                'raster.output.pixelFacilities': raster.output.pixelFacilities
              }
            }
          ]
        : []),
      ...(raster?.output.triangleFacilities
        ? [
            {
              outputs: {'raster.output.triangleFacilities': raster.output.triangleFacilities},
              inputs: {
                'raster.output.triangles': raster.output.triangles,
                'raster.output.status.count': raster.output.status?.count,
                'raster.output.pixelFacilities': raster.output.pixelFacilities,
                parameters: props.parameters
              }
            }
          ]
        : []),
      ...(cellOutline?.byFacility
        ? [
            {
              outputs: {'cellOutline.cellFacilities': cellOutline.cellFacilities},
              inputs: {
                costs: props.costs,
                assignments: props.assignments,
                'cellOutline.table.cells': cellOutline.table?.cells,
                'cellOutline.table.status.count': cellOutline.table?.status.count
              }
            }
          ]
        : [])
    ]);
  }

  /**
   * Returns the optional search nodes, then for the raster path: band parameters, encoded fill,
   * edge splat, decode and the isobands nodes; for the cell path: node mask, point keying (H3),
   * cell aggregation and the outline nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, nodeCount, mode} = this;
    const {raster, cellOutline} = props;
    const {offsets, neighbors, weights, nodePositions} = props.topology;
    const namedViews = collectNamedIsochroneViews(props);
    const views = Object.values(namedViews);
    validateGraphViewsBelongToGraph(id, graph, views);
    validateGraphChunkBindingRanges(graph, id, namedViews);
    if (raster?.rings) {
      validateCapacity(
        id,
        'raster.rings.edgeCapacity',
        raster.rings.edgeCapacity,
        1,
        getGPUIsobandRingsMaximumEdgeCapacity(graph.device)
      );
      validateMaximumCount(
        id,
        'raster.rings.output ring capacity',
        raster.rings.output.ringOffsets.length - 1,
        getGPUIsobandRingsMaximumRingCapacity(graph.device)
      );
    }
    const pixelCount = raster ? raster.width * raster.height : 0;
    const capacity = this.cellTableCapacity;
    validateNetworkViewBindings(id, graph, views, {
      ...(raster
        ? {
            'raster encoded': pixelCount * 4,
            ...(!raster.output.values ? {'raster values': pixelCount * 4} : {}),
            'raster bandParameters': GPU_ISOBANDS_PARAMETER_LENGTH * 4,
            'raster inclusiveBreaks': props.costBreaks.length * 4
          }
        : {}),
      ...(cellOutline
        ? {
            'cellOutline reachedMask': nodeCount * 4,
            ...(!cellOutline.table
              ? {
                  'cellOutline table.cells': capacity * 8,
                  'cellOutline table.counts': capacity * 4,
                  'cellOutline table.status.count': 4,
                  'cellOutline table.status.overflow': 4,
                  'cellOutline table.status.candidateOverflow': 4
                }
              : {}),
            ...(cellOutline.family === 'h3' || cellOutline.byFacility
              ? {'cellOutline nodeCells': nodeCount * 8}
              : {}),
            ...(cellOutline.byFacility
              ? {
                  'cellOutline cellMinimumCosts': capacity * 4,
                  ...(!cellOutline.cellFacilities
                    ? {'cellOutline cellFacilities': capacity * 4}
                    : {})
                }
              : {})
          }
        : {})
    });
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (props.startNodes && props.assignments) {
      nodes.push(
        ...new GPUNetworkServiceAreas({
          id: `${id}-search`,
          topology: props.topology,
          facilities: props.startNodes,
          facilityCosts: props.startCosts,
          facilityCount: props.startCount,
          costLimit: props.costLimit,
          maximumIterations: props.maximumIterations,
          maximumLabelIterations: props.maximumLabelIterations,
          localIterations: props.localIterations,
          output: {
            assignments: props.assignments,
            costs: props.costs,
            status: props.output?.status
          }
        }).getCommandNodes(graph)
      );
    } else if (props.startNodes) {
      nodes.push(
        ...new GPUNetworkReachability({
          id: `${id}-search`,
          topology: props.topology,
          startNodes: props.startNodes,
          startCosts: props.startCosts,
          startCount: props.startCount,
          costLimit: props.costLimit,
          maximumIterations: props.maximumIterations,
          localIterations: props.localIterations,
          output: {costs: props.costs, status: props.output?.status}
        }).getCommandNodes(graph)
      );
    }

    if (raster) {
      const {width, height} = raster;
      const pixelCount = width * height;
      // The isobands read their extent from the per-frame parameters, so the grid only sizes them.
      const grid: GPURasterMetadata = {
        width,
        height,
        affine: [1, 0, 0, 0, 1, 0],
        pixelInterpretation: 'point'
      };
      const encoded = createTransientView(graph, `${id}-encoded`, 'uint32', pixelCount);
      const values =
        raster.output.values ?? createTransientView(graph, `${id}-values`, 'float32', pixelCount);
      const bandParameters = createTransientView(
        graph,
        `${id}-band-parameters`,
        'float32',
        GPU_ISOBANDS_PARAMETER_LENGTH
      );
      // Upper-inclusive bands: GPUIsobands reads each break moved to the next float32 above it.
      const inclusiveBreaks = createTransientView(
        graph,
        `${id}-inclusive-breaks`,
        'float32',
        props.costBreaks.length
      );
      nodes.push(
        createIsochronesInclusiveBreaksNode<Parameters>(graph, {
          id: `${id}-inclusive-breaks`,
          costBreaks: props.costBreaks,
          inclusiveBreaks
        }),
        createIsochronesBandParametersNode<Parameters>(graph, {
          id: `${id}-band-parameters`,
          width,
          height,
          parameters: props.parameters,
          bandParameters
        }),
        createFillNode<Parameters>(graph, {
          id: `${id}-encoded-fill`,
          operation: 'GPUNetworkIsochrones',
          view: encoded,
          type: 'u32',
          value: `${getIsochronesEncodedInitialValue(mode)}u`
        }),
        createIsochronesSplatNode<Parameters>(graph, {
          id: `${id}-splat`,
          nodeCount,
          width,
          height,
          mode,
          maximumBufferPixels: raster.maximumBufferPixels ?? DEFAULT_MAXIMUM_BUFFER_PIXELS,
          maximumSamplesPerEdge: raster.maximumSamplesPerEdge ?? DEFAULT_MAXIMUM_SAMPLES_PER_EDGE,
          offsets,
          neighbors,
          weights,
          costs: props.costs,
          nodePositions,
          parameters: props.parameters,
          encoded,
          assignments: raster.output.pixelFacilities ? props.assignments : undefined
        }),
        ...(raster.sourcePaths
          ? [
              createIsochronesSplatNode<Parameters>(graph, {
                id: `${id}-source-paths-splat`,
                nodeCount: raster.sourcePaths.costs.length,
                width,
                height,
                mode,
                maximumBufferPixels: raster.maximumBufferPixels ?? DEFAULT_MAXIMUM_BUFFER_PIXELS,
                maximumSamplesPerEdge:
                  raster.maximumSamplesPerEdge ?? DEFAULT_MAXIMUM_SAMPLES_PER_EDGE,
                offsets: raster.sourcePaths.topology.offsets,
                neighbors: raster.sourcePaths.topology.neighbors,
                weights: raster.sourcePaths.topology.weights!,
                costs: raster.sourcePaths.costs,
                nodePositions: raster.sourcePaths.topology.nodePositions,
                parameters: props.parameters,
                encoded,
                assignments: raster.output.pixelFacilities
                  ? raster.sourcePaths.assignments
                  : undefined
              })
            ]
          : []),
        createIsochronesDecodeNode<Parameters>(graph, {
          id: `${id}-decode`,
          mode,
          unreachedCost: raster.unreachedCost ?? DEFAULT_UNREACHED_COST,
          encoded,
          values,
          pixelCount,
          pixelFacilities: raster.output.pixelFacilities
        }),
        ...new GPUIsobands({
          id: `${id}-isobands`,
          grid,
          values,
          breaks: inclusiveBreaks,
          parameters: bandParameters,
          output: {
            bandClasses: raster.output.bandClasses,
            triangles: raster.output.triangles,
            triangleBands: raster.output.triangleBands,
            edges: raster.output.edges,
            edgeBands: raster.output.edgeBands,
            status: raster.output.status
          }
        }).getCommandNodes(graph)
      );
      if (raster.output.triangleFacilities) {
        nodes.push(
          ...createIsochronesTriangleFacilityNodes<Parameters>(graph, {
            id: `${id}-triangle-facilities`,
            width,
            height,
            triangles: raster.output.triangles!,
            triangleCount: raster.output.status!.count!,
            parameters: props.parameters,
            pixelFacilities: raster.output.pixelFacilities!,
            triangleFacilities: raster.output.triangleFacilities
          })
        );
      }
      if (raster.rings) {
        nodes.push(
          ...new GPUIsobandRings({
            id: `${id}-rings`,
            grid,
            values,
            breaks: inclusiveBreaks,
            parameters: bandParameters,
            edgeCapacity: raster.rings.edgeCapacity,
            vertexTolerance: raster.rings.vertexTolerance,
            output: raster.rings.output
          }).getCommandNodes(graph)
        );
      }
    }

    if (cellOutline) {
      const mask = createTransientView(graph, `${id}-reached-mask`, 'uint32', nodeCount);
      nodes.push(
        createIsochronesNodeMaskNode<Parameters>(graph, {
          id: `${id}-reached-mask`,
          costs: props.costs,
          parameters: props.parameters,
          mask
        })
      );
      const capacity = this.cellTableCapacity;
      let cellFacilities: GraphDataView<'uint32'> | undefined;
      const table: GPUCellTable = cellOutline.table ?? {
        cells: createTransientView(graph, `${id}-reached-cells`, 'uint32x2', capacity),
        counts: createTransientView(graph, `${id}-reached-counts`, 'uint32', capacity),
        status: {
          count: createTransientView(graph, `${id}-reached-count`, 'uint32', 1),
          overflow: createTransientView(graph, `${id}-reached-overflow`, 'uint32', 1),
          candidateOverflow: createTransientView(
            graph,
            `${id}-reached-candidate-overflow`,
            'uint32',
            1
          )
        }
      };
      const byFacility = Boolean(cellOutline.byFacility);
      if (cellOutline.family === 'quadbin' && !byFacility) {
        nodes.push(
          ...new GPUCellAggregation({
            id: `${id}-reached-table`,
            family: 'quadbin',
            resolution: cellOutline.resolution,
            positions: nodePositions,
            mask,
            output: table
          }).getCommandNodes(graph)
        );
      } else {
        const nodeCells = createTransientView(graph, `${id}-node-cells`, 'uint32x2', nodeCount);
        nodes.push(
          ...new GPUPointToCell({
            id: `${id}-node-cells`,
            family: cellOutline.family,
            resolution: cellOutline.resolution,
            positions: nodePositions,
            mask,
            output: {cells: nodeCells}
          }).getCommandNodes(graph),
          ...new GPUCellAggregation({
            id: `${id}-reached-table`,
            family: cellOutline.family,
            resolution: cellOutline.resolution,
            cells: nodeCells,
            output: table
          }).getCommandNodes(graph)
        );
        if (byFacility) {
          cellFacilities =
            cellOutline.cellFacilities ??
            createTransientView(graph, `${id}-cell-facilities`, 'uint32', capacity);
          nodes.push(
            ...createIsochronesCellFacilityNodes<Parameters>(graph, {
              id: `${id}-cell-facilities`,
              nodeCount,
              nodeCells,
              tableCells: table.cells,
              tableCount: table.status.count,
              costs: props.costs,
              assignments: props.assignments!,
              cellMinimumCosts: createTransientView(
                graph,
                `${id}-cell-minimum-costs`,
                'uint32',
                capacity
              ),
              cellFacilities
            })
          );
        }
      }
      // Snapshot source loss before publishing caller outputs: disjoint status slices may
      // share an allocation, but a single WebGPU dispatch cannot read and write that allocation.
      const outlineSourceStatus = {
        count: table.status.count,
        overflow: createTransientView(graph, `${id}-outline-source-overflow`, 'uint32', 1),
        candidateOverflow: table.status.candidateOverflow
          ? createTransientView(graph, `${id}-outline-source-candidate-overflow`, 'uint32', 1)
          : undefined
      };
      for (const name of ['overflow', 'candidateOverflow'] as const) {
        const source = table.status[name];
        const destination = outlineSourceStatus[name];
        if (!source || !destination) continue;
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-outline-capture-${name}`,
            operation: 'GPUNetworkIsochrones',
            variant: 'capture-cell-source-loss',
            invocationCount: 1,
            bindings: [
              {name: 'source', view: source, type: 'u32', access: 'read'},
              {name: 'destination', view: destination, type: 'u32', access: 'read_write'}
            ],
            body: 'destination[destinationOffset] = source[sourceOffset];'
          })
        );
      }
      nodes.push(
        ...new GPUCellSetOutline({
          id: `${id}-outline`,
          family: cellOutline.family,
          source: {cells: table.cells, status: outlineSourceStatus},
          groups: cellFacilities,
          output: cellOutline.output,
          rings: cellOutline.rings
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}

type IsochroneOptionFields = {readonly [name: string]: true | IsochroneOptionFields};

const ISOCHRONE_TOPOLOGY_FIELDS = {
  offsets: true,
  neighbors: true,
  weights: true,
  nodePositions: true
} as const;
const ISOCHRONE_COMPACT_STATUS_FIELDS = {
  count: true,
  requiredCount: true,
  overflow: true
} as const;
const ISOCHRONE_RING_OUTPUT_FIELDS = {
  ringOffsets: true,
  positions: true,
  ringAreas: true,
  ringIsHole: true,
  ringShells: true,
  ringGroups: true,
  segmentRings: true,
  segmentVertices: true,
  segmentFlags: true,
  polygons: {
    kind: true,
    positions: true,
    ringOffsets: true,
    polygonOffsets: true,
    featureOffsets: true,
    sourceIds: true
  },
  status: {
    ...ISOCHRONE_COMPACT_STATUS_FIELDS,
    openSegmentCount: true,
    touchingSegmentCount: true,
    edgeCount: true,
    edgeOverflow: true,
    candidateOverflow: true,
    complete: true,
    polygonCount: true
  }
} as const;
/** Read declared fields by name so inherited/non-enumerable getters survive the snapshot. */
const ISOCHRONE_OPTION_FIELDS = {
  id: true,
  topology: ISOCHRONE_TOPOLOGY_FIELDS,
  spatialContext: true,
  costs: true,
  startNodes: true,
  assignments: true,
  maximumLabelIterations: true,
  startCosts: true,
  startCount: true,
  costLimit: true,
  maximumIterations: true,
  localIterations: true,
  output: {status: {converged: true, iterationLimitReached: true}},
  costBreaks: true,
  parameters: true,
  raster: {
    width: true,
    height: true,
    mode: true,
    maximumBufferPixels: true,
    maximumSamplesPerEdge: true,
    unreachedCost: true,
    sourcePaths: {topology: ISOCHRONE_TOPOLOGY_FIELDS, costs: true, assignments: true},
    output: {
      bandClasses: true,
      triangles: true,
      triangleBands: true,
      edges: true,
      edgeBands: true,
      values: true,
      pixelFacilities: true,
      triangleFacilities: true,
      status: {
        ...ISOCHRONE_COMPACT_STATUS_FIELDS,
        vertexCount: true,
        edgeCount: true,
        edgeOverflow: true,
        edgeRequiredCount: true
      }
    },
    rings: {edgeCapacity: true, vertexTolerance: true, output: ISOCHRONE_RING_OUTPUT_FIELDS}
  },
  cellOutline: {
    family: true,
    resolution: true,
    tableCapacity: true,
    byFacility: true,
    cellFacilities: true,
    table: {
      cells: true,
      counts: true,
      sums: true,
      sumValues: true,
      minimums: true,
      maximums: true,
      status: {...ISOCHRONE_COMPACT_STATUS_FIELDS, candidateOverflow: true, sumOverflow: true}
    },
    output: {
      rows: true,
      cells: true,
      edgeIndices: true,
      endpoints: true,
      groups: true,
      status: {...ISOCHRONE_COMPACT_STATUS_FIELDS, candidateOverflow: true}
    },
    rings: {vertexTolerance: true, normalizeWinding: true, output: ISOCHRONE_RING_OUTPUT_FIELDS}
  }
} as const satisfies Record<keyof GPUNetworkIsochronesProps, true | IsochroneOptionFields>;

/** Freeze declared option/port maps, retaining graph views and the resolved context identity. */
function snapshotIsochroneOptions<Value>(value: Value, fields: IsochroneOptionFields): Value {
  if (!value || typeof value !== 'object') return value;
  const snapshot = snapshotConstructionFields(value, Object.keys(fields) as (keyof Value)[]);
  return Object.freeze(
    Object.fromEntries(
      Object.entries(snapshot).map(([name, child]) => [
        name,
        fields[name] === true ? child : snapshotIsochroneOptions(child, fields[name])
      ])
    )
  ) as Value;
}

/** Retain caller-facing paths for ownership, usage, and alias diagnostics. */
function collectNamedIsochroneViews(bag: object, prefix = ''): Record<string, GraphDataView> {
  return Object.fromEntries(
    Object.entries(bag).flatMap<readonly [string, GraphDataView]>(([field, value]) => {
      const name = prefix ? `${prefix}.${field}` : field;
      if (isGraphDataView(value)) return [[name, value]];
      if (isGraphVectorView(value)) {
        return getGraphViewChunks(value).map(
          (chunk, index) => [`${name}.chunks[${index}]`, chunk] as const
        );
      }
      if (!value || typeof value !== 'object') return [];
      return Object.entries(collectNamedIsochroneViews(value, name));
    })
  );
}

/** Physical read/write and scalar-status separation, with both offending port paths. */
function validateIsochronePhysicalSeparation(
  id: string,
  outputs: Record<string, GraphDataView>,
  inputs: Record<string, GraphDataView>,
  message: string
): void {
  const inputNames = new Map(
    Object.entries(inputs).map(([name, view]) => [getNetworkPhysicalBuffer(view), name])
  );
  for (const [name, view] of Object.entries(outputs)) {
    const inputName = inputNames.get(getNetworkPhysicalBuffer(view));
    validateRequirement(id, inputName === undefined, `${message}: ${name} and ${inputName}`);
  }
}
