// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotFlowAggregationConstructionProps} from './flow-aggregation-construction';
import {
  validateNetworkViewBindings,
  validateNetworkOutputsDisjointFromInputs
} from '../network-view-utils';

import {
  createTransientView,
  doGraphDataViewsOverlap,
  getViewBindingRange,
  getBoundedDispatchLayout,
  GPUGroupAggregation,
  GPUGather,
  GPUHashIndex,
  GPUHashIndexQuery,
  GPUReduction,
  GPUScan,
  GPUSort,
  GPU_HASH_INDEX_STATISTICS_LENGTH,
  GPU_HASH_QUERY_STATISTICS_LENGTH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  createPublishNode,
  createWGSLActiveCountDispatch
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUFloat32Positions} from '@luma.gl/graph/geospatial';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCapacity,
  validateMaximumCount,
  validateSinglePackedView,
  validatePowerOfTwo,
  validateCompactOutput,
  validateRequirement,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH} from '../../dataframe-analysis/time-window-filter/time-window-parameters';
import {GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH} from '../../dataframe-analysis/time-window-filter/time-words';
import {
  getSortedSegmentSumNodes,
  SEGMENT_WORKGROUP_SIZE
} from '../../analysis-core/kernels/sorted-segment-sums';
import {getTimeWindowClassifyNodes} from '../../dataframe-analysis/time-window-filter/time-window-classify-node';
import {
  FLOW_CHUNK_WORKGROUP_SIZE,
  createFlowContributionNodes,
  createFlowDecodeZonesNode,
  createFlowExtentToFloatNode,
  createFlowGatherIdsNode,
  createFlowGatherValuesNode,
  createFlowPairKeysNode,
  createFlowPairOverflowNode,
  createFlowPairSortPrepareNode,
  createFlowPairSumNode,
  createFlowSlotKeysNode,
  createFlowWeightKeysNode,
  getFlowWeightKeyBits,
  createFlowZoneNode,
  type FlowResolvedBounds
} from './flow-aggregation-kernels';
import {
  GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT,
  GPU_FLOW_AGGREGATION_MAXIMUM_PACKED_ZONE_COUNT
} from './flow-aggregation-pairs';
import {
  createFlowLexicographicPairNodes,
  createFlowLexicographicSlotNodes
} from './flow-lexicographic-pairs';

/** Literal bounds and radii are compiled as f32, whose largest finite magnitude is this value. */
const GPU_FLOW_AGGREGATION_MAXIMUM_LITERAL_MAGNITUDE = 3.4028234663852886e38;
/** Positive literal radii must remain nonzero after conversion to the shader's f32 type. */
const GPU_FLOW_AGGREGATION_MINIMUM_LITERAL_RADIUS = 2 ** -149;
/** The shared mapper clamps f32-to-i32 at 2^31 - 128; exclude that index from valid cells. */
const GPU_FLOW_AGGREGATION_MAXIMUM_HEXAGON_AXIS_SIZE = 2 ** 31 - 128;
/** Output capacity literals and aggregate hash-probe counters are represented as uint32. */
const GPU_FLOW_AGGREGATION_MAXIMUM_UINT32_COUNT = 0xffffffff;

/** Upper bound for the fixed sorts, scans and reductions, including published primitive internals. */
const FLOW_COMMAND_NODE_ALLOWANCE = 4096;

/** Dispatch parameter slots use the published 256-byte storage binding alignment. */
const FLOW_CHUNK_PARAMETER_SLOT_BYTE_LENGTH = 256;

/**
 * Inclusive `[minX, minY, maxX, maxY]` domain of a grid or hexagon zone lattice.
 *
 * A literal is compile-time topology. A GPU view is per-frame: one `float32x4` row or four packed
 * `float32` rows, the shape of a `GPUParameterBuffer` with `format: 'float32', length: 4`.
 */
export type GPUFlowAggregationBounds =
  | readonly [number, number, number, number]
  | GraphDataView<'float32x4'>
  | GraphDataView<'float32'>;

/**
 * Per-frame lattice size of grid and hexagon zones: a packed `uint32` view of at least two rows
 * `[columns, rows]`, for example a `GPUParameterBuffer` with `format: 'uint32'`.
 *
 * Each value is clamped to `1..gridSize` of the compile-time capacity. Zone IDs are row-major in
 * the active size (`row * columns + column`), cells outside it are rejected, and zone outputs keep
 * their capacity length: rows at and after `columns * rows` stay zero. Pair keys are built with the
 * origin and destination zone columns, so ranking and ties are unaffected by the active size beyond the zone IDs.
 * Pair this with a per-frame hexagon `radius` so the lattice always covers the bounds.
 */
export type GPUFlowAggregationActiveGridSize = GraphDataView<'uint32'>;

/**
 * How rows are assigned origin and destination zones.
 *
 * `kind`, `gridSize`, and `zoneCount` are compile-time. Literal bounds and a literal radius are
 * compile-time; GPU bounds and a GPU radius are per-frame. `gridSize` is the capacity of the
 * lattice: with `activeGridSize` the lattice size is per-frame and `gridSize` is only its upper
 * bound.
 */
export type GPUFlowAggregationZones =
  | {
      /** Row-major rectangular cells, same cell formula as `GPUGridBinning`. */
      kind: 'grid';
      /** Domain covered by the cells. */
      bounds: GPUFlowAggregationBounds;
      /** `[columns, rows]` positive integers. Compile-time capacity. */
      gridSize: readonly [number, number];
      /** Optional per-frame `[columns, rows]` of the lattice in use. See {@link GPUFlowAggregationActiveGridSize}. */
      activeGridSize?: GraphDataView<'uint32'>;
    }
  | {
      /** Pointy-top odd-r hexagons, same lattice as `GPUPointDensity`. */
      kind: 'hexagon';
      /** Domain covered by the lattice. Points outside it are rejected. */
      bounds: GPUFlowAggregationBounds;
      /** `[columns, rows]` positive integers, each at most 2^31 - 128 for signed cell coordinates. Compile-time capacity. */
      gridSize: readonly [number, number];
      /** Optional per-frame `[columns, rows]` of the lattice in use. See {@link GPUFlowAggregationActiveGridSize}. */
      activeGridSize?: GraphDataView<'uint32'>;
      /** Center-to-vertex radius. A number is compile-time; a one-row float32 view is per-frame. */
      radius: number | GraphDataView<'float32'>;
    }
  | {
      /** Caller-supplied zone IDs per row (`originZoneIds` and `destinationZoneIds`). */
      kind: 'ids';
      /** Number of zones. IDs at or above it are rejected. Compile-time. */
      zoneCount: number;
    };

/**
 * Optional per-frame instant time gate (a row is accepted when its timestamp is inside the window).
 *
 * Two modes, chosen by the `timestamps` format:
 * - Float: packed `float32` timestamps, optional `timestampsLow`, and a `float32` window written with
 *   `getGPUTimeWindowFilterParameterValues`.
 * - Int64 words: packed `uint32x2` `(low, high)` timestamps (for example epoch milliseconds from an
 *   Arrow `Int64` column) and a packed `uint32` window written with
 *   `getGPUTimeWindowFilterWordParameterValues`. Acceptance is exact.
 */
export type GPUFlowAggregationTimeWindow = {
  /** Packed `float32` timestamp per row, or packed `uint32x2` Int64 words per row. Per-frame contents. */
  timestamps: GraphDataView<'float32'> | GraphDataView<'uint32x2'>;
  /** Optional double-single low parts from `splitTimestamps`. Float timestamps only. Per-frame contents. */
  timestampsLow?: GraphDataView<'float32'>;
  /**
   * Float timestamps: packed window of at least 8 `float32` rows from `getGPUTimeWindowFilterParameterValues`.
   * Word timestamps: packed window of at least 8 `uint32` rows from `getGPUTimeWindowFilterWordParameterValues`.
   * Per-frame.
   */
  window: GraphDataView<'float32'> | GraphDataView<'uint32'>;
};

/**
 * Order in which float weight sums are accumulated.
 *
 * - `'sorted'` (default): rows are stably sorted by key and reduced per key in a fixed tree, so sums
 *   are bitwise identical across runs and devices. Costs extra sort, scan, and gather passes; about
 *   13-24 ms for 1M rows whether flows are concentrated or spread over 4096 pairs.
 * - `'atomic'`: compare-exchange float addition, rounding order depends on GPU scheduling. Skips the
 *   sorts, and was about 4x faster (21 ms against 85 ms for 1M rows) only when almost every row has
 *   its own pair (about 600k distinct pairs). Lanes that share a pair or zone retry on one address
 *   and serialize: 1M rows over 4096 pairs took 70 ms against 13 ms, and 1M rows with 90% on one
 *   pair took 12-17 seconds against 22 ms.
 */
export type GPUFlowAggregationSumOrder = 'atomic' | 'sorted';

/**
 * Properties for {@link GPUFlowAggregation}.
 *
 * Compile-time: zone kind, grid size, zone count, `pairCapacity`, `maximumProbeCount`,
 * `excludeSelfFlows`, literal bounds and radius, all lengths, and which optional views exist.
 * Per-frame: the contents of every input view (positions, zone IDs, weights, mask, timestamps, time
 * window, GPU bounds and radius).
 */
export type GPUFlowAggregationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'flow-aggregation'`. */
  id?: string;
  /** Zone assignment. Compile-time. */
  zones: GPUFlowAggregationZones;
  /** Origin positions for grid and hexagon zones (single view or chunked vector). Per-frame contents. */
  origins?: GPUFloat32Positions;
  /** Destination positions, same length as `origins`. Per-frame contents. */
  destinations?: GPUFloat32Positions;
  /** Packed origin zone IDs for `'ids'` zones. IDs `>= zoneCount` reject the row. Per-frame contents. */
  originZoneIds?: GraphDataView<'uint32'>;
  /** Packed destination zone IDs for `'ids'` zones, same length as `originZoneIds`. Per-frame contents. */
  destinationZoneIds?: GraphDataView<'uint32'>;
  /**
   * Float weight sum order for `flowWeights`, `zoneOutWeights`, and `zoneInWeights`. Compile-time.
   * Defaults to `'sorted'`.
   */
  sumOrder?: GPUFlowAggregationSumOrder;
  /** Optional per-row weight with one row per flow row. Non-finite weights are ignored in sums. Per-frame contents. */
  weights?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional packed per-row accept mask; nonzero accepts. Presence is compile-time, contents per-frame. */
  mask?: GraphDataView<'uint32'>;
  /** Optional instant time gate. Presence is compile-time, contents per-frame. */
  timeWindow?: GPUFlowAggregationTimeWindow;
  /** Reject rows whose origin and destination zones are equal. Compile-time. Defaults to `false`. */
  excludeSelfFlows?: boolean;
  /** Hash table slots, a power of two: the maximum number of distinct pairs retained. Compile-time. */
  pairCapacity: number;
  /**
   * Forwarded to `GPUHashIndex` and its query. Defaults to `pairCapacity`. Compile-time. The
   * aggregate probe count `rows x maximumProbeCount` is at most 4294967295 (2^32 - 1), so large
   * inputs need a probe limit below the default.
   */
  maximumProbeCount?: number;
  /** Top-K outputs. `ids` is the packed pair key through 65535 zones; above that it is the lowest source row of the pair. `ids.length` is K. */
  output: GPUFlowAggregationOutput;
};

/** GPU-written outputs of {@link GPUFlowAggregation}, beside the top-K {@link GPUCompactOutput}. */
export type GPUFlowAggregationOutput = GPUCompactOutput & {
  /** Status words beside the compact top-K cardinality. */
  status: GPUCompactOutput['status'] & {
    /** Optional one row: 1 when an accepted row's pair could not be stored in the table. */
    pairOverflow?: GraphDataView<'uint32'>;
  };
  /** Optional K rows of origin zone IDs aligned with `output.ids`. */
  flowOriginZoneIds?: GraphDataView<'uint32'>;
  /** Optional K rows of destination zone IDs aligned with `output.ids`. */
  flowDestinationZoneIds?: GraphDataView<'uint32'>;
  /** Optional K rows: accepted rows in each reported pair. */
  flowCounts?: GraphDataView<'uint32'>;
  /** Optional K rows: sum of finite weights per pair, or the count as float32 without `weights`. */
  flowWeights?: GraphDataView<'float32'>;
  /** Optional `zoneCount` rows: accepted rows per origin zone. */
  zoneOutCounts?: GraphDataView<'uint32'>;
  /** Optional `zoneCount` rows: accepted rows per destination zone. */
  zoneInCounts?: GraphDataView<'uint32'>;
  /** Optional `zoneCount` rows: sum of finite weights per origin zone. Requires `weights`. */
  zoneOutWeights?: GraphDataView<'float32'>;
  /** Optional `zoneCount` rows: sum of finite weights per destination zone. Requires `weights`. */
  zoneInWeights?: GraphDataView<'float32'>;
  /** Optional one row receiving the clamped flow count, for an indirect instanced draw. */
  drawInstanceCount?: GraphDataView<'uint32'>;
  /**
   * Optional two `float32` rows `[minimum, maximum]` of the nonzero per-origin-zone counts (both 0
   * when no zone has a count). Computed on the GPU, so a color range needs no readback. Layers can
   * take it directly as a `[min, max]` extent buffer.
   */
  zoneOutCountExtent?: GraphDataView<'float32'>;
  /** Same as `zoneOutCountExtent` for per-destination-zone counts. */
  zoneInCountExtent?: GraphDataView<'float32'>;
};

/**
 * Aggregates origin-destination movements into a weight-sorted top-K flow list and per-zone totals.
 *
 * Each source row is one movement. Rows are assigned origin and destination zones (grid cells,
 * hexagons, or caller IDs). Above 65535 zones, accepted rows are first stably sorted by separate
 * origin and destination columns and each tuple is identified by its lowest source row. Otherwise
 * accepted rows are grouped by the legacy `originZone * zoneCount + destinationZone` key
 * in a fixed-capacity `GPUHashIndex`, counts and weight sums are accumulated with
 * `GPUGroupAggregation` over each pair's lowest source row, and the table slots are ranked with two
 * stable `GPUSort` passes.
 *
 * Accepted row: both zones valid (below `zoneCount`; for grid and hexagon, a finite position inside
 * the inclusive bounds and inside the lattice), mask nonzero when given, time window accepts when
 * given, and not a self flow when `excludeSelfFlows`.
 *
 * Output: flows are sorted by weight descending, ties by pair key ascending (origin zone, then
 * destination zone); without `weights` the weight is the count. `output.status.requiredCount` is the number
 * of distinct pairs retained in the table, `output.status.count = min(requiredCount, K)`, and
 * `output.status.overflow` is 1 when `requiredCount > K`. Optional `status.candidateOverflow`
 * reports pair-table loss separately (the same condition as `status.pairOverflow`); then
 * `requiredCount` is a lower bound on the full pair requirement. `requiredCount > K` alone means
 * the list is a top-K truncation but every aggregate is exact. `status.pairOverflow` means distinct pairs
 * exceeded `pairCapacity` (or the probe limit): aggregates are incomplete and which pairs were
 * retained is unspecified (the `GPUHashIndex` contract). Rows `k >= count` of every K-row output are
 * rewritten each encoding with sentinels (`0xffffffff` IDs, zero counts and weights). Zone totals
 * count every accepted row, including rows whose pair overflowed the table.
 *
 * Counts are exact. With `sumOrder: 'sorted'` (default) every float sum (`flowWeights`, `zoneOutWeights`, `zoneInWeights`) is accumulated in source row
 * order within each key by a fixed tree, so it is bitwise identical across runs and devices, at the
 * cost of a stable sort, a scan, and gather passes per sum (about 3 sorts for pair and zone sums).
 * With `sumOrder: 'atomic'` float sums use compare-exchange atomic addition, so their rounding order
 * is not deterministic (small integer weights stay exact), and rows that share a pair or zone
 * serialize on one address; use it only when nearly every row has a distinct pair.
 * Both routes pack finite weight contributions once from source chunks into one row-aligned view,
 * shared by the pair, origin and destination sums. Chunked atomic input therefore adds one pack
 * dispatch per chunk, rather than three separately specialized aggregation passes.
 * Weight ranking uses the final float sum either way.
 *
 * The time gate accepts float timestamps (optionally double-single) or exact Int64 word timestamps
 * (`uint32x2`), see {@link GPUFlowAggregationTimeWindow}.
 *
 * Above 65535 zones `output.ids` holds the lowest accepted source row of each pair; use
 * `flowOriginZoneIds` and `flowDestinationZoneIds` for zone identities. The uint32 output layout
 * and the legacy packed IDs through 65535 zones are preserved. The hash probe count
 * `rows * maximumProbeCount` remains bounded by 4294967295.
 *
 * Construction snapshots controls, zone literals and nested time/output/status port maps before
 * validation. Graph views retain their identities and their contents remain live across submissions.
 * Non-goals: geodesic zones, flow bundling, chunked zone IDs, masks or timestamps, table resizing.

 */
export class GPUFlowAggregation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUFlowAggregationProps;
  /** Number of zones. */
  readonly zoneCount: number;
  /** Number of source rows. */
  readonly rowCount: number;
  /** Float weight sum order. */
  readonly sumOrder: GPUFlowAggregationSumOrder;

  /** Upper bound including one weight-pack dispatch per chunk, shared by every requested sum. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUFlowAggregationProps) {
    props = snapshotFlowAggregationConstructionProps(props);
    this.id = props.id ?? 'flow-aggregation';
    this.props = props;
    const id = this.id;
    const {zones, output, weights} = props;
    // Atomic and sorted routes pack each weight chunk once. All requested atomic aggregations
    // then consume the packed view, so their fixed passes are covered by the allowance.
    this.estimatedCommandNodeCount =
      FLOW_COMMAND_NODE_ALLOWANCE +
      [props.origins, props.destinations, props.weights].reduce(
        (count, view) => count + (view ? getGraphViewChunks(view).length : 0),
        0
      );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    this.sumOrder = props.sumOrder ?? 'sorted';
    if (this.sumOrder !== 'atomic' && this.sumOrder !== 'sorted') {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }

    if (zones.kind === 'ids') {
      this.zoneCount = zones.zoneCount;
    } else if (zones.kind === 'grid' || zones.kind === 'hexagon') {
      const [columns, rows] = zones.gridSize;
      validateCapacity(id, 'gridSize columns', columns);
      validateCapacity(id, 'gridSize rows', rows);
      if (zones.kind === 'hexagon') {
        validateCapacity(
          id,
          'hexagon gridSize columns',
          columns,
          1,
          GPU_FLOW_AGGREGATION_MAXIMUM_HEXAGON_AXIS_SIZE
        );
        validateCapacity(
          id,
          'hexagon gridSize rows',
          rows,
          1,
          GPU_FLOW_AGGREGATION_MAXIMUM_HEXAGON_AXIS_SIZE
        );
      }
      this.zoneCount = columns * rows;
    } else {
      throw new Error(`${id} zones.kind must be grid, hexagon, or ids`);
    }
    validateCapacity(id, 'zone count', this.zoneCount, 1, GPU_FLOW_AGGREGATION_MAXIMUM_ZONE_COUNT);

    if (zones.kind === 'ids') {
      if (!props.originZoneIds || !props.destinationZoneIds) {
        throw new Error(`${id} ids zones require originZoneIds and destinationZoneIds`);
      }
      if (props.origins || props.destinations) {
        throw new Error(`${id} ids zones do not accept origins or destinations`);
      }
      validateSinglePackedView(props.originZoneIds, ['uint32'], `${id} originZoneIds`);
      validateSinglePackedView(props.destinationZoneIds, ['uint32'], `${id} destinationZoneIds`);
      this.rowCount = props.originZoneIds.length;
      if (props.destinationZoneIds.length !== this.rowCount) {
        throw new Error(`${id} destinationZoneIds length must equal originZoneIds length`);
      }
    } else {
      if (!props.origins || !props.destinations) {
        throw new Error(`${id} ${zones.kind} zones require origins and destinations`);
      }
      if (props.originZoneIds || props.destinationZoneIds) {
        throw new Error(`${id} ${zones.kind} zones do not accept zone IDs`);
      }
      for (const chunk of getGraphViewChunks(props.origins)) {
        validateSinglePackedView(chunk, ['float32x2'], `${id} origins`);
      }
      for (const chunk of getGraphViewChunks(props.destinations)) {
        validateSinglePackedView(chunk, ['float32x2'], `${id} destinations`);
      }
      planGraphChunkDispatch(id, {origins: props.origins});
      planGraphChunkDispatch(id, {destinations: props.destinations});
      this.rowCount = props.origins.length;
      if (props.destinations.length !== this.rowCount) {
        throw new Error(`${id} destinations length must equal origins length`);
      }
      validateFlowBounds(id, zones.bounds);
      if (zones.activeGridSize) {
        validateSinglePackedView(zones.activeGridSize, ['uint32'], `${id} activeGridSize`);
        if (zones.activeGridSize.length < 2) {
          throw new Error(`${id} activeGridSize must contain two uint32 rows [columns, rows]`);
        }
      }
      if (zones.kind === 'hexagon') {
        const radius = zones.radius;
        if (typeof radius === 'number') {
          if (!Number.isFinite(radius) || radius <= 0) {
            throw new Error(`${id} hexagon radius must be positive and finite`);
          }
          validateRequirement(
            id,
            radius >= GPU_FLOW_AGGREGATION_MINIMUM_LITERAL_RADIUS &&
              radius <= GPU_FLOW_AGGREGATION_MAXIMUM_LITERAL_MAGNITUDE,
            `hexagon radius must be a positive float32 in ${GPU_FLOW_AGGREGATION_MINIMUM_LITERAL_RADIUS}..${GPU_FLOW_AGGREGATION_MAXIMUM_LITERAL_MAGNITUDE}`
          );
        } else {
          validateSinglePackedView(radius, ['float32'], `${id} hexagon radius`);
          if (radius.length < 1) {
            throw new Error(`${id} hexagon radius must contain one float32 row`);
          }
        }
      }
    }

    validateCapacity(id, 'rows', this.rowCount, 0, GPU_FLOW_AGGREGATION_MAXIMUM_UINT32_COUNT);
    if (weights) {
      for (const chunk of getGraphViewChunks(weights)) {
        validateSinglePackedView(chunk, ['float32'], `${id} weights`);
      }
      planGraphChunkDispatch(id, {weights});
      if (weights.length !== this.rowCount) {
        throw new Error(`${id} weights length must equal the row count`);
      }
    }
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    if (props.timeWindow) {
      const {timestamps, timestampsLow, window} = props.timeWindow;
      validateSinglePackedView(timestamps, ['float32', 'uint32x2'], `${id} timestamps`);
      if (timestamps.length !== this.rowCount) {
        throw new Error(`${id} timestamps length must equal the row count`);
      }
      const isWordTime = timestamps.format === 'uint32x2';
      if (timestampsLow) {
        if (isWordTime) {
          throw new Error(`${id} timestampsLow requires float32 timestamps`);
        }
        validateSinglePackedView(timestampsLow, ['float32'], `${id} timestampsLow`);
        if (timestampsLow.length !== this.rowCount) {
          throw new Error(`${id} timestampsLow length must equal the row count`);
        }
      }
      if (isWordTime) {
        validateSinglePackedView(window, ['uint32'], `${id} word window`);
        if (window.length < GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH) {
          throw new Error(
            `${id} word window must hold ${GPU_TIME_WINDOW_FILTER_WORD_PARAMETER_LENGTH} uint32 values`
          );
        }
      } else {
        validateSinglePackedView(window, ['float32'], `${id} window`);
        if (window.length < GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH) {
          throw new Error(
            `${id} window must hold ${GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH} float32 values`
          );
        }
      }
    }

    const capacity = props.pairCapacity;
    validatePowerOfTwo(id, 'pairCapacity', capacity, 1, 0x80000000);
    // The sorted pair sum runs one 256-lane workgroup per stored pair (rule 16: checked here, not
    // at graph build).
    if (
      props.weights &&
      this.sumOrder === 'sorted' &&
      capacity * SEGMENT_WORKGROUP_SIZE > 0xffffffff
    ) {
      throw new Error(`${id} pairCapacity must be at most 2^23 with weights and sumOrder 'sorted'`);
    }
    const probeLimit = props.maximumProbeCount ?? capacity;
    validateCapacity(id, 'maximumProbeCount', probeLimit, 1, capacity);
    // The hash index counts `rows * maximumProbeCount` aggregate probes in one uint32.
    validateMaximumCount(
      id,
      'rows x maximumProbeCount',
      this.rowCount * probeLimit,
      GPU_FLOW_AGGREGATION_MAXIMUM_UINT32_COUNT
    );

    validateCompactOutput(id, output);
    const topCount = output.ids.length;
    if (topCount < 1) {
      throw new Error(`${id} output.ids must contain at least one row`);
    }
    validateCapacity(id, 'output.ids rows', topCount, 1, GPU_FLOW_AGGREGATION_MAXIMUM_UINT32_COUNT);
    for (const [name, view] of [
      ['flowOriginZoneIds', output.flowOriginZoneIds],
      ['flowDestinationZoneIds', output.flowDestinationZoneIds],
      ['flowCounts', output.flowCounts]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length !== topCount) {
          throw new Error(`${id} output.${name} length must equal output.ids length`);
        }
      }
    }
    if (output.flowWeights) {
      validateSinglePackedView(output.flowWeights, ['float32'], `${id} output.flowWeights`);
      if (output.flowWeights.length !== topCount) {
        throw new Error(`${id} output.flowWeights length must equal output.ids length`);
      }
    }
    for (const [name, view] of [
      ['status.pairOverflow', output.status.pairOverflow],
      ['drawInstanceCount', output.drawInstanceCount]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.${name} must contain one uint32 row`);
        }
      }
    }
    for (const [name, view] of [
      ['zoneOutCounts', output.zoneOutCounts],
      ['zoneInCounts', output.zoneInCounts]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length !== this.zoneCount) {
          throw new Error(`${id} output.${name} length must equal the zone count`);
        }
      }
    }
    for (const [name, view] of [
      ['zoneOutCountExtent', output.zoneOutCountExtent],
      ['zoneInCountExtent', output.zoneInCountExtent]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length !== 2) {
          throw new Error(`${id} output.${name} must contain two float32 rows`);
        }
      }
    }
    for (const [name, view] of [
      ['zoneOutWeights', output.zoneOutWeights],
      ['zoneInWeights', output.zoneInWeights],
      ['flowWeights', output.flowWeights]
    ] as const) {
      if (!view) {
        continue;
      }
      if (name !== 'flowWeights' && !weights) {
        throw new Error(`${id} output.${name} requires weights`);
      }
      validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
      if (name !== 'flowWeights' && view.length !== this.zoneCount) {
        throw new Error(`${id} output.${name} length must equal the zone count`);
      }
    }

    validateNetworkOutputsDisjointFromInputs(
      id,
      getFlowAggregationOutputViews(this.props),
      getFlowAggregationInputViews(this.props)
    );
    const outputs = getFlowAggregationOutputViews(this.props).filter(
      (view): view is GraphDataView => Boolean(view)
    );
    const inputCoreBuffers = new Set(
      getFlowAggregationInputViews(this.props).flatMap(view =>
        view
          ? getGraphViewChunks(view).flatMap(chunk =>
              chunk.buffer.defaultBuffer ? [getCoreDefaultBuffer(chunk.buffer.defaultBuffer)] : []
            )
          : []
      )
    );
    for (const view of outputs) {
      validateRequirement(
        id,
        !view.buffer.defaultBuffer ||
          !inputCoreBuffers.has(getCoreDefaultBuffer(view.buffer.defaultBuffer)),
        'outputs must not share buffers with inputs'
      );
    }
    for (const [index, first] of outputs.entries()) {
      for (const second of outputs.slice(index + 1)) {
        validateRequirement(
          id,
          first.buffer === second.buffer ||
            !first.buffer.defaultBuffer ||
            !second.buffer.defaultBuffer ||
            getCoreDefaultBuffer(first.buffer.defaultBuffer) !==
              getCoreDefaultBuffer(second.buffer.defaultBuffer),
          'outputs sharing physical storage must use one graph buffer handle'
        );
        validateRequirement(
          id,
          !doGraphDataViewsOverlap(first, second),
          'output writable byte ranges must not overlap'
        );
      }
    }
    // Decoding and publication read these caller outputs while writing other caller fields.
    validateNetworkOutputsDisjointFromInputs(
      id,
      [output.flowOriginZoneIds, output.flowDestinationZoneIds],
      [output.ids]
    );
    validateNetworkOutputsDisjointFromInputs(
      id,
      [
        output.status.count,
        output.status.overflow,
        output.status.candidateOverflow,
        output.status.requiredCount,
        output.drawInstanceCount
      ],
      [output.status.pairOverflow]
    );
    const outputBindingGroups: (GraphDataView | undefined)[][] = [
      [output.flowOriginZoneIds, output.flowDestinationZoneIds],
      [output.flowCounts, output.flowWeights],
      [
        output.status.count,
        output.status.overflow,
        output.status.candidateOverflow,
        output.status.requiredCount,
        output.drawInstanceCount
      ]
    ];
    for (const group of outputBindingGroups) {
      validateDisjointWritableViews(
        id,
        Object.fromEntries(group.map((view, index) => [`output[${index}]`, view])),
        {strictAlignedBindings: true}
      );
    }
  }

  /**
   * Returns zone, time, pair-key, hash, aggregation, ranking, gather, publish, and zone-total
   * nodes in order. Declares work only.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, zoneCount, rowCount} = this;
    const {zones, weights, output} = props;
    const capacity = props.pairCapacity;
    const maxProbeCount = props.maximumProbeCount ?? capacity;
    const topCount = output.ids.length;
    validateFlowAggregationBindings(graph, this);
    const usesLexicographicPairs = zoneCount > GPU_FLOW_AGGREGATION_MAXIMUM_PACKED_ZONE_COUNT;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const createRowBases = (name: string, views: readonly GraphDataView[]) => {
      // Each scalar is at a 256-byte boundary, so all dispatches use the same binding offset.
      const wordsPerSlot = FLOW_CHUNK_PARAMETER_SLOT_BYTE_LENGTH / 4;
      const values = new Uint32Array(Math.max(views.length, 1) * wordsPerSlot);
      let rowBase = 0;
      for (const [index, view] of views.entries()) {
        values[index * wordsPerSlot] = rowBase;
        values[index * wordsPerSlot + 1] = view.length;
        values[index * wordsPerSlot + 2] = (view.byteOffset - getViewBindingRange(view).offset) / 4;
        const layout = getBoundedDispatchLayout(
          id,
          view.length,
          FLOW_CHUNK_WORKGROUP_SIZE,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
        values[index * wordsPerSlot + 3] = layout.x;
        values[index * wordsPerSlot + 4] = layout.y;
        rowBase += view.length;
      }
      const uploaded = createSpatialStatisticsParameterNodes(graph, `${id}-${name}-bases`, values);
      nodes.push(...uploaded.nodes);
      const parameterView = uploaded.parameters;
      return views.map((_, index) =>
        graph.createDataView(parameterView.buffer, {
          format: 'uint32',
          byteOffset: index * FLOW_CHUNK_PARAMETER_SLOT_BYTE_LENGTH,
          length: 5
        })
      );
    };
    let originZones: GraphDataView<'uint32'>;
    let destinationZones: GraphDataView<'uint32'>;
    if (zones.kind === 'ids') {
      originZones = props.originZoneIds as GraphDataView<'uint32'>;
      destinationZones = props.destinationZoneIds as GraphDataView<'uint32'>;
    } else {
      originZones = createTransientView(graph, `${id}-origin-zone-ids`, 'uint32', rowCount);
      destinationZones = createTransientView(
        graph,
        `${id}-destination-zone-ids`,
        'uint32',
        rowCount
      );
      const boundsView = Array.isArray(zones.bounds) ? undefined : (zones.bounds as GraphDataView);
      const bounds: FlowResolvedBounds =
        boundsView && boundsView.format === 'float32'
          ? graph.createDataView(boundsView.buffer, {
              format: 'float32x4',
              length: 1,
              byteOffset: boundsView.byteOffset
            })
          : (zones.bounds as FlowResolvedBounds);
      for (const [name, positions, target] of [
        ['origin', props.origins as GPUFloat32Positions, originZones],
        ['destination', props.destinations as GPUFloat32Positions, destinationZones]
      ] as const) {
        const chunks = getGraphViewChunks(positions);
        const rowBases = createRowBases(name, chunks);
        for (const [chunkIndex, chunk] of chunks.entries()) {
          if (chunk.length > 0) {
            nodes.push(
              createFlowZoneNode(graph, {
                id: `${id}-${name}-zones${chunks.length > 1 ? `-chunk-${chunkIndex}` : ''}`,
                kind: zones.kind,
                positions: chunk,
                zones: target,
                rowBase: rowBases[chunkIndex],
                gridSize: zones.gridSize,
                bounds,
                radius: zones.kind === 'hexagon' ? zones.radius : undefined,
                activeGridSize: zones.activeGridSize
              })
            );
          }
        }
      }
    }

    let timeMask: GraphDataView<'uint32'> | undefined;
    if (props.timeWindow && rowCount > 0) {
      timeMask = createTransientView(graph, `${id}-time-mask`, 'uint32', rowCount);
      nodes.push(
        ...getTimeWindowClassifyNodes(graph, {
          id: `${id}-time-classify`,
          timestamps: props.timeWindow.timestamps as GraphDataView<'float32'>,
          timestampsLow: props.timeWindow.timestampsLow,
          window: props.timeWindow.window,
          mask: timeMask
        })
      );
    }

    const pairKeys = createTransientView(graph, `${id}-pair-keys`, 'uint32', rowCount);
    const acceptedOriginZones = createTransientView(
      graph,
      `${id}-accepted-origin-zones`,
      'uint32',
      rowCount
    );
    const acceptedDestinationZones = createTransientView(
      graph,
      `${id}-accepted-destination-zones`,
      'uint32',
      rowCount
    );
    if (rowCount > 0) {
      nodes.push(
        createFlowPairKeysNode(graph, {
          id: `${id}-pair-keys`,
          originZones,
          destinationZones,
          mask: props.mask,
          timeMask,
          pairKeys,
          acceptedOriginZones,
          acceptedDestinationZones,
          zoneCount,
          useRowKeys: usesLexicographicPairs,
          excludeSelfFlows: props.excludeSelfFlows ?? false
        })
      );
    }

    if (usesLexicographicPairs && rowCount > 0) {
      nodes.push(
        ...createFlowLexicographicPairNodes(graph, {
          id: `${id}-lexicographic-pairs`,
          origins: acceptedOriginZones,
          destinations: acceptedDestinationZones,
          pairKeys
        })
      );
    }
    const tableKeys = createTransientView(graph, `${id}-table-keys`, 'uint32', capacity);
    const tableValues = createTransientView(graph, `${id}-table-values`, 'uint32', capacity);
    const buildStatistics = createTransientView(
      graph,
      `${id}-build-statistics`,
      'uint32',
      GPU_HASH_INDEX_STATISTICS_LENGTH
    );
    nodes.push(
      ...new GPUHashIndex({
        id: `${id}-pair-index`,
        keys: pairKeys,
        firstValue: 0,
        tableKeys,
        tableValues,
        statistics: buildStatistics,
        maxProbeCount
      }).getCommandNodes(graph)
    );

    const representativeRows = createTransientView(
      graph,
      `${id}-representative-rows`,
      'uint32',
      rowCount
    );
    if (rowCount > 0) {
      nodes.push(
        ...new GPUHashIndexQuery({
          id: `${id}-pair-query`,
          index: {
            tableKeys,
            tableValues,
            statistics: buildStatistics,
            maxProbeCount
          },
          keys: pairKeys,
          values: representativeRows,
          found: createTransientView(graph, `${id}-query-found`, 'uint32', rowCount),
          probes: createTransientView(graph, `${id}-query-probes`, 'uint32', rowCount),
          statistics: createTransientView(
            graph,
            `${id}-query-statistics`,
            'uint32',
            GPU_HASH_QUERY_STATISTICS_LENGTH
          ),
          maxProbeCount
        }).getCommandNodes(graph)
      );
    }
    const rowCounts = createTransientView(
      graph,
      `${id}-row-counts`,
      'uint32',
      Math.max(rowCount, 1)
    );
    nodes.push(
      ...new GPUGroupAggregation({
        id: `${id}-pair-counts`,
        keys: representativeRows,
        output: rowCounts
      }).getCommandNodes(graph)
    );
    const useSortedSums = this.sumOrder === 'sorted' && rowCount > 0;
    // Finite weights, or 0, one per source row; shared by every requested sum.
    let contributions: GraphDataView<'float32'> | undefined;
    if (weights && rowCount > 0) {
      contributions = createTransientView(graph, `${id}-weight-contributions`, 'float32', rowCount);
      nodes.push(
        ...createFlowContributionNodes(graph, {
          id,
          weights,
          contributions,
          rowBases: createRowBases('weights', getGraphViewChunks(weights))
        })
      );
    }
    let rowWeights: GraphDataView<'float32'> | undefined;
    if (weights) {
      rowWeights = createTransientView(
        graph,
        `${id}-row-weights`,
        'float32',
        Math.max(rowCount, 1)
      );
    }
    // Two stable ascending sorts form an LSD composite key: pair key first, then weight descending.
    const slotKeys = createTransientView(graph, `${id}-slot-keys`, 'uint32', capacity);
    const slotIndices = createTransientView(graph, `${id}-slot-indices`, 'uint32', capacity);
    const sortedPairKeys = createTransientView(graph, `${id}-sorted-pair-keys`, 'uint32', capacity);
    let slotsByPair: GraphDataView<'uint32'>;
    const weightKeys = createTransientView(graph, `${id}-weight-keys`, 'uint32', capacity);
    const sortedWeightKeys = createTransientView(
      graph,
      `${id}-sorted-weight-keys`,
      'uint32',
      capacity
    );
    const rankedSlots = createTransientView(graph, `${id}-ranked-slots`, 'uint32', capacity);
    if (usesLexicographicPairs) {
      const sorted = createFlowLexicographicSlotNodes(graph, {
        id: `${id}-sort-pairs`,
        tableValues,
        origins: acceptedOriginZones,
        destinations: acceptedDestinationZones
      });
      nodes.push(...sorted.nodes);
      slotsByPair = sorted.slots;
    } else {
      slotsByPair = createTransientView(graph, `${id}-slots-by-pair`, 'uint32', capacity);
      nodes.push(
        createFlowSlotKeysNode(graph, {
          id: `${id}-slot-keys`,
          tableKeys,
          slotKeys,
          slotIndices,
          zoneCount
        }),
        ...new GPUSort({
          id: `${id}-sort-pairs`,
          keys: slotKeys,
          values: slotIndices,
          outputKeys: sortedPairKeys,
          outputValues: slotsByPair,
          // Pair keys are below zoneCount^2, which empty slots take (see createFlowSlotKeysNode).
          keyBits: Math.max(1, (zoneCount * zoneCount).toString(2).length)
        }).getCommandNodes(graph)
      );
    }

    if (weights && rowWeights && contributions && useSortedSums) {
      // Stable sort of rows by representative row, offsets from the exact per-pair counts, gather
      // in sorted order, and one fixed-tree segment sum per stored pair.
      const sortKeys = createTransientView(graph, `${id}-pair-sum-sort-keys`, 'uint32', rowCount);
      const sortIndices = createTransientView(
        graph,
        `${id}-pair-sum-sort-indices`,
        'uint32',
        rowCount
      );
      const sortedKeys = createTransientView(
        graph,
        `${id}-pair-sum-sorted-keys`,
        'uint32',
        rowCount
      );
      const sortedIndices = createTransientView(
        graph,
        `${id}-pair-sum-sorted-indices`,
        'uint32',
        rowCount
      );
      const rowOffsets = createTransientView(graph, `${id}-pair-sum-offsets`, 'uint32', rowCount);
      // One workgroup per stored pair, not per table slot (review B-18).
      const pairSumDispatch = createWGSLActiveCountDispatch(graph, {
        id: `${id}-pair-sum-dispatch`,
        operation: 'GPUFlowAggregation',
        count: buildStatistics,
        maximumItemCount: capacity,
        workgroupSize: SEGMENT_WORKGROUP_SIZE,
        invocationsPerItem: SEGMENT_WORKGROUP_SIZE
      });
      const sortedContributions = createTransientView(
        graph,
        `${id}-pair-sum-sorted-contributions`,
        'float32',
        rowCount
      );
      nodes.push(
        createFlowPairSortPrepareNode(graph, {
          id: `${id}-pair-sum-sort-prepare`,
          representativeRows,
          sortKeys,
          sortIndices
        }),
        ...new GPUSort({
          id: `${id}-pair-sum-sort`,
          keys: sortKeys,
          values: sortIndices,
          outputKeys: sortedKeys,
          outputValues: sortedIndices,
          keyBits: Math.max(1, rowCount.toString(2).length)
        }).getCommandNodes(graph),
        ...new GPUScan({
          id: `${id}-pair-sum-scan`,
          input: rowCounts,
          output: rowOffsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        ...new GPUGather({
          id: `${id}-pair-sum-gather`,
          source: contributions,
          indices: sortedIndices,
          output: sortedContributions
        }).getCommandNodes(graph),
        pairSumDispatch.updateNode,
        createFlowPairSumNode(graph, {
          id: `${id}-pair-sum`,
          slotsByPair,
          statistics: buildStatistics,
          condition: pairSumDispatch.condition,
          tableKeys,
          tableValues,
          rowCounts,
          rowOffsets,
          sortedContributions,
          rowWeights
        })
      );
    } else if (weights && rowWeights) {
      nodes.push(
        ...new GPUGroupAggregation({
          id: `${id}-pair-weights`,
          keys: representativeRows,
          values: contributions ?? weights,
          output: rowWeights,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }

    nodes.push(
      createFlowWeightKeysNode(graph, {
        id: `${id}-weight-keys`,
        slotsByPair,
        tableKeys,
        tableValues,
        rowCounts,
        rowWeights,
        weightKeys
      }),
      ...new GPUSort({
        id: `${id}-sort-weights`,
        keys: weightKeys,
        values: slotsByPair,
        outputKeys: sortedWeightKeys,
        outputValues: rankedSlots,
        keyBits: getFlowWeightKeyBits(Boolean(rowWeights), rowCounts.length)
      }).getCommandNodes(graph)
    );

    nodes.push(
      createFlowGatherIdsNode(graph, {
        id: `${id}-gather-ids`,
        statistics: buildStatistics,
        rankedSlots,
        tableKeys,
        ids: output.ids
      })
    );
    if (output.flowCounts || output.flowWeights) {
      nodes.push(
        createFlowGatherValuesNode(graph, {
          id: `${id}-gather-values`,
          capacity: topCount,
          statistics: buildStatistics,
          rankedSlots,
          tableValues,
          rowCounts,
          rowWeights,
          flowCounts: output.flowCounts,
          flowWeights: output.flowWeights
        })
      );
    }
    if (output.flowOriginZoneIds || output.flowDestinationZoneIds) {
      nodes.push(
        createFlowDecodeZonesNode(graph, {
          id: `${id}-decode-zones`,
          zoneCount,
          ids: output.ids,
          sourceOrigins: usesLexicographicPairs ? acceptedOriginZones : undefined,
          sourceDestinations: usesLexicographicPairs ? acceptedDestinationZones : undefined,
          originZoneIds: output.flowOriginZoneIds,
          destinationZoneIds: output.flowDestinationZoneIds
        })
      );
    }

    const pairOverflow =
      output.status.pairOverflow ?? createTransientView(graph, `${id}-pair-overflow`, 'uint32', 1);
    nodes.push(
      createFlowPairOverflowNode(graph, {
        id: `${id}-pair-overflow`,
        statistics: buildStatistics,
        pairOverflow
      }),
      createPublishNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUFlowAggregation',
        requiredCount: graph.createDataView(buildStatistics.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: buildStatistics.byteOffset
        }),
        output,
        candidateOverflowSources: output.status.candidateOverflow ? [pairOverflow] : [],
        extraCounts: output.drawInstanceCount ? [output.drawInstanceCount] : []
      })
    );

    for (const [name, keys, requestedCounts, sums, extent] of [
      [
        'out',
        acceptedOriginZones,
        output.zoneOutCounts,
        output.zoneOutWeights,
        output.zoneOutCountExtent
      ],
      [
        'in',
        acceptedDestinationZones,
        output.zoneInCounts,
        output.zoneInWeights,
        output.zoneInCountExtent
      ]
    ] as const) {
      const counts =
        requestedCounts ??
        (extent
          ? createTransientView(graph, `${id}-zone-${name}-extent-counts`, 'uint32', zoneCount)
          : undefined);
      if (counts) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-zone-${name}-counts`,
            keys,
            output: counts
          }).getCommandNodes(graph)
        );
      }
      if (counts && extent) {
        // Extent over occupied zones only: any nonzero count selects a row.
        const countExtent = createTransientView(graph, `${id}-zone-${name}-extent`, 'uint32', 2);
        nodes.push(
          ...new GPUReduction({
            id: `${id}-zone-${name}-extent`,
            input: counts,
            mask: counts,
            output: countExtent,
            operation: 'extent'
          }).getCommandNodes(graph),
          createFlowExtentToFloatNode(graph, {
            id: `${id}-zone-${name}-extent-float`,
            extent: countExtent,
            output: extent
          })
        );
      }
      if (sums && weights && contributions && useSortedSums) {
        nodes.push(
          ...getSortedSegmentSumNodes<Parameters>(graph, {
            id: `${id}-zone-${name}-sorted`,
            operation: 'GPUFlowAggregation',
            segmentCount: zoneCount,
            segmentKeys: keys,
            sumContributions: contributions,
            sums
          })
        );
      } else if (sums && weights) {
        nodes.push(
          ...new GPUGroupAggregation({
            id: `${id}-zone-${name}-weights`,
            keys,
            values: contributions ?? weights,
            output: sums,
            operation: 'sum'
          }).getCommandNodes(graph)
        );
      }
    }
    return nodes;
  }
}

/** Checks bindings and scratch before allocation for internal flow composition. @internal */
export function validateFlowAggregationBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUFlowAggregation
): void {
  const {id, props, zoneCount, rowCount, sumOrder} = contributor;
  const {weights, output} = props;
  const capacity = props.pairCapacity;
  validateNetworkViewBindings(id, graph, [
    ...getFlowAggregationInputViews(props),
    ...getFlowAggregationOutputViews(props)
  ]);
  for (const view of [
    ...getFlowAggregationInputViews(props),
    ...getFlowAggregationOutputViews(props)
  ]) {
    for (const chunk of view ? getGraphViewChunks(view) : []) {
      validateMaximumCount(
        id,
        'packed view buffer byteLength',
        chunk.buffer.byteLength,
        graph.device.limits.maxBufferSize
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'packed view',
        getViewBindingRange(chunk).size,
        'reduce the packed view length'
      );
    }
  }
  for (const [name, bytes] of [
    ['row scratch', Math.max(rowCount, 1) * 4],
    ['pair scratch', capacity * 4],
    [
      'zone sum offsets',
      (output.zoneOutWeights || output.zoneInWeights) && weights && sumOrder === 'sorted'
        ? (zoneCount + 1) * 4
        : 0
    ],
    [
      'zone extent counts',
      output.zoneOutCountExtent || output.zoneInCountExtent ? zoneCount * 4 : 0
    ],
    ['chunk parameters', 20],
    ['statistics', Math.max(GPU_HASH_INDEX_STATISTICS_LENGTH, GPU_HASH_QUERY_STATISTICS_LENGTH) * 4]
  ] as const) {
    validateMaximumCount(id, `${name} byteLength`, bytes, graph.device.limits.maxBufferSize);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      bytes,
      'reduce row count, pairCapacity or zone count'
    );
  }

  for (const [name, view] of [
    ['origins', props.origins],
    ['destinations', props.destinations],
    ['weights', rowCount > 0 ? weights : undefined]
  ] as const) {
    if (view)
      validateMaximumCount(
        id,
        `${name} chunk parameter buffer byteLength`,
        Math.max(getGraphViewChunks(view).length, 1) * FLOW_CHUNK_PARAMETER_SLOT_BYTE_LENGTH,
        graph.device.limits.maxBufferSize
      );
  }
}

function getFlowAggregationInputViews(
  props: GPUFlowAggregationProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {zones} = props;
  return [
    props.origins,
    props.destinations,
    props.originZoneIds,
    props.destinationZoneIds,
    props.weights,
    props.mask,
    props.timeWindow?.timestamps,
    props.timeWindow?.timestampsLow,
    props.timeWindow?.window,
    zones.kind !== 'ids' && !Array.isArray(zones.bounds)
      ? (zones.bounds as GraphDataView)
      : undefined,
    zones.kind === 'hexagon' && typeof zones.radius !== 'number' ? zones.radius : undefined,
    zones.kind !== 'ids' ? zones.activeGridSize : undefined
  ];
}

function getFlowAggregationOutputViews(
  props: GPUFlowAggregationProps
): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.ids,
    output.status.count,
    output.status.overflow,
    output.status.candidateOverflow,
    output.status.requiredCount,
    output.flowOriginZoneIds,
    output.flowDestinationZoneIds,
    output.flowCounts,
    output.flowWeights,
    output.status.pairOverflow,
    output.zoneOutCounts,
    output.zoneInCounts,
    output.zoneOutWeights,
    output.zoneInWeights,
    output.drawInstanceCount,
    output.zoneOutCountExtent,
    output.zoneInCountExtent
  ];
}

/** Validates literal or GPU-resident zone bounds. */
function validateFlowBounds(id: string, bounds: GPUFlowAggregationBounds): void {
  if (Array.isArray(bounds)) {
    const [minX, minY, maxX, maxY] = bounds as readonly number[];
    if (bounds.length !== 4 || !bounds.every(Number.isFinite) || minX > maxX || minY > maxY) {
      throw new Error(`${id} bounds must be finite [minX, minY, maxX, maxY]`);
    }
    validateRequirement(
      id,
      bounds.every(value => Math.abs(value) <= GPU_FLOW_AGGREGATION_MAXIMUM_LITERAL_MAGNITUDE),
      `bounds literal magnitude must be at most ${GPU_FLOW_AGGREGATION_MAXIMUM_LITERAL_MAGNITUDE} (finite float32)`
    );
    return;
  }
  const view = bounds as GraphDataView;
  validateSinglePackedView(view, ['float32x4', 'float32'], `${id} bounds`);
  if (
    (view.format === 'float32x4' && view.length !== 1) ||
    (view.format === 'float32' && view.length !== 4)
  ) {
    throw new Error(`${id} bounds view must be one float32x4 row or four float32 rows`);
  }
}
