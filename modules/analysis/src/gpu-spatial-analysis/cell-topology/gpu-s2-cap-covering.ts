// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {validateS2CoveringOptions} from './s2-covering-option-validation';
import {
  GPUScan,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {
  GPU_S2_COVERING_MAXIMUM_LEVEL,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
} from '../global-coordinates/spherical-limits';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateAngularTolerance} from '../global-coordinates/spherical-limits';
import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {
  getScanCommandNodeCount,
  createS2CoveringRoundParameters,
  validateS2CoveringRoundParameters
} from './s2-covering-support';
import {
  validateCellDispatchBindings,
  validateCellDispatchOutputs
} from '../cell-indexing/cell-dispatch-validation';

/** Storage scalar indices, lengths and serialized count literals must fit uint32. */
const MAXIMUM_S2_COVERING_COMPONENT_COUNT = 0xffffffff;

/**
 * Largest `level` of the uniform enumeration: each invocation examines all `6 * 4^level` cells of
 * the sphere, 393,216 at level 8, which is where one thread per cap stops finishing within the
 * GPU watchdog. Level 8 cells are about 40 km across.
 */
export const GPU_S2_CAP_COVERING_MAXIMUM_UNIFORM_LEVEL = 8;

/**
 * Largest `level` of the level-synchronous refinement. Cells at level 15 are a few hundred meters
 * across, near the resolution of f32 unit vectors and the 1e-5 radian (64 m) default tolerance.
 */
export const GPU_S2_CAP_COVERING_MAXIMUM_REFINED_LEVEL = GPU_S2_COVERING_MAXIMUM_LEVEL;

/** Smallest `tolerance` (radians): below it f32 cell-to-cap distances are rounding noise. */
export const GPU_S2_CAP_COVERING_MINIMUM_TOLERANCE = 1e-6;

/** Largest `tolerance` (radians), about 64 km: a larger margin changes the cover's meaning. */
export const GPU_S2_CAP_COVERING_MAXIMUM_TOLERANCE = 1e-2;

/**
 * Slack (radians) added to the pruning test of a parent cell. A child's distance to the cap is at
 * least its parent's, but each f32 distance carries rounding, so pruning without slack could drop
 * a parent whose child is selected at the target level.
 */
const REFINEMENT_PRUNING_MARGIN = 1e-6;

/** Fewest frontier cells per cap: the six cube-face cells it starts from. */
const MINIMUM_FRONTIER_CAPACITY = 6;

/**
 * Default frontier cells per capacity cell: four children of each kept parent, times two for a
 * cover that overflows `capacity` by a factor of about two.
 */
const DEFAULT_FRONTIER_CAPACITY_FACTOR = 8;
/** Default frontier cells added for the coarse levels, which hold at least the six face cells. */
const DEFAULT_FRONTIER_CAPACITY_MARGIN = 64;

/** Status of one cap: complete. */
export const GPU_S2_CAP_COVERING_STATUS_COMPLETE = 0;
/** Status bit: invalid cap (non-unit or non-finite center, radius outside `[0, pi]`). */
export const GPU_S2_CAP_COVERING_STATUS_INVALID = 1;
/** Status bit: more cells than `capacity`; `counts[1]` holds the required count. */
export const GPU_S2_CAP_COVERING_STATUS_CAPACITY = 2;
/**
 * Status bit, level-synchronous only: a refinement round needed more than `frontierCapacity`
 * cells for this cap. The cover is incomplete, no cells are written and `counts` is `(0, 0)`:
 * the required cell count is not known (unlike {@link GPU_S2_CAP_COVERING_STATUS_CAPACITY}).
 */
export const GPU_S2_CAP_COVERING_STATUS_FRONTIER = 4;

/**
 * How cells are enumerated.
 * - `'uniform'`: each cap's invocation tests every cell at `level` (level at most
 *   {@link GPU_S2_CAP_COVERING_MAXIMUM_UNIFORM_LEVEL}). The reference tier.
 * - `'level-synchronous'`: one invocation per frontier cell per level, pruning cells that cannot
 *   contain a selected descendant. The cells are the same, in the same ascending ID order (also for
 *   a zero-radius cap on a face center, see `s2AnalysisDistance`), and
 *   `level` may reach {@link GPU_S2_CAP_COVERING_MAXIMUM_REFINED_LEVEL}.
 */
export type GPUS2CapCoveringRefinement = 'uniform' | 'level-synchronous';

export type GPUS2CapCoveringProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  /** Unit center xyz, cap radius radians [0, pi]. */
  caps: GraphDataView<'float32x4'>;
  /**
   * Uniform covering level. `'uniform'`: 0 to 8, each invocation examines 6 * 4^level cells.
   * `'level-synchronous'`: 0 to 15.
   */
  level: number;
  mode?: 'exterior' | 'interior';
  capacity: number;
  /** Exterior expands / interior shrinks by this angular safety margin, default 1e-5. */
  tolerance?: number;
  /**
   * Enumeration. Defaults to `'uniform'`, the reference tier whose outputs are pinned by recorded
   * fixtures.
   */
  refinement?: GPUS2CapCoveringRefinement;
  /**
   * `'level-synchronous'` only: cells held per cap in each refinement round, at least `capacity`
   * and 6. A round holds four children of every kept parent, and a kept parent can have no selected
   * child (it only lies within the pruning margin), so the frontier can reach about four times the
   * final cell count, not the final count itself. Default `8 * capacity + 64`, which holds a cover
   * that overflows `capacity` by a factor of about two, so the caller still gets status 2 and the
   * required count to resize with. A cap whose frontier outgrows it reports the frontier status
   * bit, writes no cells and reports counts `(0, 0)`: a truncated frontier cannot know how many
   * cells the full cover has, so no required count is available; raise `frontierCapacity` (or
   * `capacity`) and run again.
   */
  frontierCapacity?: number;
  output: {
    cells: GraphDataView<'uint32x2'>;
    /** (written, required) per cap. */
    counts: GraphDataView<'uint32x2'>;
    /**
     * Per cap, a bit set: {@link GPU_S2_CAP_COVERING_STATUS_COMPLETE} (0),
     * {@link GPU_S2_CAP_COVERING_STATUS_INVALID} (1), {@link GPU_S2_CAP_COVERING_STATUS_CAPACITY} (2),
     * {@link GPU_S2_CAP_COVERING_STATUS_FRONTIER} (4).
     */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Bounded uniform-level spherical-cap covering. The `'uniform'` enumeration is the reference tier
 * for levels up to 8; `refinement: 'level-synchronous'` refines from the six face cells and is
 * parallel over frontier cells. No host readback or hidden refinement scheduler.
 */
export class GPUS2CapCovering implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  /** Cells held per cap in each refinement round (level-synchronous only). */
  readonly frontierCapacity: number;
  /** Exact number of command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;
  readonly props: GPUS2CapCoveringProps;
  constructor(props: GPUS2CapCoveringProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'caps',
      'level',
      'mode',
      'capacity',
      'tolerance',
      'refinement',
      'frontierCapacity',
      'output'
    ]);
    props = this.props = Object.freeze({
      ...captured,
      output: snapshotConstructionFields(captured.output, ['cells', 'counts', 'status']),
      ...(captured.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)}
        : {})
    });
    this.id = props.id ?? 's2-cap-covering';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    validateSinglePackedView(props.caps, ['float32x4'], this.id);
    validateSinglePackedView(props.output.cells, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.counts, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.status, ['uint32'], this.id);
    validateCapacity(
      this.id,
      'output.cells components',
      props.output.cells.length * 2,
      0,
      MAXIMUM_S2_COVERING_COMPONENT_COUNT
    );
    validateCapacity(
      this.id,
      'caps components',
      props.caps.length * 4,
      0,
      MAXIMUM_S2_COVERING_COMPONENT_COUNT
    );
    validateS2CoveringOptions(this.id, props, ['uniform', 'level-synchronous']);
    const refined = props.refinement === 'level-synchronous';
    // Uniform enumeration is intentionally bounded; the refined form is an adaptive frontier.
    validateCapacity(
      this.id,
      'level',
      props.level,
      0,
      refined
        ? GPU_S2_CAP_COVERING_MAXIMUM_REFINED_LEVEL
        : GPU_S2_CAP_COVERING_MAXIMUM_UNIFORM_LEVEL
    );
    validateCapacity(
      this.id,
      'capacity',
      props.capacity,
      1,
      Math.min(MAXIMUM_S2_COVERING_COMPONENT_COUNT, 6 * 4 ** props.level)
    );
    validateRequirement(
      this.id,
      props.output.cells.length === props.caps.length * props.capacity &&
        props.output.counts.length === props.caps.length &&
        props.output.status.length === props.caps.length,
      'output.cells must hold caps times capacity entries, and output.counts and output.status one entry per cap'
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-5,
      GPU_S2_CAP_COVERING_MINIMUM_TOLERANCE,
      GPU_S2_CAP_COVERING_MAXIMUM_TOLERANCE
    );
    this.frontierCapacity =
      props.frontierCapacity ??
      DEFAULT_FRONTIER_CAPACITY_FACTOR * props.capacity + DEFAULT_FRONTIER_CAPACITY_MARGIN;
    if (refined) {
      validateCapacity(
        this.id,
        'frontierCapacity',
        this.frontierCapacity,
        Math.max(props.capacity, MINIMUM_FRONTIER_CAPACITY),
        0xffffffff
      );
      // The frontier lists are uint32x2 rows in one binding of the default 128 MiB limit.
      validateMaximumCount(
        this.id,
        'caps times frontierCapacity',
        props.caps.length * this.frontierCapacity,
        2 ** 24
      );
    } else {
      validateRequirement(
        this.id,
        props.frontierCapacity === undefined,
        "frontierCapacity needs refinement 'level-synchronous'"
      );
    }
    // Start, cells and publish nodes, then per round a classify, the scan over every frontier row
    // (plus the total row) and a write.
    this.estimatedCommandNodeCount = refined
      ? 4 +
        (props.level + 1) *
          (2 + getScanCommandNodeCount(props.caps.length * this.frontierCapacity + 1))
      : 1;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateCellDispatchOutputs(
      this.id,
      props.output,
      refined ? [{counts: props.output.counts, status: props.output.status}] : [props.output]
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.caps]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const refined = props.refinement === 'level-synchronous';
    const rows = props.caps.length * this.frontierCapacity;
    validateCellDispatchBindings(
      graph,
      id,
      {caps: props.caps, ...props.output},
      refined
        ? [
            ['frontier scratch', rows * 8],
            ['count and offset scratch', (rows + 1) * 4],
            ['list count scratch', props.caps.length * 4]
          ]
        : []
    );
    if (refined) validateS2CoveringRoundParameters(graph, id, props.level + 1);
    validateGraphViewsBelongToGraph(id, graph, [props.caps, ...Object.values(props.output)]);
    if (props.refinement === 'level-synchronous') {
      return this.getLevelSynchronousNodes(graph);
    }
    const toleranceLiteral = getWGSLFloatLiteral(props.tolerance ?? 1e-5);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUS2CapCovering',
        variant: props.mode ?? 'exterior',
        invocationCount: props.caps.length,
        bindings: [
          {name: 'caps', view: props.caps, type: 'f32', access: 'read'},
          {name: 'cells', view: props.output.cells, type: 'u32', access: 'read_write'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${dggs.source}\n${S2_ANALYSIS_WGSL}`,
        body: `let offset = capsOffset + index * 4u; let center = vec3f(caps[offset], caps[offset + 1u], caps[offset + 2u]); let radius = caps[offset + 3u];
var failure = 0u; var count = 0u;
if ((bitcast<u32>(dot(center, center)) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(radius) & 0x7f800000u) == 0x7f800000u || abs(dot(center, center) - 1.0) > 1e-4 || radius < 0.0 || radius > 3.1415928) { failure = 1u; }
if (failure == 0u) {
  for (var candidate = 0u; candidate < ${6 * 4 ** props.level}u; candidate++) {
    let face = candidate / ${4 ** props.level}u; let position = candidate % ${4 ** props.level}u;
    let cell = dggs_u64_shift_left(vec2u(0u, face), 61u) | dggs_u64_shift_left(vec2u(0u, position), ${61 - 2 * props.level}u) | dggs_u64_shift_left(vec2u(0u, 1u), ${60 - 2 * props.level}u);
    let selected = ${props.mode === 'interior' ? `3.141592653589793 - s2AnalysisDistance(cell, -center) <= radius - ${toleranceLiteral}` : `s2AnalysisDistance(cell, center) <= radius + ${toleranceLiteral}`};
    if (selected) {
      if (count < ${props.capacity}u) { let output = cellsOffset + (index * ${props.capacity}u + count) * 2u; cells[output] = cell.y; cells[output + 1u] = cell.x; }
      count += 1u;
    }
  }
}
for (var slot = min(count, ${props.capacity}u); slot < ${props.capacity}u; slot++) { let output = cellsOffset + (index * ${props.capacity}u + slot) * 2u; cells[output] = 0u; cells[output + 1u] = 0u; }
if (count > ${props.capacity}u) { failure |= 2u; }
counts[countsOffset + index * 2u] = min(count, ${props.capacity}u); counts[countsOffset + index * 2u + 1u] = count; status[statusOffset + index] = failure;`
      })
    ];
  }

  /**
   * Level-synchronous refinement. Every cap owns `frontierCapacity` consecutive rows of one shared
   * list. Round `r` classifies the level `r` cells of every cap (drop 0, split 4, and at the
   * target level keep 1), one global exclusive scan numbers the survivors, and the per-cap base
   * offset (the scan value at the cap's first row) turns the global numbering into a per-cap one.
   * Children are written in order, so each cap's cells stay in ascending ID order without a sort.
   */
  private getLevelSynchronousNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, frontierCapacity} = this;
    const operation = 'GPUS2CapCovering';
    const variant = `${props.mode ?? 'exterior'}-level-synchronous`;
    const capCount = props.caps.length;
    const rowCount = capCount * frontierCapacity;
    const roundCount = props.level + 1;
    const toleranceLiteral = getWGSLFloatLiteral(props.tolerance ?? 1e-5);
    const pruningMarginLiteral = getWGSLFloatLiteral(REFINEMENT_PRUNING_MARGIN);
    const roundParameters = createS2CoveringRoundParameters(graph, id, roundCount);
    const lists = [
      createTransientView(graph, `${id}-frontier-a`, 'uint32x2', rowCount),
      createTransientView(graph, `${id}-frontier-b`, 'uint32x2', rowCount)
    ];
    // One extra row: the scan total, which is the last cap's end offset.
    const counts = createTransientView(graph, `${id}-round-counts`, 'uint32', rowCount + 1);
    const offsets = createTransientView(graph, `${id}-round-offsets`, 'uint32', rowCount + 1);
    const listCounts = createTransientView(graph, `${id}-list-counts`, 'uint32', capCount);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...roundParameters.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-start`,
        operation,
        variant,
        invocationCount: capCount,
        bindings: [
          {name: 'caps', view: props.caps, type: 'f32', access: 'read'},
          {name: 'list', view: lists[0]!, type: 'u32', access: 'read_write'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
          {name: 'listCounts', view: listCounts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `let offset = capsOffset + index * 4u; let center = vec3f(caps[offset], caps[offset + 1u], caps[offset + 2u]); let radius = caps[offset + 3u];
var failure = 0u;
if ((bitcast<u32>(dot(center, center)) & 0x7f800000u) == 0x7f800000u || (bitcast<u32>(radius) & 0x7f800000u) == 0x7f800000u || abs(dot(center, center) - 1.0) > 1e-4 || radius < 0.0 || radius > 3.1415928) { failure = 1u; }
status[statusOffset + index] = failure; listCounts[listCountsOffset + index] = select(6u, 0u, failure != 0u);
if (failure == 0u) {
  for (var face = 0u; face < 6u; face++) {
    let row = (index * ${frontierCapacity}u + face) * 2u;
    list[listOffset + row] = 0u; list[listOffset + row + 1u] = (2u * face + 1u) << 28u;
  }
}
if (index == 0u) { counts[countsOffset + ${rowCount}u] = 0u; }`
      })
    ];
    for (let round = 0; round < roundCount; round++) {
      const input = lists[round % 2]!;
      const output = lists[(round + 1) % 2]!;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-classify-${round}`,
          operation,
          variant,
          invocationCount: rowCount,
          bindings: [
            {
              name: 'roundParameters',
              view: roundParameters.slots[round],
              type: 'u32',
              access: 'read'
            },
            {name: 'caps', view: props.caps, type: 'f32', access: 'read'},
            {name: 'list', view: input, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
            {name: 'listCounts', view: listCounts, type: 'u32', access: 'read'}
          ],
          declarations: `${dggs.source}\n${S2_ANALYSIS_WGSL}`,
          body: `let capIndex = index / ${frontierCapacity}u; let slot = index % ${frontierCapacity}u;
var count = 0u;
if (slot < min(listCounts[listCountsOffset + capIndex], ${frontierCapacity}u)) {
  let offset = capsOffset + capIndex * 4u; let center = vec3f(caps[offset], caps[offset + 1u], caps[offset + 2u]); let radius = caps[offset + 3u];
  ${props.mode === 'interior' ? `if (radius <= ${toleranceLiteral}) { counts[countsOffset + index] = 0u; return; }` : ''}
  let cell = vec2u(list[listOffset + index * 2u + 1u], list[listOffset + index * 2u]);
  let nearest = s2AnalysisDistance(cell, center);
  if (roundParameters[roundParametersOffset] == ${props.level}u) {
    let selected = ${props.mode === 'interior' ? `3.141592653589793 - s2AnalysisDistance(cell, -center) <= radius - ${toleranceLiteral}` : `nearest <= radius + ${toleranceLiteral}`};
    count = select(0u, 1u, selected);
  } else {
    // A child is no closer to the cap than its parent, so the parent bounds every descendant.
    let kept = nearest <= radius ${props.mode === 'interior' ? '-' : '+'} ${toleranceLiteral} + ${pruningMarginLiteral};
    count = select(0u, 4u, kept);
  }
}
counts[countsOffset + index] = count;`
        }),
        ...new GPUScan({
          id: `${id}-scan-${round}`,
          input: counts,
          output: offsets,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode(graph, {
          id: `${id}-write-${round}`,
          operation,
          variant,
          invocationCount: rowCount,
          bindings: [
            {name: 'source', view: input, type: 'u32', access: 'read'},
            {name: 'destination', view: output, type: 'u32', access: 'read_write'},
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'listCounts', view: listCounts, type: 'u32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations: dggs.source,
          body: `let capIndex = index / ${frontierCapacity}u; let slot = index % ${frontierCapacity}u;
let base = offsets[offsetsOffset + capIndex * ${frontierCapacity}u];
let count = counts[countsOffset + index]; let start = offsets[offsetsOffset + index] - base;
if (slot == 0u) {
  // The next list length is unclamped so a frontier overflow stays visible.
  let total = offsets[offsetsOffset + (capIndex + 1u) * ${frontierCapacity}u] - base;
  listCounts[listCountsOffset + capIndex] = total;
  if (total > ${frontierCapacity}u) { atomicOr(&status[statusOffset + capIndex], 4u); }
}
if (count != 0u) {
  let cell = vec2u(source[sourceOffset + index * 2u + 1u], source[sourceOffset + index * 2u]);
  let row = capIndex * ${frontierCapacity}u + start;
  if (count == 1u) {
    if (start < ${frontierCapacity}u) { destination[destinationOffset + row * 2u] = cell.y; destination[destinationOffset + row * 2u + 1u] = cell.x; }
  } else {
    let level = dggs_s2_get_level(cell);
    let marker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - level)); let childMarker = dggs_u64_shift_right(marker, 2u);
    var child = dggs_u64_add(dggs_u64_subtract(cell, marker), childMarker); let step = dggs_u64_shift_left(childMarker, 1u);
    for (var childSlot = 0u; childSlot < 4u; childSlot++) {
      if (start + childSlot < ${frontierCapacity}u) { destination[destinationOffset + (row + childSlot) * 2u] = child.y; destination[destinationOffset + (row + childSlot) * 2u + 1u] = child.x; }
      child = dggs_u64_add(child, step);
    }
  }
}`
        })
      );
    }
    const cover = lists[roundCount % 2]!;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-cells`,
        operation,
        variant,
        invocationCount: capCount * props.capacity,
        bindings: [
          {name: 'list', view: cover, type: 'u32', access: 'read'},
          {name: 'listCounts', view: listCounts, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
          {name: 'cells', view: props.output.cells, type: 'u32', access: 'read_write'}
        ],
        // Slots past the cover, and every slot of an invalid or incomplete cap, are zero.
        body: `let capIndex = index / ${props.capacity}u; let slot = index % ${props.capacity}u;
var low = 0u; var high = 0u;
if ((status[statusOffset + capIndex] & 5u) == 0u && slot < min(listCounts[listCountsOffset + capIndex], ${frontierCapacity}u)) {
  let row = (capIndex * ${frontierCapacity}u + slot) * 2u; low = list[listOffset + row]; high = list[listOffset + row + 1u];
}
cells[cellsOffset + index * 2u] = low; cells[cellsOffset + index * 2u + 1u] = high;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation,
        variant,
        invocationCount: capCount,
        bindings: [
          {name: 'listCounts', view: listCounts, type: 'u32', access: 'read'},
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `var failure = status[statusOffset + index]; var written = 0u; var required = 0u;
if ((failure & 5u) == 0u) {
  required = listCounts[listCountsOffset + index]; written = min(required, ${props.capacity}u);
  if (required > ${props.capacity}u) { failure |= 2u; }
}
counts[countsOffset + index * 2u] = written; counts[countsOffset + index * 2u + 1u] = required; status[statusOffset + index] = failure;`
      })
    );
    return nodes;
  }
}
