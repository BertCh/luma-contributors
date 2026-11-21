// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateS2CoveringOptions} from './s2-covering-option-validation';
import {validateUnitVector} from '../global-coordinates/assert-unit-vector';
import {
  GPU_S2_COVERING_MAXIMUM_LEVEL,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from '../global-coordinates/spherical-limits';
import {
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
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createS2CoveringParallelNodes,
  getS2CoveringParallelNodeCount
} from './s2-covering-parallel';
export {MAXIMUM_S2_COVERING_INPUT_EDGES_PER_INVOCATION} from './s2-covering-parallel-wgsl';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateCellDispatchOutputs} from '../cell-indexing/cell-dispatch-validation';

/** Storage scalar indices, lengths and serialized count literals must fit uint32. */
const MAXIMUM_S2_COVERING_COMPONENT_COUNT = 0xffffffff;

/**
 * Largest `maximumLevel`. Level 15 cells are a few hundred meters across, already near the
 * resolution of f32 unit vectors and of the largest accepted `tolerance` (1e-3 radians is 6 km,
 * the default 1e-6 is 6 m); deeper levels classify against rounding noise.
 */
export const GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL = GPU_S2_COVERING_MAXIMUM_LEVEL;

/** Cube faces at level zero, the roots of every S2 covering traversal. */
const S2_ROOT_FACE_COUNT = 6;

/** Fewest visited cells a traversal can have: the six cube-face cells it starts from. */
export const GPU_S2_POLYGON_COVERING_MINIMUM_VISITED_CELLS = S2_ROOT_FACE_COUNT;

/**
 * Largest `maximumVisitedCells`, on either route. Each configured level holds at most this
 * many pending cells; its cell binding is at most 8 MiB. All tree fields together cost 40 bytes
 * per budget cell per level (640 MiB at level 15 and the maximum budget), before input scratch.
 * The whole-round live frontier on the level-synchronous route also includes retained cells.
 */
export const GPU_S2_POLYGON_COVERING_MAXIMUM_VISITED_CELLS = 1048576;

/** Default `maximumVisitedCells`: 64 Ki cells per level, 512 KiB per candidate-cell binding. */
export const GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS = 65536;

/** Status bit: invalid input (a malformed or non-simple ring, a non-unit vertex, an empty hemisphere). */
export const GPU_S2_POLYGON_COVERING_STATUS_INVALID = 1;
/** Status bit: more cells than `capacity`; `counts[1]` holds the required count. */
export const GPU_S2_POLYGON_COVERING_STATUS_CAPACITY = 2;
/**
 * Status bit: the traversal needed more than `maximumVisitedCells` cells. The cover is incomplete:
 * the serial traversal keeps the cells emitted so far and `counts[1]` is not a complete
 * requirement; the level-synchronous traversal writes no cells and reports `counts` of `(0, 0)`.
 */
export const GPU_S2_POLYGON_COVERING_STATUS_BUDGET = 4;

/**
 * How the quad-tree is traversed.
 * - `'serial'`: parallel predicates and subtree sizes recover the depth-first order. Exhaustion
 *   of `maximumVisitedCells` keeps exactly that traversal's emitted prefix.
 * - `'level-synchronous'`: each of the `maximumLevel + 1` rounds classifies a complete frontier,
 *   using bounded ring/tile jobs, then scans and expands its children. A failed round suppresses
 *   the whole result. Completed covers and visits equal DFS, in ascending cell-ID order.
 */
export type GPUS2PolygonCoveringRefinement = 'serial' | 'level-synchronous';

export type GPUS2PolygonCoveringProps = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  /** One simple spherical polygon: shell then holes, either orientation, unit xyz minor arcs. */
  positions: GraphDataView<'float32x3'>;
  ringOffsets: GraphDataView<'uint32'>;
  complement?: boolean;
  minimumLevel?: number;
  maximumLevel: number;
  mode?: 'exterior' | 'interior';
  /** Maximum output S2 cells, across all levels, in either traversal route. */
  capacity: number;
  /**
   * Maximum visited hierarchy cells. Exhaustion is explicit; default
   * {@link GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS}. In the level-synchronous traversal it
   * is also the frontier capacity.
   */
  maximumVisitedCells?: number;
  tolerance?: number;
  /** Optional common hemisphere pole. Omitted: each ring uses its normalized vertex mean. */
  hemisphereCenter?: readonly [number, number, number];
  /**
   * Traversal. Defaults to `'serial'`, whose outputs, including partial results on exhaustion, are
   * pinned by recorded fixtures. Both routes use parallel input preparation and predicates.
   */
  refinement?: GPUS2PolygonCoveringRefinement;
  output: {
    cells: GraphDataView<'uint32x2'>;
    /** (written, required so far), not a complete requirement when status has bit 4. */
    counts: GraphDataView<'uint32x2'>;
    /**
     * One word, a bit set: {@link GPU_S2_POLYGON_COVERING_STATUS_INVALID} (1),
     * {@link GPU_S2_POLYGON_COVERING_STATUS_CAPACITY} (2),
     * {@link GPU_S2_POLYGON_COVERING_STATUS_BUDGET} (4); 0 is complete.
     */
    status: GraphDataView<'uint32'>;
    visitedCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Adaptive mixed-level S2 interior/exterior cover of one polygon, in a bounded GPU traversal.
 * Both routes prepare rings and classify cells in parallel. One invocation processes at most
 * 4,096 polygon edges or edge pairs, independently of ring size; longer rings use integer winding
 * partials. Range searches take at most 32 bisections per invocation. Orientation and center
 * reductions process one vertex per invocation per stage.
 * Traversal is limited separately to level 15 and 1,048,576 visited cells (default 65,536).
 * The serial route publishes the original ordered depth-first prefix using subtree sizes; it
 * does not switch to breadth-first exhaustion semantics. Level-synchronous failure suppresses
 * the whole cover. Pending candidates are bounded by maximumVisitedCells at each level.
 *
 * Scratch includes per-vertex center/area reductions, ring metadata, a candidate tree of 40 bytes
 * per budget cell per level, and batched integer winding scratch (at most 128 MiB per binding).
 * Large inputs increase batch/node counts and remain subject to the shared command-node and
 * device storage budgets. Every configured level dispatches its bounded candidate list.
 */
export class GPUS2PolygonCovering implements GPUCommandNodeProducer {
  readonly id: string;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  /** Exact number of command nodes `getCommandNodes` returns. */
  readonly estimatedCommandNodeCount: number;
  readonly props: GPUS2PolygonCoveringProps;
  constructor(props: GPUS2PolygonCoveringProps) {
    const captured = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'positions',
      'ringOffsets',
      'complement',
      'minimumLevel',
      'maximumLevel',
      'mode',
      'capacity',
      'maximumVisitedCells',
      'tolerance',
      'hemisphereCenter',
      'refinement',
      'output'
    ]);
    props = this.props = Object.freeze({
      ...captured,
      output: snapshotConstructionFields(captured.output, [
        'cells',
        'counts',
        'status',
        'visitedCount'
      ]),
      ...(captured.hemisphereCenter
        ? {
            hemisphereCenter: Object.freeze([...captured.hemisphereCenter]) as readonly [
              number,
              number,
              number
            ]
          }
        : {}),
      ...(captured.spatialContext
        ? {
            spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)
          }
        : {})
    });
    this.id = props.id ?? 's2-polygon-covering';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    if (props.hemisphereCenter)
      validateUnitVector(props.hemisphereCenter, this.id, 'hemisphereCenter');
    validateSinglePackedView(props.positions, ['float32x3'], this.id);
    validateSinglePackedView(props.ringOffsets, ['uint32'], this.id);
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
      'positions components',
      props.positions.length * 3,
      0,
      MAXIMUM_S2_COVERING_COMPONENT_COUNT
    );
    validateCapacity(
      this.id,
      'ringOffsets.length',
      props.ringOffsets.length,
      2,
      MAXIMUM_S2_COVERING_COMPONENT_COUNT
    );
    if (props.output.visitedCount) {
      validateSinglePackedView(props.output.visitedCount, ['uint32'], this.id);
      validateRequirement(
        this.id,
        props.output.visitedCount.length === 1,
        'output.visitedCount must hold one entry'
      );
    }
    validateCapacity(
      this.id,
      'maximumLevel',
      props.maximumLevel,
      0,
      GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL
    );
    validateCapacity(this.id, 'minimumLevel', props.minimumLevel ?? 0, 0, props.maximumLevel);
    validateCapacity(this.id, 'capacity', props.capacity);
    validateRequirement(
      this.id,
      props.output.cells.length === props.capacity &&
        props.output.counts.length === 1 &&
        props.output.status.length === 1,
      'output.cells must hold capacity entries, and output.counts and output.status one entry'
    );
    validateCapacity(
      this.id,
      'maximumVisitedCells',
      props.maximumVisitedCells ?? GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS,
      GPU_S2_POLYGON_COVERING_MINIMUM_VISITED_CELLS,
      GPU_S2_POLYGON_COVERING_MAXIMUM_VISITED_CELLS
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE
    );
    validateS2CoveringOptions(this.id, props, ['serial', 'level-synchronous']);
    validateRequirement(
      this.id,
      props.ringOffsets.length >= 2,
      'ringOffsets must hold at least 2 entries (one ring)'
    );
    this.estimatedCommandNodeCount = getS2CoveringParallelNodeCount(props);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateCellDispatchOutputs(
      this.id,
      props.output,
      props.refinement === 'level-synchronous'
        ? [
            {
              counts: props.output.counts,
              status: props.output.status,
              visitedCount: props.output.visitedCount
            }
          ]
        : [props.output]
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateGraphViewsBelongToGraph(this.id, graph, [
      this.props.positions,
      this.props.ringOffsets,
      ...Object.values(this.props.output)
    ]);
    return createS2CoveringParallelNodes(graph, this.props, this.id);
  }
}
