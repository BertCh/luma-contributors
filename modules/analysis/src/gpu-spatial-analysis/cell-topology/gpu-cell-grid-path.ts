// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createCellRowKernelNode} from '../cell-aggregation/cell-row-kernel';
import {snapshotCellConstruction} from '../cell-indexing/cell-construction-snapshot';
import {
  getBoundedDispatchLayout,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {dggs} from '@luma.gl/shaders';

import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import type {GPUCellWordOrder} from '../cell-aggregation/gpu-cell-aggregation';
import type {GPUCellTopologyFamily} from '../cell-indexing/cell-index-families';
import {
  getCellGridPathGridKernel,
  GPU_CELL_GRID_PATH_S2_MAXIMUM_RADIUS
} from './cell-grid-path-wgsl';
import {H3_NEIGHBOR_WGSL} from './h3-neighbor-wgsl';
import {H3_LOCAL_IJ_WGSL} from './h3-local-ij-wgsl';

/** Fixed workgroup size makes chunk lengths runtime dispatch metadata. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Storage component indices of whole fixed-stride records cannot wrap uint32. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;

const OPERATION = 'GPUCellGridPath';

/** Value of `output.distances` for pairs whose grid distance is undefined (H3 `gridDistance` fails). */
export const GPU_CELL_GRID_DISTANCE_UNDEFINED = 0xffffffff;

/** Largest `maximumPathLength` (cells stored per pair). */
export const GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH = 256;

/** Caller-owned outputs of {@link GPUCellGridPath}, all row-aligned with the input pairs. */
export type GPUCellGridPathOutput = {
  /**
   * `uint32` grid distance of each pair (H3 `gridDistance`), or
   * {@link GPU_CELL_GRID_DISTANCE_UNDEFINED} for invalid, masked and unreachable pairs: different
   * resolutions, invalid cells, base cells that are not adjacent, or cells separated by the deleted
   * sector of a pentagon.
   */
  distances?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /**
   * `uint32x2` little-endian `(low, high)` path cells, `rows * maximumPathLength` entries,
   * row-major: the cells of H3 `gridPathCells`, origin first, destination last, zero padded
   * (at an exact tie of an interpolation step the cell can be the other adjacent one; see the
   * class documentation, about 0.2% of random pairs). Paths longer than `maximumPathLength` keep
   * their first `maximumPathLength` cells.
   */
  cells?: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /**
   * `uint32` number of cells of each complete path (`distance + 1`), also when it exceeds
   * `maximumPathLength` (then `counts[row] > maximumPathLength` means the row overflowed and
   * `cells` holds only the first `maximumPathLength` cells). Zero for pairs without a path.
   */
  counts?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

/**
 * Properties for {@link GPUCellGridPath}.
 *
 * Per-frame (no recompile): the contents of `origins`, `destinations` and `mask`. Topology: the
 * view layouts and `maximumPathLength`.
 */
export type GPUCellGridPathProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-grid-path'`. */
  id?: string;
  /** Cell family: H3 local-grid paths, Quadbin Chebyshev paths, or S2 edge paths. */
  family: GPUCellTopologyFamily;
  /** S2 only: optional distance bound, at most 0xfffffffe. Omit for every valid equal-level pair. */
  maximumSearchRadius?: number;
  /** Input `uint32x2` origin cells, one per pair. */
  origins: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Input `uint32x2` destination cells, one per pair, same length as `origins`. */
  destinations: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Word order of `origins`, `destinations` and `output.cells`. Defaults to `'little-endian'`. */
  wordOrder?: GPUCellWordOrder;
  /** Optional per-pair mask; zero produces an undefined distance and an empty path. */
  mask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /**
   * Cells stored per pair when `output.cells` is present, at most
   * {@link GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH}. Required then, ignored otherwise.
   */
  maximumPathLength?: number;
  /** Caller-owned outputs; at least one of `distances` and `cells`. */
  output: GPUCellGridPathOutput;
};

/**
 * Grid distances and paths for H3, Quadbin and S2 cell pairs.
 *
 * Quadbin uses Chebyshev paths; S2 uses edge paths between equal-level cells at levels 0–30.
 * H3 supplies `gridDistance`, `gridPathCells`, and `are_neighbor_cells` as `distance == 1`.
 *
 * For H3, one kernel converts both cells to local IJK coordinates anchored at the origin
 * (`cellToLocalIjk`, including the pentagon rotations) and takes the IJK distance, which equals
 * H3 `gridDistance` for every pair H3 accepts and is undefined exactly where H3 fails (checked
 * against h3-js on pentagon neighborhoods and random pairs at resolutions 1-15). Distances of any
 * length are exact integers.
 *
 * Path cells follow the H3 algorithm: the cube coordinates of the endpoints are interpolated
 * linearly, rounded with `cubeRound` and mapped back to a cell. The cell is found as the unique
 * neighbor of the previous path cell whose local IJK equals the rounded coordinate, which is what
 * H3's `localIjkToCell` returns for every path that stays off the 12 pentagon base cells. Measured
 * against h3-js on random and pentagon-centered pairs of resolutions 3-15, two deviations remain:
 *
 * - Cube rounding is evaluated in exact integer arithmetic, H3 in doubles, so at the exact `x.5`
 *   ties of an interpolation step H3's result depends on rounding noise. About 0.2% of random
 *   pairs contain such a tie where H3's cell is the other adjacent one; the path stays a valid
 *   chain of the same length with the same endpoints.
 * - Where a path crosses a pentagon base cell the rounded coordinate may match no neighbor
 *   uniquely. Such pairs (about 6% of the paths that touch a pentagon base cell, under 1% of random
 *   pairs) report `counts = 0` and an empty path row, keeping their distance; about 1% of the
 *   others differ from H3 by one or two cells.
 *
 * Output rows are bounded by `maximumPathLength`; `counts` reports the true length so overflow is
 * `counts[row] > maximumPathLength`. Everything is integer arithmetic, deterministic on every device.
 *
 * Quadbin uses wrapped-column Chebyshev distance (matching GPUCellTopology): each path step
 * advances both axes until each reaches its destination. S2 unfolds the integer cube grid for an
 * exact global edge distance, then takes neighbors that reduce the remaining distance by one.
 * An explicit maximumSearchRadius bounds accepted distances; by default every equal-level pair
 * is accepted. These routes use integer keys through
 * Quadbin resolution 26 and S2 level 30. S2 ties visit left, right, down, up in face coordinates.
 *
 * Consumes cell keys only, so it takes no `spatialContext`.
 *
 * Accepts packed or chunked row inputs; output chunks mirror inputs and keep whole fixed-stride records.
 */
export class GPUCellGridPath implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellGridPathProps;
  /** Cells stored per pair (0 without `output.cells`). */
  readonly maximumPathLength: number;

  /** Upload plus one dispatch per nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly recordChunks: Readonly<Record<string, readonly GraphDataView[]>>;

  constructor(props: GPUCellGridPathProps) {
    props = snapshotCellConstruction(props, [
      'id',
      'family',
      'maximumSearchRadius',
      'origins',
      'destinations',
      'wordOrder',
      'mask',
      'maximumPathLength',
      ['output', ['cells', 'distances', 'counts']]
    ]);
    this.id = props.id ?? 'cell-grid-path';
    this.props = props;
    const id = this.id;
    validateRequirement(
      id,
      ['h3', 'quadbin', 's2'].includes(props.family),
      "family must be 'h3', 'quadbin' or 's2'"
    );
    validateCapacity(
      id,
      'maximumSearchRadius',
      props.maximumSearchRadius ?? GPU_CELL_GRID_PATH_S2_MAXIMUM_RADIUS,
      0,
      GPU_CELL_GRID_PATH_S2_MAXIMUM_RADIUS
    );
    validateRequirement(
      id,
      props.wordOrder === undefined || ['little-endian', 'high-low'].includes(props.wordOrder),
      "wordOrder must be 'little-endian' or 'high-low'"
    );
    const {output} = props;
    validateRequirement(
      id,
      Boolean(output.distances || output.cells),
      'output needs distances or cells'
    );
    validateRequirement(
      id,
      !output.counts || Boolean(output.cells),
      'output.counts needs output.cells'
    );
    this.maximumPathLength = output.cells ? (props.maximumPathLength as number) : 0;
    if (output.cells)
      validateCapacity(
        id,
        'maximumPathLength',
        this.maximumPathLength,
        1,
        GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH
      );
    this.chunkPlan = planGraphChunkDispatch(id, {
      origins: props.origins,
      destinations: props.destinations,
      mask: props.mask,
      distances: output.distances,
      counts: output.counts
    });
    validateCapacity(id, 'origins row count', this.chunkPlan.totalLength);
    for (const chunk of this.chunkPlan.chunks) {
      if (chunk.views['origins'])
        validateSinglePackedView(
          chunk.views['origins'],
          ['uint32x2'],
          `${id} origins chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'origins chunk components',
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['destinations'])
        validateSinglePackedView(
          chunk.views['destinations'],
          ['uint32x2'],
          `${id} destinations chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'destinations chunk components',
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['mask'])
        validateSinglePackedView(
          chunk.views['mask'],
          ['uint32'],
          `${id} mask chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'mask chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['distances'])
        validateSinglePackedView(
          chunk.views['distances'],
          ['uint32'],
          `${id} distances chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'distances chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['counts'])
        validateSinglePackedView(
          chunk.views['counts'],
          ['uint32'],
          `${id} counts chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'counts chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
    }
    const recordChunks: Record<string, readonly GraphDataView[]> = {};
    recordChunks['pathCells'] = output.cells ? getGraphViewChunks(output.cells) : [];
    if (output.cells) {
      validateRequirement(
        id,
        output.cells.length === this.chunkPlan.totalLength * this.maximumPathLength,
        'output.cells length must equal input rows * record stride'
      );
      validateRequirement(
        id,
        recordChunks['pathCells'].length === this.chunkPlan.chunks.length,
        'output.cells must match input chunk topology'
      );
      for (const chunk of this.chunkPlan.chunks) {
        const view = recordChunks['pathCells'][chunk.chunkIndex];
        validateSinglePackedView(view, ['uint32x2'], `${id} pathCells chunk ${chunk.chunkIndex}`);
        validateRequirement(
          id,
          view.length === chunk.length * this.maximumPathLength,
          `output.cells chunk ${chunk.chunkIndex} length must equal input chunk rows * record stride`
        );
        validateMaximumCount(
          id,
          'pathCells chunk components',
          view.length * 2,
          MAXIMUM_CHUNK_COMPONENT_COUNT
        );
      }
    }
    this.recordChunks = recordChunks;
    validateDisjointWritableViews(id, props.output);
    this.estimatedCommandNodeCount =
      (this.chunkPlan.chunks.length > 0 ? 1 : 0) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.distances, output.cells, output.counts],
      [props.origins, props.destinations, props.mask]
    );
  }

  /** Uploads chunk parameters and dispatches each nonempty pair chunk. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumPathLength} = this;
    const {output} = props;
    validateMaximumCount(
      id,
      'storage bindings',
      3 + Number(Boolean(props.mask)) + Object.values(props.output).filter(Boolean).length,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {'output.cells': output.cells});
    const invocationsPerRow = 1;
    for (const chunk of this.chunkPlan.chunks) {
      getBoundedDispatchLayout(
        id,
        Math.max(chunk.length, 1),
        CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      );
    }
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    const chunkNodes = this.chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const bindings: WGSLKernelBinding[] = [
          {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
          {name: 'origins', view: chunk.views['origins'], type: 'u32', access: 'read'},
          {name: 'destinations', view: chunk.views['destinations'], type: 'u32', access: 'read'},
          ...(props.mask
            ? [{name: 'rowMask', view: chunk.views['mask'], type: 'u32', access: 'read'} as const]
            : []),
          ...(output.distances
            ? [
                {
                  name: 'outDistances',
                  view: chunk.views['distances'],
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.cells
            ? [
                {
                  name: 'outCells',
                  view: this.recordChunks['pathCells'][chunk.chunkIndex],
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : []),
          ...(output.counts
            ? [
                {
                  name: 'outCounts',
                  view: chunk.views['counts'],
                  type: 'u32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ];
        const isHighLow = props.wordOrder === 'high-low';
        const readCell = (
          name: string
        ) => `let ${name}Words = vec2u(${name}s[${name}sOffset + 2u * index], ${name}s[${name}sOffset + 2u * index + 1u]);
  let ${name} = ${isHighLow ? `${name}Words` : `${name}Words.yx`};`;
        const writeCell = (slot: string, cell: string) => `
    outCells[outCellsOffset + 2u * (${slot})] = ${cell}.${isHighLow ? 'x' : 'y'};
    outCells[outCellsOffset + 2u * (${slot}) + 1u] = ${cell}.${isHighLow ? 'y' : 'x'};`;

        const pathBody = output.cells
          ? `
  var pathLength = 0u;
  var isPathValid = false;
  if (pathDistance != GRID_DISTANCE_UNDEFINED) {
    isPathValid = true;
    let distanceValue = i32(pathDistance);
    let startCube = cellTopologyH3GetCube(originLocal.ijk);
    let delta = cellTopologyH3GetCube(destinationLocal.ijk) - startCube;
    // floor((delta * step) / distance), advanced one step at a time without overflow.
    var quotientStep = vec3i(0);
    var remainderStep = vec3i(0);
    if (distanceValue > 0) {
      quotientStep = delta / vec3i(distanceValue);
      remainderStep = delta - quotientStep * distanceValue;
      for (var axis = 0; axis < 3; axis++) {
        if (remainderStep[axis] < 0) {
          remainderStep[axis] += distanceValue;
          quotientStep[axis] -= 1;
        }
      }
    }
    cellGridPath[0] = origin;
    pathLength = 1u;
    var quotient = vec3i(0);
    var remainder = vec3i(0);
    var previous = origin;
    let storedLength = min(pathDistance + 1u, PATH_STRIDE);
    for (var pathStep = 1u; pathStep < storedLength; pathStep++) {
      quotient += quotientStep;
      remainder += remainderStep;
      for (var axis = 0; axis < 3; axis++) {
        if (remainder[axis] >= distanceValue) {
          remainder[axis] -= distanceValue;
          quotient[axis] += 1;
        }
      }
      let roundedTarget = cellTopologyH3RoundCubeStep(startCube, quotient, remainder, distanceValue);
      var matchCount = 0u;
      var matchedNeighbor = vec2u(0u);
      for (var direction = 1u; direction <= 6u; direction++) {
        let neighbor = cellTopologyH3Neighbor(previous, direction);
        if (neighbor.x == 0u && neighbor.y == 0u) {
          continue;
        }
        let neighborLocal = cellTopologyH3GetLocalIjk(origin, neighbor);
        if (
          neighborLocal.valid != 0u &&
          neighborLocal.ijk.x - neighborLocal.ijk.z == roundedTarget.x &&
          neighborLocal.ijk.y - neighborLocal.ijk.z == roundedTarget.y
        ) {
          matchCount++;
          matchedNeighbor = neighbor;
        }
      }
      if (matchCount != 1u) {
        isPathValid = false;
        break;
      }
      cellGridPath[pathStep] = matchedNeighbor;
      previous = matchedNeighbor;
      pathLength = pathStep + 1u;
    }
  }
  for (var slot = 0u; slot < PATH_STRIDE; slot++) {
    let row = index * PATH_STRIDE + slot;
    if (isPathValid && slot < pathLength) {${writeCell('row', 'cellGridPath[slot]')}
    } else {${writeCell('row', 'vec2u(0u)')}
    }
  }
  ${output.counts ? 'outCounts[outCountsOffset + index] = select(0u, pathDistance + 1u, isPathValid);' : ''}`
          : '';

        const regularGrid =
          props.family === 'h3'
            ? undefined
            : getCellGridPathGridKernel(props.family, {
                readCell,
                writeCell,
                maximumPathLength,
                maximumSearchRadius:
                  props.maximumSearchRadius ?? GPU_CELL_GRID_PATH_S2_MAXIMUM_RADIUS,
                hasMask: Boolean(props.mask),
                distances: Boolean(output.distances),
                cells: Boolean(output.cells),
                counts: Boolean(output.counts)
              });
        const kernelProps: WGSLKernelProps<Parameters> = {
          id: `${id}-path`,
          operation: OPERATION,
          variant: `${props.family}-${output.cells ? 'path' : 'distance'}`,
          bindings,
          invocationCount: chunk.length * invocationsPerRow,
          declarations:
            regularGrid?.declarations ??
            `const GRID_DISTANCE_UNDEFINED: u32 = 0xffffffffu;
const PATH_STRIDE: u32 = ${Math.max(maximumPathLength, 1)}u;
var<private> cellGridPath: array<vec2u, PATH_STRIDE>;
${dggs.source}
${CELL_KEY_WGSL}
${H3_NEIGHBOR_WGSL}
${H3_LOCAL_IJ_WGSL}`,
          body:
            regularGrid?.body ??
            `${readCell('origin')}
  ${readCell('destination')}
  var isValid = true;
  ${props.mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  var pathDistance = GRID_DISTANCE_UNDEFINED;
  var originLocal = CellTopologyH3LocalIjk(vec3i(0), 0u);
  var destinationLocal = CellTopologyH3LocalIjk(vec3i(0), 0u);
  if (isValid) {
    originLocal = cellTopologyH3GetLocalIjk(origin, origin);
    destinationLocal = cellTopologyH3GetLocalIjk(origin, destination);
    if (originLocal.valid != 0u && destinationLocal.valid != 0u) {
      pathDistance = cellTopologyH3GetGridDistance(originLocal.ijk, destinationLocal.ijk);
    }
  }
  ${output.distances ? 'outDistances[outDistancesOffset + index] = pathDistance;' : ''}
  ${pathBody}`
        };
        const components: Record<string, number> = {
          origins: 2,
          destinations: 2,
          rowMask: 1,
          outDistances: 1,
          outCells: maximumPathLength * 2,
          outCounts: 1
        };
        return createCellRowKernelNode(graph, {
          id: `${kernelProps.id}-${chunk.chunkIndex}`,
          kernelId: kernelProps.operation,
          operation: kernelProps.operation,
          variant: kernelProps.variant!,
          length: chunk.length,
          parameters: slots[chunk.chunkIndex],
          declarations: kernelProps.declarations,
          body: kernelProps.body,
          bindings: kernelProps.bindings
            .filter(binding => binding.name !== 'chunkParameters')
            .map(binding => ({...binding, components: components[binding.name]}))
        });
      });
    return [...nodes, ...chunkNodes];
  }
}
