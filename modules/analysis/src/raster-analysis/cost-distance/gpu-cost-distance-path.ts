// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from '../raster-grid/raster-storage-preflight';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  createPublishNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterGrid} from '../raster-band-utils';
import {validateRasterWritableDispatches} from '../raster-grid/raster-view-validation';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createContributorTransientView} from './raster-grid-utils';
import {GPU_COST_DISTANCE_NONE, GPU_COST_DISTANCE_UNRESOLVED} from './gpu-cost-distance';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

const OPERATION = 'GPUCostDistancePath';

/** Maximum whole-grid cell count using the single-thread reference path extractor. */
export const MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS = 256;

/** One-row status words of {@link GPUCostDistancePath}. */
export type GPUCostDistancePathStatus = {
  /**
   * One row: 1 when the walk met a `GPU_COST_DISTANCE_UNRESOLVED` back-link, that is,
   * {@link GPUCostDistance} ran out of `maximumIterations` or `maximumTieIterations` before this
   * path was certified; the path is then empty. Otherwise 0.
   */
  iterationLimitReached?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUCostDistancePath}. */
export type GPUCostDistancePathOutput = {
  /**
   * Bounded compact output. `ids` holds cell indices from the target (first) to the source
   * cell (last); capacity is `ids.length` (compile-time). `count` is clamped, `overflow` is 1 when
   * the path is longer than the capacity, `requiredCount` is the exact path length. The path is empty
   * (count 0) when the target is out of range or unreached, when the walk meets a
   * `GPU_COST_DISTANCE_UNRESOLVED` link (see `status.iterationLimitReached`), and when the links are
   * not a valid {@link GPUCostDistance} forest (an invalid code, a link off the grid or into an
   * unreached cell, or a cycle). A path is only ever reported complete, from the target to a
   * source.
   */
  path: GPUCompactOutput;
  /** Optional one-row status words. */
  status?: GPUCostDistancePathStatus;
};

/**
 * Properties for {@link GPUCostDistancePath}.
 *
 * Compile-time: `grid` size, the capacity of `output.path.ids`, and the presence of
 * `output.status.iterationLimitReached`. Per-frame: the contents of `backLinks` and `target`.
 */
export type GPUCostDistancePathProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'cost-distance-path'`. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /** Back-link raster from {@link GPUCostDistance}, one value per cell. Per-frame contents. */
  backLinks: GraphDataView<'uint32'>;
  /** One-row target cell index (`row * width + column`). Per-frame contents. */
  target: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUCostDistancePathOutput;
};

/**
 * Extracts the least-cost path from a target cell back to its source by following the back-links
 * written by {@link GPUCostDistance}.
 *
 * Grids with at most MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS = 256 total cells use a
 * single-thread reference walk. Larger grids use parallel pointer doubling and ordered gathering;
 * six scratch uint32 planes need `24 * cellCount` bytes total, each in a `4 * cellCount` byte
 * storage binding checked against device limits before allocation. The exact length is always reported.
 * A walk that cannot reach a source (an unresolved link from a truncated
 * `GPUCostDistance` run, or links that are not a valid forest) reports an empty path instead of
 * a truncated one. Non-goals: multiple targets per encoding and path simplification.
 */
export class GPUCostDistancePath implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCostDistancePathProps;

  constructor(inputProps: GPUCostDistancePathProps) {
    const captured = snapshotConstructionFields(inputProps, [
      'id',
      'grid',
      'backLinks',
      'target',
      'output'
    ]);
    const output = snapshotConstructionFields(captured.output, ['path', 'status']);
    const capturedPath = snapshotConstructionFields(output.path, ['ids', 'status']);
    const pathStatus = snapshotConstructionFields(capturedPath.status, [
      'count',
      'overflow',
      'requiredCount',
      'candidateOverflow'
    ]);
    const grid = snapshotConstructionFields(captured.grid, [
      'width',
      'height',
      'affine',
      'pixelInterpretation',
      'coordinateReferenceSystem',
      'level',
      'levelZeroOrigin'
    ]);
    this.props = Object.freeze({
      ...captured,
      grid: Object.freeze({
        ...grid,
        affine: Object.freeze([...grid.affine]) as GPURasterMetadata['affine']
      }),
      output: Object.freeze({
        ...output,
        path: Object.freeze({...capturedPath, status: pathStatus}),
        ...(output.status
          ? {status: snapshotConstructionFields(output.status, ['iterationLimitReached'])}
          : {})
      })
    });
    const props = this.props;
    const {path} = props.output;
    this.id = props.id ?? 'cost-distance-path';
    const {id} = this;
    const cellCount = validateRasterGrid(id, props.grid);
    const iterationLimitReached = props.output.status?.iterationLimitReached;
    for (const [name, view] of [
      ['backLinks', props.backLinks],
      ['target', props.target],
      ['output.path.ids', path.ids],
      ['output.path.count', path.status.count],
      ['output.path.overflow', path.status.overflow],
      ['output.path.requiredCount', path.status.requiredCount],
      ['output.path.status.candidateOverflow', path.status.candidateOverflow],
      ['output.status.iterationLimitReached', iterationLimitReached]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    if (props.backLinks.length !== cellCount) {
      throw new Error(`${id} backLinks must contain one value per cell`);
    }
    for (const [name, view] of [
      ['target', props.target],
      ['output.path.count', path.status.count],
      ['output.path.overflow', path.status.overflow],
      ['output.path.requiredCount', path.status.requiredCount],
      ['output.path.status.candidateOverflow', path.status.candidateOverflow],
      ['output.status.iterationLimitReached', iterationLimitReached]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    if (path.ids.length < 1) {
      throw new Error(`${id} output.path.ids must have a capacity of at least one`);
    }
    const publishOutputs = [
      path.status.count,
      path.status.overflow,
      path.status.requiredCount,
      path.status.candidateOverflow
    ].filter(view => view !== undefined);
    const walkOutputs = [path.ids, iterationLimitReached].filter(view => view !== undefined);
    validateRasterWritableDispatches(
      id,
      [props.backLinks, props.target],
      [path.ids, ...publishOutputs, ...(iterationLimitReached ? [iterationLimitReached] : [])],
      [
        {inputs: [props.backLinks, props.target], outputs: walkOutputs},
        // The publisher reads a fresh internal total, never the caller's walk inputs or IDs.
        {inputs: [], outputs: publishOutputs}
      ]
    );
  }

  /** Returns reference-walk or parallel traversal nodes, followed by compact status publication. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        storageProps.backLinks,
        storageProps.target,
        ...getRasterPreflightOutputViews(storageProps.output.path),
        ...Object.values(storageProps.output.status ?? {})
      ],
      {
        'path total': 4,
        'path jump planes':
          storageProps.backLinks.length > MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS
            ? storageProps.backLinks.length * 4
            : 0
      }
    );

    const {id, props} = this;
    const {output} = props;
    const {path} = output;
    const {width, height} = props.grid;
    const iterationLimitReached = output.status?.iterationLimitReached;
    validateGraphViewsBelongToGraph(id, graph, [
      props.backLinks,
      props.target,
      path.ids,
      path.status.count,
      path.status.overflow,
      path.status.requiredCount,
      path.status.candidateOverflow,
      iterationLimitReached
    ]);
    const cellCount = width * height;
    const total = createContributorTransientView(graph, OPERATION, id, `${id}-total`, 'uint32', 1);
    if (cellCount > MAXIMUM_COST_DISTANCE_PATH_REFERENCE_CELLS) {
      return this.getParallelNodes(graph, total);
    }
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-walk`,
        operation: OPERATION,
        variant: 'walk',
        bindings: [
          {name: 'backLinks', view: props.backLinks, type: 'u32', access: 'read'},
          {name: 'targetCells', view: props.target, type: 'u32', access: 'read'},
          {name: 'ids', view: path.ids, type: 'u32', access: 'read_write'},
          {name: 'total', view: total, type: 'u32', access: 'read_write'},
          ...(iterationLimitReached
            ? [
                {
                  name: 'iterationLimitReached',
                  view: iterationLimitReached,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: `const GRID_WIDTH: u32 = ${width}u;
const GRID_HEIGHT: u32 = ${height}u;
const CELL_COUNT: u32 = ${cellCount}u;
const CAPACITY: u32 = ${path.ids.length}u;
const NONE: u32 = ${GPU_COST_DISTANCE_NONE}u;
const UNRESOLVED: u32 = ${GPU_COST_DISTANCE_UNRESOLVED}u;

fn getColumnOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  return offsets[direction];
}
fn getRowOffset(direction: u32) -> i32 {
  var offsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  return offsets[direction];
}
fn getNeighbor(cell: u32, direction: u32) -> u32 {
  let column = i32(cell % GRID_WIDTH) + getColumnOffset(direction);
  let row = i32(cell / GRID_WIDTH) + getRowOffset(direction);
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) { return NONE; }
  return u32(row) * GRID_WIDTH + u32(column);
}`,
        body: `let targetCell = targetCells[targetCellsOffset];
  var pathLength = 0u;
  var unresolved = false;
  if (targetCell < CELL_COUNT && backLinks[backLinksOffset + targetCell] != NONE) {
    var cell = targetCell;
    // Only a walk that ends at a source (code 0) is a path; every other exit reports none.
    var complete = false;
    for (var walkIndex = 0u; walkIndex < CELL_COUNT; walkIndex++) {
      let code = backLinks[backLinksOffset + cell];
      if (code == UNRESOLVED) { unresolved = true; break; }
      if (pathLength < CAPACITY) { ids[idsOffset + pathLength] = cell; }
      pathLength++;
      if (code == 0u) { complete = true; break; }
      if (code > 128u || countOneBits(code) != 1u) { break; }
      let next = getNeighbor(cell, firstTrailingBit(code));
      if (next == NONE || backLinks[backLinksOffset + next] == NONE) { break; }
      cell = next;
    }
    if (!complete) { pathLength = 0u; }
  }
  total[totalOffset] = pathLength;
  ${iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, unresolved);' : ''}`
      }),
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output: path
      })
    ];
  }

  /** Parallel list ranking certifies the chain while doubling its ordered scratch prefix. */
  private getParallelNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    total: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {path} = props.output;
    const cellCount = props.grid.width * props.grid.height;
    const roundCount = Math.ceil(Math.log2(cellCount));
    const createPlane = (name: string) =>
      createContributorTransientView(graph, OPERATION, id, `${id}-${name}`, 'uint32', cellCount);
    let successors = createPlane('successors-first');
    let lengths = createPlane('lengths-first');
    let states = createPlane('states-first');
    let nextSuccessors = createPlane('successors-second');
    let nextLengths = createPlane('lengths-second');
    let nextStates = createPlane('states-second');
    // State 0 is pending (or a cycle after the last round), 1 source, 2 invalid, 3 unresolved.
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'parallel-initialize',
        invocationCount: cellCount,
        bindings: [
          {name: 'backLinks', view: props.backLinks, type: 'u32', access: 'read'},
          {name: 'targetCells', view: props.target, type: 'u32', access: 'read'},
          {name: 'successors', view: successors, type: 'u32', access: 'read_write'},
          {name: 'lengths', view: lengths, type: 'u32', access: 'read_write'},
          {name: 'states', view: states, type: 'u32', access: 'read_write'},
          {name: 'ids', view: path.ids, type: 'u32', access: 'read_write'}
        ],
        declarations: `const WIDTH: u32 = ${props.grid.width}u;
const HEIGHT: u32 = ${props.grid.height}u;
const NONE: u32 = ${GPU_COST_DISTANCE_NONE}u;
const UNRESOLVED: u32 = ${GPU_COST_DISTANCE_UNRESOLVED}u;`,
        body: `let code = backLinks[backLinksOffset + index];
  var successor = index;
  var length = 0u;
  var state = 2u;
  if (code == 0u) { length = 1u; state = 1u; }
  else if (code == UNRESOLVED) { state = 3u; }
  else if (code <= 128u && countOneBits(code) == 1u) {
    let direction = firstTrailingBit(code);
    var columns = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
    var rows = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
    let column = i32(index % WIDTH) + columns[direction];
    let row = i32(index / WIDTH) + rows[direction];
    if (column >= 0 && row >= 0 && column < i32(WIDTH) && row < i32(HEIGHT)) {
      let neighbor = u32(row) * WIDTH + u32(column);
      if (backLinks[backLinksOffset + neighbor] != NONE) {
        successor = neighbor; length = 1u; state = 0u;
      }
    }
  }
  successors[successorsOffset + index] = successor;
  lengths[lengthsOffset + index] = length;
  states[statesOffset + index] = state;
  if (index == 0u) {
    let targetCell = targetCells[targetCellsOffset];
    ids[idsOffset] = select(0u, targetCell, targetCell < ${cellCount}u);
  }`
      })
    ];
    for (let round = 0; round < roundCount; round++) {
      const stride = 2 ** round;
      const gatherCount = Math.min(stride, path.ids.length - stride, cellCount - stride);
      if (gatherCount > 0) {
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-gather-${round}`,
            operation: OPERATION,
            variant: 'parallel-gather',
            invocationCount: gatherCount,
            bindings: [
              {name: 'successors', view: successors, type: 'u32', access: 'read'},
              {name: 'ids', view: path.ids, type: 'u32', access: 'read_write'}
            ],
            // The existing prefix is [0, stride); every write is in [stride, 2 * stride).
            // Each jump follows stride edges, so the concatenation keeps target-to-source order.
            // Terminated chains may repeat their terminal cell outside the final readable count.
            body: `let cell = ids[idsOffset + index];
  ids[idsOffset + ${stride}u + index] = successors[successorsOffset + cell];`
          })
        );
      }
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-jump-${round}`,
          operation: OPERATION,
          variant: 'parallel-jump',
          invocationCount: cellCount,
          bindings: [
            {name: 'successors', view: successors, type: 'u32', access: 'read'},
            {name: 'lengths', view: lengths, type: 'u32', access: 'read'},
            {name: 'states', view: states, type: 'u32', access: 'read'},
            {name: 'nextSuccessors', view: nextSuccessors, type: 'u32', access: 'read_write'},
            {name: 'nextLengths', view: nextLengths, type: 'u32', access: 'read_write'},
            {name: 'nextStates', view: nextStates, type: 'u32', access: 'read_write'}
          ],
          body: `var successor = successors[successorsOffset + index];
  var length = lengths[lengthsOffset + index];
  var state = states[statesOffset + index];
  if (state == 0u) {
    let next = successor;
    successor = successors[successorsOffset + next];
    length += lengths[lengthsOffset + next];
    state = states[statesOffset + next];
  }
  nextSuccessors[nextSuccessorsOffset + index] = successor;
  nextLengths[nextLengthsOffset + index] = length;
  nextStates[nextStatesOffset + index] = state;`
        })
      );
      [successors, nextSuccessors] = [nextSuccessors, successors];
      [lengths, nextLengths] = [nextLengths, lengths];
      [states, nextStates] = [nextStates, states];
    }
    const iterationLimitReached = props.output.status?.iterationLimitReached;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-certify`,
        operation: OPERATION,
        variant: 'parallel-certify',
        invocationCount: 1,
        bindings: [
          {name: 'lengths', view: lengths, type: 'u32', access: 'read'},
          {name: 'states', view: states, type: 'u32', access: 'read'},
          {name: 'targetCells', view: props.target, type: 'u32', access: 'read'},
          {name: 'total', view: total, type: 'u32', access: 'read_write'},
          ...(iterationLimitReached
            ? [
                {
                  name: 'iterationLimitReached',
                  view: iterationLimitReached,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `let targetCell = targetCells[targetCellsOffset];
  var length = 0u;
  var state = 2u;
  if (targetCell < ${cellCount}u) {
    state = states[statesOffset + targetCell];
    if (state == 1u) { length = lengths[lengthsOffset + targetCell]; }
  }
  total[totalOffset] = length;
  ${iterationLimitReached ? 'iterationLimitReached[iterationLimitReachedOffset] = u32(state == 3u);' : ''}`
      }),
      createPublishNode(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output: path
      })
    );
    return nodes;
  }
}
