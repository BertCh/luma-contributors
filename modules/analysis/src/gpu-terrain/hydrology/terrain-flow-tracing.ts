// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createContributorTransientView} from '../../raster-analysis/cost-distance/raster-grid-utils';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createRasterIterationFinalizeNode,
  createRasterGatedLoopNodes,
  createRasterIterationResetNode,
  createRasterIterationState,
  type RasterIterationState
} from '../../raster-analysis/cost-distance/raster-relaxation';

/**
 * How a caller-supplied D8 direction grid numbers its eight directions. Compile-time.
 *
 * **The codes are raster offsets, not compass bearings.** In the tables below "south" means the
 * next row (`row + 1`), "north" the previous row and "east" the next column, whatever the grid
 * affine says: no hydrology class reads the row sign `grid.affine[4]`. On a north-up raster (rows
 * advance south, `grid.affine[4] < 0`), which is how ArcGIS, WhiteboxTools and TauDEM write their
 * grids, the names are true compass directions. `GPUTerrainFlow` writes its `flowDirections` by
 * the same rule, so its output round-trips on any grid. If you hold a tool's direction grid with
 * its rows flipped (rows advancing north, `grid.affine[4] > 0`), its codes still name the tool's
 * compass directions and would be followed mirrored north to south with no error: flip the rows
 * back, or exchange the north and south codes, before passing it in.
 *
 * - `'esri'` (default): powers of two clockwise from east: 1 east, 2 south-east, 4 south, 8
 *   south-west, 16 west, 32 north-west, 64 north, 128 north-east. The encoding of
 *   `GPUTerrainFlow.flowDirections`, of ArcGIS and pysheds, and of WhiteboxTools with `--esri_pntr`.
 * - `'whitebox'`: the WhiteboxTools default pointer, powers of two clockwise from north-east: 1
 *   north-east, 2 east, 4 south-east, 8 south, 16 south-west, 32 west, 64 north-west, 128 north.
 * - `'taudem'`: the TauDEM D8 grid, 1 to 8 counter-clockwise from east: 1 east, 2 north-east, 3
 *   north, 4 north-west, 5 west, 6 south-west, 7 south, 8 south-east.
 *
 * In every encoding 0 is a terminal cell, `0xffffffff` an invalid cell and `0xfffffffe` an
 * unresolved direction, so map the tool's own no-data value to `0xffffffff` when you import the
 * grid as `uint32`. A word outside the encoding has no receiver, as an invalid ESRI code has none.
 */
export type GPUTerrainFlowDirectionEncoding = 'esri' | 'whitebox' | 'taudem';

const TERRAIN_FLOW_DIRECTION_ENCODINGS: readonly GPUTerrainFlowDirectionEncoding[] = [
  'esri',
  'whitebox',
  'taudem'
];

/** Throws unless `directionEncoding` is undefined or one of the three encodings. @internal */
export function validateTerrainFlowDirectionEncoding(
  id: string,
  directionEncoding: GPUTerrainFlowDirectionEncoding | undefined
): void {
  validateRequirement(
    id,
    directionEncoding === undefined || TERRAIN_FLOW_DIRECTION_ENCODINGS.includes(directionEncoding),
    `directionEncoding must be 'esri', 'whitebox' or 'taudem' (got ${String(directionEncoding)})`
  );
}

/**
 * WGSL `getFlowDirectionCode(word)`: the ESRI code of one word of a caller-supplied direction
 * grid. The single decoder of every terrain contributor that takes `directionEncoding`. Zero and
 * the two sentinels pass through; a word outside the encoding becomes `INVALID_DIRECTION_CODE`
 * (nonzero, no receiver), which every consumer treats as it treats a multi-bit ESRI code.
 */
function getFlowDirectionCodeWGSL(directionEncoding: GPUTerrainFlowDirectionEncoding): string {
  switch (directionEncoding) {
    case 'esri':
      return 'fn getFlowDirectionCode(word: u32) -> u32 { return word; }';
    case 'whitebox':
      // One step clockwise ahead of ESRI: north-east is bit 0, so rotate the eight bits right by one.
      return `fn getFlowDirectionCode(word: u32) -> u32 {
  if (word == 0u || word >= UNRESOLVED_DIRECTION) { return word; }
  if (word > 128u || (word & (word - 1u)) != 0u) { return INVALID_DIRECTION_CODE; }
  return select(word >> 1u, 128u, word == 1u);
}`;
    case 'taudem':
      // Counter-clockwise from east, so direction index (east first, clockwise) = (9 - word) mod 8.
      return `fn getFlowDirectionCode(word: u32) -> u32 {
  if (word == 0u || word >= UNRESOLVED_DIRECTION) { return word; }
  if (word > 8u) { return INVALID_DIRECTION_CODE; }
  return 1u << ((9u - word) % 8u);
}`;
  }
}

/**
 * WGSL constants and helpers shared by the D8 tracing kernels: `GRID_WIDTH`, `GRID_HEIGHT`,
 * `NO_CELL`, `UNRESOLVED_DIRECTION`, `getFlowDirectionCode(word)` and `getFlowReceiver(cell, code)`.
 *
 * `getFlowReceiver` takes an ESRI code. A code has a receiver only when it is a single bit in
 * `1..128` (ESRI D8: `1 << direction`, east first, clockwise) and the neighbor lies inside the
 * grid. Terminal (0), invalid (`0xffffffff`), unresolved (`0xfffffffe`), multi-bit, and
 * out-of-grid codes have no receiver; a path that stops at an unresolved cell is itself
 * unresolved, which the callers report. A kernel that reads a caller-supplied direction grid
 * passes each word through `getFlowDirectionCode` first, which turns `directionEncoding` into
 * ESRI codes (the identity for `'esri'`). The helpers do not read the grid `settings`, so kernels
 * that use them need no settings binding.
 *
 * @internal
 */
export function getTerrainFlowTracingWGSL(
  width: number,
  height: number,
  directionEncoding: GPUTerrainFlowDirectionEncoding = 'esri'
): string {
  return /* wgsl */ `
const GRID_WIDTH: u32 = ${width}u;
const GRID_HEIGHT: u32 = ${height}u;
const NO_CELL: u32 = 0xffffffffu;
// GPU_TERRAIN_FLOW_UNRESOLVED: a valid cell whose direction is unknown. It has no receiver but is
// not a terminal cell, so consumers must not read it as an outlet or a sink.
const UNRESOLVED_DIRECTION: u32 = 0xfffffffeu;
// A direction word outside the contributor's encoding: nonzero and without a receiver.
const INVALID_DIRECTION_CODE: u32 = 0x100u;

${getFlowDirectionCodeWGSL(directionEncoding)}

fn getFlowReceiver(cell: u32, code: u32) -> u32 {
  if (code == 0u || code > 128u || (code & (code - 1u)) != 0u) { return NO_CELL; }
  var columnOffsets = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
  var rowOffsets = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);
  let direction = firstTrailingBit(code);
  let column = i32(cell % GRID_WIDTH) + columnOffsets[direction];
  let row = i32(cell / GRID_WIDTH) + rowOffsets[direction];
  if (column < 0 || row < 0 || column >= i32(GRID_WIDTH) || row >= i32(GRID_HEIGHT)) {
    return NO_CELL;
  }
  return u32(row) * GRID_WIDTH + u32(column);
}`;
}

/**
 * Which cells stop pointer jumping.
 *
 * - `'none'`: only cells without a receiver stop, so the fixpoint is the terminal cell.
 * - `'streams'`: valid cells (direction code other than `0xffffffff`) with a non-zero `streams` word.
 * - `'markers'`: cells whose `markers` word is not `0xffffffff`.
 *
 * @internal
 */
export type TerrainFlowStopRule =
  | {kind: 'none'}
  | {kind: 'streams'; streams: GraphDataView<'uint32'>}
  | {kind: 'markers'; markers: GraphDataView<'uint32'>};

/** Returns the WGSL `isStopCell(cell, code)` function and its bindings for a stop rule. @internal */
function getStopRuleSource(stops: TerrainFlowStopRule): {
  binding?: WGSLKernelBinding;
  declarations: string;
} {
  switch (stops.kind) {
    case 'none':
      return {declarations: 'fn isStopCell(cell: u32, code: u32) -> bool { return false; }'};
    case 'streams':
      return {
        binding: {name: 'stops', view: stops.streams, type: 'u32', access: 'read'},
        declarations: `fn isStopCell(cell: u32, code: u32) -> bool {
  return code != NO_CELL && stops[stopsOffset + cell] != 0u;
}`
      };
    case 'markers':
      return {
        binding: {name: 'stops', view: stops.markers, type: 'u32', access: 'read'},
        declarations: `fn isStopCell(cell: u32, code: u32) -> bool {
  return stops[stopsOffset + cell] != NO_CELL;
}`
      };
  }
}

// No upstream list-ranking or tree-accumulation primitive exists (gpu-graph has none), so pointer jumping stays local.
/**
 * Returns the pointer-jumping nodes: initialization, loop reset, `maxIterations` gated doubling
 * rounds, and the optional converged-flag publication.
 *
 * `pointer[c]` starts as `c` for stop cells and cells without a receiver, otherwise as the D8
 * receiver. Each round sets `pointer[c] = pointer[pointer[c]]`. With stableRounds, every read
 * comes from a separate snapshot copied before that round; otherwise updates are in place.
 * A read of a partially
 * updated pointer is still a cell on the same downstream path, so on acyclic paths the fixpoint (the first stop cell
 * or the terminal cell downstream) is unique and independent of the schedule. Path lengths up to
 * `2^k` need `k` doubling rounds plus one round that confirms nothing changed. Receiver cycles
 * can collapse to self-pointers, so consumers must verify original receivers and stop rules
 * before treating pointer stability as drainage convergence.
 *
 * @internal
 */
export function createTerrainFlowPointerNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    contributorId: string;
    operation: string;
    width: number;
    height: number;
    maxIterations: number;
    /** Each round reads a separate snapshot, for deterministic explicit-budget watersheds. */
    stableRounds?: boolean;
    flowDirections: GraphDataView<'uint32'>;
    /** Encoding of `flowDirections`. Defaults to `'esri'`. */
    directionEncoding?: GPUTerrainFlowDirectionEncoding;
    pointer: GraphDataView<'uint32'>;
    stops: TerrainFlowStopRule;
    converged?: GraphDataView<'uint32'>;
    iterationLimitReached?: GraphDataView<'uint32'>;
    iterationCount?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const cellCount = props.width * props.height;
  const state = createRasterIterationState(
    graph,
    `${props.id}-pointer`,
    props.operation,
    cellCount,
    props.contributorId
  );
  const stopRule = getStopRuleSource(props.stops);
  const previousPointer = props.stableRounds
    ? createContributorTransientView(
        graph,
        props.operation,
        props.contributorId,
        `${props.id}-previous-pointer`,
        'uint32',
        cellCount
      )
    : undefined;
  const tracing = getTerrainFlowTracingWGSL(props.width, props.height, props.directionEncoding);
  const initBindings: WGSLKernelBinding[] = [
    {name: 'flowDirections', view: props.flowDirections, type: 'u32', access: 'read'},
    ...(stopRule.binding ? [stopRule.binding] : []),
    {name: 'pointer', view: props.pointer, type: 'u32', access: 'read_write'}
  ];
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-pointer-init`,
      operation: props.operation,
      variant: 'pointer-init',
      bindings: initBindings,
      invocationCount: cellCount,
      declarations: `${tracing}\n${stopRule.declarations}`,
      body: `let code = getFlowDirectionCode(flowDirections[flowDirectionsOffset + index]);
  var jumpTarget = index;
  if (!isStopCell(index, code)) {
    let receiver = getFlowReceiver(index, code);
    if (receiver != NO_CELL) { jumpTarget = receiver; }
  }
  pointer[pointerOffset + index] = jumpTarget;`
    }),
    createRasterIterationResetNode<Parameters>(graph, {
      id: `${props.id}-pointer-reset`,
      operation: props.operation,
      state
    })
  ];
  const roundBindings: WGSLKernelBinding[] = [
    {name: 'pointer', view: props.pointer, type: 'atomic<u32>', access: 'read_write'},
    ...(previousPointer
      ? [
          {
            name: 'previousPointer',
            view: previousPointer,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : []),
    {name: 'status', view: state.status, type: 'atomic<u32>', access: 'read_write'}
  ];
  const getRoundBody = (
    markChangedWGSL: string
  ) => `let current = ${previousPointer ? 'previousPointer[previousPointerOffset + index]' : 'atomicLoad(&pointer[pointerOffset + index])'};
  let next = ${previousPointer ? 'previousPointer[previousPointerOffset + current]' : 'atomicLoad(&pointer[pointerOffset + current])'};
  if (next != current) {
    atomicStore(&pointer[pointerOffset + index], next);
    ${markChangedWGSL}
  }`;
  const snapshots = new Map<string, GPUCommandNode<Parameters>>();
  const roundNodes = createRasterGatedLoopNodes<Parameters>(graph, {
    id: `${props.id}-pointer-round`,
    gateId: `${props.id}-pointer-gate`,
    operation: props.operation,
    state,
    maxIterations: props.maxIterations,
    createRound: ({nodeId, slotBinding, markChangedWGSL, condition, extraResources}) => {
      if (previousPointer)
        snapshots.set(
          nodeId,
          createWGSLKernelNode<Parameters>(graph, {
            id: `${nodeId}-snapshot`,
            operation: props.operation,
            variant: 'pointer-snapshot',
            bindings: [
              {name: 'pointer', view: props.pointer, type: 'u32', access: 'read'},
              {name: 'previousPointer', view: previousPointer, type: 'u32', access: 'read_write'}
            ],
            invocationCount: cellCount,
            body: 'previousPointer[previousPointerOffset + index] = pointer[pointerOffset + index];',
            condition,
            extraResources
          })
        );
      return createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: props.operation,
        variant: 'pointer-round',
        bindings: [...roundBindings, slotBinding],
        invocationCount: cellCount,
        body: getRoundBody(markChangedWGSL),
        condition,
        extraResources
      });
    }
  });
  for (const node of roundNodes) {
    const snapshot = snapshots.get(node.id);
    if (snapshot) nodes.push(snapshot);
    nodes.push(node);
  }
  if (props.converged || props.iterationCount) {
    nodes.push(
      createRasterIterationFinalizeNode<Parameters>(graph, {
        id: `${props.id}-pointer-finalize`,
        operation: props.operation,
        state,
        converged: props.converged,
        iterationCount: props.iterationCount
      })
    );
  }
  if (props.iterationLimitReached) {
    nodes.push(
      createTerrainIterationLimitNode(graph, {
        id: `${props.id}-pointer-limit`,
        operation: props.operation,
        state,
        iterationLimitReached: props.iterationLimitReached
      })
    );
  }
  return nodes;
}

/**
 * Publishes `iterationLimitReached`, one when the gated loop of `state` ran out of iterations
 * before it converged. It sits beside `converged` and `iterationCount`. @internal
 */
export function createTerrainIterationLimitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: RasterIterationState<Parameters>;
    iterationLimitReached: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'iteration-limit',
    bindings: [
      {name: 'status', view: props.state.status, type: 'u32', access: 'read'},
      {
        name: 'iterationLimitReached',
        view: props.iterationLimitReached,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: 1,
    body: 'iterationLimitReached[iterationLimitReachedOffset] = 1u - status[statusOffset + 3u];'
  });
}

/** Throws unless `view` is a packed view of `format` with one value per cell. @internal */
export function validateTerrainTracingCellView(
  id: string,
  name: string,
  view: GraphDataView,
  format: 'uint32' | 'float32',
  cellCount: number
): void {
  validateSinglePackedView(view, [format], `${id} ${name}`);
  if (view.length !== cellCount) {
    throw new Error(`${id} ${name} must contain one value per cell`);
  }
}

/** One-row status words of a pointer-jumping loop. @internal */
export type TerrainFlowTracingStatus = {
  /** One row set to 1 when pointer jumping reached its fixpoint. */
  converged?: GraphDataView<'uint32'>;
  /** One row set to 1 when the iteration budget ran out before pointer jumping converged. */
  iterationLimitReached?: GraphDataView<'uint32'>;
  /** One row: gated pointer-jumping rounds executed. */
  iterationCount?: GraphDataView<'uint32'>;
};

/** Throws unless every status view is a packed one-row `uint32` view. @internal */
export function validateTerrainTracingStatus(
  id: string,
  status: TerrainFlowTracingStatus | undefined
): void {
  for (const [name, view] of [
    ['converged', status?.converged],
    ['iterationLimitReached', status?.iterationLimitReached],
    ['iterationCount', status?.iterationCount]
  ] as const) {
    if (!view) {
      continue;
    }
    validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
    if (view.length !== 1) {
      throw new Error(`${id} output.status.${name} must contain one row`);
    }
  }
}
