// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  captureGraphCommandNodes,
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getRasterGroundScaleWGSL,
  validateRasterCellSizeMode,
  writeRasterGridParameterValues,
  type GPURasterCellSizeMode,
  type RasterGridAffine
} from '../../analysis-core/contracts/raster-grid';
import type {GPUStatusScalar} from '../../analysis-core/contracts/index';
import {
  getRasterBufferBandNodes,
  validateParameterTarget,
  validateRasterBand,
  validateRasterGrid,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from '../../raster-analysis/raster-band-utils';
import {
  createRecipeKernelNode,
  getRecipeView,
  validateRecipeBoundary
} from '../recipes/expansion-recipe-utils';
import {addRecipeCountNodes, RecipeBuilder, type GPURecipeResult} from '../recipes/recipe-utils';
import {
  addSparseResistanceConnectivityRecipe,
  estimateSparseResistanceCommandNodeCount,
  SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_CONNECTIVITY_ITERATIONS,
  SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_COUNT,
  SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_ITERATIONS,
  SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_VERTEX_COUNT
} from './sparse-resistance-connectivity';

const OPERATION = 'addRasterResistanceConnectivityRecipe';

/** Largest grid cell count: every cell is a vertex of the sparse solve, which takes 1,048,576. */
export const RASTER_RESISTANCE_CONNECTIVITY_MAXIMUM_CELL_COUNT =
  SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_VERTEX_COUNT;

/**
 * Largest edge slot count, `cells * slotsPerCell`: every slot is an edge row of the sparse solve,
 * which takes 2,097,152. Four-neighbour grids use two slots per cell (east, south), so they reach
 * the cell limit; eight-neighbour grids use four (east, south, south-east, south-west), so they
 * stop at 524,288 cells.
 */
export const RASTER_RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_SLOT_COUNT =
  SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_COUNT;

/** Number of float32 values read from `GPURasterResistanceConnectivityProps.parameters`. */
export const GPU_RASTER_RESISTANCE_CONNECTIVITY_PARAMETER_LENGTH = 8;

/** Parameter slots: affine `a, b, d, e` (0 to 3), cell size (4, 5), north and south edges (6, 7). */
const AFFINE_SLOT = 0;
const GRID_SLOTS = {cellSizeIndex: 4, northEdgeIndex: 6} as const;

/**
 * Command nodes the adapter adds around the sparse solve: band canonicalisation (one published
 * `GPURasterNeighborhood` pass or a texture gather), the edge kernel, the invalid-cell count kernel
 * and its published `GPUReduction` (a few levels at a million rows), and the status kernel. The
 * allowance is an upper bound; a test counts the real nodes.
 */
export const RASTER_RESISTANCE_ADAPTER_COMMAND_NODE_ALLOWANCE = 16;

/** Edge slots per cell for each neighbourhood: east and south, plus south-east and south-west. */
function getEdgeSlotsPerCell(neighborhood: 4 | 8): number {
  return neighborhood === 4 ? 2 : 4;
}

/**
 * Packs the per-frame parameters of {@link addRasterResistanceConnectivityRecipe}: the affine
 * `a, b, d, e` (edge lengths in `'uniform'` mode), the cell size and the north and south row edges
 * (edge lengths in `'web-mercator'` and `'geographic'` mode). Changing the grid georeference
 * rewrites these values without recompiling.
 */
export function getRasterResistanceConnectivityParameterValues(
  grid: RasterGridAffine,
  target: Float32Array = new Float32Array(GPU_RASTER_RESISTANCE_CONNECTIVITY_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(OPERATION, target, GPU_RASTER_RESISTANCE_CONNECTIVITY_PARAMETER_LENGTH);
  const [a, b, , d, e] = grid.affine;
  target[AFFINE_SLOT] = a;
  target[AFFINE_SLOT + 1] = b;
  target[AFFINE_SLOT + 2] = d;
  target[AFFINE_SLOT + 3] = e;
  writeRasterGridParameterValues(target, grid, GRID_SLOTS);
  return target;
}

/** Caller-owned outputs of {@link addRasterResistanceConnectivityRecipe}. */
export type GPURasterResistanceConnectivityOutput = {
  /** `float32`, one per cell: potential of every cell in the sink's component, else NaN. */
  potentials: GraphDataView<'float32'>;
  /** `float32`, one per cell: Circuitscape-style node current (the current map), else NaN. */
  vertexCurrents: GraphDataView<'float32'>;
  /** `float32x4`, one row: effective resistance, true residual, dissipated energy, recursive residual. */
  diagnostics: GraphDataView<'float32x4'>;
  /** `uint32x4`, one row: the sparse solve's `[converged, iterations, reachable, invalidEdges]`. */
  solverStatus: GraphDataView<'uint32x4'>;
  /** Optional `float32`, one per edge slot: signed current from the cell to its neighbour. */
  currents?: GraphDataView<'float32'>;
  /** Optional `uint32x2`, one per edge slot: `(cell, neighbour)`; `(cell, cell)` off the grid. */
  edges?: GraphDataView<'uint32x2'>;
  /** Optional `float32`, one per edge slot: the conductance the solve used (0 barrier, NaN invalid). */
  conductances?: GraphDataView<'float32'>;
  /** Optional one-row scalars of the result status stages. */
  status?: {
    /**
     * Cells with data and a zero or negative resistance, plus cells with data whose edge to
     * another positive resistance has a conductance that overflows float32 (stage `resistance`).
     */
    invalidCellCount?: GPUStatusScalar;
    /** Stage `solve`. */
    converged?: GPUStatusScalar;
    /** Stage `solve`: conjugate gradient iterations run. */
    iterationCount?: GPUStatusScalar;
    /** Stage `solve`: 1 when the solve ran to `maximumIterations` without converging. */
    iterationLimitReached?: GPUStatusScalar;
    /** Stage `solve`: invalid edges (NaN conductance; each touches an invalid cell). */
    invalidEdgeCount?: GPUStatusScalar;
  };
};

export type GPURasterResistanceConnectivityProps = {
  /** Node and transient ID prefix. Default `'raster-resistance-connectivity'`. */
  id?: string;
  /** The raster grid: `width * height` cells, affine and pixel interpretation. */
  grid: GPURasterMetadata;
  /**
   * Resistance per cell (any published band: buffer or texture, scale, offset and no-data are
   * resolved first). No-data cells (zero validity, the band's no-data value, or a non-finite value,
   * including +Infinity) are removed from the circuit. A cell with data and a zero or negative
   * resistance is invalid: it is counted, and every edge it shares with a cell with data is invalid,
   * which blocks the solve. Two positive resistances so small that their conductance
   * `2 / (R(a) + R(b)) / edgeLength` exceeds the float32 maximum (about 3.4e38) make their edge
   * invalid the same way, and both cells are counted.
   */
  resistance: GPURasterBand;
  /**
   * Per-frame parameters with at least 8 float32 values, packed by
   * {@link getRasterResistanceConnectivityParameterValues} from the same `grid`.
   */
  parameters: GraphDataView<'float32'>;
  /** Neighbours of a cell: 4 (rook) or 8 (queen). Default 8. Compile-time. */
  neighborhood?: 4 | 8;
  /**
   * Ground model for edge lengths. Default `'uniform'` (affine units, exact for rotated and
   * sheared grids). `'web-mercator'` and `'geographic'` measure WGS84 ground meters at the midpoint
   * row of each edge and need an axis-aligned grid. Compile-time.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Source cell index, `row * width + column`. Compile-time. */
  source: number;
  /** Sink cell index, `row * width + column`; held at potential zero. Compile-time. */
  sink: number;
  /** Passed to the sparse solve: 1 to 4,096, default 1,024. Compile-time. */
  maximumIterations?: number;
  /** Passed to the sparse solve: absolute current residual, default 1e-4. */
  tolerance?: number;
  /** Passed to the sparse solve: 1 to 1,024, default 128. Compile-time. */
  connectivityIterations?: number;
  output: GPURasterResistanceConnectivityOutput;
};

export type GPURasterResistanceConnectivityRecipeResult = GPURecipeResult<
  {
    potentials: GraphDataView<'float32'>;
    vertexCurrents: GraphDataView<'float32'>;
    diagnostics: GraphDataView<'float32x4'>;
    solverStatus: GraphDataView<'uint32x4'>;
    currents: GraphDataView<'float32'>;
  },
  {
    resistance: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    edges: GraphDataView<'uint32x2'>;
    conductances: GraphDataView<'float32'>;
  }
> & {
  /** Upper bound of the command nodes this recipe added to the graph. */
  estimatedCommandNodeCount: number;
  provenance: ReturnType<typeof addSparseResistanceConnectivityRecipe>['provenance'] & {
    grid: GPURasterMetadata;
    neighborhood: 4 | 8;
    cellSizeMode: GPURasterCellSizeMode;
    conductance: '2 / (R(a) + R(b)) / edgeLength';
  };
};

/** Upper bound of the command nodes {@link addRasterResistanceConnectivityRecipe} adds. */
export function estimateRasterResistanceCommandNodeCount(
  maximumIterations: number,
  connectivityIterations: number
): number {
  return (
    RASTER_RESISTANCE_ADAPTER_COMMAND_NODE_ALLOWANCE +
    estimateSparseResistanceCommandNodeCount(maximumIterations, connectivityIterations)
  );
}

/** WGSL `getEdgeLength(row, deltaColumn, deltaRow)` in ground units for one cell size mode. */
function getEdgeLengthWGSL(cellSizeMode: GPURasterCellSizeMode): string {
  const word = (slot: number) => `parameters[parametersOffset + ${slot}u]`;
  if (cellSizeMode === 'uniform') {
    // The affine maps a pixel step to a world step, so skewed and rotated grids are exact.
    return `fn getEdgeLength(row: u32, deltaColumn: f32, deltaRow: f32) -> f32 {
  return length(vec2<f32>(${word(AFFINE_SLOT)} * deltaColumn + ${word(AFFINE_SLOT + 1)} * deltaRow, ${word(AFFINE_SLOT + 2)} * deltaColumn + ${word(AFFINE_SLOT + 3)} * deltaRow));
}`;
  }
  // Ground cell size at the midpoint row of the edge, the rule of the terrain flow measures.
  return `fn getEdgeLength(row: u32, deltaColumn: f32, deltaRow: f32) -> f32 {
  let rowCoordinate = f32(row) + 0.5 + 0.5 * deltaRow;
  let edge = mix(${word(GRID_SLOTS.northEdgeIndex)}, ${word(GRID_SLOTS.northEdgeIndex + 1)}, rowCoordinate / f32(HEIGHT));
  ${getRasterGroundScaleWGSL(cellSizeMode)}
  let ground = vec2<f32>(${word(GRID_SLOTS.cellSizeIndex)}, ${word(GRID_SLOTS.cellSizeIndex + 1)}) * groundScale;
  return length(vec2<f32>(ground.x * abs(deltaColumn), ground.y * abs(deltaRow)));
}`;
}

/**
 * Circuit connectivity across a resistance raster, for one source and one sink. Each cell is a
 * vertex; each pair of neighbouring cells with data is an edge with conductance
 * `2 / (R(a) + R(b)) / edgeLength`, where `edgeLength` is the ground distance between the cell
 * centres (the diagonal length for diagonal neighbours) under `cellSizeMode`. Resistance is
 * therefore per unit ground length, and conductance is per unit ground length too. The edge list
 * is built on the GPU, one invocation per edge slot, then solved by
 * {@link addSparseResistanceConnectivityRecipe} with unit current from `source` to `sink`.
 *
 * Relation to Circuitscape's raster pairwise mode: this is not the same convention. Circuitscape
 * (average-resistance connection) uses `1 / mean(R)` for cardinal neighbours and
 * `1 / (sqrt(2) * mean(R))` for diagonal ones and never divides by the cell size. On a planar grid
 * of square cells of size `s` (`'uniform'` mode) every conductance here is Circuitscape's divided
 * by `s`: the current map is the same, and potentials and effective resistance are `s` times
 * Circuitscape's (two cells with `R = 1` and `s = 30` give 30, Circuitscape gives 1). With
 * rectangular cells, and in `'geographic'` and `'web-mercator'` mode where the edge length changes
 * with the row, the conductance ratios differ and so does the current map.
 *
 * Edge slots have a fixed layout, `cell * slotsPerCell + direction`, with directions east and south
 * (plus south-east and south-west for `neighborhood: 8`); a slot whose neighbour is off the grid is
 * a self edge of zero conductance, and a slot touching a no-data cell has zero conductance.
 *
 * No-data: zero validity, the band's no-data value or a non-finite resistance (including
 * +Infinity) removes the cell; its potential and current are NaN. Zero or negative resistance with
 * data is invalid: such cells are counted in stage `resistance` (`invalidCount`), each edge they
 * share with a cell with data gets a NaN conductance, and the sparse solve then reports those
 * edges in stage `solve` (`invalidCount`) and publishes no potentials. A positive resistance is
 * invalid in the same way when an edge between it and another positive resistance has a
 * conductance that is not finite in float32 (for example `R = 1e-35` on both sides of a `1e-4`
 * cell): the edge gets a NaN conductance and both cells are counted, so `invalidCount` of stage
 * `resistance` is nonzero whenever stage `solve` reports invalid edges. Clamp resistances to a
 * positive minimum (or mark them no-data) before calling.
 *
 * Limits: at most 1,048,576 cells and at most 2,097,152 edge slots (524,288 cells with eight
 * neighbours), the limits of the sparse solve. Compile-time: the grid size, `neighborhood`,
 * `cellSizeMode`, `source`, `sink`, `tolerance` and the iteration budgets. Per-frame: `parameters`
 * (the grid georeference) and the resistance values.
 */
export function addRasterResistanceConnectivityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterResistanceConnectivityProps
): GPURasterResistanceConnectivityRecipeResult {
  const id = props.id ?? 'raster-resistance-connectivity';
  const {grid, output} = props;
  const neighborhood = props.neighborhood ?? 8;
  const cellSizeMode = props.cellSizeMode ?? 'uniform';
  const maximumIterations = props.maximumIterations ?? 1024;
  const connectivityIterations = props.connectivityIterations ?? 128;
  const cellCount = validateRasterGrid(id, grid);
  validateRequirement(id, neighborhood === 4 || neighborhood === 8, 'neighborhood must be 4 or 8');
  validateMaximumCount(
    id,
    'grid cells (width * height)',
    cellCount,
    RASTER_RESISTANCE_CONNECTIVITY_MAXIMUM_CELL_COUNT
  );
  const slotsPerCell = getEdgeSlotsPerCell(neighborhood);
  const edgeCount = cellCount * slotsPerCell;
  validateMaximumCount(
    id,
    `edge slots (grid cells * ${slotsPerCell} for neighborhood ${neighborhood})`,
    edgeCount,
    RASTER_RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_SLOT_COUNT
  );
  validateRequirement(id, cellCount >= 2, 'grid must have at least 2 cells');
  validateCapacity(id, 'source', props.source, 0, cellCount - 1);
  validateCapacity(id, 'sink', props.sink, 0, cellCount - 1);
  validateRequirement(id, props.source !== props.sink, 'source and sink must be different cells');
  // The solve's own limits, checked here so that nothing is added to the graph before they hold.
  validateCapacity(
    id,
    'maximumIterations',
    maximumIterations,
    1,
    SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_ITERATIONS
  );
  validateCapacity(
    id,
    'connectivityIterations',
    connectivityIterations,
    1,
    SPARSE_RESISTANCE_CONNECTIVITY_MAXIMUM_CONNECTIVITY_ITERATIONS
  );
  validateRequirement(
    id,
    props.tolerance === undefined || (Number.isFinite(props.tolerance) && props.tolerance > 0),
    'tolerance must be a positive finite number'
  );
  validateRasterCellSizeMode(id, cellSizeMode);
  validateRasterGridAxisAligned(id, grid, cellSizeMode);
  validateRasterBand(props.resistance, grid, `${id} resistance`);
  validateRequirement(
    id,
    !props.resistance.validity || props.resistance.validity.length === cellCount,
    'resistance.validity must hold one uint32 word per grid cell'
  );
  validateRasterParameters(
    id,
    props.parameters,
    GPU_RASTER_RESISTANCE_CONNECTIVITY_PARAMETER_LENGTH
  );
  validateViewLength(id, 'output.potentials', output.potentials, cellCount);
  validateViewLength(id, 'output.vertexCurrents', output.vertexCurrents, cellCount);
  if (output.currents) validateViewLength(id, 'output.currents', output.currents, edgeCount);
  const estimatedCommandNodeCount = estimateRasterResistanceCommandNodeCount(
    maximumIterations,
    connectivityIterations
  );
  validateCommandNodeBudget(id, estimatedCommandNodeCount);

  const edges = getRecipeView(graph, `${id}-edges`, 'uint32x2', edgeCount, output.edges);
  const conductances = getRecipeView(
    graph,
    `${id}-conductances`,
    'float32',
    edgeCount,
    output.conductances
  );
  const statusView = (name: keyof NonNullable<GPURasterResistanceConnectivityOutput['status']>) =>
    getRecipeView(graph, `${id}-${name}`, 'uint32', 1, output.status?.[name]);
  const invalidCellCount = statusView('invalidCellCount');
  const converged = statusView('converged');
  const iterationCount = statusView('iterationCount');
  const iterationLimitReached = statusView('iterationLimitReached');
  const invalidEdgeCount = statusView('invalidEdgeCount');
  const statusViews = [
    invalidCellCount,
    converged,
    iterationCount,
    iterationLimitReached,
    invalidEdgeCount
  ];
  validateRecipeBoundary(
    graph,
    id,
    [
      props.parameters,
      ...(props.resistance.storage.kind === 'buffer' ? [props.resistance.storage.values] : []),
      props.resistance.validity
    ],
    [
      output.potentials,
      output.vertexCurrents,
      output.diagnostics,
      output.solverStatus,
      output.currents,
      edges,
      conductances,
      ...statusViews
    ]
  );

  const builder = new RecipeBuilder(graph);
  // Scale, offset, no-data and texture storage resolve into float32 values plus validity.
  const resolved = getRasterBufferBandNodes(
    graph,
    `${id}-resistance`,
    props.resistance,
    grid,
    'reuse-canonical'
  );
  if (resolved.nodes.length > 0) builder.addGlue('resistance-band', resolved.nodes);
  const resistance = resolved.band.storage.values as GraphDataView<'float32'>;
  const validity = resolved.band.validity!;

  const cellBindings = [
    {name: 'resistance', view: resistance, type: 'f32', access: 'read'},
    {name: 'validity', view: validity, type: 'u32', access: 'read'}
  ] as const;
  const cellDeclarations = `${IS_FINITE_VALUE_WGSL}
fn hasData(cell: u32) -> bool {
  return validity[validityOffset + cell] != 0u && isFiniteValue(resistance[resistanceOffset + cell]);
}`;
  // One invocation per edge slot: endpoints, then the conductance of the two cells' mean resistance.
  builder.addGlue(
    'edges',
    createRecipeKernelNode(graph, {
      id: `${id}-edges`,
      operation: OPERATION,
      variant: `neighborhood-${neighborhood}-${cellSizeMode}`,
      invocationCount: edgeCount,
      declarations: `override WIDTH: u32;
override HEIGHT: u32;
override SLOTS_PER_CELL: u32;
${cellDeclarations}
${GET_NAN_WGSL}
${getEdgeLengthWGSL(cellSizeMode)}`,
      constants: {WIDTH: grid.width, HEIGHT: grid.height, SLOTS_PER_CELL: slotsPerCell},
      bindings: [
        ...cellBindings,
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'edges', view: edges, type: 'u32', access: 'read_write'},
        {name: 'conductances', view: conductances, type: 'f32', access: 'read_write'}
      ],
      body: `let cell = index / SLOTS_PER_CELL;
  let direction = index % SLOTS_PER_CELL;
  let column = cell % WIDTH;
  let row = cell / WIDTH;
  // Directions: 0 east, 1 south, 2 south-east, 3 south-west.
  let deltaColumn = select(select(select(-1i, 1i, direction == 2u), 0i, direction == 1u), 1i, direction == 0u);
  let deltaRow = select(1i, 0i, direction == 0u);
  let otherColumn = i32(column) + deltaColumn;
  let otherRow = row + u32(deltaRow);
  let inside = otherColumn >= 0i && otherColumn < i32(WIDTH) && otherRow < HEIGHT;
  let other = select(cell, otherRow * WIDTH + u32(max(otherColumn, 0i)), inside);
  edges[edgesOffset + index * 2u] = cell;
  edges[edgesOffset + index * 2u + 1u] = other;
  var conductance = 0.0;
  if (inside && hasData(cell) && hasData(other)) {
    let first = resistance[resistanceOffset + cell];
    let second = resistance[resistanceOffset + other];
    conductance = getNaN();
    if (first > 0.0 && second > 0.0) {
      let value = 2.0 / (first + second) / getEdgeLength(row, f32(deltaColumn), f32(deltaRow));
      if (isFiniteValue(value)) { conductance = value; }
    }
  }
  conductances[conductancesOffset + index] = conductance;`
    })
  );
  // A cell with data is invalid when its resistance is not positive, or when an edge to another
  // positive resistance got a NaN conductance (`2 / (R(a) + R(b)) / edgeLength` overflowed
  // float32). The second case reads the conductances the edge kernel wrote, so the count and the
  // solve's invalid edges agree exactly. Incoming slots: the west cell's east slot (0), the north
  // cell's south slot (1) and, with eight neighbours, the north-west cell's south-east slot (2)
  // and the north-east cell's south-west slot (3).
  addRecipeCountNodes(builder, {
    id: `${id}-invalid-cells`,
    operation: OPERATION,
    rowCount: cellCount,
    bindings: [
      ...cellBindings,
      {name: 'edges', view: edges, type: 'u32', access: 'read'},
      {name: 'conductances', view: conductances, type: 'f32', access: 'read'}
    ],
    declarations: `${cellDeclarations}
const WIDTH: u32 = ${grid.width}u;
const SLOTS_PER_CELL: u32 = ${slotsPerCell}u;
fn isOverflowSlot(slot: u32) -> bool {
  let bits = bitcast<u32>(conductances[conductancesOffset + slot]) & 0x7fffffffu;
  let first = edges[edgesOffset + slot * 2u];
  let second = edges[edgesOffset + slot * 2u + 1u];
  return bits > 0x7f800000u &&
    resistance[resistanceOffset + first] > 0.0 && resistance[resistanceOffset + second] > 0.0;
}
fn isInvalidCell(cell: u32) -> bool {
  if (!hasData(cell)) {
    return false;
  }
  if (!(resistance[resistanceOffset + cell] > 0.0)) {
    return true;
  }
  let column = cell % WIDTH;
  var invalid = false;
  for (var direction = 0u; direction < SLOTS_PER_CELL; direction++) {
    invalid = invalid || isOverflowSlot(cell * SLOTS_PER_CELL + direction);
  }
  if (column > 0u) {
    invalid = invalid || isOverflowSlot((cell - 1u) * SLOTS_PER_CELL);
  }
  if (cell >= WIDTH) {
    let north = cell - WIDTH;
    invalid = invalid || isOverflowSlot(north * SLOTS_PER_CELL + 1u);
    if (SLOTS_PER_CELL == 4u && column > 0u) {
      invalid = invalid || isOverflowSlot((north - 1u) * SLOTS_PER_CELL + 2u);
    }
    if (SLOTS_PER_CELL == 4u && column + 1u < WIDTH) {
      invalid = invalid || isOverflowSlot((north + 1u) * SLOTS_PER_CELL + 3u);
    }
  }
  return invalid;
}`,
    counts: [
      {
        name: 'invalidCells',
        output: invalidCellCount,
        value: 'select(0u, 1u, isInvalidCell(index))'
      }
    ]
  });

  let solve: ReturnType<typeof addSparseResistanceConnectivityRecipe> | undefined;
  const solveNodes = captureGraphCommandNodes(graph, () => {
    solve = addSparseResistanceConnectivityRecipe(graph, {
      id: `${id}-solve`,
      edges,
      conductances,
      vertexCount: cellCount,
      source: props.source,
      sink: props.sink,
      maximumIterations,
      tolerance: props.tolerance,
      connectivityIterations,
      output: {
        potentials: output.potentials,
        currents: output.currents,
        vertexCurrents: output.vertexCurrents,
        diagnostics: output.diagnostics,
        status: output.solverStatus
      }
    });
  });
  builder.addGlue('solve', solveNodes);
  const solved = solve!;

  builder.addGlue(
    'solve-status',
    createRecipeKernelNode(graph, {
      id: `${id}-solve-status`,
      operation: OPERATION,
      invocationCount: 1,
      bindings: [
        {name: 'solverStatus', view: output.solverStatus, type: 'u32', access: 'read'},
        {name: 'converged', view: converged, type: 'u32', access: 'read_write'},
        {name: 'iterationCount', view: iterationCount, type: 'u32', access: 'read_write'},
        {name: 'limitReached', view: iterationLimitReached, type: 'u32', access: 'read_write'},
        {name: 'invalidEdges', view: invalidEdgeCount, type: 'u32', access: 'read_write'}
      ],
      body: `let isConverged = solverStatus[solverStatusOffset];
  let iterations = solverStatus[solverStatusOffset + 1u];
  let ran = solverStatus[solverStatusOffset + 2u] != 0u && solverStatus[solverStatusOffset + 3u] == 0u;
  converged[convergedOffset] = isConverged;
  iterationCount[iterationCountOffset] = iterations;
  limitReached[limitReachedOffset] = select(0u, 1u, ran && isConverged == 0u && iterations >= ${maximumIterations}u);
  invalidEdges[invalidEdgesOffset] = solverStatus[solverStatusOffset + 3u];`
    })
  );

  return {
    contributors: builder.contributors,
    outputs: {
      potentials: solved.outputs.potentials,
      vertexCurrents: solved.outputs.vertexCurrents,
      diagnostics: solved.outputs.diagnostics,
      solverStatus: solved.status,
      currents: solved.outputs.currents
    },
    intermediates: {resistance, validity, edges, conductances},
    status: {
      stages: [
        {stage: 'resistance', status: {invalidCount: invalidCellCount}},
        {
          stage: 'solve',
          status: {converged, iterationCount, iterationLimitReached, invalidCount: invalidEdgeCount}
        }
      ]
    },
    estimatedCommandNodeCount,
    provenance: {
      ...solved.provenance,
      grid,
      neighborhood,
      cellSizeMode,
      conductance: '2 / (R(a) + R(b)) / edgeLength'
    }
  };
}
