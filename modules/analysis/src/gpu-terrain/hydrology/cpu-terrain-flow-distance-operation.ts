// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize,
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import type {GPUTerrainFlowDirectionEncoding} from './terrain-flow-tracing';
import {
  getTerrainDrainageDirection,
  getTerrainDrainageReceiver,
  TERRAIN_DRAINAGE_NONE,
  TERRAIN_DRAINAGE_UNRESOLVED,
  validateTerrainDrainageOptions
} from './cpu-terrain-drainage-utils';
import {
  GPUTerrainFlowDistanceOperation,
  type TerrainFlowDistanceOperationProps
} from './gpu-terrain-flow-distance-operation';

export type CPUTerrainFlowDistanceOptions = {
  grid: GPURasterMetadata;
  directionEncoding?: GPUTerrainFlowDirectionEncoding;
  cellSizeMode?: GPURasterCellSizeMode;
  maximumSteps?: number;
};
export type CPUTerrainFlowDistanceResult = {
  distance: Float32Array;
  validity: Uint32Array;
  terminalCells: Uint32Array;
};

/**
 * Resolves downstream distances and terminal cells with memoized float64 path sums. A terminal
 * contributes one cell to maximumSteps, including when a stream stops the path. Bad directions
 * take precedence over stream stops, as in the GPU seed kernel.
 *
 * What differs from the GPU contributor:
 * - Edge lengths and sums use float64 and round once at the output, rather than float32
 *   pointer-doubling sums. Resolved distances agree within float32 accumulation error.
 * - No command-node or pointer-round limit is needed. maximumSteps is retained as the documented
 *   path-length definition; cycles, overlong paths, bad costs and unknown inputs stay unresolved.
 * - Finite float64 totals that overflow float32 are invalidated at output; GPU intermediate
 *   overflow can invalidate additional paths. All terminal and missing codes otherwise match.
 */
export function computeTerrainFlowDistanceOnCPU(
  flowDirections: Uint32Array,
  options: CPUTerrainFlowDistanceOptions,
  streams?: Uint32Array,
  transitionCosts?: Float32Array
): CPUTerrainFlowDistanceResult {
  const count = validateDistanceOptions('computeTerrainFlowDistanceOnCPU', options);
  if (flowDirections.length !== count)
    throw new Error('flowDirections must contain one word per grid cell');
  if (streams && streams.length !== count)
    throw new Error('streams must contain one word per grid cell');
  if (transitionCosts && transitionCosts.length !== count)
    throw new Error('transitionCosts must contain one value per grid cell');
  const {grid, directionEncoding} = options;
  const georeference = getRasterGridGeoreference(grid);
  const receivers = new Uint32Array(count).fill(TERRAIN_DRAINAGE_NONE);
  const totals = new Float64Array(count);
  const lengths = new Float64Array(count).fill(1);
  const terminals = new Uint32Array(count).fill(TERRAIN_DRAINAGE_UNRESOLVED);
  const state = new Uint8Array(count);
  for (let cell = 0; cell < count; cell++) {
    const code = getTerrainDrainageDirection(flowDirections[cell], directionEncoding);
    const receiver = getTerrainDrainageReceiver(cell, code, grid.width, grid.height);
    const stream = streams?.[cell] ?? 0;
    if (
      (code !== 0 && receiver === TERRAIN_DRAINAGE_NONE) ||
      stream >= TERRAIN_DRAINAGE_UNRESOLVED
    ) {
      state[cell] = 2;
      continue;
    }
    if (stream !== 0 || receiver === TERRAIN_DRAINAGE_NONE) {
      terminals[cell] = cell;
      state[cell] = 2;
      continue;
    }
    const columnDifference = (receiver % grid.width) - (cell % grid.width);
    const row = Math.floor(cell / grid.width);
    const receiverRow = Math.floor(receiver / grid.width);
    const rowDifference = receiverRow - row;
    let cost: number;
    if (transitionCosts) cost = transitionCosts[cell];
    else if ((options.cellSizeMode ?? 'uniform') === 'uniform') {
      const [columnX, rowX, , columnY, rowY] = grid.affine;
      cost = Math.hypot(
        columnX * columnDifference + rowX * rowDifference,
        columnY * columnDifference + rowY * rowDifference
      );
    } else {
      const [groundX, groundY] = getRasterGroundCellSize(
        options.cellSizeMode!,
        georeference,
        (row + receiverRow + 1) / 2,
        grid.height
      );
      cost = Math.hypot(groundX * columnDifference, groundY * rowDifference);
    }
    if (!Number.isFinite(cost) || cost < 0) {
      state[cell] = 2;
      continue;
    }
    totals[cell] = cost;
    receivers[cell] = receiver;
  }
  for (let start = 0; start < count; start++) {
    if (state[start] === 2) continue;
    const path: number[] = [];
    let cell = start;
    while (state[cell] === 0) {
      state[cell] = 1;
      path.push(cell);
      cell = receivers[cell];
    }
    const terminal = state[cell] === 1 ? TERRAIN_DRAINAGE_UNRESOLVED : terminals[cell];
    let total = state[cell] === 1 ? 0 : totals[cell];
    let length = state[cell] === 1 ? 0 : lengths[cell];
    for (let position = path.length - 1; position >= 0; position--) {
      const member = path[position];
      total += totals[member];
      length++;
      terminals[member] = terminal;
      totals[member] = total;
      lengths[member] = length;
      state[member] = 2;
    }
  }
  const result = {
    distance: new Float32Array(count).fill(Number.NaN),
    validity: new Uint32Array(count),
    terminalCells: new Uint32Array(count).fill(TERRAIN_DRAINAGE_UNRESOLVED)
  };
  const maximumSteps = options.maximumSteps ?? count;
  for (let cell = 0; cell < count; cell++) {
    if (flowDirections[cell] === TERRAIN_DRAINAGE_NONE) {
      result.terminalCells[cell] = TERRAIN_DRAINAGE_NONE;
      continue;
    }
    if (terminals[cell] === TERRAIN_DRAINAGE_UNRESOLVED || lengths[cell] > maximumSteps) continue;
    result.distance[cell] = totals[cell];
    if (!Number.isFinite(result.distance[cell])) {
      result.distance[cell] = Number.NaN;
      continue;
    }
    result.validity[cell] = 1;
    result.terminalCells[cell] = terminals[cell];
  }
  return result;
}

function validateDistanceOptions(id: string, options: CPUTerrainFlowDistanceOptions): number {
  const count = validateTerrainDrainageOptions(id, options.grid, options.directionEncoding);
  validateCapacity(id, 'maximumSteps', options.maximumSteps ?? count, 1);
  validateRasterCellSizeMode(id, options.cellSizeMode);
  validateRequirement(
    id,
    (options.cellSizeMode ?? 'uniform') === 'uniform' ||
      (options.grid.affine[1] === 0 && options.grid.affine[3] === 0),
    'grid.affine must be axis-aligned for geographic and web-mercator cellSizeMode'
  );
  return count;
}

/** CPU flow-distance twin; preserves maximumSteps, formats, validity and missing terminal codes. */
export class CPUTerrainFlowDistanceOperation extends GPUCPUOperation {
  constructor(props: TerrainFlowDistanceOperationProps) {
    const operation = new GPUTerrainFlowDistanceOperation(props);
    const {id} = operation;
    validateDistanceOptions(id, props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const result = computeTerrainFlowDistanceOnCPU(
          getCPUOperationUint32Rows(id, inputs, 'flowDirections'),
          props,
          props.streams !== undefined
            ? getCPUOperationUint32Rows(id, inputs, 'streams')
            : undefined,
          props.transitionCosts !== undefined
            ? getCPUOperationFloat32Rows(id, inputs, 'transitionCosts')
            : undefined
        );
        return {
          [props.output.distance]: result.distance,
          [props.output.validity]: result.validity,
          [props.output.terminalCells]: result.terminalCells
        };
      }
    });
  }
}
