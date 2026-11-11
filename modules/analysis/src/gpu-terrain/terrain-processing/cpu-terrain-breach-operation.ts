// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getTerrainDrainageReceiver,
  TERRAIN_DRAINAGE_NONE
} from '../hydrology/cpu-terrain-drainage-utils';
import {
  TERRAIN_OPERATION_MAXIMUM_WORD_COUNT,
  validateCPUTerrainOperationGrid
} from './cpu-terrain-operation-grid';
import {
  GPUTerrainBreachOperation,
  type TerrainBreachOperationProps
} from './gpu-terrain-breach-operation';

export type CPUTerrainBreachInput = {
  elevation: Float32Array;
  validity?: Uint32Array;
  pitCells: Uint32Array;
  burnDepth?: Float32Array;
};
export type CPUTerrainBreachOptions = Pick<
  TerrainBreachOperationProps,
  'grid' | 'maximumSteps' | 'maximumCutDepth' | 'elevationDecrement'
>;
export type CPUTerrainBreachResult = {
  elevation: Float32Array;
  cutDepth: Float32Array;
  modificationMask: Uint32Array;
  pathCells: Uint32Array;
  pathLengths: Uint32Array;
  pitStatus: Uint32Array;
  unresolvedCount: Uint32Array;
};

/**
 * Finds minimum-total-excavation paths on the bounded step-layered D8 graph. Ties choose fewer
 * steps, then the lowest outlet ID; predecessor ties choose the lowest cell ID. All pits search
 * the original surface. Their minimum-elevation proposals combine simultaneously before burning.
 * Paths remain published if the final strict-descent check changes pitStatus from 1 to 2.
 *
 * What differs from the GPU contributor:
 * - Levels, costs, cuts and burns use float64, rounded once at output. Compare finite nonzero
 *   elevations/depths with relative tolerance 2e-5 away from near ties and feasibility bounds;
 *   near ties can choose different integer paths or modification flags.
 * - No GPU scratch-word or storage-binding cap is imposed: one pit's search is held at a time.
 *   maximumSteps and maximumCutDepth remain the definition of the bounded optimization problem.
 * - Positive finite decrements need not be representable in float32 or change the float32 pit
 *   elevation. The search uses float64; final paths must descend strictly in the rounded output,
 *   so collapsed output steps have status 2. A decrement too small to change float64 base has
 *   status 0. GPU-only representability rejection can therefore become a CPU search result.
 * - The pure function supports no pits (burn-only); the operation form requires pitCount above
 *   zero because GPUContributorOperation requires nonempty output vectors.
 * - Subnormals are retained; GPU arithmetic may flush them to zero.
 *
 * @throws If elevation, validity or burnDepth does not contain exactly one value per grid cell.
 */
export function computeTerrainBreachOnCPU(
  input: CPUTerrainBreachInput,
  options: CPUTerrainBreachOptions
): CPUTerrainBreachResult {
  const count = validateBreachOptions('computeTerrainBreachOnCPU', options);
  const {pitCells, validity, burnDepth} = input;
  if (input.elevation.length !== count)
    throw new Error('elevation must contain one value per grid cell');
  if (validity && validity.length !== count)
    throw new Error('validity must contain one word per grid cell');
  if (burnDepth && burnDepth.length !== count)
    throw new Error('burnDepth must contain one value per grid cell');
  const {grid, maximumSteps, maximumCutDepth, elevationDecrement} = options;
  const stride = maximumSteps + 1;
  validateMaximumCount(
    'computeTerrainBreachOnCPU',
    'pitCells.length * (maximumSteps + 1)',
    pitCells.length * stride,
    TERRAIN_OPERATION_MAXIMUM_WORD_COUNT
  );
  const source = Float64Array.from(input.elevation, (value, cell) =>
    (validity?.[cell] ?? 1) !== 0 ? value : NaN
  );
  const edited = source.slice();
  const pathCells = new Uint32Array(pitCells.length * stride).fill(TERRAIN_DRAINAGE_NONE);
  const pathLengths = new Uint32Array(pitCells.length);
  const pitStatus = new Uint32Array(pitCells.length);
  const searchWidth = Math.min(grid.width, maximumSteps * 2 + 1);
  const searchHeight = Math.min(grid.height, maximumSteps * 2 + 1);
  const searchCount = searchWidth * searchHeight;
  for (let pitIndex = 0; pitIndex < pitCells.length; pitIndex++) {
    const pit = pitCells[pitIndex];
    if (
      pit >= count ||
      !Number.isFinite(source[pit]) ||
      !(source[pit] - elevationDecrement < source[pit])
    )
      continue;
    const base = source[pit];
    const originColumn = Math.max(
      0,
      Math.min(grid.width - searchWidth, (pit % grid.width) - maximumSteps)
    );
    const originRow = Math.max(
      0,
      Math.min(grid.height - searchHeight, Math.floor(pit / grid.width) - maximumSteps)
    );
    const getCell = (localCell: number) =>
      (originRow + Math.floor(localCell / searchWidth)) * grid.width +
      originColumn +
      (localCell % searchWidth);
    const getLocalCell = (cell: number) => {
      const column = cell % grid.width,
        row = Math.floor(cell / grid.width);
      return column < originColumn ||
        row < originRow ||
        column >= originColumn + searchWidth ||
        row >= originRow + searchHeight
        ? TERRAIN_DRAINAGE_NONE
        : (row - originRow) * searchWidth + column - originColumn;
    };
    let previous = new Float64Array(searchCount).fill(Infinity);
    previous[getLocalCell(pit)] = 0;
    const predecessors = new Float64Array(stride * searchCount).fill(TERRAIN_DRAINAGE_NONE);
    let bestCost = Infinity,
      bestCell = TERRAIN_DRAINAGE_NONE,
      bestStep = 0;
    for (let step = 1; step <= maximumSteps; step++) {
      const next = new Float64Array(searchCount).fill(Infinity);
      const level = base - step * elevationDecrement;
      for (let localCell = 0; localCell < searchCount; localCell++) {
        const cell = getCell(localCell),
          original = source[cell];
        const cut = Math.max(original - level, 0);
        if (!Number.isFinite(original) || cut > maximumCutDepth) continue;
        let cost = Infinity,
          predecessor = TERRAIN_DRAINAGE_NONE;
        for (let direction = 0; direction < 8; direction++) {
          const neighbor = getTerrainDrainageReceiver(
            cell,
            1 << direction,
            grid.width,
            grid.height
          );
          if (neighbor === TERRAIN_DRAINAGE_NONE) continue;
          const localNeighbor = getLocalCell(neighbor);
          if (localNeighbor === TERRAIN_DRAINAGE_NONE) continue;
          const candidate = previous[localNeighbor] + cut;
          if (candidate < cost || (candidate === cost && neighbor < predecessor)) {
            cost = candidate;
            predecessor = neighbor;
          }
        }
        next[localCell] = cost;
        predecessors[step * searchCount + localCell] = predecessor;
        if (original <= level && cost < bestCost) {
          bestCost = cost;
          bestCell = cell;
          bestStep = step;
        }
      }
      previous = next;
    }
    if (bestCell === TERRAIN_DRAINAGE_NONE) {
      pitStatus[pitIndex] = 2;
      continue;
    }
    let current = bestCell;
    for (let step = bestStep; step > 0; step--) {
      pathCells[pitIndex * stride + step] = current;
      current = predecessors[step * searchCount + getLocalCell(current)];
    }
    pathCells[pitIndex * stride] = pit;
    pathLengths[pitIndex] = bestStep + 1;
    pitStatus[pitIndex] = 1;
    for (let step = 1; step <= bestStep; step++) {
      const cell = pathCells[pitIndex * stride + step];
      edited[cell] = Math.min(edited[cell], base - step * elevationDecrement);
    }
  }
  const elevation = new Float32Array(count),
    cutDepth = new Float32Array(count),
    modificationMask = new Uint32Array(count);
  for (let cell = 0; cell < count; cell++) {
    const original = source[cell];
    let value = edited[cell],
      flags = 0;
    if (Number.isFinite(original)) {
      if (value < original) flags |= 1;
      if (burnDepth) {
        const depth = burnDepth[cell];
        if (!Number.isFinite(depth) || depth < 0) {
          value = NaN;
          flags |= 4;
        } else if (depth > 0) {
          value -= depth;
          flags |= 2;
        }
      }
    }
    elevation[cell] = value;
    cutDepth[cell] = original - value;
    modificationMask[cell] = flags;
  }
  let unresolved = 0;
  for (let pitIndex = 0; pitIndex < pitCells.length; pitIndex++) {
    if (pitStatus[pitIndex] === 1) {
      for (let step = 1; step < pathLengths[pitIndex]; step++) {
        const previous = pathCells[pitIndex * stride + step - 1],
          current = pathCells[pitIndex * stride + step];
        if (
          ((modificationMask[previous] | modificationMask[current]) & 4) !== 0 ||
          !Number.isFinite(elevation[previous]) ||
          !Number.isFinite(elevation[current]) ||
          !(elevation[current] < elevation[previous])
        )
          pitStatus[pitIndex] = 2;
      }
    }
    if (pitStatus[pitIndex] !== 1) unresolved++;
  }
  return {
    elevation,
    cutDepth,
    modificationMask,
    pathCells,
    pathLengths,
    pitStatus,
    unresolvedCount: Uint32Array.of(unresolved)
  };
}

function validateBreachOptions(id: string, options: CPUTerrainBreachOptions): number {
  const count = validateCPUTerrainOperationGrid(id, options.grid);
  validateCapacity(id, 'maximumSteps', options.maximumSteps);
  validateFiniteNumber(id, 'maximumCutDepth', options.maximumCutDepth, 0);
  validateRequirement(
    id,
    Number.isFinite(options.elevationDecrement) && options.elevationDecrement > 0,
    'elevationDecrement must be a positive finite number'
  );
  return count;
}

/** CPU breach twin; searches one pit at a time, without GPU scratch/binding limits. */
export class CPUTerrainBreachOperation extends GPUCPUOperation {
  constructor(props: TerrainBreachOperationProps) {
    const id = props.id ?? 'terrain-breach';
    validateBreachOptions(id, props);
    validateCapacity(id, 'pitCount', props.pitCount);
    validateMaximumCount(
      id,
      'pitCount * (maximumSteps + 1)',
      props.pitCount * (props.maximumSteps + 1),
      TERRAIN_OPERATION_MAXIMUM_WORD_COUNT
    );
    const operation = new GPUTerrainBreachOperation(props);
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
        const pitCells = getCPUOperationUint32Rows(id, inputs, 'pitCells');
        if (pitCells.length !== props.pitCount)
          throw new Error(`${id} pitCells length must equal pitCount`);
        const result = computeTerrainBreachOnCPU(
          {
            elevation: getCPUOperationFloat32Rows(id, inputs, 'elevation'),
            pitCells,
            validity:
              props.validity !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'validity')
                : undefined,
            burnDepth:
              props.burnDepth !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'burnDepth')
                : undefined
          },
          props
        );
        return {
          [props.output.elevation]: result.elevation,
          [props.output.cutDepth]: result.cutDepth,
          [props.output.modificationMask]: result.modificationMask,
          [props.output.pathCells]: result.pathCells,
          [props.output.pathLengths]: result.pathLengths,
          [props.output.pitStatus]: result.pitStatus,
          [props.output.unresolvedCount]: result.unresolvedCount
        };
      }
    });
  }
}
