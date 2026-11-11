// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
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
  getCPUTerrainGroundMeasures,
  validateCPUTerrainOperationGrid
} from './cpu-terrain-operation-grid';
import {
  GPUTerrainBranchingFlowOperation,
  type TerrainBranchingFlowOperationProps
} from './gpu-terrain-branching-flow-operation';

export type CPUTerrainBranchingFlowInput = {
  fractions: Float32Array;
  values: Float32Array;
  edgeLengths?: Float32Array;
  validity?: Uint32Array;
};
export type CPUTerrainBranchingFlowOptions = Pick<
  TerrainBranchingFlowOperationProps,
  'grid' | 'mode' | 'cellSizeMode'
>;
export type CPUTerrainBranchingFlowResult = {
  values: Float32Array;
  status: Uint32Array;
  unresolvedCount: Uint32Array;
};

/**
 * Resolves transport or downstream branching recurrences with a topological dependency queue.
 * Fractions are not renormalized. Invalid local inputs have status 0; dependencies on invalid
 * cells, cycles and invalid used lengths have status 2. unresolvedCount counts both 0 and 2.
 *
 * What differs from the GPU contributor:
 * - Float64 edge lengths, fraction sums and recurrences round once at output; compare resolved
 *   values with relative tolerance 2e-5. Rows at the fraction-sum tolerance can classify differently.
 * - No sweep or successor-walk budget is used. Every finite DAG recurrence completes; status 2
 *   never means an exhausted budget. maximumIterations and maximumWalkLength are ignored.
 * - Float64 intermediates can resolve paths that overflow on GPU. A final value outside float32
 *   is published as NaN/status 2; it may still contribute to a representable dependent result.
 * - Subnormal inputs are retained; GPU arithmetic may flush them to zero.
 *
 * @throws If fractions, values, edgeLengths or validity have inconsistent lengths.
 */
export function computeTerrainBranchingFlowOnCPU(
  input: CPUTerrainBranchingFlowInput,
  options: CPUTerrainBranchingFlowOptions
): CPUTerrainBranchingFlowResult {
  const count = validateBranchingOptions('computeTerrainBranchingFlowOnCPU', options);
  const {fractions, values, edgeLengths, validity} = input;
  if (fractions.length !== count * 8)
    throw new Error('fractions must contain eight values per grid cell');
  if (values.length !== count) throw new Error('values must contain one value per grid cell');
  if (edgeLengths && edgeLengths.length !== count * 8)
    throw new Error('edgeLengths must contain eight values per grid cell');
  if (validity && validity.length !== count)
    throw new Error('validity must contain one word per grid cell');
  const {grid, mode} = options;
  const transport = mode === 'transport';
  const distanceMode = mode !== 'transport' && mode !== 'fractional-catchment';
  const measures = getCPUTerrainGroundMeasures(grid, options.cellSizeMode);
  const getNeighbor = (cell: number, direction: number) =>
    getTerrainDrainageReceiver(cell, 1 << direction, grid.width, grid.height);
  const totals = new Float64Array(count);
  const status = new Uint32Array(count);
  const pending = new Uint32Array(count);
  const blocked = new Uint8Array(count);
  const dependents: number[][] = Array.from({length: count}, () => []);
  for (let cell = 0; cell < count; cell++) {
    let valid = Number.isFinite(values[cell]) && values[cell] >= 0 && (validity?.[cell] ?? 1) !== 0;
    if (mode === 'fractional-catchment') valid = valid && values[cell] <= 1;
    let total = 0;
    for (let direction = 0; direction < 8; direction++) {
      const fraction = fractions[cell * 8 + direction];
      valid = valid && Number.isFinite(fraction) && fraction >= 0;
      if (fraction > 0 && getNeighbor(cell, direction) === TERRAIN_DRAINAGE_NONE) valid = false;
      total += fraction;
    }
    valid = valid && (total === 0 || Math.abs(total - 1) <= 0.00001);
    status[cell] = valid ? 2 : 0;
  }
  const queue: number[] = [];
  for (let cell = 0; cell < count; cell++) {
    if (status[cell] === 0) continue;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor === TERRAIN_DRAINAGE_NONE) continue;
      const fraction =
        fractions[transport ? neighbor * 8 + ((direction + 4) % 8) : cell * 8 + direction];
      // Transport checks every neighbor's facing fraction, even on an invalid neighbor.
      if (!Number.isFinite(fraction) || fraction < 0) {
        blocked[cell] = 1;
        continue;
      }
      if (fraction === 0) continue;
      if (distanceMode) {
        const length = edgeLengths?.[cell * 8 + direction] ?? measures.edgeLength(cell, neighbor);
        if (!Number.isFinite(length) || length < 0) blocked[cell] = 1;
      }
      pending[cell]++;
      dependents[neighbor].push(cell);
    }
    if (pending[cell] === 0 && !blocked[cell]) queue.push(cell);
  }
  for (let position = 0; position < queue.length; position++) {
    const cell = queue[position];
    let value = transport ? values[cell] : 0;
    let minimum = Infinity;
    let hasReceiver = false;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(cell, direction);
      if (neighbor === TERRAIN_DRAINAGE_NONE) continue;
      const fraction =
        fractions[transport ? neighbor * 8 + ((direction + 4) % 8) : cell * 8 + direction];
      if (fraction === 0) continue;
      hasReceiver = true;
      if (!distanceMode) value += fraction * totals[neighbor];
      else {
        const candidate =
          totals[neighbor] +
          (edgeLengths?.[cell * 8 + direction] ?? measures.edgeLength(cell, neighbor));
        if (mode === 'minimum-distance') minimum = Math.min(minimum, candidate);
        else if (mode === 'maximum-distance') value = Math.max(value, candidate);
        else value += fraction * candidate;
      }
    }
    if (mode === 'minimum-distance') value = hasReceiver ? minimum : 0;
    if (mode === 'fractional-catchment' && !hasReceiver) value = values[cell];
    if (!Number.isFinite(value)) continue;
    totals[cell] = value;
    status[cell] = 1;
    for (const dependent of dependents[cell]) {
      pending[dependent]--;
      if (pending[dependent] === 0 && !blocked[dependent]) queue.push(dependent);
    }
  }
  const output = new Float32Array(count).fill(NaN);
  let unresolved = 0;
  for (let cell = 0; cell < count; cell++) {
    if (status[cell] === 1) {
      output[cell] = totals[cell];
      if (!Number.isFinite(output[cell])) {
        output[cell] = NaN;
        status[cell] = 2;
      }
    }
    if (status[cell] !== 1) unresolved++;
  }
  return {values: output, status, unresolvedCount: Uint32Array.of(unresolved)};
}

function validateBranchingOptions(id: string, options: CPUTerrainBranchingFlowOptions): number {
  const count = validateCPUTerrainOperationGrid(id, options.grid, options.cellSizeMode);
  validateRequirement(
    id,
    [
      'transport',
      'expected-distance',
      'minimum-distance',
      'maximum-distance',
      'fractional-catchment'
    ].includes(options.mode),
    'mode must name a supported branching recurrence'
  );
  return count;
}

/** CPU twin; GPU-only sweep and walk caps are dropped. */
export class CPUTerrainBranchingFlowOperation extends GPUCPUOperation {
  constructor(props: TerrainBranchingFlowOperationProps) {
    const id = props.id ?? 'terrain-branching-flow';
    validateBranchingOptions(id, props);
    validateCapacity(id, 'maximumIterations', props.maximumIterations ?? 128);
    validateCapacity(id, 'maximumWalkLength', props.maximumWalkLength ?? 32);
    const operation = new GPUTerrainBranchingFlowOperation(props);
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
        const result = computeTerrainBranchingFlowOnCPU(
          {
            fractions: getCPUOperationFloat32Rows(id, inputs, 'fractions'),
            values: getCPUOperationFloat32Rows(id, inputs, 'values'),
            edgeLengths:
              props.edgeLengths !== undefined
                ? getCPUOperationFloat32Rows(id, inputs, 'edgeLengths')
                : undefined,
            validity:
              props.validity !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'validity')
                : undefined
          },
          props
        );
        return {
          [props.output.values]: result.values,
          [props.output.status]: result.status,
          [props.output.unresolvedCount]: result.unresolvedCount
        };
      }
    });
  }
}
