// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type RegionalizationWeightsStatus,
  validateRegionalizationWeightsStatus
} from './adjacency-loss-suppression';

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeightsInformation} from '../spatial-weights/spatial-weights';

/** Packed regionalization inputs shared by the Max-P and AZP twins. @internal */
export type CPURegionInput = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  distances?: Float32Array;
  values: Float32Array;
  parameters: Uint32Array;
  status?: RegionalizationWeightsStatus<Uint32Array>;
};

/** Adjacency loss is unavailable input, not an invalid feature or neighbor ID. @internal */
export function hasCPURegionInputLoss(input: CPURegionInput, id = 'regionalization'): boolean {
  validateRegionalizationWeightsStatus(id, input.status);
  let incomplete = false;
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    const flag = input.status?.[name];
    if (flag) {
      validateCapacity(id, `status.${name} length`, flag.length, 1);
      incomplete ||= flag[0] !== 0;
    }
  }
  return incomplete;
}

/** Checks array shapes; invalid neighbor IDs remain status-bearing input. @internal */
export function validateCPURegionInput(
  input: CPURegionInput,
  columnCount: number,
  parameterLength: number,
  id = 'regionalization'
): number {
  const {offsets, neighbors, weights, distances, values, parameters} = input;
  const rowCount = offsets.length - 1;
  validateCapacity(id, 'columnCount', columnCount);
  validateCapacity(id, 'rowCount', rowCount);
  validateRequirement(id, offsets[0] === 0, 'offsets must start at zero', {
    code: 'REQUIREMENT',
    path: 'offsets[0]',
    received: offsets[0],
    expected: 0
  });
  for (let row = 1; row <= rowCount; row++) {
    validateRequirement(
      id,
      offsets[row] >= offsets[row - 1] && offsets[row] <= neighbors.length,
      'offsets must be nondecreasing CSR offsets within neighbors',
      {
        code: 'REQUIREMENT',
        path: `offsets[${row}]`,
        received: offsets[row],
        expected: {minimum: offsets[row - 1], maximum: neighbors.length}
      }
    );
  }
  if (weights.length !== neighbors.length)
    validateRequirement(
      id,
      false,
      `weights length must equal neighbors length; received weights.length=${weights.length}, neighbors.length=${neighbors.length}`,
      {
        code: 'VIEW_LENGTH',
        path: 'weights.length',
        received: weights.length,
        expected: 'weights length must equal neighbors length'
      }
    );
  if (distances && distances.length !== neighbors.length)
    validateRequirement(
      id,
      false,
      `distances length must equal neighbors length; received distances.length=${distances.length}, neighbors.length=${neighbors.length}`,
      {
        code: 'VIEW_LENGTH',
        path: 'distances.length',
        received: distances.length,
        expected: 'distances length must equal neighbors length'
      }
    );
  if (values.length !== rowCount * columnCount)
    validateRequirement(
      id,
      false,
      `values length must equal rowCount * columnCount; received values.length=${values.length}`,
      {
        code: 'VIEW_LENGTH',
        path: 'values.length',
        received: values.length,
        expected: 'values length must equal rowCount * columnCount'
      }
    );
  if (parameters.length < parameterLength)
    validateRequirement(
      id,
      false,
      `parameters must hold the seed layout; received parameters.length=${parameters.length}`,
      {
        code: 'VIEW_LENGTH',
        path: 'parameters.length',
        received: parameters.length,
        expected: 'parameters must hold the seed layout'
      }
    );
  return rowCount;
}

/** Same adjacency order and duplicate slots as createSymmetricAdjacency. @internal */
export function getCPURegionAdjacency(
  input: CPURegionInput,
  information?: GPUSpatialWeightsInformation
): number[][] {
  const rowCount = input.offsets.length - 1;
  const adjacency: number[][] = Array.from({length: rowCount}, () => []);
  for (let row = 0; row < rowCount; row++)
    for (let slot = input.offsets[row]; slot < input.offsets[row + 1]; slot++) {
      const neighbor = input.neighbors[slot];
      adjacency[row].push(neighbor < rowCount ? neighbor : row);
    }
  if (information?.symmetric !== true)
    for (let row = 0; row < rowCount; row++)
      for (let slot = input.offsets[row]; slot < input.offsets[row + 1]; slot++) {
        const neighbor = input.neighbors[slot];
        if (neighbor < rowCount) adjacency[neighbor].push(row);
      }
  return adjacency;
}

/** Float64 population standardization: only zero deviation makes a constant column. @internal */
export function getCPURegionTable(
  values: Float32Array,
  columnCount: number,
  standardize = true
): Float64Array {
  const table = Float64Array.from(values);
  if (!standardize) return table;
  const rowCount = values.length / columnCount;
  for (let column = 0; column < columnCount; column++) {
    let origin = 0;
    let count = 0;
    let sum = 0;
    let compensation = 0;
    for (let row = 0; row < rowCount; row++) {
      const value = values[row * columnCount + column];
      if (!Number.isFinite(value)) continue;
      if (count === 0) origin = value;
      const term = value - origin;
      const next = sum + term;
      compensation += Math.abs(sum) >= Math.abs(term) ? sum - next + term : term - next + sum;
      sum = next;
      count++;
    }
    const divisor = Math.max(count, 1);
    const mean = sum / divisor;
    // Retain the correction separately: adding it to a large mean can erase a real residual.
    const meanResidual = (sum - mean * divisor + compensation) / divisor;
    let squares = 0;
    for (let row = 0; row < rowCount; row++) {
      const value = values[row * columnCount + column];
      if (Number.isFinite(value)) squares += (value - origin - mean - meanResidual) ** 2;
    }
    const deviation = Math.sqrt(squares / Math.max(count, 1));
    // Float32 input differences and their squares fit float64 even at the smallest positive
    // float32 scale. An absolute cutoff would erase information after changing column units.
    const inverse = deviation > 0 ? 1 / deviation : 0;
    for (let row = 0; row < rowCount; row++) {
      const index = row * columnCount + column;
      if (Number.isFinite(values[index]))
        table[index] = (values[index] - origin - mean - meanResidual) * inverse;
    }
  }
  return table;
}

/** Direct deviations about centered means avoid subtracting squared moments. @internal */
export function getCPURegionObjective(
  table: Float64Array,
  labels: Uint32Array,
  columnCount: number
): number {
  const members = new Map<number, number[]>();
  for (let row = 0; row < labels.length; row++) {
    const rows = members.get(labels[row]) ?? [];
    rows.push(row);
    members.set(labels[row], rows);
  }
  let objective = 0;
  for (const rows of members.values())
    for (let column = 0; column < columnCount; column++) {
      const origin = table[rows[0] * columnCount + column];
      let sum = 0;
      for (const row of rows) sum += table[row * columnCount + column] - origin;
      const mean = sum / rows.length;
      for (const row of rows) objective += (table[row * columnCount + column] - origin - mean) ** 2;
    }
  return objective;
}

/** Counts original bad slots and feature entries, without inspecting unused capacity. @internal */
export function countCPURegionInvalid(input: CPURegionInput): number {
  let invalidCount = 0;
  for (const value of input.values) if (!Number.isFinite(value)) invalidCount++;
  const rowCount = input.offsets.length - 1;
  for (let slot = 0; slot < input.offsets[rowCount]; slot++)
    if (input.neighbors[slot] >= rowCount) invalidCount++;
  return invalidCount;
}
