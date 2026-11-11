// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeightsInformation} from '../spatial-weights/spatial-weights';

/** Packed regionalization inputs shared by the Max-P and AZP twins. @internal */
export type CPURegionInput = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  distances?: Float32Array;
  values: Float32Array;
  parameters: Uint32Array;
};

/** Checks array shapes; invalid neighbor IDs remain status-bearing input. @internal */
export function validateCPURegionInput(
  input: CPURegionInput,
  columnCount: number,
  parameterLength: number
): number {
  const {offsets, neighbors, weights, distances, values, parameters} = input;
  const rowCount = offsets.length - 1;
  validateCapacity('regionalization', 'columnCount', columnCount);
  if (
    rowCount < 1 ||
    offsets[0] !== 0 ||
    offsets.some(
      (offset, row) => offset > neighbors.length || (row > 0 && offset < offsets[row - 1])
    )
  )
    throw new Error('offsets must be nondecreasing CSR offsets starting at zero within neighbors');
  if (weights.length !== neighbors.length)
    throw new Error('weights length must equal neighbors length');
  if (distances && distances.length !== neighbors.length)
    throw new Error('distances length must equal neighbors length');
  if (values.length !== rowCount * columnCount)
    throw new Error('values length must equal rowCount * columnCount');
  if (parameters.length < parameterLength) throw new Error('parameters must hold the seed layout');
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

/** Float64 population standardization, including the GPU's tiny-deviation rule. @internal */
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
    for (let row = 0; row < rowCount; row++) {
      const value = values[row * columnCount + column];
      if (!Number.isFinite(value)) continue;
      if (count === 0) origin = value;
      sum += value - origin;
      count++;
    }
    const mean = sum / Math.max(count, 1);
    let squares = 0;
    for (let row = 0; row < rowCount; row++) {
      const value = values[row * columnCount + column];
      if (Number.isFinite(value)) squares += (value - origin - mean) ** 2;
    }
    const deviation = Math.sqrt(squares / Math.max(count, 1));
    const inverse = deviation > 1e-20 ? 1 / deviation : 0;
    for (let row = 0; row < rowCount; row++) {
      const index = row * columnCount + column;
      if (Number.isFinite(values[index])) table[index] = (values[index] - origin - mean) * inverse;
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
