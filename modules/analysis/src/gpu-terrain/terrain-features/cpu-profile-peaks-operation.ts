// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_PROFILE_PEAKS_PARAMETER_LENGTH} from './gpu-profile-peaks';
import {
  GPUProfilePeaksOperation,
  snapshotProfilePeaksOperationProps,
  type ProfilePeaksOperationProps
} from './gpu-profile-peaks-operation';

export type CPUProfilePeaksOptions = {
  window: number;
  minimumSide?: number;
  suppressionRadius?: number;
  wrap?: boolean;
  maximumSuppressionIterations?: number;
  capacity: number;
};

/** Dense rows and compact IDs, with scalar status in one-element arrays. */
export type CPUProfilePeaksResult = {
  prominence: Float32Array;
  refinedIndex: Float32Array;
  refinedValue: Float32Array;
  peakMask: Uint32Array;
  ids: Uint32Array;
  count: Uint32Array;
  requiredCount: Uint32Array;
  overflow: Uint32Array;
  candidateOverflow: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/**
 * Finds window-prominent maxima, refines them parabolically and completes ranked greedy suppression.
 * Parameters use the layout of getGPUProfilePeaksParameterValues (minimumProminence).
 *
 * What differs from the GPU contributor:
 * - All intermediate arithmetic and ranking use float64; outputs round once to float32. Near
 *   prominence ties or the inclusive suppression boundary, float32 GPU decisions can differ.
 * - Suppression honors the round budget (default 16). Within a round, decisions are visited in
 *   source-row order; GPU invocation scheduling can decide a different subset at a small budget.
 * - GPU radius limits are dropped; window and radius may exceed 4096.
 * - Unused compact ID rows are zero-filled; only the first count rows are defined by the GPU.
 */
export function computeProfilePeaksOnCPU(
  values: Float32Array,
  offsets: Uint32Array,
  parameters: Float32Array,
  options: CPUProfilePeaksOptions,
  validity?: Uint32Array
): CPUProfilePeaksResult {
  if (validity && validity.length !== values.length)
    throw new Error('validity must match values length');
  if (parameters.length < GPU_PROFILE_PEAKS_PARAMETER_LENGTH)
    throw new Error('parameters is too short');
  if (
    offsets.length < 1 ||
    offsets.some(
      (offset, index) => offset > values.length || (index > 0 && offset < offsets[index - 1])
    )
  ) {
    throw new Error('offsets must be nondecreasing and within values length');
  }
  validateProfileOptions('computeProfilePeaksOnCPU', options);
  const minimumSide = options.minimumSide ?? 4;
  const radius = options.suppressionRadius ?? Math.max(2, Math.round(options.window / 4));
  const [minimumProminence] = parameters;
  const result: CPUProfilePeaksResult = {
    prominence: new Float32Array(values.length).fill(Number.NaN),
    refinedIndex: new Float32Array(values.length).fill(Number.NaN),
    refinedValue: new Float32Array(values.length).fill(Number.NaN),
    peakMask: new Uint32Array(values.length),
    ids: new Uint32Array(options.capacity),
    count: new Uint32Array(1),
    requiredCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    candidateOverflow: new Uint32Array(1),
    converged: new Uint32Array([1]),
    iterationLimitReached: new Uint32Array(1)
  };
  const keptRows: number[] = [];
  for (let profile = 0; profile + 1 < offsets.length; profile++) {
    const start = offsets[profile];
    const length = offsets[profile + 1] - start;
    const isValid = (local: number) =>
      local >= 0 &&
      local < length &&
      (!validity || validity[start + local] !== 0) &&
      Number.isFinite(values[start + local]);
    const getNeighbor = (local: number, displacement: number) =>
      options.wrap ? (((local + displacement) % length) + length) % length : local + displacement;
    const candidates: {local: number; prominence: number; position: number; value: number}[] = [];
    for (let local = options.wrap ? 0 : 1; local < (options.wrap ? length : length - 1); local++) {
      if (!isValid(local)) continue;
      const height = values[start + local];
      let maximum = true;
      for (let displacement = -2; displacement <= 2; displacement++) {
        const neighbor = getNeighbor(local, displacement);
        if (neighbor === local || !isValid(neighbor)) continue;
        const other = values[start + neighbor];
        if (other > height || (displacement < 0 && other === height)) maximum = false;
      }
      if (!maximum) continue;
      const sides = [-1, 1].map(direction => {
        let lowest = height;
        let validCount = 0;
        const limit = options.wrap
          ? Math.min(options.window, length - 1)
          : Math.min(options.window, length);
        for (let step = 1; step <= limit; step++) {
          const neighbor = getNeighbor(local, direction * step);
          if (!isValid(neighbor)) break;
          validCount++;
          const other = values[start + neighbor];
          if (other > height) break;
          lowest = Math.min(lowest, other);
        }
        return {lowest, validCount};
      });
      if (sides.some(side => side.validCount < minimumSide)) continue;
      const prominence = height - Math.max(sides[0].lowest, sides[1].lowest);
      if (prominence < minimumProminence) continue;
      const previous = getNeighbor(local, -1);
      const next = getNeighbor(local, 1);
      let displacement = 0;
      let value = height;
      if (isValid(previous) && isValid(next)) {
        const difference = values[start + previous] - values[start + next];
        const denominator = values[start + previous] - 2 * height + values[start + next];
        if (denominator < 0) {
          displacement = Math.max(-0.5, Math.min(0.5, (0.5 * difference) / denominator));
          value -= 0.25 * difference * displacement;
        }
      }
      candidates.push({local, prominence, position: local + displacement, value});
    }
    const states = new Uint8Array(candidates.length).fill(1);
    for (let round = 0; round < (options.maximumSuppressionIterations ?? 16); round++) {
      let changed = false;
      for (const [index, candidate] of candidates.entries()) {
        if (states[index] !== 1) continue;
        let pending = false;
        let suppressed = false;
        for (const [otherIndex, other] of candidates.entries()) {
          if (
            !(
              other.prominence > candidate.prominence ||
              (other.prominence === candidate.prominence && other.local < candidate.local)
            )
          )
            continue;
          const distance = Math.abs(other.position - candidate.position);
          if ((options.wrap ? Math.min(distance, length - distance) : distance) > radius) continue;
          if (states[otherIndex] === 2) {
            suppressed = true;
            break;
          }
          pending ||= states[otherIndex] === 1;
        }
        if (suppressed || !pending) {
          states[index] = suppressed ? 3 : 2;
          changed = true;
        }
      }
      if (!changed) break;
    }
    for (const [index, candidate] of candidates.entries()) {
      const row = start + candidate.local;
      if (states[index] === 1) {
        result.peakMask[row] = 0xffffffff;
        result.prominence[row] = result.refinedIndex[row] = result.refinedValue[row] = Infinity;
        result.converged[0] = 0;
        result.iterationLimitReached[0] = result.candidateOverflow[0] = 1;
      }
      if (states[index] !== 2) continue;
      result.peakMask[row] = 1;
      result.prominence[row] = candidate.prominence;
      let refinedIndex = candidate.position;
      if (options.wrap && refinedIndex < 0) refinedIndex += length;
      if (options.wrap && refinedIndex >= length) refinedIndex -= length;
      result.refinedIndex[row] = refinedIndex;
      // Final float32 storage can round a wrapped position back up to the profile length.
      if (options.wrap && result.refinedIndex[row] >= length) result.refinedIndex[row] = 0;
      result.refinedValue[row] = candidate.value;
      keptRows.push(row);
    }
  }
  keptRows.sort((first, second) => first - second);
  result.ids.set(keptRows.slice(0, options.capacity));
  result.count[0] = Math.min(keptRows.length, options.capacity);
  result.requiredCount[0] = keptRows.length;
  result.overflow[0] = Number(keptRows.length > options.capacity);
  return result;
}

function validateProfileOptions(id: string, options: CPUProfilePeaksOptions): void {
  validateCapacity(id, 'window', options.window, 1, Number.MAX_SAFE_INTEGER);
  validateCapacity(id, 'minimumSide', options.minimumSide ?? 4, 0, Number.MAX_SAFE_INTEGER);
  const radius = options.suppressionRadius ?? Math.max(2, Math.round(options.window / 4));
  validateFiniteNumber(id, 'suppressionRadius', radius);
  validateRequirement(id, radius >= 0, 'suppressionRadius must be at least 0');
  validateCapacity(id, 'capacity', options.capacity, 1);
  validateCapacity(
    id,
    'maximumSuppressionIterations',
    options.maximumSuppressionIterations ?? 16,
    0,
    Number.MAX_SAFE_INTEGER
  );
}

/** CPU twin with the same suppression budget and separate capacity/candidate status. */
export class CPUProfilePeaksOperation extends GPUCPUOperation {
  constructor(props: ProfilePeaksOperationProps) {
    props = snapshotProfilePeaksOperationProps(props);
    const operation = new GPUProfilePeaksOperation(props);
    const {id} = operation;
    const options = {...props, capacity: props.output.peaks?.capacity ?? props.sampleCount};
    validateCapacity(id, 'sampleCount', props.sampleCount, 1);
    validateProfileOptions(id, options);
    validateRequirement(
      id,
      Object.keys(operation.outputs).length > 0,
      'output must contain at least one port'
    );
    if (props.maximumSuppressionIterations !== undefined)
      validateCapacity(
        id,
        'maximumSuppressionIterations',
        props.maximumSuppressionIterations,
        0,
        Number.MAX_SAFE_INTEGER
      );
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
        const values = getCPUOperationFloat32Rows(id, inputs, 'values');
        if (values.length !== props.sampleCount)
          throw new Error(`${id} values must match sampleCount`);
        const result = computeProfilePeaksOnCPU(
          values,
          getCPUOperationUint32Rows(id, inputs, 'offsets'),
          getCPUOperationFloat32Rows(id, inputs, 'parameters'),
          options,
          props.validity !== undefined
            ? getCPUOperationUint32Rows(id, inputs, 'validity')
            : undefined
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUProfilePeaksResult]
          ])
        );
      }
    });
  }
}
