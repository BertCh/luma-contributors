// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSamplingSeed} from './sampling-seed';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPhilox4x32, PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUSpatialBalancedSampleOperation,
  snapshotBalancedSampleOperation,
  type SpatialBalancedSampleOperationProps
} from './gpu-spatial-balanced-sample-operation';

/** Stratum IDs share a counter word with the four-bit hierarchy level. */
const MAXIMUM_STRATUM_ID_EXCLUSIVE = 16777216;
/** Inclusion masses and their total must be exact float32 integers in this design. */
const MAXIMUM_TICKET_TOTAL = 16777216;

/** Packed finite-frame input; IDs are returned verbatim, with row order breaking hierarchy ties. */
export type CPUSpatialBalancedSampleInput = {
  positions: Float32Array;
  frameIds: Uint32Array;
  strata?: Uint32Array;
  inclusionTickets?: Uint32Array;
  parameters?: Uint32Array;
};

/** Numeric design properties, shared with the contributor. */
export type CPUSpatialBalancedSampleOptions = Pick<
  SpatialBalancedSampleOperationProps,
  'id' | 'spatialContext' | 'allocations' | 'bounds' | 'seed' | 'replicate'
>;

/** Fixed-size design records and [selectedCount, status] per allocation. */
export type CPUSpatialBalancedSampleResult = {
  frameIds: Uint32Array;
  inclusionProbabilities: Float32Array;
  designWeights: Float32Array;
  status: Uint32Array;
};

function validateOptions(
  options: CPUSpatialBalancedSampleOptions,
  rowCount: number,
  hasStrata: boolean,
  resident = false
): void {
  const id = options.id ?? 'spatial-balanced-sample';
  resolveGPUSpatialContext(id, options.spatialContext);
  validateCapacity(id, 'rowCount', rowCount);
  validateCapacity(id, 'allocations', options.allocations.length);
  const stratumIds = new Set<number>();
  let sampleCount = 0;
  for (const allocation of options.allocations) {
    validateCapacity(
      id,
      'allocation stratumId',
      allocation.stratumId,
      0,
      MAXIMUM_STRATUM_ID_EXCLUSIVE - 1
    );
    validateRequirement(
      id,
      !stratumIds.has(allocation.stratumId),
      'allocation stratumId must be unique'
    );
    stratumIds.add(allocation.stratumId);
    validateCapacity(id, 'allocation sampleCount', allocation.sampleCount, 1, rowCount);
    sampleCount += allocation.sampleCount;
  }
  validateRequirement(
    id,
    sampleCount <= rowCount,
    'allocations must not sample more than rowCount'
  );
  validateRequirement(
    id,
    hasStrata || (options.allocations.length === 1 && options.allocations[0].stratumId === 0),
    'without strata, allocations must be the single stratum 0'
  );
  validateRequirement(
    id,
    options.bounds.length === 4 &&
      options.bounds.every(value => Number.isFinite(Math.fround(value))),
    'bounds must contain four finite float32 values'
  );
  validateRequirement(
    id,
    Math.fround(options.bounds[2]) > Math.fround(options.bounds[0]) &&
      Math.fround(options.bounds[3]) > Math.fround(options.bounds[1]),
    'bounds must have a positive extent on both axes'
  );
  validateSamplingSeed(id, options.seed, resident);
  validateCapacity(id, 'replicate', options.replicate ?? 0, 0);
}

/**
 * Computes the stratified randomized-quadtree circular-systematic design in float64. All rows
 * must be finite, enclosed by bounds and assigned to an allocation; otherwise every stratum is
 * invalidated. Invalid ticket totals invalidate only their own stratum. Missing output IDs are
 * 0xffffffff and their probabilities and weights are NaN. IDs need not be unique for execution;
 * uniqueness is the caller's design responsibility, as on GPU.
 *
 * What differs from the GPU contributor:
 * - Float64 normalization can assign a different hierarchy cell at a float32 cell boundary;
 *   Philox words, rotations, row-index tie breaking and integer selection are otherwise identical.
 * - Float64 probabilities and reciprocal weights are rounded independently once at output;
 *   compare positive floats with relative tolerance 2e-6.
 * - The GPU-only 4096-row and 256-stratum caps and 128-draw rejection budget are removed.
 *   Rejection continues until accepted, so RNG-exhaustion status 2 is never emitted.
 * - Bounds still round to float32, and the 24-bit stratum IDs and inclusion-ticket total remain
 *   limits of the design. Non-finite positions produce invalid-frame status 1, as on GPU.
 * - Integer sample-rank arithmetic does not wrap at 2^32 on larger CPU frames.
 *
 * @throws If positions, frameIds, strata, inclusionTickets or parameters lengths disagree.
 */
export function computeSpatialBalancedSampleOnCPU(
  input: CPUSpatialBalancedSampleInput,
  options: CPUSpatialBalancedSampleOptions
): CPUSpatialBalancedSampleResult {
  const {positions, frameIds, strata, inclusionTickets, parameters} = input;
  const rowCount = frameIds.length;
  if (positions.length !== rowCount * 2)
    throw new Error('positions length must equal frameIds length * 2');
  if (strata && strata.length !== rowCount)
    throw new Error('strata length must equal frameIds length');
  if (inclusionTickets && inclusionTickets.length !== rowCount)
    throw new Error('inclusionTickets length must equal frameIds length');
  if (parameters && parameters.length !== 2)
    throw new Error('parameters must contain [seed, replicate]');
  validateOptions(options, rowCount, strata !== undefined, parameters !== undefined);
  const [seed, replicate] = parameters ?? [options.seed!, options.replicate ?? 0];
  const drawWord = (first: number, second: number) =>
    getPhilox4x32(
      [first, second, replicate, 0],
      [seed, PHILOX_PURPOSES.spatialBalancedSampling.id]
    )[0];
  const bounds = options.bounds.map(Math.fround);
  const allocatedStrata = new Set(options.allocations.map(allocation => allocation.stratumId));
  const codes = new Uint32Array(rowCount);
  let validFrame = true;
  for (let row = 0; row < rowCount; row++) {
    const horizontal = positions[row * 2];
    const vertical = positions[row * 2 + 1];
    const stratum = strata?.[row] ?? 0;
    const valid =
      Number.isFinite(horizontal) &&
      Number.isFinite(vertical) &&
      horizontal >= bounds[0] &&
      horizontal <= bounds[2] &&
      vertical >= bounds[1] &&
      vertical <= bounds[3] &&
      allocatedStrata.has(stratum);
    validFrame &&= valid;
    if (!valid) continue;
    const horizontalCell = Math.floor(
      Math.min(1, Math.max(0, (horizontal - bounds[0]) / (bounds[2] - bounds[0]))) * 65535
    );
    const verticalCell = Math.floor(
      Math.min(1, Math.max(0, (vertical - bounds[1]) / (bounds[3] - bounds[1]))) * 65535
    );
    let prefix = 0;
    let code = 0;
    for (let level = 0; level < 16; level++) {
      const shift = 15 - level;
      const child = ((horizontalCell >>> shift) & 1) | (((verticalCell >>> shift) & 1) << 1);
      const rotation = drawWord(prefix, (stratum << 4) | level) & 3;
      code = ((code << 2) | ((child + rotation) & 3)) >>> 0;
      prefix = ((prefix << 2) | child) >>> 0;
    }
    codes[row] = code;
  }
  const sampleCount = options.allocations.reduce(
    (total, allocation) => total + allocation.sampleCount,
    0
  );
  const result: CPUSpatialBalancedSampleResult = {
    frameIds: new Uint32Array(sampleCount).fill(0xffffffff),
    inclusionProbabilities: new Float32Array(sampleCount).fill(NaN),
    designWeights: new Float32Array(sampleCount).fill(NaN),
    status: new Uint32Array(options.allocations.length * 2)
  };
  let outputOffset = 0;
  for (const [allocationIndex, allocation] of options.allocations.entries()) {
    const members: number[] = [];
    let ticketTotal = 0;
    let maximumTicket = 0;
    let valid = validFrame;
    for (let row = 0; row < rowCount; row++) {
      if ((strata?.[row] ?? 0) !== allocation.stratumId) continue;
      members.push(row);
      const ticket = inclusionTickets?.[row] ?? 1;
      ticketTotal += ticket;
      maximumTicket = Math.max(maximumTicket, ticket);
      valid &&= ticket > 0 && ticketTotal <= MAXIMUM_TICKET_TOTAL;
    }
    const step = ticketTotal / allocation.sampleCount;
    valid &&= members.length >= allocation.sampleCount;
    if (inclusionTickets) valid &&= Number.isInteger(step) && step > 0 && maximumTicket <= step;
    result.status[allocationIndex * 2 + 1] = valid ? 0 : 1;
    if (valid) {
      members.sort((first, second) => codes[first] - codes[second] || first - second);
      const drawLimit = inclusionTickets ? step : members.length;
      const threshold = (4294967296 - drawLimit) % drawLimit;
      let attempt = 0;
      let draw: number;
      do {
        draw = drawWord(allocation.stratumId, (0x80000000 | attempt++) >>> 0);
      } while (draw < threshold);
      const shift = draw % drawLimit;
      const firstTickets = new Float64Array(members.length);
      let consumed = 0;
      for (const [index, row] of members.entries()) {
        firstTickets[index] = consumed;
        consumed += inclusionTickets?.[row] ?? 1;
      }
      for (let sample = 0; sample < allocation.sampleCount; sample++) {
        const rank = inclusionTickets
          ? shift + sample * step
          : (shift + Math.floor((sample * members.length) / allocation.sampleCount)) %
            members.length;
        const memberIndex = firstTickets.findIndex(
          (first, index) =>
            first <= rank && rank < first + (inclusionTickets?.[members[index]] ?? 1)
        );
        const row = members[memberIndex];
        const probability = inclusionTickets
          ? inclusionTickets[row] / step
          : allocation.sampleCount / members.length;
        result.frameIds[outputOffset + sample] = frameIds[row];
        result.inclusionProbabilities[outputOffset + sample] = probability;
        result.designWeights[outputOffset + sample] = 1 / probability;
      }
      result.status[allocationIndex * 2] = allocation.sampleCount;
    }
    outputOffset += allocation.sampleCount;
  }
  return result;
}

/** CPU twin; reference-tier GPU work caps are dropped, with identical graph values. */
export class CPUSpatialBalancedSampleOperation extends GPUCPUOperation {
  constructor(props: SpatialBalancedSampleOperationProps) {
    props = snapshotBalancedSampleOperation(props);
    const id = props.id ?? 'spatial-balanced-sample';
    validateOptions(
      props,
      props.rowCount,
      props.strata !== undefined,
      props.parameters !== undefined
    );
    const operation = new GPUSpatialBalancedSampleOperation(props);
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
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must equal rowCount * 2`);
        const result = computeSpatialBalancedSampleOnCPU(
          {
            positions,
            frameIds: getCPUOperationUint32Rows(id, inputs, 'frameIds'),
            strata:
              props.strata !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'strata')
                : undefined,
            inclusionTickets:
              props.inclusionTickets !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'inclusionTickets')
                : undefined,
            parameters:
              props.parameters !== undefined
                ? getCPUOperationUint32Rows(id, inputs, 'parameters')
                : undefined
          },
          props
        );
        return Object.fromEntries(
          Object.entries(props.output).map(([port, value]) => [
            value,
            result[port as keyof CPUSpatialBalancedSampleResult]
          ])
        );
      }
    });
  }
}
