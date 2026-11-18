// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {POPULATION_SUPPORT_ALIGNMENT_CODES} from './gpu-population-support-alignment';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUPopulationSupportAlignmentOperation,
  type PopulationSupportAlignmentOperationProps
} from './gpu-population-support-alignment-operation';

export type CPUPopulationSupportAlignmentInput = {
  /** Packed focal/complement pairs, one pair per source ID. Values are copied unchanged. */
  populations: Float32Array;
  sourceIds: Uint32Array;
  targetIds: Uint32Array;
};

export type CPUPopulationSupportAlignmentResult = {
  populations: Float32Array;
  /** 0 matched, 1 no source, 2 ambiguous source or target; missing takes precedence. */
  alignmentCode: Uint32Array;
  /** 1 for a unique source and target match, otherwise 0; usable as a downstream mask. */
  matchedMask: Uint32Array;
  /** [aligned, missingTargetRows, ambiguousTargetRows, excludedSourceRows]. */
  status: Uint32Array;
};

/**
 * Exact ID alignment in O(sourceCount + targetCount) expected time using maps. Population values
 * are copied, including NaNs, negative values and infinities; only IDs determine alignment.
 * All uint32 IDs are valid. Duplicate source IDs invalidate any target naming them. A repeated
 * target with no source has code 1; all other ambiguous targets have code 2. Excluded source
 * rows are counted individually, including duplicates whose ID never appears among the targets.
 *
 * What differs from the GPU contributor:
 * - GPU storage-binding limits and sorting scratch do not apply to CPU maps.
 * - The function checks source/population lengths before joining. Otherwise outputs, status
 *   words and row order are identical; no floating-point arithmetic is needed.
 */
export function computePopulationSupportAlignmentOnCPU(
  input: CPUPopulationSupportAlignmentInput
): CPUPopulationSupportAlignmentResult {
  const id = 'population-support-alignment';
  validateCapacity(id, 'sourceIds.length', input.sourceIds.length, 1);
  validateCapacity(id, 'targetIds.length', input.targetIds.length, 1);
  validateRequirement(
    id,
    input.populations.length === 2 * input.sourceIds.length,
    'populations must have 2 values per sourceIds row'
  );
  const sources = new Map<number, {row: number; count: number}>();
  input.sourceIds.forEach((identifier, row) => {
    const existing = sources.get(identifier);
    if (existing) existing.count++;
    else sources.set(identifier, {row, count: 1});
  });
  const targets = new Map<number, number>();
  for (const identifier of input.targetIds)
    targets.set(identifier, (targets.get(identifier) ?? 0) + 1);
  const populations = new Float32Array(2 * input.targetIds.length).fill(NaN);
  const alignmentCode = new Uint32Array(input.targetIds.length);
  const matchedMask = new Uint32Array(input.targetIds.length);
  let missingCount = 0;
  let duplicateCount = 0;
  input.targetIds.forEach((identifier, row) => {
    const source = sources.get(identifier);
    if (!source) {
      alignmentCode[row] = POPULATION_SUPPORT_ALIGNMENT_CODES.MISSING;
      missingCount++;
    } else if (source.count !== 1 || targets.get(identifier) !== 1) {
      alignmentCode[row] = POPULATION_SUPPORT_ALIGNMENT_CODES.AMBIGUOUS;
      duplicateCount++;
    } else {
      alignmentCode[row] = POPULATION_SUPPORT_ALIGNMENT_CODES.MATCHED;
      matchedMask[row] = 1;
      populations.set(input.populations.subarray(2 * source.row, 2 * source.row + 2), 2 * row);
    }
  });
  let excludedCount = 0;
  for (const identifier of input.sourceIds) if (!targets.has(identifier)) excludedCount++;
  return {
    populations,
    alignmentCode,
    matchedMask,
    status: Uint32Array.of(
      Number(missingCount === 0 && duplicateCount === 0 && excludedCount === 0),
      missingCount,
      duplicateCount,
      excludedCount
    )
  };
}

/** CPU support alignment stage for mixed GPUGraph programs; same ports as the GPU operation. */
export class CPUPopulationSupportAlignmentOperation extends GPUCPUOperation {
  constructor(props: PopulationSupportAlignmentOperationProps) {
    const id = props.id ?? 'population-support-alignment';
    validateCapacity(id, 'targetCount', props.targetCount, 1);
    const operation = new GPUPopulationSupportAlignmentOperation(props);
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
        const targetIds = getCPUOperationUint32Rows(id, inputs, 'targetIds');
        validateRequirement(
          id,
          targetIds.length === props.targetCount,
          'targetIds must have targetCount rows'
        );
        const result = computePopulationSupportAlignmentOnCPU({
          populations: getCPUOperationFloat32Rows(id, inputs, 'inputPopulations', 'float32x2'),
          sourceIds: getCPUOperationUint32Rows(id, inputs, 'sourceIds'),
          targetIds
        });
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUPopulationSupportAlignmentResult]
          ])
        );
      }
    });
  }
}
