// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {GPU_SPATIAL_WEIGHTS_VIOLATION} from './gpu-spatial-weights-validation';
import type {SpatialWeightsValidationOperationProps} from './gpu-spatial-weights-validation-operation';

/**
 * Checks whole CSR and returns the contributor's exact per-row violation bits and invalid count.
 * Only slots in [begin, min(end, capacity)) are inspected, even for malformed offsets. Offset
 * violations are attributed to row 0, the decreasing row, or the final row exactly as on the GPU.
 *
 * What differs from the GPU contributor:
 * - Zero neighbor capacity is accepted: offsets=[0,0] with empty neighbors and weights returns
 *   flags=[0] and invalidCount=[0]. The GPU contributor requires positive neighbor capacity.
 * - Chunked CSR is omitted from the operation form.
 *
 * @throws If offsets has fewer than two entries or weights and neighbors have different lengths.
 * Invalid CSR contents are reported in flags, not thrown.
 */
export function computeSpatialWeightsValidationOnCPU(
  offsets: Uint32Array,
  neighbors: Uint32Array,
  weights: Float32Array
): {flags: Uint32Array; invalidCount: Uint32Array} {
  if (offsets.length < 2)
    throw new Error(
      'computeSpatialWeightsValidationOnCPU offsets must contain at least two entries'
    );
  if (weights.length !== neighbors.length)
    throw new Error(
      'computeSpatialWeightsValidationOnCPU weights length must equal neighbors length'
    );
  const rowCount = offsets.length - 1;
  const capacity = neighbors.length;
  const flags = new Uint32Array(rowCount);
  const bits = GPU_SPATIAL_WEIGHTS_VIOLATION;
  let invalidCount = 0;
  for (let row = 0; row < rowCount; row++) {
    const begin = offsets[row];
    const end = offsets[row + 1];
    let rowFlags = 0;
    if ((row === 0 && begin !== 0) || begin > end || (row + 1 === rowCount && end > capacity))
      rowFlags |= bits.offsets;
    let previous = 0;
    for (let slot = begin; slot < Math.min(end, capacity); slot++) {
      const neighbor = neighbors[slot];
      const weight = weights[slot];
      if (neighbor >= rowCount) rowFlags |= bits.neighborRange;
      if (neighbor === row) rowFlags |= bits.self;
      if (slot > begin && neighbor <= previous) rowFlags |= bits.order;
      if (!Number.isFinite(weight) || weight < 0) rowFlags |= bits.weight;
      previous = neighbor;
    }
    flags[row] = rowFlags;
    if (rowFlags !== 0) invalidCount++;
  }
  return {flags, invalidCount: Uint32Array.of(invalidCount)};
}

/** CPU twin for whole packed CSR. Optional flags are computed internally when omitted. */
export class CPUSpatialWeightsValidationOperation extends GPUCPUOperation {
  constructor(props: SpatialWeightsValidationOperationProps) {
    const id = props.id ?? 'spatial-weights-validation';
    validateCapacity(id, 'rowCount', props.rowCount, 1, 0xfffffffe);
    super({
      id,
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights
      },
      outputs: {
        [props.output.status.invalidCount]: gpuGraph.vector('uint32', {length: 1}),
        ...(props.output.flags
          ? {[props.output.flags]: gpuGraph.vector('uint32', {length: props.rowCount})}
          : {})
      },
      execute({inputs}) {
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        if (offsets.length !== props.rowCount + 1)
          throw new Error(`${id} offsets length must equal rowCount + 1`);
        const result = computeSpatialWeightsValidationOnCPU(
          offsets,
          getCPUOperationUint32Rows(id, inputs, 'neighbors'),
          getCPUOperationFloat32Rows(id, inputs, 'weights')
        );
        return {
          [props.output.status.invalidCount]: result.invalidCount,
          ...(props.output.flags ? {[props.output.flags]: result.flags} : {})
        };
      }
    });
  }
}
