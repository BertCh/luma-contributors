// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSpatialWeightsValidation} from './gpu-spatial-weights-validation';
import {snapshotWeightsValidationOperationProps} from './weights-construction-snapshot';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveSpatialWeightsValidationOptions,
  type SpatialWeightsValidationOptions
} from './spatial-weights-validation-options';

/** Whole, packed CSR graph values, shared by the GPU validator and its CPU twin. */
export type SpatialWeightsValidationOperationProps = SpatialWeightsValidationOptions & {
  id?: string;
  weights: {offsets: string; neighbors: string; weights: string};
  /** offsets length minus one; explicitly carried for the optional flags output shape. */
  rowCount: number;
  output: {flags?: string; status: {invalidCount: string}};
};

/**
 * Operation form for whole CSR only. Chunked CSR is unsupported by the mixed executor.
 * Optional distances and input metadata/status are omitted: no validation kernel reads them.
 * Graph input contents are per-frame; all props are compile-time.
 */
export class GPUSpatialWeightsValidationOperation extends GPUContributorOperation {
  constructor(props: SpatialWeightsValidationOperationProps) {
    const {columnCount, diagonal} = props;
    props = Object.freeze({
      ...snapshotWeightsValidationOperationProps(props),
      columnCount,
      diagonal
    });
    const id = props.id ?? 'spatial-weights-validation';
    const {rowCount} = props;
    const outputFlags = props.output.flags;
    validateCapacity(id, 'rowCount', rowCount, 1, 0xfffffffe);
    const options = resolveSpatialWeightsValidationOptions(id, rowCount, props);
    super({
      id,
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights
      },
      outputs: {
        invalidCount: {value: props.output.status.invalidCount, format: 'uint32', length: 1},
        ...(outputFlags
          ? {flags: {value: outputFlags, format: 'uint32' as const, length: rowCount}}
          : {})
      },
      createContributor: getView => {
        const offsets = getView<'uint32'>('offsets');
        validateRequirement(
          id,
          offsets.length === rowCount + 1,
          'offsets length must equal rowCount + 1'
        );
        return new GPUSpatialWeightsValidation({
          id,
          ...options,
          weights: {
            offsets,
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          output: {
            flags: outputFlags ? getView<'uint32'>('flags') : undefined,
            status: {invalidCount: getView<'uint32'>('invalidCount')}
          }
        });
      }
    });
  }
}
