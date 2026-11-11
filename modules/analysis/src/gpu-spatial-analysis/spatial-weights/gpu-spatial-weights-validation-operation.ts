// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUSpatialWeightsValidation} from './gpu-spatial-weights-validation';

/** Whole, packed CSR graph values, shared by the GPU validator and its CPU twin. */
export type SpatialWeightsValidationOperationProps = {
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
    const id = props.id ?? 'spatial-weights-validation';
    super({
      id,
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights
      },
      outputs: {
        invalidCount: {value: props.output.status.invalidCount, format: 'uint32', length: 1},
        ...(props.output.flags
          ? {flags: {value: props.output.flags, format: 'uint32' as const, length: props.rowCount}}
          : {})
      },
      createContributor: getView =>
        new GPUSpatialWeightsValidation({
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          output: {
            flags: props.output.flags ? getView<'uint32'>('flags') : undefined,
            status: {invalidCount: getView<'uint32'>('invalidCount')}
          }
        })
    });
  }
}
