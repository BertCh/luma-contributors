// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUSpatialTwoStageLeastSquares,
  type GPUSpatialTwoStageLeastSquaresProps
} from './gpu-spatial-two-stage-least-squares';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
} from './spatial-two-stage-least-squares-parameters';

/** Graph names replace every view; rowCount explicitly sizes the residual output. */
export type SpatialTwoStageLeastSquaresOperationProps = Pick<
  GPUSpatialTwoStageLeastSquaresProps,
  'id' | 'predictorCount' | 'tileRowCount' | 'instrumentOrder'
> & {
  rowCount: number;
  weights: {offsets: string; neighbors: string; weights: string};
  predictors: string;
  response: string;
  robust?: {
    method: 'HC0' | 'HC1' | 'HAC';
    kernel?: {offsets: string; neighbors: string; weights: string; overflow?: string};
    output: {covariance: string; status: string};
  };
  output: {status: {code: string}; table: string; summary: string; residuals?: string};
};

/** Whole-CSR GPU IV operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: contents of the input graph values.
 * Unused model weights metadata is omitted; HAC overflow is a separate optional input.
 */
export class GPUSpatialTwoStageLeastSquaresOperation extends GPUContributorOperation {
  constructor(props: SpatialTwoStageLeastSquaresOperationProps) {
    const id = props.id ?? 'spatial-two-stage-least-squares';
    const robust = props.robust;
    const coefficientCount = props.predictorCount + 2;
    super({
      id,
      inputs: {
        ...props.weights,
        predictors: props.predictors,
        response: props.response,
        ...(robust?.kernel
          ? {
              kernelOffsets: robust.kernel.offsets,
              kernelNeighbors: robust.kernel.neighbors,
              kernelWeights: robust.kernel.weights,
              ...(robust.kernel.overflow ? {kernelOverflow: robust.kernel.overflow} : {})
            }
          : {})
      },
      outputs: {
        code: {value: props.output.status.code, format: 'uint32', length: 1},
        table: {
          value: props.output.table,
          format: 'float32',
          length: coefficientCount * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
        },
        ...(props.output.residuals
          ? {
              residuals: {
                value: props.output.residuals,
                format: 'float32' as const,
                length: props.rowCount
              }
            }
          : {}),
        ...(robust
          ? {
              robustCovariance: {
                value: robust.output.covariance,
                format: 'float32' as const,
                length: coefficientCount ** 2
              },
              robustStatus: {value: robust.output.status, format: 'uint32' as const, length: 1}
            }
          : {})
      },
      createContributor: getView =>
        new GPUSpatialTwoStageLeastSquares({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
          },
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          robust: robust
            ? {
                method: robust.method,
                kernel: robust.kernel
                  ? {
                      offsets: getView<'uint32'>('kernelOffsets'),
                      neighbors: getView<'uint32'>('kernelNeighbors'),
                      weights: getView<'float32'>('kernelWeights'),
                      status: robust.kernel.overflow
                        ? {overflow: getView<'uint32'>('kernelOverflow')}
                        : undefined
                    }
                  : undefined,
                output: {
                  covariance: getView<'float32'>('robustCovariance'),
                  status: getView<'uint32'>('robustStatus')
                }
              }
            : undefined,
          output: {
            status: {code: getView<'uint32'>('code')},
            table: getView<'float32'>('table'),
            summary: getView<'float32'>('summary'),
            residuals: props.output.residuals ? getView<'float32'>('residuals') : undefined
          }
        })
    });
  }
}
