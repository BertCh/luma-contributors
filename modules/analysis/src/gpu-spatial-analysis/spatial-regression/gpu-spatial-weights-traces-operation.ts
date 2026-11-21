// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRegressionOperationProps} from './regression-construction-snapshot';

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateCompleteRegressionWeights} from './cpu-regression-weights';
import {
  GPUSpatialWeightsTraces,
  type GPUSpatialWeightsTracesProps
} from './gpu-spatial-weights-traces';
import {GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_LENGTH} from './spatial-weights-traces-parameters';

/** Whole or chunked CSR; each chunk array is a separate packed graph value. */
export type SpatialWeightsTracesOperationProps = Pick<
  GPUSpatialWeightsTracesProps,
  | 'id'
  | 'maximumPower'
  | 'mode'
  | 'probeCount'
  | 'includeGramTraces'
  | 'tileRowCount'
  | 'probeBatchSize'
> & {
  rowCount: number;
  weights:
    | {offsets: string; neighbors: string; weights: string}
    | {
        rowCount: number;
        chunks: readonly {
          rowStart: number;
          rowCount: number;
          weights: {offsets: string; neighbors: string; weights: string};
        }[];
      };
  parameters?: string;
  output: {
    powerTraces: string;
    powerTraceStandardErrors?: string;
    powerSums: string;
    gramTraces?: string;
    infinityNorm?: string;
    status?: {approximation?: string};
  };
};

/** GPU trace operation; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSpatialWeightsTracesOperation extends GPUContributorOperation {
  constructor(props: SpatialWeightsTracesOperationProps) {
    props = snapshotRegressionOperationProps(props, ['approximation']);
    const id = props.id ?? 'spatial-weights-traces';
    validateCompleteRegressionWeights(id, props.weights);
    super({
      id,
      inputs: {
        ...('chunks' in props.weights
          ? Object.fromEntries(
              props.weights.chunks.flatMap((chunk, index) => [
                [`chunk${index}Offsets`, chunk.weights.offsets],
                [`chunk${index}Neighbors`, chunk.weights.neighbors],
                [`chunk${index}Weights`, chunk.weights.weights]
              ])
            )
          : props.weights),
        ...(props.parameters ? {parameters: props.parameters} : {})
      },
      outputs: {
        powerTraces: {
          value: props.output.powerTraces,
          format: 'float32',
          length: props.maximumPower
        },
        powerSums: {value: props.output.powerSums, format: 'float32', length: props.maximumPower},
        ...(props.output.powerTraceStandardErrors
          ? {
              powerTraceStandardErrors: {
                value: props.output.powerTraceStandardErrors,
                format: 'float32' as const,
                length: props.maximumPower
              }
            }
          : {}),
        ...(props.output.gramTraces
          ? {
              gramTraces: {
                value: props.output.gramTraces,
                format: 'float32' as const,
                length: GPU_SPATIAL_WEIGHTS_TRACES_GRAM_TRACE_LENGTH
              }
            }
          : {}),
        ...(props.output.infinityNorm
          ? {
              infinityNorm: {
                value: props.output.infinityNorm,
                format: 'float32' as const,
                length: 1
              }
            }
          : {}),
        ...(props.output.status?.approximation
          ? {
              approximation: {
                value: props.output.status.approximation,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUSpatialWeightsTraces({
          ...props,
          id,
          weights:
            'chunks' in props.weights
              ? {
                  rowCount: props.weights.rowCount,
                  chunks: props.weights.chunks.map((chunk, index) => ({
                    rowStart: chunk.rowStart,
                    rowCount: chunk.rowCount,
                    weights: {
                      offsets: getView<'uint32'>(`chunk${index}Offsets`),
                      neighbors: getView<'uint32'>(`chunk${index}Neighbors`),
                      weights: getView<'float32'>(`chunk${index}Weights`)
                    }
                  }))
                }
              : {
                  offsets: getView<'uint32'>('offsets'),
                  neighbors: getView<'uint32'>('neighbors'),
                  weights: getView<'float32'>('weights')
                },
          parameters: props.parameters ? getView<'uint32'>('parameters') : undefined,
          output: {
            powerTraces: getView<'float32'>('powerTraces'),
            powerSums: getView<'float32'>('powerSums'),
            powerTraceStandardErrors: props.output.powerTraceStandardErrors
              ? getView<'float32'>('powerTraceStandardErrors')
              : undefined,
            gramTraces: props.output.gramTraces ? getView<'float32'>('gramTraces') : undefined,
            infinityNorm: props.output.infinityNorm
              ? getView<'float32'>('infinityNorm')
              : undefined,
            status: props.output.status?.approximation
              ? {approximation: getView<'uint32'>('approximation')}
              : undefined
          }
        })
    });
  }
}
