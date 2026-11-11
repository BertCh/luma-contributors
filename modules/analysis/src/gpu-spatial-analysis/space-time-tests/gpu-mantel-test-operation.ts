// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUMantelTest, type GPUMantelTransform} from './gpu-mantel-test';
import {GPU_SPACE_TIME_SUMMARY_LENGTH} from './space-time-parameters';

/** Shared graph-value properties; parameters retain getGPUSpaceTimeParameterValues layout. */
export type MantelTestOperationProps = {
  id?: string;
  rowCount: number;
  /** Weights are shape-checked by the contributor but do not enter the Mantel statistic. */
  pairs: {offsets: string; neighbors: string; weights: string; distances: string};
  times: string;
  parameters: string;
  maximumPermutations: number;
  stopAfterExceedances?: number;
  distanceTransform?: GPUMantelTransform;
  timeTransform?: GPUMantelTransform;
  output: {statistics: string; summary: string; permutationCount?: string};
};

/**
 * Vector operation form. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Optional pair metadata/status ports are omitted: GPUMantelTest never reads them.
 */
export class GPUMantelTestOperation extends GPUContributorOperation {
  constructor(props: MantelTestOperationProps) {
    super({
      id: props.id ?? 'mantel-test',
      inputs: {...props.pairs, times: props.times, parameters: props.parameters},
      outputs: {
        statistics: {
          value: props.output.statistics,
          format: 'float32',
          length: props.maximumPermutations + 1
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_SPACE_TIME_SUMMARY_LENGTH
        },
        ...(props.output.permutationCount === undefined
          ? {}
          : {
              permutationCount: {
                value: props.output.permutationCount,
                format: 'uint32' as const,
                length: 1
              }
            })
      },
      createContributor: getView =>
        new GPUMantelTest({
          ...props,
          pairs: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            distances: getView<'float32'>('distances')
          },
          times: getView<'float32'>('times'),
          parameters: getView<'uint32'>('parameters'),
          output: {
            statistics: getView<'float32'>('statistics'),
            summary: getView<'float32'>('summary'),
            permutationCount:
              props.output.permutationCount === undefined
                ? undefined
                : getView<'uint32'>('permutationCount')
          }
        })
    });
  }
}
