// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUEnsembleScores} from './gpu-ensemble-scores';

/** Shared graph-value props. Counts, fair and variogramOrder are compile-time. */
export type EnsembleScoresOperationProps = {
  id?: string;
  ensemble: string;
  observations: string;
  memberCount: number;
  dimension: number;
  fair?: boolean;
  variogramOrder?: number;
  variogramPairs?: string;
  variogramWeights?: string;
  output: {crps?: string; energyScore?: string; variogramScore?: string};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUEnsembleScoresOperation extends GPUContributorOperation {
  constructor(props: EnsembleScoresOperationProps) {
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const [port, value] of Object.entries(props.output)) {
      if (value !== undefined)
        outputs[port] = {value, format: 'float32', length: port === 'crps' ? props.dimension : 1};
    }
    super({
      id: props.id ?? 'ensemble-scores',
      inputs: {
        ensemble: props.ensemble,
        observations: props.observations,
        ...(props.variogramPairs !== undefined ? {variogramPairs: props.variogramPairs} : {}),
        ...(props.variogramWeights !== undefined ? {variogramWeights: props.variogramWeights} : {})
      },
      outputs,
      createContributor: getView =>
        new GPUEnsembleScores({
          id: props.id,
          memberCount: props.memberCount,
          dimension: props.dimension,
          fair: props.fair,
          variogramOrder: props.variogramOrder,
          ensemble: getView<'float32'>('ensemble'),
          observations: getView<'float32'>('observations'),
          variogramPairs:
            props.variogramPairs !== undefined ? getView<'uint32x2'>('variogramPairs') : undefined,
          variogramWeights:
            props.variogramWeights !== undefined
              ? getView<'float32'>('variogramWeights')
              : undefined,
          output: {
            crps: props.output.crps !== undefined ? getView<'float32'>('crps') : undefined,
            energyScore:
              props.output.energyScore !== undefined
                ? getView<'float32'>('energyScore')
                : undefined,
            variogramScore:
              props.output.variogramScore !== undefined
                ? getView<'float32'>('variogramScore')
                : undefined
          }
        })
    });
  }
}
