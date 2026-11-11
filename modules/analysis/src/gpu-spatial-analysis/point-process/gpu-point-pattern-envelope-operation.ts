// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUPointPatternEnvelope} from './gpu-point-pattern-envelope';

/** Shared graph-value properties. All properties are compile-time. */
export type PointPatternEnvelopeOperationProps = {
  id?: string;
  observed: string;
  simulations: string;
  reference: string;
  integrationWeights: string;
  curveRowCount: number;
  simulationCount: number;
  rank?: number;
  statistic?: 'maximum-deviation' | 'integrated-squared-deviation' | 'extreme-rank';
  significanceLevel?: number;
  nullModel: string;
  fittedParameters: 'none' | 'fixed-external' | 'refit-each-replicate';
  output: {
    lower: string;
    upper: string;
    statistics: string;
    status: string;
    pValueInterval?: string;
  };
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUPointPatternEnvelopeOperation extends GPUContributorOperation {
  constructor(props: PointPatternEnvelopeOperationProps) {
    super({
      id: props.id ?? 'point-pattern-envelope',
      inputs: {
        observed: props.observed,
        simulations: props.simulations,
        reference: props.reference,
        integrationWeights: props.integrationWeights
      },
      outputs: {
        lower: {value: props.output.lower, format: 'float32', length: props.curveRowCount},
        upper: {value: props.output.upper, format: 'float32', length: props.curveRowCount},
        statistics: {value: props.output.statistics, format: 'float32', length: 4},
        status: {value: props.output.status, format: 'uint32', length: 1},
        ...(props.output.pValueInterval === undefined
          ? {}
          : {
              pValueInterval: {
                value: props.output.pValueInterval,
                format: 'float32' as const,
                length: 2
              }
            })
      },
      createContributor: getView =>
        new GPUPointPatternEnvelope({
          ...props,
          observed: getView<'float32'>('observed'),
          simulations: getView<'float32'>('simulations'),
          reference: getView<'float32'>('reference'),
          integrationWeights: getView<'float32'>('integrationWeights'),
          output: {
            lower: getView<'float32'>('lower'),
            upper: getView<'float32'>('upper'),
            statistics: getView<'float32'>('statistics'),
            status: getView<'uint32'>('status'),
            pValueInterval:
              props.output.pValueInterval === undefined
                ? undefined
                : getView<'float32'>('pValueInterval')
          }
        })
    });
  }
}
