// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  captureEnsembleScoreOutputs,
  resolveEnsembleScoresEstimator,
  type EnsembleScore,
  type EnsembleScoresEstimatorOptions,
  type EnsembleScoresOutputSelection
} from './ensemble-scores-options';
import {GPUEnsembleScores} from './gpu-ensemble-scores';

/** Shared graph-value props. Counts, estimator and variogramOrder are captured at construction. */
export type EnsembleScoresOperationProps<
  Output extends EnsembleScoresOutputSelection = EnsembleScoresOutputSelection
> = EnsembleScoresEstimatorOptions & {
  id?: string;
  ensemble: string;
  observations: string;
  memberCount: number;
  dimension: number;
  variogramOrder?: number;
  variogramPairs?: string;
  variogramWeights?: string;
  output: Output;
};

/** Props for the GPU operation, retaining the shared props name for compatibility. */
export type GPUEnsembleScoresOperationProps<
  Output extends EnsembleScoresOutputSelection = EnsembleScoresOutputSelection
> = EnsembleScoresOperationProps<Output>;

/** Descriptors preserve the selected keys and literal graph destinations. */
type EnsembleScoreOutputDescriptor<Value> = {
  value: Exclude<Value, undefined> & string;
  format: 'float32';
  length: number;
};

type DeclaredEnsembleScoreOutputKeys<Output> = keyof {
  [Key in keyof Output as string extends Key
    ? never
    : number extends Key
      ? never
      : symbol extends Key
        ? never
        : Key]: unknown;
};

export type EnsembleScoresOperationOutputs<Output extends EnsembleScoresOutputSelection> =
  Output extends unknown
    ? {
        [Port in keyof Output &
          EnsembleScore as Port extends DeclaredEnsembleScoreOutputKeys<Output>
          ? undefined extends Output[Port]
            ? never
            : Port
          : never]: EnsembleScoreOutputDescriptor<Output[Port]>;
      } & {
        [Port in keyof Output & EnsembleScore as Output[Port] extends undefined
          ? never
          : Port extends DeclaredEnsembleScoreOutputKeys<Output>
            ? undefined extends Output[Port]
              ? Port
              : never
            : Port]?: EnsembleScoreOutputDescriptor<Output[Port]>;
      }
    : never;

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUEnsembleScoresOperation<
  const Output extends EnsembleScoresOutputSelection = EnsembleScoresOutputSelection
> extends GPUContributorOperation<EnsembleScoresOperationOutputs<Output>> {
  constructor(props: EnsembleScoresOperationProps<Output>) {
    const {estimator: requestedEstimator} = props;
    const estimator = resolveEnsembleScoresEstimator(
      props.id ?? 'ensemble-scores',
      'fair' in props ? props : {estimator: requestedEstimator}
    );
    props = {
      id: props.id,
      ensemble: props.ensemble,
      observations: props.observations,
      memberCount: props.memberCount,
      dimension: props.dimension,
      estimator: requestedEstimator,
      variogramOrder: props.variogramOrder,
      variogramPairs: props.variogramPairs,
      variogramWeights: props.variogramWeights,
      output: props.output
    };
    const selected = captureEnsembleScoreOutputs(props.id ?? 'ensemble-scores', props.output);
    const id = props.id ?? 'ensemble-scores';
    validateCapacity(id, 'memberCount', props.memberCount, estimator === 'fair' ? 2 : 1);
    validateCapacity(id, 'dimension', props.dimension, 1);
    validateRequirement(
      id,
      selected.some(([port]) => port === 'variogramScore') === (props.variogramPairs !== undefined),
      'output.variogramScore and variogramPairs must go together'
    );
    validateRequirement(
      id,
      props.variogramWeights === undefined || props.variogramPairs !== undefined,
      'variogramWeights requires variogramPairs'
    );
    const descriptors = selected.map(
      ([port, value]) =>
        [
          port,
          {
            value,
            format: 'float32' as const,
            length: port === 'crps' ? props.dimension : 1
          }
        ] as const
    );
    // The validated selection establishes these keys; Object.fromEntries cannot preserve a
    // generic map or literal destination names itself. The bridge freezes each descriptor.
    const outputs = Object.fromEntries(descriptors) as EnsembleScoresOperationOutputs<Output>;
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
          estimator,
          variogramOrder: props.variogramOrder,
          ensemble: getView('ensemble', 'float32'),
          observations: getView('observations', 'float32'),
          variogramPairs:
            props.variogramPairs !== undefined ? getView('variogramPairs', 'uint32x2') : undefined,
          variogramWeights:
            props.variogramWeights !== undefined
              ? getView('variogramWeights', 'float32')
              : undefined,
          output: Object.fromEntries(
            descriptors.map(([port, descriptor]) => [port, getView(port, descriptor.format)])
          )
        })
    });
  }
}
