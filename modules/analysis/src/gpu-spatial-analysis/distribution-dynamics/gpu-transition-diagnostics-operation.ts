// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationProps
} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUTransitionDiagnostics,
  GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS,
  GPU_TRANSITION_HOMOGENEITY_LENGTH,
  GPU_TRANSITION_STATE_SPARSITY_LENGTH
} from './gpu-transition-diagnostics';

/** Graph names in place of the contributor's views. Output lengths are fixed by class counts. */
export type TransitionDiagnosticsOperationProps = {
  id?: string;
  counts: string;
  classCount: number;
  conditionCount?: number;
  minimumTransitions?: number;
  output: {
    homogeneity?: string;
    steadyStates?: string;
    stateSparsity?: string;
    status?: {converged?: string};
  };
};

/**
 * Reads every declared scalar and nested port field exactly once, including inherited and
 * accessor fields, so validation and lowering see the same values. A spread would drop an
 * inherited `minimumTransitions` and let it pass as the default. @internal
 */
export function captureTransitionDiagnosticsOperationProps(
  props: TransitionDiagnosticsOperationProps
): TransitionDiagnosticsOperationProps {
  const {id, counts, classCount, conditionCount, minimumTransitions, output} = props;
  const {homogeneity, steadyStates, stateSparsity, status} = output;
  return {
    id,
    counts,
    classCount,
    conditionCount,
    minimumTransitions,
    output: {
      homogeneity,
      steadyStates,
      stateSparsity,
      status: status ? {converged: status.converged} : undefined
    }
  };
}

/** Graph-port validation shared independently of CPU/GPU numerical limits. @internal */
export function getTransitionDiagnosticsOperationProps(
  props: TransitionDiagnosticsOperationProps
): GPUContributorOperationProps {
  props = captureTransitionDiagnosticsOperationProps(props);
  const id = props.id ?? 'transition-diagnostics';
  const {output} = props;
  const conditionCount = props.conditionCount ?? 1;
  const chainCount = conditionCount > 1 ? conditionCount + 1 : 1;
  return {
    id,
    inputs: {counts: props.counts},
    outputs: {
      ...(output.homogeneity
        ? {
            homogeneity: {
              value: output.homogeneity,
              format: 'float32' as const,
              length: GPU_TRANSITION_HOMOGENEITY_LENGTH
            }
          }
        : {}),
      ...(output.steadyStates
        ? {
            steadyStates: {
              value: output.steadyStates,
              format: 'float32' as const,
              length: chainCount * props.classCount
            }
          }
        : {}),
      ...(output.stateSparsity
        ? {
            stateSparsity: {
              value: output.stateSparsity,
              format: 'uint32' as const,
              length: GPU_TRANSITION_STATE_SPARSITY_LENGTH
            }
          }
        : {}),
      ...(output.status?.converged
        ? {converged: {value: output.status.converged, format: 'uint32' as const, length: 1}}
        : {})
    },
    createContributor: getView =>
      new GPUTransitionDiagnostics({
        ...props,
        id,
        counts: getView<'uint32'>('counts'),
        output: {
          homogeneity: output.homogeneity ? getView<'float32'>('homogeneity') : undefined,
          steadyStates: output.steadyStates ? getView<'float32'>('steadyStates') : undefined,
          stateSparsity: output.stateSparsity ? getView<'uint32'>('stateSparsity') : undefined,
          status: {
            converged: output.status?.converged ? getView<'uint32'>('converged') : undefined
          }
        }
      })
  };
}

/** Transition diagnostics operation; compile with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. */
export class GPUTransitionDiagnosticsOperation extends GPUContributorOperation {
  constructor(props: TransitionDiagnosticsOperationProps) {
    props = captureTransitionDiagnosticsOperationProps(props);
    // The GPU sparsity threshold is a uint32 shader constant: validateCapacity's default
    // maximum 0xffffffff gives [1, 2^32 - 1].
    validateCapacity(
      props.id ?? 'transition-diagnostics',
      'minimumTransitions',
      props.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS,
      1
    );
    super(getTransitionDiagnosticsOperationProps(props));
  }
}
