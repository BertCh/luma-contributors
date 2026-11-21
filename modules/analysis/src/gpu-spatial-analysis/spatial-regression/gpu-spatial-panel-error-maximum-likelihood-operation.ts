// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRegressionWeights} from './regression-weights-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRegressionOperationProps} from './regression-construction-snapshot';
import {
  GPUSpatialPanelErrorMaximumLikelihood,
  type GPUSpatialPanelErrorMaximumLikelihoodProps
} from './gpu-spatial-panel-error-maximum-likelihood';
import type {SpatialErrorMaximumLikelihoodOperationProps} from './gpu-spatial-error-maximum-likelihood-operation';

/** Named values replace views; entityCount is the whole entity CSR row count. */
export type SpatialPanelErrorMaximumLikelihoodOperationProps = Omit<
  SpatialErrorMaximumLikelihoodOperationProps,
  'rowCount' | 'weights'
> &
  Pick<GPUSpatialPanelErrorMaximumLikelihoodProps, 'periodCount' | 'effects'> & {
    entityCount: number;
    weights: {
      offsets: string;
      neighbors: string;
      weights: string;
      status?: {
        complete?: string;
        overflow?: string;
        candidateOverflow?: string;
        count?: string;
        requiredCount?: string;
      };
    };
    diagnostics?: {tests: string; status: string};
  };

/** Capture panel scalars and nested diagnostic port maps before either execution form. @internal */
export function snapshotPanelErrorOperationProps(
  props: SpatialPanelErrorMaximumLikelihoodOperationProps
): SpatialPanelErrorMaximumLikelihoodOperationProps {
  validateRequirement(
    props.id ?? 'spatial-panel-error-maximum-likelihood',
    !('exactSpectrumRepetitionCount' in props),
    'exactSpectrumRepetitionCount is not a panel prop; supply entityCount or entityCount * periodCount eigenvalues'
  );
  props = snapshotConstructionFields(props, [
    'id',
    'weights',
    'entityCount',
    'periodCount',
    'effects',
    'predictorCount',
    'predictors',
    'response',
    'maximumPower',
    'logDeterminant',
    'eigenvalues',
    'traceMode',
    'traceProbeCount',
    'traceParameters',
    'tailExtrapolation',
    'tileRowCount',
    'diagnostics',
    'output'
  ]);
  const {entityCount, periodCount, effects, diagnostics} = props;
  const weights = snapshotRegressionWeights(props.weights);
  const captured = snapshotRegressionOperationProps({...props, weights});
  return Object.freeze({
    ...captured,
    weights,
    entityCount,
    periodCount,
    effects,
    diagnostics: diagnostics && snapshotConstructionFields(diagnostics, ['tests', 'status'])
  });
}

/** Balanced fixed-effects error ML operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSpatialPanelErrorMaximumLikelihoodOperation extends GPUContributorOperation {
  constructor(props: SpatialPanelErrorMaximumLikelihoodOperationProps) {
    props = snapshotPanelErrorOperationProps(props);
    const id = props.id ?? 'spatial-panel-error-maximum-likelihood';
    for (const name of ['count', 'requiredCount', 'complete'] as const) {
      const value = props.weights.status?.[name];
      if (value !== undefined)
        validateRequirement(
          id,
          typeof value === 'string' && value.length > 0,
          `weights.status.${name} must name a non-empty graph value`
        );
    }
    super({
      id,
      inputs: {
        offsets: props.weights.offsets,
        neighbors: props.weights.neighbors,
        weights: props.weights.weights,
        ...(props.weights.status?.complete !== undefined
          ? {complete: props.weights.status.complete}
          : {}),
        ...(props.weights.status?.count ? {count: props.weights.status.count} : {}),
        ...(props.weights.status?.requiredCount
          ? {requiredCount: props.weights.status.requiredCount}
          : {}),
        predictors: props.predictors,
        response: props.response,
        ...(props.weights.status?.overflow !== undefined
          ? {overflow: props.weights.status.overflow}
          : {}),
        ...(props.weights.status?.candidateOverflow !== undefined
          ? {candidateOverflow: props.weights.status.candidateOverflow}
          : {}),
        ...(props.eigenvalues !== undefined ? {eigenvalues: props.eigenvalues} : {}),
        ...(props.traceParameters !== undefined ? {traceParameters: props.traceParameters} : {})
      },
      outputs: {
        table: {
          value: props.output.table,
          format: 'float32',
          length: (props.predictorCount + 2) * 4
        },
        summary: {value: props.output.summary, format: 'float32', length: 22},
        code: {value: props.output.status.code, format: 'uint32', length: 1},
        ...(props.output.status.converged !== undefined
          ? {
              converged: {
                value: props.output.status.converged,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {}),
        ...(props.output.residuals !== undefined
          ? {
              residuals: {
                value: props.output.residuals,
                format: 'float32' as const,
                length: props.entityCount * props.periodCount
              }
            }
          : {}),
        ...(props.output.powerTraceStandardErrors !== undefined
          ? {
              powerTraceStandardErrors: {
                value: props.output.powerTraceStandardErrors,
                format: 'float32' as const,
                length: props.maximumPower ?? 128
              }
            }
          : {}),
        ...(props.diagnostics
          ? {
              diagnosticTests: {
                value: props.diagnostics.tests,
                format: 'float32' as const,
                length: 6
              },
              diagnosticStatus: {
                value: props.diagnostics.status,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView => {
        // entityCount specifies named output shapes; it must agree with the resolved entity
        // graph even when a caller omits residuals (whose output length would expose it).
        validateRequirement(
          id,
          getView('offsets').length === props.entityCount + 1,
          'entityCount must equal weights.offsets.length - 1'
        );
        return new GPUSpatialPanelErrorMaximumLikelihood({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            status: props.weights.status
              ? {
                  complete:
                    props.weights.status.complete !== undefined
                      ? getView<'uint32'>('complete')
                      : undefined,
                  count: props.weights.status.count ? getView<'uint32'>('count') : undefined,
                  requiredCount: props.weights.status.requiredCount
                    ? getView<'uint32'>('requiredCount')
                    : undefined,
                  overflow:
                    props.weights.status.overflow !== undefined
                      ? getView<'uint32'>('overflow')
                      : undefined,
                  candidateOverflow:
                    props.weights.status.candidateOverflow !== undefined
                      ? getView<'uint32'>('candidateOverflow')
                      : undefined
                }
              : undefined
          },
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          eigenvalues:
            props.eigenvalues !== undefined ? getView<'float32'>('eigenvalues') : undefined,
          traceParameters:
            props.traceParameters !== undefined ? getView<'uint32'>('traceParameters') : undefined,
          output: {
            table: getView<'float32'>('table'),
            summary: getView<'float32'>('summary'),
            residuals:
              props.output.residuals !== undefined ? getView<'float32'>('residuals') : undefined,
            powerTraceStandardErrors:
              props.output.powerTraceStandardErrors !== undefined
                ? getView<'float32'>('powerTraceStandardErrors')
                : undefined,
            status: {
              code: getView<'uint32'>('code'),
              converged:
                props.output.status.converged !== undefined
                  ? getView<'uint32'>('converged')
                  : undefined
            }
          },
          diagnostics: props.diagnostics
            ? {
                tests: getView<'float32'>('diagnosticTests'),
                status: getView<'uint32'>('diagnosticStatus')
              }
            : undefined
        });
      }
    });
  }
}
