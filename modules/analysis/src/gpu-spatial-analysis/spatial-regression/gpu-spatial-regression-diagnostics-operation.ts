// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  GPUSpatialRegressionDiagnostics,
  type GPUSpatialRegressionDiagnosticsProps
} from './gpu-spatial-regression-diagnostics';
import {
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
} from './spatial-regression-diagnostics-parameters';

/** Whole CSR graph values and explicit input row count. Unused weights metadata is omitted. */
export type SpatialRegressionDiagnosticsOperationProps = Pick<
  GPUSpatialRegressionDiagnosticsProps,
  'id' | 'predictorCount' | 'restandardizeRows' | 'tileRowCount'
> & {
  rowCount: number;
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
  predictors: string;
  response: string;
  residuals: string;
  mask?: string;
  output: {tests: string; summary: string; status: {code: string}};
};

/** Capture declared operation ports once for contract, lowering and CPU execution. @internal */
export function snapshotDiagnosticsOperationProps(
  props: SpatialRegressionDiagnosticsOperationProps
): SpatialRegressionDiagnosticsOperationProps {
  const captured = snapshotConstructionFields(props, [
    'id',
    'rowCount',
    'predictorCount',
    'restandardizeRows',
    'tileRowCount',
    'weights',
    'predictors',
    'response',
    'residuals',
    'mask',
    'output'
  ]);
  const weights = snapshotConstructionFields(captured.weights, [
    'offsets',
    'neighbors',
    'weights',
    'status'
  ]);
  const output = snapshotConstructionFields(captured.output, ['tests', 'summary', 'status']);
  return Object.freeze({
    ...captured,
    weights: Object.freeze({
      ...weights,
      ...(weights.status
        ? {
            status: snapshotConstructionFields(weights.status, [
              'complete',
              'overflow',
              'candidateOverflow',
              'count',
              'requiredCount'
            ])
          }
        : {})
    }),
    output: Object.freeze({...output, status: snapshotConstructionFields(output.status, ['code'])})
  });
}

/** GPU spatial regression report as a graph operation, with the contributor's minimum sizes. */
export class GPUSpatialRegressionDiagnosticsOperation extends GPUContributorOperation {
  constructor(props: SpatialRegressionDiagnosticsOperationProps) {
    props = snapshotDiagnosticsOperationProps(props);
    const id = props.id ?? 'spatial-regression-diagnostics';
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
        ...(props.weights.status?.count ? {weightsCount: props.weights.status.count} : {}),
        ...(props.weights.status?.requiredCount
          ? {weightsRequiredCount: props.weights.status.requiredCount}
          : {}),
        ...(props.weights.status?.overflow ? {weightsOverflow: props.weights.status.overflow} : {}),
        ...(props.weights.status?.candidateOverflow
          ? {weightsCandidateOverflow: props.weights.status.candidateOverflow}
          : {}),
        predictors: props.predictors,
        response: props.response,
        residuals: props.residuals,
        ...(props.mask ? {mask: props.mask} : {})
      },
      outputs: {
        tests: {
          value: props.output.tests,
          format: 'float32',
          length: GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
        },
        summary: {
          value: props.output.summary,
          format: 'float32',
          length: GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
        },
        code: {value: props.output.status.code, format: 'uint32', length: 1}
      },
      createContributor: getView => {
        validateRequirement(
          id,
          getView('response').length === props.rowCount,
          'response length must equal rowCount'
        );
        return new GPUSpatialRegressionDiagnostics({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights'),
            status: {
              complete:
                props.weights.status?.complete !== undefined
                  ? getView<'uint32'>('complete')
                  : undefined,
              count: props.weights.status?.count ? getView<'uint32'>('weightsCount') : undefined,
              requiredCount: props.weights.status?.requiredCount
                ? getView<'uint32'>('weightsRequiredCount')
                : undefined,
              overflow: props.weights.status?.overflow
                ? getView<'uint32'>('weightsOverflow')
                : undefined,
              candidateOverflow: props.weights.status?.candidateOverflow
                ? getView<'uint32'>('weightsCandidateOverflow')
                : undefined
            }
          },
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          residuals: getView<'float32'>('residuals'),
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          output: {
            tests: getView<'float32'>('tests'),
            summary: getView<'float32'>('summary'),
            status: {code: getView<'uint32'>('code')}
          }
        });
      }
    });
  }
}
