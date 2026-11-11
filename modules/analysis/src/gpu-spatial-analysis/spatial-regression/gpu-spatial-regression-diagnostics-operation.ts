// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
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
  weights: {offsets: string; neighbors: string; weights: string};
  predictors: string;
  response: string;
  residuals: string;
  mask?: string;
  output: {tests: string; summary: string; status: {code: string}};
};

/** GPU spatial regression report as a graph operation, with the contributor's minimum sizes. */
export class GPUSpatialRegressionDiagnosticsOperation extends GPUContributorOperation {
  constructor(props: SpatialRegressionDiagnosticsOperationProps) {
    const id = props.id ?? 'spatial-regression-diagnostics';
    super({
      id,
      inputs: {
        ...props.weights,
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
      createContributor: getView =>
        new GPUSpatialRegressionDiagnostics({
          ...props,
          id,
          weights: {
            offsets: getView<'uint32'>('offsets'),
            neighbors: getView<'uint32'>('neighbors'),
            weights: getView<'float32'>('weights')
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
        })
    });
  }
}
