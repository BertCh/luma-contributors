// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUProbitRegression, type GPUProbitRegressionProps} from './gpu-probit-regression';

/** Graph names in place of views; rowCount explicitly sizes the fitted output. */
export type ProbitRegressionOperationProps = Pick<
  GPUProbitRegressionProps,
  'id' | 'predictorCount' | 'maximumIterations' | 'tolerance'
> & {
  rowCount: number;
  predictors: string;
  response: string;
  mask?: string;
  spatialDiagnostics?: {
    weights: {offsets: string; neighbors: string; weights: string; overflow?: string};
    output: {tests: string; status: string};
  };
  output: {
    coefficients: string;
    standardErrors: string;
    fitted: string;
    marginalEffects: string;
    summary: string;
    status: string;
  };
};

/** GPU probit as a graph operation. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Compile-time: all props. Per-frame: contents of the input graph values.
 * Optional spatial diagnostics support whole CSR; unused weights metadata is omitted.
 */
export class GPUProbitRegressionOperation extends GPUContributorOperation {
  constructor(props: ProbitRegressionOperationProps) {
    const id = props.id ?? 'probit-regression';
    const diagnostics = props.spatialDiagnostics;
    super({
      id,
      inputs: {
        predictors: props.predictors,
        response: props.response,
        ...(props.mask ? {mask: props.mask} : {}),
        ...(diagnostics
          ? {
              offsets: diagnostics.weights.offsets,
              neighbors: diagnostics.weights.neighbors,
              weights: diagnostics.weights.weights,
              ...(diagnostics.weights.overflow ? {overflow: diagnostics.weights.overflow} : {})
            }
          : {})
      },
      outputs: {
        coefficients: {
          value: props.output.coefficients,
          format: 'float32',
          length: props.predictorCount + 1
        },
        standardErrors: {
          value: props.output.standardErrors,
          format: 'float32',
          length: props.predictorCount + 1
        },
        fitted: {value: props.output.fitted, format: 'float32', length: props.rowCount},
        marginalEffects: {
          value: props.output.marginalEffects,
          format: 'float32',
          length: props.predictorCount
        },
        summary: {value: props.output.summary, format: 'float32', length: 4},
        status: {value: props.output.status, format: 'uint32', length: 1},
        ...(diagnostics
          ? {
              diagnosticTests: {
                value: diagnostics.output.tests,
                format: 'float32' as const,
                length: 6
              },
              diagnosticStatus: {
                value: diagnostics.output.status,
                format: 'uint32' as const,
                length: 1
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPUProbitRegression({
          ...props,
          id,
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          spatialDiagnostics: diagnostics
            ? {
                weights: {
                  offsets: getView<'uint32'>('offsets'),
                  neighbors: getView<'uint32'>('neighbors'),
                  weights: getView<'float32'>('weights'),
                  status: diagnostics.weights.overflow
                    ? {overflow: getView<'uint32'>('overflow')}
                    : undefined
                },
                output: {
                  tests: getView<'float32'>('diagnosticTests'),
                  status: getView<'uint32'>('diagnosticStatus')
                }
              }
            : undefined,
          output: {
            coefficients: getView<'float32'>('coefficients'),
            standardErrors: getView<'float32'>('standardErrors'),
            fitted: getView<'float32'>('fitted'),
            marginalEffects: getView<'float32'>('marginalEffects'),
            summary: getView<'float32'>('summary'),
            status: getView<'uint32'>('status')
          }
        })
    });
  }
}
