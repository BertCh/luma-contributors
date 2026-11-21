// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUProbitRegression, type GPUProbitRegressionProps} from './gpu-probit-regression';
import {resolveProbitConvergenceTolerance} from './probit-convergence';

/** Graph names in place of views; rowCount explicitly sizes the fitted output. */
export type ProbitRegressionOperationProps = Pick<
  GPUProbitRegressionProps,
  'id' | 'predictorCount' | 'maximumIterations' | 'tolerance'
> & {
  rowCount: number;
  predictors: string;
  response: string;
  mask?: string;
  spatialLag?: {
    weights: {
      offsets: string;
      neighbors: string;
      weights: string;
      status?: Partial<
        Record<
          | 'complete'
          | 'overflow'
          | 'count'
          | 'requiredCount'
          | 'candidateOverflow'
          | 'invalidCount'
          | 'iterationLimitReached'
          | 'converged'
          | 'approximation'
          | 'uncertainCount',
          string
        >
      >;
    };
    output: {coefficients: string; standardErrors: string; status: string};
  };
  spatialDiagnostics?: {
    weights: {
      offsets: string;
      neighbors: string;
      weights: string;
      status?: Partial<
        Record<
          | 'complete'
          | 'overflow'
          | 'count'
          | 'requiredCount'
          | 'candidateOverflow'
          | 'invalidCount'
          | 'iterationLimitReached'
          | 'converged'
          | 'approximation'
          | 'uncertainCount',
          string
        >
      >;
    };
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
    props = {
      ...props,
      output: {...props.output},
      spatialDiagnostics: props.spatialDiagnostics
        ? {
            ...props.spatialDiagnostics,
            weights: {
              ...props.spatialDiagnostics.weights,
              status: {...props.spatialDiagnostics.weights.status}
            },
            output: {...props.spatialDiagnostics.output}
          }
        : undefined,
      spatialLag: props.spatialLag
        ? {
            ...props.spatialLag,
            weights: {...props.spatialLag.weights, status: {...props.spatialLag.weights.status}},
            output: {...props.spatialLag.output}
          }
        : undefined
    };
    const id = props.id ?? 'probit-regression';
    resolveProbitConvergenceTolerance(id, props.tolerance);
    // Optional names are tested against undefined, not for truthiness, so that an explicitly
    // empty name reaches the shared graph-name validation instead of being discarded.
    const diagnostics = props.spatialDiagnostics;
    const spatialLag = props.spatialLag;
    const diagnosticStatuses = diagnostics?.weights.status ?? {};
    const lagStatuses = spatialLag?.weights.status ?? {};
    super({
      id,
      inputs: {
        predictors: props.predictors,
        response: props.response,
        ...(props.mask !== undefined ? {mask: props.mask} : {}),
        ...(diagnostics
          ? {
              offsets: diagnostics.weights.offsets,
              neighbors: diagnostics.weights.neighbors,
              weights: diagnostics.weights.weights,
              ...Object.fromEntries(
                Object.entries(diagnosticStatuses).map(([name, value]) => [
                  `diagnosticWeight${name}`,
                  value
                ])
              )
            }
          : {}),
        ...(spatialLag
          ? {
              lagOffsets: spatialLag.weights.offsets,
              lagNeighbors: spatialLag.weights.neighbors,
              lagWeights: spatialLag.weights.weights,
              ...Object.fromEntries(
                Object.entries(lagStatuses).map(([name, value]) => [`lagWeight${name}`, value])
              )
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
        ...(spatialLag
          ? {
              lagCoefficients: {
                value: spatialLag.output.coefficients,
                format: 'float32' as const,
                length: props.predictorCount + 2
              },
              lagStandardErrors: {
                value: spatialLag.output.standardErrors,
                format: 'float32' as const,
                length: props.predictorCount + 2
              },
              lagStatus: {value: spatialLag.output.status, format: 'uint32' as const, length: 1}
            }
          : {}),
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
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          spatialDiagnostics: diagnostics
            ? {
                weights: {
                  offsets: getView<'uint32'>('offsets'),
                  neighbors: getView<'uint32'>('neighbors'),
                  weights: getView<'float32'>('weights'),
                  status: Object.fromEntries(
                    Object.keys(diagnosticStatuses).map(name => [
                      name,
                      getView<'uint32'>(`diagnosticWeight${name}`)
                    ])
                  )
                },
                output: {
                  tests: getView<'float32'>('diagnosticTests'),
                  status: getView<'uint32'>('diagnosticStatus')
                }
              }
            : undefined,
          spatialLag: spatialLag
            ? {
                weights: {
                  offsets: getView<'uint32'>('lagOffsets'),
                  neighbors: getView<'uint32'>('lagNeighbors'),
                  weights: getView<'float32'>('lagWeights'),
                  status: Object.fromEntries(
                    Object.keys(lagStatuses).map(name => [
                      name,
                      getView<'uint32'>(`lagWeight${name}`)
                    ])
                  )
                },
                output: {
                  coefficients: getView<'float32'>('lagCoefficients'),
                  standardErrors: getView<'float32'>('lagStandardErrors'),
                  status: getView<'uint32'>('lagStatus')
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
