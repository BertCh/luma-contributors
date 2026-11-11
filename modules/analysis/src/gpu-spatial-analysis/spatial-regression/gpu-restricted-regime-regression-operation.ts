// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPURestrictedRegimeRegression,
  type GPURestrictedRegimeRegressionProps
} from './gpu-restricted-regime-regression';

/** Graph names replace views; rowCount supplies the response length. */
export type RestrictedRegimeRegressionOperationProps = Pick<
  GPURestrictedRegimeRegressionProps,
  'id' | 'predictorCount' | 'regimeCount' | 'common'
> & {
  rowCount: number;
  predictors: string;
  response: string;
  regimes: string;
  mask?: string;
  output: {
    coefficients: string;
    covariance: string;
    fitted: string;
    chow: string;
    stability?: string;
    status: string;
  };
};

/** GPU restricted regimes as a graph operation, with the contributor's exact output sizes. */
export class GPURestrictedRegimeRegressionOperation extends GPUContributorOperation {
  constructor(props: RestrictedRegimeRegressionOperationProps) {
    const id = props.id ?? 'restricted-regime-regression';
    const coefficientCount = props.common.reduce(
      (count, shared) => count + (shared ? 1 : props.regimeCount),
      0
    );
    super({
      id,
      inputs: {
        predictors: props.predictors,
        response: props.response,
        regimes: props.regimes,
        ...(props.mask ? {mask: props.mask} : {})
      },
      outputs: {
        coefficients: {
          value: props.output.coefficients,
          format: 'float32',
          length: coefficientCount
        },
        covariance: {
          value: props.output.covariance,
          format: 'float32',
          length: coefficientCount ** 2
        },
        fitted: {value: props.output.fitted, format: 'float32', length: props.rowCount},
        chow: {value: props.output.chow, format: 'float32', length: 3},
        status: {value: props.output.status, format: 'uint32', length: 1},
        ...(props.output.stability
          ? {
              stability: {
                value: props.output.stability,
                format: 'float32' as const,
                length: (props.predictorCount + 2) * 3
              }
            }
          : {})
      },
      createContributor: getView =>
        new GPURestrictedRegimeRegression({
          ...props,
          id,
          predictors: getView<'float32'>('predictors'),
          response: getView<'float32'>('response'),
          regimes: getView<'uint32'>('regimes'),
          mask: props.mask ? getView<'uint32'>('mask') : undefined,
          output: {
            coefficients: getView<'float32'>('coefficients'),
            covariance: getView<'float32'>('covariance'),
            fitted: getView<'float32'>('fitted'),
            chow: getView<'float32'>('chow'),
            status: getView<'uint32'>('status'),
            stability: props.output.stability ? getView<'float32'>('stability') : undefined
          }
        })
    });
  }
}
