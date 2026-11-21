// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotRestrictedRegimeKernelStatus,
  snapshotRestrictedRegimeKernelInformation
} from './restricted-regime-kernel-snapshot';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import type {
  RestrictedRegimeRobustOptions,
  RestrictedRegimeRobustOutput
} from './restricted-regime-robust';
import {
  GPURestrictedRegimeRegression,
  type GPURestrictedRegimeRegressionProps
} from './gpu-restricted-regime-regression';

/** Graph-value names of a kernel CSR, including any producer status ports. */
export type RestrictedRegimeKernelNames = {
  offsets: string;
  neighbors: string;
  weights: string;
  status?: Partial<Record<keyof NonNullable<GPUSpatialWeights['status']>, string>>;
  information?: GPUSpatialWeights['information'];
};

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
  robust?: RestrictedRegimeRobustOptions<RestrictedRegimeKernelNames>;
  output: {
    robust?: RestrictedRegimeRobustOutput<string, string>;
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
    props = {
      ...props,
      common: [...props.common],
      output: {...props.output, robust: props.output.robust ? {...props.output.robust} : undefined},
      robust: props.robust ? {...props.robust} : undefined
    };
    if (props.robust?.type === 'hac') {
      const {offsets, neighbors, weights, status, information} = props.robust.kernelWeights;
      props.robust = {
        ...props.robust,
        kernelWeights: {
          offsets,
          neighbors,
          weights,
          status: status && snapshotRestrictedRegimeKernelStatus(status),
          information: snapshotRestrictedRegimeKernelInformation(information)
        }
      };
    }
    const id = props.id ?? 'restricted-regime-regression';
    validateRequirement(
      id,
      Boolean(props.robust) === Boolean(props.output.robust),
      'robust and output.robust must be supplied together'
    );
    if (props.robust) {
      validateRequirement(
        id,
        props.robust.type === 'white' || props.robust.type === 'hac',
        "robust.type must be 'white' or 'hac'"
      );
      if (props.robust.type === 'white')
        validateRequirement(
          id,
          props.robust.smallSample === undefined || typeof props.robust.smallSample === 'boolean',
          'robust.smallSample must be boolean'
        );
    }
    const kernel = props.robust?.type === 'hac' ? props.robust.kernelWeights : undefined;
    const robustOutput = props.output.robust;
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
        ...(props.mask ? {mask: props.mask} : {}),
        ...(kernel
          ? {
              kernelOffsets: kernel.offsets,
              kernelNeighbors: kernel.neighbors,
              kernelWeights: kernel.weights,
              ...Object.fromEntries(
                Object.entries(kernel.status ?? {}).map(([name, value]) => [
                  `kernel_${name}`,
                  value
                ])
              )
            }
          : {})
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
        ...(robustOutput
          ? {
              robustCovariance: {
                value: robustOutput.covariance,
                format: 'float32' as const,
                length: coefficientCount ** 2
              },
              robustChow: {value: robustOutput.chow, format: 'float32' as const, length: 3},
              robustStability: {
                value: robustOutput.stability,
                format: 'float32' as const,
                length: (props.predictorCount + 2) * 3
              },
              robustStatus: {value: robustOutput.status, format: 'uint32' as const, length: 1}
            }
          : {}),
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
          robust:
            props.robust?.type === 'hac'
              ? {
                  type: 'hac',
                  kernelWeights: {
                    offsets: getView<'uint32'>('kernelOffsets'),
                    neighbors: getView<'uint32'>('kernelNeighbors'),
                    weights: getView<'float32'>('kernelWeights'),
                    information: kernel?.information,
                    status: Object.fromEntries(
                      Object.keys(kernel?.status ?? {}).map(name => [
                        name,
                        getView<'uint32'>(`kernel_${name}`)
                      ])
                    )
                  }
                }
              : props.robust,
          output: {
            robust: robustOutput
              ? {
                  covariance: getView<'float32'>('robustCovariance'),
                  chow: getView<'float32'>('robustChow'),
                  stability: getView<'float32'>('robustStability'),
                  status: getView<'uint32'>('robustStatus')
                }
              : undefined,
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
