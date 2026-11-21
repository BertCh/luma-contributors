// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRegressionWeights} from './regression-weights-snapshot';
import type {GPUSpatialWeights, GPUChunkedSpatialWeights} from '../spatial-weights/index';
import {getRegressionCompletenessViews} from './regression-weights-completeness';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Capture declared output maps without cloning GPU views. */
function snapshotOutput<Output extends object>(
  output: Output,
  fields: readonly string[],
  statusFields: readonly string[] = ['code'],
  statusShape: 'map' | 'scalar' = 'map'
): Output {
  const captured = snapshotConstructionFields(
    output,
    fields as readonly (keyof Output)[]
  ) as Output & {status?: object | string};
  return Object.freeze({
    ...captured,
    ...(captured.status
      ? {
          status:
            statusShape === 'scalar'
              ? captured.status
              : snapshotConstructionFields(captured.status as object, statusFields as never[])
        }
      : {})
  });
}

/** Named GPU/CPU forms capture scalar options and port-name maps before lowering. */
export function snapshotRegressionOperationProps<Props extends {weights: object; output: object}>(
  input: Props,
  statusFields: readonly string[] = ['code', 'converged']
): Props {
  const props = snapshotConstructionFields(input, [
    'id',
    'rowCount',
    'weights',
    'predictors',
    'response',
    'predictorCount',
    'estimator',
    'traceMode',
    'traceProbeCount',
    'traceParameters',
    'tileRowCount',
    'maximumPower',
    'mode',
    'probeCount',
    'parameters',
    'includeGramTraces',
    'probeBatchSize',
    'tailExtrapolation',
    'logDeterminant',
    'eigenvalues',
    'instrumentOrder',
    'robust',
    'output'
  ] as readonly (keyof Props)[]);
  const robust = (props as Props & {robust?: {kernel?: object; output: object}}).robust;
  const capturedRobust =
    robust &&
    snapshotConstructionFields(robust, [
      'method',
      'clusters',
      'clusterCount',
      'kernel',
      'output'
    ] as never[]);
  return Object.freeze({
    ...props,
    weights: snapshotRegressionWeights(props.weights),
    output: snapshotOutput(
      props.output,
      [
        'table',
        'summary',
        'residuals',
        'status',
        'powerTraces',
        'powerSums',
        'powerTraceStandardErrors',
        'gramTraces',
        'infinityNorm'
      ],
      statusFields
    ),
    ...(capturedRobust
      ? {
          robust: Object.freeze({
            ...capturedRobust,
            ...(capturedRobust.kernel
              ? {
                  kernel: snapshotRegressionWeights(capturedRobust.kernel)
                }
              : {}),
            output: snapshotOutput(capturedRobust.output, ['covariance', 'status'])
          })
        }
      : {})
  });
}

/** Construction-time scalars and nested maps, including inherited getters, are read once. */
export function snapshotRegressionConstructionProps<
  Props extends {id?: string; weights: GPUSpatialWeights | GPUChunkedSpatialWeights; output: object}
>(
  input: Props,
  fields: readonly string[],
  outputFields: readonly string[],
  defaultId: string,
  statusFields: readonly string[] = ['code', 'converged'],
  statusShape: 'map' | 'scalar' = 'map'
): Props {
  const props = snapshotConstructionFields(input, fields as readonly (keyof Props)[]);
  const weights = snapshotRegressionWeights(props.weights);
  const id = props.id ?? defaultId;
  for (const view of getRegressionCompletenessViews(weights)) {
    validateSinglePackedView(view, ['uint32'], `${id} weights status`);
    validateRequirement(id, view.length >= 1, 'weights status must hold at least one uint32');
  }
  const robust = (props as Props & {robust?: {kernel?: GPUSpatialWeights; output: object}}).robust;
  const capturedRobust =
    robust &&
    snapshotConstructionFields(robust, [
      'method',
      'clusters',
      'clusterCount',
      'kernel',
      'output'
    ] as never[]);
  return Object.freeze({
    ...props,
    weights,
    output: snapshotOutput(props.output, outputFields, statusFields, statusShape),
    ...(capturedRobust
      ? {
          robust: Object.freeze({
            ...capturedRobust,
            ...(capturedRobust.kernel
              ? {kernel: snapshotRegressionWeights(capturedRobust.kernel)}
              : {}),
            output: snapshotOutput(capturedRobust.output, ['covariance', 'status'])
          })
        }
      : {})
  });
}
