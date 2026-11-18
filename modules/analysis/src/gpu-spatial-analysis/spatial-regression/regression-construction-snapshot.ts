// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotWeightsPort} from '../spatial-weights/weights-construction-snapshot';
import type {GPUSpatialWeights, GPUChunkedSpatialWeights} from '../spatial-weights/index';
import {getRegressionLossViews} from './regression-weights-completeness';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Capture declared output maps without cloning GPU views. */
function snapshotOutput<Output extends object>(output: Output, fields: readonly string[]): Output {
  const captured = snapshotConstructionFields(
    output,
    fields as readonly (keyof Output)[]
  ) as Output & {status?: object | string};
  return Object.freeze({
    ...captured,
    ...(captured.status
      ? {
          status:
            typeof captured.status !== 'object' || 'buffer' in captured.status
              ? captured.status
              : snapshotConstructionFields(captured.status, [
                  'code',
                  'converged',
                  'approximation'
                ] as never[])
        }
      : {})
  });
}

/** Named GPU/CPU forms capture scalar options and port-name maps before lowering. */
export function snapshotRegressionOperationProps<Props extends {weights: object; output: object}>(
  input: Props
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
    weights: snapshotWeightsPort(props.weights),
    output: snapshotOutput(props.output, [
      'table',
      'summary',
      'residuals',
      'status',
      'powerTraces',
      'powerSums',
      'powerTraceStandardErrors',
      'gramTraces',
      'infinityNorm'
    ]),
    ...(capturedRobust
      ? {
          robust: Object.freeze({
            ...capturedRobust,
            ...(capturedRobust.kernel ? {kernel: snapshotWeightsPort(capturedRobust.kernel)} : {}),
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
  defaultId: string
): Props {
  const props = snapshotConstructionFields(input, fields as readonly (keyof Props)[]);
  const weights = snapshotWeightsPort(props.weights);
  const id = props.id ?? defaultId;
  for (const view of getRegressionLossViews(weights)) {
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
    output: snapshotOutput(props.output, outputFields),
    ...(capturedRobust
      ? {
          robust: Object.freeze({
            ...capturedRobust,
            ...(capturedRobust.kernel ? {kernel: snapshotWeightsPort(capturedRobust.kernel)} : {}),
            output: snapshotOutput(capturedRobust.output, ['covariance', 'status'])
          })
        }
      : {})
  });
}
