// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRegressionConstructionProps} from './regression-construction-snapshot';
import {getRegressionCertificateViews} from './regression-weights-completeness';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialWeights, GPUChunkedSpatialWeights} from '../spatial-weights/index';

/** Capture regression inputs and validate optional positive completeness certificates. @internal */
export function snapshotRegressionConstructionPropsWithCompleteness<
  Props extends {id?: string; weights: GPUSpatialWeights | GPUChunkedSpatialWeights; output: object}
>(
  input: Props,
  fields: readonly string[],
  outputFields: readonly string[],
  defaultId: string,
  statusFields: readonly string[] = ['code', 'converged'],
  statusShape: 'map' | 'scalar' = 'map'
): Props {
  const props = snapshotRegressionConstructionProps(
    input,
    fields,
    outputFields,
    defaultId,
    statusFields,
    statusShape
  );
  for (const view of getRegressionCertificateViews(props.weights)) {
    validateSinglePackedView(view, ['uint32'], `${props.id ?? defaultId} weights.status.complete`);
    validateRequirement(
      props.id ?? defaultId,
      view.length >= 1,
      'weights.status.complete must hold at least one uint32'
    );
  }
  return props;
}
