// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialWeightsTransformProps} from './gpu-spatial-weights-transform';

/** Only caller views bound by the selected route; ownership still checks the full descriptor. @internal */
export function getSpatialWeightsTransformBindingViews(
  props: Pick<GPUSpatialWeightsTransformProps, 'weights' | 'operation' | 'output' | 'bandwidth'>
) {
  const {weights, operation, output, bandwidth} = props;
  return {
    offsets: weights.offsets,
    neighbors: operation === 'average-reciprocal' ? weights.neighbors : undefined,
    source: operation === 'kernel' ? undefined : weights.weights,
    output: output ?? weights.weights,
    distances: operation === 'kernel' ? weights.distances : undefined,
    bandwidth: operation === 'kernel' && typeof bandwidth === 'object' ? bandwidth : undefined
  };
}
