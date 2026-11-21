// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWeightsLossSources} from '../spatial-weights/spatial-weights-loss';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import type {GPUChunkedSpatialWeights} from '../spatial-weights/gpu-chunked-spatial-weights';

/** Keep positive certificates distinct from negative loss flags. @internal */
export function getPermutationWeightsLossChecks(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
) {
  const sources = getWeightsLossSources([weights], 'permutation weights', true);
  return [
    ...[...new Set([...sources.overflow, ...sources.candidateOverflow])].map(view => ({
      view,
      incompleteWhen: '!='
    })),
    ...sources.complete.map(view => ({view, incompleteWhen: '=='}))
  ];
}
