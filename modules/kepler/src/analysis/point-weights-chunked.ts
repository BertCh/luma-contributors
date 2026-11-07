// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUSpatialWeightsOrChunksViews,
  isGPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeights
} from '@luma-contributors/graph/gpu-spatial-analysis';
import type {GraphDataView} from '@luma-contributors/graph/gpu-core';

/** Point weights as one CSR, or as row chunks when the single CSR exceeded the binding limit. */
export type KeplerPointWeightsStorage = GPUSpatialWeights | GPUChunkedSpatialWeights;

/** Source rows of either form of point weights. */
export function getKeplerPointWeightsRowCount(weights: KeplerPointWeightsStorage): number {
  return isGPUChunkedSpatialWeights(weights) ? weights.rowCount : weights.offsets.length - 1;
}

/** Every defined view of either form, for graph-membership and overlap checks. */
export function getKeplerPointWeightsViews(
  weights: KeplerPointWeightsStorage
): GraphDataView<'uint32' | 'float32'>[] {
  return getGPUSpatialWeightsOrChunksViews(weights).filter(view => view !== undefined);
}

/** Narrows to one CSR for consumers that have no chunk-aware kernels. */
export function requireKeplerWholePointWeights(
  weights: KeplerPointWeightsStorage
): GPUSpatialWeights {
  // This analysis needs a single-CSR weights matrix; lower the neighbor capacity so it fits one binding.
  if (isGPUChunkedSpatialWeights(weights))
    throw new Error('Analysis requires single-CSR point weights');
  return weights;
}
