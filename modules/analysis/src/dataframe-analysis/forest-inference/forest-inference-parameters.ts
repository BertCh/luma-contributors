// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

const ID = 'GPUForestInference';
/** Bounded traversal and local score storage, independent of the number of input rows. */
export const GPU_FOREST_INFERENCE_MAXIMUM_TREE_COUNT = 1024;
export const GPU_FOREST_INFERENCE_MAXIMUM_DEPTH = 64;
export const GPU_FOREST_INFERENCE_MAXIMUM_FEATURE_COUNT = 256;
export const GPU_FOREST_INFERENCE_MAXIMUM_OUTPUT_COUNT = 32;
/** Sparse nodes occupy 24 MiB at this cap; dense trees are never materialized. */
export const GPU_FOREST_INFERENCE_MAXIMUM_NODE_COUNT = 1048576;
/** Limits each row batch's worst-case traversal plus score accumulation work; split larger batches. */
export const GPU_FOREST_INFERENCE_MAXIMUM_WORK = 5000000000;
/** Each row batch emits at most 16 band-preparation dispatches, then inference and publication. */
export const GPU_FOREST_INFERENCE_MAXIMUM_RASTER_BAND_COUNT = 16;
/** Total rows use uint32 chunk bases; per-batch elements, traversal work and bindings are checked separately. */
export const GPU_FOREST_INFERENCE_MAXIMUM_ROW_COUNT = 0xffffffff;
/**
 * [feature, left, right, threshold bits, flags, original leaf id]. Leaf feature is 0xffffffff.
 * Flags: bit 0 default-left; bit 1 accepts missing; bit 2 zero (abs <= 1e-35) is missing;
 * bit 3 NaN converts to zero before the missing test; bit 4 strict < instead of <=. Other bits zero.
 */
export const GPU_FOREST_INFERENCE_NODE_STRIDE = 6;
/** [root node, exclusive end node, weight bits, reserved (zero)]. */
export const GPU_FOREST_INFERENCE_TREE_STRIDE = 4;

export type GPUForestInferenceSettings = {
  /** Initial raw score per output; zero for a random forest. */
  baseScores: readonly number[];
  /** Positive logistic link multiplier; LightGBM's sigmoid, otherwise one. */
  sigmoid?: number;
};

/** One logistic multiplier followed by outputCount initial raw scores. */
export function getGPUForestInferenceParameterLength(outputCount: number): number {
  validateCapacity(ID, 'outputCount', outputCount, 1, GPU_FOREST_INFERENCE_MAXIMUM_OUTPUT_COUNT);
  return outputCount + 1;
}

/** Packs [sigmoid, ...baseScores] for replay without recompilation. */
export function getGPUForestInferenceParameterValues(
  settings: GPUForestInferenceSettings,
  target: Float32Array = new Float32Array(
    getGPUForestInferenceParameterLength(settings.baseScores.length)
  )
): Float32Array {
  const length = getGPUForestInferenceParameterLength(settings.baseScores.length);
  validateRequirement(
    ID,
    target.length >= length,
    `parameter target length must be at least ${length}`
  );
  const sigmoid = settings.sigmoid ?? 1;
  validateFiniteNumber(ID, 'sigmoid', sigmoid);
  validateRequirement(ID, Math.fround(sigmoid) > 0, 'sigmoid must be greater than 0 as float32');
  target[0] = sigmoid;
  settings.baseScores.forEach((score, index) => {
    validateFiniteNumber(ID, `baseScores[${index}]`, score);
    target[index + 1] = score;
  });
  return target;
}

/** Compile-time shape shared by the importer and GPU contributor. */
export type GPUForestInferenceModelShape = {
  featureCount: number;
  treeCount: number;
  nodeCount: number;
  outputCount: number;
  maximumDepth: number;
  task: 'regression' | 'classification';
  transform: 'identity' | 'sigmoid' | 'softmax';
};

/** @internal Validates model topology once, before buffers or nodes are created. */
export function validateForestInferenceModelShape(
  id: string,
  model: GPUForestInferenceModelShape
): void {
  validateCapacity(
    id,
    'featureCount',
    model.featureCount,
    1,
    GPU_FOREST_INFERENCE_MAXIMUM_FEATURE_COUNT
  );
  validateCapacity(id, 'treeCount', model.treeCount, 1, GPU_FOREST_INFERENCE_MAXIMUM_TREE_COUNT);
  validateCapacity(id, 'nodeCount', model.nodeCount, 1, GPU_FOREST_INFERENCE_MAXIMUM_NODE_COUNT);
  validateCapacity(
    id,
    'outputCount',
    model.outputCount,
    1,
    GPU_FOREST_INFERENCE_MAXIMUM_OUTPUT_COUNT
  );
  validateCapacity(id, 'maximumDepth', model.maximumDepth, 0, GPU_FOREST_INFERENCE_MAXIMUM_DEPTH);
  validateRequirement(
    id,
    model.task === 'regression' || model.task === 'classification',
    'task must be regression or classification'
  );
  validateRequirement(
    id,
    ['identity', 'sigmoid', 'softmax'].includes(model.transform),
    'transform must be identity, sigmoid or softmax'
  );
  validateRequirement(
    id,
    model.task === 'classification'
      ? model.outputCount >= 1
      : model.outputCount === 1 && model.transform === 'identity',
    'outputCount must be at least 1 for classification or 1 with identity for regression'
  );
  validateRequirement(
    id,
    model.transform !== 'sigmoid' || model.outputCount === 2,
    'sigmoid outputCount must equal 2'
  );
}
