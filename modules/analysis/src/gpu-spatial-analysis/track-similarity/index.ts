// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUTrackSimilarity,
  GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT,
  GPU_TRACK_SIMILARITY_STATUS
} from './gpu-track-similarity';
export type {GPUTrackSimilarityOutput, GPUTrackSimilarityProps} from './gpu-track-similarity';
export {GPUTrackSimilaritySearch} from './gpu-track-similarity-search';
export type {GPUTrackSimilaritySearchProps} from './gpu-track-similarity-search';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeTrackSimilarityOnCPU,
  CPUTrackSimilarityOperation
} from './cpu-track-similarity-operation';
export type {
  CPUTrackSimilarityInput,
  CPUTrackSimilarityResult,
  CPUTrackSimilarityOptions
} from './cpu-track-similarity-operation';
export {GPUTrackSimilarityOperation} from './gpu-track-similarity-operation';
export type {TrackSimilarityOperationProps} from './gpu-track-similarity-operation';
