// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {GPUSpatialInteraction} from './gpu-spatial-interaction';
export type {GPUSpatialInteractionProps} from './gpu-spatial-interaction';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeSpatialInteractionOnCPU,
  CPUSpatialInteractionOperation
} from './cpu-spatial-interaction-operation';
export type {
  CPUSpatialInteractionInput,
  CPUSpatialInteractionOptions,
  CPUSpatialInteractionResult
} from './cpu-spatial-interaction-operation';
export {GPUSpatialInteractionOperation} from './gpu-spatial-interaction-operation';
export type {SpatialInteractionOperationProps} from './gpu-spatial-interaction-operation';

export {
  GPU_SPATIAL_INTERACTION_DENSE_MAXIMUM_PAIRS,
  GPU_SPATIAL_INTERACTION_DENSE_MAXIMUM_GROUPS,
  GPU_SPATIAL_INTERACTION_SPARSE_MAXIMUM_PAIRS,
  GPU_SPATIAL_INTERACTION_SPARSE_MAXIMUM_GROUPS
} from './gpu-spatial-interaction';
