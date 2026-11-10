// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_SINKHORN_DIVERGENCE_DEFAULT_MAXIMUM_ITERATIONS,
  GPU_SINKHORN_DIVERGENCE_MAXIMUM_GRID_SIDE,
  GPU_SINKHORN_DIVERGENCE_MAXIMUM_ITERATIONS,
  GPUSinkhornDivergence
} from './gpu-sinkhorn-divergence';
export type {
  GPUSinkhornDivergenceOutput,
  GPUSinkhornDivergenceProps
} from './gpu-sinkhorn-divergence';
export {
  getGPUSinkhornDivergenceParameterValues,
  GPU_SINKHORN_DIVERGENCE_DEFAULT_TOLERANCE,
  GPU_SINKHORN_DIVERGENCE_PARAMETER_LENGTH
} from './sinkhorn-divergence-parameters';
export type {GPUSinkhornDivergenceSettings} from './sinkhorn-divergence-parameters';
