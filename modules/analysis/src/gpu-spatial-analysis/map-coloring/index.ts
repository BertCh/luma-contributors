// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  getGPUMapColoringParameterValues,
  GPUMapColoring,
  GPU_MAP_COLORING_DEFAULT_MAXIMUM_ITERATIONS,
  GPU_MAP_COLORING_PARAMETER_LENGTH,
  GPU_MAP_COLORING_UNCOLORED
} from './gpu-map-coloring';
export type {
  GPUMapColoringOutput,
  GPUMapColoringProps,
  GPUMapColoringSettings
} from './gpu-map-coloring';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {computeMapColoringOnCPU, CPUMapColoringOperation} from './cpu-map-coloring-operation';
export type {CPUMapColoringResult} from './cpu-map-coloring-operation';
export {GPUMapColoringOperation} from './gpu-map-coloring-operation';
export type {MapColoringOperationProps} from './gpu-map-coloring-operation';
