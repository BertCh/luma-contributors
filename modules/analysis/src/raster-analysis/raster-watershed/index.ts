// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeRasterWatershedSegmentationOnCPU,
  CPURasterWatershedSegmentationOperation
} from './cpu-raster-watershed-segmentation-operation';
export type {
  CPURasterWatershedSegmentationOptions,
  CPURasterWatershedSegmentationResult
} from './cpu-raster-watershed-segmentation-operation';
export {GPURasterWatershedSegmentationOperation} from './gpu-raster-watershed-segmentation-operation';
export type {RasterWatershedSegmentationOperationProps} from './gpu-raster-watershed-segmentation-operation';
