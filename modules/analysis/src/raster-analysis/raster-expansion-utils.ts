// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../analysis-core/kernels/gpu-contributor-utils';
import {validateRasterBuffersDistinct, validateRasterGrid} from './raster-band-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

/** Shared exact-length and ownership checks for bounded raster producers. @internal */
export function validateRasterExpansionViews(
  id: string,
  grid: GPURasterMetadata,
  inputs: readonly (GraphDataView | undefined)[],
  outputs: readonly (GraphDataView | undefined)[]
): number {
  const count = validateRasterGrid(id, grid);
  for (const view of [...inputs, ...outputs]) {
    if (view) validatePackedView(view, [view.format], id);
  }
  validateRasterBuffersDistinct(id, outputs, inputs);
  return count;
}

/** @internal */
export function validateRasterExpansionGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: readonly (GraphDataView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, views);
}

/** @internal */
export function requireRasterLength(view: GraphDataView | undefined, length: number): void {
  // Every present output must cover the requested dense rows.
  assert(!view || view.length >= length);
}
