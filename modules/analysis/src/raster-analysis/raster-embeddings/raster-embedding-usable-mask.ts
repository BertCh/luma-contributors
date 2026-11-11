// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {RASTER_FINITE_WGSL, rasterBinding} from '../remote-sensing/raster-kernel-utils';

/**
 * Builds the per-pixel mask the reductions of {@link GPURasterEmbeddingCovariance} and
 * {@link GPURasterEmbeddingZoneMean} share: 1 where `mask` is nonzero (when given) and every
 * component of the vector is finite. A reduction that read each component separately would
 * weight the bands differently when only some of a pixel's bands are missing.
 *
 * @internal
 */
export function createRasterEmbeddingUsableMask<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    pixelCount: number;
    dimensions: number;
    vectors: GraphDataView<'float32'>;
    mask?: GraphDataView<'uint32'>;
  }
): {usable: GraphDataView<'uint32'>; node: GPUCommandNode<Parameters>} {
  const usable = createTransientView(graph, `${props.id}-usable`, 'uint32', props.pixelCount);
  const node = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-usable`,
    operation: props.operation,
    variant: 'usable-mask',
    invocationCount: props.pixelCount,
    workgroupSize: 64,
    bindings: [
      rasterBinding('vectors', props.vectors),
      ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
      rasterBinding('usable', usable, true)
    ],
    declarations: RASTER_FINITE_WGSL,
    body: `var isUsable = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
for (var band = 0u; band < ${props.dimensions}u; band++) {
  if (!finite(vectors[vectorsOffset + index * ${props.dimensions}u + band])) { isUsable = false; }
}
usable[usableOffset + index] = select(0u, 1u, isUsable);`
  });
  return {usable, node};
}
