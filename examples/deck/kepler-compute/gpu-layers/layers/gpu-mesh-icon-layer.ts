// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUData} from '@luma.gl/graph/gpu-data';
import {GPUScatterplotLayer, type GPUScatterplotLayerProps} from './gpu-scatterplot-layer';

/** Exact triangle icons using borrowed mesh and instance storage. */
export type GPUMeshIconLayerProps = Omit<
  GPUScatterplotLayerProps,
  | 'getLineColor'
  | 'getLineWidth'
  | 'stroked'
  | 'filled'
  | 'lineWidthScale'
  | 'lineWidthMinPixels'
  | 'lineWidthMaxPixels'
  | 'lineWidthUnits'
> & {
  /** Unit-size triangle-list vertices, length divisible by three. XY defines the shape;
   * local Z is retained in ground-plane mode. No readback, rasterization or mesh upload occurs.
   * Native SVG adapters supply their normalized [x, -y, z] tessellation unchanged.
   */
  iconGeometry: GPUData<'float32x3'>;
};

/**
 * Instanced SVG-compatible triangle meshes with GPUScatterplotLayer's position, radius,
 * billboard, chunking, selection and picking contracts. The fragment shader fills the exact
 * triangles, including tessellated holes, without circular clipping. Geometry and all source
 * vectors remain caller-owned; only the model, uniform storage and default attributes are owned.
 * Antialiasing follows native mesh-icon vertex padding; triangle edges use render-target sampling.
 */
export class GPUMeshIconLayer extends GPUScatterplotLayer<GPUMeshIconLayerProps> {
  static override layerName = 'GPUMeshIconLayer';

  protected override getIconGeometry(): GPUData<'float32x3'> {
    const geometry = this.props.iconGeometry;
    // Icons require fixed-width XYZ triangles; null vertices are not a supported mesh topology.
    assert(geometry.format === 'float32x3' && geometry.length % 3 === 0 && !geometry.nullBitmap);
    return geometry;
  }
}
