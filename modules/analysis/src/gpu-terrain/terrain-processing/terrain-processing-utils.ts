// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GraphDataView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  rasterBinding,
  validateRasterViews
} from '../../raster-analysis/remote-sensing/raster-kernel-utils';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Shares the raster validator: packed views, distinct output buffers, outputs disjoint from inputs and each other. @internal */
export function checkTerrainViews(
  id: string,
  inputs: GraphDataView[],
  outputs: GraphDataView[]
): void {
  validateRasterViews(id, inputs, outputs);
  assert(new Set(outputs.map(view => view.buffer)).size === outputs.length);
}
/** @internal */
export function checkTerrainGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: GraphDataView[]
): void {
  validateGraphViewsBelongToGraph(id, graph, views);
}
/** Maps uint32 to u32, sint32 to i32 and float32 to f32. @internal */
export function terrainBinding(
  name: string,
  view: GraphDataView,
  access: 'read' | 'read_write' = 'read'
): WGSLKernelBinding {
  return rasterBinding(name, view, access === 'read_write');
}
/** @internal */
export const TERRAIN_FINITE_WGSL = `fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn missing() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
fn infinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }`;
