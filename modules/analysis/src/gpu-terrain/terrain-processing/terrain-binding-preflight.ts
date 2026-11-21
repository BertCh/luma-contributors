// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GraphVectorView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  isGraphVectorView,
  isGraphDataView,
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_GATE_INTERVAL} from '../../raster-analysis/cost-distance/raster-relaxation';

/** Gate tables use 256-byte slots so every one-word subview is storage aligned. */
const TERRAIN_GATE_SLOT_BYTE_LENGTH = 256;

/** The shared extrema pyramid reduces four levels per sixteen-cell tile edge. */
const TERRAIN_PYRAMID_LEVELS_PER_PARAMETER_SLOT = 4;
/** A parameter slot contains six base words and four (width, height, offset) triples. */
const TERRAIN_PYRAMID_PARAMETER_BINDING_BYTE_LENGTH = (6 + 4 * 3) * 4;
/** Each fused dispatch addresses one storage-aligned slot, even though it binds only 72 bytes. */
const TERRAIN_PYRAMID_PARAMETER_SLOT_BYTE_LENGTH = 256;

/** Distinguishes a bound slot from the allocation containing all its aligned slots. @internal */
type TerrainScratchBinding = {bindingByteLength: number; allocationByteLength: number};

/** Scratch layout of the shared extrema pyramid's fused parameter upload. @internal */
export function getTerrainPyramidParameterScratch(
  layout: {levels: readonly unknown[]} | null | undefined
): TerrainScratchBinding | 0 {
  const dispatchCount = layout
    ? Math.ceil((layout.levels.length - 1) / TERRAIN_PYRAMID_LEVELS_PER_PARAMETER_SLOT)
    : 0;
  return dispatchCount > 0
    ? {
        bindingByteLength: TERRAIN_PYRAMID_PARAMETER_BINDING_BYTE_LENGTH,
        allocationByteLength: dispatchCount * TERRAIN_PYRAMID_PARAMETER_SLOT_BYTE_LENGTH
      }
    : 0;
}

/** Size of the complete table bound by the raster gated-loop initializer. @internal */
export function getTerrainGateScratchByteLength(iterations: number): number {
  return (
    (Math.min(RASTER_GATE_INTERVAL, Math.max(iterations, 1)) + 1) * TERRAIN_GATE_SLOT_BYTE_LENGTH
  );
}

/** Collects view metadata, never buffer contents or the graph owning a resource. @internal */
export function getTerrainBindingViews(
  fields: unknown
): Record<string, GraphDataView | GraphVectorView> {
  const views: Record<string, GraphDataView | GraphVectorView> = {};
  const visited = new Map<object, boolean>();
  const visit = (value: unknown, name: string): void => {
    if (!value || typeof value !== 'object' || visited.has(value)) return;
    visited.set(value, true);
    if (isGraphVectorView(value)) {
      views[name] = value;
    } else if (isGraphDataView(value)) {
      views[name] = value as GraphDataView;
    } else if ('words' in value) {
      visit(value.words, `${name}.words`);
    } else if (!('graph' in value) && !ArrayBuffer.isView(value)) {
      for (const [key, child] of Object.entries(value)) visit(child, name ? `${name}.${key}` : key);
    }
  };
  visit(fields, '');
  return views;
}

/**
 * Runs before graph declarations. Caller views use their actual aligned ranges and STORAGE flags.
 * Scratch entries bound the largest binding/allocation of each nested route: smaller ping-pong
 * planes and scan/reduction workspaces are covered by the same bound. Scratch is created with
 * STORAGE by the public transient-view helpers. This performs no alias or CPU data validation.
 * @internal
 */
export function validateTerrainBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  fields: unknown,
  scratch: Readonly<Record<string, number | TerrainScratchBinding>> = {}
): void {
  validateGraphChunkBindingRanges(graph, id, getTerrainBindingViews(fields));
  for (const [name, layout] of Object.entries(scratch)) {
    const byteLength = typeof layout === 'number' ? layout : layout.bindingByteLength;
    const allocationByteLength = typeof layout === 'number' ? layout : layout.allocationByteLength;
    if (byteLength === 0 && allocationByteLength === 0) continue;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'use smaller terrain tiles'
    );
    validateCapacity(
      id,
      `${name} allocation byteLength`,
      allocationByteLength,
      0,
      graph.device.limits.maxBufferSize
    );
  }
}
