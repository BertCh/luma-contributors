// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import type {GPUCommandGraph, GPUCommandNode} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelProps} from './wgsl-kernel-nodes';

/**
 * Splits one linear `createWGSLKernelNode` kernel into consecutive row tiles, one dispatch each.
 *
 * Every tile shares one shader module: the tile base is the pipeline constant `TILE_ROW_BASE` and
 * a nested block shadows `index` with the global row index, so the original `body` and any
 * per-row random stream or self-exclusion keyed by `index` are unchanged. Results therefore do not
 * depend on `tileSize`. With no `tileSize`, or one at least `invocationCount`, this is exactly one
 * ordinary node. `body` must not declare a module-scope name `TILE_ROW_BASE`.
 *
 * @internal
 */
export function createTiledWGSLKernelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>,
  tileSize?: number
): GPUCommandNode<Parameters>[] {
  const rowCount = props.invocationCount;
  const tileRowCount = clamp(rowCount, 1, tileSize ?? rowCount);
  if (tileRowCount >= rowCount) {
    return [createWGSLKernelNode<Parameters>(graph, props)];
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let tileBase = 0; tileBase < rowCount; tileBase += tileRowCount) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        ...props,
        id: `${props.id}-tile-${tileBase}`,
        invocationCount: Math.min(tileRowCount, rowCount - tileBase),
        constants: {...props.constants, TILE_ROW_BASE: tileBase},
        declarations: `override TILE_ROW_BASE: u32;\n${props.declarations ?? ''}`,
        body: `{ let index = index + TILE_ROW_BASE;\n${props.body}\n}`
      })
    );
  }
  return nodes;
}
