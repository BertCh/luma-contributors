// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {createGraphChunkGroupKernelNode} from '../../analysis-core/kernels/graph-chunk-groups';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Raster interpolation tiles share one pipeline and carry the global row base in a uniform. */
export function createInterpolationTiledNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>,
  tileSize?: number
): GPUCommandNode<Parameters>[] {
  const tileLength = Math.min(props.invocationCount, tileSize ?? props.invocationCount);
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let rowBase = 0; rowBase < props.invocationCount; rowBase += tileLength) {
    nodes.push(
      createGraphChunkGroupKernelNode(graph, {
        id: tileLength >= props.invocationCount ? props.id : `${props.id}-tile-${rowBase}`,
        pipelineId: props.id,
        operation: props.operation,
        variant: props.variant,
        bindings: props.bindings,
        workgroupSize: props.workgroupSize ?? 64,
        invocationCount: Math.min(tileLength, props.invocationCount - rowBase),
        rowBase,
        totalLength: props.invocationCount,
        declarations: props.declarations,
        body: `let interpolationRow = getChunkParameter(0u) + index;\n${props.body.replace(/\bindex\b/g, 'interpolationRow')}`
      })
    );
  }
  return nodes;
}
