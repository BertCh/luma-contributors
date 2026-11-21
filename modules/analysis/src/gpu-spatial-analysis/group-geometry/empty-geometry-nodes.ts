// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Defined fixed-row outputs for an empty resident geometry source. @internal */
export function getEmptyGeometryNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  columns: Readonly<Record<string, {view?: GraphDataView; value: number}>>
): GPUCommandNode<Parameters>[] {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(Object.entries(columns).map(([name, column]) => [name, column.view]))
  );
  return Object.entries(columns).flatMap(([name, {view, value}]) =>
    !view
      ? []
      : [
          createFillNode(graph, {
            id: `${id}-empty-${name}`,
            operation,
            view,
            type: view.format.startsWith('float') ? 'f32' : 'u32',
            value: Number.isNaN(value)
              ? 'bitcast<f32>(0x7fc00000u | (index & 0u))'
              : view.format.startsWith('float')
                ? `${value}.0`
                : `${value}u`
          })
        ]
  );
}
