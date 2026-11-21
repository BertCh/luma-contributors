// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {getEmptyGeometryNodes} from '../group-geometry/empty-geometry-nodes';

/**
 * Empty-route fills for refinement contributors. A zero-row column needs no write and binds
 * nothing, so it is omitted before the fill helper validates binding ranges; a minimally backed
 * zero-row view (smaller than one row) is therefore accepted on this route. @internal
 */
export function getRefinementEmptyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  columns: Readonly<Record<string, {view?: GraphDataView; value: number}>>
): GPUCommandNode<Parameters>[] {
  return getEmptyGeometryNodes(
    graph,
    id,
    operation,
    Object.fromEntries(
      Object.entries(columns).filter(([, column]) => column.view && column.view.length > 0)
    )
  );
}
