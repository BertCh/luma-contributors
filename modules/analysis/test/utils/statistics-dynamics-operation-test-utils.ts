// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUGraphVectorFormat,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUContributorOperationViewGetter} from '../../src/analysis-core/operations/gpu-contributor-operation';

/** Resolves operation ports to real graph views on the null device for constructor checks. */
export function createOperationTestViewGetter(
  graph: GPUCommandGraph,
  ports: Record<string, {format: GPUGraphVectorFormat; length: number}>
): GPUContributorOperationViewGetter {
  const views = new Map<string, GraphDataView>();
  return <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> => {
    const descriptor = ports[port];
    if (!descriptor) throw new Error(`Missing test port ${port}`);
    let view = views.get(port);
    if (!view) {
      view = createTransientView(graph, port, descriptor.format, descriptor.length);
      views.set(port, view);
    }
    return view as GraphDataView<Format>;
  };
}
