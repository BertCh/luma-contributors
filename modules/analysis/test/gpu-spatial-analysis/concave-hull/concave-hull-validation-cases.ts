// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getDelaunayMaximumTriangleCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';

/** Public packed views shared by node and browser constructor-validation cases. */
export function createConcaveHullValidationProps(
  graph: GPUCommandGraph,
  rows = 4,
  groups = 1,
  triangleCapacity = getDelaunayMaximumTriangleCount(rows) * groups
) {
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    maximumCircumradius: 1,
    output: {
      triangles: createTransientView(graph, 'triangles', 'uint32x3', triangleCapacity),
      triangleValidity: createTransientView(graph, 'triangle-validity', 'uint32', triangleCapacity),
      boundaryEdges: createTransientView(graph, 'edges', 'uint32x2', 3 * triangleCapacity),
      edgeValidity: createTransientView(graph, 'edge-validity', 'uint32', 3 * triangleCapacity),
      status: {converged: createTransientView(graph, 'converged', 'uint32', groups)}
    }
  };
}
