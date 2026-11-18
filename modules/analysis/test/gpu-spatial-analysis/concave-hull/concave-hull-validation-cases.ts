// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Device} from '@luma.gl/core';
import {getDelaunayMaximumTriangleCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createOutputBuffer} from '../../utils/gpu-contributor-test-utils';

/** Caller-owned output allocation shared by CPU preflight and headless execution fixtures. */
export function createConcaveHullTestOutput<Format extends 'uint32' | 'uint32x2' | 'uint32x3'>(
  graph: GPUCommandGraph,
  device: Device,
  name: string,
  format: Format,
  length: number
) {
  const components = format === 'uint32x3' ? 3 : format === 'uint32x2' ? 2 : 1;
  // Published getViewBindingRange binds at least one physical row, including for length 0.
  const buffer = createOutputBuffer(device, Math.max(length, 1) * components);
  return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
}

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
