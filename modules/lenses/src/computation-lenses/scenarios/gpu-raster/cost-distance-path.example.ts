// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  type GPUCommandGraph,
  type GraphDataView,
  GPUHistogram,
  createTransientView
} from '@luma.gl/graph/gpu-core';
import {type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUParameterBuffer, createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {
  GPUCostDistance,
  GPU_COST_DISTANCE_PARAMETER_LENGTH,
  GPUCostDistancePath,
  type GPUCostDistancePathOutput,
  makeGPURasterBand,
  getGPUCostDistanceParameterValues
} from '@luma-contributors/analysis/raster-analysis';

/** Graph, grid and caller-owned views for cost-distance-path. */
export type CostDistancePathExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  friction: GraphDataView<'float32'>;
  sourceMask: GraphDataView<'uint32'>;
  costs: GraphDataView<'float32'>;
  backLinks: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationCount: GraphDataView<'uint32'>;
  targets: {
    target: GraphDataView<'uint32'>;
    path: GPUCostDistancePathOutput['path'];
    mask: GraphDataView<'uint32'>;
  }[];
};

/** Adds the recorded cost-distance-path computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: CostDistancePathExampleInput) {
  const {graph, grid, friction, sourceMask, costs, backLinks, converged, iterationCount, targets} =
    input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'cost-path-parameters',
    format: 'float32',
    length: GPU_COST_DISTANCE_PARAMETER_LENGTH,
    values: getGPUCostDistanceParameterValues({}, grid)
  });
  graph.add(
    new GPUCostDistance({
      grid,
      friction: makeGPURasterBand(friction, grid),
      sourceMask,
      maximumIterations: 128,
      parameters: parameters.importToGraph(graph),
      output: {costs, backLinks, status: {converged, iterationCount}}
    })
  );
  for (const [index, {target, path, mask}] of targets.entries()) {
    graph.add(
      new GPUCostDistancePath({id: `walk-${index}`, grid, backLinks, target, output: {path}})
    );
    addCostDistancePathMembership(graph, path, mask, `mask-${index}`);
  }
  // #endregion example
  return {parameters};
}

/** Count bounds the readable ID prefix; every spare membership row is overwritten on every run. */
export function addCostDistancePathMembership(
  graph: GPUCommandGraph,
  path: GPUCostDistancePathOutput['path'],
  membership: GraphDataView<'uint32'>,
  id: string
): void {
  const ids = path.ids;
  const count = path.status.count;
  const rowMask = createTransientView(graph, `${id}-rows`, 'uint32', ids.length);
  graph.add({
    getCommandNodes: currentGraph => [
      createWGSLKernelNode(currentGraph, {
        id: `${id}-prefix`,
        operation: 'CostDistancePathMembership',
        invocationCount: ids.length,
        bindings: [
          {name: 'count', view: count, type: 'u32', access: 'read'},
          {name: 'rowMask', view: rowMask, type: 'u32', access: 'read_write'}
        ],
        body: 'rowMask[rowMaskOffset + index] = select(0u, 1u, index < count[countOffset]);'
      })
    ]
  });
  graph.add(
    new GPUHistogram({
      id,
      input: ids,
      mask: rowMask,
      domain: [0, membership.length],
      output: membership
    })
  );
}
