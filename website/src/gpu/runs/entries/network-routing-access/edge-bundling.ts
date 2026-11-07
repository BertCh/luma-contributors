// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUEdgeBundling: density-based bundling of flow lines. The muted straight lines are the input
// edges; the bright curves are each edge's bundled polyline, pulled toward the shared corridors.

import {
  GPUEdgeBundling,
  getGPUEdgeBundlingParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './edge-bundling.parameters';

const POINT_COUNT = 64;
const EDGE_COUNT = 140;
const POINTS_PER_EDGE = 16;
const ITERATIONS = 15;

export default defineOperationRun({
  parameters,
  caption: 'Straight flow lines (muted) and their bundled paths.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, POINT_COUNT, 3, 0);
    const sources = new Uint32Array(EDGE_COUNT);
    const targets = new Uint32Array(EDGE_COUNT);
    for (let edge = 0; edge < EDGE_COUNT; edge++) {
      sources[edge] = Math.floor(random() * POINT_COUNT);
      do targets[edge] = Math.floor(random() * POINT_COUNT);
      while (points.labels[targets[edge]] === points.labels[sources[edge]]);
    }
    const pairs = new Uint32Array(EDGE_COUNT * 2);
    for (let edge = 0; edge < EDGE_COUNT; edge++)
      pairs.set([sources[edge], targets[edge]], edge * 2);
    // Consecutive path points of every edge, as (point, next point) rows.
    const pathPairs = new Uint32Array(EDGE_COUNT * (POINTS_PER_EDGE - 1) * 2);
    let row = 0;
    for (let edge = 0; edge < EDGE_COUNT; edge++) {
      for (let step = 0; step < POINTS_PER_EDGE - 1; step++) {
        pathPairs.set(
          [edge * POINTS_PER_EDGE + step, edge * POINTS_PER_EDGE + step + 1],
          row++ * 2
        );
      }
    }
    return {points, sources, targets, pairs, pathPairs};
  },
  build(builder, inputs) {
    const nodes = builder.upload('nodes', 'float32x2', inputs.points.positions);
    const paths = builder.allocate('paths', 'float32x2', EDGE_COUNT * POINTS_PER_EDGE);
    const parameterBuffer = builder.allocate('parameters', 'float32', 5);
    builder.graph.add(
      new GPUEdgeBundling({
        nodePositions: nodes.view,
        sourceNodes: builder.upload('sources', 'uint32', inputs.sources).view,
        targetNodes: builder.upload('targets', 'uint32', inputs.targets).view,
        pointsPerEdge: POINTS_PER_EDGE,
        maximumIterations: ITERATIONS,
        parameters: parameterBuffer.view,
        output: {paths: paths.view}
      })
    );
    return {
      outputs: {
        nodes,
        straight: builder.upload('straight', 'uint32x2', inputs.pairs),
        paths,
        pathPairs: builder.upload('path-pairs', 'uint32x2', inputs.pathPairs)
      },
      update: values =>
        parameterBuffer.write(
          getGPUEdgeBundlingParameterValues(
            {
              activeIterations: values.activeIterations,
              kernelRadius: values.kernelRadius,
              smoothing: values.smoothing
            },
            'float32'
          )
        )
    };
  },
  draw: ({outputs}) => [
    {layer: 'segments', context: true, nodes: outputs.nodes, edges: outputs.straight, radius: 0.8},
    {
      layer: 'segments',
      nodes: outputs.paths,
      edges: outputs.pathPairs,
      radius: 1.2,
      color: [1, 0.78, 0.25, 0.55]
    }
  ]
});
