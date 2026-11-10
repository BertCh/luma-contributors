// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUEdgeBundling: density-based bundling of flow lines. The muted straight lines are the input
// edges; the bright curves are each edge's bundled polyline, pulled toward the shared corridors.

import {
  GPUEdgeBundling,
  getGPUEdgeBundlingParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {COMMUTE_FLOWS, getCommutePositions} from '../../../data/commute-flows';
import {bindAxes} from '../../interaction';
import {fitPositions} from './shared/graphs';
import {parameters} from './edge-bundling.parameters';

const EDGE_COUNT = 140;
const POINTS_PER_EDGE = 16;
const ITERATIONS = 15;

export default defineOperationRun({
  parameters,
  caption: 'The 140 largest commuter flows in the Chicago extract, bundled into shared corridors.',
  createInputs: ({world}) => {
    const positions = fitPositions(getCommutePositions(world), world);
    const flows = COMMUTE_FLOWS.slice(0, EDGE_COUNT);
    const sources = Uint32Array.from(flows, ([home]) => home);
    const targets = Uint32Array.from(flows, ([, work]) => work);
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
    return {positions, sources, targets, pairs, pathPairs};
  },
  build(builder, inputs) {
    const nodes = builder.upload('nodes', 'float32x2', inputs.positions);
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
  // x is the bundling iteration reached, y the kernel radius that pulls edges together.
  interaction: bindAxes(parameters, {
    kind: 'time',
    hint: 'scrub the bundling',
    x: 'activeIterations',
    y: 'kernelRadius',
    decorate: current => ({
      readout: `iteration ${current.activeIterations} / ${ITERATIONS} · kernelRadius ${current.kernelRadius.toFixed(3)}`
    })
  }),
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
