// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNetworkLouvain: modularity communities of a small weighted graph. Nodes are 32 places whose
// tie strength decays with distance (the muted lines); each dot takes the color of its community.
// A higher resolution splits the graph into more, smaller communities.

import {GPUNetworkLouvain} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './network-louvain.parameters';

const NODE_COUNT = 32;

export default defineOperationRun({
  parameters,
  caption: 'Communities (colors) of a distance-weighted tie graph; ties are the muted lines.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, NODE_COUNT, 4, 0.1);
    return {
      positions: points.positions,
      weights: new Float32Array(NODE_COUNT * NODE_COUNT),
      lines: new Float32Array(NODE_COUNT * NODE_COUNT * 4),
      lineCount: {value: 0}
    };
  },
  build(builder, inputs) {
    const weights = builder.allocate('weights', 'float32', NODE_COUNT * NODE_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const communities = builder.allocate('communities', 'uint32', NODE_COUNT);
    const objectiveTrace = builder.allocate('objective-trace', 'float32', 129);
    const summary = builder.allocate('summary', 'uint32', 4);
    builder.graph.add(
      new GPUNetworkLouvain({
        nodeCount: NODE_COUNT,
        seed: 4332,
        weights: weights.view,
        parameters: parameterBuffer.view,
        output: {
          communities: communities.view,
          objectiveTrace: objectiveTrace.view,
          summary: summary.view
        }
      })
    );
    const ties = builder.upload('ties', 'float32x4', inputs.lines);
    return {
      outputs: {
        communities,
        ties,
        positions: builder.upload('positions', 'float32x2', inputs.positions)
      },
      update: values => {
        // Symmetric weights decaying with distance; weak ties are dropped.
        let lineCount = 0;
        for (let first = 0; first < NODE_COUNT; first++) {
          for (let second = 0; second < NODE_COUNT; second++) {
            const distance = Math.hypot(
              inputs.positions[first * 2] - inputs.positions[second * 2],
              inputs.positions[first * 2 + 1] - inputs.positions[second * 2 + 1]
            );
            const weight = first === second ? 0 : Math.exp(-((distance / values.sigma) ** 2));
            const kept = weight > 0.08 ? weight : 0;
            inputs.weights[first * NODE_COUNT + second] = kept;
            if (kept > 0 && first < second) {
              inputs.lines.set(
                [
                  inputs.positions[first * 2],
                  inputs.positions[first * 2 + 1],
                  inputs.positions[second * 2],
                  inputs.positions[second * 2 + 1]
                ],
                lineCount++ * 4
              );
            }
          }
        }
        inputs.lines.fill(-100, lineCount * 4);
        inputs.lineCount.value = lineCount;
        weights.write(inputs.weights);
        ties.write(inputs.lines);
        parameterBuffer.write(Float32Array.of(values.resolution, 0.000001));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.ties, radius: 1.2},
    {layer: 'dots', positions: outputs.positions, values: outputs.communities, radius: 9}
  ]
});
