// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNeighborSearch over clustered points. In knn mode each point links to its k nearest; in
// radius mode to every point within the radius. Points are colored by the neighbor count the
// operation wrote.

import {
  GPUNeighborSearch,
  getGPUNeighborSearchParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './neighbor-search.parameters';
import {addCardinality, addLinkSegments, allocateWeights, drawLinks} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Neighbor links found by the GPU grid search.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 260, 5, 0.12),
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const isRadius = values.mode === 1;
    const weights = allocateWeights(
      builder,
      'search',
      points.count,
      points.count * (isRadius ? 24 : values.k),
      true
    );
    builder.graph.add(
      new GPUNeighborSearch({
        mode: isRadius ? 'radius' : 'knn',
        k: isRadius ? undefined : values.k,
        positions: positions.view,
        gridSize: [24, 24],
        parameters: parameterBuffer.view,
        output: weights.port
      })
    );
    const cardinality = addCardinality(builder, 'search', weights);
    const links = addLinkSegments(builder, 'search', weights, positions);
    return {
      outputs: {positions, cardinality, links},
      update: current =>
        parameterBuffer.write(
          getGPUNeighborSearchParameterValues({
            bounds: [0, 0, ...worldBounds(points.positions)],
            radius: current.radius
          })
        )
    };
  },
  rebuildOn: ['mode', 'k'],
  draw: ({outputs}) => [
    drawLinks(outputs.links, [0.95, 0.55, 0.2, 0.45]),
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.cardinality,
      radius: 3.2
    }
  ]
});

function worldBounds(positions: Float32Array): [number, number] {
  let maximumX = 0;
  let maximumY = 0;
  for (let index = 0; index < positions.length; index += 2) {
    maximumX = Math.max(maximumX, positions[index]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  return [maximumX + 0.01, maximumY + 0.01];
}
