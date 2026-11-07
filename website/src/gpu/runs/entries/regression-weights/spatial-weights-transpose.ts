// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsTranspose of directed k-nearest-neighbor weights. Every point has k outgoing
// links; the transpose lists the incoming ones, so points colored by their transposed row length
// are the hubs that many others pick as neighbor.

import {GPUSpatialWeightsTranspose} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints, createKnnWeights} from '../../inputs';
import {parameters} from './spatial-weights-transpose.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  createSeededRandom,
  uploadWeightsBuffers
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Points colored by in-degree: the length of their row in the transpose.',
  createInputs: ({seed, world}) => ({seed, world}),
  build(builder, {seed, world}, values) {
    const points = createClusteredPoints(createSeededRandom(seed), world, 180, 5, values.scatter);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const directed = uploadWeightsBuffers(
      builder,
      'directed',
      createKnnWeights(points.positions, values.k)
    );
    const transposed = allocateWeights(
      builder,
      'transposed',
      points.count,
      directed.capacity,
      true
    );
    builder.graph.add(
      new GPUSpatialWeightsTranspose({
        weights: directed.input,
        output: transposed.port
      })
    );
    return {
      outputs: {
        positions,
        inDegree: addCardinality(builder, 'transposed', transposed),
        links: addLinkSegments(builder, 'transposed', transposed, positions)
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'lines',
      segments: outputs.links,
      color: [0.6, 0.65, 0.85, 0.4],
      radius: 1.1
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.inDegree,
      radius: 3.6
    }
  ]
});
