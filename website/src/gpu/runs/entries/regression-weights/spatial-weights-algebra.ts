// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsAlgebra combines two weights: k nearest neighbors (blue) and a distance band
// (green). The thick orange links underneath are the combined weights the operation wrote.

import {GPUSpatialWeightsAlgebra} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createKnnWeights, createUniformPoints} from '../../inputs';
import {parameters} from './spatial-weights-algebra.parameters';
import {addLinkSegments, allocateWeights, uploadWeightsBuffers} from './shared/weights';

const OPERATIONS = ['union', 'intersection', 'difference', 'symmetricDifference'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Left (blue) and right (green) weights, and their combination (orange).',
  createInputs: ({random, world}) => createUniformPoints(random, world, 70, 0.1),
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const left = uploadWeightsBuffers(
      builder,
      'left',
      createKnnWeights(points.positions, values.k)
    );
    const right = uploadWeightsBuffers(
      builder,
      'right',
      createDistanceBandWeights(points.positions, values.radius)
    );
    const combined = allocateWeights(
      builder,
      'combined',
      points.count,
      left.capacity + right.capacity
    );
    builder.graph.add(
      new GPUSpatialWeightsAlgebra({
        operation: OPERATIONS[values.operation],
        left: left.input,
        right: right.input,
        output: combined.port
      })
    );
    return {
      outputs: {
        positions,
        leftLinks: addLinkSegments(builder, 'left', left, positions),
        rightLinks: addLinkSegments(builder, 'right', right, positions),
        combinedLinks: addLinkSegments(builder, 'combined', combined, positions)
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'lines',
      segments: outputs.combinedLinks,
      color: [1, 0.55, 0.2, 0.9],
      radius: 3.2
    },
    {
      layer: 'lines',
      segments: outputs.leftLinks,
      color: [0.35, 0.6, 1, 0.9],
      radius: 1
    },
    {
      layer: 'lines',
      segments: outputs.rightLinks,
      color: [0.35, 0.85, 0.5, 0.9],
      radius: 1
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3}
  ]
});
