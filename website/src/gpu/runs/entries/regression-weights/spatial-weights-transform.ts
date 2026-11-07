// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsTransform on a distance-band graph. Left: points colored by the sum of their
// transformed weights (1 for `row`). Right: every transformed weight against its link distance,
// which traces the kernel profile for `kernel`.

import {GPUSpatialWeightsTransform} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createUniformPoints} from '../../inputs';
import {parameters} from './spatial-weights-transform.parameters';
import {addLinkSegments, addRowSums, uploadWeightsBuffers} from './shared/weights';

const OPERATIONS = ['row', 'binary', 'kernel', 'double', 'variance', 'symmetrize'] as const;
const KERNELS = ['gaussian', 'triangular', 'epanechnikov', 'bisquare', 'uniform'] as const;
const BAND_RADIUS = 0.1;

export default defineOperationRun({
  parameters,
  caption: 'Transformed weights per link (right) and their row sums (left).',
  createInputs: ({random, world}) =>
    createUniformPoints(random, {width: world.width * 0.55, height: world.height}, 220, 0.1),
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const band = uploadWeightsBuffers(
      builder,
      'band',
      createDistanceBandWeights(points.positions, BAND_RADIUS)
    );
    const transformed = builder.allocate('transformed', 'float32', band.capacity);
    builder.graph.add(
      new GPUSpatialWeightsTransform({
        operation: OPERATIONS[values.operation],
        weights: band.input,
        output: transformed.view,
        kernel: KERNELS[values.kernel],
        bandwidth: values.bandwidth
      })
    );
    return {
      outputs: {
        positions,
        distances: band.distances!,
        transformed,
        rowSums: addRowSums(builder, 'transformed', band, transformed),
        links: addLinkSegments(builder, 'band', band, positions)
      }
    };
  },
  rebuildOn: ['operation', 'kernel', 'bandwidth'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.links},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.rowSums,
      radius: 3.4
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.transformed,
      xValues: outputs.distances,
      xRange: [0, BAND_RADIUS],
      rectangle: [0.62, 0.14, 0.96, 0.86],
      radius: 2
    }
  ]
});
