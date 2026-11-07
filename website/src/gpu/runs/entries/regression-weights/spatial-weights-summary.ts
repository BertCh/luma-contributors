// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsSummary of a distance-band graph whose weights carry noise. Left: points colored
// by the per-row cardinality the operation wrote. Right: the six counts (slots, asymmetric slots,
// isolates, minimum and maximum cardinality, empty rows) as bars of log(1 + count).

import {
  GPUSpatialWeightsSummary,
  GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createUniformPoints} from '../../inputs';
import {parameters} from './spatial-weights-summary.parameters';
import {addLinkSegments, addLogCounts, uploadWeightsBuffers} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Cardinality per point (left) and the summary counts (right, log(1 + count)).',
  createInputs: ({random, world}) =>
    createUniformPoints(random, {width: world.width * 0.55, height: world.height}, 200, 0.1),
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const csr = createDistanceBandWeights(points.positions, 0.09);
    const band = uploadWeightsBuffers(builder, 'band', csr);
    const statistics = builder.allocate(
      'statistics',
      'float32',
      GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.statistics.length
    );
    const counts = builder.allocate(
      'counts',
      'uint32',
      GPU_SPATIAL_WEIGHTS_SUMMARY_LAYOUT.counts.length
    );
    const cardinality = builder.allocate('cardinality', 'uint32', points.count);
    builder.graph.add(
      new GPUSpatialWeightsSummary({
        weights: band.input,
        symmetryTolerance: values.symmetryTolerance,
        output: {
          statistics: statistics.view,
          counts: counts.view,
          cardinality: cardinality.view
        }
      })
    );
    return {
      outputs: {
        positions,
        loggedCounts: addLogCounts(builder, 'summary', counts),
        cardinality,
        links: addLinkSegments(builder, 'band', band, positions)
      },
      // Independent noise on every slot, so w_ij and w_ji differ.
      update: current =>
        band.weights.write(
          Float32Array.from({length: band.capacity}, (_, slot) => {
            const hash = Math.sin(slot * 12.9898) * 43758.5453;
            return 1 + (hash - Math.floor(hash) - 0.5) * current.noise;
          })
        )
    };
  },
  rebuildOn: ['symmetryTolerance'],
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.links},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.cardinality,
      ramp: 'sequential',
      radius: 3.6
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.loggedCounts,
      ramp: 'sequential',
      rectangle: [0.64, 0.14, 0.96, 0.86]
    }
  ]
});
