// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUIslandAttachment on a distance-band graph. Points with no neighbor in the band (islands) get
// an edge to their nearest neighbor. Band links are muted; the added links are drawn bright, and
// points are colored by their neighbor count after attachment.

import {GPUIslandAttachment} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints, createDistanceBandWeights} from '../../inputs';
import {parameters} from './island-attachment.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  uploadWeightsBuffers
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Isolated points are attached to their nearest neighbor.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 200, 4, 0.18),
  build(builder, points, values) {
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const band = uploadWeightsBuffers(
      builder,
      'band',
      createDistanceBandWeights(points.positions, values.radius)
    );
    const attached = allocateWeights(
      builder,
      'attached',
      points.count,
      band.capacity + points.count * 2
    );
    const attachment = builder.allocate('attachment', 'uint32', points.count);
    const added = builder.allocate('added', 'uint32', attached.capacity);
    builder.graph.add(
      new GPUIslandAttachment({
        weights: band.input,
        positions: positions.view,
        reciprocal: values.reciprocal === 1,
        output: {
          ...attached.port,
          attachment: attachment.view,
          added: added.view
        }
      })
    );
    const cardinality = addCardinality(builder, 'attached', attached);
    return {
      outputs: {
        positions,
        cardinality,
        bandLinks: addLinkSegments(builder, 'band', band, positions),
        addedLinks: addLinkSegments(builder, 'added', attached, positions, {onlyWhere: added})
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.bandLinks},
    {
      layer: 'lines',
      segments: outputs.addedLinks,
      color: [1, 0.45, 0.2, 0.95],
      radius: 1.8
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.cardinality,
      radius: 3.4,
      range: [0, 8]
    }
  ]
});
