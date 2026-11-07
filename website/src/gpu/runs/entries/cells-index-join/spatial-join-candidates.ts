// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialJoinCandidates is the cheap first stage of a spatial join: it pairs every point with
// each zone whose bounding box (grown by `distance`) contains it, without testing the exact
// geometry. Lines are the candidate pairs; many reach points outside the zone itself.

import {GPUSpatialJoinCandidates} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPairSegments} from './support/kernels';
import {createRandomPoints, createZones, scaleZones, uploadZones} from './support/join';
import {parameters} from './spatial-join-candidates.parameters';

const POINT_COUNT = 300;
const PAIR_CAPACITY = 4096;

export default defineOperationRun({
  parameters,
  caption: 'Bounding-box candidate pairs: a superset of the exact matches.',
  createInputs: ({random, world}) => ({
    zones: createZones(random, world),
    points: createRandomPoints(random, world, POINT_COUNT)
  }),
  build(builder, inputs) {
    const points = builder.upload('points', 'float32x2', inputs.points);
    const zones = uploadZones(builder, inputs.zones);
    const distance = builder.allocate('distance', 'float32', 1);
    const leftIds = builder.allocate('left-ids', 'uint32', PAIR_CAPACITY);
    const rightIds = builder.allocate('right-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUSpatialJoinCandidates({
        left: {kind: 'points', positions: points.view},
        right: zones.geometry,
        distance: distance.view,
        output: {
          leftIds: leftIds.view,
          rightIds: rightIds.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {
        points,
        zones,
        pairs: addPairSegments(builder, 'candidates', {
          leftPositions: points,
          rightPositions: zones.centers,
          leftIds,
          rightIds,
          count
        })
      },
      update: current => {
        distance.write(Float32Array.of(current.distance));
        const scaled = scaleZones(inputs.zones, current.size);
        zones.positions.write(scaled.positions);
        zones.outline.write(scaled.outline);
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.zones.outline,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    },
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'lines', segments: outputs.pairs, radius: 1, color: [0.45, 0.75, 1, 0.75]}
  ]
});
