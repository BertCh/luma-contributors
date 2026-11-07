// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialPredicateJoin pairs points with the zones they satisfy a predicate with: 'within' the
// zone, or 'dwithin' a distance of it. Each line is one matched (point, zone) pair the operation
// wrote, drawn from the point to the zone center.

import {GPUSpatialPredicateJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPairSegments} from './support/kernels';
import {createRandomPoints, createZones, uploadZones} from './support/join';
import {parameters} from './spatial-predicate-join.parameters';

const POINT_COUNT = 360;
const PAIR_CAPACITY = 4096;

export default defineOperationRun({
  parameters,
  caption: 'Matched (point, zone) pairs; the lines are the join result.',
  createInputs: ({random, world}) => ({
    zones: createZones(random, world),
    points: createRandomPoints(random, world, POINT_COUNT)
  }),
  build(builder, inputs, values) {
    const points = builder.upload('points', 'float32x2', inputs.points);
    const zones = uploadZones(builder, inputs.zones);
    const distance = builder.allocate('distance', 'float32', 1);
    const leftIds = builder.allocate('left-ids', 'uint32', PAIR_CAPACITY);
    const rightIds = builder.allocate('right-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const isDistance = values.predicate === 0;
    builder.graph.add(
      new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: points.view},
        right: zones.geometry,
        predicate: isDistance ? 'dwithin' : 'within',
        distance: isDistance ? distance.view : undefined,
        candidateCapacity: PAIR_CAPACITY,
        output: {
          pairs: {
            leftIds: leftIds.view,
            rightIds: rightIds.view,
            status: {count: count.view, overflow: overflow.view}
          }
        }
      })
    );
    return {
      outputs: {
        points,
        zones,
        pairs: addPairSegments(builder, 'pairs', {
          leftPositions: points,
          rightPositions: zones.centers,
          leftIds,
          rightIds,
          count
        })
      },
      update: current => distance.write(Float32Array.of(current.distance))
    };
  },
  rebuildOn: ['predicate'],
  draw: ({outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.zones.outline,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    },
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'lines', segments: outputs.pairs, radius: 1, color: [1, 0.7, 0.25, 0.8]}
  ]
});
