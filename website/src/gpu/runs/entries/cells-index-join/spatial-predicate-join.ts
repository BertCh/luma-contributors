// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialPredicateJoin pairs points with the zones they satisfy a predicate with: 'within' the
// zone, or 'dwithin' a distance of it. Each line is one matched (point, zone) pair the operation
// wrote, drawn from the point to the zone center.

import {GPUSpatialPredicateJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {findNearestSegment} from '../../pick';
import {addPairedRightIds, addPairSegments} from './support/kernels';
import {createRandomPoints, createZones, uploadZones} from './support/join';
import {findPointAtDistance} from './support/pointer';
import {parameters} from './spatial-predicate-join.parameters';

const POINT_COUNT = 360;
const PAIR_CAPACITY = 4096;
const DISTANCE_PARAMETER = findParameter(parameters, 'distance');

export default defineOperationRun({
  parameters,
  caption: 'Matched (point, zone) pairs; the lines are the join result.',
  createInputs: ({random, world}) => ({
    zones: createZones(world),
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
        count,
        pointZones: addPairedRightIds(builder, 'pairs', {
          rowCount: POINT_COUNT,
          leftIds,
          rightIds,
          count
        }),
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
  // Reach: the distance from the pointer to the nearest zone boundary is the dwithin distance.
  interaction: {
    kind: 'reach',
    hint: 'measure from the shore',
    rest: ({toUnit, inputs}) =>
      toUnit(
        findPointAtDistance(
          inputs.zones.outline,
          DISTANCE_PARAMETER.default,
          (x, y) => findNearestSegment(inputs.zones.outline, x, y)?.distance ?? Infinity
        ) ?? [0.5, 0.5]
      ),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const hit = findNearestSegment(inputs.zones.outline, pointer.x, pointer.y);
      if (!hit) return {};
      const distance = snapToParameter(DISTANCE_PARAMETER, hit.distance);
      const count = peek(outputs.count);
      return {
        values: {distance},
        glyphs: [
          {shape: 'line', from: hit.point, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: hit.point},
          {shape: 'ring', at: [pointer.x, pointer.y], radius: distance, dashed: true}
        ],
        readout: `${values.predicate === 0 ? `dwithin ${distance.toFixed(3)}` : 'within'} · ${count ? count[0] : '…'} pairs`
      };
    }
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
    {layer: 'lines', segments: outputs.pairs, radius: 0.8, color: [1, 0.7, 0.25, 0.22]},
    // Matched points take the color of the zone they were paired with.
    {layer: 'dots', positions: outputs.points, values: outputs.pointZones, radius: 3.6}
  ]
});
