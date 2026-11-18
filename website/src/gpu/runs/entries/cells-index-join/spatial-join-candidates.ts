// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialJoinCandidates is the cheap first stage of a spatial join: it pairs every point with
// each zone whose bounding box (grown by `distance`) contains it, without testing the exact
// geometry. Lines are the candidate pairs; many reach points outside the zone itself.

import {GPUSpatialJoinCandidates} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findNearestSegment} from '../../pick';
import {findParameter, snapToParameter, type RunGlyph} from '../../interaction';
import {addPairSegments} from './support/kernels';
import {
  createBoundsSegments,
  createRandomPoints,
  createZones,
  getZoneBounds,
  scaleZones,
  uploadZones
} from './support/join';
import {findPointAtDistance} from './support/pointer';
import {parameters} from './spatial-join-candidates.parameters';

const POINT_COUNT = 300;
const PAIR_CAPACITY = 4096;
const DISTANCE_PARAMETER = findParameter(parameters, 'distance');

export default defineOperationRun({
  parameters,
  caption: 'Bounding-box candidate pairs: a superset of the exact matches.',
  createInputs: ({random, world}) => ({
    zones: createZones(world),
    points: createRandomPoints(random, world, POINT_COUNT)
  }),
  build(builder, inputs) {
    const points = builder.upload('points', 'float32x2', inputs.points);
    const zones = uploadZones(builder, inputs.zones);
    const distance = builder.allocate('distance', 'float32', 1);
    const boxes = builder.allocate('zone-bounds', 'float32x4', inputs.zones.featureCount * 4);
    const leftRows = builder.allocate('left-ids', 'uint32', PAIR_CAPACITY);
    const rightRows = builder.allocate('right-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('count', 'uint32', 1);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUSpatialJoinCandidates({
        left: {kind: 'points', positions: points.view},
        right: zones.geometry,
        distance: distance.view,
        output: {
          leftRows: leftRows.view,
          rightRows: rightRows.view,
          status: {count: count.view, overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {
        points,
        zones,
        boxes,
        count,
        pairs: addPairSegments(builder, 'candidates', {
          leftPositions: points,
          rightPositions: zones.centers,
          leftRows,
          rightRows,
          count
        })
      },
      update: current => {
        distance.write(Float32Array.of(current.distance));
        const scaled = scaleZones(inputs.zones, current.size);
        zones.positions.write(scaled.positions);
        zones.outline.write(scaled.outline);
        // The bounding boxes the operation tests against, grown by the current distance.
        boxes.write(
          createBoundsSegments(getZoneBounds(inputs.zones, scaled.positions), current.distance)
        );
      }
    };
  },
  // Reach: the distance from the pointer to the nearest bounding-box edge grows every box.
  interaction: {
    kind: 'reach',
    hint: 'grow the boxes',
    rest: ({toUnit, inputs, values}) => {
      const edges = createBoundsSegments(
        getZoneBounds(inputs.zones, scaleZones(inputs.zones, values.size).positions)
      );
      return toUnit(
        findPointAtDistance(
          edges,
          DISTANCE_PARAMETER.default,
          (x, y) => findNearestSegment(edges, x, y)?.distance ?? Infinity
        ) ?? [0.5, 0.5]
      );
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      const edges = createBoundsSegments(
        getZoneBounds(inputs.zones, scaleZones(inputs.zones, values.size).positions)
      );
      const nearest = findNearestSegment(edges, pointer.x, pointer.y);
      if (!nearest) return {};
      const distance = snapToParameter(DISTANCE_PARAMETER, nearest.distance);
      const count = peek(outputs.count);
      const glyphs: RunGlyph[] = [
        {shape: 'line', from: nearest.point, to: [pointer.x, pointer.y]},
        {shape: 'dot', at: nearest.point},
        {shape: 'crosshair', at: [pointer.x, pointer.y], size: 6}
      ];
      return {
        values: {distance},
        glyphs,
        readout: `distance ${distance.toFixed(3)} · ${count ? count[0] : '…'} candidate pairs`
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
    {
      layer: 'lines',
      segments: outputs.boxes,
      radius: 1.2,
      color: [1, 0.85, 0.3, 0.9],
      context: true
    },
    {layer: 'lines', segments: outputs.pairs, radius: 1, color: [0.45, 0.75, 1, 0.75]}
  ]
});
