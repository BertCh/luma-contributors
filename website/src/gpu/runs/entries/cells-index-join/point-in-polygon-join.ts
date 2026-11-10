// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointInPolygonJoin writes, for every point, the id of the zone that contains it. Points are
// colored by that zone id; points in no zone keep the muted input color.

import {GPUPointInPolygonJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import lakes from '../../../data/natural-earth-geometry.json';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {findNearestSegment} from '../../pick';
import {createRandomPoints, createZones, scaleZones, uploadZones} from './support/join';
import {parameters} from './point-in-polygon-join.parameters';

const POINT_COUNT = 900;
const ZONE_NAMES = Object.keys(lakes.greatLakes);

export default defineOperationRun({
  parameters,
  caption: 'Each point colored by the zone that contains it.',
  createInputs: ({random, world}) => ({
    zones: createZones(world),
    points: createRandomPoints(random, world, POINT_COUNT),
    // The last point row is the pointer.
    pointer: [world.width / 2, 0.5] as [number, number]
  }),
  build(builder, inputs) {
    const rowCount = POINT_COUNT + 1;
    const points = builder.allocate('points', 'float32x2', rowCount);
    const zones = uploadZones(builder, inputs.zones);
    const pointFeatureIds = builder.allocate('point-feature-ids', 'uint32', rowCount);
    const featureCounts = builder.allocate('feature-counts', 'uint32', inputs.zones.featureCount);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUPointInPolygonJoin({
        points: {kind: 'points', positions: points.view},
        polygons: zones.geometry,
        candidateCapacity: 1 << 14,
        output: {
          pointFeatureIds: pointFeatureIds.view,
          featureCounts: featureCounts.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {points, pointFeatureIds, featureCounts, zones},
      update: current => {
        points.write(
          Float32Array.from([
            ...Float32Array.from(inputs.points, (value, index) =>
              index % 2 === 0 ? value + current.shift : value
            ),
            ...inputs.pointer
          ])
        );
        const scaled = scaleZones(inputs.zones, current.size);
        zones.positions.write(scaled.positions);
        zones.outline.write(scaled.outline);
      }
    };
  },
  // The pointer is one more point in the join: the zone the operation wrote for it is lit.
  interaction: {
    kind: 'place',
    hint: 'move a point',
    rest: ({toUnit, inputs}) => toUnit([inputs.zones.centers[0], inputs.zones.centers[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const glyphs: RunGlyph[] = [
        {shape: 'crosshair', at: inputs.pointer, size: 8},
        {shape: 'dot', at: inputs.pointer}
      ];
      const zoneIds = peek(outputs.pointFeatureIds);
      const featureCounts = peek(outputs.featureCounts);
      let readout = 'zone …';
      if (zoneIds && featureCounts) {
        const zone = zoneIds[POINT_COUNT];
        if (zone >= featureCounts.length) {
          // Not joined: a dashed link to the nearest zone edge shows how far outside it is.
          const hit = findNearestSegment(
            scaleZones(inputs.zones, values.size).outline,
            pointer.x,
            pointer.y
          );
          if (hit) {
            const ring = inputs.zones.ringOffsets.findIndex(
              (offset, row) => hit.index >= offset && hit.index < inputs.zones.ringOffsets[row + 1]
            );
            glyphs.push(
              {shape: 'line', from: [pointer.x, pointer.y], to: hit.point, dashed: true},
              {shape: 'ring', at: hit.point, radius: 0.012}
            );
            readout = `joins no zone · ${hit.distance.toFixed(2)} from ${ZONE_NAMES[ring]}`;
          } else {
            readout = 'joins no zone';
          }
        } else {
          const scaled = scaleZones(inputs.zones, values.size);
          glyphs.push({
            shape: 'polyline',
            points: scaled.positions.subarray(
              inputs.zones.ringOffsets[zone] * 2,
              inputs.zones.ringOffsets[zone + 1] * 2
            ),
            closed: true
          });
          // The count includes the pointer's own row.
          readout = `in zone ${zone} (${ZONE_NAMES[zone]}) · ${featureCounts[zone] - 1} points`;
        }
      }
      return {rerun: true, glyphs, readout};
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'dots', positions: outputs.points, values: outputs.pointFeatureIds, radius: 3.5},
    {
      layer: 'lines',
      context: true,
      segments: outputs.zones.outline,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    }
  ]
});
