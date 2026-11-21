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
// A lattice of probe points follows the pointer; the first rows of the join are the lattice.
const LATTICE_COLUMNS = 9;
const LATTICE_ROWS = 5;
const LATTICE_COUNT = LATTICE_COLUMNS * LATTICE_ROWS;
const LATTICE_SPACING = 0.04;
const CENTER_ROW = (LATTICE_ROWS >> 1) * LATTICE_COLUMNS + (LATTICE_COLUMNS >> 1);
const ZONE_NAMES = Object.keys(lakes.greatLakes);

export default defineOperationRun({
  parameters,
  caption: 'Each point colored by the zone that contains it.',
  createInputs: ({random, world}) => ({
    zones: createZones(world),
    points: createRandomPoints(random, world, POINT_COUNT),
    // The pointer is the center of a lattice of probe points.
    pointer: [world.width / 2, 0.5] as [number, number]
  }),
  build(builder, inputs) {
    const rowCount = LATTICE_COUNT + POINT_COUNT;
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
          polygonAssignmentCounts: featureCounts.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {points, pointFeatureIds, featureCounts, zones},
      update: current => {
        const lattice = new Float32Array(LATTICE_COUNT * 2);
        for (let row = 0; row < LATTICE_COUNT; row++) {
          lattice[row * 2] =
            inputs.pointer[0] +
            ((row % LATTICE_COLUMNS) - (LATTICE_COLUMNS >> 1)) * LATTICE_SPACING;
          lattice[row * 2 + 1] =
            inputs.pointer[1] +
            (Math.floor(row / LATTICE_COLUMNS) - (LATTICE_ROWS >> 1)) * LATTICE_SPACING;
        }
        points.write(
          Float32Array.from([
            ...lattice,
            ...Float32Array.from(inputs.points, (value, index) =>
              index % 2 === 0 ? value + current.shift : value
            )
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
    hint: 'move the probe lattice',
    rest: ({toUnit, inputs}) => toUnit([inputs.zones.centers[0], inputs.zones.centers[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const glyphs: RunGlyph[] = [
        {
          shape: 'box',
          minimum: [
            pointer.x - (LATTICE_COLUMNS >> 1) * LATTICE_SPACING - 0.02,
            pointer.y - (LATTICE_ROWS >> 1) * LATTICE_SPACING - 0.02
          ],
          maximum: [
            pointer.x + (LATTICE_COLUMNS >> 1) * LATTICE_SPACING + 0.02,
            pointer.y + (LATTICE_ROWS >> 1) * LATTICE_SPACING + 0.02
          ],
          dashed: true
        },
        {shape: 'crosshair', at: inputs.pointer, size: 8}
      ];
      const zoneIds = peek(outputs.pointFeatureIds);
      const featureCounts = peek(outputs.featureCounts);
      let readout = 'zone …';
      if (zoneIds && featureCounts) {
        const zone = zoneIds[CENTER_ROW];
        let joined = 0;
        for (let row = 0; row < LATTICE_COUNT; row++) {
          if (zoneIds[row] < featureCounts.length) joined++;
        }
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
            readout = `${joined} of ${LATTICE_COUNT} probes in a zone · ${hit.distance.toFixed(2)} from ${ZONE_NAMES[ring]}`;
          } else {
            readout = `${joined} of ${LATTICE_COUNT} probes in a zone`;
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
          readout = `${joined} of ${LATTICE_COUNT} probes in a zone · centre in ${ZONE_NAMES[zone]}`;
        }
      }
      return {rerun: true, glyphs, readout};
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.points, radius: 3},
    {layer: 'dots', positions: outputs.points, values: outputs.pointFeatureIds, radius: 3.5},
    // The probe lattice, joined by the same operation, drawn larger.
    {
      layer: 'dots',
      positions: outputs.points,
      values: outputs.pointFeatureIds,
      count: LATTICE_COUNT,
      radius: 6.5
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.zones.outline,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    }
  ]
});
