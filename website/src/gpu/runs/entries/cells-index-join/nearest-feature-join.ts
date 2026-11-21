// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNearestFeatureJoin finds, for every query point, its k nearest stations within a radius. Each
// line runs from a query to the nearest point of a neighbor station (the foot point the operation
// wrote); queries with no station in range stay unconnected.

import {GPUNearestFeatureJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {createPointIndex} from '../../pick';
import {addPointPairSegments} from './support/kernels';
import {getRainStations} from '../../../data/point-observations';
import {parameters} from './nearest-feature-join.parameters';

const QUERY_COUNT = 120;
const STATION_COUNT = 12;

export default defineOperationRun({
  parameters,
  caption: 'Each query linked to its k nearest stations inside the search radius.',
  createInputs: ({random, world}) => {
    const {positions} = getRainStations(random, world, QUERY_COUNT + STATION_COUNT, true);
    return {
      queries: positions.slice(0, QUERY_COUNT * 2),
      stations: positions.slice(QUERY_COUNT * 2),
      stationIndex: createPointIndex(positions.slice(QUERY_COUNT * 2)),
      world,
      // The last query row is the pointer.
      pointer: [world.width / 2, 0.5] as [number, number]
    };
  },
  build(builder, inputs, values) {
    const k = values.neighborCount;
    const rowCount = QUERY_COUNT + 1;
    const queries = builder.allocate('queries', 'float32x2', rowCount);
    const stations = builder.upload('stations', 'float32x2', inputs.stations);
    const radius = builder.allocate('radius', 'float32', 1);
    const slots = rowCount * k;
    const neighborIds = builder.allocate('neighbor-ids', 'uint32', slots);
    const neighborCounts = builder.allocate('neighbor-counts', 'uint32', rowCount);
    const neighborDistances = builder.allocate('neighbor-distances', 'float32', slots);
    const footPoints = builder.allocate('foot-points', 'float32x2', slots);
    const queryPoints = builder.allocate('query-points', 'float32x2', slots);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: queries.view},
        features: {kind: 'points', positions: stations.view},
        radius: radius.view,
        neighborCount: k,
        output: {
          neighborIds: neighborIds.view,
          neighborCounts: neighborCounts.view,
          neighborDistances: neighborDistances.view,
          neighborFootPoints: footPoints.view,
          neighborQueryPoints: queryPoints.view,
          status: {overflow: overflow.view}
        }
      })
    );
    return {
      outputs: {
        queries,
        stations,
        neighborCounts,
        neighborDistances,
        links: addPointPairSegments(builder, 'links', queryPoints, footPoints)
      },
      update: current => {
        radius.write(Float32Array.of(current.radius));
        queries.write(Float32Array.from([...inputs.queries, ...inputs.pointer]));
      }
    };
  },
  rebuildOn: ['neighborCount'],
  // The pointer is one more query row and its distance to the nearest station is the search radius:
  // every query links to the stations within that reach, so the links grow as the pointer moves
  // away from the stations. The ring is that radius.
  interaction: {
    kind: 'reach',
    hint: 'move away from a station',
    rest: ({inputs, toUnit}) => {
      // The grid point whose nearest station is closest to the default radius away.
      const target = parameters[1].default;
      let best: [number, number] = [inputs.world.width / 2, 0.5];
      let bestError = Infinity;
      for (let column = 0; column < 80; column++) {
        for (let row = 0; row < 40; row++) {
          const x = ((column + 0.5) / 80) * inputs.world.width;
          const y = (row + 0.5) / 40;
          const hit = inputs.stationIndex.nearest(x, y);
          const error = Math.abs(inputs.stationIndex.distanceTo(hit, x, y) - target);
          if (hit >= 0 && error < bestError) {
            bestError = error;
            best = [x, y];
          }
        }
      }
      return toUnit(best);
    },
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const station = inputs.stationIndex.nearest(pointer.x, pointer.y);
      const radius =
        station >= 0
          ? Math.min(
              Math.max(
                // Rounded up to the slider step, so the nearest station is always inside the ring.
                Number(
                  (
                    Math.ceil(
                      inputs.stationIndex.distanceTo(station, pointer.x, pointer.y) * 100 - 1e-6
                    ) / 100
                  ).toFixed(2)
                ),
                0.05
              ),
              0.6
            )
          : values.radius;
      const glyphs: RunGlyph[] = [
        {shape: 'crosshair', at: inputs.pointer},
        {shape: 'ring', at: inputs.pointer, radius, dashed: true}
      ];
      const links = peek(outputs.links);
      const distances = peek(outputs.neighborDistances);
      const counts = peek(outputs.neighborCounts);
      const first = QUERY_COUNT * values.neighborCount;
      let linked = '…';
      // After a rebuild with another k the read-back still has the old size until the next run.
      if (
        links &&
        distances &&
        counts &&
        distances.length === (QUERY_COUNT + 1) * values.neighborCount
      ) {
        let queriesLinked = 0;
        for (let row = 0; row < QUERY_COUNT; row++) if (counts[row] > 0) queriesLinked++;
        linked = `${queriesLinked} of ${QUERY_COUNT} queries linked`;
        const found = counts[QUERY_COUNT];
        for (let slot = 0; slot < Math.min(found, values.neighborCount); slot++) {
          const row = (first + slot) * 4;
          const to = [links[row + 2], links[row + 3]] as const;
          glyphs.push({shape: 'line', from: inputs.pointer, to}, {shape: 'dot', at: to});
        }
      }
      return {
        rerun: true,
        values: {radius},
        glyphs,
        readout: `radius ${radius.toFixed(2)} · k ${values.neighborCount} · ${linked}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.queries, radius: 3},
    {layer: 'lines', segments: outputs.links, radius: 1.2, color: [1, 0.7, 0.25, 0.85]},
    {layer: 'dots', positions: outputs.stations, radius: 7, color: [0.45, 0.8, 1, 1]}
  ]
});
