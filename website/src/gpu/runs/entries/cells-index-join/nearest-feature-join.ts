// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNearestFeatureJoin finds, for every query point, its k nearest stations within a radius. Each
// line runs from a query to the nearest point of a neighbor station (the foot point the operation
// wrote); queries with no station in range stay unconnected.

import {GPUNearestFeatureJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
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
      // The last query row is the pointer.
      pointer: [world.width / 2, 0.5] as [number, number]
    };
  },
  build(builder, inputs, values) {
    const k = values.k;
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
        k,
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
  rebuildOn: ['k'],
  // The pointer is one more query row: its links are drawn bold, and the radius ring shows what
  // the search could reach.
  interaction: {
    kind: 'place',
    hint: 'move a query',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const glyphs: RunGlyph[] = [
        {shape: 'crosshair', at: inputs.pointer},
        {shape: 'ring', at: inputs.pointer, radius: values.radius, dashed: true}
      ];
      const links = peek(outputs.links);
      const distances = peek(outputs.neighborDistances);
      const counts = peek(outputs.neighborCounts);
      const first = QUERY_COUNT * values.k;
      let nearest = '…';
      // After a rebuild with another k the read-back still has the old size until the next run.
      if (links && distances && counts && distances.length === (QUERY_COUNT + 1) * values.k) {
        const found = counts[QUERY_COUNT];
        for (let slot = 0; slot < Math.min(found, values.k); slot++) {
          const row = (first + slot) * 4;
          const to = [links[row + 2], links[row + 3]] as const;
          glyphs.push({shape: 'line', from: inputs.pointer, to}, {shape: 'dot', at: to});
        }
        nearest = found > 0 ? `nearest ${distances[first].toFixed(2)}` : 'no station in range';
      }
      return {
        rerun: true,
        glyphs,
        readout: `k ${values.k} · ${nearest} (radius ${values.radius.toFixed(2)})`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.queries, radius: 3},
    {layer: 'lines', segments: outputs.links, radius: 1.2, color: [1, 0.7, 0.25, 0.85]},
    {layer: 'dots', positions: outputs.stations, radius: 7, color: [0.45, 0.8, 1, 1]}
  ]
});
