// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNearestFeatureJoin finds, for every query point, its k nearest stations within a radius. Each
// line runs from a query to the nearest point of a neighbor station (the foot point the operation
// wrote); queries with no station in range stay unconnected.

import {GPUNearestFeatureJoin} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addPointPairSegments} from './support/kernels';
import {createRandomPoints} from './support/join';
import {parameters} from './nearest-feature-join.parameters';

const QUERY_COUNT = 120;
const STATION_COUNT = 12;

export default defineOperationRun({
  parameters,
  caption: 'Each query linked to its k nearest stations inside the search radius.',
  createInputs: ({random, world}) => ({
    queries: createRandomPoints(random, world, QUERY_COUNT),
    stations: createRandomPoints(random, world, STATION_COUNT)
  }),
  build(builder, inputs, values) {
    const k = values.k;
    const queries = builder.upload('queries', 'float32x2', inputs.queries);
    const stations = builder.upload('stations', 'float32x2', inputs.stations);
    const radius = builder.allocate('radius', 'float32', 1);
    const slots = QUERY_COUNT * k;
    const neighborIds = builder.allocate('neighbor-ids', 'uint32', slots);
    const neighborCounts = builder.allocate('neighbor-counts', 'uint32', QUERY_COUNT);
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
        links: addPointPairSegments(builder, 'links', queryPoints, footPoints)
      },
      update: current => radius.write(Float32Array.of(current.radius))
    };
  },
  rebuildOn: ['k'],
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.queries, radius: 3},
    {layer: 'lines', segments: outputs.links, radius: 1.2, color: [1, 0.7, 0.25, 0.85]},
    {layer: 'dots', positions: outputs.stations, radius: 7, color: [0.45, 0.8, 1, 1]}
  ]
});
