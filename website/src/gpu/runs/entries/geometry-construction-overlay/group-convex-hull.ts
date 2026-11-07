// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGroupConvexHull builds the exact convex hull of every label group. Points are muted; each
// group's hull ring has its own color. A hull with more vertices than `maximumVerticesPerGroup` is
// dropped whole, and labels at or above `groupCount` (and the noise label) are excluded.

import {GPUGroupConvexHull} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {addPackedSegments} from './shared/lines';
import {parameters} from './group-convex-hull.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Convex hull per label group; each ring has its group color.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 320, 4, 0.1),
  build(builder, points, values) {
    const groupCount = values.groupCount;
    const maximumVertices = values.maximumVerticesPerGroup;
    const vertexCapacity = groupCount * maximumVertices;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const labels = builder.upload('labels', 'uint32', points.labels);
    const vertexIndices = builder.allocate('hull-vertex-indices', 'uint32', vertexCapacity);
    const vertexPositions = builder.allocate('hull-vertex-positions', 'float32x2', vertexCapacity);
    const offsets = builder.allocate('hull-offsets', 'uint32', groupCount + 1);
    const counts = builder.allocate('hull-counts', 'uint32', groupCount);
    const overflow = builder.allocate('hull-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUGroupConvexHull({
        positions: positions.view,
        labels: labels.view,
        groupCount,
        noiseLabel: 4,
        maximumVerticesPerGroup: maximumVertices,
        vertexCapacity,
        output: {
          vertexIndices: vertexIndices.view,
          vertexPositions: vertexPositions.view,
          offsets: offsets.view,
          counts: counts.view,
          status: {overflow: overflow.view}
        }
      })
    );
    const drawing = addPackedSegments(builder, 'hull', vertexPositions, offsets, {closed: true});
    return {outputs: {positions, labels, ...drawing}};
  },
  rebuildOn: ['maximumVerticesPerGroup', 'groupCount'],
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, context: true, radius: 3},
    {
      layer: 'segments',
      nodes: outputs.points,
      edges: outputs.edges,
      values: outputs.lineIds,
      radius: 2.6
    }
  ]
});
