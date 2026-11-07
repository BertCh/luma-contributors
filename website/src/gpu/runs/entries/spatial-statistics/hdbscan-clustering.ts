// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUHierarchicalDensityClustering (HDBSCAN): builds the density hierarchy and keeps the stable
// clusters, so clusters of different density are found without an epsilon. Noise points are
// drawn in the muted color; each cluster gets its own color.

import {GPUHierarchicalDensityClustering} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './hdbscan-clustering.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Stable density clusters; noise is muted.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 700, 5, 0.12),
  build(builder, points, values) {
    const labels = builder.allocate('labels', 'uint32', points.count);
    const coreDistances = builder.allocate('core-distances', 'float32', points.count);
    builder.graph.add(
      new GPUHierarchicalDensityClustering({
        positions: builder.upload('positions', 'float32x2', points.positions).view,
        minimumClusterSize: values.minimumClusterSize,
        minimumSamples: values.minimumSamples,
        output: {labels: labels.view, coreDistances: coreDistances.view}
      })
    );
    return {
      outputs: {labels, positions: builder.upload('points', 'float32x2', points.positions)}
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2},
    {layer: 'dots', positions: outputs.positions, values: outputs.labels, radius: 4.5}
  ]
});
