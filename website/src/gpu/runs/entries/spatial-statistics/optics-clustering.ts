// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUOPTICS: orders the points by density reachability and extracts clusters with the xi method.
// Colors are the cluster labels; dots too sparse for any cluster are muted. The size of each dot
// follows nothing: the reachability and ordering stay on the GPU for a reachability plot.

import {GPUOPTICS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {parameters} from './optics-clustering.parameters';

export default defineOperationRun({
  parameters,
  caption: 'OPTICS clusters extracted with the xi method.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 500, 4, 0.12),
  build(builder, points, values) {
    const ordering = builder.allocate('ordering', 'uint32', points.count);
    const coreDistances = builder.allocate('core-distances', 'float32', points.count);
    const reachability = builder.allocate('reachability', 'float32', points.count);
    const predecessors = builder.allocate('predecessors', 'uint32', points.count);
    const labels = builder.allocate('labels', 'uint32', points.count);
    const status = builder.allocate('status', 'uint32', 3);
    builder.graph.add(
      new GPUOPTICS({
        positions: builder.upload('positions', 'float32x2', points.positions).view,
        minimumSamples: values.minimumSamples,
        maximumDistance: values.maximumDistance,
        extractionMethod: 'xi',
        xi: values.xi,
        output: {
          ordering: ordering.view,
          coreDistances: coreDistances.view,
          reachability: reachability.view,
          predecessors: predecessors.view,
          labels: labels.view,
          status: status.view
        }
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
