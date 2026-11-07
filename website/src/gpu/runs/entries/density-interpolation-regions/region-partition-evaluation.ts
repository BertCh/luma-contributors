// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURegionPartitionEvaluation scores a partition (here the Ward regions on the map). Bars show
// the within-region sum of squares of each region (indexed by label); the summary also holds the
// between and total sums of squares, the pseudo-F and the share of adjacencies that cross a border.

import {
  getGPUWardRegionsParameterValues,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPURegionPartitionEvaluation,
  GPUWardRegions
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {makeRandom} from '../../../demos/kit';
import {uploadSpatialWeights} from '../../inputs';
import {parameters} from './region-partition-evaluation.parameters';
import {uploadTractMesh} from './support/geometry';
import {createRegionScene, drawLabels} from './support/regions';

export default defineOperationRun({
  parameters,
  caption: 'Within-region variance of each region of a partition.',
  createInputs: ({world}) => createRegionScene(makeRandom(7), world, 6, 2),
  build(builder, scene, current) {
    const {tracts, attributes, columnCount} = scene;
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const values = builder.upload('attributes', 'float32', attributes);
    const labels = builder.allocate('labels', 'uint32', tracts.count);
    builder.graph.add(
      new GPUWardRegions({
        weights,
        values: values.view,
        columnCount,
        standardize: current.standardize === 1,
        targetRegionCount: current.regionCount,
        maximumIterations: tracts.count - current.regionCount,
        mergeSchedule: 'mutual-nearest',
        parameters: builder.upload(
          'ward-parameters',
          'uint32',
          getGPUWardRegionsParameterValues({seed: 1})
        ).view,
        output: {
          labels: labels.view,
          objective: builder.allocate('ward-objective', 'float32', 1).view,
          status: builder.allocate('ward-status', 'uint32', 5).view
        }
      })
    );
    const within = builder.allocate('region-within', 'float32', tracts.count);
    builder.graph.add(
      new GPURegionPartitionEvaluation({
        values: values.view,
        columnCount,
        labels: labels.view,
        weights,
        output: {
          summary: builder.allocate(
            'summary',
            'float32',
            GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
          ).view,
          regionSizes: builder.allocate('region-sizes', 'uint32', tracts.count).view,
          regionWithinSumOfSquares: within.view
        }
      })
    );
    return {outputs: {labels, within, ...uploadTractMesh(builder, tracts)}};
  },
  draw: ({inputs, outputs}) => [
    ...drawLabels(inputs, outputs, outputs.labels),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.within,
      range: 'auto',
      rectangle: [0.04, 0.04, 0.96, 0.2],
      ramp: 'sequential'
    }
  ]
});
