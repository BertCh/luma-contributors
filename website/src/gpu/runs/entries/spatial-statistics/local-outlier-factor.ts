// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalOutlierFactor: compares each point's local density with the density around its
// neighbors. Points that are much sparser than their neighbors (factor above the threshold)
// are flagged. Flagged outliers are orange, the rest blue; the muted dots underneath show the factor itself.

import {
  getGPULocalOutlierFactorParameterValues,
  GPULocalOutlierFactor,
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints, createKnnWeights, uploadSpatialWeights} from '../../inputs';
import {parameters} from './local-outlier-factor.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Points flagged as outliers (orange) by the local outlier factor.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 600, 4, 0.06),
  build(builder, points, values) {
    const neighbors = uploadSpatialWeights(
      builder,
      createKnnWeights(points.positions, values.neighborCount)
    );
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH
    );
    const kDistance = builder.allocate('k-distance', 'float32', points.count);
    const density = builder.allocate('density', 'float32', points.count);
    const lof = builder.allocate('lof', 'float32', points.count);
    const outlier = builder.allocate('outlier', 'uint32', points.count);
    const outlierCount = builder.allocate('outlier-count', 'uint32', 1);
    builder.graph.add(
      new GPULocalOutlierFactor({
        neighbors,
        parameters: parameterBuffer.view,
        output: {
          kDistance: kDistance.view,
          localReachabilityDensity: density.view,
          lof: lof.view,
          outlier: outlier.view,
          outlierCount: outlierCount.view
        }
      })
    );
    return {
      outputs: {
        lof,
        outlier,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current =>
        parameterBuffer.write(
          getGPULocalOutlierFactorParameterValues({threshold: current.threshold})
        )
    };
  },
  rebuildOn: ['neighborCount'],
  draw: ({outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.lof,
      radius: 2.5,
      range: [1, 2.5]
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.outlier,
      ramp: 'categorical',
      radius: 4.5
    }
  ]
});
