// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocalOutlierFactor: compares each point's local density with the density around its
// neighbors. Points that are much sparser than their neighbors (factor above the threshold)
// are flagged. Every dot is colored by its factor and flagged outliers get an orange disc; hovering a
// point shows its k nearest neighbors and the k-distance that sets its local density.

import {
  getGPULocalOutlierFactorParameterValues,
  GPULocalOutlierFactor,
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints, createKnnWeights, uploadSpatialWeights} from '../../inputs';
import {chooseSticky, createPointIndex} from '../../pick';
import {createNeighborGlyphs, getNeighborRows} from './shared/point-field';
import {parameters} from './local-outlier-factor.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Points flagged as outliers (orange) by the local outlier factor.',
  createInputs: ({random, world}) => {
    const points = createClusteredPoints(random, world, 600, 4, 0.06);
    return {...points, index: createPointIndex(points.positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const weights = createKnnWeights(points.positions, values.neighborCount);
    const neighbors = uploadSpatialWeights(builder, weights);
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
        kDistance,
        weights,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current =>
        parameterBuffer.write(
          getGPULocalOutlierFactorParameterValues({threshold: current.threshold})
        )
    };
  },
  rebuildOn: ['neighborCount'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.6, 0.45],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const at = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const factor = peek(outputs.lof)?.[row];
      const kDistance = peek(outputs.kDistance)?.[row];
      return {
        glyphs: [
          ...createNeighborGlyphs(inputs.positions, outputs.weights, row),
          ...(kDistance === undefined
            ? []
            : [{shape: 'ring' as const, at, radius: kDistance, dashed: true}])
        ],
        readout: `LOF ${factor === undefined ? '…' : factor.toFixed(2)} ${
          factor !== undefined && factor > values.threshold ? '>' : '≤'
        } threshold ${values.threshold} · ${getNeighborRows(outputs.weights, row).length} neighbors`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.lof,
      radius: 3.2,
      range: [0.2, 2.2]
    },
    // Flagged outliers get a larger disc; the faded mask hides the rest.
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.outlier,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      alpha: 0.85,
      radius: 6.5
    }
  ]
});
