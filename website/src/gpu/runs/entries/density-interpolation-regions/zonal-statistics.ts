// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUZonalStatistics summarizes the point values that fall inside each polygon: count, mean,
// extremes, spread and sum per tract. The tracts are filled with the chosen output buffer.

import {GPUZonalStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, createUniformPoints} from '../../inputs';
import {parameters} from './zonal-statistics.parameters';
import {addSlice} from './support/gather';
import {getPackedPolygons, uploadPackedPolygons, uploadTractMesh} from './support/geometry';

export default defineOperationRun({
  parameters,
  caption: 'Per-polygon statistics of the points inside each tract.',
  createInputs: ({random, world}) => ({
    tracts: createTracts(random, world, 6, 0.2),
    points: createUniformPoints(random, world, 6000, 0.25)
  }),
  build(builder, {tracts, points}, current) {
    const count = current.pointCount;
    const featureCount = tracts.count;
    const statistics = {
      counts: builder.allocate('counts', 'uint32', featureCount),
      means: builder.allocate('means', 'float32', featureCount),
      minima: builder.allocate('minima', 'float32', featureCount),
      maxima: builder.allocate('maxima', 'float32', featureCount),
      standardDeviations: builder.allocate('standard-deviations', 'float32', featureCount),
      sums: builder.allocate('sums', 'float32', featureCount)
    };
    builder.graph.add(
      new GPUZonalStatistics({
        features: {
          kind: 'polygons',
          ...uploadPackedPolygons(builder, getPackedPolygons(tracts)),
          candidateCapacity: count * 8
        },
        positions: builder.upload('positions', 'float32x2', points.positions.slice(0, count * 2))
          .view,
        values: builder.upload('values', 'float32', points.values.slice(0, count)).view,
        sumOrder: current.sumOrder === 0 ? 'atomic' : 'sorted',
        output: {
          counts: statistics.counts.view,
          means: statistics.means.view,
          minima: statistics.minima.view,
          maxima: statistics.maxima.view,
          standardDeviations: statistics.standardDeviations.view,
          sums: statistics.sums.view,
          status: {overflow: builder.allocate('overflow', 'uint32', 1).view}
        }
      })
    );
    return {
      outputs: {
        ...statistics,
        countValues: addSlice(builder, statistics.counts, 0, featureCount, 'count-values'),
        ...uploadTractMesh(builder, tracts),
        dots: builder.upload('dots', 'float32x2', points.positions),
        dotCount: count
      }
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: [
        outputs.counts,
        outputs.means,
        outputs.minima,
        outputs.maxima,
        outputs.standardDeviations,
        outputs.sums
      ][values.statistic],
      triangleCount: inputs.tracts.triangleCount,
      ramp: 'sequential'
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.tracts.outlineCount,
      alpha: 0.5
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.dots,
      count: outputs.dotCount,
      radius: 1.2,
      alpha: 0.6
    }
  ]
});
