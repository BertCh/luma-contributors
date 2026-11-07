// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addPointsInPolygonsChoroplethRecipe: points joined to zones, one statistic per zone and class
// breaks. Zones are filled by their class (light to dark); the points are the muted input.

import {
  getGPUClassBreaksParameterValues,
  getGPUColorScaleParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {addPointsInPolygonsChoroplethRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {createQuadZones, getPolygonPort, getQuadFill, uploadQuadZones} from './lib/quads';
import {parameters} from './points-in-polygons-choropleth-recipe.parameters';

const MAXIMUM_CLASSES = 6;
const POINT_COUNT = 2500;

export default defineOperationRun({
  parameters,
  caption: 'Per-zone statistic of the points inside, classed into a choropleth.',
  createInputs: ({random, world}) => ({
    world,
    zones: createQuadZones(random, world),
    points: createClusteredPoints(random, world, POINT_COUNT, 3, 0.25)
  }),
  build(builder, {zones, points, world}, values) {
    const mesh = uploadQuadZones(builder, zones);
    const pointValues = builder.allocate('point-values', 'float32', POINT_COUNT);
    const counts = builder.allocate('counts', 'uint32', zones.count);
    const featureValues = builder.allocate('feature-values', 'float32', zones.count);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const breaks = builder.allocate('breaks', 'float32', MAXIMUM_CLASSES + 1);
    const classCount = builder.allocate('class-count', 'uint32', 1);
    const colors = builder.allocate('colors', 'uint32', zones.count);
    const classIndices = builder.allocate('class-indices', 'uint32', zones.count);
    const classBreaks = builder.allocate(
      'class-breaks-parameters',
      'float32',
      getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 3}, MAXIMUM_CLASSES)
        .length
    );
    const scale = builder.upload(
      'color-scale-parameters',
      'float32',
      getGPUColorScaleParameterValues({
        scale: 'quantile',
        domainCount: MAXIMUM_CLASSES - 1,
        paletteCount: MAXIMUM_CLASSES
      })
    );
    addPointsInPolygonsChoroplethRecipe(builder.graph, {
      statistic: (['mean', 'sum', 'maximum'] as const)[values.statistic],
      points: {
        kind: 'points',
        positions: builder.upload('points', 'float32x2', points.positions).view
      },
      values: pointValues.view,
      polygons: {...getPolygonPort(mesh), candidateCapacity: 1 << 16},
      outputs: {
        counts: counts.view,
        featureValues: featureValues.view,
        color: {
          breaks: breaks.view,
          classCount: classCount.view,
          colors: colors.view,
          classIndices: classIndices.view
        },
        status: {overflow: overflow.view}
      },
      color: {
        classBreaksParameters: classBreaks.view,
        maximumClassCount: MAXIMUM_CLASSES,
        methods: ['equal-interval'],
        colorScaleParameters: scale.view,
        palette: builder.upload(
          'palette',
          'uint32',
          Uint32Array.from({length: MAXIMUM_CLASSES}, (_, index) => 0xff000000 + index * 0x202020)
        ).view,
        maximumPaletteCount: MAXIMUM_CLASSES
      }
    });
    return {
      outputs: {
        classIndices,
        mesh,
        points: builder.upload('point-positions', 'float32x2', points.positions)
      },
      update: current => {
        pointValues.write(
          Float32Array.from(
            points.values,
            (value, index) =>
              1 + 9 * value + current.trend * 12 * (points.positions[index * 2] / world.width)
          )
        );
        classBreaks.write(
          getGPUClassBreaksParameterValues(
            {method: 'equal-interval', classCount: current.classCount},
            MAXIMUM_CLASSES
          )
        );
      }
    };
  },
  rebuildOn: ['statistic'],
  draw: ({inputs, outputs, values}) => [
    getQuadFill(outputs.mesh, inputs.zones, outputs.classIndices, {
      ramp: 'sequential',
      range: [0, values.classCount - 1]
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    },
    {layer: 'dots', context: true, positions: outputs.points, radius: 1.5}
  ]
});
