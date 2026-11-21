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
import {isInsideRing} from '../../pick';
import {createQuadZones, getPolygonPort, getQuadFill, uploadQuadZones} from './lib/quads';
import {findZoneAt, getZoneRing} from './lib/zone-lookup';
import {parameters} from './points-in-polygons-choropleth-recipe.parameters';

const MAXIMUM_CLASSES = 6;
const POINT_COUNT = 2500;

export default defineOperationRun({
  parameters,
  caption: 'Per-zone statistic of the points inside, classed into a choropleth.',
  createInputs: ({random, world}) => ({
    world,
    zones: createQuadZones(random, world),
    points: createClusteredPoints(random, world, POINT_COUNT, 3, 0.25),
    focus: {zone: -1}
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
      output: {
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
    const highlight = builder.allocate('highlight', 'float32', POINT_COUNT);
    const legendPositions = builder.upload(
      'legend-positions',
      'float32x2',
      Float32Array.from({length: MAXIMUM_CLASSES * 2}, (_, index) =>
        index % 2 === 0 ? (0.05 + 0.04 * (index / 2)) * world.width : 0.04
      )
    );
    const legendValues = builder.upload(
      'legend-values',
      'float32',
      Float32Array.from({length: MAXIMUM_CLASSES}, (_, index) => index)
    );
    return {
      outputs: {
        classIndices,
        featureValues,
        highlight,
        legendPositions,
        legendValues,
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
  // The pointer picks a zone: the points joined to it light up, with its statistic and class.
  interaction: {
    kind: 'probe',
    hint: 'hover a zone',
    rest: ({toUnit, inputs}) =>
      toUnit([inputs.zones.centers[2 * 70], inputs.zones.centers[2 * 70 + 1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const zone = findZoneAt(inputs.zones, pointer.x, pointer.y);
      if (zone !== inputs.focus.zone) {
        inputs.focus.zone = zone;
        const ring = getZoneRing(inputs.zones, zone);
        // 1 for the points inside the zone; the dots layer fades the rest out.
        const padded = new Float32Array(POINT_COUNT);
        for (let point = 0; point < POINT_COUNT; point++) {
          padded[point] = isInsideRing(
            ring,
            inputs.points.positions[point * 2],
            inputs.points.positions[point * 2 + 1]
          )
            ? 1
            : 0;
        }
        outputs.highlight.write(padded);
      }
      const classes = peek(outputs.classIndices);
      const statistics = peek(outputs.featureValues);
      const name = ['mean', 'sum', 'maximum'][values.statistic];
      const hasPoints = classes && statistics && classes[zone] < values.classCount;
      return {
        glyphs: [{shape: 'polyline', points: getZoneRing(inputs.zones, zone), closed: true}],
        readout: hasPoints
          ? `zone ${zone} · ${name} ${statistics[zone].toFixed(1)} · class ${classes[zone] + 1} of ${values.classCount}`
          : `zone ${zone} · no points`
      };
    }
  },
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
    {layer: 'dots', context: true, positions: outputs.points, radius: 1.5},
    {
      layer: 'dots',
      positions: outputs.points,
      values: outputs.highlight,
      ramp: 'grey',
      range: [0, 1],
      fade: true,
      count: POINT_COUNT,
      radius: 3
    },
    // Legend row: one dot per class, from the lowest to the highest.
    {
      layer: 'dots',
      positions: outputs.legendPositions,
      values: outputs.legendValues,
      ramp: 'sequential',
      range: [0, values.classCount - 1],
      count: values.classCount,
      radius: 9
    }
  ]
});
