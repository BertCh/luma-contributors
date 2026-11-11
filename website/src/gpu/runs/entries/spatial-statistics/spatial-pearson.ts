// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialPearson between a variable and a second variable that follows it plus noise. The
// pointer is a curtain: the first variable on its left, the second on its right (one color scale).
// The bar is the statistic the GPU wrote (one float32); the line through it is zero. The small
// scatter plots the two input variables against each other, point for point.

import {GPUSpatialPearson} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {uploadSpatialWeights} from '../../inputs';
import {drawSignedBars} from '../regression-weights/shared/regression';
import {createBandGraph, createPointField, mixNoise} from './shared/point-field';
import {parameters} from './spatial-pearson.parameters';

const CHART = [0.62, 0.04, 0.95, 0.24] as const;
const SCATTER = [0.08, 0.04, 0.4, 0.3] as const;

export default defineOperationRun({
  parameters,
  caption: 'Spatial Pearson correlation of two variables over the neighbor graph.',
  createInputs: ({random, world}) => createPointField(random, world, 700),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const first = builder.upload('first', 'float32', points.field);
    const second = builder.allocate('second', 'float32', points.count);
    const statistic = builder.allocate('statistic', 'float32', 1);
    builder.graph.add(
      new GPUSpatialPearson({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: first.view,
        secondValues: second.view,
        output: {statistic: statistic.view}
      })
    );
    return {
      outputs: {
        first,
        second,
        statistic,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => second.write(mixNoise(points.field, points.noise, current.noise))
    };
  },
  rebuildOn: ['radius'],
  interaction: {
    kind: 'swipe',
    hint: 'swipe the two variables',
    rest: [0.5, 0.4],
    resolve(pointer, {outputs, world, peek}) {
      const statistic = peek(outputs.statistic)?.[0];
      const zero = (CHART[1] + (CHART[3] - CHART[1]) / 2) * world.height;
      return {
        curtain: pointer.x,
        glyphs: [
          {shape: 'curtain', x: pointer.x},
          {shape: 'line', from: [CHART[0] * world.width, zero], to: [CHART[2] * world.width, zero]}
        ],
        readout: `x | y · spatial Pearson ${statistic === undefined ? '…' : statistic.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, interaction, world}) => {
    const scatterRectangle = getChartRectangle(world, SCATTER);
    const axes = createChartAxes({
      rectangle: getChartRectangle(world, CHART),
      y: {range: [-1, 1], ticks: [-1, 0, 1], title: 'spatial Pearson r'}
    });
    const scatterAxes = createChartAxes({
      rectangle: scatterRectangle,
      x: {range: [0, 1], ticks: [0, 1], title: 'x'},
      y: {range: [0, 1], ticks: [1], title: 'y', grid: false}
    });
    return [
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.first,
        radius: 5.5,
        range: [0, 1],
        clip: {side: 'left', x: interaction.curtain ?? world.width / 2}
      },
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.second,
        radius: 5.5,
        range: [0, 1],
        clip: {side: 'right', x: interaction.curtain ?? world.width / 2}
      },
      {
        layer: 'chart',
        kind: 'scatter',
        context: true,
        values: outputs.second,
        xValues: outputs.first,
        xRange: [0, 1],
        range: [0, 1],
        rectangle: SCATTER,
        radius: 3
      },
      ...drawSignedBars(outputs.statistic, CHART, 1),
      {
        layer: 'text',
        ...mergeText(axes, scatterAxes, {
          labels: [
            {
              text: 'inputs: y against x',
              at: [SCATTER[0] * world.width, SCATTER[3] * world.height],
              anchor: 'bottom-left',
              offset: [0, 14],
              role: 'label'
            }
          ]
        })
      }
    ];
  }
});
