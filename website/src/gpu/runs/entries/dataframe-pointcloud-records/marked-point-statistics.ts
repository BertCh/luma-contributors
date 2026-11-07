// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMarkedPointStatistics on two point types (colors) with a size mark. The GPU writes four
// columns per radius; the cyan line is the cross K function of the chosen type pair, the orange
// line its pair correlation (1 means no interaction), and the white line the mark correlation.

import {GPUMarkedPointStatistics} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {drawNormal} from './_lib/helpers';
import {parameters} from './marked-point-statistics.parameters';

const POINT_COUNT = 500;
const PLOT = [0.58, 0.12, 0.96, 0.88] as const;
const TYPE_PAIRS = [
  [0, 1],
  [0, 0],
  [1, 1]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Cross K, pair correlation and mark correlation of two point types.',
  createInputs: ({random, world}) => {
    const windowWidth = world.width * 0.5;
    const parents = Float32Array.from({length: 24}, () => random());
    return {
      windowWidth,
      membership: Float32Array.from({length: POINT_COUNT}, () => random()),
      uniform: Float32Array.from({length: POINT_COUNT * 2}, () => random()),
      normals: Float32Array.from({length: POINT_COUNT * 2}, () => drawNormal(random)),
      parents,
      marks: Float32Array.from({length: POINT_COUNT}, () => 0.5 + random())
    };
  },
  build(builder, inputs, values) {
    const radiusCount = values.radiusCount;
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const types = builder.allocate('types', 'uint32', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 6);
    const statistics = builder.allocate('statistics', 'float32', radiusCount * 4);
    builder.graph.add(
      new GPUMarkedPointStatistics({
        positions: positions.view,
        types: types.view,
        typePair: TYPE_PAIRS[values.typePair],
        marks: builder.upload('marks', 'float32', inputs.marks).view,
        parameters: parameterBuffer.view,
        radiusCount,
        output: {
          statistics: statistics.view,
          pairCounts: builder.allocate('pair-counts', 'uint32', radiusCount).view,
          status: builder.allocate('status', 'uint32', radiusCount).view
        }
      })
    );
    return {
      outputs: {
        positions,
        types,
        crossK: addFloatCopy(builder, 'cross-k', statistics, radiusCount, {stride: 4}),
        pairCorrelation: addFloatCopy(builder, 'pair-correlation', statistics, radiusCount, {
          stride: 4,
          offset: 2
        }),
        markCorrelation: addFloatCopy(builder, 'mark-correlation', statistics, radiusCount, {
          stride: 4,
          offset: 3
        })
      },
      update: current => {
        const coordinates = new Float32Array(POINT_COUNT * 2);
        const typeValues = new Uint32Array(POINT_COUNT);
        for (let index = 0; index < POINT_COUNT; index++) {
          // First half are type 0 (random); type 1 points follow a type 0 point with `attraction`.
          const isSecond = index >= POINT_COUNT / 2;
          typeValues[index] = isSecond ? 1 : 0;
          let x = inputs.uniform[index * 2];
          let y = inputs.uniform[index * 2 + 1];
          if (isSecond && inputs.membership[index] < current.attraction) {
            const anchor = index - POINT_COUNT / 2;
            x = inputs.uniform[anchor * 2] + inputs.normals[index * 2] * 0.02;
            y = inputs.uniform[anchor * 2 + 1] + inputs.normals[index * 2 + 1] * 0.02;
          }
          coordinates[index * 2] = Math.min(1, Math.max(0, x)) * inputs.windowWidth;
          coordinates[index * 2 + 1] = Math.min(1, Math.max(0, y));
        }
        positions.write(coordinates);
        types.write(typeValues);
        parameterBuffer.write(Float32Array.of(0, 0, inputs.windowWidth, 1, 0.2, 0.000001));
      }
    };
  },
  rebuildOn: ['typePair', 'radiusCount'],
  draw: ({outputs}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.types, radius: 2.8},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.crossK,
      range: [0, 0.14],
      rectangle: PLOT,
      color: [0.3, 0.75, 1, 0.3]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.crossK,
      range: [0, 0.14],
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.pairCorrelation,
      range: [0, 4],
      rectangle: PLOT,
      radius: 2.2,
      color: [1, 0.6, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.markCorrelation,
      range: [0, 2],
      rectangle: PLOT,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    }
  ]
});
