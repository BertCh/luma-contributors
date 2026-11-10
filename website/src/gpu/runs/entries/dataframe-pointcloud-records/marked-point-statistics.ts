// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMarkedPointStatistics on two point types (colors) with a size mark. The GPU writes four
// columns per radius; the cyan line is the cross K function of the chosen type pair, the orange
// line its pair correlation (1 means no interaction), and the white line the mark correlation.

import {GPUMarkedPointStatistics} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {addFloatCopy} from './_lib/convert';
import {sampleFlashes} from '../../../data/lightning';
import {
  DISTANCE_PLOT,
  RING_HIGHLIGHT_LIMIT,
  createRuleSegments,
  getAxisFraction,
  getAxisPointer,
  getBinBox,
  getBinIndex,
  getHighlightCount
} from './_lib/distance-probe';
import {parameters} from './marked-point-statistics.parameters';

const POINT_COUNT = 500;
const PLOT = DISTANCE_PLOT;
const MAXIMUM_DISTANCE = 0.2;
const CORRELATION_RANGE = [0, 3] as const;
const ANCHOR_COUNT = 6;
const REST_FRACTION = 0.3;
const TYPE_PAIRS = [
  [0, 1],
  [0, 0],
  [1, 1]
] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Cross K (cyan), pair correlation (orange) and mark correlation (white) of two flash size classes; the faint rule is 1, no interaction.',
  createInputs: ({world}) => {
    const windowWidth = world.width * 0.5;
    const flashes = sampleFlashes({width: windowWidth, height: 1}, POINT_COUNT);
    return {windowWidth, flashes, world};
  },
  build(builder, inputs, values) {
    const radiusCount = values.radiusCount;
    const positions = builder.upload('positions', 'float32x2', inputs.flashes.positions);
    const types = builder.allocate('types', 'uint32', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 6);
    const statistics = builder.allocate('statistics', 'float32', radiusCount * 4);
    builder.graph.add(
      new GPUMarkedPointStatistics({
        positions: positions.view,
        types: types.view,
        typePair: TYPE_PAIRS[values.typePair],
        marks: builder.upload('marks', 'float32', inputs.flashes.energy).view,
        parameters: parameterBuffer.view,
        radiusCount,
        output: {
          statistics: statistics.view,
          pairCounts: builder.allocate('pair-counts', 'uint32', radiusCount).view,
          status: builder.allocate('status', 'uint32', radiusCount).view
        }
      })
    );
    const highlight = builder.allocate('highlight', 'float32x2', RING_HIGHLIGHT_LIMIT);
    const unity = builder.upload(
      'unity',
      'float32x4',
      createRuleSegments(PLOT, inputs.world, CORRELATION_RANGE, [1])
    );
    return {
      outputs: {
        highlight,
        unity,
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
        types.write(
          Uint32Array.from(inputs.flashes.area, area => Number(area >= current.areaThreshold))
        );
        parameterBuffer.write(Float32Array.of(0, 0, inputs.windowWidth, 1, 0.2, 0.000001));
      }
    };
  },
  rebuildOn: ['typePair', 'radiusCount'],
  // The pointer is a radius on the chart's distance axis: rings of that radius around a few points
  // of the first type, and the points of the second type inside them are brightened.
  interaction: {
    kind: 'reach',
    hint: 'hover a radius',
    rest: [getAxisPointer(REST_FRACTION), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const radiusCount = values.radiusCount;
      const bin = getBinIndex(getAxisFraction(pointer.unitX), radiusCount);
      const radius = (MAXIMUM_DISTANCE * (bin + 1)) / radiusCount;
      const [firstType, secondType] = TYPE_PAIRS[values.typePair];
      const {positions, area} = inputs.flashes;
      const typeOf = (row: number) => Number(area[row] >= values.areaThreshold);
      const firstRows: number[] = [];
      for (let row = 0; row < POINT_COUNT; row++)
        if (typeOf(row) === firstType) firstRows.push(row);
      const anchors = Array.from(
        {length: Math.min(ANCHOR_COUNT, firstRows.length)},
        (_, index) => firstRows[Math.floor(((index + 0.5) * firstRows.length) / ANCHOR_COUNT)]
      );
      const brightened = new Float32Array(RING_HIGHLIGHT_LIMIT * 2).fill(-10);
      let found = 0;
      for (let row = 0; row < POINT_COUNT && found < RING_HIGHLIGHT_LIMIT; row++) {
        if (typeOf(row) !== secondType || anchors.includes(row)) continue;
        const isInside = anchors.some(
          anchor =>
            Math.hypot(
              positions[row * 2] - positions[anchor * 2],
              positions[row * 2 + 1] - positions[anchor * 2 + 1]
            ) <= radius
        );
        if (!isInside) continue;
        brightened.set([positions[row * 2], positions[row * 2 + 1]], found * 2);
        found++;
      }
      outputs.highlight.write(brightened);
      const text = (buffer: typeof outputs.crossK, digits: number) => {
        const value = peek(buffer)?.[bin];
        return value === undefined ? '…' : value.toFixed(digits);
      };
      return {
        spotlight: null,
        focus: found,
        glyphs: [
          getBinBox(PLOT, inputs.world, bin, radiusCount),
          ...anchors.map(anchor => ({
            shape: 'ring' as const,
            at: [positions[anchor * 2], positions[anchor * 2 + 1]] as const,
            radius
          }))
        ],
        readout: `r ${radius.toFixed(2)} · K ${text(outputs.crossK, 3)} · g ${text(outputs.pairCorrelation, 1)} · mark ${text(outputs.markCorrelation, 1)}`
      };
    }
  },
  draw: ({outputs, interaction}) => [
    {layer: 'dots', positions: outputs.positions, values: outputs.types, radius: 2.8},
    {
      layer: 'dots',
      positions: outputs.highlight,
      count: getHighlightCount(interaction.focus),
      radius: 4.2,
      color: [1, 1, 1, 0.95]
    },
    {layer: 'lines', context: true, segments: outputs.unity, count: 1, color: [1, 1, 1, 0.5]},
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
      range: CORRELATION_RANGE,
      rectangle: PLOT,
      radius: 2.2,
      color: [1, 0.6, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.markCorrelation,
      range: CORRELATION_RANGE,
      rectangle: PLOT,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    }
  ]
});
