// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUJacquezTest on the Kīlauea summit earthquakes: J counts (event, neighbor) pairs where the
// neighbor is among the k nearest events in space and also among the k nearest in time. Dots are
// events colored by day; the bars are the permuted J values with the observed J marked (solid) and
// their mean (dashed). Hovering an event links its k spatial neighbors: solid when also k-nearest
// in time.

import {
  getGPUSpaceTimeParameterValues,
  GPUJacquezTest,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {createKnnWeights, uploadSpatialWeights, type CsrWeights} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky} from '../../pick';
import {createKilaueaEvents, formatValue, getRow, shuffleTimes} from './_lib/events';
import {drawReferenceHistogram} from './_lib/permutation-chart';
import {drawPanel, type Panel, uploadPanel} from './_lib/panel';
import {parameters} from './jacquez-test.parameters';

const MAXIMUM_PERMUTATIONS = 499;
const HISTOGRAM_BINS = 28;
const CHART = [0.06, 0.04, 0.94, 0.22] as const;

type Inputs = ReturnType<typeof createKilaueaEvents> & {worldWidth: number};
type Outputs = {
  panel: Panel;
  positions: RunBuffer;
  times: RunBuffer;
  statistics: RunBuffer;
  summary: RunBuffer;
  bins: RunBuffer<'float32'>;
  weights: CsrWeights;
};

/** The k events nearest in time to `row` (ties to the lower row), excluding the event itself. */
function getTemporalNeighbors(times: Float32Array, row: number, k: number): Set<number> {
  const order = Array.from({length: times.length}, (_, index) => index).filter(
    index => index !== row
  );
  order.sort(
    (first, second) =>
      Math.abs(times[first] - times[row]) - Math.abs(times[second] - times[row]) || first - second
  );
  return new Set(order.slice(0, k));
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Kīlauea summit earthquakes, May–August 2018, colored by day; bars are J under permuted times, the solid tick the observed J.',
  createInputs: ({world}) => ({
    ...createKilaueaEvents([0, 0.28, world.width, world.height]),
    worldWidth: world.width
  }),
  build(builder, events, values) {
    const weights = createKnnWeights(events.positions, values.k);
    const times = builder.allocate('times', 'float32', events.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    const statistics = builder.allocate('statistics', 'uint32', MAXIMUM_PERMUTATIONS + 1);
    const summary = builder.allocate('summary', 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    builder.graph.add(
      new GPUJacquezTest({
        neighbors: uploadSpatialWeights(builder, weights),
        times: times.view,
        parameters: parameterBuffer.view,
        k: values.k,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {statistics: statistics.view, summary: summary.view}
      })
    );
    return {
      outputs: {
        positions: builder.upload('positions', 'float32x2', events.positions),
        times,
        statistics,
        summary,
        weights,
        panel: uploadPanel(builder, CHART, {width: events.worldWidth, height: 1}),
        bins: builder.allocate('bins', 'float32', HISTOGRAM_BINS)
      },
      update: current => {
        times.write(shuffleTimes(events.days, current.observedShare));
        parameterBuffer.write(
          getGPUSpaceTimeParameterValues({seed: 3, permutationCount: current.permutationCount})
        );
      }
    };
  },
  rebuildOn: ['k'],
  buildCache: 5,
  interaction: {
    kind: 'probe',
    hint: 'hover an event',
    rest: [0.48, 0.3],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const summary = peek(outputs.summary);
      const observed = summary?.[GPU_SPACE_TIME_SUMMARY.observed] ?? Number.NaN;
      const chart = drawReferenceHistogram({
        histogram: outputs.bins,
        samples: peek(outputs.statistics),
        start: 1,
        sampleCount: values.permutationCount,
        observed,
        rectangle: CHART,
        world
      });
      const total = `J ${formatValue(observed, 0)} vs ${formatValue(
        summary?.[GPU_SPACE_TIME_SUMMARY.permutationMean],
        0
      )} · p ${formatValue(summary?.[GPU_SPACE_TIME_SUMMARY.pseudoPGreater], 3)}`;
      if (pointer.y < 0.26) return {glyphs: chart.glyphs, readout: total};
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {glyphs: chart.glyphs, readout: total};
      const times = shuffleTimes(inputs.days, values.observedShare);
      const temporal = getTemporalNeighbors(times, row, values.k);
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [...chart.glyphs, {shape: 'crosshair', at: from, size: 12}];
      let bothCount = 0;
      for (const neighbor of getRow(outputs.weights, row)) {
        const isBoth = temporal.has(neighbor);
        if (isBoth) bothCount++;
        const to = [inputs.positions[neighbor * 2], inputs.positions[neighbor * 2 + 1]] as const;
        glyphs.push({shape: 'line', from, to, dashed: !isBoth}, {shape: 'dot', at: to});
      }
      return {
        focus: row,
        glyphs,
        readout: `${bothCount} of k ${values.k} also nearest in time · ${total}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.times,
      range: [0, 120],
      radius: 3.5
    },
    drawPanel(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bins,
      range: [0, 1],
      rectangle: CHART,
      color: [...theme.categories[0], 0.9]
    }
  ]
});
