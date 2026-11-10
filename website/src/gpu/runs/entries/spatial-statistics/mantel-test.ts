// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMantelTest: correlation between spatial closeness and time difference over event pairs. The
// bars are a histogram of the statistic of each shuffle (the reference distribution) with the
// observed statistic marked on it (solid line) and the shuffled mean (dashed). Events are colored by
// time; hovering one links it to its spatial neighbors, solid when also within a week in time.

import {
  getGPUSpaceTimeParameterValues,
  GPUMantelTest,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {drawReferenceHistogram} from './shared/permutation-chart';
import {
  createBandGraph,
  createSpaceTimeEvents,
  createTimedNeighborGlyphs,
  getEventTimes,
  pickProbeRow
} from './shared/point-field';
import {parameters} from './mantel-test.parameters';

const MAXIMUM_PERMUTATIONS = 499;
const HISTOGRAM_BINS = 24;
const CHART = [0.08, 0.04, 0.92, 0.22] as const;
const TIME_WINDOW = 7;

export default defineOperationRun({
  parameters,
  caption: 'Space and time differences among Kīlauea earthquakes, with a shuffled reference.',
  createInputs: ({random, world}) => createSpaceTimeEvents(random, world),
  build(builder, events, values) {
    const graph = createBandGraph(events.positions, values.radius);
    const times = builder.allocate('times', 'float32', events.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    const statistics = builder.allocate('statistics', 'float32', MAXIMUM_PERMUTATIONS + 1);
    const summary = builder.allocate('summary', 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    builder.graph.add(
      new GPUMantelTest({
        pairs: uploadSpatialWeights(builder, graph.weights),
        times: times.view,
        parameters: parameterBuffer.view,
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        output: {statistics: statistics.view, summary: summary.view}
      })
    );
    return {
      outputs: {
        times,
        statistics,
        summary,
        graph,
        bins: builder.allocate('bins', 'float32', HISTOGRAM_BINS),
        positions: builder.upload('positions', 'float32x2', events.drawnPositions)
      },
      update: current => {
        times.write(getEventTimes(events, current.clustering));
        parameterBuffer.write(
          getGPUSpaceTimeParameterValues({
            seed: current.seed,
            permutationCount: current.permutationCount
          })
        );
      }
    };
  },
  rebuildOn: ['radius'],
  interaction: {
    kind: 'probe',
    hint: 'hover an event',
    rest: [0.5, 0.4],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const summary = peek(outputs.summary);
      const observed = summary?.[GPU_SPACE_TIME_SUMMARY.observed] ?? NaN;
      const {glyphs: chartGlyphs} = drawReferenceHistogram({
        histogram: outputs.bins,
        samples: peek(outputs.statistics),
        start: 1,
        sampleCount: values.permutationCount,
        observed,
        rectangle: CHART,
        world
      });
      const pseudoP = summary?.[GPU_SPACE_TIME_SUMMARY.pseudoPGreater];
      const summaryText = `Mantel r ${Number.isFinite(observed) ? observed.toFixed(2) : '…'} · p ${
        pseudoP === undefined ? '…' : pseudoP.toFixed(3)
      }`;
      const row = pickProbeRow(inputs, pointer);
      if (row < 0) return {glyphs: chartGlyphs, readout: summaryText};
      const probe = createTimedNeighborGlyphs(
        inputs.drawnPositions,
        outputs.graph.weights,
        row,
        getEventTimes(inputs, values.clustering),
        TIME_WINDOW
      );
      return {
        glyphs: [...chartGlyphs, ...probe.glyphs],
        readout: `${probe.neighborCount} neighbors within ${values.radius} km, mean Δ ${probe.meanDifference.toFixed(1)} days · ${summaryText}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.times,
      range: [0, 120],
      radius: 4
    },
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
