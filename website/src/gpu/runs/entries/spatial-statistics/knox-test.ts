// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUKnoxTest: events that are close in space and also close in time, against the same count when
// the times are shuffled. Dots are events colored by time. The bars are a histogram of the shuffled
// counts of close pairs with the observed count marked on it (solid line) and the shuffled mean
// (dashed). Hovering an event links it to its spatial neighbors: solid when also close in time.

import {
  getGPUSpaceTimeParameterValues,
  GPUKnoxTest,
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
import {parameters} from './knox-test.parameters';

const MAXIMUM_PERMUTATIONS = 499;
const HISTOGRAM_BINS = 24;
const CHART = [0.08, 0.04, 0.92, 0.22] as const;

export default defineOperationRun({
  parameters,
  caption: 'Space–time proximity of Kīlauea earthquakes against permuted origin times.',
  createInputs: ({random, world}) => createSpaceTimeEvents(random, world),
  build(builder, events, values) {
    const graph = createBandGraph(events.positions, values.radius);
    const times = builder.allocate('times', 'float32', events.count);
    const parameterBuffer = builder.allocate(
      'parameters',
      'uint32',
      GPU_SPACE_TIME_PARAMETER_LENGTH
    );
    const statistics = builder.allocate('statistics', 'uint32', MAXIMUM_PERMUTATIONS + 1);
    const summary = builder.allocate('summary', 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
    builder.graph.add(
      new GPUKnoxTest({
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
        summary,
        statistics,
        graph,
        bins: builder.allocate('bins', 'float32', HISTOGRAM_BINS),
        positions: builder.upload('positions', 'float32x2', events.drawnPositions)
      },
      update: current => {
        times.write(getEventTimes(events, current.clustering));
        parameterBuffer.write(
          getGPUSpaceTimeParameterValues({
            seed: 1,
            permutationCount: current.permutationCount,
            timeThreshold: current.timeThreshold
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
      const statistics = peek(outputs.statistics);
      const observed = summary?.[GPU_SPACE_TIME_SUMMARY.observed] ?? NaN;
      const {glyphs: chartGlyphs} = drawReferenceHistogram({
        histogram: outputs.bins,
        samples: statistics,
        start: 1,
        sampleCount: values.permutationCount,
        observed,
        rectangle: CHART,
        world
      });
      const pseudoP = summary?.[GPU_SPACE_TIME_SUMMARY.pseudoPGreater];
      const summaryText = `Knox ${Number.isFinite(observed) ? observed : '…'} · p ${
        pseudoP === undefined ? '…' : pseudoP.toFixed(2)
      }`;
      const row = pickProbeRow(inputs, pointer);
      if (row < 0) return {glyphs: chartGlyphs, readout: summaryText};
      const times = getEventTimes(inputs, values.clustering);
      const probe = createTimedNeighborGlyphs(
        inputs.drawnPositions,
        outputs.graph.weights,
        row,
        times,
        values.timeThreshold
      );
      return {
        glyphs: [...chartGlyphs, ...probe.glyphs],
        readout: `${probe.closeCount} of ${probe.neighborCount} within ${values.timeThreshold} d and ${values.radius} km · ${summaryText}`
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
