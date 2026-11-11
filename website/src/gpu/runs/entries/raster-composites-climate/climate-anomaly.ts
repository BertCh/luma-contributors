// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterClimateAnomaly reads the climatology a GPURasterStreamingAccumulator holds (a mean and
// standard deviation per cell and calendar slot) and expresses a new slice as an anomaly against it,
// either `value - mean` or, with `standardized`, a z-score. The data are real: the sixteen Sentinel-2
// NDVI scenes of the Dixie Fire (2017 to 2024), all from one summer slot. The first
// `activeSliceCount` scenes are streamed in as the baseline (the eight before the fire by default;
// raise it and post-fire scenes leak into the baseline and weaken the anomaly) and the pointer picks
// the scene evaluated. The strip shows the mean anomaly of every scene, on the CPU from the same
// data, with a cursor on the scene shown.

import {
  GPURasterClimateAnomaly,
  GPURasterStreamingAccumulator,
  getGPURasterStreamingAccumulatorParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {parameters} from './climate-anomaly.parameters';
import {
  ACQUISITION_DATES,
  createAcquisitionStack,
  type AcquisitionStack
} from './support/acquisitions';
import {createPanelData, getPanelLayers, getThemeColor, uploadPanel} from './support/panel';

const PANEL = [0.03, 0.03, 0.97, 0.22] as const;
const STRIP = [0.06, 0.06, 0.94, 0.2] as const;
const STRIP_RANGE = [-0.5, 0.1] as const;

/** Mean over cells of each scene minus the mean of its first `baselineCount` scenes (CPU). */
function getMeanAnomalies(stack: AcquisitionStack, baselineCount: number): Float32Array {
  const {cellCount, sliceCount, observations} = stack;
  const totals = new Float64Array(sliceCount);
  const counts = new Float64Array(sliceCount);
  for (let cell = 0; cell < cellCount; cell++) {
    let sum = 0;
    let count = 0;
    for (let slice = 0; slice < baselineCount; slice++) {
      const value = observations[cell * sliceCount + slice];
      if (Number.isFinite(value)) {
        sum += value;
        count++;
      }
    }
    if (count === 0) continue;
    for (let slice = 0; slice < sliceCount; slice++) {
      const value = observations[cell * sliceCount + slice];
      if (Number.isFinite(value)) {
        totals[slice] += value - sum / count;
        counts[slice]++;
      }
    }
  }
  return Float32Array.from(totals, (total, slice) => (counts[slice] ? total / counts[slice] : 0));
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['standardized'],
  buildCache: 2,
  caption: 'Real Dixie NDVI scenes in; one scene against the pre-fire baseline out.',
  createInputs: ({aspect, world}) => ({
    ...createAcquisitionStack(aspect, 24),
    panel: createPanelData(world, [PANEL])
  }),
  build(builder, stack, values) {
    const {cellCount, sliceCount} = stack;
    const state = {
      counts: builder.allocate('state-counts', 'uint32', cellCount),
      sums: builder.allocate('state-sums', 'uint32', cellCount * 3),
      moments: builder.allocate('state-moments', 'uint32', cellCount * 3)
    };
    const parameterBuffer = builder.allocate('accumulator-parameters', 'float32', 5);
    builder.graph.add(
      new GPURasterStreamingAccumulator({
        id: 'baseline',
        cellCount,
        chunkSliceCount: sliceCount,
        slotCount: 1,
        values: builder.upload('baseline-values', 'float32', stack.observations).view,
        slotIndices: builder.upload('baseline-slots', 'uint32', new Uint32Array(sliceCount)).view,
        parameters: parameterBuffer.view,
        state: {counts: state.counts.view, sums: state.sums.view, moments: state.moments.view}
      })
    );
    const evaluated = builder.allocate('evaluated', 'float32', cellCount);
    const evaluatedSlot = builder.upload('evaluated-slot', 'uint32', Uint32Array.of(0));
    const anomaly = builder.allocate('anomaly', 'float32', cellCount);
    const climatologyMean = builder.allocate('climatology-mean', 'float32', cellCount);
    builder.graph.add(
      new GPURasterClimateAnomaly({
        cellCount,
        slotCount: 1,
        state: {counts: state.counts.view, sums: state.sums.view, moments: state.moments.view},
        accumulatorParameters: parameterBuffer.view,
        chunkSliceCount: 1,
        values: evaluated.view,
        slotIndices: evaluatedSlot.view,
        standardized: values.standardized === 1,
        output: {anomaly: anomaly.view, climatologyMean: climatologyMean.view}
      })
    );
    const strip = builder.allocate('strip', 'float32', sliceCount);
    return {
      outputs: {anomaly, strip, panel: uploadPanel(builder, stack.panel)},
      update: current => {
        parameterBuffer.write(
          getGPURasterStreamingAccumulatorParameterValues(
            {
              reset: true,
              activeSliceCount: Math.round(current.activeSliceCount),
              maximumValueMagnitude: 1
            },
            sliceCount
          )
        );
        const scene = Math.round(current.scene);
        evaluated.write(
          Float32Array.from(
            {length: cellCount},
            (_, cell) => stack.observations[cell * sliceCount + scene]
          )
        );
        strip.write(getMeanAnomalies(stack, Math.round(current.activeSliceCount)));
      }
    };
  },
  interaction: bindAxes<AcquisitionStack>(parameters, {
    kind: 'time',
    hint: 'scrub the scene',
    x: 'scene',
    decorate: (values, _pointer, {inputs, world}) => {
      const scene = Math.round(values.scene);
      const baseline = Math.round(values.activeSliceCount);
      const mean = getMeanAnomalies(inputs, baseline)[scene];
      const x =
        world.width * (STRIP[0] + ((scene + 0.5) / inputs.sliceCount) * (STRIP[2] - STRIP[0]));
      return {
        glyphs: [
          {
            shape: 'line',
            from: [x, world.height * STRIP[1]],
            to: [x, world.height * STRIP[3]],
            dashed: false
          }
        ],
        readout: `${ACQUISITION_DATES[scene]} vs first ${baseline} scenes · mean NDVI anomaly ${formatNumber(mean, 2)}${
          values.standardized === 1 ? ' · map in z-scores' : ''
        }`
      };
    }
  }),
  draw: ({inputs, outputs, values, theme}) => [
    {
      layer: 'raster',
      values: outputs.anomaly,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'diverging',
      range: values.standardized === 1 ? [-6, 6] : [-0.6, 0.6]
    },
    ...getPanelLayers(outputs.panel, theme),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.strip,
      range: STRIP_RANGE,
      rectangle: STRIP,
      color: getThemeColor(theme.categories[0])
    }
  ]
});
