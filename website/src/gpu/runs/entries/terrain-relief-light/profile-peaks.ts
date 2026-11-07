// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUProfilePeaks on a set of noisy one-dimensional profiles (one per row of a table, stored end to
// end): local maxima with enough prominence within a window, with close neighbors suppressed. The
// line is the input series; the bars mark the peaks the GPU kept. The minimum prominence is
// a per-frame value; window and suppression radius are construction props.

import {
  getGPUProfilePeaksParameterValues,
  GPUProfilePeaks
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {parameters} from './profile-peaks.parameters';

const PROFILE_COUNT = 4;
const SAMPLE_COUNT = 96;

export default defineOperationRun({
  parameters,
  rebuildOn: ['window', 'suppressionRadius'],
  caption: 'Profiles in; the prominent peaks of each profile out.',
  createInputs: ({random}) => {
    const waves = Array.from({length: PROFILE_COUNT}, () => ({
      first: 2 + Math.floor(random() * 3),
      second: 7 + Math.floor(random() * 5),
      phase: random() * Math.PI * 2
    }));
    return {
      noise: Float32Array.from({length: PROFILE_COUNT * SAMPLE_COUNT}, () => random() - 0.5),
      waves
    };
  },
  build(builder, inputs, values) {
    const series = builder.allocate('values', 'float32', PROFILE_COUNT * SAMPLE_COUNT);
    const offsets = builder.upload(
      'offsets',
      'uint32',
      Uint32Array.from({length: PROFILE_COUNT + 1}, (_, index) => index * SAMPLE_COUNT)
    );
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const peakMask = builder.allocate('peak-mask', 'uint32', PROFILE_COUNT * SAMPLE_COUNT);
    const refinedIndex = builder.allocate('refined-index', 'float32', PROFILE_COUNT * SAMPLE_COUNT);
    builder.graph.add(
      new GPUProfilePeaks({
        values: series.view,
        offsets: offsets.view,
        window: values.window,
        suppressionRadius: values.suppressionRadius,
        wrap: true,
        parameters: parameterBuffer.view,
        output: {peakMask: peakMask.view, refinedIndex: refinedIndex.view}
      })
    );
    return {
      outputs: {series, peakMask},
      update: current => {
        series.write(
          Float32Array.from({length: PROFILE_COUNT * SAMPLE_COUNT}, (_, index) => {
            const profile = Math.floor(index / SAMPLE_COUNT);
            const angle = ((index % SAMPLE_COUNT) / SAMPLE_COUNT) * Math.PI * 2;
            const wave = inputs.waves[profile];
            return (
              Math.sin(angle * wave.first + wave.phase) +
              0.6 * Math.sin(angle * wave.second) +
              current.noise * inputs.noise[index] * 2
            );
          })
        );
        parameterBuffer.write(
          getGPUProfilePeaksParameterValues({minimumProminence: current.minimumProminence})
        );
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'chart', context: true, kind: 'line', values: outputs.series, range: [-2.6, 2.6]},
    {layer: 'chart', kind: 'bars', values: outputs.peakMask, ramp: 'sequential', range: [0, 1]}
  ]
});
