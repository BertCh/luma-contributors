// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUProfilePeaks on a set of noisy one-dimensional profiles (one per row of a table, stored end to
// end): local maxima with enough prominence within a window, with close neighbors suppressed. The
// four profiles are drawn stacked, the dots are the peaks the GPU kept. The minimum prominence is
// a per-frame value; window and suppression radius are construction props.

import {
  getGPUProfilePeaksParameterValues,
  GPUProfilePeaks
} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {sampleElevation} from '../../../data/elevation';

import {parameters} from './profile-peaks.parameters';

const PROFILE_COUNT = 4;
const SAMPLE_COUNT = 96;
// Vertical layout of the stacked profiles in world units of height.
const BAND_HEIGHT = 0.2;
const BAND_GAP = 0.04;
const BAND_BOTTOM = 0.04;
const MARGIN = 0.04;

/** World position of sample `sample` of profile `profile` at `height` meters. */
function getProfilePoint(
  worldWidth: number,
  profile: number,
  sample: number,
  height: number,
  range: readonly [number, number]
): [number, number] {
  // The top profile (row 0 of the table) is drawn highest.
  const band = PROFILE_COUNT - 1 - profile;
  const bandBottom = BAND_BOTTOM + band * (BAND_HEIGHT + BAND_GAP);
  const unit = (height - range[0]) / (range[1] - range[0]);
  return [
    MARGIN * worldWidth + (sample / (SAMPLE_COUNT - 1)) * (1 - 2 * MARGIN) * worldWidth,
    bandBottom + Math.min(Math.max(unit, 0), 1) * BAND_HEIGHT
  ];
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['window', 'suppressionRadius'],
  caption: 'Four east–west Grand Canyon transects in; their prominent peaks out.',
  createInputs: ({random, world}) => {
    const {elevation} = sampleElevation(SAMPLE_COUNT, 60);
    const profiles = Float32Array.from({length: PROFILE_COUNT * SAMPLE_COUNT}, (_, index) => {
      const row = Math.floor(((Math.floor(index / SAMPLE_COUNT) + 0.5) * 60) / PROFILE_COUNT);
      return elevation[row * SAMPLE_COUNT + (index % SAMPLE_COUNT)];
    });
    // Each profile fills its own band: heights are scaled to its own minimum and maximum, with room
    // for the noise.
    const ranges = Array.from({length: PROFILE_COUNT}, (_, profile) => {
      const own = profiles.subarray(profile * SAMPLE_COUNT, (profile + 1) * SAMPLE_COUNT);
      return [Math.min(...own) - 60, Math.max(...own) + 60] as const;
    });
    return {
      ranges,
      profiles,
      noise: Float32Array.from(profiles, () => random() - 0.5),
      width: world.width
    };
  },
  build(builder, inputs, values) {
    const sampleTotal = PROFILE_COUNT * SAMPLE_COUNT;
    const series = builder.allocate('values', 'float32', sampleTotal);
    const offsets = builder.upload(
      'offsets',
      'uint32',
      Uint32Array.from({length: PROFILE_COUNT + 1}, (_, index) => index * SAMPLE_COUNT)
    );
    const parameterBuffer = builder.allocate('parameters', 'float32', 1);
    const peakMask = builder.allocate('peak-mask', 'uint32', sampleTotal);
    const refinedIndex = builder.allocate('refined-index', 'float32', sampleTotal);
    builder.graph.add(
      new GPUProfilePeaks({
        values: series.view,
        offsets: offsets.view,
        window: values.window,
        suppressionRadius: values.suppressionRadius,
        wrap: false,
        parameters: parameterBuffer.view,
        output: {peakMask: peakMask.view, refinedIndex: refinedIndex.view}
      })
    );
    // Render-only: the profiles as segments and the kept peaks as dots, in world units.
    const lines = builder.allocate('profile-lines', 'float32x4', sampleTotal);
    const peakDots = builder.allocate('peak-dots', 'float32x2', sampleTotal);
    return {
      outputs: {series, peakMask, lines, peakDots},
      update: current => {
        const noisy = inputs.profiles.map(
          (height, index) => height + current.noise * inputs.noise[index]
        );
        series.write(noisy);
        const segments = new Float32Array(sampleTotal * 4);
        for (let profile = 0; profile < PROFILE_COUNT; profile++) {
          for (let sample = 0; sample < SAMPLE_COUNT - 1; sample++) {
            const index = profile * SAMPLE_COUNT + sample;
            segments.set(
              [
                ...getProfilePoint(
                  inputs.width,
                  profile,
                  sample,
                  noisy[index],
                  inputs.ranges[profile]
                ),
                ...getProfilePoint(
                  inputs.width,
                  profile,
                  sample + 1,
                  noisy[index + 1],
                  inputs.ranges[profile]
                )
              ],
              index * 4
            );
          }
        }
        lines.write(segments);
        parameterBuffer.write(
          getGPUProfilePeaksParameterValues({minimumProminence: current.minimumProminence})
        );
      }
    };
  },
  // The pointer height is minimumProminence (up is more): every kept peak hangs a bar of that
  // prominence below it, and the ruler marks the same height.
  interaction: {
    kind: 'level',
    hint: 'raise the prominence bar',
    rest: [0.5, 1 - getFractionOf(findParameter(parameters, 'minimumProminence'), 30)],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const minimumProminence = getParameterAt(
        findParameter(parameters, 'minimumProminence'),
        1 - pointer.unitY
      );
      const mask = peek(outputs.peakMask);
      const series = peek(outputs.series);
      const glyphs: RunGlyph[] = [
        {shape: 'line', from: [0, pointer.y], to: [world.width, pointer.y], dashed: true}
      ];
      let peakCount = 0;
      if (mask && series) {
        // The kept peaks, as dots on the profiles; unused slots sit outside the picture.
        const dots = new Float32Array(mask.length * 2).fill(-10);
        for (let index = 0; index < mask.length; index++) {
          if (mask[index] === 0) continue;
          peakCount++;
          const profile = Math.floor(index / SAMPLE_COUNT);
          const range = inputs.ranges[profile];
          const [x, y] = getProfilePoint(
            inputs.width,
            profile,
            index % SAMPLE_COUNT,
            series[index],
            range
          );
          const barHeight = (minimumProminence / (range[1] - range[0])) * BAND_HEIGHT;
          dots.set([x, y], index * 2);
          glyphs.push({shape: 'line', from: [x, y], to: [x, y - barHeight]});
        }
        outputs.peakDots.write(dots);
      }
      return {
        values: {minimumProminence},
        glyphs,
        readout: `minimumProminence ${minimumProminence} m${mask ? ` · ${peakCount} peaks` : ''}${
          values.noise > 0 ? ` · noise ${values.noise} m` : ''
        }`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'lines', segments: outputs.lines, radius: 2},
    {
      layer: 'dots',
      positions: outputs.peakDots,
      radius: 9,
      color: [0.95, 0.45, 0.1, 1]
    }
  ]
});
