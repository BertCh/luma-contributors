// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRegistration estimates the sub-pixel shift between two images by phase correlation. The
// reference image is the moving image displaced by a true shift (the sliders, in pixels, plus
// a little noise); the operation reports the shift that aligns them.
// The plot is a pixel ruler (grid lines every tenth of a pixel); the hollow dot is the true shift
// and the bright dot is the shift the operation wrote. `upsampleFactor` sets how finely the
// correlation peak is located (1/upsampleFactor of a pixel), so low factors snap to a coarse grid.

import {
  GPURasterRegistration,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './raster-registration.parameters';

const SIZE = 64;

/** A periodic scene (integer wavenumbers), so shifting it is exact for FFT registration. */
function getScene(column: number, row: number, phases: readonly number[]): number {
  let value = 0;
  for (let wave = 0; wave < 6; wave++) {
    const kx = (wave * 3 + 1) % 7;
    const ky = (wave * 5 + 2) % 6;
    value +=
      Math.sin(2 * Math.PI * ((kx * column + ky * row) / SIZE) + phases[wave]) / (1 + wave * 0.3);
  }
  return value;
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['upsampleFactor'],
  caption: 'Two images in; their sub-pixel offset out (dot position, in pixels).',
  createInputs: ({random}) => ({
    phases: Array.from({length: 6}, () => random() * Math.PI * 2),
    noise: Float32Array.from({length: SIZE * SIZE}, () => random() - 0.5),
    // Pixel ruler: lines every tenth of a pixel, drawn in the unit square of the plot.
    ruler: Float32Array.from(
      Array.from({length: 11}, (_, tick) => [
        [tick / 10, 0, tick / 10, 1],
        [0, tick / 10, 1, tick / 10]
      ]).flat(2)
    )
  }),
  build(builder, inputs, values) {
    const reference = builder.allocate('reference', 'float32', SIZE * SIZE);
    const moving = builder.allocate('moving', 'float32', SIZE * SIZE);
    const shift = builder.allocate('shift', 'float32x2', 1);
    const peak = builder.allocate('peak', 'float32', 1);
    const status = builder.allocate('status', 'uint32', 1);
    builder.graph.add(
      new GPURasterRegistration({
        grid: makeGPURasterMetadata({width: SIZE, height: SIZE}),
        reference: reference.view,
        moving: moving.view,
        upsampleFactor: values.upsampleFactor,
        output: {shift: shift.view, peak: peak.view, status: status.view}
      })
    );
    const trueShift = builder.allocate('true-shift', 'float32x2', 1);
    const ruler = builder.upload('ruler', 'float32x4', inputs.ruler);
    return {
      outputs: {shift, trueShift, ruler},
      update: current => {
        const referenceData = new Float32Array(SIZE * SIZE);
        const movingData = new Float32Array(SIZE * SIZE);
        for (let row = 0; row < SIZE; row++) {
          for (let column = 0; column < SIZE; column++) {
            const index = row * SIZE + column;
            movingData[index] = getScene(column, row, inputs.phases);
            referenceData[index] =
              getScene(column - current.shiftX, row - current.shiftY, inputs.phases) +
              0.1 * inputs.noise[index];
          }
        }
        reference.write(referenceData);
        moving.write(movingData);
        trueShift.write(Float32Array.of(current.shiftX, current.shiftY));
      }
    };
  },
  draw: ({outputs, theme}) => [
    {layer: 'lines', segments: outputs.ruler, color: [...theme.ink, 0.25]},
    {layer: 'dots', context: true, positions: outputs.trueShift, radius: 34},
    {layer: 'dots', positions: outputs.shift, radius: 24, color: [...theme.ink, 1]}
  ]
});
