// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRegistration estimates the sub-pixel shift between two images by phase correlation. The
// reference image is the moving image displaced by a true shift (the sliders, in pixels, plus
// a little noise); the operation reports the shift that aligns them.
// The plot is a pixel ruler (grid lines every tenth of a pixel); the hollow dot is the true shift
// and the bright dot is the shift the operation wrote. `upsampleFactor` sets how finely the
// correlation peak is located (1/upsampleFactor of a pixel), so low factors snap to a coarse grid.
// The pointer sets the true shift, the displacement of the reference image that the operation
// recovers; the ruler is centred in the frame.

import {
  GPURasterRegistration,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {sampleReflectance} from '../../../data/reflectance';
import {clamp, findParameter, snapToParameter} from '../../interaction';
import {parameters} from './raster-registration.parameters';

const SIZE = 64;
const SHIFT_X = findParameter(parameters, 'shiftX');
const SHIFT_Y = findParameter(parameters, 'shiftY');

/** Left edge of the unit-square ruler that centres it in the world. */
const getOffset = (world: {width: number}) => (world.width - 1) / 2;

const COSINE = Float64Array.from({length: SIZE}, (_, index) =>
  Math.cos((2 * Math.PI * index) / SIZE)
);
const SINE = Float64Array.from({length: SIZE}, (_, index) =>
  Math.sin((2 * Math.PI * index) / SIZE)
);

/** Separable DFT of a SIZE x SIZE complex image; `sign` -1 forward, +1 inverse (unscaled). */
function transform(real: Float64Array, imaginary: Float64Array, sign: number): void {
  const outputReal = new Float64Array(SIZE);
  const outputImaginary = new Float64Array(SIZE);
  for (const stride of [1, SIZE]) {
    for (let line = 0; line < SIZE; line++) {
      const start = stride === 1 ? line * SIZE : line;
      for (let frequency = 0; frequency < SIZE; frequency++) {
        let sumReal = 0;
        let sumImaginary = 0;
        for (let position = 0; position < SIZE; position++) {
          const phase = (frequency * position) % SIZE;
          const cosine = COSINE[phase];
          const sine = sign * SINE[phase];
          const index = start + position * stride;
          sumReal += real[index] * cosine - imaginary[index] * sine;
          sumImaginary += real[index] * sine + imaginary[index] * cosine;
        }
        outputReal[frequency] = sumReal;
        outputImaginary[frequency] = sumImaginary;
      }
      for (let position = 0; position < SIZE; position++) {
        real[start + position * stride] = outputReal[position];
        imaginary[start + position * stride] = outputImaginary[position];
      }
    }
  }
}

/** The spectrum of an image, for exact periodic sub-pixel shifts by the Fourier shift theorem. */
function createSpectrum(image: Float32Array) {
  const real = Float64Array.from(image);
  const imaginary = new Float64Array(SIZE * SIZE);
  transform(real, imaginary, -1);
  return {real, imaginary};
}

/** The image displaced by (shiftX, shiftY) pixels, periodic like the registration's FFT. */
function shiftImage(
  spectrum: ReturnType<typeof createSpectrum>,
  shiftX: number,
  shiftY: number
): Float32Array {
  const real = new Float64Array(SIZE * SIZE);
  const imaginary = new Float64Array(SIZE * SIZE);
  for (let row = 0; row < SIZE; row++) {
    const rowFrequency = row <= SIZE / 2 ? row : row - SIZE;
    for (let column = 0; column < SIZE; column++) {
      const columnFrequency = column <= SIZE / 2 ? column : column - SIZE;
      const angle = (-2 * Math.PI * (columnFrequency * shiftX + rowFrequency * shiftY)) / SIZE;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const index = row * SIZE + column;
      real[index] = spectrum.real[index] * cosine - spectrum.imaginary[index] * sine;
      imaginary[index] = spectrum.real[index] * sine + spectrum.imaginary[index] * cosine;
    }
  }
  transform(real, imaginary, 1);
  return Float32Array.from(real, value => value / (SIZE * SIZE));
}

export default defineOperationRun({
  parameters,
  rebuildOn: ['upsampleFactor'],
  caption: 'Two images in; their sub-pixel offset out (dot position, in pixels).',
  createInputs: ({random, world}) => ({
    offset: getOffset(world),
    image: sampleReflectance(SIZE, SIZE).nearInfrared,
    spectrum: createSpectrum(sampleReflectance(SIZE, SIZE).nearInfrared),
    noise: Float32Array.from({length: SIZE * SIZE}, () => random() - 0.5),
    // Pixel ruler: lines every tenth of a pixel, drawn in the unit square of the plot.
    ruler: Float32Array.from(
      Array.from({length: 11}, (_, tick) => [
        [getOffset(world) + tick / 10, 0, getOffset(world) + tick / 10, 1],
        [getOffset(world), tick / 10, getOffset(world) + 1, tick / 10]
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
        const referenceData = shiftImage(inputs.spectrum, current.shiftX, current.shiftY);
        for (let index = 0; index < referenceData.length; index++) {
          referenceData[index] += 0.002 * inputs.noise[index];
        }
        reference.write(referenceData);
        moving.write(inputs.image);
        trueShift.write(Float32Array.of(inputs.offset + current.shiftX, current.shiftY));
      }
    };
  },
  // The estimate is read back and drawn inside the centred ruler.
  async afterRun({inputs, outputs, upload}) {
    const bytes = await outputs.shift.buffer.readAsync();
    const shift = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 8));
    return {
      estimate: upload(
        'estimate',
        'float32x2',
        Float32Array.of(inputs.offset + shift[0], shift[1])
      ),
      shift
    };
  },
  interaction: {
    kind: 'drag',
    hint: 'drag the true shift',
    rest: ({world}) => [(getOffset(world) + SHIFT_X.default) / world.width, 1 - SHIFT_Y.default],
    resolve: (pointer, {inputs, outputs, values, peek}) => {
      const shiftX = snapToParameter(
        SHIFT_X,
        clamp(pointer.x - inputs.offset, SHIFT_X.min, SHIFT_X.max)
      );
      const shiftY = snapToParameter(SHIFT_Y, clamp(pointer.y, SHIFT_Y.min, SHIFT_Y.max));
      const estimate = peek(outputs.shift);
      return {
        values: {shiftX, shiftY},
        glyphs: [{shape: 'crosshair', at: [inputs.offset + shiftX, shiftY]}],
        readout: `true [${shiftX.toFixed(2)}, ${shiftY.toFixed(2)}] | got ${estimate ? `[${estimate[0].toFixed(2)}, ${estimate[1].toFixed(2)}]` : '-'} ±${(0.5 / values.upsampleFactor).toFixed(2)} px`
      };
    }
  },
  draw: ({outputs, extra, theme}) => [
    {layer: 'lines', segments: outputs.ruler, color: [...theme.ink, 0.25]},
    {layer: 'dots', context: true, positions: outputs.trueShift, radius: 34},
    ...(extra
      ? [
          {
            layer: 'dots' as const,
            positions: extra.estimate,
            radius: 24,
            color: [...theme.ink, 1] as const
          }
        ]
      : [])
  ]
});
