// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterRegistration estimates the sub-pixel shift between two images by phase correlation. The
// reference image is the moving image displaced by a true shift (the sliders, in pixels, plus
// a little noise); the operation reports the shift that aligns them.
// Left: the moving image (near-infrared reflectance). Right: the reference minus the moving
// image, subtracted by GPURasterArithmetic; the fringes along the edges of the scene grow with the
// shift. The inset is a pixel ruler (grid lines every tenth of a pixel, x right, y down like the
// image rows): the large dot is the true shift and the dark dot is the shift the operation
// wrote. `upsampleFactor` sets how finely the correlation peak is located (1/upsampleFactor of a
// pixel), so low factors snap to a coarse grid. The pointer sets the true shift, the displacement
// of the reference image that the operation recovers: x across the frame, y down it.

import {
  GPURasterRegistration,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunLabel} from '../../text';
import {sampleReflectance} from '../../../data/reflectance';
import {clamp, findParameter, snapToParameter} from '../../interaction';
import {parameters} from './raster-registration.parameters';
import {addArithmetic} from './support/indices';

const SIZE = 64;
const SHIFT_X = findParameter(parameters, 'shiftX');
const SHIFT_Y = findParameter(parameters, 'shiftY');

const INSET_SIZE = 0.34;
const INSET_MARGIN = 0.05;

/** The pixel-ruler inset in the lower right: left, bottom and side in world units. */
const getInset = (world: {width: number}) => ({
  left: world.width - INSET_MARGIN - INSET_SIZE,
  bottom: INSET_MARGIN,
  size: INSET_SIZE
});

/** Position of a shift (pixels, y down) inside the inset. */
function getInsetPoint(world: {width: number}, shiftX: number, shiftY: number): [number, number] {
  const inset = getInset(world);
  return [inset.left + shiftX * inset.size, inset.bottom + (1 - shiftY) * inset.size];
}

/** Two 64 x 64 images side by side as one 128 x 64 raster. */
function joinSideBySide(left: Float32Array, right: Float32Array, target: Float32Array): void {
  for (let row = 0; row < SIZE; row++) {
    target.set(left.subarray(row * SIZE, (row + 1) * SIZE), row * 2 * SIZE);
    target.set(right.subarray(row * SIZE, (row + 1) * SIZE), row * 2 * SIZE + SIZE);
  }
}

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
  caption:
    'Moving image and reference minus moving in; the sub-pixel offset out (dot in the pixel ruler).',
  createInputs: ({random, world}) => {
    const inset = getInset(world);
    const image = sampleReflectance(SIZE, SIZE).nearInfrared;
    // Ranges from the data, so the scene fills the ramps: the image's own extent, and the
    // difference over a shift of about one pixel (a high percentile of the cell-to-cell steps).
    const steps = Array.from(image, (value, index) => Math.abs(value - image[index + 1])).filter(
      (_, index) => index % SIZE !== SIZE - 1
    );
    steps.sort((a, b) => a - b);
    return {
      world: {width: world.width},
      imageRange: [Math.min(...image), Math.max(...image)] as const,
      differenceRange: Math.max(steps[Math.floor(steps.length * 0.99)], 0.005),
      image,
      spectrum: createSpectrum(sampleReflectance(SIZE, SIZE).nearInfrared),
      noise: Float32Array.from({length: SIZE * SIZE}, () => random() - 0.5),
      // Pixel ruler: lines every tenth of a pixel, drawn in the inset square.
      ruler: Float32Array.from(
        Array.from({length: 11}, (_, tick) => [
          [
            inset.left + (tick / 10) * inset.size,
            inset.bottom,
            inset.left + (tick / 10) * inset.size,
            inset.bottom + inset.size
          ],
          [
            inset.left,
            inset.bottom + (tick / 10) * inset.size,
            inset.left + inset.size,
            inset.bottom + (tick / 10) * inset.size
          ]
        ]).flat(2)
      )
    };
  },
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
    // [moving | reference] minus [nothing | moving], by GPURasterArithmetic: the left half is the
    // moving image, the right half the reference minus the moving image.
    const pairFirst = builder.allocate('pair-first', 'float32', 2 * SIZE * SIZE);
    const pairSecond = builder.allocate('pair-second', 'float32', 2 * SIZE * SIZE);
    const pair = addArithmetic(builder, 'pair', 'subtract', pairFirst, pairSecond);
    const pairFirstData = new Float32Array(2 * SIZE * SIZE);
    const pairSecondData = new Float32Array(2 * SIZE * SIZE);
    joinSideBySide(new Float32Array(SIZE * SIZE), inputs.image, pairSecondData);
    pairSecond.write(pairSecondData);
    const ruler = builder.upload('ruler', 'float32x4', inputs.ruler);
    return {
      outputs: {shift, pair, ruler},
      update: current => {
        const referenceData = shiftImage(inputs.spectrum, current.shiftX, current.shiftY);
        for (let index = 0; index < referenceData.length; index++) {
          referenceData[index] += 0.002 * inputs.noise[index];
        }
        reference.write(referenceData);
        moving.write(inputs.image);
        joinSideBySide(inputs.image, referenceData, pairFirstData);
        pairFirst.write(pairFirstData);
      }
    };
  },
  // The estimate and the true shift are drawn inside the inset ruler.
  async afterRun({inputs, outputs, values, upload}) {
    const bytes = await outputs.shift.buffer.readAsync();
    const shift = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 8));
    return {
      estimate: upload(
        'estimate',
        'float32x2',
        Float32Array.of(...getInsetPoint(inputs.world, shift[0], shift[1]))
      ),
      truth: upload(
        'truth',
        'float32x2',
        Float32Array.of(...getInsetPoint(inputs.world, values.shiftX, values.shiftY))
      ),
      shift
    };
  },
  interaction: {
    kind: 'drag',
    hint: 'drag the true shift',
    rest: () => [
      (SHIFT_X.default - SHIFT_X.min) / (SHIFT_X.max - SHIFT_X.min),
      (SHIFT_Y.default - SHIFT_Y.min) / (SHIFT_Y.max - SHIFT_Y.min)
    ],
    resolve: (pointer, {values, world, peek, outputs}) => {
      const shiftX = snapToParameter(
        SHIFT_X,
        SHIFT_X.min + clamp(pointer.x / world.width, 0, 1) * (SHIFT_X.max - SHIFT_X.min)
      );
      const shiftY = snapToParameter(
        SHIFT_Y,
        SHIFT_Y.min + clamp(1 - pointer.y, 0, 1) * (SHIFT_Y.max - SHIFT_Y.min)
      );
      const estimate = peek(outputs.shift);
      return {
        values: {shiftX, shiftY},
        glyphs: [
          {shape: 'ring', at: getInsetPoint(world, shiftX, shiftY), radius: 0.035, dashed: true}
        ],
        readout: `true [${shiftX.toFixed(2)}, ${shiftY.toFixed(2)}] | got ${estimate ? `[${estimate[0].toFixed(2)}, ${estimate[1].toFixed(2)}]` : '-'} ±${(0.5 / values.upsampleFactor).toFixed(2)} px`
      };
    }
  },
  draw: ({inputs, outputs, extra, theme, world}) => {
    const middle = world.width / 2;
    const inset = getInset(world);
    const look = {layer: 'raster', values: outputs.pair, columns: 2 * SIZE, rows: SIZE} as const;
    const labels: RunLabel[] = [
      {text: 'moving image (near-infrared)', at: [middle / 2, 0.97], anchor: 'top', role: 'title'},
      {text: 'reference − moving', at: [middle * 1.5, 0.97], anchor: 'top', role: 'title'},
      {
        text: 'shift in px: true (large dot), recovered (dark dot)',
        at: [inset.left, inset.bottom + inset.size + 0.012],
        anchor: 'bottom-left',
        role: 'label'
      },
      {text: '0', at: [inset.left, inset.bottom], anchor: 'top-right', role: 'tick'},
      {text: '1', at: [inset.left + inset.size, inset.bottom], anchor: 'top', role: 'tick'}
    ];
    return [
      {...look, ramp: 'grey', range: inputs.imageRange, clip: {side: 'left', x: middle}},
      {
        ...look,
        ramp: 'diverging',
        range: [-inputs.differenceRange, inputs.differenceRange],
        clip: {side: 'right', x: middle}
      },
      {layer: 'lines', segments: outputs.ruler, color: [...theme.ink, 0.5]},
      ...(extra
        ? [
            {layer: 'dots' as const, positions: extra.truth, radius: 18},
            {
              layer: 'dots' as const,
              positions: extra.estimate,
              radius: 11,
              color: [...theme.ink, 1] as const
            }
          ]
        : []),
      {layer: 'text', labels}
    ];
  }
});
