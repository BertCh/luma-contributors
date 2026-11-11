// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR,
  VARIOGRAM_MODEL_SHAPE_WGSL
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review, interpolation-density-08. `VARIOGRAM_MODEL_SHAPE_WGSL` is the one
// float32 definition of the variogram shapes `1 - correlation` used by `GPUKriging` and
// `GPUVariogramModelFit`. The direct forms cancel at lags that are a small fraction of the
// range: at 1e-4 of the range the Gaussian shape came out twice too large, and below about 1e-8
// every exponential family returned 0. The expected values are float64 with `Math.expm1`, so no
// cancellation and no shared arithmetic.

const MODEL_COUNT = 5;

/** Shape `1 - correlation` of model code 0..4 in float64, without cancellation. */
function getReferenceShape(modelCode: number, scaled: number): number {
  switch (modelCode) {
    case 0:
      return scaled >= 1 ? 1 : 1.5 * scaled - 0.5 * scaled ** 3;
    case 1:
      return -Math.expm1(-3 * scaled);
    case 2:
      return -Math.expm1(-3 * scaled * scaled);
    case 3: {
      const argument = MATERN_3_2_RANGE_FACTOR * scaled;
      // 1 - (1 + a) exp(-a) = -expm1(-a) - a exp(-a); for small a the two terms nearly cancel
      // (to a^2 / 2), so the float64 series takes over there.
      return argument < 1e-3
        ? argument ** 2 / 2 - argument ** 3 / 3 + argument ** 4 / 8
        : -Math.expm1(-argument) - argument * Math.exp(-argument);
    }
    default: {
      const argument = MATERN_5_2_RANGE_FACTOR * scaled;
      return argument < 1e-3
        ? argument ** 2 / 6 - argument ** 4 / 24 + argument ** 5 / 45
        : -Math.expm1(-argument) - (argument + argument ** 2 / 3) * Math.exp(-argument);
    }
  }
}

/** The argument each model applies its small-argument series to. */
function getSeriesArgument(modelCode: number, scaled: number): number {
  return [
    Number.POSITIVE_INFINITY,
    3 * scaled,
    3 * scaled * scaled,
    MATERN_3_2_RANGE_FACTOR * scaled,
    MATERN_5_2_RANGE_FACTOR * scaled
  ][modelCode];
}

it('VARIOGRAM_MODEL_SHAPE_WGSL keeps float32 accuracy from 1e-9 of the range to beyond it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Six lags per decade from 1e-9 to about 3 ranges, rounded to float32 as the kernel sees them.
  const scaledValues = Array.from({length: 58}, (_, index) => Math.fround(10 ** (-9 + index / 6)));
  const lagCount = scaledValues.length;
  const scaledBuffer = createInputBuffer(device, Float32Array.from(scaledValues));
  const shapesBuffer = createOutputBuffer(device, MODEL_COUNT * lagCount);
  const graph = new GPUCommandGraph(device, {id: 'variogram-model-shape'});
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'variogram-model-shape',
      operation: 'VariogramModelShapeTest',
      bindings: [
        {
          name: 'scaledLags',
          view: importGraphBuffer(graph, 'scaled-lags', scaledBuffer, 'float32', lagCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'shapes',
          view: importGraphBuffer(graph, 'shapes', shapesBuffer, 'float32', MODEL_COUNT * lagCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: MODEL_COUNT * lagCount,
      declarations: VARIOGRAM_MODEL_SHAPE_WGSL,
      body: `let modelCode = index / ${lagCount}u;
  let lag = index % ${lagCount}u;
  shapes[shapesOffset + index] = getVariogramModelShape(modelCode, scaledLags[scaledLagsOffset + lag]);`
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const shapes = await readFloat32(shapesBuffer, MODEL_COUNT * lagCount);
  compiled.destroy();
  scaledBuffer.destroy();
  shapesBuffer.destroy();

  for (let modelCode = 0; modelCode < MODEL_COUNT; modelCode++) {
    for (const [lag, scaled] of scaledValues.entries()) {
      const expected = getReferenceShape(modelCode, scaled);
      const actual = shapes[modelCode * lagCount + lag];
      const label = `model ${modelCode} at ${scaled} of the range`;
      expect(actual, label).toBeGreaterThan(0);
      const relativeError = Math.abs(actual / expected - 1);
      const argument = getSeriesArgument(modelCode, scaled);
      if (argument < 0.09 || modelCode === 0) {
        // Series region (and the spherical polynomial): a few float32 steps.
        expect(relativeError, label).toBeLessThan(2e-6);
      } else if (argument < 1) {
        // Closed form just above the switch: one float32 step of a correlation near 1 is up to
        // 6e-8 / 1.7e-3 of the Matern 5/2 shape at an argument of 0.1.
        expect(relativeError, label).toBeLessThan(1e-4);
      } else {
        expect(relativeError, label).toBeLessThan(2e-6);
      }
    }
  }
});
