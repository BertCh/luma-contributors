// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  getGPURasterStretchParameterValues,
  GPU_RASTER_STRETCH_PARAMETER_LENGTH,
  GPURasterStretch,
  type GPURasterStretchOutput,
  type GPURasterStretchSettings
} from '../../../src/raster-analysis/raster-stretch/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from '../raster-algebra/raster-algebra-test-utils';
import {
  computeRasterStretchLutOnCPU,
  computeRasterStretchStatistics,
  getPaletteColorOnCPU,
  packColor,
  stretchRasterOnCPU,
  type RasterStretchScene
} from './raster-stretch-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import stretchReference from '../../oracles/fixtures/raster-stretch-references.json';

type OutputName = 'stretched' | 'colors' | 'lut' | 'lutColors' | 'histogram' | 'statistics';
const ALL_OUTPUTS: OutputName[] = [
  'stretched',
  'colors',
  'lut',
  'lutColors',
  'histogram',
  'statistics'
];

type HarnessOptions = {
  scene: RasterStretchScene;
  palette?: Uint32Array;
  outputs?: OutputName[];
};

/** Compiles one graph; `run` rewrites parameters, mask and palette and returns the readbacks. */
function createHarness(device: Device, options: HarnessOptions) {
  const {scene} = options;
  const {width, height, binCount, lutSize} = scene;
  const cellCount = width * height;
  const outputs =
    options.outputs ??
    ALL_OUTPUTS.filter(name => options.palette || (name !== 'colors' && name !== 'lutColors'));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'stretch-parameters',
    format: 'float32',
    length: GPU_RASTER_STRETCH_PARAMETER_LENGTH
  });
  const valuesBuffer = new GPUParameterBuffer(device, {
    id: 'stretch-values',
    format: 'float32',
    length: cellCount,
    values: scene.values
  });
  const maskBuffer = scene.regionMask
    ? new GPUParameterBuffer(device, {
        id: 'stretch-mask',
        format: 'uint32',
        length: cellCount,
        values: scene.regionMask
      })
    : undefined;
  const paletteBuffer = options.palette
    ? new GPUParameterBuffer(device, {
        id: 'stretch-palette',
        format: 'uint32',
        length: options.palette.length,
        values: options.palette
      })
    : undefined;
  const validityBuffer = scene.validity ? createInputBuffer(device, scene.validity) : undefined;
  const lengths: Record<OutputName, number> = {
    stretched: cellCount,
    colors: cellCount,
    lut: lutSize,
    lutColors: lutSize,
    histogram: binCount,
    statistics: 8
  };

  const formats: Record<OutputName, 'float32' | 'uint32'> = {
    stretched: 'float32',
    colors: 'uint32',
    lut: 'float32',
    lutColors: 'uint32',
    histogram: 'uint32',
    statistics: 'float32'
  };

  const outputBuffers = Object.fromEntries(
    outputs.map(name => [name, createOutputBuffer(device, lengths[name])])
  ) as Record<OutputName, ReturnType<typeof createOutputBuffer>>;
  const graph = new GPUCommandGraph(device, {id: 'stretch-graph'});
  const createSelection = (name: OutputName): GPURasterStretchOutput => {
    const buffer = outputBuffers[name];
    switch (name) {
      case 'stretched':
        return {
          stretched: importGraphBuffer(graph, `out-${name}`, buffer, 'float32', lengths[name])
        };
      case 'colors':
        return {colors: importGraphBuffer(graph, `out-${name}`, buffer, 'uint32', lengths[name])};
      case 'lut':
        return {lut: importGraphBuffer(graph, `out-${name}`, buffer, 'float32', lengths[name])};
      case 'lutColors':
        return {
          lutColors: importGraphBuffer(graph, `out-${name}`, buffer, 'uint32', lengths[name])
        };
      case 'histogram':
        return {
          histogram: importGraphBuffer(graph, `out-${name}`, buffer, 'uint32', lengths[name])
        };
      case 'statistics':
        return {
          statistics: importGraphBuffer(graph, `out-${name}`, buffer, 'float32', lengths[name])
        };
    }
  };
  const [firstOutput, ...remainingOutputs] = outputs;
  if (firstOutput === undefined)
    throw new Error('stretch fixture requires a requested terminal output');
  const output = createSelection(firstOutput);
  for (const name of remainingOutputs) Object.assign(output, createSelection(name));
  graph.add(
    new GPURasterStretch({
      id: 'stretch',
      values: valuesBuffer.importToGraph(graph),
      grid: makeGPURasterMetadata({width, height}),
      noDataValue: scene.noDataValue,
      mask: validityBuffer
        ? importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', cellCount)
        : undefined,
      regionMask: maskBuffer?.importToGraph(graph),
      parameters: parameterBuffer.importToGraph(graph),
      palette: paletteBuffer?.importToGraph(graph),
      binCount,
      lutSize,
      output
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(
      settings: GPURasterStretchSettings,
      update?: {mask?: Uint32Array; palette?: Uint32Array}
    ) {
      if (update?.mask) {
        maskBuffer!.write(update.mask);
      }
      if (update?.palette) {
        paletteBuffer!.write(update.palette);
      }
      const parameters = getGPURasterStretchParameterValues(settings);
      parameterBuffer.write(parameters);
      submitGraph(device, compiled, undefined);
      const result: Partial<Record<OutputName, number[]>> = {};
      for (const name of outputs) {
        result[name] =
          formats[name] === 'float32'
            ? await readFloat32(outputBuffers[name], lengths[name])
            : await readUint32(outputBuffers[name], lengths[name]);
      }
      return {parameters, result};
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      valuesBuffer.destroy();
      maskBuffer?.destroy();
      paletteBuffer?.destroy();
      validityBuffer?.destroy();
      for (const buffer of Object.values(outputBuffers)) {
        buffer.destroy();
      }
    }
  };
}

type Harness = ReturnType<typeof createHarness>;

function expectClose(actual: number, expected: number, tolerance: number, label: string): void {
  if (Number.isNaN(expected) || Number.isNaN(actual)) {
    expect(Number.isNaN(actual), `${label} nodata (gpu ${actual}, cpu ${expected})`).toBe(
      Number.isNaN(expected)
    );
    return;
  }
  expect(Math.abs(actual - expected), `${label} gpu ${actual} cpu ${expected}`).toBeLessThanOrEqual(
    tolerance
  );
}

type CheckOptions = {
  palette?: Uint32Array;
  /** Allowed L1 distance between GPU and CPU histograms (0 when bin arithmetic is exact). */
  histogramTolerance?: number;
  /** Tolerance of lo/hi in bin widths; the GPU divides in f32. */
  boundTolerance?: number;
  /** Absolute tolerance of the stretched values and the LUT. */
  valueTolerance?: number;
};

/** Runs one frame and compares every output with the CPU oracle. */
async function checkFrame(
  harness: Harness,
  scene: RasterStretchScene,
  settings: GPURasterStretchSettings,
  options: CheckOptions = {},
  update?: {mask?: Uint32Array; palette?: Uint32Array}
) {
  const {parameters, result} = await harness.run(settings, update);
  if (update?.mask) {
    // The GPU keeps the written mask for later frames.
    scene.regionMask = update.mask;
  }
  const live = scene;
  const stats = computeRasterStretchStatistics(live, parameters);
  const histogramTolerance = options.histogramTolerance ?? 0;
  const histogram = result.histogram!;
  let distance = 0;
  for (let bin = 0; bin < scene.binCount; bin++) {
    distance += Math.abs(histogram[bin] - stats.histogram[bin]);
  }
  expect(distance, 'histogram L1 distance').toBeLessThanOrEqual(histogramTolerance);
  if (histogramTolerance === 0) {
    expect(histogram).toEqual(Array.from(stats.histogram));
  }
  const statistics = result.statistics!;
  const nan = Number.NaN;
  expect(statistics[4], 'validCount').toBe(stats.validCount);
  expect(statistics[6]).toBe(0);
  expect(statistics[7]).toBe(0);
  expectClose(statistics[0], stats.isEmpty ? nan : stats.domainMinimum, 0, 'domainMinimum');
  expectClose(statistics[1], stats.isEmpty ? nan : stats.domainMaximum, 0, 'domainMaximum');
  expectClose(
    statistics[5],
    stats.isEmpty ? nan : stats.binWidth,
    1e-6 * stats.binWidth,
    'binWidth'
  );
  const boundTolerance = (options.boundTolerance ?? 1e-4) * Math.max(stats.binWidth, 1e-12);
  expectClose(statistics[2], stats.isEmpty ? nan : stats.lo, boundTolerance, 'lo');
  expectClose(statistics[3], stats.isEmpty ? nan : stats.hi, boundTolerance, 'hi');
  // Compare the per-cell maps against the oracle evaluated at the GPU's own bounds, so that f32
  // division in the bound search does not leak into the per-cell tolerance.
  const gpuStats = {
    ...stats,
    lo: stats.isEmpty ? 0 : statistics[2],
    hi: stats.isEmpty ? 0 : statistics[3]
  };
  const valueTolerance = options.valueTolerance ?? 2e-5;
  const expectedStretched = stretchRasterOnCPU(live, parameters, gpuStats);
  const stretched = result.stretched!;
  for (let cell = 0; cell < expectedStretched.length; cell++) {
    expectClose(stretched[cell], expectedStretched[cell], valueTolerance, `stretched[${cell}]`);
    if (!Number.isNaN(stretched[cell])) {
      expect(stretched[cell]).toBeGreaterThanOrEqual(0);
      expect(stretched[cell]).toBeLessThanOrEqual(1);
    }
  }
  const expectedLut = computeRasterStretchLutOnCPU(live, parameters, gpuStats);
  const lut = result.lut!;
  for (let index = 0; index < expectedLut.length; index++) {
    expectClose(lut[index], expectedLut[index], valueTolerance, `lut[${index}]`);
  }
  if (options.palette) {
    const palette = update?.palette ?? options.palette;
    const linear = settings.paletteInterpolation === 'linear';
    const compareColors = (actual: number[], source: number[], label: string) => {
      for (let index = 0; index < source.length; index++) {
        const expected = getPaletteColorOnCPU(source[index], palette, linear);
        if (linear) {
          for (let channel = 0; channel < 4; channel++) {
            const shift = 8 * channel;
            const delta = Math.abs(
              ((actual[index] >>> shift) & 255) - ((expected >>> shift) & 255)
            );
            expect(delta, `${label}[${index}] channel ${channel}`).toBeLessThanOrEqual(1);
          }
        } else {
          expect(actual[index], `${label}[${index}]`).toBe(expected);
        }
      }
    };
    // Colors derive from the GPU's own t, so nearest lookups are exact.
    compareColors(result.colors!, stretched, 'colors');
    compareColors(result.lutColors!, lut, 'lutColors');
  }
  return {parameters, result, stats, scene: live};
}

/** Integers in [0, 1024] with both endpoints present, NaN, sentinel and zero-validity cells. */
function createIntegerScene(seed: number, width: number, height: number): RasterStretchScene {
  const random = createRandom(seed);
  const cellCount = width * height;
  const values = new Float32Array(cellCount);
  const validity = new Uint32Array(cellCount).fill(1);
  for (let cell = 0; cell < cellCount; cell++) {
    // Skewed so that equalization differs from a linear stretch.
    values[cell] = Math.round(random() ** 2 * 1024);
    const roll = random();
    if (roll < 0.03) {
      values[cell] = NaN;
    } else if (roll < 0.06) {
      values[cell] = -9999;
    } else if (roll < 0.09) {
      validity[cell] = 0;
    } else if (roll < 0.1) {
      values[cell] = roll < 0.095 ? Infinity : -Infinity;
    }
  }
  values[3] = 0;
  values[4] = 1024;
  validity[3] = 1;
  validity[4] = 1;
  return {values, width, height, noDataValue: -9999, validity, binCount: 1024, lutSize: 64};
}

const PALETTE = Uint32Array.from([
  packColor(0, 0, 128, 255),
  packColor(0, 200, 255, 255),
  packColor(255, 255, 0, 200),
  packColor(255, 0, 0, 255),
  packColor(120, 10, 10, 255)
]);

it('GPURasterStretch matches the CPU oracle across modes, domains, windows and masks without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createIntegerScene(5, 37, 29);
  scene.regionMask = new Uint32Array(37 * 29).map((_, cell) => ((cell * 7) % 5 === 0 ? 0 : 1));
  const harness = createHarness(device, {scene, palette: PALETTE});
  const options = {palette: PALETTE};
  const frames: GPURasterStretchSettings[] = [
    {mode: 'linear'},
    {mode: 'linear', domain: [0, 512], paletteInterpolation: 'linear'},
    {mode: 'percentile', percentiles: [2, 98]},
    {mode: 'percentile', percentiles: [0, 100], gamma: 0.5},
    {
      mode: 'percentile',
      percentiles: [10, 90],
      gamma: 2.2,
      sigmoidContrast: 6,
      sigmoidMidpoint: 0.4
    },
    {mode: 'equalize', paletteInterpolation: 'linear'},
    {mode: 'equalize', domain: [0, 512], sigmoidContrast: 4},
    {mode: 'linear', window: [10, 5, 30, 20]},
    {mode: 'equalize', window: [10, 5, 30, 20], domain: [0, 1024]},
    {mode: 'percentile', window: [-5, -5, 1000, 1000], percentiles: [5, 95], gamma: 1.5}
  ];
  for (const [frameIndex, settings] of frames.entries()) {
    // Mask contents change between frames; frame 0 keeps the initial mask.
    const update =
      frameIndex === 4
        ? {mask: new Uint32Array(37 * 29).map((_, cell) => (cell % 3 === 0 ? 1 : 0))}
        : undefined;
    const {stats} = await checkFrame(harness, scene, settings, options, update);
    expect(stats.validCount, `frame ${frameIndex} has data`).toBeGreaterThan(50);
  }
  // Palette contents change per frame too.
  await checkFrame(harness, scene, {mode: 'percentile'}, options, {
    palette: Uint32Array.from([
      packColor(1, 2, 3, 4),
      packColor(250, 251, 252, 253),
      0,
      7,
      0xffffffff
    ])
  });
  expect(harness.rebuildCount).toBe(0);
  harness.destroy();
});

it('GPURasterStretch automatic domain is the exact minimum and maximum, percentile bounds honor the bin error', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(21);
  const width = 64;
  const height = 48;
  const values = new Float32Array(width * height);
  for (let cell = 0; cell < values.length; cell++) {
    values[cell] = (random() - 0.3) * 50 + random() * random() * 7;
  }
  values[11] = NaN;
  const scene: RasterStretchScene = {values, width, height, binCount: 1024, lutSize: 256};
  const harness = createHarness(device, {scene});
  // The f32 bin arithmetic is not exact for this domain: allow cells near bin edges to differ.
  const {stats, result} = await checkFrame(
    harness,
    scene,
    {mode: 'percentile', percentiles: [2, 98]},
    {histogramTolerance: 24, boundTolerance: 3}
  );
  const finite = Array.from(values)
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  expect(stats.domainMinimum).toBe(finite[0]);
  expect(stats.domainMaximum).toBe(finite[finite.length - 1]);
  // lo/hi are the centre of the bin that holds order statistic floor((N - 1) * p) of the sorted
  // data: half a bin width, plus the float32 rounding of a value that sits on a bin edge.
  const exactLow = finite[Math.floor(0.02 * (finite.length - 1))];
  const exactHigh = finite[Math.floor(0.98 * (finite.length - 1))];
  expect(Math.abs(result.statistics![2] - exactLow)).toBeLessThanOrEqual(0.75 * stats.binWidth);
  expect(Math.abs(result.statistics![3] - exactHigh)).toBeLessThanOrEqual(0.75 * stats.binWidth);
  // Equalization is a step function of the bin, so a value on a bin edge (where the float32 bin
  // arithmetic of GPU and oracle may differ) moves by the share of one bin, plus the cells the
  // histogram tolerance allows.
  const equalizeStatistics = computeRasterStretchStatistics(
    scene,
    getGPURasterStretchParameterValues({mode: 'equalize'})
  );
  const largestBinCount = equalizeStatistics.histogram.reduce(
    (largest, count) => Math.max(largest, count),
    0
  );
  await checkFrame(
    harness,
    scene,
    {mode: 'equalize'},
    {histogramTolerance: 24, valueTolerance: (largestBinCount + 24) / equalizeStatistics.total}
  );
  harness.destroy();
});

it('GPURasterStretch defines all-nodata, empty-window and constant rasters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // All nodata: NaN everywhere, zero counts, no crash.
  const nodata: RasterStretchScene = {
    values: new Float32Array(20).fill(NaN),
    width: 5,
    height: 4,
    binCount: 16,
    lutSize: 8
  };
  const nodataHarness = createHarness(device, {scene: nodata, palette: PALETTE});
  for (const mode of ['linear', 'percentile', 'equalize'] as const) {
    const {result} = await nodataHarness.run({mode});
    expect(result.stretched!.every(Number.isNaN)).toBe(true);
    expect(result.colors!.every(color => color === 0)).toBe(true);
    expect(result.histogram!.every(count => count === 0)).toBe(true);
    expect(result.statistics![4]).toBe(0);
    expect([0, 1, 2, 3, 5].every(index => Number.isNaN(result.statistics![index]))).toBe(true);
  }
  // An explicit domain stays defined without data.
  const explicit = await nodataHarness.run({domain: [2, 6]});
  expect(Array.from(explicit.result.statistics!)).toEqual([2, 6, 2, 6, 0, 0.25, 0, 0]);
  expect(nodataHarness.rebuildCount).toBe(0);
  nodataHarness.destroy();

  // Constant raster: lo == hi, every cell maps to 0.5, histogram holds all cells in bin 0.
  const constant: RasterStretchScene = {
    values: new Float32Array(30).fill(7),
    width: 6,
    height: 5,
    binCount: 16,
    lutSize: 8
  };
  const constantHarness = createHarness(device, {scene: constant, palette: PALETTE});
  for (const mode of ['linear', 'percentile', 'equalize'] as const) {
    const {result} = await constantHarness.run({mode, gamma: 2, sigmoidContrast: 5});
    expect(Array.from(result.statistics!)).toEqual([7, 7, 7, 7, 30, 0, 0, 0]);
    expect(result.stretched!.every(value => value === 0.5)).toBe(true);
    expect(result.histogram![0]).toBe(30);
    expect(result.lut!.every(value => value === 0.5)).toBe(true);
    // Nearest lookup of t = 0.5 in a five-color palette is index 2.
    expect(result.colors!.every(color => color === PALETTE[2])).toBe(true);
  }
  // A wider explicit domain makes the constant a normal value.
  const wide = await constantHarness.run({domain: [0, 10]});
  expect(wide.result.stretched!.every(value => Math.abs(value - 0.7) < 1e-6)).toBe(true);
  // Values outside an explicit domain clamp.
  const clamped = await constantHarness.run({domain: [8, 9]});
  expect(clamped.result.stretched!.every(value => value === 0)).toBe(true);
  expect(clamped.result.histogram!.every(count => count === 0)).toBe(true);
  expect(clamped.result.statistics![4]).toBe(30);
  expect(constantHarness.rebuildCount).toBe(0);
  constantHarness.destroy();

  // A window with no cells reports empty statistics but the apply step still covers every cell:
  // valid cells fall back to the degenerate mapping against lo = hi = 0.
  const mixed = createIntegerScene(8, 12, 9);
  const mixedHarness = createHarness(device, {scene: mixed});
  const {result: empty} = await mixedHarness.run({window: [3, 3, 3, 3]});
  expect(empty.statistics![4]).toBe(0);
  expect(Number.isNaN(empty.statistics![2])).toBe(true);
  for (let cell = 0; cell < mixed.values.length; cell++) {
    const value = mixed.values[cell];
    const isNoData = Number.isNaN(value) || value === -9999 || mixed.validity![cell] === 0;
    const expected = isNoData ? NaN : value < 0 ? 0 : value > 0 ? 1 : 0.5;
    expectClose(empty.stretched![cell], expected, 0, `stretched[${cell}]`);
  }
  mixedHarness.destroy();
});

it('GPURasterStretch supports output subsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene = createIntegerScene(3, 20, 10);
  const histogramOnly = createHarness(device, {scene, outputs: ['histogram', 'statistics']});
  const {parameters, result} = await histogramOnly.run({mode: 'percentile'});
  expect(result.histogram).toEqual(
    Array.from(computeRasterStretchStatistics(scene, parameters).histogram)
  );
  histogramOnly.destroy();
  // Only lutColors (and the palette); then only colors.
  const lutOnly = createHarness(device, {scene, palette: PALETTE, outputs: ['lutColors']});
  const lutResult = await lutOnly.run({mode: 'equalize', paletteInterpolation: 'linear'});
  expect(lutResult.result.lutColors!.length).toBe(scene.lutSize);
  expect(lutResult.result.lutColors!.some(color => color !== 0)).toBe(true);
  lutOnly.destroy();
  const colorsOnly = createHarness(device, {scene, palette: PALETTE, outputs: ['colors']});
  const colorResult = await colorsOnly.run({});
  expect(colorResult.result.colors!.some(color => color !== 0)).toBe(true);
  colorsOnly.destroy();
});

it('GPURasterStretch matches the CPU oracle over many workgroup tiles (privatized histogram and tree reduction)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 199 x 157 = 31243 cells: several tiles of the workgroup-private reduce and histogram passes,
  // with a ragged last tile.
  const scene = createIntegerScene(9, 199, 157);
  const harness = createHarness(device, {scene, palette: PALETTE});
  for (const frame of [
    {mode: 'linear'},
    {mode: 'percentile', percentiles: [2, 98]},
    {mode: 'equalize'}
  ] as GPURasterStretchSettings[]) {
    await checkFrame(harness, scene, frame, {palette: PALETTE});
  }
  harness.destroy();
});

// ---------------------------------------------------------------------------------------------
// Hand-computed cases (math correctness review 2026-11-10, raster-imagery-05, -06 and -12).
// Every expected number below is worked out in the comment next to it, not taken from the oracle.

async function runSmallCase(
  values: number[],
  width: number,
  settings: GPURasterStretchSettings,
  palette?: Uint32Array,
  binCount = 1024
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scene: RasterStretchScene = {
    values: Float32Array.from(values),
    width,
    height: values.length / width,
    binCount,
    lutSize: 4
  };
  const harness = createHarness(device, {scene, palette});
  const {result} = await harness.run(settings);
  harness.destroy();
  return result;
}

it('GPURasterStretch linear, gamma and sigmoid match hand-computed values', async () => {
  // Domain [2, 10]: t = (v - 2) / 8.
  const linear = await runSmallCase([2, 4, 6, 10], 2, {mode: 'linear'});
  expect(linear.statistics!.slice(0, 5)).toEqual([2, 10, 2, 10, 4]);
  [0, 0.25, 0.5, 1].forEach((expected, cell) =>
    expectClose(linear.stretched![cell], expected, 1e-6, `linear[${cell}]`)
  );
  // Display gamma 2: t ** (1 / 2).
  const gamma = await runSmallCase([2, 4, 6, 10], 2, {mode: 'linear', gamma: 2});
  [0, 0.5, Math.SQRT1_2, 1].forEach((expected, cell) =>
    expectClose(gamma.stretched![cell], expected, 1e-6, `gamma[${cell}]`)
  );
  // Sigmoid, contrast 10, midpoint 0.5 (rio-color): s(t) = 1 / (1 + e^(10 (0.5 - t))).
  // s(0) = 0.0066929, s(1) = 0.9933071, s(0.25) = 0.0758582, s(0.5) = 0.5.
  // (s(0.25) - s(0)) / (s(1) - s(0)) = 0.0691653 / 0.9866143 = 0.0701037.
  const sigmoid = await runSmallCase([2, 4, 6, 10], 2, {mode: 'linear', sigmoidContrast: 10});
  [0, 0.0701037, 0.5, 1].forEach((expected, cell) =>
    expectClose(sigmoid.stretched![cell], expected, 2e-6, `sigmoid[${cell}]`)
  );
});

it('GPURasterStretch percentile bounds follow the rank rule of GPURasterGlobalPercentile', async () => {
  const values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
  // N = 10, 1024 bins of width 90 / 1024 over [0, 90].
  // p = 0.50: rank floor(9 * 0.50) = 4, the value 40, bin floor(40 * 1024 / 90) = 455,
  //           centre 455.5 * 90 / 1024 = 40.0342.
  // p = 0.98: rank floor(9 * 0.98) = 8, the value 80, bin floor(80 * 1024 / 90) = 910,
  //           centre 910.5 * 90 / 1024 = 80.0244.
  // (The rule before this pass gave 40.08 and 89.98; NumPy's linear rule gives 45 and 88.2.)
  const result = await runSmallCase(values, 5, {mode: 'percentile', percentiles: [50, 98]});
  expectClose(result.statistics![2], 40.0342, 1e-3, 'lo');
  expectClose(result.statistics![3], 80.0244, 1e-3, 'hi');
  // Applied: (60 - 40.0342) / (80.0244 - 40.0342) = 0.49927; values at or below lo give 0 and
  // values at or above hi give 1.
  const expected = [0, 0, 0, 0, 0, 0.2492, 0.49927, 0.74932, 0.99939, 1];
  expected.forEach((value, cell) =>
    expectClose(result.stretched![cell], value, 5e-5, `stretched[${cell}]`)
  );
  // p = 0 and p = 1 are the exact extrema.
  const full = await runSmallCase(values, 5, {mode: 'percentile', percentiles: [0, 100]});
  expect(full.statistics!.slice(2, 4)).toEqual([0, 90]);
  // p = 0.10: rank floor(0.9) = 0, the value 0, bin 0, centre 0.5 * 90 / 1024 = 0.043945.
  // p = 0.90: rank floor(8.1) = 8, as above.
  const tails = await runSmallCase(values, 5, {mode: 'percentile', percentiles: [10, 90]});
  expectClose(tails.statistics![2], 0.043945, 1e-5, 'lo at 10 percent');
  expectClose(tails.statistics![3], 80.0244, 1e-3, 'hi at 90 percent');
});

it('GPURasterStretch equalization maps the middle of three equally frequent values to 0.5', async () => {
  // GPURasterContrast and OpenCV equalizeHist: (cdf[bin] - cdfFirst) / (total - cdfFirst).
  // Four cells of each value: cdf is 4, 8, 12, cdfFirst is 4, so the values map to
  // (4 - 4) / 8 = 0, (8 - 4) / 8 = 0.5 and (12 - 4) / 8 = 1 wherever the middle value sits.
  // (The rule before this pass gave 0.523, 0.333 and 0.379 for the three middle values.)
  for (const middle of [100, 127.5, 200]) {
    const values = [0, middle, 255, 0, middle, 255, 0, middle, 255, 0, middle, 255];
    const result = await runSmallCase(values, 4, {mode: 'equalize'});
    values.forEach((value, cell) =>
      expect(result.stretched![cell], `middle ${middle} cell ${cell}`).toBe(
        value === 0 ? 0 : value === 255 ? 1 : 0.5
      )
    );
  }
  // Counts 1, 2, 3, 4 of the values 1, 2, 3, 4: cdf 1, 3, 6, 10, cdfFirst 1.
  // (1 - 1) / 9 = 0, (3 - 1) / 9 = 0.2222, (6 - 1) / 9 = 0.5556, (10 - 1) / 9 = 1.
  const ramp = await runSmallCase([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], 5, {mode: 'equalize'});
  [0, 2 / 9, 2 / 9, 5 / 9, 5 / 9, 5 / 9, 1, 1, 1, 1].forEach((expected, cell) =>
    expectClose(ramp.stretched![cell], expected, 1e-6, `ramp[${cell}]`)
  );
});

it('GPURasterStretch nearest palette lookup picks floor(t * size)', async () => {
  // Linear over [0, 8]: t = 0, 0.25, 0.5, 1; four colours: index min(floor(4 t), 3) = 0, 1, 2, 3.
  const palette = Uint32Array.from([11, 22, 33, 44]);
  const result = await runSmallCase([0, 2, 4, 8], 2, {mode: 'linear'}, palette);
  expect(Array.from(result.colors!)).toEqual([11, 22, 33, 44]);
});

// ---------------------------------------------------------------------------------------------
// Reference-library fixture (`test/oracles/raster-stretch-references.py`: OpenCV and NumPy).

it('GPURasterStretch equalization matches cv2.equalizeHist on an 8-bit raster', async () => {
  const {width, values, expectedOpenCV} = stretchReference.equalize;
  // The automatic domain is [0, 255] and 256 bins hold one value each, as in OpenCV.
  const result = await runSmallCase(values, width, {mode: 'equalize'}, undefined, 256);
  expect(result.statistics!.slice(0, 2)).toEqual([0, 255]);
  // OpenCV rounds (cdf - cdfFirst) * 255 / (total - cdfFirst) to 8 bits.
  let largestDistance = 0;
  values.forEach((_, cell) => {
    largestDistance = Math.max(
      largestDistance,
      Math.abs(result.stretched![cell] * 255 - expectedOpenCV[cell])
    );
  });
  expect(largestDistance).toBeLessThanOrEqual(0.5 + 1e-3);
});

it('GPURasterStretch percentile bounds are the bin centre of numpy.percentile(method="lower")', async () => {
  const {width, values, percentiles} = stretchReference.float;
  for (const entry of percentiles) {
    const result = await runSmallCase(values, width, {
      mode: 'percentile',
      percentiles: [entry.percentiles[0], entry.percentiles[1]]
    });
    const [lo, hi] = result.statistics!.slice(2, 4);
    if (entry.percentiles[0] === 0 && entry.percentiles[1] === 100) {
      // Exact extrema with an automatic domain.
      expect([lo, hi]).toEqual(entry.lower);
      continue;
    }
    // The order statistic lies in the reported bin, so it is within half a bin of its centre.
    const halfBin = 0.5 * result.statistics![5] * (1 + 1e-3);
    expect(Math.abs(lo - entry.lower[0]), `lo at ${entry.percentiles[0]}`).toBeLessThanOrEqual(
      halfBin
    );
    expect(Math.abs(hi - entry.lower[1]), `hi at ${entry.percentiles[1]}`).toBeLessThanOrEqual(
      halfBin
    );
  }
});

it('GPURasterStretch linear stretch, gamma and sigmoid match the NumPy evaluation of the rio-color formula', async () => {
  const {width, values, curve} = stretchReference.float;
  const result = await runSmallCase(values, width, {
    mode: 'linear',
    gamma: curve.gamma,
    sigmoidContrast: curve.sigmoidContrast,
    sigmoidMidpoint: curve.sigmoidMidpoint
  });
  curve.expected.forEach((expected, cell) =>
    expectClose(result.stretched![cell], expected, 2e-5, `curve[${cell}]`)
  );
});
