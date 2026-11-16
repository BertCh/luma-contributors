// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  getGPUGTWRParameterValues,
  type GPUGeographicallyTemporallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression';
import {
  GPUMultiscaleGeographicallyTemporallyWeightedRegression,
  getGPUMGTWRParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-temporally-weighted-regression';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Observations = {
  positions: Float32Array;
  times: Float32Array;
  predictors: Float32Array;
  response: Float32Array;
  mask: Uint32Array;
};
type Frame = {
  bandwidth: number;
  timeScale: number;
  positions?: Float32Array;
  times?: Float32Array;
  mask?: Uint32Array;
};
type Configuration = Partial<
  Pick<
    GPUGeographicallyTemporallyWeightedRegressionProps,
    'kernel' | 'gaussianCutoff' | 'searchStructure' | 'adaptive' | 'predictorCount'
  >
> & {viewByteOffset?: number; packedFitOutputs?: boolean};

async function runGPU(data: Observations, frames: Frame[], configuration: Configuration = {}) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {
    viewByteOffset = 0,
    predictorCount = 1,
    packedFitOutputs = false,
    ...regressionConfiguration
  } = configuration;
  const prefixWords = viewByteOffset / 4;
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const rows = data.times.length;
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    values: Float32Array | Uint32Array,
    format: Format,
    components = 1
  ) => {
    const padded =
      prefixWords === 0
        ? values
        : values instanceof Uint32Array
          ? new Uint32Array(prefixWords + values.length)
          : new Float32Array(prefixWords + values.length);
    if (prefixWords > 0) padded.set(values, prefixWords);
    const buffer = createInputBuffer(device, padded);
    buffers.push(buffer);
    const imported = importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      values.length / components
    );
    return {
      buffer,
      view: graph.createDataView(imported.buffer, {
        format,
        length: values.length / components,
        byteOffset: viewByteOffset
      })
    };
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, prefixWords + length);
    buffers.push(buffer);
    const imported = importGraphBuffer(graph, `output-${serial++}`, buffer, format, length);
    return {
      view: graph.createDataView(imported.buffer, {format, length, byteOffset: viewByteOffset}),
      // Keep large readbacks packed; Array.from plus slice duplicates boxed row values.
      read: async () => {
        const bytes = await buffer.readAsync();
        const byteOffset = bytes.byteOffset + viewByteOffset;
        return format === 'uint32'
          ? new Uint32Array(bytes.buffer, byteOffset, length)
          : new Float32Array(bytes.buffer, byteOffset, length);
      }
    };
  };
  const positions = input(data.positions, 'float32x2', 2);
  const times = input(data.times, 'float32');
  const mask = input(data.mask, 'uint32');
  const parameters = input(getGPUGTWRParameterValues(frames[0]), 'float32');
  const coefficients = output(rows * (predictorCount + 1), 'float32');
  const packed = packedFitOutputs
    ? (() => {
        const hatByteOffset = Math.ceil((viewByteOffset + rows * 4) / 256) * 256;
        const buffer = createOutputBuffer(device, hatByteOffset / 4 + rows);
        buffers.push(buffer);
        // One physical import, with two aligned nonoverlapping writable binding ranges.
        const imported = importGraphBuffer(
          graph,
          'packed-fit-and-hat',
          buffer,
          'float32',
          hatByteOffset / 4 + rows
        );
        const range = (byteOffset: number) => ({
          view: graph.createDataView(imported.buffer, {
            format: 'float32' as const,
            length: rows,
            byteOffset
          }),
          read: async () => {
            const bytes = await buffer.readAsync();
            return new Float32Array(bytes.buffer, bytes.byteOffset + byteOffset, rows);
          }
        });
        return {fitted: range(viewByteOffset), hat: range(hatByteOffset)};
      })()
    : undefined;
  const fitted = packed?.fitted ?? output(rows, 'float32');
  const hat = packed?.hat ?? output(rows, 'float32');
  const localStatus = output(rows, 'uint32');
  const residuals = output(rows, 'float32');
  const summary = output(12, 'float32');
  graph.add(
    new GPUGeographicallyTemporallyWeightedRegression({
      positions: positions.view,
      times: times.view,
      mask: mask.view,
      parameters: parameters.view,
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      predictors: input(data.predictors, 'float32').view,
      response: input(data.response, 'float32').view,
      predictorCount,
      kernel: 'bisquare',
      ...regressionConfiguration,
      output: {
        coefficients: coefficients.view,
        fitted: fitted.view,
        hatDiagonal: hat.view,
        localStatus: localStatus.view,
        residuals: residuals.view,
        summary: summary.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    const results = [];
    for (const frame of frames) {
      parameters.buffer.write(getGPUGTWRParameterValues(frame), viewByteOffset);
      if (frame.positions) positions.buffer.write(frame.positions, viewByteOffset);
      if (frame.times) times.buffer.write(frame.times, viewByteOffset);
      if (frame.mask) mask.buffer.write(frame.mask, viewByteOffset);
      submitGraph(device, compiled, undefined);
      results.push({
        coefficients: await coefficients.read(),
        fitted: await fitted.read(),
        hat: await hat.read(),
        localStatus: await localStatus.read(),
        residuals: await residuals.read(),
        summary: await summary.read()
      });
    }
    return results;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

// Independent scalar two-parameter weighted regression formula; no GPU solver or grid code.
function fitOnCPU(data: Observations, frame: Frame, configuration: Configuration = {}) {
  const rows = data.times.length;
  const coefficients = new Float64Array(rows * 2).fill(NaN);
  const fitted = new Float64Array(rows).fill(NaN);
  const hat = new Float64Array(rows).fill(NaN);
  const localStatus = new Uint32Array(rows).fill(1);
  const included = (row: number) =>
    Boolean(data.mask[row]) &&
    [
      data.positions[row * 2],
      data.positions[row * 2 + 1],
      data.times[row],
      data.predictors[row],
      data.response[row]
    ].every(Number.isFinite);
  for (let focal = 0; focal < rows; focal++) {
    if (!included(focal)) {
      localStatus[focal] = 2;
      continue;
    }
    const distances = Array.from({length: rows}, (_, row) => row)
      .filter(included)
      .map(row =>
        Math.hypot(
          data.positions[row * 2] - data.positions[focal * 2],
          data.positions[row * 2 + 1] - data.positions[focal * 2 + 1],
          Math.sqrt(frame.timeScale) * (data.times[row] - data.times[focal])
        )
      )
      .sort((first, second) => first - second);
    const bandwidth = configuration.adaptive
      ? Math.fround(Math.fround(distances[frame.bandwidth - 1]) * Math.fround(1.0000001))
      : frame.bandwidth;
    if (!(bandwidth > 0 && Number.isFinite(bandwidth))) continue;
    let weightSum = 0;
    let weightedPredictor = 0;
    let weightedSquares = 0;
    let weightedResponse = 0;
    let weightedCross = 0;
    for (let row = 0; row < rows; row++) {
      if (!included(row)) continue;
      const distance = Math.hypot(
        data.positions[row * 2] - data.positions[focal * 2],
        data.positions[row * 2 + 1] - data.positions[focal * 2 + 1],
        Math.sqrt(frame.timeScale) * (data.times[row] - data.times[focal])
      );
      const ratio = distance / bandwidth;
      const weight =
        configuration.kernel === 'gaussian'
          ? distance > (configuration.gaussianCutoff ?? Infinity) * bandwidth
            ? 0
            : Math.exp(-0.5 * ratio ** 2)
          : ratio >= 1
            ? 0
            : (1 - ratio ** 2) ** 2;
      const predictor = data.predictors[row] - data.predictors[focal];
      weightSum += weight;
      weightedPredictor += weight * predictor;
      weightedSquares += weight * predictor ** 2;
      weightedResponse += weight * data.response[row];
      weightedCross += weight * predictor * data.response[row];
    }
    const determinant = weightSum * weightedSquares - weightedPredictor ** 2;
    if (!(determinant > 1e-10)) continue;
    const slope = (weightSum * weightedCross - weightedPredictor * weightedResponse) / determinant;
    fitted[focal] =
      (weightedSquares * weightedResponse - weightedPredictor * weightedCross) / determinant;
    coefficients[focal * 2] = fitted[focal] - slope * data.predictors[focal];
    coefficients[focal * 2 + 1] = slope;
    hat[focal] = weightedSquares / determinant;
    localStatus[focal] = 0;
  }
  return {coefficients, fitted, hat, localStatus};
}

function expectValues(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-4) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) expect(actual[index], `row/entry ${index}`).toBeNaN();
    else
      expect(Math.abs(actual[index] - expected[index]), `row/entry ${index}`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(expected[index]))
      );
  }
}

function createSmallData(): Observations {
  const rows = 64;
  const positions = new Float32Array(rows * 2);
  const times = new Float32Array(rows);
  const predictors = new Float32Array(rows);
  const response = new Float32Array(rows);
  const mask = new Uint32Array(rows).fill(1);
  for (let row = 0; row < rows; row++) {
    positions[row * 2] = row % 8;
    positions[row * 2 + 1] = Math.floor(row / 8);
    times[row] = row % 3;
    predictors[row] = (row % 7) - 3;
    response[row] = 3 + 2 * predictors[row] + ((row % 5) - 2) * 0.125;
  }
  mask[7] = 0;
  times[9] = NaN;
  response[13] = NaN;
  positions[126] = 1000; // Isolated, rank-deficient local design.
  return {positions, times, predictors, response, mask};
}

for (const adaptive of [false, true])
  for (const configuration of [
    {kernel: 'bisquare' as const},
    {kernel: 'gaussian' as const, gaussianCutoff: 2}
  ]) {
    it(`compares indexed ${configuration.kernel}, adaptive ${adaptive} with a CPU oracle and the exact route, including masks and singular rows`, async () => {
      const data = createSmallData();
      for (const timeScale of [0, 0.75]) {
        const frame = {bandwidth: adaptive ? 16 : 3, timeScale};
        const [indexed] = await runGPU(data, [frame], {
          ...configuration,
          adaptive,
          searchStructure: 'grid'
        });
        const [dense] = await runGPU(data, [frame], {
          ...configuration,
          adaptive,
          searchStructure: 'brute-force'
        });
        const expected = fitOnCPU(data, frame, {...configuration, adaptive});
        for (const name of ['coefficients', 'fitted', 'hat', 'localStatus'] as const) {
          expectValues(indexed[name], expected[name]);
          expectValues(indexed[name], dense[name]);
        }
        let residualSquares = 0;
        let trace = 0;
        let observations = 0;
        let singular = 0;
        for (let row = 0; row < data.times.length; row++) {
          if (expected.localStatus[row] === 0) {
            residualSquares += (data.response[row] - expected.fitted[row]) ** 2;
            trace += expected.hat[row];
            observations++;
          } else if (expected.localStatus[row] === 1) singular++;
        }
        expectValues(indexed.summary, dense.summary, 1e-3);
        expectValues(
          [indexed.summary[0], indexed.summary[1], indexed.summary[4], indexed.summary[6]],
          [residualSquares, trace, observations, singular],
          1e-3
        );
      }
    });
  }

it('includes the exact Gaussian cutoff boundary and rebuilds the index across parameter and data changes', async () => {
  const data: Observations = {
    positions: new Float32Array([0, 0, 0, 0, 1, 0, 1.0001, 0, 100, 0, 100, 0]),
    times: new Float32Array([0, 0, 0, 0, 1, 1]),
    predictors: new Float32Array([-1, 1, 0, 2, -1, 1]),
    response: new Float32Array([1, 5, 7, 20, 5, 9]),
    mask: new Uint32Array(6).fill(1)
  };
  const movedPositions = data.positions.map(value => value / 100);
  const emptyMask = new Uint32Array(6);
  const frames = [
    {bandwidth: 0.5, timeScale: 1},
    {bandwidth: 0.5, timeScale: 0, positions: movedPositions},
    {bandwidth: 0.5, timeScale: 0, mask: emptyMask}
  ];
  const configuration = {
    kernel: 'gaussian' as const,
    gaussianCutoff: 2,
    searchStructure: 'grid' as const
  };
  const results = await runGPU(data, frames, configuration);
  for (let frameIndex = 0; frameIndex < frames.length; frameIndex++) {
    const expected = fitOnCPU(
      {
        ...data,
        positions: frameIndex ? movedPositions : data.positions,
        mask: frameIndex === 2 ? emptyMask : data.mask
      },
      frames[frameIndex],
      configuration
    );
    for (const name of ['coefficients', 'fitted', 'hat', 'localStatus'] as const)
      expectValues(results[frameIndex][name], expected[name]);
  }
  expect(results[2].summary[4]).toBe(0);
  expect(results[2].summary[11]).toBe(0);
  expect(results[2].summary[2]).toBeNaN();
});

for (const adaptive of [false, true])
  it(`fits beyond the old cap against closed-form coefficients, hats and summaries, adaptive ${adaptive}`, async () => {
    const rows = adaptive ? 40_000 : 1_000_000;
    const positions = new Float32Array(rows * 2);
    const times = new Float32Array(rows);
    const predictors = new Float32Array(rows);
    const response = new Float32Array(rows);
    const mask = new Uint32Array(rows).fill(1);
    for (let row = 0; row < rows; row++) {
      const cluster = Math.floor(row / 4);
      positions[row * 2] = (cluster % 63) * 8;
      positions[row * 2 + 1] = (Math.floor(cluster / 63) % 63) * 8;
      times[row] = Math.floor(cluster / (63 * 63)) * 8;
      predictors[row] = (row % 4) - 1;
      response[row] = 3 + 2 * predictors[row] + (row % 4 === 0 || row % 4 === 3 ? 0.125 : -0.125);
    }
    const [result] = await runGPU(
      {positions, times, predictors, response, mask},
      [{bandwidth: adaptive ? 8 : 1, timeScale: 1}],
      {adaptive}
    );
    let maximumError = 0;
    let invalidRows = 0;
    for (let row = 0; row < rows; row++) {
      maximumError = Math.max(
        maximumError,
        Math.abs(result.coefficients[row * 2] - 3),
        Math.abs(result.coefficients[row * 2 + 1] - 2),
        Math.abs(result.fitted[row] - (3 + 2 * predictors[row])),
        Math.abs(result.hat[row] - (row % 4 === 0 || row % 4 === 3 ? 0.7 : 0.3)),
        Math.abs(result.residuals[row] - (row % 4 === 0 || row % 4 === 3 ? 0.125 : -0.125))
      );
      if (result.localStatus[row] !== 0) invalidRows++;
    }
    expect(maximumError).toBeLessThan(1e-4);
    expect(invalidRows).toBe(0);
    expect(result.summary[0]).toBeCloseTo(rows / 64, 1);
    expect(result.summary[1]).toBeCloseTo(rows / 2, 1);
    expect(result.summary[4]).toBe(rows);
    expect(result.summary[6]).toBe(0);
    expect(result.summary[11]).toBe(rows);
  }, 120_000);

it('keeps adaptive zero-distance ties and invalid neighbor counts singular', async () => {
  const data: Observations = {
    positions: new Float32Array(8),
    times: new Float32Array(4),
    predictors: new Float32Array([-1, 0, 1, 2]),
    response: new Float32Array([1, 3, 5, 7]),
    mask: new Uint32Array(4).fill(1)
  };
  const frames = [1, 4, 5, 2.5, 129].map(bandwidth => ({bandwidth, timeScale: 1}));
  for (const searchStructure of ['grid', 'brute-force'] as const) {
    const results = await runGPU(data, frames, {adaptive: true, searchStructure});
    for (const result of results) {
      expect(Array.from(result.localStatus)).toEqual([1, 1, 1, 1]);
      expect(Array.from(result.fitted).every(Number.isNaN)).toBe(true);
      expect(result.summary[6]).toBe(4);
      expect(result.summary[4]).toBe(0);
    }
  }
});

for (const rows of [10, 36]) {
  it(`preserves zero TSS and NaN R squared for ${rows} coincident constant responses`, async () => {
    const data: Observations = {
      positions: new Float32Array(rows * 2),
      times: new Float32Array(rows),
      predictors: Float32Array.from({length: rows}, (_, row) => row - rows / 2),
      response: new Float32Array(rows).fill(1),
      mask: new Uint32Array(rows).fill(1)
    };
    const frame = {bandwidth: 1, timeScale: 1};
    const [grid] = await runGPU(data, [frame], {searchStructure: 'grid'});
    const [reference] = await runGPU(data, [frame], {searchStructure: 'brute-force'});
    expect(Array.from(grid.localStatus)).toEqual(new Array(rows).fill(0));
    expectValues(grid.localStatus, reference.localStatus);
    expectValues(grid.fitted, new Float32Array(rows).fill(1));
    expect(grid.summary[3]).toBeNaN();
    expect(reference.summary[3]).toBeNaN();
    expect(grid.summary[4]).toBe(rows);
  });
}

it('rejects MGTWR Gaussian cutoff 1 for balanced clusters separated by three bandwidths', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputBuffers = new Map<GraphDataView, Buffer>();
  let compiled: ReturnType<typeof graph.compile> | undefined;
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2'>(
    values: Float32Array,
    format: Format,
    components = 1
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `mgtwr-cutoff-${serial++}`,
      buffer,
      format,
      values.length / components
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    const view = importGraphBuffer(graph, `mgtwr-output-${serial++}`, buffer, format, length);
    outputBuffers.set(view, buffer);
    return view;
  };
  try {
    const settings = {bandwidth: 1, timeScale: 0};
    const requested = {
      id: 'mgtwr-unsupported-cutoff',
      positions: input(
        new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0]),
        'float32x2',
        2
      ),
      times: input(new Float32Array(8), 'float32'),
      predictors: input(new Float32Array([-1, 1, -1, 1, -1, 1, -1, 1]), 'float32'),
      response: input(new Float32Array([0, 2, 0, 2, 99, 101, 99, 101]), 'float32'),
      predictorCount: 1,
      temporalContext: {
        unit: 'second' as const,
        epoch: 0n,
        encoding: 'float32-relative' as const,
        axis: {kind: 'irregular' as const}
      },
      kernel: 'gaussian' as const,
      gaussianCutoff: 1,
      iterationCount: 1,
      parameters: input(
        getGPUMGTWRParameterValues({initial: settings, components: [settings, settings]}),
        'float32'
      ),
      output: {
        coefficients: output(16, 'float32'),
        fitted: output(8, 'float32'),
        residuals: output(8, 'float32'),
        localStatus: output(8, 'uint32'),
        scores: output(1, 'float32')
      }
    };
    expect(() => new GPUMultiscaleGeographicallyTemporallyWeightedRegression(requested)).toThrow(
      'mgtwr-unsupported-cutoff gaussianCutoff is not supported'
    );
    const {gaussianCutoff: unsupportedCutoff, ...supported} = requested;
    expect(unsupportedCutoff).toBe(1);
    graph.add(new GPUMultiscaleGeographicallyTemporallyWeightedRegression(supported));
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const coefficients = await readFloat32(outputBuffers.get(requested.output.coefficients)!, 16);
    const otherClusterWeight = Math.exp(-4.5);
    const firstIntercept = (1 + 100 * otherClusterWeight) / (1 + otherClusterWeight);
    const secondIntercept = (100 + otherClusterWeight) / (1 + otherClusterWeight);
    for (let row = 0; row < 8; row++) {
      expectValues(
        [coefficients[row * 2], coefficients[row * 2 + 1]],
        [row < 4 ? firstIntercept : secondIntercept, 1]
      );
    }
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const rows of [255, 256, 257, 32768]) {
  it(`matches an independent oracle with shuffled offset views, masks and cell-boundary neighbors at ${rows} rows`, async () => {
    const data: Observations = {
      positions: new Float32Array(rows * 2),
      times: new Float32Array(rows),
      predictors: new Float32Array(rows),
      response: new Float32Array(rows),
      mask: new Uint32Array(rows)
    };
    // Only a bounded subset is usable in the 32K reference case, keeping the exact scan cheap.
    // The invertible permutation spreads that subset throughout the caller's shuffled storage.
    for (let row = 0; row < rows; row++) {
      const source = (row * 73 + 13) % rows;
      data.positions[row * 2] = (source % 8) * 0.5;
      data.positions[row * 2 + 1] = Math.floor(source / 8) * 0.5;
      data.times[row] = (source % 3) * 0.25;
      data.predictors[row] = (source % 5) - 2;
      data.response[row] = 3 + 2 * data.predictors[row] + ((source % 3) - 1) * 0.125;
      data.mask[row] = Number(source < 32 && source % 11 !== 0);
    }
    const frame = {bandwidth: 1.4, timeScale: 0.75};
    const expected = fitOnCPU(data, frame);
    const [grid] = await runGPU(data, [frame], {searchStructure: 'grid', viewByteOffset: 260});
    const [reference] = await runGPU(data, [frame], {
      searchStructure: 'brute-force',
      viewByteOffset: 260
    });
    for (const name of ['coefficients', 'fitted', 'hat', 'localStatus'] as const) {
      expectValues(grid[name], expected[name]);
      expectValues(reference[name], expected[name]);
    }
  });
}

for (const timeScale of [1e-30, 1e30])
  for (const requested of [1, 128]) {
    it(`matches adaptive k=${requested} through extreme time scale ${timeScale} and empty shells`, async () => {
      const rows = 160;
      const data: Observations = {
        positions: new Float32Array(rows * 2),
        times: new Float32Array(rows),
        predictors: new Float32Array(rows),
        response: new Float32Array(rows),
        mask: new Uint32Array(rows).fill(1)
      };
      const timeUnit = timeScale > 1 ? 1e-15 : 1e15;
      for (let row = 0; row < rows; row++) {
        data.positions[row * 2] = (row % 8) * 0.125;
        data.positions[row * 2 + 1] = (Math.floor(row / 8) % 8) * 0.125;
        data.times[row] = (row < 128 ? row * 0.01 : 1e4 + row) * timeUnit;
        data.predictors[row] = (row % 11) - 5;
        data.response[row] = 3 + 2 * data.predictors[row] + ((row % 3) - 1) * 0.125;
      }
      const frame = {bandwidth: requested, timeScale};
      const expected = fitOnCPU(data, frame, {adaptive: true});
      for (const searchStructure of ['grid', 'brute-force'] as const) {
        const [actual] = await runGPU(data, [frame], {adaptive: true, searchStructure});
        for (const name of ['coefficients', 'fitted', 'hat', 'localStatus'] as const)
          expectValues(actual[name], expected[name], 5e-4);
      }
    });
  }

it('keeps k=128 singular when only 127 rows are usable', async () => {
  const rows = 160;
  const data: Observations = {
    positions: new Float32Array(rows * 2),
    times: new Float32Array(rows),
    predictors: Float32Array.from({length: rows}, (_, row) => row % 5),
    response: new Float32Array(rows).fill(1),
    mask: Uint32Array.from({length: rows}, (_, row) => Number(row < 127))
  };
  const [actual] = await runGPU(data, [{bandwidth: 128, timeScale: 1}], {adaptive: true});
  expect(Array.from(actual.localStatus)).toEqual(
    Array.from({length: rows}, (_, row) => (row < 127 ? 1 : 2))
  );
  expect(actual.summary[4]).toBe(0);
  expect(actual.summary[6]).toBe(127);
});

it('preserves both clusters when opposite finite extrema overflow their extent', async () => {
  const data: Observations = {
    positions: new Float32Array([
      -3e38, 0, -3e38, 0, -3e38, 0, -3e38, 0, 3e38, 0, 3e38, 0, 3e38, 0, 3e38, 0
    ]),
    times: new Float32Array(8),
    predictors: new Float32Array([-1, 0, 1, 2, -1, 0, 1, 2]),
    response: new Float32Array([1, 3, 5, 7, 1, 3, 5, 7]),
    mask: new Uint32Array(8).fill(1)
  };
  for (const searchStructure of ['grid', 'brute-force'] as const) {
    const [actual] = await runGPU(data, [{bandwidth: 1, timeScale: 1}], {searchStructure});
    expectValues(
      actual.coefficients,
      new Float32Array([3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2])
    );
    expectValues(actual.hat, new Float32Array([0.7, 0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.7]));
    expect(Array.from(actual.localStatus)).toEqual(new Array(8).fill(0));
  }
});

for (const adaptive of [false, true]) {
  it(`fits 40K ${adaptive ? 'adaptive' : 'fixed'} rows at the maximum predictor count against an orthogonal closed form`, async () => {
    // Fixed size above the old 32,768-row limit, independent of device maxima. The separate
    // million-row, one-predictor fixture covers scale without a large coefficient readback.
    const rows = 40_000;
    const predictorCount = 11;
    const data: Observations = {
      positions: new Float32Array(rows * 2),
      times: new Float32Array(rows),
      predictors: new Float32Array(rows * predictorCount),
      response: new Float32Array(rows),
      mask: new Uint32Array(rows).fill(1)
    };
    const walsh = (value: number) => {
      let parity = 0;
      for (let bit = 0; bit < 5; bit++) parity ^= (value >> bit) & 1;
      return parity ? -1 : 1;
    };
    for (let row = 0; row < rows; row++) {
      const cluster = Math.floor(row / 32);
      const within = row % 32;
      data.positions[row * 2] = (cluster % 63) * 8;
      data.positions[row * 2 + 1] = (Math.floor(cluster / 63) % 63) * 8;
      data.times[row] = Math.floor(cluster / (63 * 63)) * 8;
      let fitted = 3;
      for (let column = 0; column < predictorCount; column++) {
        const value = walsh(within & (column + 1));
        data.predictors[row * predictorCount + column] = value;
        fitted += (value * (column + 1)) / 8;
      }
      data.response[row] = fitted + 0.125 * walsh(within & 31);
    }
    const [actual] = await runGPU(data, [{bandwidth: adaptive ? 64 : 1, timeScale: 1}], {
      adaptive,
      predictorCount
    });
    let maximumError = 0;
    let invalidRows = 0;
    for (let row = 0; row < rows; row++) {
      let expectedFitted = 3;
      maximumError = Math.max(
        maximumError,
        Math.abs(actual.coefficients[row * 12] - 3),
        Math.abs(actual.hat[row] - 12 / 32)
      );
      for (let column = 0; column < predictorCount; column++) {
        maximumError = Math.max(
          maximumError,
          Math.abs(actual.coefficients[row * 12 + column + 1] - (column + 1) / 8)
        );
        expectedFitted += (data.predictors[row * predictorCount + column] * (column + 1)) / 8;
      }
      maximumError = Math.max(maximumError, Math.abs(actual.fitted[row] - expectedFitted));
      if (actual.localStatus[row] !== 0) invalidRows++;
    }
    expect(maximumError).toBeLessThan(2e-4);
    expect(invalidRows).toBe(0);
    expect(actual.summary[0]).toBeCloseTo(rows / 64, 0);
    expect(actual.summary[1]).toBeCloseTo((rows * 12) / 32, 0);
    expect(actual.summary[4]).toBe(rows);
  }, 120_000);
}

for (const searchStructure of ['grid', 'brute-force'] as const) {
  it(`accepts singly imported packed writable GTWR outputs against the CPU oracle, ${searchStructure}`, async () => {
    const data = createSmallData();
    const frame = {bandwidth: 3, timeScale: 0.75};
    const [actual] = await runGPU(data, [frame], {
      searchStructure,
      packedFitOutputs: true,
      viewByteOffset: 4
    });
    const expected = fitOnCPU(data, frame);
    expectValues(actual.coefficients, expected.coefficients);
    expectValues(actual.fitted, expected.fitted);
    expectValues(actual.hat, expected.hat);
    expect(Array.from(actual.localStatus)).toEqual(Array.from(expected.localStatus));
  });
}

it('accepts singly imported packed MGTWR fitted values and residuals against an affine closed form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let compiled: ReturnType<typeof graph.compile> | undefined;
  let serial = 0;
  const rows = 8;
  const input = <Format extends 'float32' | 'float32x2'>(
    values: Float32Array,
    format: Format,
    components = 1
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `packed-mgtwr-input-${serial++}`,
      buffer,
      format,
      values.length / components
    );
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `packed-mgtwr-output-${serial++}`, buffer, format, length)
    };
  };
  try {
    const predictors = Float32Array.from({length: rows}, (_, row) => (row % 2 ? 1 : -1));
    const response = Float32Array.from(predictors, value => 3 + 2 * value);
    const packed = output(64 + rows, 'float32');
    const fitted = graph.createDataView(packed.view.buffer, {format: 'float32', length: rows});
    const residuals = graph.createDataView(packed.view.buffer, {
      format: 'float32',
      length: rows,
      byteOffset: 256
    });
    const coefficients = output(rows * 2, 'float32');
    const localStatus = output(rows, 'uint32');
    const scores = output(1, 'float32');
    const settings = {bandwidth: 1, timeScale: 0};
    graph.add(
      new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
        positions: input(new Float32Array(rows * 2), 'float32x2', 2),
        times: input(new Float32Array(rows), 'float32'),
        temporalContext: {
          unit: 'second',
          epoch: 0n,
          encoding: 'float32-relative',
          axis: {kind: 'irregular'}
        },
        predictors: input(predictors, 'float32'),
        response: input(response, 'float32'),
        predictorCount: 1,
        parameters: input(
          getGPUMGTWRParameterValues({initial: settings, components: [settings, settings]}),
          'float32'
        ),
        kernel: 'bisquare',
        iterationCount: 1,
        output: {
          coefficients: coefficients.view,
          fitted,
          residuals,
          localStatus: localStatus.view,
          scores: scores.view
        }
      })
    );
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const values = await readFloat32(packed.buffer);
    expectValues(values.slice(0, rows), response);
    expectValues(values.slice(64, 64 + rows), new Float32Array(rows));
    expectValues(
      await readFloat32(coefficients.buffer),
      Float32Array.from({length: rows * 2}, (_, index) => (index % 2 ? 2 : 3))
    );
    expect(await readUint32(localStatus.buffer)).toEqual(new Array(rows).fill(0));
    expectValues(await readFloat32(scores.buffer), [0]);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
