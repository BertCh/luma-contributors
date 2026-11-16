// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity,
  type GPUKernelDensityEdgeCorrection
} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {
  createDensityWeightSumNodes,
  needsDensitySplitScratch
} from '../../../src/gpu-spatial-analysis/point-density/density-scratch';
import {withDensityTestBindingLimit} from './density-test-device';
import {getGPUKernelDensityBinnedErrorBound} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {computeGridDensity, computeHexagonDensity} from './point-density-oracle';
import {
  getExactDensityModel,
  getBinnedDensityModel,
  getPolygonBinnedDensityOracle
} from './kernel-density-binned-oracle';
import {getBandwidthOracle} from './kernel-density-bandwidth-oracle';
import {getSharedOutputDensityFixture} from './shared-output-density-fixture';
import {sumFixedPointOracle} from '../../utils/fixed-point-oracle';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import type {OracleStudyWindow} from '../../analysis-core/study-window-oracle';
import {getPolygonKernelDensityWeights} from './point-density-oracle';

type Fixture = {positions: Float32Array; weights: Float32Array};
type RunOptions = {
  kind: 'grid' | 'hexagon' | 'kde';
  chunkLengths?: number[];
  weightChunkLengths?: number[];
  weighted?: boolean;
  mask?: Uint32Array;
  bandwidthRule?: 'scott' | 'silverman';
  inputOffsets?: boolean;
  window?: OracleStudyWindow;
  updates?: readonly {weights: Float32Array; bandwidth: number; positions?: Float32Array}[];
  probability?: boolean;
  edgeCorrection?: GPUKernelDensityEdgeCorrection;
  scratchBindingByteLength?: number;
  scratchTileSize?: number;
  expectedScratchRoute?: 'split' | 'tiled' | 'chunked-grid';
};

function createFixture(rowCount: number): Fixture {
  const positions = new Float32Array(rowCount * 2);
  const weights = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    positions[row * 2] = (((row * 17) % 127) + 0.25) / 16;
    positions[row * 2 + 1] = (((row * 31) % 61) + 0.25) / 16;
    weights[row] = ((row % 13) - 3) / 8;
  }
  return {positions, weights};
}

function getFloatBits(values: number[]): number[] {
  return Array.from(new Uint32Array(Float32Array.from(values).buffer));
}

function getFloats(bits: number[]): Float32Array {
  return new Float32Array(Uint32Array.from(bits).buffer);
}

async function runDensity(device: Device, fixture: Fixture, options: RunOptions) {
  const rowCount = fixture.weights.length;
  const graph = new GPUCommandGraph(device);
  const addDensity = (createDensity: () => GPUCommandNodeProducer) =>
    withDensityTestBindingLimit(device, options.scratchBindingByteLength, () => {
      const density = createDensity();
      if (options.expectedScratchRoute === 'split') {
        expect(density).toBeInstanceOf(GPUKernelDensity);
        expect(needsDensitySplitScratch(graph, rowCount)).toBe(true);
      } else if (options.expectedScratchRoute) {
        expect(density).toBeInstanceOf(GPUPointDensity);
        expect((density as GPUPointDensity).usesTiledScratch).toBe(
          options.expectedScratchRoute === 'tiled'
        );
      }
      graph.add(density);
    });
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const weightUploads: {buffer: Buffer; rowStart: number; length: number; prefixWords: number}[] =
    [];
  const positionUploads: typeof weightUploads = [];
  let parameterBuffer: Buffer | undefined;
  const importRows = <Format extends 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    values: Float32Array,
    lengths?: number[]
  ): GraphDataView<Format> | GraphVectorView<Format> => {
    const components = format === 'float32x2' ? 2 : 1;
    let rowStart = 0;
    const chunks = (lengths ?? [rowCount]).map((length, chunkIndex) => {
      const prefixWords = options.inputOffsets ? chunkIndex + 1 : 0;
      const chunk = new Float32Array(prefixWords + Math.max(components, length * components));
      chunk.fill(-999999);
      chunk.set(
        values.subarray(rowStart * components, (rowStart + length) * components),
        prefixWords
      );
      const buffer = track(createInputBuffer(device, chunk));
      if (name === 'weights') {
        weightUploads.push({buffer, rowStart, length, prefixWords});
      } else {
        positionUploads.push({buffer, rowStart, length, prefixWords});
      }
      rowStart += length;
      const whole = importGraphBuffer(
        graph,
        `${name}-${chunkIndex}`,
        buffer,
        format,
        Math.floor(chunk.length / components)
      );
      return graph.createDataView(whole.buffer, {format, length, byteOffset: prefixWords * 4});
    });
    expect(rowStart).toBe(rowCount);
    return lengths ? createVectorView(name, format, chunks) : chunks[0];
  };
  const cellCount = options.kind === 'kde' ? 32 : 192;
  const outputBuffer = track(createOutputBuffer(device, cellCount));
  const countsBuffer = track(createOutputBuffer(device, cellCount));
  const sumsBuffer = track(createOutputBuffer(device, cellCount));
  const meansBuffer = track(createOutputBuffer(device, cellCount));
  const weightSumBuffer = track(createOutputBuffer(device, 1));
  const bandwidthBuffer = track(createOutputBuffer(device, 1));
  const invalidCountBuffer = track(createOutputBuffer(device, 1));
  const positions = importRows('positions', 'float32x2', fixture.positions, options.chunkLengths);
  const weights = options.weighted
    ? importRows(
        'weights',
        'float32',
        fixture.weights,
        options.weightChunkLengths ?? options.chunkLengths
      )
    : undefined;
  const values = importGraphBuffer(graph, 'values', outputBuffer, 'float32', cellCount);
  const mask = options.mask
    ? importGraphBuffer(
        graph,
        'mask',
        track(createInputBuffer(device, options.mask)),
        'uint32',
        rowCount
      )
    : undefined;
  if (options.kind === 'kde') {
    const grid = makeExtentGrid(8, 4, [0, 0, 8, 4]);
    const upload = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      data: Float32Array | Uint32Array
    ) => importGraphBuffer(graph, name, track(createInputBuffer(device, data)), format);
    const studyWindow = options.window
      ? new GPUStudyWindowPreparation({
          positions: upload(
            'window-positions',
            'float32x2',
            Float32Array.from(options.window.positions)
          ),
          featureOffsets: upload(
            'window-features',
            'uint32',
            options.window.featureOffsets ??
              new Uint32Array([0, options.window.polygonOffsets.length - 1])
          ),
          polygonOffsets: upload('window-polygons', 'uint32', options.window.polygonOffsets),
          ringOffsets: upload('window-rings', 'uint32', options.window.ringOffsets),
          samplingGrid: [8, 8]
        })
      : undefined;
    if (studyWindow) {
      graph.add(studyWindow);
    }
    parameterBuffer = track(
      createInputBuffer(
        device,
        getGPUKernelDensityParameterValues(
          {bandwidth: 3, window: options.window ? 'study-window' : undefined},
          grid
        )
      )
    );
    const parameters = importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 9);
    addDensity(
      () =>
        new GPUKernelDensity({
          positions,
          weights,
          parameters,
          studyWindow,
          grid,
          indexGridSize: [4, 2],
          indexBounds: [0, 0, 8, 4],
          method: 'binned',
          binnedPadding: 12,
          bandwidthRule: options.bandwidthRule,
          edgeCorrection: options.edgeCorrection ?? 'none',
          normalization: options.probability ? 'probability' : 'intensity',
          output: {
            values,
            weightSum: importGraphBuffer(graph, 'weight-sum', weightSumBuffer, 'float32', 1),
            bandwidth: options.bandwidthRule
              ? importGraphBuffer(graph, 'bandwidth', bandwidthBuffer, 'float32', 1)
              : undefined,
            status: {
              invalidCount: importGraphBuffer(
                graph,
                'invalid-count',
                invalidCountBuffer,
                'uint32',
                1
              )
            }
          }
        })
    );
  } else {
    const binning = options.kind;
    addDensity(
      () =>
        new GPUPointDensity({
          scratchTileSize: options.scratchTileSize,
          positions,
          weights,
          mask,
          binning,
          hexagonRadius: options.kind === 'hexagon' ? 0.5 : undefined,
          bounds: [0, 0, 8, 4],
          gridSize: [16, 12],
          statistic: options.weighted ? 'sum' : 'count',
          output: {
            values,
            counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', cellCount),
            sums: options.weighted
              ? importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', cellCount)
              : undefined,
            means: options.weighted
              ? importGraphBuffer(graph, 'means', meansBuffer, 'float32', cellCount)
              : undefined
          }
        })
    );
  }
  const compiled = graph.compile();
  const readResult = async () => ({
    values: await readUint32(outputBuffer, cellCount),
    counts: options.kind === 'kde' ? [] : await readUint32(countsBuffer, cellCount),
    sums: options.weighted && options.kind !== 'kde' ? await readUint32(sumsBuffer, cellCount) : [],
    means:
      options.weighted && options.kind !== 'kde' ? await readUint32(meansBuffer, cellCount) : [],
    weightSum: options.kind === 'kde' ? await readUint32(weightSumBuffer, 1) : [],
    invalidCount: options.kind === 'kde' ? await readUint32(invalidCountBuffer, 1) : [],
    bandwidth: options.bandwidthRule ? await readUint32(bandwidthBuffer, 1) : []
  });
  try {
    submitGraph(device, compiled, undefined);
    const initial = await readResult();
    const updates: Awaited<ReturnType<typeof readResult>>[] = [];
    for (const update of options.updates ?? []) {
      if (update.positions) {
        for (const upload of positionUploads) {
          const values = new Float32Array(upload.prefixWords + Math.max(upload.length * 2, 2));
          values.fill(-999999);
          values.set(
            update.positions.subarray(upload.rowStart * 2, (upload.rowStart + upload.length) * 2),
            upload.prefixWords
          );
          upload.buffer.write(values);
        }
      }
      for (const upload of weightUploads) {
        const values = new Float32Array(upload.prefixWords + Math.max(upload.length, 1));
        values.fill(-999999);
        values.set(
          update.weights.subarray(upload.rowStart, upload.rowStart + upload.length),
          upload.prefixWords
        );
        upload.buffer.write(values);
      }
      parameterBuffer?.write(
        getGPUKernelDensityParameterValues(
          {bandwidth: update.bandwidth, window: options.window ? 'study-window' : undefined},
          makeExtentGrid(8, 4, [0, 0, 8, 4])
        )
      );
      submitGraph(device, compiled, undefined);
      updates.push(await readResult());
    }
    return {...initial, updates};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

for (const kind of ['grid', 'hexagon'] as const) {
  it(`point density ${kind}: packed and three uneven chunks are bit-identical and match CPU`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(1019);
    fixture.positions[10] = Number.NaN;
    fixture.weights[8] = Number.NaN;
    const packed = await runDensity(device, fixture, {kind, weighted: true});
    const chunked = await runDensity(device, fixture, {
      kind,
      weighted: true,
      chunkLengths: [7, 264, 748],
      weightChunkLengths: [258, 3, 758]
    });
    expect(chunked).toEqual(packed);
    const computeDensity = kind === 'grid' ? computeGridDensity : computeHexagonDensity;
    const oracle = computeDensity(
      Array.from(fixture.positions),
      Array.from(fixture.weights),
      [0, 0, 8, 4],
      [16, 12],
      0.5
    );
    expect(chunked.counts).toEqual(oracle.counts);
    expect(chunked.sums).toEqual(getFloatBits(oracle.sums));
    expect(chunked.values).toEqual(chunked.sums);
  });
}

it('point density: packed mask rows stay aligned with uneven position chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(1019);
  const mask = Uint32Array.from({length: 1019}, (_, row) => row % 3);
  for (const kind of ['grid', 'hexagon'] as const) {
    const packed = await runDensity(device, fixture, {kind, weighted: true, mask});
    expect(
      await runDensity(device, fixture, {kind, weighted: true, mask, chunkLengths: [7, 264, 748]})
    ).toEqual(packed);
    const accepted = Array.from(fixture.positions, (value, index) =>
      mask[Math.floor(index / 2)] ? value : Number.NaN
    );
    const compute = kind === 'grid' ? computeGridDensity : computeHexagonDensity;
    const expected = compute(accepted, Array.from(fixture.weights), [0, 0, 8, 4], [16, 12], 0.5);
    expect(packed.counts).toEqual(expected.counts);
    expect(packed.sums).toEqual(getFloatBits(expected.sums));
  }
});

it('split KDE recomputes Diggle probability density, bandwidth and status after input rewrites', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(5019);
  fixture.weights.fill(0.5);
  fixture.positions[18] = Number.NaN;
  fixture.weights[23] = -1;
  const changedWeights = Float32Array.from(fixture.weights, (_, row) =>
    row % 5 === 0 ? 0.25 : 0.75
  );
  changedWeights[17] = Number.NaN;
  const changedPositions = Float32Array.from(fixture.positions);
  changedPositions[18] = 2;
  changedPositions[42] = Number.POSITIVE_INFINITY;
  const result = await runDensity(device, fixture, {
    kind: 'kde',
    weighted: true,
    bandwidthRule: 'scott',
    probability: true,
    edgeCorrection: 'diggle',
    window: POLYGON_WINDOW,
    chunkLengths: [1000, 1000, 0, 1000, 1000, 1019],
    inputOffsets: true,
    scratchBindingByteLength: 16384,
    expectedScratchRoute: 'split',
    updates: [{weights: changedWeights, positions: changedPositions, bandwidth: 2}]
  });
  for (const [actual, input] of [
    [result, fixture],
    [result.updates[0], {positions: changedPositions, weights: changedWeights}]
  ] as const) {
    const bandwidth = getBandwidthOracle(input.positions, input.weights).isotropic;
    const oracleInput = {
      points: Array.from(input.positions),
      weights: Array.from(input.weights),
      width: 8,
      height: 4,
      start: [0, 0],
      cellSize: [1, 1],
      bandwidth,
      padding: 12,
      window: POLYGON_WINDOW,
      kernel: 'gaussian',
      edgeCorrection: 'diggle',
      normalization: 'probability'
    } as const;
    const expected = getPolygonBinnedDensityOracle(oracleInput);
    const expectedWeights = getPolygonKernelDensityWeights(oracleInput);
    expect(getFloats(actual.bandwidth)[0]).toBeCloseTo(bandwidth, 5);
    expect(getFloats(actual.weightSum)[0]).toBe(expectedWeights.weightSum);
    expect(actual.invalidCount).toEqual([expectedWeights.invalidCount]);
    for (const [cell, value] of getFloats(actual.values).entries()) {
      if (Number.isNaN(expected[cell])) expect(value).toBeNaN();
      else
        expect(Math.abs(value - expected[cell])).toBeLessThan(
          5e-4 * Math.max(1e-3, expected[cell])
        );
    }
  }
});

for (const weighted of [false, true]) {
  it(`binned KDE: packed and three uneven chunks are bit-identical (${weighted ? 'weighted' : 'unweighted'})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(1019);
    const packed = await runDensity(device, fixture, {kind: 'kde', weighted});
    const chunked = await runDensity(device, fixture, {
      kind: 'kde',
      weighted,
      chunkLengths: [7, 264, 748]
    });
    expect(chunked).toEqual(packed);
    const weights = weighted
      ? Array.from(fixture.weights, weight => Math.max(0, weight))
      : undefined;
    const total = weights?.reduce((sum, weight) => sum + weight, 0) ?? 1019;
    const exact = getExactDensityModel({
      points: Array.from(fixture.positions),
      weights,
      width: 8,
      height: 4,
      start: [0, 0],
      cellSize: [1, 1],
      bandwidth: 3
    });
    const bound = (getGPUKernelDensityBinnedErrorBound(1 / 3) * total) / (2 * Math.PI * 9);
    for (const [cell, value] of getFloats(chunked.values).entries()) {
      expect(Math.abs(value - exact[cell])).toBeLessThanOrEqual(bound);
    }
    expect(getFloats(chunked.weightSum)[0]).toBe(total);
  });
}

it('binned KDE: chunk boundaries preserve bandwidth rules, exclusions, Diggle weights and probability', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(1019);
  fixture.positions[14] = Number.NaN;
  fixture.weights[271] = Number.NaN;
  for (const bandwidthRule of [undefined, 'scott'] as const) {
    const options = {
      kind: 'kde',
      weighted: true,
      probability: true,
      bandwidthRule,
      edgeCorrection: 'diggle'
    } as const;
    const packed = await runDensity(device, fixture, options);
    expect(await runDensity(device, fixture, {...options, chunkLengths: [7, 264, 748]})).toEqual(
      packed
    );
    expect(getFloats(packed.weightSum)[0]).toBeGreaterThan(0);
    expect(packed.invalidCount[0]).toBeGreaterThan(2);
    expect(Array.from(getFloats(packed.values)).every(Number.isFinite)).toBe(true);
  }
});

it('point density: three chunks accumulate three million rows into one grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 3_000_019;
  const fixture = createFixture(rowCount);
  const packed = await runDensity(device, fixture, {kind: 'grid'});
  const chunked = await runDensity(device, fixture, {
    kind: 'grid',
    chunkLengths: [17, 1_000_003, rowCount - 1_000_020]
  });
  expect(chunked).toEqual(packed);
  expect(chunked.counts.reduce((sum, count) => sum + count, 0)).toBe(rowCount);
  expect(chunked.values).toEqual(getFloatBits(chunked.counts));
});

const BOUNDARY_CHUNKS = [63, 0, 64, 65, 255, 0, 256, 257];

for (const kind of ['grid', 'hexagon', 'kde'] as const) {
  it(`${kind}: workgroup boundaries, empty chunks and nonzero view offsets match independent values`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(960);
    const packed = await runDensity(device, fixture, {kind, weighted: true});
    const chunked = await runDensity(device, fixture, {
      kind,
      weighted: true,
      chunkLengths: BOUNDARY_CHUNKS,
      inputOffsets: true
    });
    expect(chunked).toEqual(packed);
    if (kind === 'kde') {
      const expected = getBinnedDensityModel({
        points: Array.from(fixture.positions),
        weights: Array.from(fixture.weights, value => Math.max(0, value)),
        width: 8,
        height: 4,
        start: [0, 0],
        cellSize: [1, 1],
        bandwidth: 3,
        padding: 12
      });
      for (const [cell, value] of getFloats(chunked.values).entries()) {
        expect(Math.abs(value - expected[cell])).toBeLessThan(2e-4 * Math.max(1, expected[cell]));
      }
    } else {
      const compute = kind === 'grid' ? computeGridDensity : computeHexagonDensity;
      const oracle = compute(
        Array.from(fixture.positions),
        Array.from(fixture.weights),
        [0, 0, 8, 4],
        [16, 12],
        0.5
      );
      expect(chunked.counts).toEqual(oracle.counts);
      expect(chunked.sums).toEqual(getFloatBits(oracle.sums));
    }
  });
}

it('point-density hotspots preserve signed cancellation across wide weight magnitudes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(960);
  for (let row = 0; row < 960; row++) {
    fixture.positions[row * 2] = 0.25;
    fixture.positions[row * 2 + 1] = 0.25;
    fixture.weights[row] = [2 ** 60, 1, -(2 ** 60), 2 ** -30][row % 4];
  }
  const expected = sumFixedPointOracle(fixture.weights, new Uint32Array(960), 192).sums;
  for (const kind of ['grid', 'hexagon'] as const) {
    const packed = await runDensity(device, fixture, {kind, weighted: true});
    const chunked = await runDensity(device, fixture, {
      kind,
      weighted: true,
      chunkLengths: BOUNDARY_CHUNKS,
      inputOffsets: true
    });
    expect(chunked.sums).toEqual(getFloatBits(expected));
    expect(chunked).toEqual(packed);
    expect(chunked.counts).toEqual([960, ...new Array(191).fill(0)]);
  }
});

it('one compiled density graph clears all accumulators when every chunk is rewritten and then zeroed', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(960);
  const changed = Float32Array.from(fixture.weights, (_, row) => ((row * 7) % 19) / 8);
  for (const kind of ['grid', 'hexagon', 'kde'] as const) {
    const result = await runDensity(device, fixture, {
      kind,
      weighted: true,
      chunkLengths: BOUNDARY_CHUNKS,
      inputOffsets: true,
      updates: [
        {weights: changed, bandwidth: 2.25},
        {weights: new Float32Array(960), bandwidth: 4}
      ]
    });
    const afterChange = result.updates[0];
    const afterZero = result.updates[1];
    if (kind === 'kde') {
      const expected = getBinnedDensityModel({
        points: Array.from(fixture.positions),
        weights: Array.from(changed),
        width: 8,
        height: 4,
        start: [0, 0],
        cellSize: [1, 1],
        bandwidth: 2.25,
        padding: 12
      });
      for (const [cell, value] of getFloats(afterChange.values).entries()) {
        expect(Math.abs(value - expected[cell])).toBeLessThan(2e-4 * Math.max(1, expected[cell]));
      }
      expect(getFloats(afterChange.weightSum)[0]).toBe(
        changed.reduce((sum, value) => sum + value, 0)
      );
      expect(getFloats(afterZero.weightSum)[0]).toBe(0);
    } else {
      const compute = kind === 'grid' ? computeGridDensity : computeHexagonDensity;
      const expected = compute(
        Array.from(fixture.positions),
        Array.from(changed),
        [0, 0, 8, 4],
        [16, 12],
        0.5
      );
      expect(afterChange.sums).toEqual(getFloatBits(expected.sums));
      expect(afterZero.sums).toEqual(new Array(192).fill(0));
      expect(afterZero.counts.reduce((sum, count) => sum + count, 0)).toBe(960);
    }
    expect(afterZero.values).toEqual(new Array(kind === 'kde' ? 32 : 192).fill(0));
  }
});

const RECTANGLE_WINDOW: OracleStudyWindow = {
  positions: new Float64Array([0, 0, 8, 0, 8, 4, 0, 4]),
  polygonOffsets: new Uint32Array([0, 1]),
  ringOffsets: new Uint32Array([0, 4])
};
const POLYGON_WINDOW: OracleStudyWindow = {
  positions: new Float64Array([0, 0, 8, 0, 8, 2, 4, 2, 4, 4, 0, 4]),
  polygonOffsets: new Uint32Array([0, 1]),
  ringOffsets: new Uint32Array([0, 6])
};

for (const bandwidthRule of ['scott', 'silverman'] as const) {
  for (const polygon of [false, true]) {
    it(`chunked ${bandwidthRule} bandwidth, Diggle and probability match numerical oracles (${polygon ? 'polygon' : 'rectangle'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createFixture(960);
      fixture.positions[126] = Number.NaN;
      fixture.weights[256] = Number.NaN;
      const bandwidth = getBandwidthOracle(fixture.positions, fixture.weights).isotropic;
      const window = polygon ? POLYGON_WINDOW : RECTANGLE_WINDOW;
      const input = {
        points: Array.from(fixture.positions),
        weights: Array.from(fixture.weights),
        width: 8,
        height: 4,
        start: [0, 0] as const,
        cellSize: [1, 1] as const,
        bandwidth,
        padding: 12,
        window,
        kernel: 'gaussian',
        edgeCorrection: 'diggle',
        normalization: 'probability'
      } as const;
      const expected = getPolygonBinnedDensityOracle(input);
      const expectedWeights = getPolygonKernelDensityWeights(input);
      const options = {
        kind: 'kde',
        weighted: true,
        bandwidthRule,
        probability: true,
        edgeCorrection: 'diggle',
        window: polygon ? window : undefined
      } as const;
      const packed = await runDensity(device, fixture, options);
      const chunked = await runDensity(device, fixture, {
        ...options,
        chunkLengths: BOUNDARY_CHUNKS,
        inputOffsets: true
      });
      expect(chunked).toEqual(packed);
      expect(Math.abs(getFloats(chunked.bandwidth)[0] - bandwidth)).toBeLessThan(1e-5 * bandwidth);
      expect(getFloats(chunked.weightSum)[0]).toBe(expectedWeights.weightSum);
      expect(chunked.invalidCount[0]).toBe(expectedWeights.invalidCount);
      for (const [cell, value] of getFloats(chunked.values).entries()) {
        if (Number.isNaN(expected[cell])) {
          expect(value).toBeNaN();
        } else {
          expect(Math.abs(value - expected[cell])).toBeLessThan(
            5e-4 * Math.max(1e-3, expected[cell])
          );
        }
      }
    });
  }
}

it('packed point density writes independent expected values to aligned output fields in one buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positionsBuffer = createInputBuffer(
    device,
    new Float32Array([0.25, 0.5, 0.75, 0.5, 1.25, 0.5, 1.75, 0.5])
  );
  const weightsBuffer = createInputBuffer(device, new Float32Array([1, 3, 2, 4]));
  const outputBuffer = createOutputBuffer(device, 512);
  const sentinel = 0xdeadbeef;
  outputBuffer.write(new Uint32Array(512).fill(sentinel));
  const graph = new GPUCommandGraph(device);
  const wholeOutput = importGraphBuffer(graph, 'shared-output', outputBuffer, 'float32', 512);
  graph.add(
    new GPUPointDensity({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 4),
      weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 4),
      bounds: [0, 0, 2, 1],
      gridSize: [2, 1],
      statistic: 'sum',
      output: {
        values: graph.createDataView(wholeOutput.buffer, {format: 'float32', length: 2}),
        counts: graph.createDataView(wholeOutput.buffer, {
          format: 'uint32',
          length: 2,
          byteOffset: 256
        }),
        sums: graph.createDataView(wholeOutput.buffer, {
          format: 'float32',
          length: 2,
          byteOffset: 512
        }),
        means: graph.createDataView(wholeOutput.buffer, {
          format: 'float32',
          length: 2,
          byteOffset: 768
        }),
        extent: graph.createDataView(wholeOutput.buffer, {
          format: 'float32',
          length: 2,
          byteOffset: 1024
        }),
        histogram: graph.createDataView(wholeOutput.buffer, {
          format: 'uint32',
          length: 4,
          byteOffset: 1280
        }),
        status: {
          overflow: graph.createDataView(wholeOutput.buffer, {
            format: 'uint32',
            length: 2,
            byteOffset: 1536
          })
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const bits = await readUint32(outputBuffer, 512);
    expect(bits.slice(0, 2)).toEqual(getFloatBits([4, 6]));
    expect(bits.slice(64, 66)).toEqual([2, 2]);
    expect(bits.slice(128, 130)).toEqual(getFloatBits([4, 6]));
    expect(bits.slice(192, 194)).toEqual(getFloatBits([2, 3]));
    expect(bits.slice(256, 258)).toEqual(getFloatBits([4, 6]));
    expect(bits.slice(320, 324)).toEqual([1, 0, 0, 1]);
    expect(bits.slice(384, 386)).toEqual([0, sentinel]);
    expect(bits[2]).toBe(sentinel);
    // Reuse one compiled graph; every staged field must be republished on each encoding.
    weightsBuffer.write(new Float32Array([2, 4, 6, 8]));
    submitGraph(device, compiled, undefined);
    const changed = await readUint32(outputBuffer, 512);
    expect(changed.slice(0, 2)).toEqual(getFloatBits([6, 14]));
    expect(changed.slice(128, 130)).toEqual(getFloatBits([6, 14]));
    expect(changed.slice(192, 194)).toEqual(getFloatBits([3, 7]));
    expect(changed.slice(256, 258)).toEqual(getFloatBits([6, 14]));
    weightsBuffer.write(new Float32Array(4));
    submitGraph(device, compiled, undefined);
    const zeroed = await readUint32(outputBuffer, 512);
    expect(zeroed.slice(0, 2)).toEqual([0, 0]);
    expect(zeroed.slice(64, 66)).toEqual([2, 2]);
    expect(zeroed.slice(128, 130)).toEqual([0, 0]);
    expect(zeroed.slice(192, 194)).toEqual([0, 0]);
    expect(zeroed.slice(256, 258)).toEqual([0, 0]);
    expect(zeroed.slice(384, 386)).toEqual([0, sentinel]);
  } finally {
    compiled.destroy();
    positionsBuffer.destroy();
    weightsBuffer.destroy();
    outputBuffer.destroy();
  }
});

it.each([
  undefined,
  2
])('density skips end-of-buffer empty chunks and publishes adjacent columns (tile %s)', async scratchTileSize => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positionsBuffer = createInputBuffer(
    device,
    Float32Array.of(0.25, 0.5, 0.75, 0.5, 1.25, 0.5)
  );
  const outputBuffer = createOutputBuffer(device, 5);
  outputBuffer.write(Uint32Array.of(0, 0, 0, 0, 0xdeadbeef));
  const graph = new GPUCommandGraph(device);
  const wholeOutput = importGraphBuffer(graph, 'adjacent-output', outputBuffer, 'uint32', 5);
  const wholePositions = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 3);
  let rowOffset = 0;
  const positions = createVectorView(
    'position-chunks',
    'float32x2',
    [0, 1, 2, 0].map(length => {
      const chunk = graph.createDataView(wholePositions.buffer, {
        format: 'float32x2',
        length,
        byteOffset: rowOffset * 8
      });
      rowOffset += length;
      return chunk;
    })
  );
  const density = new GPUPointDensity({
    positions,
    scratchTileSize,
    bounds: [0, 0, 2, 1],
    gridSize: [2, 1],
    output: {
      values: graph.createDataView(wholeOutput.buffer, {format: 'float32', length: 2}),
      counts: graph.createDataView(wholeOutput.buffer, {
        format: 'uint32',
        length: 2,
        byteOffset: 8
      })
    }
  });
  expect(density.usesTiledScratch).toBe(scratchTileSize !== undefined);
  graph.add(density);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputBuffer, 5)).toEqual([...getFloatBits([2, 1]), 2, 1, 0xdeadbeef]);
  } finally {
    compiled.destroy();
    positionsBuffer.destroy();
    outputBuffer.destroy();
  }
});

for (const method of ['exact', 'auto', 'binned'] as const) {
  for (const edgeCorrection of [undefined, 'none'] as const) {
    it(`${method} KDE publishes shared-buffer values, bandwidth and diagnostics against numerical expectations (${edgeCorrection ?? 'default uniform'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = getSharedOutputDensityFixture(method, edgeCorrection);
      const {points, weights, grid, bandwidth, expectedValues} = fixture;
      const positionsBuffer = createInputBuffer(device, Float32Array.from(points));
      const weightsBuffer = createInputBuffer(device, Float32Array.from(weights));
      const parametersBuffer = createInputBuffer(
        device,
        getGPUKernelDensityParameterValues({bandwidth: 1}, grid)
      );
      const outputBuffer = createOutputBuffer(device, 384);
      const sentinel = 0xdeadbeef;
      outputBuffer.write(new Uint32Array(384).fill(sentinel));
      const graph = new GPUCommandGraph(device);
      const whole = importGraphBuffer(graph, 'shared-output', outputBuffer, 'uint32', 384);
      const positions = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 5);
      const inputWeights = importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', 5);
      // The binned path also exercises three uneven chunks sharing one input allocation.
      const chunks = <Format extends 'float32x2' | 'float32'>(view: GraphDataView<Format>) =>
        createVectorView(view.buffer.id, view.format, [
          graph.createDataView(view.buffer, {format: view.format, length: 1}),
          graph.createDataView(view.buffer, {
            format: view.format,
            length: 3,
            byteOffset: view.rowByteLength
          }),
          graph.createDataView(view.buffer, {
            format: view.format,
            length: 1,
            byteOffset: 4 * view.rowByteLength
          })
        ]);
      graph.add(
        new GPUKernelDensity({
          positions: method === 'binned' ? chunks(positions) : positions,
          weights: method === 'binned' ? chunks(inputWeights) : inputWeights,
          parameters: importGraphBuffer(graph, 'parameters', parametersBuffer, 'float32', 9),
          grid,
          indexGridSize: [2, 2],
          indexBounds: [0, 0, 2, 2],
          ...fixture.settings,
          binnedPadding: 8,
          bandwidthRule: 'scott',
          normalization: 'probability',
          output: {
            // Oversized views must preserve rows beyond the four result cells.
            values: graph.createDataView(whole.buffer, {format: 'float32', length: 8}),
            weightSum: graph.createDataView(whole.buffer, {
              format: 'float32',
              length: 1,
              byteOffset: 256
            }),
            bandwidth: graph.createDataView(whole.buffer, {
              format: 'float32',
              length: 1,
              byteOffset: 512
            }),
            status: {
              invalidCount: graph.createDataView(whole.buffer, {
                format: 'uint32',
                length: 1,
                byteOffset: 768
              }),
              insufficientPadding:
                method === 'exact'
                  ? undefined
                  : graph.createDataView(whole.buffer, {
                      format: 'uint32',
                      length: 1,
                      byteOffset: 1024
                    })
            }
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const bits = await readUint32(outputBuffer, 384);
        for (const [cell, value] of getFloats(bits.slice(0, 4)).entries()) {
          expect(value).toBeCloseTo(expectedValues[cell], 5);
        }
        expect(bits.slice(4, 8)).toEqual(new Array(4).fill(sentinel));
        expect(getFloats(bits.slice(64, 65))[0]).toBe(10);
        expect(getFloats(bits.slice(128, 129))[0]).toBeCloseTo(bandwidth, 6);
        expect(bits[192]).toBe(1);
        expect(bits[256]).toBe(method === 'exact' ? sentinel : 0);
        expect(bits[65]).toBe(sentinel);
        // Re-encoding must not publish stale density, bandwidth, weight sum or status words.
        weightsBuffer.write(new Float32Array(5));
        submitGraph(device, compiled, undefined);
        const zeroed = await readUint32(outputBuffer, 384);
        expect(Array.from(getFloats(zeroed.slice(0, 4)))).toEqual(new Array(4).fill(Number.NaN));
        expect(getFloats(zeroed.slice(64, 65))[0]).toBe(0);
        expect(getFloats(zeroed.slice(128, 129))[0]).toBeNaN();
        expect(zeroed[192]).toBe(1);
        expect(zeroed.slice(4, 8)).toEqual(new Array(4).fill(sentinel));
      } finally {
        compiled.destroy();
        positionsBuffer.destroy();
        weightsBuffer.destroy();
        parametersBuffer.destroy();
        outputBuffer.destroy();
      }
    });
  }
}

it('unmasked weighted grid density exceeds a small scratch binding with independent cell totals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(5019);
  const chunked = await runDensity(device, fixture, {
    kind: 'grid',
    weighted: true,
    chunkLengths: [1000, 1000, 0, 1000, 1000, 1019],
    weightChunkLengths: [997, 1001, 1024, 999, 998],
    inputOffsets: true,
    scratchBindingByteLength: 16384,
    expectedScratchRoute: 'chunked-grid'
  });
  const oracle = computeGridDensity(
    Array.from(fixture.positions),
    Array.from(fixture.weights),
    [0, 0, 8, 4],
    [16, 12]
  );
  expect(chunked.counts).toEqual(oracle.counts);
  expect(chunked.sums).toEqual(getFloatBits(oracle.sums));
  expect(chunked.values).toEqual(getFloatBits(oracle.sums));
});

it('binned KDE exceeds a small scratch binding with independent density, bandwidth and status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(5019);
  // Positive binary weights give a closed-form weight sum and exact exclusions.
  fixture.weights.fill(0.5);
  fixture.weights[7] = -1;
  fixture.positions[28] = Number.NaN;
  for (const bandwidthRule of [undefined, 'scott'] as const) {
    const result = await runDensity(device, fixture, {
      kind: 'kde',
      weighted: true,
      bandwidthRule,
      chunkLengths: [1000, 1000, 0, 1000, 1000, 1019],
      inputOffsets: true,
      scratchBindingByteLength: 16384,
      expectedScratchRoute: 'split'
    });
    expect(result.invalidCount).toEqual([2]);
    expect(getFloats(result.weightSum)[0]).toBe(5017 * 0.5);
    const bandwidth = bandwidthRule
      ? getBandwidthOracle(fixture.positions, fixture.weights).isotropic
      : 3;
    if (bandwidthRule) expect(getFloats(result.bandwidth)[0]).toBeCloseTo(bandwidth, 5);
    const expected = getBinnedDensityModel({
      points: Array.from(fixture.positions),
      weights: Array.from(fixture.weights),
      width: 8,
      height: 4,
      start: [0, 0],
      cellSize: [1, 1],
      bandwidth,
      padding: 12
    });
    for (const [cell, value] of getFloats(result.values).entries()) {
      expect(value).toBeCloseTo(expected[cell], 3);
    }
  }
});

it('density weight totals preserve cancellation and propagate non-finite terms across chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {terms, expected} of [
    // Opposite float32 values cancel exactly; the remaining terms sum to three.
    {terms: [1e20, 1, -1e20, 2], expected: 3},
    {terms: [1, Number.POSITIVE_INFINITY, 2], expected: Number.POSITIVE_INFINITY},
    {terms: [1, Number.NEGATIVE_INFINITY, 2], expected: Number.NEGATIVE_INFINITY},
    {terms: [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY], expected: Number.NaN},
    {terms: [1, Number.NaN, 2], expected: Number.NaN},
    {terms: [], expected: 0}
  ]) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const chunks = terms.map((term, index) => {
      const buffer = createInputBuffer(device, Float32Array.of(term));
      buffers.push(buffer);
      return importGraphBuffer(graph, `term-${index}`, buffer, 'float32', 1);
    });
    const outputBuffer = createOutputBuffer(device, 1);
    buffers.push(outputBuffer);
    const output = importGraphBuffer(graph, 'total', outputBuffer, 'float32', 1);
    for (const node of createDensityWeightSumNodes(
      graph,
      'density-total',
      createVectorView('terms', 'float32', chunks),
      output
    ))
      graph.add(node);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const total = getFloats(await readUint32(outputBuffer, 1))[0];
      if (Number.isNaN(expected)) expect(total).toBeNaN();
      else expect(total).toBe(expected);
      // Reuse the compiled module after every special value. The non-finite flags and
      // result must refresh; every input chunk now contributes the finite number two.
      for (const buffer of buffers.slice(0, chunks.length)) buffer.write(Float32Array.of(2));
      submitGraph(device, compiled, undefined);
      expect(getFloats(await readUint32(outputBuffer, 1))[0]).toBe(chunks.length * 2);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

for (const kind of ['grid', 'hexagon'] as const)
  it(`fixed point tiles handle ${kind} scratch beyond a small binding and retain CPU totals`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(5019);
    const mask =
      kind === 'grid'
        ? Uint32Array.from({length: 5019}, (_, row) => (row % 3 === 0 ? 0 : 1))
        : undefined;
    const result = await runDensity(device, fixture, {
      kind,
      weighted: true,
      mask,
      chunkLengths: [1000, 0, 1000, 1000, 1000, 1019],
      weightChunkLengths: [997, 1001, 1024, 999, 998],
      scratchBindingByteLength: mask ? 32768 : 16384,
      expectedScratchRoute: 'tiled',
      inputOffsets: true
    });
    const accepted = Array.from(fixture.positions, (value, component) =>
      mask && mask[Math.floor(component / 2)] === 0 ? Number.NaN : value
    );
    const compute = kind === 'grid' ? computeGridDensity : computeHexagonDensity;
    const oracle = compute(accepted, Array.from(fixture.weights), [0, 0, 8, 4], [16, 12], 0.5);
    expect(result.counts).toEqual(oracle.counts);
    expect(result.sums).toEqual(getFloatBits(oracle.sums));
    expect(result.values).toEqual(getFloatBits(oracle.sums));
  });
