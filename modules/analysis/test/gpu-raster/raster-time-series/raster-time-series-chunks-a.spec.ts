// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterBestAvailablePixel} from '../../../src/raster-analysis/raster-time-series/gpu-raster-best-available-pixel';
import {GPURasterBFASTMonitor} from '../../../src/raster-analysis/raster-time-series/gpu-raster-bfast-monitor';
import {GPURasterClimateAnomaly} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-anomaly';
import {GPURasterClimateRunAccumulator} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-run-accumulator';
import {GPURasterCOLDMonitor} from '../../../src/raster-analysis/raster-time-series/gpu-raster-cold-monitor';
import {GPURasterEOFProjection} from '../../../src/raster-analysis/raster-time-series/gpu-raster-eof-projection';
import {GPURasterGeometricMedianComposite} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';
import {GPURasterHANTS} from '../../../src/raster-analysis/raster-time-series/gpu-raster-hants';
import {GPURasterHarmonicRegression} from '../../../src/raster-analysis/raster-time-series/gpu-raster-harmonic-regression';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type Family =
  | 'best'
  | 'bfast'
  | 'anomaly'
  | 'runs'
  | 'cold'
  | 'eof'
  | 'geometric'
  | 'hants'
  | 'harmonic';
const MISSING_CELL = 2;
const FAMILIES = [
  'best',
  'bfast',
  'anomaly',
  'runs',
  'cold',
  'eof',
  'geometric',
  'hants',
  'harmonic'
] as const;

type RecordedOutput = {buffer: Buffer; prefix: number; length: number; integer: boolean};

it('harmonic reference fits a constant series identically with pooled and separate outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'uint32x2'>(
    name: string,
    format: Format,
    data: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      format === 'uint32x2' ? data.length / 2 : data.length
    );
  };
  const allocate = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const values = input('constant', 'float32', new Float32Array(16).fill(3));
  const parameters = input('period', 'float32', Float32Array.of(1));
  const dates = BigInt64Array.from({length: 16}, (_, slice) => BigInt(slice) * 7_889_238n);
  const dateAxis = {
    acquisitionTimes: input('dates', 'uint32x2', new Uint32Array(dates.buffer)),
    temporalContext: {
      encoding: 'int64-words',
      unit: 'second',
      epoch: 0n,
      axis: {kind: 'irregular'}
    } as const
  };
  const pooled = allocate(256);
  const pooledView = importGraphBuffer(graph, 'pooled', pooled, 'float32', 256);
  const coefficients = allocate(4);
  const fitted = allocate(16);
  const summaries = [allocate(2), allocate(2)];
  for (const [index, output] of [
    {
      coefficients: graph.createDataView(pooledView.buffer, {format: 'float32', length: 4}),
      fitted: graph.createDataView(pooledView.buffer, {
        format: 'float32',
        length: 16,
        byteOffset: 256
      })
    },
    {
      coefficients: importGraphBuffer(graph, 'coefficients', coefficients, 'float32', 4),
      fitted: importGraphBuffer(graph, 'fitted', fitted, 'float32', 16)
    }
  ].entries()) {
    graph.add(
      new GPURasterHarmonicRegression({
        id: `constant-${index}`,
        mode: 'reference',
        cellCount: 1,
        sliceCount: 16,
        harmonicCount: 1,
        values,
        parameters,
        dateAxis,
        output: {
          ...output,
          summary: importGraphBuffer(graph, `summary-${index}`, summaries[index], 'uint32', 2)
        }
      })
    );
  }
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const packed = await readFloat32(pooled, 80);
    const separateCoefficients = await readFloat32(coefficients, 4);
    const separateFitted = await readFloat32(fitted, 16);
    for (const [index, value] of separateCoefficients.entries()) {
      expect(value).toBeCloseTo(index === 0 ? 3 : 0, 5);
      expect(packed[index]).toBeCloseTo(value, 5);
    }
    for (const [index, value] of separateFitted.entries()) {
      expect(value).toBeCloseTo(3, 5);
      expect(packed[64 + index]).toBeCloseTo(value, 5);
    }
    for (const summary of summaries) expect(await readUint32(summary, 2)).toEqual([0, 16]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

// Gaussian elimination of the normal equations is independent of the contributor's QR solver.
function fitBFASTConstantWithNoise(cell: number): number[] {
  const design = Array.from({length: 12}, (_, slice) => {
    const years = (slice - 7.5) / 4;
    return [1, years, Math.cos(2 * Math.PI * years), Math.sin(2 * Math.PI * years)];
  });
  const system = Array.from({length: 4}, (_, column) => [
    ...Array.from({length: 4}, (_, other) =>
      design.reduce((sum, row) => sum + row[column] * row[other], 0)
    ),
    design.reduce(
      (sum, row, slice) => sum + row[column] * (cell + 2 + (slice % 2 ? -0.125 : 0.125)),
      0
    )
  ]);
  for (let column = 0; column < 4; column++) {
    const pivot = system[column][column];
    for (let entry = column; entry <= 4; entry++) system[column][entry] /= pivot;
    for (let row = 0; row < 4; row++) {
      if (row === column) continue;
      const multiplier = system[row][column];
      for (let entry = column; entry <= 4; entry++)
        system[row][entry] -= multiplier * system[column][entry];
    }
  }
  return system.map(row => row[4]);
}

async function runCase(
  family: Family,
  slices = 16,
  seasonalModel = false,
  finalChunkCellCount = 1
) {
  const cellCount = 6 + finalChunkCellCount;
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs: Record<string, RecordedOutput[]>[] = [];
  let serial = 0;
  const packed = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `table-${serial++}`,
      buffer,
      format,
      format === 'uint32x2' ? values.length / 2 : values.length
    );
  };
  const seconds = BigInt64Array.from({length: slices}, (_, slice) => BigInt(slice) * 7_889_238n);
  const dateAxis = {
    acquisitionTimes: packed('uint32x2', new Uint32Array(seconds.buffer)),
    temporalContext: {
      encoding: 'int64-words',
      unit: 'second',
      epoch: 0n,
      axis: {kind: 'irregular'}
    } as const
  };
  const configurations =
    family === 'harmonic' && slices <= 256
      ? [
          {cells: [cellCount], reference: false},
          {cells: [3, 3, finalChunkCellCount], reference: false},
          {cells: [cellCount], reference: true}
        ]
      : [
          {cells: [cellCount], reference: false},
          {cells: [3, 3, finalChunkCellCount], reference: false}
        ];
  for (const [route, configuration] of configurations.entries()) {
    const {cells} = configuration;
    const recorded: Record<string, RecordedOutput[]> = {};
    outputs.push(recorded);
    const rows = <Format extends 'float32' | 'uint32'>(
      name: string,
      format: Format,
      width: number,
      value?: (cell: number, entry: number) => number
    ): GraphDataView<Format> | GraphVectorView<Format> => {
      let cellBase = 0;
      const chunks = cells.map((cellCount, chunkIndex) => {
        const prefix = 1 + chunkIndex * 3 + (serial % 13);
        const length = cellCount * width;
        const buffer = createOutputBuffer(device, prefix + length);
        buffers.push(buffer);
        if (value) {
          const data =
            format === 'float32'
              ? new Float32Array(prefix + length)
              : new Uint32Array(prefix + length);
          for (let entry = 0; entry < length; entry++)
            data[prefix + entry] = value(cellBase + Math.floor(entry / width), entry % width);
          buffer.write(data);
        } else {
          recorded[name] ??= [];
          recorded[name].push({buffer, prefix, length, integer: format === 'uint32'});
        }
        const backing = importGraphBuffer(
          graph,
          `rows-${serial++}`,
          buffer,
          format,
          prefix + length
        );
        cellBase += cellCount;
        return graph.createDataView(backing.buffer, {format, length, byteOffset: prefix * 4});
      });
      return cells.length === 1 ? chunks[0] : createVectorView(name, format, chunks);
    };
    const base = {id: `${family}-${route}`, cellCount, sliceCount: slices};
    if (family === 'anomaly') {
      graph.add(
        new GPURasterClimateAnomaly({
          ...base,
          slotCount: 2,
          chunkSliceCount: slices,
          standardized: true,
          accumulatorParameters: packed('float32', new Float32Array([0, 0, 0, 0, 1000])),
          state: {
            counts: rows('counts', 'uint32', 2, cell => (cell === MISSING_CELL ? 0 : 2)),
            sums: rows('sums', 'uint32', 6, (cell, entry) =>
              entry % 3 ? 0 : 2 * (cell + 2 + entry / 3)
            ),
            moments: rows('moments', 'uint32', 6, (_cell, entry) => (entry % 3 ? 0 : 4))
          },
          values: rows('values', 'float32', slices, (cell, slice) => cell + 2 + slice),
          slotIndices: packed(
            'uint32',
            Uint32Array.from({length: slices}, (_, slice) => slice % 2)
          ),
          output: {
            climatologyMean: rows('mean', 'float32', 2),
            standardDeviation: rows('deviation', 'float32', 2),
            anomaly: rows('anomaly', 'float32', slices)
          }
        })
      );
      continue;
    }
    const values = rows('values', 'float32', slices, (cell, slice) => {
      const years = (slice - (slices - 1) / 2) / 4;
      if (seasonalModel)
        return (
          cell +
          2 +
          0.25 * years +
          0.5 * Math.cos(2 * Math.PI * years) -
          0.75 * Math.sin(2 * Math.PI * years)
        );
      return (
        cell +
        2 +
        (family === 'eof'
          ? (cell + 1) * slice
          : family === 'bfast'
            ? slice % 2
              ? -0.125
              : 0.125
            : 0)
      );
    });
    const mask = rows('mask', 'uint32', slices, (cell, slice) =>
      cell === MISSING_CELL || (seasonalModel && cell === 1 && slice === 2) ? 0 : 1
    );
    const common = {...base, values, mask};
    switch (family) {
      case 'best':
        graph.add(
          new GPURasterBestAvailablePixel({
            ...common,
            bandCount: 1,
            parameters: packed('float32', new Float32Array([1, 1, 1, 2000, 1, 0, 0, 1, 0])),
            sliceAttributes: packed(
              'float32',
              Float32Array.from({length: slices * 2}, (_, entry) =>
                entry % 2 ? 2000 : 1 + entry / 2
              )
            ),
            output: {
              bestSlice: rows('bestSlice', 'uint32', 1),
              bestScore: rows('score', 'float32', 1),
              composite: rows('composite', 'float32', 1)
            }
          })
        );
        break;
      case 'bfast':
        graph.add(
          new GPURasterBFASTMonitor({
            ...common,
            dateAxis,
            harmonicCount: 1,
            historySliceCount: 12,
            parameters: packed('float32', new Float32Array([1, 0.25, 1000])),
            output: {
              coefficients: rows('coefficients', 'float32', 4),
              scores: rows('scores', 'float32', 2),
              cellSummary: rows('summary', 'uint32', 3),
              process: rows('process', 'float32', 4)
            }
          })
        );
        break;
      case 'runs':
        graph.add(
          new GPURasterClimateRunAccumulator({
            ...common,
            chunkSliceCount: slices,
            parameters: packed('float32', new Float32Array([1, slices, 0, 0])),
            state: {runs: rows('runs', 'uint32', 6), degreeDays: rows('degreeDays', 'uint32', 3)}
          })
        );
        break;
      case 'cold':
        graph.add(
          new GPURasterCOLDMonitor({
            ...common,
            dateAxis,
            bandCount: 1,
            minimumObservations: 5,
            maximumCoefficientCount: 4,
            maximumSegmentCount: 2,
            maximumFitCount: 96,
            parameters: packed('float32', new Float32Array([1, 100, 10000, 0, 0.5, 1, 1])),
            output: {
              segments: rows('segments', 'uint32', 8),
              coefficients: rows('coefficients', 'float32', 16),
              segmentStatistics: rows('statistics', 'float32', 4),
              cellSummary: rows('summary', 'uint32', 3)
            }
          })
        );
        break;
      case 'eof':
        graph.add(
          new GPURasterEOFProjection({
            ...common,
            modeCount: 2,
            weights: rows('weights', 'float32', 1, () => 0.25),
            parameters: packed('float32', new Float32Array([0, 10000])),
            temporalModes: packed(
              'float32',
              Float32Array.from({length: slices * 2}, (_, entry) =>
                entry === 0 || entry === slices * 2 - 1 ? 1 : 0
              )
            ),
            singularValues: packed('float32', new Float32Array([2, 1])),
            output: {eofs: rows('eofs', 'float32', 2), status: rows('status', 'uint32', 1)}
          })
        );
        break;
      case 'geometric':
        graph.add(
          new GPURasterGeometricMedianComposite({
            ...common,
            bandCount: 1,
            iterationCount: 2,
            parameters: packed('float32', new Float32Array([1e-5])),
            output: {
              composite: rows('composite', 'float32', 1),
              status: rows('status', 'uint32', 1),
              lastStep: rows('step', 'float32', 1)
            }
          })
        );
        break;
      case 'hants':
        graph.add(
          new GPURasterHANTS({
            ...common,
            dateAxis,
            harmonicCount: 1,
            rejectionPassCount: 0,
            parameters: packed('float32', new Float32Array([1, 100, 1])),
            output: {
              smoothed: rows('smoothed', 'float32', slices),
              coefficients: rows('coefficients', 'float32', 3),
              status: rows('status', 'uint32', 1),
              diagnostics: rows('diagnostics', 'uint32', 2)
            }
          })
        );
        break;
      case 'harmonic':
        graph.add(
          new GPURasterHarmonicRegression({
            ...common,
            dateAxis,
            harmonicCount: 1,
            mode: configuration.reference ? 'reference' : 'storage',
            parameters: packed('float32', new Float32Array([1])),
            output: {
              coefficients: rows('coefficients', 'float32', 4),
              summary: rows('summary', 'uint32', 2),
              fitted: rows('fitted', 'float32', slices),
              rmse: rows('rmse', 'float32', 1)
            }
          })
        );
        break;
    }
  }
  const compiled = graph.compile();
  try {
    const results: Record<string, number[]>[] = [];
    // Encode twice: immutable dispatch uniforms and transient QR scratch must survive repeated runs.
    for (let run = 0; run < 2; run++) {
      submitGraph(device, compiled, undefined);
      results.length = 0;
      for (const recorded of outputs) {
        const result: Record<string, number[]> = {};
        for (const [name, chunks] of Object.entries(recorded)) {
          result[name] = [];
          for (const {buffer, prefix, length, integer} of chunks)
            result[name].push(
              ...(await (integer ? readUint32 : readFloat32)(buffer, prefix + length)).slice(prefix)
            );
        }
        results.push(result);
      }
      for (const result of results.slice(1)) {
        for (const name of Object.keys(results[0])) {
          expect(result[name]).toHaveLength(results[0][name].length);
          for (let entry = 0; entry < result[name].length; entry++) {
            const expected = results[0][name][entry];
            if (Number.isNaN(expected)) expect(result[name][entry]).toBeNaN();
            else expect(Math.abs(result[name][entry] - expected)).toBeLessThanOrEqual(5e-4);
          }
        }
      }
    }
    return results[1];
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
}

function assertFamilyOracle(result: Record<string, number[]>, family: Family, cellCount: number) {
  for (let cell = 0; cell < cellCount; cell++) {
    if (cell === MISSING_CELL) continue;
    const value = cell + 2;
    switch (family) {
      case 'best':
        expect(result['bestSlice'][cell]).toBe(0);
        expect(result['score'][cell]).toBeCloseTo(1, 5);
        expect(result['composite'][cell]).toBe(value);
        break;
      case 'geometric':
        expect(result['composite'][cell]).toBeCloseTo(value, 4);
        expect(result['step'][cell]).toBeCloseTo(0, 4);
        expect(result['status'][cell]).toBe(0);
        break;
      case 'eof':
        expect(result['eofs'][cell * 2]).toBeCloseTo(-1.875 * (cell + 1), 4);
        expect(result['eofs'][cell * 2 + 1]).toBeCloseTo(3.75 * (cell + 1), 4);
        expect(result['status'][cell]).toBe(0);
        break;
      case 'runs':
        expect(result['runs'].slice(cell * 6, cell * 6 + 6)).toEqual([16, 16, 16, 1, 16, 0]);
        expect(result['degreeDays'].slice(cell * 3, cell * 3 + 3)).toEqual([16 * value, 0, 0]);
        break;
      case 'anomaly':
        expect(result['mean'].slice(cell * 2, cell * 2 + 2)).toEqual([value, value + 1]);
        expect(result['deviation'].slice(cell * 2, cell * 2 + 2)).toEqual([2, 2]);
        expect(result['anomaly'].slice(cell * 16, cell * 16 + 16)).toEqual(
          Array.from({length: 16}, (_, slice) => Math.floor(slice / 2))
        );
        break;
      case 'hants':
      case 'harmonic': {
        const columns = family === 'hants' ? 3 : 4;
        for (let column = 0; column < columns; column++)
          expect(result['coefficients'][cell * columns + column]).toBeCloseTo(
            column ? 0 : value,
            3
          );
        for (const fitted of result[family === 'hants' ? 'smoothed' : 'fitted'].slice(
          cell * 16,
          cell * 16 + 16
        ))
          expect(fitted).toBeCloseTo(value, 3);
        expect(family === 'hants' ? result['status'][cell] : result['summary'][cell * 2]).toBe(0);
        break;
      }
      case 'bfast': {
        const expected = fitBFASTConstantWithNoise(cell);
        expected.forEach((coefficient, column) =>
          expect(result['coefficients'][cell * 4 + column]).toBeCloseTo(coefficient, 3)
        );
        expect(result['summary'].slice(cell * 3, cell * 3 + 3)).toEqual([0xffffffff, 0, 12]);
        break;
      }
      case 'cold':
        expect(result['segments'].slice(cell * 8, cell * 8 + 4)).toEqual([0, 15, 0xffffffff, 4]);
        expect(result['coefficients'][cell * 16]).toBeCloseTo(value, 3);
        expect(result['summary'].slice(cell * 3, cell * 3 + 2)).toEqual([1, 0]);
        break;
    }
  }
  // Missing support is in an earlier chunk; the final chunk must compute valid distinct values.
  if (family === 'best') expect(result['bestSlice'][MISSING_CELL]).toBe(0xffffffff);
  if (family === 'geometric' || family === 'eof' || family === 'hants')
    expect(result['status'][MISSING_CELL]).toBe(1);
  if (family === 'runs')
    expect(result['runs'].slice(MISSING_CELL * 6, MISSING_CELL * 6 + 6)).toEqual([
      0, 0, 0, 0, 0, 16
    ]);
  if (family === 'harmonic')
    expect(result['summary'].slice(MISSING_CELL * 2, MISSING_CELL * 2 + 2)).toEqual([1, 0]);
  if (family === 'bfast')
    expect(result['summary'].slice(MISSING_CELL * 3, MISSING_CELL * 3 + 3)).toEqual([
      0xffffffff, 1, 0
    ]);
  if (family === 'cold')
    expect(result['summary'].slice(MISSING_CELL * 3, MISSING_CELL * 3 + 3)).toEqual([0, 1, 0]);
  if (family === 'anomaly')
    expect(
      result['anomaly'].slice(MISSING_CELL * 16, MISSING_CELL * 16 + 16).every(Number.isNaN)
    ).toBe(true);
}

for (const family of FAMILIES) {
  for (const finalChunkCellCount of [1, 7, 8, 9, 15, 16, 17, 63, 64, 65]) {
    it(`${family} computes a valid final chunk of ${finalChunkCellCount} cells with distinct values`, async () => {
      const result = await runCase(family, 16, false, finalChunkCellCount);
      assertFamilyOracle(result, family, 6 + finalChunkCellCount);
    });
  }
}

for (const sliceCount of [512, 4096]) {
  it(`fits ${sliceCount} slices beyond the old private-array limit with a closed-form constant model`, async () => {
    const result = await runCase('harmonic', sliceCount);
    for (let cell = 0; cell < 7; cell++) {
      if (cell === MISSING_CELL) continue;
      expect(result['summary'].slice(cell * 2, cell * 2 + 2)).toEqual([0, sliceCount]);
      for (let column = 0; column < 4; column++)
        expect(result['coefficients'][cell * 4 + column]).toBeCloseTo(column ? 0 : cell + 2, 3);
      for (const fitted of result['fitted'].slice(
        cell * sliceCount,
        cell * sliceCount + sliceCount
      ))
        expect(fitted).toBeCloseTo(cell + 2, 3);
      expect(result['rmse'][cell]).toBeCloseTo(0, 3);
    }
  });
}

it('matches the reference QR and a closed-form trend and seasonal model with a masked gap', async () => {
  const slices = 64;
  const result = await runCase('harmonic', slices, true);
  for (let cell = 0; cell < 7; cell++) {
    if (cell === MISSING_CELL) continue;
    const expected = [cell + 2, 0.25, 0.5, -0.75];
    expected.forEach((coefficient, column) =>
      expect(result['coefficients'][cell * 4 + column]).toBeCloseTo(coefficient, 3)
    );
    expect(result['summary'].slice(cell * 2, cell * 2 + 2)).toEqual([0, cell === 1 ? 63 : 64]);
    for (let slice = 0; slice < slices; slice++) {
      const years = (slice - (slices - 1) / 2) / 4;
      const fitted =
        cell +
        2 +
        0.25 * years +
        0.5 * Math.cos(2 * Math.PI * years) -
        0.75 * Math.sin(2 * Math.PI * years);
      expect(result['fitted'][cell * slices + slice]).toBeCloseTo(fitted, 3);
    }
    expect(result['rmse'][cell]).toBeCloseTo(0, 3);
  }
});

/** Small test rig with partial readback, so the scratch-boundary case never reads millions of rows. */
function createChunkHarness(device: import('@luma.gl/core').Device) {
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  return {
    graph,
    field<Format extends 'float32' | 'uint32' | 'uint32x2'>(
      format: Format,
      length: number,
      values?: Float32Array | Uint32Array,
      prefix = 1 + (serial % 17)
    ) {
      const components = format === 'uint32x2' ? 2 : 1;
      const buffer = createOutputBuffer(device, prefix + length * components);
      if (values) buffer.write(values, prefix * 4);
      const backing = importGraphBuffer(
        graph,
        `review-field-${serial++}`,
        buffer,
        format === 'float32' ? 'float32' : 'uint32'
      );
      const view = graph.createDataView(backing.buffer, {format, length, byteOffset: prefix * 4});
      buffers.set(view, buffer);
      return view;
    },
    write(view: GraphDataView, values: Float32Array | Uint32Array) {
      buffers.get(view)!.write(values, view.byteOffset);
    },
    async read(view: GraphDataView, firstScalar = 0, scalarCount = view.length) {
      const bytes = await buffers
        .get(view)!
        .readAsync(view.byteOffset + firstScalar * 4, scalarCount * 4);
      return Array.from(
        view.format === 'float32'
          ? new Float32Array(bytes.buffer, bytes.byteOffset, scalarCount)
          : new Uint32Array(bytes.buffer, bytes.byteOffset, scalarCount)
      );
    },
    destroy() {
      for (const buffer of buffers.values()) buffer.destroy();
    }
  };
}

it('fits expected values on both sides of the 524288-cell automatic QR scratch boundary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createChunkHarness(device);
  const cellCount = 524_289;
  const sliceCount = 16;
  const values = new Float32Array(cellCount * sliceCount);
  for (let cell = 0; cell < cellCount; cell++)
    values.fill(2 + (cell % 17), cell * sliceCount, (cell + 1) * sliceCount);
  const seconds = BigInt64Array.from(
    {length: sliceCount},
    (_, slice) => BigInt(slice) * 7_889_238n
  );
  const coefficients = harness.field('float32', cellCount * 4);
  const summary = harness.field('uint32', cellCount * 2);
  const rmse = harness.field('float32', cellCount);
  harness.graph.add(
    new GPURasterHarmonicRegression({
      cellCount,
      sliceCount,
      harmonicCount: 1,
      values: harness.field('float32', values.length, values),
      parameters: harness.field('float32', 1, new Float32Array([1])),
      dateAxis: {
        acquisitionTimes: harness.field('uint32x2', sliceCount, new Uint32Array(seconds.buffer)),
        temporalContext: {
          encoding: 'int64-words',
          unit: 'second',
          epoch: 0n,
          axis: {kind: 'irregular'}
        }
      },
      output: {coefficients, summary, rmse}
    })
  );
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const cell of [0, 1, 524_286, 524_287, 524_288]) {
      expect(await harness.read(summary, cell * 2, 2)).toEqual([0, 16]);
      const fitted = await harness.read(coefficients, cell * 4, 4);
      fitted.forEach((value, column) => expect(value).toBeCloseTo(column ? 0 : 2 + (cell % 17), 3));
      expect((await harness.read(rmse, cell, 1))[0]).toBeCloseTo(0, 3);
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
}, 120_000);

/** Binary64 normal-equation oracle; the GPU routes use two-pass modified Gram-Schmidt. */
function fitHarmonicOnCPU(years: number[], values: Float32Array, mask: Uint32Array) {
  const columns = 4;
  const design = years.map(year => [
    1,
    year,
    Math.cos(2 * Math.PI * year),
    Math.sin(2 * Math.PI * year)
  ]);
  const kept = years
    .map((_, slice) => slice)
    .filter(slice => mask[slice] !== 0 && Number.isFinite(values[slice]));
  const missing = (status: number) => ({
    status,
    count: kept.length,
    coefficients: Array(columns).fill(Number.NaN) as number[],
    fitted: Array(years.length).fill(Number.NaN) as number[],
    rmse: Number.NaN
  });
  if (kept.length <= columns) return missing(1);
  const system = Array.from({length: columns}, (_, column) => [
    ...Array.from({length: columns}, (_, other) =>
      kept.reduce((sum, slice) => sum + design[slice][column] * design[slice][other], 0)
    ),
    kept.reduce((sum, slice) => sum + design[slice][column] * values[slice], 0)
  ]);
  const maximumDiagonal = Math.max(...system.map((row, index) => Math.abs(row[index])));
  for (let column = 0; column < columns; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < columns; row++)
      if (Math.abs(system[row][column]) > Math.abs(system[pivotRow][column])) pivotRow = row;
    if (Math.abs(system[pivotRow][column]) < maximumDiagonal * 1e-12) return missing(2);
    [system[column], system[pivotRow]] = [system[pivotRow], system[column]];
    const pivot = system[column][column];
    for (let entry = column; entry <= columns; entry++) system[column][entry] /= pivot;
    for (let row = 0; row < columns; row++) {
      if (row === column) continue;
      const multiplier = system[row][column];
      for (let entry = column; entry <= columns; entry++)
        system[row][entry] -= multiplier * system[column][entry];
    }
  }
  const coefficients = system.map(row => row[columns]);
  const fitted = design.map(row =>
    row.reduce((sum, value, column) => sum + value * coefficients[column], 0)
  );
  const residual = kept.reduce((sum, slice) => sum + (values[slice] - fitted[slice]) ** 2, 0);
  return {
    status: 0,
    count: kept.length,
    coefficients,
    fitted,
    rmse: Math.sqrt(residual / (kept.length - columns))
  };
}

it('compares storage and reference with a CPU fit on irregular dates, missing data, nonzero residuals and rank deficiency', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createChunkHarness(device);
  const yearSlots = [0, 0.17, 0.55, 1, 1.2, 1.71, 2, 2.37, 2.91, 3, 3.22, 3.66, 4, 4.1, 4.4, 5];
  const seconds = BigInt64Array.from(yearSlots, year => BigInt(Math.round(year * 31556952)));
  const midpoint = seconds[0] + (seconds[seconds.length - 1] - seconds[0]) / 2n;
  const years = Array.from(seconds, second => Number(second - midpoint) / 31556952);
  const values = Array.from({length: 3}, (_, cell) =>
    Float32Array.from(
      years,
      (year, slice) =>
        4 +
        cell +
        0.35 * year +
        0.4 * Math.cos(2 * Math.PI * year) -
        0.2 * Math.sin(2 * Math.PI * year) +
        0.15 * ((slice % 3) - 1)
    )
  );
  values[0][4] = Number.NaN;
  const masks = [
    new Uint32Array(16).fill(1),
    Uint32Array.from(years, (_, slice) => ([1, 5, 12].includes(slice) ? 0 : 1)),
    Uint32Array.from(yearSlots, year => (Number.isInteger(year) ? 1 : 0))
  ];
  const expected = values.map((value, cell) => fitHarmonicOnCPU(years, value, masks[cell]));
  expect(expected.map(fit => fit.status)).toEqual([0, 0, 2]);
  expect(expected[0].rmse).toBeGreaterThan(0.05);
  expect(expected[1].rmse).toBeGreaterThan(0.05);
  const outputs: {
    coefficients: GraphVectorView<'float32'>;
    summary: GraphVectorView<'uint32'>;
    fitted: GraphVectorView<'float32'>;
    rmse: GraphVectorView<'float32'>;
  }[] = [];
  const dateAxis = {
    acquisitionTimes: harness.field('uint32x2', 16, new Uint32Array(seconds.buffer)),
    temporalContext: {
      encoding: 'int64-words',
      unit: 'second',
      epoch: 0n,
      axis: {kind: 'irregular'}
    } as const
  };
  const input = createVectorView(
    'irregular-values',
    'float32',
    values.map(value => harness.field('float32', 16, value))
  );
  const mask = createVectorView(
    'irregular-mask',
    'uint32',
    masks.map(value => harness.field('uint32', 16, value))
  );
  for (const mode of ['storage', 'reference'] as const) {
    const output = {
      coefficients: createVectorView(
        `${mode}-coefficients`,
        'float32',
        values.map(() => harness.field('float32', 4))
      ),
      summary: createVectorView(
        `${mode}-summary`,
        'uint32',
        values.map(() => harness.field('uint32', 2))
      ),
      fitted: createVectorView(
        `${mode}-fitted`,
        'float32',
        values.map(() => harness.field('float32', 16))
      ),
      rmse: createVectorView(
        `${mode}-rmse`,
        'float32',
        values.map(() => harness.field('float32', 1))
      )
    };
    outputs.push(output);
    harness.graph.add(
      new GPURasterHarmonicRegression({
        id: `irregular-${mode}`,
        cellCount: 3,
        sliceCount: 16,
        harmonicCount: 1,
        mode,
        values: input,
        mask,
        parameters: harness.field('float32', 1, new Float32Array([1])),
        dateAxis,
        output
      })
    );
  }
  const compiled = harness.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const output of outputs)
      for (const [cell, fit] of expected.entries()) {
        expect(await harness.read(output.summary.data[cell])).toEqual([fit.status, fit.count]);
        const compare = (actual: number[], reference: number[]) =>
          actual.forEach((value, entry) => {
            if (Number.isNaN(reference[entry])) expect(value).toBeNaN();
            else expect(Math.abs(value - reference[entry])).toBeLessThanOrEqual(5e-4);
          });
        compare(await harness.read(output.coefficients.data[cell]), fit.coefficients);
        compare(await harness.read(output.fitted.data[cell]), fit.fitted);
        compare(await harness.read(output.rmse.data[cell]), [fit.rmse]);
      }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('preserves chunked runs and exact fixed-point carries across successive encodings with reset false', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createChunkHarness(device);
  const cells = [3, 3, 1];
  const values = createVectorView(
    'stream-values',
    'float32',
    cells.map(count => harness.field('float32', count * 4))
  );
  const runs = createVectorView(
    'stream-runs',
    'uint32',
    cells.map(count => harness.field('uint32', count * 6))
  );
  const degreeDays = createVectorView(
    'stream-degree-days',
    'uint32',
    cells.map(count => harness.field('uint32', count * 3))
  );
  const parameters = harness.field('float32', 4);
  harness.graph.add(
    new GPURasterClimateRunAccumulator({
      cellCount: 7,
      chunkSliceCount: 4,
      values,
      parameters,
      minimumSpellLength: 6,
      state: {runs, degreeDays}
    })
  );
  const compiled = harness.graph.compile();
  try {
    let samplesSeen = 0;
    for (const [call, sliceCount] of [4, 4, 2].entries()) {
      harness.write(parameters, new Float32Array([call === 0 ? 1 : 0, sliceCount, 0, -30]));
      let cellBase = 0;
      for (const [chunkIndex, count] of cells.entries()) {
        harness.write(
          values.data[chunkIndex],
          Float32Array.from(
            {length: count * 4},
            (_, entry) => cellBase + Math.floor(entry / 4) + samplesSeen + (entry % 4) + 1
          )
        );
        cellBase += count;
      }
      samplesSeen += sliceCount;
      submitGraph(device, compiled, undefined);
      cellBase = 0;
      for (const [chunkIndex, count] of cells.entries()) {
        const runWords = await harness.read(runs.data[chunkIndex]);
        const sumWords = await harness.read(degreeDays.data[chunkIndex]);
        for (let localCell = 0; localCell < count; localCell++) {
          const cell = cellBase + localCell;
          expect(runWords.slice(localCell * 6, localCell * 6 + 6)).toEqual([
            samplesSeen,
            samplesSeen,
            samplesSeen,
            samplesSeen >= 6 ? 1 : 0,
            samplesSeen >= 6 ? samplesSeen : 0,
            0
          ]);
          // sum_{sample=1..n}(cell + sample), quantized at unit 2^-30, with independent bigint carries.
          const total = BigInt(samplesSeen * cell + (samplesSeen * (samplesSeen + 1)) / 2) << 30n;
          expect(sumWords.slice(localCell * 3, localCell * 3 + 3)).toEqual([
            Number(total & 0xffffffffn),
            Number((total >> 32n) & 0xffffffffn),
            Number(total >> 64n)
          ]);
        }
        cellBase += count;
      }
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});
