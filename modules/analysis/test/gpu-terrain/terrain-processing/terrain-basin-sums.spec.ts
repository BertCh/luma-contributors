// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUTerrainCatchmentSummary} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary';
import {GPUTerrainDepressionHierarchy} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/raster-grid';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review 2026-11-10, terrain-hydrology-02: sums over the cells of one basin must
// not be serial float32 sums. Every expected value below is a float64 sum of the float32 terms.
// A plain float32 sum of a million equal terms of 2000 is 0.85 % off, and of 262,144 terms of 0.7
// it is 0.26 % off, so a tolerance of 1e-5 separates the two.

async function setup() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let identifier = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, `input-${identifier++}`, buffer, format, values.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${identifier++}`, buffer, format, length),
      async read(count = length) {
        return Array.from(
          await (format === 'float32' ? readFloat32(buffer, count) : readUint32(buffer, count))
        );
      }
    };
  };
  return {
    graph,
    input,
    output,
    run() {
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      return () => {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      };
    }
  };
}

it('the mean of a million-cell basin at a constant 2000 m is 2000 m', async () => {
  const {graph, input, output, run} = await setup();
  const width = 1000,
    height = 1000,
    count = width * height;
  const statistics = output(6, 'float32'),
    hypsometry = output(2, 'float32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid: {width, height, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      elevation: input(new Float32Array(count).fill(2000), 'float32'),
      basinIds: input(new Uint32Array(count), 'uint32'),
      basinCount: 1,
      streams: input(new Uint32Array(count), 'uint32'),
      flowDirections: input(new Uint32Array(count), 'uint32'),
      histogramBins: 2,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: output(count * 3, 'float32').view,
        profileStatus: output(count, 'uint32').view
      }
    })
  );
  const destroy = run();
  try {
    // [area, mean, minimum, maximum, stream length, drainage density]
    expect(await statistics.read()).toEqual([count, 2000, 2000, 2000, 0, 0]);
    expect(await hypsometry.read()).toEqual([count, count]);
  } finally {
    destroy();
  }
}, 240000);

it('mean, ground area, stream length and hypsometry of a million-cell basin match float64', async () => {
  const {graph, input, output, run} = await setup();
  const width = 1000,
    height = 1000,
    count = width * height;
  // One degree of latitude from 60 N: the ground area of a cell changes by 3 % over the rows.
  const grid = {
    width,
    height,
    affine: [0.001, 0, 10, 0, -0.001, 60] as const,
    pixelInterpretation: 'area' as const
  };
  // Elevations between 1500 and 2500 m from a fixed linear congruential sequence.
  const elevation = new Float32Array(count);
  let state = 12345;
  for (let cell = 0; cell < count; cell++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    elevation[cell] = 1500 + (1000 * state) / 4294967296;
  }
  // Every cell is a stream cell flowing east; the last column is terminal.
  const directions = new Uint32Array(count).fill(1);
  for (let row = 0; row < height; row++) {
    directions[row * width + width - 1] = 0;
  }
  const histogramBins = 4;
  const statistics = output(6, 'float32'),
    hypsometry = output(histogramBins, 'float32');
  graph.add(
    new GPUTerrainCatchmentSummary({
      grid,
      cellSizeMode: 'geographic',
      elevation: input(elevation, 'float32'),
      basinIds: input(new Uint32Array(count), 'uint32'),
      basinCount: 1,
      streams: input(new Uint32Array(count).fill(1), 'uint32'),
      flowDirections: input(directions, 'uint32'),
      histogramBins,
      output: {
        statistics: statistics.view,
        hypsometry: hypsometry.view,
        profiles: output(count * 3, 'float32').view,
        profileStatus: output(count, 'uint32').view
      }
    })
  );
  const destroy = run();
  try {
    // Float64 reference: the terms are the float32 values the kernel sums.
    const georeference = getRasterGridGeoreference(grid);
    let expectedSum = 0,
      expectedArea = 0,
      expectedLength = 0,
      minimum = Infinity,
      maximum = -Infinity;
    const rowAreas = new Float64Array(height);
    for (let row = 0; row < height; row++) {
      const [cellWidth, cellHeight] = getRasterGroundCellSize(
        'geographic',
        georeference,
        row + 0.5,
        height
      );
      rowAreas[row] = Math.fround(Math.fround(cellWidth) * Math.fround(cellHeight));
      expectedArea += width * rowAreas[row];
      expectedLength += (width - 1) * Math.fround(cellWidth);
    }
    for (let cell = 0; cell < count; cell++) {
      expectedSum += elevation[cell];
      minimum = Math.min(minimum, elevation[cell]);
      maximum = Math.max(maximum, elevation[cell]);
    }
    const expectedBins = new Float64Array(histogramBins);
    for (let bin = 0; bin < histogramBins; bin++) {
      const threshold = Math.fround(minimum + ((maximum - minimum) * (bin + 1)) / histogramBins);
      for (let cell = 0; cell < count; cell++) {
        if (elevation[cell] <= threshold || bin + 1 === histogramBins) {
          expectedBins[bin] += rowAreas[Math.floor(cell / width)];
        }
      }
    }
    const actual = await statistics.read();
    expect(actual[0] / expectedArea - 1, 'area').toBeLessThan(1e-5);
    expect(actual[0] / expectedArea - 1, 'area').toBeGreaterThan(-1e-5);
    // A float32 near 2000 is spaced 1.2e-4 apart; the serial sum was 0.46 m off.
    expect(Math.abs(actual[1] - expectedSum / count), 'mean').toBeLessThan(5e-4);
    expect(actual[2]).toBe(minimum);
    expect(actual[3]).toBe(maximum);
    expect(Math.abs(actual[4] / expectedLength - 1), 'stream length').toBeLessThan(1e-5);
    expect(Math.abs(actual[5] / (expectedLength / expectedArea) - 1), 'density').toBeLessThan(2e-5);
    const bins = await hypsometry.read();
    for (let bin = 0; bin < histogramBins; bin++) {
      // The bin edge is a float32 threshold: allow the few cells that sit on it.
      expect(Math.abs(bins[bin] / expectedBins[bin] - 1), `bin ${bin}`).toBeLessThan(2e-5);
    }
    expect(Math.abs(bins[histogramBins - 1] / expectedArea - 1)).toBeLessThan(1e-5);
  } finally {
    destroy();
  }
}, 240000);

it('stage area and volume of a 262,144-cell depression match float64', async () => {
  const {graph, input, output, run} = await setup();
  const width = 512,
    height = 512,
    count = width * height;
  // Cell area 0.7 (not a dyadic number, so partial sums round), one flat basin at elevation 0.
  const grid = {
    width,
    height,
    affine: [0.7, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const stage = 2000.37;
  const curves = output(2, 'float32'),
    status = output(3, 'uint32');
  graph.add(
    new GPUTerrainDepressionHierarchy({
      grid,
      elevation: input(new Float32Array(count), 'float32'),
      outlets: input(new Uint32Array(count), 'uint32'),
      stages: input(Float32Array.of(stage), 'float32'),
      nodeCapacity: 1,
      output: {
        nodes: output(4, 'uint32').view,
        levels: output(2, 'float32').view,
        cellNodes: output(count, 'uint32').view,
        curves: curves.view,
        status: status.view
      }
    })
  );
  const destroy = run();
  try {
    // One node holds every cell: [stored, required, overflow].
    expect(await status.read()).toEqual([1, 1, 0]);
    const cellArea = Math.fround(0.7);
    const expectedArea = count * cellArea;
    const expectedVolume = count * Math.fround(Math.fround(stage) * cellArea);
    const [area, volume] = await curves.read();
    expect(Math.abs(area / expectedArea - 1), 'area').toBeLessThan(1e-6);
    expect(Math.abs(volume / expectedVolume - 1), 'volume').toBeLessThan(1e-6);
  } finally {
    destroy();
  }
}, 240000);
