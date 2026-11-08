// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';

import {expect, it} from 'vitest';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import reference from '../../oracles/fixtures/raster-remote-sensing.json';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterConservativeRemapping,
  GPURasterTexture,
  GPURasterTemporal,
  GPURasterRegistration,
  GPURasterRegionGraph,
  GPURasterSpectralUnmixing,
  GPURasterSeasonalBreak
} from '../../../src/raster-analysis/remote-sensing/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('RS03 conserves source totals and distinguishes uncovered from dynamic missing support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const normalization of ['extensive', 'destination-area', 'covered-area'] as const) {
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterConservativeRemapping({
        source: {width: 2, height: 1, origin: [10000000, 0], spacing: [1, 1]},
        destination: {width: 3, height: 1, origin: [9999999.5, 0], spacing: [1, 1]},
        values: harness.input([2, 6]),
        validity: harness.mask([1, 1]),
        normalization,
        output: {
          values: harness.output('values', 'float32', 3),
          coveredArea: harness.output('area', 'float32', 3),
          status: harness.output('status', 'uint32', 3)
        }
      })
    );
    expect(result['area']).toEqual([0.5, 1, 0.5]);
    expect(result['status']).toEqual([0, 0, 0]);
    expect(result['values']).toEqual(normalization === 'covered-area' ? [2, 4, 6] : [1, 4, 3]);
  }
  const harness = createRemoteSensingHarness(device);
  const missing = await harness.run(
    new GPURasterConservativeRemapping({
      source: {width: 2, height: 1, origin: [0, 0], spacing: [1, 1]},
      destination: {width: 1, height: 1, origin: [0, 0], spacing: [2, 1]},
      values: harness.input([2, 6]),
      validity: harness.mask([0, 0]),
      normalization: 'covered-area',
      output: {
        values: harness.output('values', 'float32', 1),
        coveredArea: harness.output('area', 'float32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(missing['values'][0]).toBeNaN();
  expect(missing['status']).toEqual([1]);
});

it('RS04 exact checkerboard pair counts and normalized Haralick properties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const result = await harness.run(
    new GPURasterTexture({
      grid: makeGPURasterMetadata({width: 3, height: 3}),
      values: harness.input([0, 1, 0, 1, 0, 1, 0, 1, 0]),
      levels: 2,
      range: [0, 1],
      radius: 1,
      offset: [1, 0],
      symmetric: true,
      features: harness.output('features', 'float32', 63),
      pairCount: harness.output('count', 'uint32', 9)
    })
  );
  expect(result['count'][4]).toBe(12);
  const expected = [1, 1, 0.5, 0.5, Math.SQRT1_2, -1, Math.log(2)];
  expected.forEach((value, index) => expect(result['features'][28 + index]).toBeCloseTo(value, 5));
});

it('RS06 SG preserves quadratic polynomials at interpolation edges and through QA gaps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const values = Array.from({length: 9}, (_, slice) => 2 + 3 * slice + slice * slice);
  const result = await harness.run(
    new GPURasterTemporal({
      cellCount: 1,
      sliceCount: 9,
      values: harness.input(values),
      validity: harness.mask([1, 1, 0, 1, 1, 1, 0, 1, 1]),
      operation: 'savitzky-golay',
      windowLength: 7,
      polynomialOrder: 2,
      output: {
        values: harness.output('values', 'float32', 9),
        counts: harness.output('count', 'uint32', 9),
        chosenSlice: harness.output('date', 'uint32', 9),
        status: harness.output('status', 'uint32', 9)
      }
    })
  );
  expect(result['status']).toEqual(Array(9).fill(0));
  values.forEach((value, index) => expect(result['values'][index]).toBeCloseTo(value, 3));
  expect(result['date']).toEqual(Array(9).fill(0xffffffff));
});

it('RS06 composites retain original date provenance and omit QA/nonfinite values', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const operation of ['mean', 'minimum', 'maximum', 'median'] as const) {
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterTemporal({
        cellCount: 2,
        sliceCount: 5,
        values: harness.input([9, 3, 3, 100, NaN, NaN, NaN, NaN, NaN, NaN]),
        validity: harness.mask([1, 1, 1, 0, 1, 1, 1, 1, 1, 1]),
        operation,
        output: {
          values: harness.output('values', 'float32', 2),
          counts: harness.output('count', 'uint32', 2),
          chosenSlice: harness.output('date', 'uint32', 2),
          status: harness.output('status', 'uint32', 2)
        }
      })
    );
    expect(result['values'][0]).toBe(operation === 'mean' ? 5 : operation === 'maximum' ? 9 : 3);
    expect(result['values'][1]).toBeNaN();
    expect(result['count']).toEqual([3, 0]);
    expect(result['status']).toEqual([0, 1]);
    expect(result['date'][0]).toBe(
      operation === 'mean'
        ? 0xffffffff
        : operation === 'maximum'
          ? 0
          : operation === 'median'
            ? 2
            : 1
    );
  }
});

it('RS07 phase-correlation sign and subpixel refinement recover known periodic translations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 16;
  const height = 16;
  const shift = [1.25, -2.35];
  // Sum every non-Nyquist frequency with deterministic phases to avoid a repeated pattern.
  const sample = (column: number, row: number) => {
    let value = 0;
    for (let horizontal = -7; horizontal <= 7; horizontal++)
      for (let vertical = -7; vertical <= 7; vertical++) {
        if (horizontal === 0 && vertical === 0) continue;
        value += Math.cos(
          2 * Math.PI * ((horizontal * column) / width + (vertical * row) / height) +
            Math.sin(horizontal * 17 + vertical * 31)
        );
      }
    return value;
  };
  const harness = createRemoteSensingHarness(device);
  const result = await harness.run(
    new GPURasterRegistration({
      grid: makeGPURasterMetadata({width, height}),
      reference: harness.input(
        Array.from({length: 256}, (_, index) => sample(index % width, Math.floor(index / width)))
      ),
      moving: harness.input(
        Array.from({length: 256}, (_, index) =>
          sample((index % width) - shift[0], Math.floor(index / width) - shift[1])
        )
      ),
      upsampleFactor: 20,
      output: {
        shift: harness.output('shift', 'float32x2', 1),
        peak: harness.output('peak', 'float32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  expect(result['status']).toEqual([0]);
  expect(Math.abs(result['shift'][0] + shift[0])).toBeLessThanOrEqual(0.02501);
  expect(Math.abs(result['shift'][1] + shift[1])).toBeLessThanOrEqual(0.02501);
});

it('RS08 aggregates symmetric boundaries and reports unfinished fixed-weight merge rounds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const maximumIterations of [1, 3]) {
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegionGraph({
        grid: makeGPURasterMetadata({width: 4, height: 1}),
        labels: harness.mask([0, 1, 2, 3]),
        values: harness.input([1, 2, 3, 4]),
        regionCount: 4,
        mergeThreshold: 1,
        maximumIterations,
        output: {
          adjacency: harness.output('adjacency', 'uint32', 16),
          weights: harness.output('weights', 'float32', 16),
          mapping: harness.output('mapping', 'uint32', 4),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['adjacency']).toEqual([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]);
    expect(result['mapping']).toEqual(maximumIterations === 1 ? [0, 0, 1, 2] : [0, 0, 0, 0]);
    expect(result['status']).toEqual([maximumIterations === 1 ? 1 : 0]);
  }
});

it('RS10 QR, NNLS and simplex recover abundances and report forced exhaustion', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const constraint of ['unconstrained', 'nonnegative', 'simplex'] as const) {
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterSpectralUnmixing({
        cellCount: 2,
        bandCount: 3,
        endmemberCount: 2,
        values: harness.input([0.25, 0.75, 1, -1, 2, 1]),
        endmembers: harness.input([1, 0, 1, 0, 1, 1]),
        constraint,
        maximumIterations: 1024,
        tolerance: 1e-6,
        output: {
          abundances: harness.output('abundances', 'float32', 4),
          residual: harness.output('residual', 'float32', 2),
          status: harness.output('status', 'uint32', 2),
          iterationCounts: harness.output('iterations', 'uint32', 2)
        }
      })
    );
    expect(result['status']).toEqual([0, 0]);
    expect(result['abundances'][0]).toBeCloseTo(0.25, 4);
    expect(result['abundances'][1]).toBeCloseTo(0.75, 4);
    const expected =
      constraint === 'unconstrained' ? [-1, 2] : constraint === 'nonnegative' ? [0, 1.5] : [0, 1];
    expected.forEach((value, index) =>
      expect(result['abundances'][index + 2]).toBeCloseTo(value, 4)
    );
  }
  const harness = createRemoteSensingHarness(device);
  const result = await harness.run(
    new GPURasterSpectralUnmixing({
      cellCount: 1,
      bandCount: 3,
      endmemberCount: 2,
      values: harness.input([0.25, 0.75, 1]),
      endmembers: harness.input([1, 0, 1, 0, 1, 1]),
      constraint: 'nonnegative',
      maximumIterations: 1,
      output: {
        abundances: harness.output('abundances', 'float32', 2),
        residual: harness.output('residual', 'float32', 1),
        status: harness.output('status', 'uint32', 1),
        iterationCounts: harness.output('iterations', 'uint32', 1)
      }
    })
  );
  expect(result['status']).toEqual([4]);
  expect(result['iterations']).toEqual([1]);
});

it('RS11 harmonic level-break model recovers acquisition date and magnitude despite gaps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const sliceCount = 36;
  const values = Array.from({length: sliceCount * 2}, (_, index) => {
    const slice = index % sliceCount;
    return (
      5 +
      0.2 * slice +
      2 * Math.cos((2 * Math.PI * slice) / 12) +
      (index >= sliceCount && slice >= 18 ? 4 : 0)
    );
  });
  const result = await harness.run(
    new GPURasterSeasonalBreak({
      cellCount: 2,
      sliceCount,
      period: 12,
      minimumSegmentLength: 8,
      values: harness.input(values),
      validity: harness.mask(values.map((_, index) => (index % 7 === 0 ? 0 : 1))),
      output: {
        coefficients: harness.output('coefficients', 'float32', 10),
        components: harness.output('components', 'float32', 216),
        breakIndex: harness.output('breakIndex', 'uint32', 2),
        improvement: harness.output('improvement', 'float32', 2),
        status: harness.output('status', 'uint32', 2)
      }
    })
  );
  expect(result['status']).toEqual([0, 0]);
  expect(result['breakIndex']).toEqual([0xffffffff, 18]);
  expect(result['coefficients'][9]).toBeCloseTo(4, 3);
  expect(result['components'][2]).toBeNaN();
});

it('RS06/RS10 agree with pinned SciPy SG, NNLS and constrained optimizer fixtures', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const temporal = reference.temporal;
  const harness = createRemoteSensingHarness(device);
  const smoothed = await harness.run(
    new GPURasterTemporal({
      cellCount: 1,
      sliceCount: temporal.values.length,
      values: harness.input(temporal.values),
      operation: 'savitzky-golay',
      windowLength: temporal.windowLength,
      polynomialOrder: temporal.polynomialOrder,
      output: {
        values: harness.output('values', 'float32', 9),
        counts: harness.output('count', 'uint32', 9),
        chosenSlice: harness.output('date', 'uint32', 9),
        status: harness.output('status', 'uint32', 9)
      }
    })
  );
  temporal.expected.forEach((value, index) =>
    expect(smoothed['values'][index]).toBeCloseTo(value, 4)
  );
  for (const constraint of ['unconstrained', 'nonnegative', 'simplex'] as const) {
    const harness = createRemoteSensingHarness(device);
    const fixture = reference.unmixing;
    const result = await harness.run(
      new GPURasterSpectralUnmixing({
        cellCount: 3,
        bandCount: 4,
        endmemberCount: 3,
        values: harness.input(fixture.values),
        endmembers: harness.input(fixture.endmembers),
        constraint,
        tolerance: 1e-7,
        maximumIterations: 4096,
        output: {
          abundances: harness.output('abundances', 'float32', 9),
          residual: harness.output('residual', 'float32', 3),
          status: harness.output('status', 'uint32', 3),
          iterationCounts: harness.output('iterations', 'uint32', 3)
        }
      })
    );
    expect(result['status']).toEqual([0, 0, 0]);
    fixture.solutions[constraint].abundances.forEach((value, index) =>
      expect(result['abundances'][index]).toBeCloseTo(value, 4)
    );
    fixture.solutions[constraint].residual.forEach((value, index) =>
      expect(result['residual'][index]).toBeCloseTo(value, 4)
    );
  }
});

it('RS07 rejects flat and repeated ambiguous alignment inputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const constant of [true, false]) {
    const harness = createRemoteSensingHarness(device);
    const values = Array.from({length: 64}, (_, index) => (constant ? 1 : index % 2));
    const result = await harness.run(
      new GPURasterRegistration({
        grid: makeGPURasterMetadata({width: 8, height: 8}),
        reference: harness.input(values),
        moving: harness.input(values),
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['status']).toEqual([constant ? 1 : 2]);
    expect(result['shift'].every(Number.isNaN)).toBe(true);
  }
});

it('RS07 rejects non-flat images without a usable common phase spectrum', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [referenceValues, movingValues] of [
    // Orthogonal row/column stripes have no common nonzero Fourier coefficient.
    // At 2x2 every peak is adjacent, so the ambiguity test cannot catch this case.
    [
      [1, -1, 1, -1],
      [1, 1, -1, -1]
    ],
    // Finite input can still overflow the FFT/product arithmetic.
    [
      [3e38, -3e38, -3e38, 3e38],
      [3e38, -3e38, -3e38, 3e38]
    ]
  ]) {
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegistration({
        grid: makeGPURasterMetadata({width: 2, height: 2}),
        reference: harness.input(referenceValues),
        moving: harness.input(movingValues),
        output: {
          shift: harness.output('shift', 'float32x2', 1),
          peak: harness.output('peak', 'float32', 1),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['status']).toEqual([1]);
    expect(result['shift'].every(Number.isNaN)).toBe(true);
    expect(result['peak'][0]).toBeNaN();
  }
});

it('RS03 compiled graph recomputes dynamic QA support without retaining previous weights', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const importView = <Format extends 'float32' | 'uint32'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    buffers.push(buffer);
    const handle = graph.importBuffer(
      {id: `repeat-${serial++}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length});
  };
  const sourceBuffer = createInputBuffer(device, new Float32Array([2, 6]));
  const maskBuffer = createInputBuffer(device, new Uint32Array([1, 1]));
  const valueBuffer = createOutputBuffer(device, 1);
  const areaBuffer = createOutputBuffer(device, 1);
  const statusBuffer = createOutputBuffer(device, 1);
  graph.add(
    new GPURasterConservativeRemapping({
      source: {width: 2, height: 1, origin: [0, 0], spacing: [1, 1]},
      destination: {width: 1, height: 1, origin: [0, 0], spacing: [2, 1]},
      values: importView(sourceBuffer, 'float32', 2),
      validity: importView(maskBuffer, 'uint32', 2),
      normalization: 'covered-area',
      output: {
        values: importView(valueBuffer, 'float32', 1),
        coveredArea: importView(areaBuffer, 'float32', 1),
        status: importView(statusBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(valueBuffer, 1)).toEqual([4]);
    expect(await readFloat32(areaBuffer, 1)).toEqual([2]);
    sourceBuffer.write(new Float32Array([10, 100]));
    maskBuffer.write(new Uint32Array([1, 0]));
    submitGraph(device, compiled, undefined);
    expect(await readFloat32(valueBuffer, 1)).toEqual([10]);
    expect(await readFloat32(areaBuffer, 1)).toEqual([1]);
    maskBuffer.write(new Uint32Array([0, 0]));
    submitGraph(device, compiled, undefined);
    expect((await readFloat32(valueBuffer, 1))[0]).toBeNaN();
    expect(await readUint32(statusBuffer, 1)).toEqual([1]);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

it('RS06/RS10/RS11 keep insufficient and singular models invalid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const smoothing = createRemoteSensingHarness(device);
  const smoothResult = await smoothing.run(
    new GPURasterTemporal({
      cellCount: 1,
      sliceCount: 7,
      values: smoothing.input([1, NaN, NaN, 4, NaN, NaN, NaN]),
      operation: 'savitzky-golay',
      windowLength: 7,
      polynomialOrder: 2,
      output: {
        values: smoothing.output('values', 'float32', 7),
        counts: smoothing.output('count', 'uint32', 7),
        chosenSlice: smoothing.output('date', 'uint32', 7),
        status: smoothing.output('status', 'uint32', 7)
      }
    })
  );
  expect(smoothResult['status']).toEqual(Array(7).fill(1));
  expect(smoothResult['values'].every(Number.isNaN)).toBe(true);
  const unmixing = createRemoteSensingHarness(device);
  const unmixResult = await unmixing.run(
    new GPURasterSpectralUnmixing({
      cellCount: 1,
      bandCount: 3,
      endmemberCount: 2,
      values: unmixing.input([1, 2, 3]),
      endmembers: unmixing.input([1, 2, 3, 2, 4, 6]),
      output: {
        abundances: unmixing.output('abundances', 'float32', 2),
        residual: unmixing.output('residual', 'float32', 1),
        status: unmixing.output('status', 'uint32', 1),
        iterationCounts: unmixing.output('iterations', 'uint32', 1)
      }
    })
  );
  expect(unmixResult['status']).toEqual([2]);
  expect(unmixResult['abundances'].every(Number.isNaN)).toBe(true);
  const seasonal = createRemoteSensingHarness(device);
  const seasonalResult = await seasonal.run(
    new GPURasterSeasonalBreak({
      cellCount: 1,
      sliceCount: 12,
      values: seasonal.input(Array(12).fill(NaN)),
      period: 6,
      minimumSegmentLength: 3,
      output: {
        coefficients: seasonal.output('coefficients', 'float32', 5),
        components: seasonal.output('components', 'float32', 36),
        breakIndex: seasonal.output('breakIndex', 'uint32', 1),
        improvement: seasonal.output('improvement', 'float32', 1),
        status: seasonal.output('status', 'uint32', 1)
      }
    })
  );
  expect(seasonalResult['status']).toEqual([1]);
  expect(seasonalResult['breakIndex']).toEqual([0xffffffff]);
  expect(seasonalResult['components'].every(Number.isNaN)).toBe(true);
});

it('RS08 preserves finite zero region means for opposite-sign extreme observations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mergeMode of ['fixed', 'adaptive'] as const) {
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterRegionGraph({
        grid: makeGPURasterMetadata({width: 3, height: 1}),
        labels: harness.mask([0, 0, 1]),
        values: harness.input([3e38, -3e38, 0]),
        regionCount: 2,
        mergeMode,
        mergeThreshold: 0,
        maximumIterations: 2,
        output: {
          adjacency: harness.output('adjacency', 'uint32', 4),
          weights: harness.output('weights', 'float32', 4),
          mapping: harness.output('mapping', 'uint32', 2),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['mapping']).toEqual([0, 0]);
    expect(result['status']).toEqual([0]);
    if (mergeMode === 'fixed') expect(result['weights'][1]).toBe(0);
  }
});

it('RS11 rejects overflowing training energy in both default and explicit harmonic modes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const seasonality of [undefined, 'harmonic'] as const) {
    const harness = createRemoteSensingHarness(device);
    const result = await harness.run(
      new GPURasterSeasonalBreak({
        cellCount: 1,
        sliceCount: 24,
        values: harness.input(Array(24).fill(1e19)),
        period: 6,
        minimumSegmentLength: 6,
        seasonality,
        output: {
          coefficients: harness.output('coefficients', 'float32', 5),
          components: harness.output('components', 'float32', 72),
          breakIndex: harness.output('breakIndex', 'uint32', 1),
          improvement: harness.output('improvement', 'float32', 1),
          status: harness.output('status', 'uint32', 1)
        }
      })
    );
    expect(result['status']).toEqual([2]);
    expect(result['breakIndex']).toEqual([0xffffffff]);
    expect(result['coefficients'].every(Number.isNaN)).toBe(true);
    expect(result['components'].every(Number.isNaN)).toBe(true);
    expect(result['improvement'].every(Number.isNaN)).toBe(true);
  }
});
