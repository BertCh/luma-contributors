// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Buffer} from '@luma.gl/core';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../../src/testing/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';
import {
  GPURasterStreamingAccumulator,
  createGPURasterStreamingAccumulatorEncodingPlan,
  getGPURasterStreamingAccumulatorParameterValues,
  getGPURasterFixedPointValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {
  GPURasterClimateRunAccumulator,
  createGPURasterClimateRunAccumulatorEncodingPlan,
  getGPURasterClimateRunAccumulatorParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-run-accumulator';
import {
  GPURasterStandardizedIndexCalibration,
  createGPURasterStandardizedIndexCalibrationEncodingPlan,
  getGPURasterStandardizedIndexCalibrationParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {
  GPURasterStandardizedIndex,
  getGPURasterStandardizedIndexParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';
import {GPURasterRollingMean} from '../../../src/raster-analysis/raster-time-series/gpu-raster-rolling-mean';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {RASTER_REGULARIZED_GAMMA_TAILS_WGSL} from '../../../src/raster-analysis/raster-time-series/raster-climate-distribution-wgsl';

it('LIF-02 retains one unit across two ordered dispatches for all three models', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const stream = createGPURasterStreamingAccumulatorEncodingPlan({maximumValueMagnitude: 1}, 1);
  const climate = createGPURasterClimateRunAccumulatorEncodingPlan(
    {threshold: 0, maximumIncrement: 1},
    1
  );
  const calibration = createGPURasterStandardizedIndexCalibrationEncodingPlan(
    {location: 0, maximumValueMagnitude: 1},
    1
  );
  const counts = harness.output('counts', 'uint32', 1);
  const sums = harness.output('sums', 'uint32', 3);
  const degreeDays = harness.output('degreeDays', 'uint32', 3);
  const calibrationCounts = harness.output('calibrationCounts', 'uint32', 3);
  const moments = harness.output('moments', 'uint32', 9);
  const fit = harness.output('fit', 'float32', 3);
  const contributors = [true, false].flatMap((reset, frame) => [
    new GPURasterStreamingAccumulator({
      id: `stream-${frame}`,
      cellCount: 1,
      chunkSliceCount: 1,
      values: harness.floats([1]),
      parameters: harness.floats(getGPURasterStreamingAccumulatorParameterValues(stream, {reset})),
      state: {counts, sums}
    }),
    new GPURasterClimateRunAccumulator({
      id: `climate-${frame}`,
      cellCount: 1,
      chunkSliceCount: 1,
      values: harness.floats([1]),
      parameters: harness.floats(
        getGPURasterClimateRunAccumulatorParameterValues(climate, {reset})
      ),
      state: {degreeDays}
    }),
    new GPURasterStandardizedIndexCalibration({
      id: `calibration-${frame}`,
      cellCount: 1,
      chunkSliceCount: 1,
      distribution: 'gamma',
      values: harness.floats([1]),
      parameters: harness.floats(
        getGPURasterStandardizedIndexCalibrationParameterValues(calibration, {reset})
      ),
      state: {counts: calibrationCounts, moments},
      fit
    })
  ]);
  const result = await harness.run(contributors);
  // Hand expectation: 1 + 1 = 2 in every sum; both calibration samples enter the continuous fit.
  expect(result['counts']).toEqual([2]);
  expect(result['calibrationCounts']).toEqual([2, 0, 2]);
  expect([...getGPURasterFixedPointValues(result['sums'], stream.sumUnitExponent)]).toEqual([2]);
  expect([
    ...getGPURasterFixedPointValues(result['degreeDays'], climate.degreeDayUnitExponent)
  ]).toEqual([2]);
  expect(
    getGPURasterFixedPointValues(result['moments'].slice(0, 3), calibration.sumUnitExponent)[0]
  ).toBe(2);
});

it('NUM-02 places the zero atom at original zero for a shifted gamma fit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterStandardizedIndex({
      cellCount: 1,
      chunkSliceCount: 2,
      distribution: 'gamma',
      zeroAtom: true,
      values: harness.floats([-0.5, 0]),
      fit: harness.floats([1, 1, 0.25]),
      parameters: harness.floats(getGPURasterStandardizedIndexParameterValues({location: -1}, 2)),
      output: {
        index: harness.output('index', 'float32', 2),
        status: harness.output('status', 'uint32', 2)
      }
    })
  );
  // scipy.special.ndtri(.75*gammainc(1,x+1)+.25*(x>=0)), for x=-.5,0.
  const expected = [-0.5385404161507371, 0.5950363673241081];
  expect(result['status']).toEqual([0, 0]);
  expect(result['index']).toHaveLength(2);
  result['index'].forEach((value, index) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(value).toBeCloseTo(expected[index], 5);
  });
});

it('NUM-06 checks both large-shape gamma tails against SciPy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const output = harness.output('tails', 'float32', 6);
  const result = await harness.run({
    getCommandNodes(graph) {
      return [
        createWGSLKernelNode(graph, {
          id: 'gamma-tails',
          operation: 'gamma-tails',
          invocationCount: 3,
          workgroupSize: 64,
          bindings: [{name: 'tails', view: output, type: 'f32', access: 'read_write'}],
          declarations: RASTER_REGULARIZED_GAMMA_TAILS_WGSL,
          body: `let pair = getConvergedRegularizedGammaTails(65536.0, 65280.0 + f32(index) * 256.0);
tails[tailsOffset + 2u * index] = pair.x;
tails[tailsOffset + 2u * index + 1u] = pair.y;`
        })
      ];
    }
  });
  // /usr/bin/python3: scipy.special.gammainc/gammaincc(65536, 65536 + k*256), k=-1,0,1.
  const expected = [
    0.15865463685376976, 0.8413453631462303, 0.5005194561383061, 0.49948054386169394,
    0.8413453597275153, 0.15865464027248466
  ];
  expect(result['tails']).toHaveLength(6);
  result['tails'].forEach((value, index) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - expected[index])).toBeLessThan(1e-5);
  });
});

it('NUM-06 distinguishes evaluation iteration exhaustion from calibration availability', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterStandardizedIndex({
      cellCount: 1,
      chunkSliceCount: 2,
      distribution: 'gamma',
      values: harness.floats([1e12, 0]),
      fit: harness.floats([1e12, 1, 0]),
      parameters: harness.floats([2, 0]),
      output: {
        index: harness.output('index', 'float32', 2),
        status: harness.output('status', 'uint32', 2)
      }
    })
  );
  // The SAME valid fit has an exact support-boundary CDF, although its bounded series at the mean exhausts.
  expect(result['status']).toEqual([3, 0]);
  expect(result['index']).toHaveLength(2);
  expect(Number.isNaN(result['index'][0])).toBe(true);
  expect(Number.isFinite(result['index'][1])).toBe(true);
  expect(result['index'][1]).toBeCloseTo(-8.21, 5);
});

it('SCL-06 retains overflow across an empty resubmission and clears it on reset', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const countsBuffer = createOutputBuffer(device, 1);
  const sumsBuffer = createOutputBuffer(device, 3);
  const statusBuffer = createOutputBuffer(device, 1);
  countsBuffer.write(Uint32Array.of(0xffffffff));
  const observedCounts: number[] = [];
  const observedStatus: number[] = [];
  try {
    for (const frame of [0, 1, 2]) {
      const graph = new GPUCommandGraph(device);
      const valuesBuffer = createInputBuffer(device, Float32Array.of(0));
      const parametersBuffer = createInputBuffer(
        device,
        getGPURasterStreamingAccumulatorParameterValues(
          {
            reset: frame === 2,
            activeSliceCount: frame === 0 ? 1 : 0,
            maximumValueMagnitude: 1,
            maximumStreamSliceCount: frame === 2 ? 0 : 0xffffffff
          },
          1
        )
      );
      const view = <Format extends 'uint32' | 'float32'>(
        buffer: Buffer,
        format: Format,
        length: number,
        name: string
      ) =>
        graph.createDataView(
          graph.importBuffer(
            {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
            buffer
          ),
          {format, length}
        );
      graph.add(
        new GPURasterStreamingAccumulator({
          id: `overflow-${frame}`,
          cellCount: 1,
          chunkSliceCount: 1,
          values: view(valuesBuffer, 'float32', 1, 'values'),
          parameters: view(parametersBuffer, 'float32', 5, 'parameters'),
          state: {
            counts: view(countsBuffer, 'uint32', 1, 'counts'),
            sums: view(sumsBuffer, 'uint32', 3, 'sums'),
            status: view(statusBuffer, 'uint32', 1, 'status')
          }
        })
      );
      const compiled = graph.compile();
      try {
        // Three actual submissions retaining the same physical state buffers.
        submitGraph(device, compiled, undefined);
        const counts = await readUint32(countsBuffer, 1);
        const status = await readUint32(statusBuffer, 1);
        expect(counts).toHaveLength(1);
        expect(status).toHaveLength(1);
        observedCounts.push(counts[0]);
        observedStatus.push(status[0]);
      } finally {
        compiled.destroy();
        valuesBuffer.destroy();
        parametersBuffer.destroy();
      }
    }
    expect(observedCounts).toEqual([0xffffffff, 0xffffffff, 0]);
    expect(observedStatus).toEqual([1, 1, 0]);
    expect(await readUint32(sumsBuffer, 3)).toEqual([0, 0, 0]);
  } finally {
    countsBuffer.destroy();
    sumsBuffer.destroy();
    statusBuffer.destroy();
  }
});

for (const masked of [false, true]) {
  it(`LIF-07 keeps the two-cell short-chunk pitch including observation masks (${masked})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const result = await harness.run(
      new GPURasterRollingMean({
        cellCount: 2,
        chunkSliceCount: 2,
        windowLength: 1,
        values: harness.floats([12, -999, 22, -999]),
        mask: harness.integers([1, 0, masked ? 0 : 1, 1]),
        parameters: harness.floats(
          getGPURasterStreamingAccumulatorParameterValues(
            {reset: true, activeSliceCount: 1, maximumValueMagnitude: 32},
            2
          )
        ),
        ring: harness.output('ring', 'float32', 2),
        seenCount: harness.output('seen', 'uint32', 2),
        output: harness.output('rolling', 'float32', 4)
      })
    );
    expect(result['rolling'][0]).toBe(12);
    if (masked) expect(Number.isNaN(result['rolling'][2])).toBe(true);
    else expect(result['rolling'][2]).toBe(22);
    expect(Number.isNaN(result['rolling'][1])).toBe(true);
    expect(Number.isNaN(result['rolling'][3])).toBe(true);
    expect(result['seen']).toEqual([1, 1]);
  });
}

it('NUM-02 counts the original zero atom before applying the continuous magnitude bound', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const plan = createGPURasterStandardizedIndexCalibrationEncodingPlan(
    {location: -1, maximumValueMagnitude: 0.5},
    3
  );
  const result = await harness.run(
    new GPURasterStandardizedIndexCalibration({
      cellCount: 1,
      chunkSliceCount: 3,
      distribution: 'gamma',
      zeroAtom: true,
      values: harness.floats([-0.75, 0, -0.5]),
      parameters: harness.floats(
        getGPURasterStandardizedIndexCalibrationParameterValues(plan, {reset: true})
      ),
      state: {
        counts: harness.output('counts', 'uint32', 3),
        moments: harness.output('moments', 'uint32', 9)
      },
      fit: harness.output('fit', 'float32', 3)
    })
  );
  // Two shifted continuous values .25 and .5, plus one original zero: q = 1 / 3.
  expect(result['counts']).toEqual([3, 1, 2]);
  expect(Number.isFinite(result['fit'][2])).toBe(true);
  expect(result['fit'][2]).toBeCloseTo(1 / 3, 6);
});

it('S1 classifies original subnormal signs and both signed zeros by bits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  // Negative subnormal, -0, +0, positive subnormal: bit identity must survive FTZ arithmetic.
  const original = new Float32Array(Uint32Array.of(0x80000001, 0x80000000, 0, 1).buffer);
  const evaluation = new GPURasterStandardizedIndex({
    cellCount: 1,
    chunkSliceCount: 4,
    distribution: 'gamma',
    zeroAtom: true,
    values: harness.floats(original),
    fit: harness.floats([1, 1, 0.25]),
    parameters: harness.floats([4, -1]),
    output: {
      index: harness.output('index', 'float32', 4),
      status: harness.output('status', 'uint32', 4)
    }
  });
  const calibration = new GPURasterStandardizedIndexCalibration({
    cellCount: 1,
    chunkSliceCount: 4,
    distribution: 'gamma',
    zeroAtom: true,
    values: harness.floats(original),
    parameters: harness.floats(
      getGPURasterStandardizedIndexCalibrationParameterValues(
        {reset: true, location: -1, maximumValueMagnitude: 2},
        4
      )
    ),
    state: {
      counts: harness.output('counts', 'uint32', 3),
      moments: harness.output('moments', 'uint32', 9)
    },
    fit: harness.output('fit', 'float32', 3)
  });
  const result = await harness.run([evaluation, calibration]);
  expect(result['counts']).toEqual([4, 2, 2]);
  expect(result['status']).toEqual([0, 0, 0, 0]);
  expect(result['index']).toHaveLength(4);
  // ndtri(.75*(1-exp(-1))) before the jump, ndtri(.75*(1-exp(-1))+.25) at/after it.
  const expected = [
    -0.06499141173357233, 0.5950363673241081, 0.5950363673241081, 0.5950363673241081
  ];
  for (let index = 0; index < 4; index++) {
    expect(Number.isFinite(result['index'][index])).toBe(true);
    expect(result['index'][index]).toBeCloseTo(expected[index], 5);
  }
});

it('EARLIER-1 scales a shifted gamma argument before its subtraction can overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRasterTimeSeriesHarness(device);
  const result = await harness.run(
    new GPURasterStandardizedIndex({
      cellCount: 1,
      chunkSliceCount: 1,
      distribution: 'gamma',
      values: harness.floats([2e38]),
      fit: harness.floats([1, 2e38, 0]),
      parameters: harness.floats([1, -2e38]),
      output: {
        index: harness.output('index', 'float32', 1),
        status: harness.output('status', 'uint32', 1)
      }
    })
  );
  // (2e38 - (-2e38))/2e38 = 2. Gamma(shape=1) CDF is 1-exp(-2); SciPy ndtri gives this literal.
  expect(result['status']).toEqual([0]);
  expect(result['index']).toHaveLength(1);
  expect(Number.isFinite(result['index'][0])).toBe(true);
  expect(result['index'][0]).toBeCloseTo(1.1015196284987498, 5);
});
