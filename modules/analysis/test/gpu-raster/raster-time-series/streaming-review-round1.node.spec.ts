// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  GPURasterStreamingAccumulator,
  createGPURasterStreamingAccumulatorEncodingPlan,
  getGPURasterStreamingAccumulatorParameterValues,
  getGPURasterFixedPointValues,
  decodeGPURasterStreamingAccumulatorStatus
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {
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
  getGPURasterStandardizedIndexParameterValues,
  decodeGPURasterStandardizedIndexStatus
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index';

it('B1 resumes baseline moment words without changing units in either model', () => {
  // Baseline squares the original double, then converts 4*m*m to float32. Units are literal
  // -66/-68 at the default 2^24 headroom. 1 at those units is [0,0,4] / [0,0,16].
  for (const [maximumValueMagnitude, exponent, highWord] of [
    [Math.SQRT2, -66, 4],
    [Math.SQRT1_2, -68, 16]
  ]) {
    const streaming = getGPURasterStreamingAccumulatorParameterValues(
      {reset: false, maximumValueMagnitude},
      1
    );
    const calibration = getGPURasterStandardizedIndexCalibrationParameterValues(
      {reset: false, maximumValueMagnitude},
      1
    );
    expect(streaming[3]).toBe(exponent);
    expect(calibration[5]).toBe(exponent);
    expect(getGPURasterFixedPointValues([0, 0, highWord], streaming[3])[0]).toBe(1);
    expect(getGPURasterFixedPointValues([0, 0, highWord], calibration[5])[0]).toBe(1);
  }
});

it('B2 keeps structurally valid legacy options with unrelated kind and width fields', () => {
  const options = {
    kind: 'options',
    chunkSliceCount: 8,
    reset: true,
    maximumValueMagnitude: 1,
    maximumIncrement: 1,
    threshold: 0
  };
  const packets = [
    getGPURasterStreamingAccumulatorParameterValues(options, 2),
    getGPURasterClimateRunAccumulatorParameterValues(options, 2),
    getGPURasterStandardizedIndexCalibrationParameterValues(options, 2)
  ];
  for (const packet of packets) {
    expect(packet[1]).toBe(2);
    expect([...packet].every(Number.isFinite)).toBe(true);
  }
});

it('B2 rejects wrong-model and corrupted plan encodings before changing any destination', () => {
  const stream = createGPURasterStreamingAccumulatorEncodingPlan({maximumValueMagnitude: 1}, 2);
  const climate = createGPURasterClimateRunAccumulatorEncodingPlan(
    {threshold: 0, maximumIncrement: 1},
    2
  );
  const calibration = createGPURasterStandardizedIndexCalibrationEncodingPlan(
    {maximumValueMagnitude: 1},
    2
  );
  const checks = [
    {
      length: 5,
      pack: (target: Float32Array, wrongModel: boolean) =>
        getGPURasterStreamingAccumulatorParameterValues(
          {
            ...stream,
            ...(wrongModel
              ? {kind: 'options'}
              : {momentUnitExponent: stream.momentUnitExponent + 1})
          } as typeof stream,
          {reset: false},
          target
        )
    },
    {
      length: 4,
      pack: (target: Float32Array, wrongModel: boolean) =>
        getGPURasterClimateRunAccumulatorParameterValues(
          {
            ...climate,
            ...(wrongModel ? {kind: 'options'} : {degreeDayUnitExponent: NaN})
          } as typeof climate,
          {reset: false},
          target
        )
    },
    {
      length: 7,
      pack: (target: Float32Array, wrongModel: boolean) =>
        getGPURasterStandardizedIndexCalibrationParameterValues(
          {
            ...calibration,
            ...(wrongModel
              ? {kind: 'options'}
              : {logarithmUnitExponent: calibration.logarithmUnitExponent + 1})
          } as typeof calibration,
          {reset: false},
          target
        )
    }
  ];
  for (const {length, pack} of checks)
    for (const wrongModel of [false, true]) {
      const target = new Float32Array(length).fill(77);
      expect(() => pack(target, wrongModel)).toThrow();
      expect([...target]).toEqual(Array(length).fill(77));
    }
});

it('B3 rejects frames larger than declared total for all three models and overloads', () => {
  const settings = {
    reset: true,
    activeSliceCount: 8,
    maximumStreamSliceCount: 1,
    maximumValueMagnitude: 1,
    maximumIncrement: 1,
    threshold: 0
  };
  const stream = createGPURasterStreamingAccumulatorEncodingPlan(settings, 8);
  const climate = createGPURasterClimateRunAccumulatorEncodingPlan(settings, 8);
  const calibration = createGPURasterStandardizedIndexCalibrationEncodingPlan(settings, 8);
  const frame = {reset: true, activeSliceCount: 8};
  const checks = [
    {
      length: 5,
      pack: (target: Float32Array, planned: boolean) =>
        planned
          ? getGPURasterStreamingAccumulatorParameterValues(stream, frame, target)
          : getGPURasterStreamingAccumulatorParameterValues(settings, 8, target)
    },
    {
      length: 4,
      pack: (target: Float32Array, planned: boolean) =>
        planned
          ? getGPURasterClimateRunAccumulatorParameterValues(climate, frame, target)
          : getGPURasterClimateRunAccumulatorParameterValues(settings, 8, target)
    },
    {
      length: 7,
      pack: (target: Float32Array, planned: boolean) =>
        planned
          ? getGPURasterStandardizedIndexCalibrationParameterValues(calibration, frame, target)
          : getGPURasterStandardizedIndexCalibrationParameterValues(settings, 8, target)
    }
  ];
  for (const {length, pack} of checks)
    for (const planned of [false, true]) {
      const target = new Float32Array(length).fill(77);
      expect(() => pack(target, planned)).toThrow(/activeSliceCount.*maximumStreamSliceCount/);
      expect([...target]).toEqual(Array(length).fill(77));
    }
});

it('B3 permits an empty reset with zero active slices and zero total headroom', () => {
  const settings = {
    reset: true,
    activeSliceCount: 0,
    maximumStreamSliceCount: 0,
    maximumValueMagnitude: 1,
    maximumIncrement: 1,
    threshold: 0
  };
  const packets = [
    getGPURasterStreamingAccumulatorParameterValues(settings, 8),
    getGPURasterClimateRunAccumulatorParameterValues(settings, 8),
    getGPURasterStandardizedIndexCalibrationParameterValues(settings, 8)
  ];
  for (const packet of packets) {
    expect(packet[0]).toBe(1);
    expect(packet[1]).toBe(0);
    expect([...packet].every(Number.isFinite)).toBe(true);
  }
  const plan = createGPURasterStreamingAccumulatorEncodingPlan(settings, 8);
  expect(
    getGPURasterStreamingAccumulatorParameterValues(plan, {reset: true, activeSliceCount: 0})[1]
  ).toBe(0);
  expect(() =>
    getGPURasterStreamingAccumulatorParameterValues(plan, {reset: false, activeSliceCount: 0})
  ).toThrow();
});

it('B4 rejects pooled read-only counts and writable status even at disjoint aligned ranges', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const pooled = createTransientView(graph, 'pooled-state', 'uint32', 128);
    const counts = graph.createDataView(pooled.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 0
    });
    const status = graph.createDataView(pooled.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    const props = {
      cellCount: 1,
      chunkSliceCount: 1,
      values: createTransientView(graph, 'values', 'float32', 1),
      parameters: createTransientView(graph, 'parameters', 'float32', 5),
      state: {counts, status, sums: createTransientView(graph, 'sums', 'uint32', 3)}
    };
    expect(() => new GPURasterStreamingAccumulator(props)).toThrow(
      /read-only and writable storage/
    );
    const valid = new GPURasterStreamingAccumulator({
      ...props,
      state: {...props.state, status: createTransientView(graph, 'separate-status', 'uint32', 1)}
    });
    expect(valid.getCommandNodes(graph)).toHaveLength(2);
    // Sums and status are writable in separate passes, so this disjoint aligned pool is safe.
    const pooledSums = graph.createDataView(pooled.buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 0
    });
    const safePool = new GPURasterStreamingAccumulator({
      ...props,
      state: {
        counts: createTransientView(graph, 'separate-counts', 'uint32', 1),
        sums: pooledSums,
        status
      }
    });
    expect(safePool.getCommandNodes(graph)).toHaveLength(2);
  } finally {
    device.destroy();
  }
});

it('B5 distinguishes iteration exhaustion from an unusable calibration', () => {
  expect(decodeGPURasterStandardizedIndexStatus([3])).toEqual({
    complete: false,
    invalidInput: false,
    calibrationUnavailable: false,
    iterationLimit: true
  });
  expect(decodeGPURasterStandardizedIndexStatus([2])).toEqual({
    complete: false,
    invalidInput: false,
    calibrationUnavailable: true,
    iterationLimit: false
  });
});

for (const [name, decode] of [
  ['streaming', decodeGPURasterStreamingAccumulatorStatus],
  ['index', decodeGPURasterStandardizedIndexStatus]
] as const) {
  it(`B6 validates original numeric ${name} status before uint32 narrowing`, () => {
    for (const value of [NaN, Infinity, -Infinity, 0.7, -1, 2 ** 32])
      expect(() => decode([value]), String(value)).toThrow();
    const bytes = new Uint8Array(11);
    new DataView(bytes.buffer).setUint32(3, 1, true);
    expect(decode(bytes.subarray(3, 7)).complete).toBe(false);
  });
}

it('S2 keeps count and target validation IDs and paths stable with structured evidence', () => {
  let caught: unknown;
  try {
    getGPURasterStreamingAccumulatorParameterValues(
      {reset: false, maximumValueMagnitude: 1, activeSliceCount: 0.1},
      8
    );
  } catch (error) {
    caught = error;
  }
  expect(caught).toMatchObject({
    id: 'getGPURasterStreamingAccumulatorParameterValues',
    path: 'activeSliceCount',
    received: 0.1,
    expected: {minimum: 0, maximum: 8}
  });
  expect(String(caught)).toContain('0.1');
  caught = undefined;
  try {
    getGPURasterStandardizedIndexParameterValues({}, 8, new Float32Array(1));
  } catch (error) {
    caught = error;
  }
  expect(caught).toMatchObject({
    id: 'getGPURasterStandardizedIndexParameterValues',
    path: 'target',
    received: 1,
    expected: {minimumWords: 2}
  });
  expect(String(caught)).toContain('received 1');
});

it('S1 and EARLIER-1 generate valid atom/shifted-ratio shaders for both climate distributions', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floats = (length: number) =>
      createTransientView(graph, `floats-${serial++}`, 'float32', length);
    const integers = (length: number) =>
      createTransientView(graph, `integers-${serial++}`, 'uint32', length);
    for (const distribution of ['gamma', 'fisk'] as const) {
      const evaluation = new GPURasterStandardizedIndex({
        cellCount: 1,
        chunkSliceCount: 4,
        distribution,
        zeroAtom: true,
        values: floats(4),
        fit: floats(3),
        parameters: floats(2),
        output: {index: floats(4), status: integers(4)}
      });
      const calibration = new GPURasterStandardizedIndexCalibration({
        cellCount: 1,
        chunkSliceCount: 4,
        distribution,
        zeroAtom: true,
        values: floats(4),
        parameters: floats(7),
        state: {counts: integers(3), moments: integers(9)},
        fit: floats(3)
      });
      for (const contributor of [evaluation, calibration])
        for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
          expect(() => new WgslReflect(kernel.source)).not.toThrow();
          expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
        }
    }
  } finally {
    device.destroy();
  }
});
