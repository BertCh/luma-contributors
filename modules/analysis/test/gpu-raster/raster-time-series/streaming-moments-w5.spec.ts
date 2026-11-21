// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPURasterStreamingAccumulator,
  createGPURasterStreamingAccumulatorEncodingPlan,
  getGPURasterStreamingAccumulatorParameterValues,
  getGPURasterFixedPointValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {
  GPURasterStandardizedIndexCalibration,
  createGPURasterStandardizedIndexCalibrationEncodingPlan,
  getGPURasterStandardizedIndexCalibrationParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../../src/testing/gpu-contributor-test-utils';
import {decodeMomentReadback} from './streaming-moments-w5.test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

type MomentRoute = 'streaming' | 'calibration';

async function runMomentProbe(
  device: Device,
  route: MomentRoute,
  observations: [number, number],
  split: boolean,
  options: {
    location?: number;
    maximumValueMagnitude?: number;
    maximumStreamSliceCount?: number;
    distribution?: 'gamma' | 'fisk';
  } = {}
) {
  const chunkSliceCount = split ? 1 : 2;
  const settings = {
    location: options.location ?? 0,
    maximumValueMagnitude:
      options.maximumValueMagnitude ??
      Math.max(...observations.map(value => Math.abs(value - (options.location ?? 0)))),
    maximumStreamSliceCount: options.maximumStreamSliceCount
  };
  const streamPlan = createGPURasterStreamingAccumulatorEncodingPlan(settings, chunkSliceCount);
  const calibrationPlan = createGPURasterStandardizedIndexCalibrationEncodingPlan(
    settings,
    chunkSliceCount
  );
  const parametersFor = (reset: boolean) =>
    route === 'calibration'
      ? getGPURasterStandardizedIndexCalibrationParameterValues(calibrationPlan, {reset})
      : getGPURasterStreamingAccumulatorParameterValues(streamPlan, {reset});
  const momentUnit =
    route === 'calibration' ? calibrationPlan.momentUnitExponent : streamPlan.momentUnitExponent;
  // Calibration's squared deviations are the third statistic, at word offset 6.
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  let fitBuffer: Buffer | undefined;
  const inputBuffer = createInputBuffer(
    device,
    Float32Array.from(observations.slice(0, chunkSliceCount))
  );
  const parameterBuffer = createInputBuffer(device, parametersFor(true));
  const maskBuffer = createInputBuffer(device, new Uint32Array(chunkSliceCount).fill(1));
  buffers.push(inputBuffer, parameterBuffer, maskBuffer);
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const counts = output('counts', route === 'calibration' ? 3 : 1);
  const sums = output('sums', 3);
  const moments = output('moments', route === 'streaming' ? 3 : 9);
  const values = importGraphBuffer(graph, 'values', inputBuffer, 'float32', chunkSliceCount);
  const parameters = importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32');
  const mask = importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', chunkSliceCount);
  try {
    if (route === 'streaming') {
      const contributor = new GPURasterStreamingAccumulator({
        cellCount: 1,
        chunkSliceCount,
        values,
        mask,
        parameters,
        state: {counts: counts.view, sums: sums.view, moments: moments.view}
      });
      for (const node of contributor.getCommandNodes(graph)) graph.add(node);
    } else {
      fitBuffer = createOutputBuffer(device, 3);
      buffers.push(fitBuffer);
      const contributor = new GPURasterStandardizedIndexCalibration({
        cellCount: 1,
        chunkSliceCount,
        values,
        mask,
        parameters,
        distribution: options.distribution ?? 'gamma',
        state: {counts: counts.view, moments: moments.view},
        fit: importGraphBuffer(graph, 'fit', fitBuffer, 'float32', 3)
      });
      for (const node of contributor.getCommandNodes(graph)) graph.add(node);
    }
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    if (split) {
      expect([
        ...decodeMomentReadback(await readUint32(moments.buffer), route, momentUnit)
      ]).toEqual([0]);
      inputBuffer.write(Float32Array.of(observations[1]));
      parameterBuffer.write(parametersFor(false));
      submitGraph(device, compiled, undefined);
    }
    const countWords = await readUint32(counts.buffer);
    expect(countWords).toEqual(route === 'streaming' ? [2] : [2, 0, 2]);
    const momentWords = await readUint32(moments.buffer);
    const decoded = decodeMomentReadback(momentWords, route, momentUnit);
    const sumWords =
      route === 'streaming' ? await readUint32(sums.buffer) : momentWords.slice(0, 3);
    expect(sumWords).toHaveLength(3);
    const decodedSums = getGPURasterFixedPointValues(
      Uint32Array.from(sumWords),
      streamPlan.sumUnitExponent
    );
    expect(decodedSums).toHaveLength(1);
    const fit = fitBuffer ? await readFloat32(fitBuffer) : undefined;
    if (fit) expect(fit).toHaveLength(3);
    return {moments: decoded, sums: decodedSums, fit, momentUnit};
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const route of ['streaming', 'calibration'] as const) {
  for (const split of [false, true]) {
    for (const reversed of [false, true]) {
      it(`SCL-01 ${route} keeps half-unit squared deviations split=${split} reversed=${reversed}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const observations: [number, number] = reversed
          ? [10000001, 10000000]
          : [10000000, 10000001];
        const actual = await runMomentProbe(device, route, observations, split);
        // Independent closed form: mean=10000000.5; .5^2 + (-.5)^2 = .5.
        expect(actual.moments.every(Number.isFinite)).toBe(true);
        expect([...actual.moments]).toEqual([0.5]);
      });
    }
  }
}

for (const route of ['streaming', 'calibration'] as const) {
  it(`SCL-01 ${route} scales before squaring a large finite residual`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const observations: [number, number] =
      route === 'streaming' ? [2 ** 70, -(2 ** 70)] : [2 ** 70, 2 ** 71];
    const actual = await runMomentProbe(device, route, observations, true);
    // Closed form for two samples: M2 = (second-first)^2/2, beyond finite f32.
    const expected = route === 'streaming' ? 2 ** 141 : 2 ** 139;
    expect(actual.moments.every(Number.isFinite)).toBe(true);
    expect([...actual.moments]).toEqual([expected]);
  });
}

for (const route of ['streaming', 'calibration'] as const) {
  for (const split of [false, true]) {
    for (const reversed of [false, true]) {
      it(`B1 ${route} wide bound with explicit two-slice headroom split=${split} reversed=${reversed}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const observations: [number, number] = [2 ** 69, 2 ** 70];
        if (reversed) observations.reverse();
        const result = await runMomentProbe(device, route, observations, split, {
          maximumStreamSliceCount: 2
        });
        // (2^70-2^69)^2/2 = 2^137; bound exponent 142 + 1 + 2 - 95 = 50.
        expect(result.momentUnit).toBe(50);
        expect([...result.moments]).toEqual([2 ** 137]);
      });
    }
  }
}
for (const split of [false, true]) {
  for (const reversed of [false, true]) {
    it(`B2 calibration retains shifted low bits split=${split} reversed=${reversed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const observations: [number, number] = [16, 16 + 2 ** -19];
      if (reversed) observations.reverse();
      const result = await runMomentProbe(device, 'calibration', observations, split, {
        location: -64,
        maximumValueMagnitude: 81,
        distribution: 'fisk'
      });
      // Translation preserves M2=(2^-19)^2/2; Python closed-form Fisk mean/SD ratio.
      expect([...result.moments]).toEqual([2 ** -39]);
      expect([...result.sums]).toEqual([160 + 2 ** -19]);
      expect(result.fit).toBeDefined();
      expect(result.fit!.every(Number.isFinite)).toBe(true);
      expect(Math.abs(result.fit![0] / 152152520.3859001 - 1)).toBeLessThan(3e-7);
      expect(Math.abs(result.fit![1] / 80.00000095367432 - 1)).toBeLessThan(3e-7);
    });
    it(`Fisk calibration preserves a finite large-offset fit split=${split} reversed=${reversed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const observations: [number, number] = [2 ** 46, 2 ** 47];
      if (reversed) observations.reverse();
      const result = await runMomentProbe(device, 'calibration', observations, split, {
        distribution: 'fisk'
      });
      // mean=3*2^45, population SD=2^45: shape=pi*sqrt(3), scale=mean/(1+1/18).
      expect([...result.moments]).toEqual([2 ** 91]);
      expect(result.fit).toBeDefined();
      expect(result.fit!.every(Number.isFinite)).toBe(true);
      expect(result.fit![0]).toBeCloseTo(5.441398092702653, 5);
      expect(Math.abs(result.fit![1] / 99997689094575.16 - 1)).toBeLessThan(3e-7);
    });
  }
}

for (const split of [false, true]) {
  for (const reversed of [false, true]) {
    it(`B2 calibration preserves nearby large-location subtraction split=${split} reversed=${reversed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const observations: [number, number] = [2 ** 70 + 2 ** 47, 2 ** 70 + 2 ** 48];
      if (reversed) observations.reverse();
      const result = await runMomentProbe(device, 'calibration', observations, split, {
        location: 2 ** 70,
        maximumValueMagnitude: 2 ** 48
      });
      // Sterbenz-exact shifted values 2^47, 2^48; (2^47)^2/2 = 2^93.
      expect([...result.moments]).toEqual([2 ** 93]);
    });
  }
}
