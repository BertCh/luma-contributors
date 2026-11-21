// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  FIXED_POINT_MOMENTS_WGSL,
  FIXED_POINT_WGSL
} from '../../src/analysis-core/kernels/fixed-point-accumulator';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPURasterStreamingAccumulator,
  createGPURasterStreamingAccumulatorEncodingPlan,
  getGPURasterStreamingAccumulatorParameterValues
} from '../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {
  GPURasterStandardizedIndexCalibration,
  createGPURasterStandardizedIndexCalibrationEncodingPlan,
  getGPURasterStandardizedIndexCalibrationParameterValues
} from '../../src/raster-analysis/raster-time-series/gpu-raster-standardized-index-calibration';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../src/testing/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {decodeMomentProbe, type MomentRoute} from './fixed-point-moments.test-utils';

async function runMomentProbe(
  device: Device,
  route: MomentRoute,
  observations: [number, number],
  split: boolean,
  maximumStreamSliceCount?: number
): Promise<number> {
  const chunkSliceCount = split ? 1 : 2;
  const settings = {
    maximumValueMagnitude: Math.max(...observations.map(Math.abs)),
    maximumStreamSliceCount
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
  // Calibration's third three-word statistic starts at word 6. Probe the shared primitive
  // there too, so the test cannot accidentally decode the sum instead of the moment.
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let compiled: CompiledGPUCommandGraph | undefined;
  const inputBuffer = createInputBuffer(
    device,
    Float32Array.from(observations.slice(0, chunkSliceCount))
  );
  const parameterBuffer = createInputBuffer(device, parametersFor(true));
  buffers.push(inputBuffer, parameterBuffer);
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
  try {
    if (route === 'primitive') {
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'fixed-point-moment-probe',
          operation: 'FixedPointMomentProbe',
          invocationCount: 1,
          bindings: [
            {name: 'values', view: values, type: 'f32', access: 'read'},
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'counts', view: counts.view, type: 'u32', access: 'read_write'},
            {name: 'sums', view: sums.view, type: 'u32', access: 'read_write'},
            {name: 'moments', view: moments.view, type: 'u32', access: 'read_write'}
          ],
          declarations: `${FIXED_POINT_WGSL}\n${FIXED_POINT_MOMENTS_WGSL}`,
          body: `if (parameters[parametersOffset] != 0.0) {
  counts[countsOffset] = 0u;
  for (var word = 0u; word < 3u; word++) {
    sums[sumsOffset + word] = 0u;
    moments[momentsOffset + 6u + word] = 0u;
  }
}
let sumUnit = i32(parameters[parametersOffset + 2u]);
let momentUnit = i32(parameters[parametersOffset + 3u]);
for (var slice = 0u; slice < ${chunkSliceCount}u; slice++) {
  let value = values[valuesOffset + slice];
  let count = counts[countsOffset];
  let priorSum = vec3<u32>(sums[sumsOffset], sums[sumsOffset + 1u], sums[sumsOffset + 2u]);
  let increment = getFixedPointMomentIncrement(priorSum, count, value, sumUnit, momentUnit);
  let moment = addFixedPoint(vec3<u32>(moments[momentsOffset + 6u],
    moments[momentsOffset + 7u], moments[momentsOffset + 8u]), increment);
  let sum = addFixedPoint(priorSum, quantizeFixedPoint(value, sumUnit));
  sums[sumsOffset] = sum.x; sums[sumsOffset + 1u] = sum.y; sums[sumsOffset + 2u] = sum.z;
  moments[momentsOffset + 6u] = moment.x;
  moments[momentsOffset + 7u] = moment.y; moments[momentsOffset + 8u] = moment.z;
  counts[countsOffset] = count + 1u;
}`
        })
      );
    } else if (route === 'streaming') {
      const contributor = new GPURasterStreamingAccumulator({
        cellCount: 1,
        chunkSliceCount,
        values,
        parameters,
        state: {counts: counts.view, sums: sums.view, moments: moments.view}
      });
      for (const node of contributor.getCommandNodes(graph)) graph.add(node);
    } else {
      const fitBuffer = createOutputBuffer(device, 3);
      buffers.push(fitBuffer);
      const contributor = new GPURasterStandardizedIndexCalibration({
        cellCount: 1,
        chunkSliceCount,
        values,
        parameters,
        distribution: 'gamma',
        state: {counts: counts.view, moments: moments.view},
        fit: importGraphBuffer(graph, 'fit', fitBuffer, 'float32', 3)
      });
      for (const node of contributor.getCommandNodes(graph)) graph.add(node);
    }
    compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    if (split) {
      expect(decodeMomentProbe(await readUint32(moments.buffer), route, momentUnit)).toBe(0);
      inputBuffer.write(Float32Array.of(observations[1]));
      parameterBuffer.write(parametersFor(false));
      submitGraph(device, compiled, undefined);
    }
    const countWords = await readUint32(counts.buffer);
    expect(countWords[route === 'calibration' ? 2 : 0]).toBe(2);
    return decodeMomentProbe(await readUint32(moments.buffer), route, momentUnit);
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const route of ['primitive', 'streaming', 'calibration'] as const) {
  for (const split of [false, true]) {
    for (const reversed of [false, true]) {
      it(`B2 ${route} retains a wide moment with two-slice headroom split=${split} reversed=${reversed}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const observations: [number, number] = reversed ? [2 ** 70, 2 ** 69] : [2 ** 69, 2 ** 70];
        // Independently M2=(2^70-2^69)^2/2=2^137; the wide bound 2^142 requires unit 2^50.
        const actual = await runMomentProbe(device, route, observations, split, 2);
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBe(2 ** 137);
      });
    }
  }
}

for (const route of ['primitive', 'streaming', 'calibration'] as const) {
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
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBe(0.5);
      });
    }
  }
}

it('SCL-01 primitive scales before squaring a large finite residual', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const actual = await runMomentProbe(device, 'primitive', [2 ** 70, -(2 ** 70)], true);
  // Mean=0, M2 = (2^70)^2 + (-2^70)^2 = 2^141, beyond finite f32 but inside the state.
  expect(Number.isFinite(actual)).toBe(true);
  expect(actual).toBe(2 ** 141);
});
