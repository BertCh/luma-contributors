// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSimpleLocalReliefParameterValues,
  GPUSimpleLocalRelief
} from '../../../src/gpu-terrain/relief-visualization/gpu-simple-local-relief';
import {
  getGPUMultiScaleReliefParameterValues,
  GPUMultiScaleRelief
} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {
  GPU_RELIEF_DEFAULT_QUANTUM,
  usesReliefSummedArea
} from '../../../src/gpu-terrain/relief-visualization/relief-visualization-utils';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createElevationBand} from './relief-test-utils';

async function runRadiusOneRelief(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  multiScale = false
): Promise<{relief: number[]; validity: number[]}> {
  const pixelCount = width * height;
  const buffers: Buffer[] = [];
  const reliefBuffer = createOutputBuffer(device, pixelCount);
  const validityBuffer = createOutputBuffer(device, pixelCount);
  buffers.push(reliefBuffer, validityBuffer);
  const settings = new GPUParameterBuffer(device, {
    id: `distant-anchor-settings-${width}`,
    format: 'float32',
    length: 4,
    values: multiScale
      ? getGPUMultiScaleReliefParameterValues()
      : getGPUSimpleLocalReliefParameterValues({radius: 1})
  });
  const graph = new GPUCommandGraph(device, {id: `distant-anchor-${width}`});
  const shared = {
    grid: makeGPURasterMetadata({width, height}),
    elevation: createElevationBand(graph, device, elevation, buffers),
    parameters: settings.importToGraph(graph),
    output: {
      relief: importGraphBuffer(graph, 'relief', reliefBuffer, 'float32', pixelCount),
      outputValidity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', pixelCount)
    }
  };
  if (multiScale) {
    const contributor = new GPUMultiScaleRelief({
      ...shared,
      cellSizeMode: 'uniform',
      resolution: 1,
      featureMinimum: 1,
      featureMaximum: 3,
      scalingFactor: 1
    });
    // i=floor((1-1)/2)=0, n=ceil((3-1)/2)=1. MSRM is z-mean(radius 1),
    // divided by n-i=1. Verify both the requested radii and the actual table declaration.
    expect(contributor.radii.radii).toEqual([0, 1]);
    expect(usesReliefSummedArea(device, pixelCount)).toBe(true);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.some(node => node.id.endsWith('-summed-area-quantize'))).toBe(true);
    for (const node of nodes) graph.add(node);
  } else {
    graph.add(
      new GPUSimpleLocalRelief({
        ...shared,
        maximumRadius: 1
      })
    );
  }
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const relief = await readFloat32(reliefBuffer, pixelCount);
  const validity = await readUint32(validityBuffer, pixelCount);
  compiled.destroy();
  settings.destroy();
  for (const buffer of buffers) buffer.destroy();
  return {relief, validity};
}

it('GPUSimpleLocalRelief keeps a haloed core independent of a distant first valid cell (RT-128)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 5 x 5 ones with a distant first valid cell [0, 0] = 30000 and centre 1 + 2^-11 (exact in
  // float32). The radius-1 window holds eight ones and the centre: z - mean = 2^-11 * 8 / 9 =
  // 1 / 2304 (closed form). The 3 x 3 crop around the centre has the same window.
  const quantum = GPU_RELIEF_DEFAULT_QUANTUM;
  const expected = 1 / 2304;
  const whole = new Float32Array(25).fill(1);
  whole[0] = 30000;
  whole[12] = 1 + 32 * quantum;
  const crop = new Float32Array(9).fill(1);
  crop[4] = 1 + 32 * quantum;
  const wholeResult = await runRadiusOneRelief(device, whole, 5, 5);
  const cropResult = await runRadiusOneRelief(device, crop, 3, 3);
  expect(wholeResult.validity[12]).toBe(1);
  expect(cropResult.validity[4]).toBe(1);
  expect(Number.isFinite(wholeResult.relief[12])).toBe(true);
  expect(Number.isFinite(cropResult.relief[4])).toBe(true);
  // Window samples are multiples of the quantum, so only float32 rounding of the deviation remains.
  const bound = quantum / 2 + 4 * 2 ** -24 * (expected + quantum);
  expect(Math.abs(cropResult.relief[4] - expected)).toBeLessThanOrEqual(bound);
  expect(Math.abs(wholeResult.relief[12] - expected)).toBeLessThanOrEqual(bound);
  expect(wholeResult.relief[12]).toBe(cropResult.relief[4]);
});

it('GPUMultiScaleRelief uses uniform radii zero and one without distant-anchor loss (RT-128)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const quantum = GPU_RELIEF_DEFAULT_QUANTUM;
  const whole = new Float32Array(25).fill(1);
  whole[0] = 30000;
  whole[12] = 1 + 32 * quantum;
  const crop = new Float32Array(9).fill(1);
  crop[4] = 1 + 32 * quantum;
  const wholeResult = await runRadiusOneRelief(device, whole, 5, 5, true);
  const cropResult = await runRadiusOneRelief(device, crop, 3, 3, true);
  // The zero-radius fine filter is z itself; n-i=1. The radius-one coarse filter covers
  // eight ones and z=1+2^-11: MSRM = 2^-11*(1-1/9) = 1/2304 for both domains.
  const expected = 1 / 2304;
  const bound = quantum / 2 + 4 * 2 ** -24 * (expected + quantum);
  for (const [result, center] of [
    [wholeResult, 12],
    [cropResult, 4]
  ] as const) {
    expect(result.validity[center]).toBe(1);
    expect(Number.isFinite(result.relief[center])).toBe(true);
    expect(Math.abs(result.relief[center] - expected)).toBeLessThanOrEqual(bound);
  }
  expect(wholeResult.relief[12]).toBe(cropResult.relief[4]);
});
