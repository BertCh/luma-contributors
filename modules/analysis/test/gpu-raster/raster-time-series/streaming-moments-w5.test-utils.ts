// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {ScalarData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../../src/testing/gpu-contributor-test-utils';
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

// Execute production kernel text, with instance-local fixes for wgsl_reflect 1.2.3
// interpreter omissions. These match the core primitive's test shims: ldexp, unsigned vector
// storage, and unsigned right shift. This remains a host check, not GPU qualification.
export function executeKernel(
  source: string,
  constants: Readonly<Record<string, number>>,
  fields: Record<string, Float32Array | Uint32Array>
): void {
  const reflection = new WgslReflect(source);
  // The interpreter leaks helper parameter names into its caller's scope; use a distinct
  // name for the equivalent load/store parameter. Ignore WGSL phony assignments as well.
  const executable = source
    .replace(/^\s*_ = .*;$/gm, '')
    .replace(/fn (?:load|store)(?:Sums|Moments)\([^}]+\}/g, helper =>
      helper.replace(/\bentry\b/g, 'stateEntry')
    );
  const execution = new WgslExec(
    new WgslParser().parse(`${executable.replaceAll('ldexp(', 'testLdexp(')}
fn testLdexp(fraction: f32, exponent: i32) -> f32 { return fraction * exp2(f32(exponent)); }`)
  );
  const getTypeInfo = execution.getTypeInfo.bind(execution);
  execution.getTypeInfo = type => {
    const information = getTypeInfo(type);
    const format = (information as {format?: {name: string}} | null)?.format;
    if ((information?.name === 'vec2' || information?.name === 'vec3') && format?.name === 'u32') {
      information.name += 'u';
    }
    return information;
  };
  const evaluateBinary = execution._evalBinaryOp.bind(execution);
  execution._evalBinaryOp = (node, context) => {
    if (node.operator === '>>') {
      const left = execution.evalExpression(node.left, context);
      const right = execution.evalExpression(node.right, context);
      if (
        left instanceof ScalarData &&
        right instanceof ScalarData &&
        left.typeInfo.name === 'u32'
      ) {
        return new ScalarData(left.value >>> right.value, left.typeInfo);
      }
    }
    return evaluateBinary(node, context);
  };
  execution.dispatchWorkgroups(
    'main',
    1,
    {0: Object.fromEntries(reflection.storage.map(field => [field.binding, fields[field.name]]))},
    {constants}
  );
}

export type MomentRoute = 'streaming' | 'calibration';

export function decodeMomentReadback(
  words: ArrayLike<number>,
  route: MomentRoute,
  exponent: number
): Float64Array {
  const offset = route === 'streaming' ? 0 : 6;
  expect(words).toHaveLength(route === 'streaming' ? 3 : 9);
  const decoded = getGPURasterFixedPointValues(
    Uint32Array.from(Array.from(words).slice(offset, offset + 3)),
    exponent
  );
  expect(decoded).toHaveLength(1);
  return decoded;
}

export function runHostMomentProbe(options: {
  route: MomentRoute;
  observations: [number, number];
  split: boolean;
  location?: number;
  maximumValueMagnitude?: number;
  maximumStreamSliceCount?: number;
  distribution?: 'gamma' | 'fisk';
}) {
  const {route, observations, split, location = 0} = options;
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const chunkSliceCount = split ? 1 : 2;
    const settings = {
      maximumValueMagnitude:
        options.maximumValueMagnitude ??
        Math.max(...observations.map(value => Math.abs(value - location))),
      maximumStreamSliceCount: options.maximumStreamSliceCount,
      location
    };
    const streamPlan = createGPURasterStreamingAccumulatorEncodingPlan(settings, chunkSliceCount);
    const calibrationPlan = createGPURasterStandardizedIndexCalibrationEncodingPlan(
      settings,
      chunkSliceCount
    );
    const parametersFor = (reset: boolean) =>
      route === 'streaming'
        ? getGPURasterStreamingAccumulatorParameterValues(streamPlan, {reset})
        : getGPURasterStandardizedIndexCalibrationParameterValues(calibrationPlan, {reset});
    const fields = {
      values: Float32Array.from(observations.slice(0, chunkSliceCount)),
      mask: new Uint32Array(chunkSliceCount).fill(1),
      parameters: parametersFor(true),
      counts: new Uint32Array(route === 'streaming' ? 1 : 3),
      sums: new Uint32Array(3),
      moments: new Uint32Array(route === 'streaming' ? 3 : 9),
      fit: new Float32Array(3)
    };
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `moment-${serial++}`, format, length);
    const values = view('float32', chunkSliceCount);
    const mask = view('uint32', chunkSliceCount);
    const parameters = view('float32', fields.parameters.length);
    const counts = view('uint32', fields.counts.length);
    const moments = view('uint32', fields.moments.length);
    const contributor =
      route === 'streaming'
        ? new GPURasterStreamingAccumulator({
            cellCount: 1,
            chunkSliceCount,
            values,
            mask,
            parameters,
            state: {counts, moments, sums: view('uint32', 3)}
          })
        : new GPURasterStandardizedIndexCalibration({
            cellCount: 1,
            chunkSliceCount,
            distribution: options.distribution ?? 'gamma',
            values,
            mask,
            parameters,
            state: {counts, moments},
            fit: view('float32', 3)
          });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    for (const kernel of kernels) executeKernel(kernel.source, kernel.constants, fields);
    if (split) {
      expect([
        ...decodeMomentReadback(fields.moments, route, calibrationPlan.momentUnitExponent)
      ]).toEqual([0]);
      fields.values.set([observations[1]]);
      fields.parameters.set(parametersFor(false));
      for (const kernel of kernels) executeKernel(kernel.source, kernel.constants, fields);
    }
    expect([...fields.counts]).toEqual(route === 'streaming' ? [2] : [2, 0, 2]);
    const exponent =
      route === 'streaming' ? streamPlan.momentUnitExponent : calibrationPlan.momentUnitExponent;
    return {
      fields,
      momentUnitExponent: exponent,
      sumUnitExponent: calibrationPlan.sumUnitExponent,
      moments: decodeMomentReadback(fields.moments, route, exponent)
    };
  } finally {
    device.destroy();
  }
}
