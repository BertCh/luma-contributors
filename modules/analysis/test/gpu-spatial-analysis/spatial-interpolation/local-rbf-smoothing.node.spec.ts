// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {SCALED_QUOTIENT_WGSL} from '../../../src/analysis-core/kernels/scaled-quotient-wgsl';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

function createContributor(
  graph: GPUCommandGraph,
  kernel: 'linear' | 'thin-plate-spline' | 'cubic' | 'quintic',
  lengths: number[] = [1]
) {
  const queryChunks = lengths.map((length, index) =>
    createTransientView(graph, `queries-${index}`, 'float32x2', length)
  );
  // Co-bound fields share one buffer, with disjoint aligned binding ranges.
  const outputChunks = lengths.map((length, index) => {
    const statusOffset = Math.ceil((length * 16) / 256) * 256;
    const backing = createTransientView(
      graph,
      `output-${index}`,
      'uint32',
      statusOffset / 4 + length
    );
    return {
      values: graph.createDataView(backing.buffer, {format: 'float32x4', length}),
      status: graph.createDataView(backing.buffer, {
        format: 'uint32',
        length,
        byteOffset: statusOffset
      })
    };
  });
  return new GPULocalRBF({
    positions: createTransientView(graph, 'positions', 'float32x2', 9),
    values: createTransientView(graph, 'values', 'float32', 9),
    sampleSmoothing: createTransientView(graph, 'smoothing', 'float32', 9),
    queries: createVectorView('queries', 'float32x2', queryChunks),
    neighborCount: 9,
    kernel,
    output: {
      values: createVectorView(
        'results',
        'float32x4',
        outputChunks.map(chunk => chunk.values)
      ),
      status: createVectorView(
        'statuses',
        'uint32',
        outputChunks.map(chunk => chunk.status)
      )
    }
  });
}

it('exposes the overflowing reciprocal that makes reassociated runtime smoothing unsafe', () => {
  const coordinateScale = Math.fround(1e-9);
  const reciprocalPower = Math.fround((1 / coordinateScale) ** 5);
  expect(reciprocalPower).toBe(Infinity);
  expect(Math.fround(0 * reciprocalPower)).toBeNaN();
  expect(Math.fround(Math.fround(1e-36) * reciprocalPower)).toBe(Infinity);
  // Binary64 forms the complete quotient without overflowing; the f32 result is normal.
  expect(Math.fround(Math.fround(1e-36) / coordinateScale ** 5)).toBeGreaterThan(1e8);
});

for (const [kernel, degree] of [
  ['linear', 1],
  ['thin-plate-spline', 2],
  ['cubic', 3],
  ['quintic', 5]
] as const) {
  it(`normalizes runtime ${kernel} smoothing with bounded mantissas and integer exponents`, () => {
    const graph = new GPUCommandGraph(device);
    const nodes = createContributor(graph, kernel).getCommandNodes(graph);
    const {source} = getGPUCommandNodeKernelSources(nodes)[0];
    // An ordinary CPU interpreter cannot reproduce Metal reassociation. This structural guard
    // rejects the former staged divisions even when their sequential CPU values happen to pass.
    const smoothingFunction = source.slice(
      source.indexOf('fn scaleSmoothing('),
      source.indexOf('fn polynomial(')
    );
    expect(source).toContain(SCALED_QUOTIENT_WGSL);
    expect(smoothingFunction).toContain('frexp(coordinateScale)');
    expect(smoothingFunction).toContain(`pow(divisor.fract, ${degree}.0)`);
    expect(smoothingFunction).toContain(`-${degree} * divisor.exp`);
    expect(smoothingFunction).not.toMatch(/\/=?\s*coordinateScale|pow\(coordinateScale/);
    expectValidQueryShaderLiterals(source);
    const words =
      source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
    expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);

    const inputs = Float32Array.from([0, 1e-9, 1e-36, 1e-9, 1e-35, 1e-7, 1e30, 1e9, 0, 1e-20]);
    const actual = new Float32Array(inputs.length / 2);
    // wgsl_reflect leaves frexp unimplemented. For normal inputs, supply exact IEEE bit
    // decomposition in the test harness; the independent oracle below uses binary64 powers.
    const executableSource = source
      .replace(/\bfrexp\(/g, 'probeFrexp(')
      .replace(/\bldexp\(/g, 'probeLdexp(');
    const probe = `${executableSource}
struct ProbeFloatParts { fract: f32, exp: i32 }
fn probeFrexp(probeValue: f32) -> ProbeFloatParts {
  let probeBits = bitcast<u32>(probeValue);
  return ProbeFloatParts(bitcast<f32>((probeBits & 0x807fffffu) | 0x3f000000u), i32((probeBits >> 23u) & 255u) - 126);
}
fn probeLdexp(probeFraction: f32, probeExponent: i32) -> f32 {
  return probeFraction * pow(2.0, f32(probeExponent));
}
@group(1) @binding(0) var<storage, read> smoothingProbeInputs: array<f32>;
@group(1) @binding(1) var<storage, read_write> smoothingProbeOutputs: array<f32>;
@compute @workgroup_size(1) fn probeSmoothing(@builtin(global_invocation_id) invocation: vec3u) {
  smoothingProbeOutputs[invocation.x] = scaleSmoothing(smoothingProbeInputs[invocation.x * 2u], smoothingProbeInputs[invocation.x * 2u + 1u]);
}`;
    expectValidQueryShaderLiterals(probe);
    const probeWords =
      probe.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
    expect(probeWords.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
    new WgslExec(new WgslParser().parse(probe)).dispatchWorkgroups(
      'probeSmoothing',
      [actual.length, 1, 1],
      {
        1: {0: inputs, 1: actual}
      }
    );
    for (let row = 0; row < actual.length; row++) {
      // Independent binary64 power/quotient on the uploaded f32 inputs.
      const expected = inputs[row * 2] / inputs[row * 2 + 1] ** degree;
      if (expected === 0) expect(actual[row]).toBe(0);
      else expect(Math.abs(actual[row] / expected - 1)).toBeLessThan(2e-6);
    }
  });
}

it('keeps runtime sample-smoothing shaders and pipelines constant across uneven chunks', () => {
  const pipelines = new Set<string>();
  const sources = new Set<string>();
  for (const chunkCount of [1, 3, 9]) {
    const graph = new GPUCommandGraph(device);
    const lengths = Array.from({length: chunkCount}, (_, index) => [31, 32, 33][index % 3]);
    const nodes = createContributor(graph, 'quintic', lengths).getCommandNodes(graph);
    expect(() => graph.add(nodes)).not.toThrow();
    expect(nodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
      lengths.map(length => Math.ceil(length / 32))
    );
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      pipelines.add(key);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
  }
  expect(pipelines.size).toBe(1);
  expect(sources.size).toBe(1);
});
