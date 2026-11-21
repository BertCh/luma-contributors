// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPUPointPatternIndices} from '../../../src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from '../../../src/analysis-core/kernels/incomplete-gamma-wgsl';
import {
  ERFC_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_NAN_BITS_WGSL
} from '../../../src/analysis-core/kernels/wgsl-common';
import {EXACT_ORIENTATION_WGSL} from '../../../src/analysis-core/kernels/exact-orientation-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getFunction(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  if (start < 0) throw new Error(`Missing WGSL function ${name}`);
  const body = source.indexOf('{', start);
  let depth = 1;
  let end = body + 1;
  while (depth && end < source.length) {
    if (source[end] === '{') depth++;
    if (source[end] === '}') depth--;
    end++;
  }
  return source.slice(start, end);
}

it('shared and point-index gamma kernels are compared against independent SciPy tails', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `numerics-${serial++}`, format, length);
    const nodes = new GPUPointPatternIndices({
      positions: view('float32x2', 4),
      parameters: view('float32', 8),
      gridSize: [4, 4],
      quadratGrid: [16, 16],
      output: {quadratStatistics: view('float32', 8)}
    }).getCommandNodes(graph);
    const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.id.endsWith('-quadrat-statistics')
    )!.source;
    const functions = [
      'getNormalPValue',
      'getLogGamma',
      'getGammaPrefix',
      'getGammaTails',
      'getChiSquarePValue'
    ]
      .map(name => getFunction(source, name))
      .join('\n');
    // /usr/bin/python3: scipy.stats.chi2.sf(x,df), scipy.stats.norm.sf(z).
    for (const [degreesOfFreedom, value, expected, z, normalExpected] of [
      [1, 9, 0.00269979606326019, 0, 0.5],
      [4, 20, 0.0004993992273873336, 3, 0.001349898031630093],
      [10000, 10000, 0.49811936596618267, 8, 6.22096057427174e-16],
      [10000, 11000, 3.61832955809647e-12, 3, 0.001349898031630093],
      [65535, 67000, 2.9268476745227096e-5, 0, 0.5]
    ]) {
      const result = new Float32Array(4);
      const probe = `${ERFC_VALUE_WGSL}\n${GET_NAN_WGSL}\n${IS_NAN_BITS_WGSL}\n${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}\n${functions}
@group(0) @binding(0) var<storage,read_write> result: array<f32>;
@compute @workgroup_size(1) fn main() {
 result[0]=getChiSquarePValue(${value}.0,${degreesOfFreedom}.0,1u);
 result[1]=getChiSquareSurvivalValue(${value}.0,${degreesOfFreedom}.0);
 result[2]=getNormalPValue(${z}.0,1u); result[3]=getNormalSurvivalValue(${z}.0);
}`;
      new WgslExec(new WgslParser().parse(probe)).dispatchWorkgroups('main', [1, 1, 1], {
        0: {0: result}
      });
      expect(Array.from(result).every(Number.isFinite)).toBe(true);
      expect(Math.abs(result[0] / expected - 1)).toBeLessThan(0.002);
      // Shared WH approximation at large df has weaker tail precision than the centered
      // exact-series route. It is not a drop-in replacement for the public indices result.
      expect(Math.abs(result[1] / expected - 1)).toBeLessThan(0.1);
      for (const index of [2, 3])
        expect(Math.abs(result[index] / normalExpected - 1)).toBeLessThan(1e-5);
    }
  } finally {
    device.destroy();
  }
});

it('shared orientation preserves independent near-collinear determinant signs', () => {
  // Exact integer determinant of (8192,8191),(8191,8190) is -1, while both
  // separately rounded float32 products are 67092480. Shared orientSign returns only
  // a sign, which cannot replace the metric determinant used in intersection distances.
  for (const [second, third, expected] of [
    [[8192, 8191], [8191, 8190], -1],
    [[8191, 8190], [8192, 8191], 1],
    [[8192, 8192], [8191, 8191], 0]
  ] as const) {
    const result = new Float32Array(1);
    const source = `${EXACT_ORIENTATION_WGSL}
@group(0) @binding(0) var<storage,read_write> result: array<f32>;
@compute @workgroup_size(1) fn main() {
 let first=vec2f(0.0); let second=vec2f(${second[0]}.0,${second[1]}.0); let third=vec2f(${third[0]}.0,${third[1]}.0);
 result[0]=f32(orientSign(first,second,third));
}`;
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
      0: {0: result}
    });
    expect(Array.from(result)).toEqual([expected]);
  }
});
