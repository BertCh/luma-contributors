// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPURipley} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley';
import {GPUSpaceTimeKFunction} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-k-function';
import {SPACE_TIME_K_EDGE_WEIGHT_WGSL} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function evaluateWeights(source: string, functionName: string): Float32Array {
  const functions = [
    ...source.matchAll(
      /fn (?:getSideAngle|getArcOverlap|getIsotropicWeight|getSpaceTimeSpatialWeight)\([^]*?\n\}/g
    )
  ]
    .map(match => match[0])
    .join('\n');
  const output = new Float32Array(4);
  const constants = [...source.matchAll(/const (?:PI|HALF_PI|WEIGHT_CAP): f32 = [^;]+;/g)].map(
    match => match[0]
  );
  expect(constants).toHaveLength(3);
  new WgslExec(
    new WgslParser().parse(`
${constants.join('\n')}
${functions}
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1) fn main() {
  let negativeZero = bitcast<f32>(0x80000000u);
  output[0] = ${functionName}(5.0, 5.0, 5.0, 5.0, 1.0);
  output[1] = ${functionName}(negativeZero, 5.0, 10.0, 5.0, 1.0);
  output[2] = ${functionName}(negativeZero, negativeZero, 10.0, 10.0, 1.0);
  output[3] = ${functionName}(0.0, 0.0, 10.0, 10.0, 100.0);
}`)
  ).dispatchWorkgroups('main', [1, 1, 1], {0: {0: output}});
  return output;
}

it('W2-01 both consumers preserve interior, signed-zero side, corner and capped weights', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 2);
  const parameters = createTransientView(graph, 'parameters', 'float32', 11);
  const ripley = new GPURipley({
    positions,
    parameters,
    gridSize: [2, 2],
    radiusCount: 1,
    output: {k: createTransientView(graph, 'ripley-k', 'float32', 1)}
  });
  const spaceTime = new GPUSpaceTimeKFunction({
    positions,
    parameters,
    times: createTransientView(graph, 'times', 'float32', 2),
    gridSize: [2, 2],
    distanceClassCount: 1,
    timeClassCount: 1,
    output: {k: createTransientView(graph, 'space-time-k', 'float32', 1)}
  });
  try {
    for (const [contributor, functionName] of [
      [ripley, 'getIsotropicWeight'],
      [spaceTime, 'getSpaceTimeSpatialWeight']
    ] as const) {
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      const source = kernels.find(kernel => kernel.source.includes(`fn ${functionName}(`))!.source;
      if (contributor === spaceTime) expect(source).toContain(SPACE_TIME_K_EDGE_WEIGHT_WGSL);
      const output = evaluateWeights(source, functionName);
      // A full circle, half circle, and quarter circle give reciprocal fractions 1,2,4.
      for (const [index, expected] of [1, 2, 4, 100].entries()) {
        expect(Number.isFinite(output[index])).toBe(true);
        expect(output[index]).toBeCloseTo(expected, 5);
      }
      // B3 mutation control: changing either consumer's generated cap must reach execution.
      const changedCap = source.replace(
        /const WEIGHT_CAP: f32 = [^;]+;/,
        'const WEIGHT_CAP: f32 = 50.0;'
      );
      expect(evaluateWeights(changedCap, functionName)[3]).toBe(50);
    }
  } finally {
    device.destroy();
  }
});
