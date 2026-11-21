// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {SPACE_TIME_K_CLASS_WGSL} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-wgsl';
import {getSpaceTimeKReference} from './space-time-k-function-reference';
import {GPUSpaceTimeKFunction} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-k-function';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getFunctions(source: string): string {
  const names = [
    'getQuadratProduct',
    'isQuadratProductBelow',
    'isSpaceTimeKClassIncluded',
    'getSpaceTimeKClass',
    'isInside'
  ];
  return names
    .map(name => {
      const start = source.indexOf(`fn ${name}(`);
      if (start < 0) return '';
      let end = source.indexOf('{', start) + 1;
      let depth = 1;
      while (depth > 0 && end < source.length) {
        if (source[end] === '{') depth++;
        if (source[end] === '}') depth--;
        end++;
      }
      return source.slice(start, end);
    })
    .join('\n');
}

it('t1 production class and temporal predicates include the maximum lag under reciprocal lowering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUSpaceTimeKFunction({
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      times: createTransientView(graph, 'times', 'float32', 4),
      parameters: createTransientView(graph, 'parameters', 'float32', 11),
      gridSize: [4, 4],
      distanceClassCount: 4,
      timeClassCount: 3,
      edgeCorrection: 'none',
      output: {
        k: createTransientView(graph, 'k', 'float32', 12),
        temporalK: createTransientView(graph, 'temporal', 'float32', 3)
      }
    }).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const {source, constants} of sources)
      expect(auditWGSLModule(source, constants)).toEqual([]);
    const pairSource = sources.find(({source}) => source.includes('let pairIntensity ='))!.source;
    const classBody = pairSource.match(
      /let pairIntensity = [^;]*;\s*([\s\S]*?)\s*if \(HAS_MARGINAL/
    )!;
    expect(classBody).not.toBeNull();
    const temporalSource = sources.find(({id}) => id.endsWith('-temporal-marginal'))!.source;
    // Model reciprocal lowering AND reassociation: round((15*3)*f32(1/15)) > 3.
    // Literal 15 is the closed last lag, and d=6 is the closed third distance edge.
    // This pair must be included in flattened bin 2*3+2=8, and in the last temporal marginal.
    const body = classBody[1]
      .replaceAll(
        'timeLag / maximumTimeLag * f32(TIME_CLASS_COUNT)',
        `timeLag * f32(TIME_CLASS_COUNT) * ${Math.fround(1 / 15)}`
      )
      .replaceAll('pairDistance / maximumDistance', 'pairDistance * 0.125');
    // Only append isInside from the second module, avoiding duplicate shared helpers.
    const temporalFunction = getFunctions(temporalSource).match(/fn isInside[\s\S]*/)?.[0];
    expect(temporalFunction).toBeDefined();
    const source = `const DISTANCE_CLASS_COUNT:u32=4u; const TIME_CLASS_COUNT:u32=3u;
${getFunctions(pairSource)}
${temporalFunction!.replaceAll('lag / maximumTimeLag * f32(TIME_CLASS_COUNT)', `lag * f32(TIME_CLASS_COUNT) * ${Math.fround(1 / 15)}`)}
@group(0) @binding(0) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(1) fn main() {
 let pairDistance=6.0; let timeLag=15.0; let maximumDistance=8.0; let maximumTimeLag=15.0;
 ${body}
 result[0]=select(0u,1u,inDistance && inLag);
 result[1]=distanceClass*3u+lagClass;
 result[2]=select(0u,1u,isInside(15.0,3.0,20.0,false,15.0));
}`;
    const result = new Uint32Array(3);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: {0: result}});
    expect(Array.from(result)).toEqual([1, 8, 1]);
  } finally {
    device.destroy();
  }
});

it('t1 exact class products keep interior ties and avoid overflowing product normalizers', () => {
  // value*count equals maximum*(class+1) exactly; the tie belongs to that lower class.
  // The last fixture would overflow either naive float32 product.
  for (const [value, maximum, count, expected] of [
    [7, 13, 13, 6],
    [13, 13, 13, 12],
    [0, 13, 13, 0],
    [2 ** 120, 2 ** 123, 64, 7],
    [3 * 2 ** 124, 2 ** 127, 64, 23]
  ]) {
    const source = `${SPACE_TIME_K_CLASS_WGSL}
@group(0) @binding(0) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(1) fn main() {result[0]=getSpaceTimeKClass(${value.toExponential(17)},${maximum.toExponential(17)},${count}u);}`;
    const result = new Uint32Array(1);
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', 1, {0: {0: result}});
    expect(Array.from(result)).toEqual([expected]);
  }
});

it('t1 host reference matches the hand-enumerated closed cumulative bins', () => {
  const reference = getSpaceTimeKReference({
    scene: {positions: [1, 1, 2, 1, 5, 1, 7, 1], times: [1, 3, 9, 16]},
    bounds: [0, 0, 10, 8],
    timeWindow: [0, 20],
    maximumDistance: 8,
    maximumTimeLag: 15,
    distanceClassCount: 4,
    timeClassCount: 3,
    edgeCorrection: 'none'
  });
  // Six unordered pairs: (d,dt)=(1,2),(4,8),(6,15),(3,6),(5,13),(2,7), counted twice.
  const cumulativeCounts = [2, 4, 4, 2, 8, 8, 2, 8, 12, 2, 8, 12];
  expect(reference.pairCounts).toEqual([2, 2, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0]);
  expect(reference.k.length).toBe(12);
  for (const [index, count] of cumulativeCounts.entries()) {
    expect(Number.isFinite(reference.k[index])).toBe(true);
    expect(reference.k[index]).toBeCloseTo((count * 400) / 3, 10);
  }
  expect(reference.temporalK).toEqual([10 / 3, 40 / 3, 20]);
});
