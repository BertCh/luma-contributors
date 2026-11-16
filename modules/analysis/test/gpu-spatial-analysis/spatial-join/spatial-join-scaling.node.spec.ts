// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {SCALED_QUOTIENT_WGSL} from '../../../src/analysis-core/kernels/scaled-quotient-wgsl';
import {getSpatialJoinCandidateNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-candidates';
import {
  QUANTIZE_AXIS_WGSL,
  getSortedFeatureBVHNodes,
  SPATIAL_JOIN_WGSL_HELPERS
} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getReservedWGSLWords} from './wgsl-reserved-words';
import {getChunkDispatches} from './chunk-dispatch-test-utils';

/** The interpreter lacks frexp/ldexp. Supply only their normal-scalar IEEE builtin semantics. */
function createScaleProbe(source: string): WgslExec {
  new WgslParser().parse(source);
  const executableSource = source
    .replace(/^\s*_ = &\w+;$/gm, '')
    .replace(/\bfrexp\(/g, 'probeFrexp(')
    .replace(/\bldexp\(/g, 'probeLdexp(');
  const probeSource = `${executableSource}
struct ProbeFraction { fract: f32, exp: i32 }
fn probeFrexp(value: f32) -> ProbeFraction {
  if (value == 0.0) { return ProbeFraction(0.0, 0); }
  let bits = bitcast<u32>(value);
  return ProbeFraction(bitcast<f32>((bits & 0x807fffffu) | 0x3f000000u), i32((bits >> 23u) & 255u) - 126);
}
fn probeLdexp(value: f32, exponent: i32) -> f32 { return value * exp2(f32(exponent)); }
`;
  expect(getReservedWGSLWords(probeSource)).toEqual([]);
  return new WgslExec(new WgslParser().parse(probeSource));
}

it('scene reduction reaches both barriers uniformly in exactly one complete workgroup', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const minima = createTransientView(graph, 'minima', 'float32x2', 257);
    const maxima = createTransientView(graph, 'maxima', 'float32x2', 257);
    const {nodes} = getSortedFeatureBVHNodes(
      graph,
      'uniform-sort',
      'UniformSort',
      minima,
      maxima,
      512,
      true
    );
    const reduction = nodes.filter(node => node.id === 'uniform-sort-sort-scene-bounds');
    expect(getChunkDispatches(device, reduction)).toEqual([[1, 1, 1]]);
    const kernel = getGPUCommandNodeKernelSources(reduction)[0];
    const entry = kernel.source.slice(kernel.source.indexOf('fn main'));
    expect(getReservedWGSLWords(kernel.source)).toEqual([]);
    expect(kernel.constants['INVOCATION_COUNT']).toBe(256);
    expect(entry.match(/workgroupBarrier\(\)/g)).toHaveLength(2);
    // The shared dispatch packer may reject padding workgroups uniformly. An invocation guard
    // or any other early return would leave only part of this workgroup reaching its barriers.
    const reductionBody = entry.replace(/if \(workgroupIndex >= \d+u\) \{ return; \}/, '');
    expect(reductionBody).not.toMatch(/\breturn\b/);
    expect(entry).toContain('stride = stride >> 1u');
  } finally {
    device.destroy();
  }
});

it('quantizes large and small finite extents with the shared exponent-scaled quotient', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const values = Float32Array.from([
      2 ** 126,
      0,
      2 ** 127,
      2 ** -102,
      0,
      2 ** -100,
      3 * 2 ** -102,
      0,
      2 ** -100,
      0,
      -(2 ** 126),
      2 ** 126
    ]);
    const cells = new Uint32Array(4);
    const node = createWGSLKernelNode(graph, {
      id: 'scaled-quantization',
      operation: 'SpatialJoinQuantization',
      invocationCount: 4,
      bindings: [
        {
          name: 'values',
          view: createTransientView(graph, 'values', 'float32', 12),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'cells',
          view: createTransientView(graph, 'cells', 'uint32', 4),
          type: 'u32',
          access: 'read_write'
        }
      ],
      declarations: `${SPATIAL_JOIN_WGSL_HELPERS}\n${QUANTIZE_AXIS_WGSL}`,
      body: `let row = valuesOffset + index * 3u;
  cells[cellsOffset + index] = quantizeAxis(values[row], values[row + 1u], values[row + 2u]);`
    });
    const kernel = getGPUCommandNodeKernelSources([node])[0];
    expect(getReservedWGSLWords(kernel.source)).toEqual([]);
    expect(kernel.source).toContain(SCALED_QUOTIENT_WGSL.trim());
    expect(kernel.source).toContain('getScaledQuotient(offset, extent, 1.0, 0)');
    createScaleProbe(kernel.source).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: {0: values, 1: cells}},
      {constants: kernel.constants}
    );
    // Closed-form halves, quarters and three-quarters of a 65535-cell extent.
    expect(Array.from(cells)).toEqual([32768, 16384, 49151, 32768]);
    const reciprocal = Math.fround(1 / 2 ** 127);
    expect(reciprocal).toBeLessThan(2 ** -126); // Metal flushes this reciprocal to zero.
  } finally {
    device.destroy();
  }
});

for (const radius of [2 ** 127, 2 ** -100]) {
  it(`expands geodesic candidate bounds with a half-radian radius at scale ${radius}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const boxes = (name: string, length: number) =>
        createTransientView(graph, name, 'float32x2', length);
      const nodes = getSpatialJoinCandidateNodes(graph, {
        id: 'scaled-geodesic-candidates',
        leftCount: 1,
        rightCount: 2,
        leftMinima: boxes('left-minima', 1),
        leftMaxima: boxes('left-maxima', 1),
        tree: {
          nodeMinima: boxes('node-minima', 3),
          nodeMaxima: boxes('node-maxima', 3),
          leafIds: createTransientView(graph, 'leaves', 'uint32', 2),
          internalNodeCount: 1
        },
        candidateCapacity: 2,
        sphereRadius: radius,
        margin: radius / 2,
        state: createTransientView(graph, 'state', 'uint32', 4),
        candidatePairs: createTransientView(graph, 'pairs', 'uint32x2', 2)
      });
      const kernel = getGPUCommandNodeKernelSources(nodes).find(source =>
        source.id.endsWith('-count-candidates')
      )!;
      expect(getReservedWGSLWords(kernel.source)).toEqual([]);
      expect(kernel.source).toContain('getScaledQuotient(margin, SPHERE_RADIUS, 1.0, 0)');
      const arrays: Record<string, Float32Array | Uint32Array> = {
        leftMinima: new Float32Array(2),
        leftMaxima: new Float32Array(2),
        nodeMinima: Float32Array.from([20, 0, 20, 0, 40, 0]),
        nodeMaxima: Float32Array.from([40, 0, 20, 0, 40, 0]),
        leafIds: Uint32Array.from([0, 1]),
        leftCounts: new Uint32Array(1)
      };
      const bindings = Object.fromEntries(
        new WgslReflect(kernel.source).storage.map(binding => [
          binding.binding,
          arrays[binding.name]
        ])
      );
      createScaleProbe(kernel.source).dispatchWorkgroups(
        'main',
        [1, 1, 1],
        {0: bindings},
        {constants: kernel.constants}
      );
      // Half a radian is 28.65 degrees: the 20-degree leaf overlaps, the 40-degree leaf does not.
      expect(Array.from(arrays['leftCounts'])).toEqual([1]);
    } finally {
      device.destroy();
    }
  });
}
