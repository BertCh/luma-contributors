// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {getNaturalBreaksNodes} from '../../../src/dataframe-analysis/column-classification/natural-breaks-nodes';
import {GPU_CLASS_BREAKS_METHOD_CODES} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {getOrderedFloat32Key} from '../../../src/dataframe-analysis/column-classification/column-classification-shared';
import {
  createKernelAuditContext,
  recordShaderModules,
  type RecordedShaderModule
} from '../../analysis-core/kernel-audit/kernel-audit-context';

function createModules(classes: number, bins: number): RecordedShaderModule[] {
  const context = createKernelAuditContext('natural-review');
  const {graph, view} = context;
  try {
    const modules: RecordedShaderModule[] = [];
    const nodes = getNaturalBreaksNodes(graph, {
      id: 'review',
      operation: 'review',
      values: view('float32', 4),
      state: view('uint32', 6),
      extremes: view('uint32', 4),
      edges: view('float32', classes + 1),
      edgeState: view('uint32', 1),
      maximumClassCount: classes,
      binCount: bins
    });
    recordShaderModules(nodes, module => modules.push(module));
    return modules;
  } finally {
    context.device.destroy();
  }
}

/** Executes the actual generated independent-row WGSL on the CPU; no GPU/device is exercised. */
function runModule(
  module: RecordedShaderModule,
  rows: number,
  buffers: Record<string, Float32Array | Uint32Array>
): void {
  // The interpreter cannot index a vec4 in a uniform array. These fixtures use zero view
  // offsets and fixed invocation counts, so substitute only that transport accessor.
  const source = module.source.replace(
    /fn getChunkParameter\(word: u32\) -> u32 \{[^}]+\}/,
    `fn getChunkParameter(word: u32) -> u32 { return select(0u, ${rows}u, word == 1u || word == 2u); }`
  );
  const reflection = new WgslReflect(source);
  const bindings: Record<number, Float32Array | Uint32Array> = {};
  for (const resource of reflection.storage) {
    const buffer = buffers[resource.name];
    expect(buffer, resource.name).toBeDefined();
    bindings[resource.binding] = buffer;
  }
  const parameters = new Uint32Array(64);
  parameters.set([0, rows, rows, 0]);
  for (const uniform of reflection.uniforms) bindings[uniform.binding] = parameters;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
    'main',
    1,
    {0: bindings},
    {constants: module.constants}
  );
}

it('B1: approximate refinement membership merges signed zeros while retaining raw extrema', () => {
  const module = createModules(4, 4).find(module => module.nodeId.endsWith('-extents'))!;
  expect(module).toBeDefined();
  const extents = new Uint32Array(12);
  extents.fill(0xffffffff, 0, 4);
  runModule(module, 4, {
    values: Float32Array.of(-2, -0, +0, 2),
    state: Uint32Array.of(GPU_CLASS_BREAKS_METHOD_CODES['natural-breaks'], 4, 0, 0, 0, 0),
    edges: Float32Array.of(-2, -0, +0, 2, 2),
    extents
  });
  // Both zero cuts compare equal: neither zero can occupy the class between them.
  expect(Array.from(extents.subarray(8))).toEqual([1, 0, 2, 1]);
  expect(extents[2]).toBe(getOrderedFloat32Key(-0));
  expect(extents[6]).toBe(getOrderedFloat32Key(+0));
});

it('B2: the bulk histogram preserves distinct normal floats beside a huge outlier', () => {
  const modules = createModules(3, 16);
  const scale = modules.find(module => module.nodeId.endsWith('-coordinate-scale'))!;
  const histogramModule = modules.find(module => module.nodeId.endsWith('-histogram'))!;
  const tiny = 2 ** -90;
  const coordinateScale = new Float32Array(4);
  const state = Uint32Array.of(GPU_CLASS_BREAKS_METHOD_CODES['natural-breaks'], 3, 0, 0, 0, 0);
  const extremes = Uint32Array.of(getOrderedFloat32Key(tiny), getOrderedFloat32Key(2 ** 70), 4, 4);
  runModule(scale, 4, {state, extremes, coordinateScale});
  // The second coarse pass narrows its extrema to the three-row bulk, on the same graph.
  extremes[1] = getOrderedFloat32Key(2 * tiny);
  const histogram = new Uint32Array(80);
  runModule(histogramModule, 4, {
    state,
    extremes,
    coordinateScale,
    histogram,
    values: Float32Array.of(tiny, tiny, 2 * tiny, 2 ** 70)
  });
  // Hand count: two minima in bin 0, one maximum in bin 15; outlier excluded by raw keys.
  expect(Array.from(histogram.subarray(0, 16))).toEqual([
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
  ]);
});

it('B2: a local refinement plan converts saved widths from the preceding scale', () => {
  const module = createModules(3, 16).find(module => module.nodeId.endsWith('-refine-0-plan'))!;
  const low = 2 ** -30;
  const high = 2 ** -29;
  const plan = new Float32Array(16);
  const breakState = new Uint32Array(12);
  const widths = new Float32Array(breakState.buffer);
  // Previous solve used 2^-64 coordinates. Its physical width is 2^-26, much larger than
  // this new extent (2^-30). Conversion must cap to the local extent before multiplication.
  widths[1] = 2 ** -90;
  runModule(module, 2, {
    coordinateScale: Float32Array.of(0, 2 ** -64, 0, 0),
    state: Uint32Array.of(GPU_CLASS_BREAKS_METHOD_CODES['natural-breaks'], 3, 0, 1, 0, 0),
    edges: Float32Array.of(low, high, 2 ** 70, 2 ** 70),
    extents: Uint32Array.of(
      getOrderedFloat32Key(low),
      getOrderedFloat32Key(high),
      getOrderedFloat32Key(2 ** 70),
      getOrderedFloat32Key(low),
      getOrderedFloat32Key(high),
      getOrderedFloat32Key(2 ** 70),
      2,
      1,
      1
    ),
    breakState,
    plan
  });
  expect(plan.every(Number.isFinite)).toBe(true);
  expect(plan[4]).toBe(high);
  expect(plan[5]).toBe(2 ** -39); // span 2^-30 / 512 cells per half window
  expect(plan[7]).toBe(1);
});
