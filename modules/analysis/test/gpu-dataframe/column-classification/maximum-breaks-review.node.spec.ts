// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {GPU_CLASS_BREAKS_METHOD_CODES} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {
  createKernelAuditContext,
  recordShaderModules,
  type RecordedShaderModule
} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {computeClassBreaksOracle, countClassesOracle} from './class-breaks-oracle';

it.each([
  {bits: [0x08800000, 0x08800001, 0x08800003], keys: [0xffffffff, 0xfffeffff, 0xfffdffff]},
  {bits: [0x88800003, 0x88800001, 0x88800000], keys: [0xffffffff, 0xfffdffff, 0xfffeffff]},
  {bits: [0x007fffff, 0x00800000, 0x00800001], keys: [0xffffffff, 0xfffffffe, 0xfffffffe]},
  {bits: [0x80ffffff, 0x00000001, 0x00ffffff], keys: [0xffffffff, 0xfeffffff, 0xff000001]},
  {bits: [0x0bffffff, 0x0c000000, 0x0c000001], keys: [0xffffffff, 0xffbfffff, 0xff7fffff]},
  // Widths 1.125*2^128 and 1.5*2^128; last row repeats so it is not another gap.
  {bits: [0xff000000, 0x7f200000, 0x7f200000], keys: [0xffffffff, 0x806fffff, 0xffffffff]},
  {bits: [0xff600000, 0x7f200000, 0x7f200000], keys: [0xffffffff, 0x803fffff, 0xffffffff]}
])('review B1 preserves tiny gap keys when subtraction results flush: %j', fixture => {
  const context = createKernelAuditContext('tiny-gap-review');
  try {
    const {graph, view} = context;
    const modules: RecordedShaderModule[] = [];
    recordShaderModules(
      [
        ...new GPUClassBreaks({
          values: view('float32', 3),
          parameters: view('float32', 11),
          maximumClassCount: 2,
          methods: ['maximum-breaks'],
          output: {breaks: view('float32', 3), classCount: view('uint32', 1)}
        }).getCommandNodes(graph)
      ],
      module => modules.push(module)
    );
    const module = modules.find(module => module.nodeId.endsWith('-maximum-breaks-gaps'))!;
    expect(module).toBeDefined();
    // Execute the actual kernel, modeling result FTZ only at its legacy subtraction site.
    // This is host interpretation, not evidence of device arithmetic.
    const source =
      module.source
        .replace(
          /fn getChunkParameter\(word: u32\) -> u32 \{[^}]+\}/,
          'fn getChunkParameter(word: u32) -> u32 { return select(0u, 3u, word == 1u || word == 2u); }'
        )
        .replace(/_ = [^;]+;/g, '')
        .replaceAll('0x1p126', '8.507059173023462e37')
        .replace('upper * 0.5 - lower * 0.5', 'forceFloat32(upper - lower) * 0.5')
        .replace('bitcast<u32>(upper - lower)', 'flushSubnormalGap(upper - lower)') +
      `
fn flushSubnormalGap(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits, 0u, (bits & 0x7f800000u) == 0u);
}
fn forceFloat32(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value));
}`;
    const gapKeys = new Uint32Array(3);
    const sortedKeys = Uint32Array.from(fixture.bits, bits =>
      (bits & 0x80000000) !== 0 ? ~bits : bits ^ 0x80000000
    );
    const buffers: Record<string, Uint32Array> = {
      state: Uint32Array.of(GPU_CLASS_BREAKS_METHOD_CODES['maximum-breaks'], 2, 0, 0, 0, 0),
      extremes: Uint32Array.of(sortedKeys[0], sortedKeys[2], 3, 3),
      sortedKeys,
      gapKeys,
      gapPositions: new Uint32Array(3)
    };
    const reflection = new WgslReflect(source);
    const bindings: Record<number, Uint32Array> = {};
    for (const resource of reflection.storage) {
      expect(buffers[resource.name], resource.name).toBeDefined();
      bindings[resource.binding] = buffers[resource.name];
    }
    const parameters = new Uint32Array(64);
    parameters.set([0, 3, 3, 0]);
    for (const uniform of reflection.uniforms) bindings[uniform.binding] = parameters;
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
      'main',
      1,
      {0: bindings},
      {constants: module.constants}
    );
    // Bits 0x08800000 have exponent -110, ULP 2^-133 = 65536 * 2^-149.
    // The two gaps are 2^-133 and 2^-132, so their complemented f32 bits differ.
    // Other literal keys were checked with Python struct.pack('<f', upper-lower), then bitwise
    // complemented: negative endpoints, the normal/subnormal boundary and a signed crossing.
    expect(Array.from(gapKeys)).toEqual(fixture.keys);
  } finally {
    context.device.destroy();
  }
});

it.each([
  {
    values: [-2 + 2 ** -23, -2, -2, -2 + 2 ** -23, -2],
    edges: [-2, -1.9999998807907104, -1.9999998807907104],
    counts: [3, 2]
  },
  {values: [-(2 ** 127), 2 ** 127], edges: [-(2 ** 127), 0, 2 ** 127], counts: [1, 1]}
])('review B2 oracle publishes separating, overflow-safe maximum-break edges: %j', fixture => {
  const values = Float32Array.from(fixture.values);
  const result = computeClassBreaksOracle({
    values,
    parameters: {method: 'maximum-breaks', classCount: 2},
    maximumClassCount: 2,
    naturalBreaksBinCount: 64
  });
  expect(result.classCount).toBe(2);
  expect(Array.from(result.breaks)).toEqual(fixture.edges);
  expect(Array.from(countClassesOracle(values, undefined, result.breaks, 2, 2))).toEqual(
    fixture.counts
  );
});
