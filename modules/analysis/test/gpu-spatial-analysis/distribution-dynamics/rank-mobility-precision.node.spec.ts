// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, getViewBindingRange} from '@luma.gl/graph/gpu-core';
import {ScalarData, WgslExec, WgslParser} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {createRankMobilityNodes} from '../../../src/gpu-spatial-analysis/distribution-dynamics/rank-markov-sorted';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  auditWGSLModule,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

/** Model the permitted f32 reciprocal multiplication that reproduced the real Metal log. */
function createMetalDivisionExecutor(source: string): WgslExec {
  const executor = new WgslExec(new WgslParser().parse(source.replace(/^\s*_\s*=[^;]*;/gm, '')));
  const evaluateBinary = executor._evalBinaryOp.bind(executor);
  executor._evalBinaryOp = (node, context) => {
    const result = evaluateBinary(node, context);
    if (node.operator === '/' && result instanceof ScalarData && result.typeInfo.name === 'f32') {
      const numerator = executor.evalExpression(node.left, context);
      const denominator = executor.evalExpression(node.right, context);
      if (numerator instanceof ScalarData && denominator instanceof ScalarData) {
        const reciprocal = Math.fround(1 / denominator.value);
        result.value = Math.fround(
          numerator.value * (Math.abs(reciprocal) < 2 ** -126 ? 0 : reciprocal)
        );
      }
    }
    return result;
  };
  return executor;
}

it('reproduces both logged Metal mobility errors with the previous f32 division', () => {
  const values = Float32Array.of(16228, 234, 0, 16370, 235, 0);
  createMetalDivisionExecutor(`
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let base = invocation.x * 3u;
  values[base + 2u] = values[base] / values[base + 1u];
}`).dispatchWorkgroups('main', [2], {0: {0: values}});
  expect(values[2]).toBe(69.35043334960938);
  expect(values[5]).toBe(69.6595687866211);
  expect(Math.fround(16228 / 234)).toBe(69.35042572021484);
  expect(Math.fround(16370 / 235)).toBe(69.65957641601562);
});

it('rounds the emitted integer mobility mean to the CPU result under Metal division lowering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const cases = [
      [16228, 234],
      [16370, 235],
      [0, 2],
      [2, 65535],
      [2147418112, 65535],
      [65535 * 65534, 65535],
      // Exact halfway cases, rounding to an even significand in opposite directions.
      [16777217, 2],
      [16777219, 2],
      [0, 0],
      [0, 1]
    ];
    for (let population = 2; population < 65535; population *= 3)
      cases.push([population * (population - 1) - 2, population]);
    const periods = cases.length + 1;
    const mobilityStorage = createTransientView(graph, 'mobility-storage', 'float32', periods + 16);
    const mobility = graph.createDataView(mobilityStorage.buffer, {
      format: 'float32',
      length: periods - 1,
      byteOffset: 64
    });
    const nodes = createRankMobilityNodes(graph, 'rounding', {
      values: createTransientView(graph, 'values', 'float32', 65535 * periods),
      periodCount: periods,
      output: {
        ranks: createTransientView(graph, 'ranks', 'uint32', 65535 * periods),
        counts: createTransientView(graph, 'unused-counts', 'uint32', 1),
        mobility
      }
    });
    const record = getWGSLKernelNodeRecord(graph, 'rounding-mobility')!;
    expect(auditWGSLModule(record.source, record.constants)).toEqual([]);
    const identifiers = stripWGSLComments(record.source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
    expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
    for (const [displacement, population] of cases) {
      const result = new Float32Array(getViewBindingRange(mobility).size / 4).fill(-1);
      createMetalDivisionExecutor(record.source).dispatchWorkgroups(
        'main',
        [...record.workgroups],
        {0: {0: Uint32Array.of(displacement, population), 1: result}},
        {constants: {...record.constants, INVOCATION_COUNT: 1}}
      );
      if (population < 2) expect(result[16]).toBeNaN();
      else
        expect(result[16], `${displacement}/${population}`).toBe(
          Math.fround(displacement / population)
        );
      expect([...result.slice(0, 16)]).toEqual(new Array(16).fill(-1));
    }
    const reduce = nodes.find(node => node.id === 'rounding-mobility-reduce')!;
    expect(getWGSLKernelNodeRecord(graph, reduce.id)!.source).toContain('vec2<u32>');
  } finally {
    device.destroy();
  }
});
