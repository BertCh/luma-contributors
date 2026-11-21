// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createInequalityGiniGapNodes} from '../../../src/dataframe-analysis/composite-indicators/inequality-gini-gaps';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET} from '../../../src/analysis-core/kernels/graph-chunk-groups';

it.each([
  {values: [0, 1], weights: [1, 2 ** 58], expected: 3.469446951953614e-18},
  {values: [0, 1], weights: [2 ** 58, 1], expected: 1},
  {values: [1, 2], weights: [2 ** 58, 1], expected: 3.469446951953614e-18},
  {values: [1, 1 + 2 ** -23], weights: [1, 1], expected: 2.9802320611338473e-8},
  {
    values: [2 ** -110, 2 ** -110 * (1 + 2 ** -23)],
    weights: [1, 1],
    expected: 2.9802320611338473e-8
  },
  {values: [0, 2 ** -80], weights: [2 ** -80, 2 ** -80], expected: 0.5},
  {values: [0, 0], weights: [2 ** -80, 2 ** -80], expected: NaN},
  {values: new Array<number>(12).fill(1), weights: new Array<number>(12).fill(1), expected: 0}
])('NUM-15 executes gap publication arithmetic against a literal pair-difference answer: %j', fixture => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const rowCount = fixture.values.length;
    const values = createTransientView(graph, 'values', 'float32', rowCount);
    const weights = createTransientView(graph, 'weights', 'float32', rowCount);
    const sortedRows = createTransientView(graph, 'rows', 'uint32', rowCount);
    const sortedZones = createTransientView(graph, 'zones', 'uint32', rowCount);
    const output = createTransientView(graph, 'gini', 'float32', 1);
    const nodes = createInequalityGiniGapNodes(graph, {
      id: 'gap',
      sortedRows,
      sortedZones,
      values,
      weights,
      zoneCount: 1,
      output,
      outputStride: 1,
      outputSlot: 0
    });
    const memory = createCPUGraphMemory();
    memory.write(values, Float32Array.from(fixture.values));
    memory.write(weights, Float32Array.from(fixture.weights));
    memory.write(
      sortedRows,
      Uint32Array.from({length: rowCount}, (_, row) => row)
    );
    memory.write(sortedZones, new Uint32Array(rowCount));
    // The interpreter does not bind chunk-group uniforms. Rebind the recorded arithmetic with
    // literal parameters, and supply direct f32 prefix/suffix coordinates (GPU scans are queued).
    const termRecord = getWGSLKernelNodeRecord(graph, 'gap-terms');
    const publishRecord = getWGSLKernelNodeRecord(graph, 'gap-publish');
    if (!termRecord || !publishRecord) throw new Error('Missing gap publication records');
    const left = new Float32Array(rowCount);
    const right = new Float32Array(rowCount);
    for (let row = 0; row < rowCount; row++) {
      left[row] = (row ? left[row - 1] : 0) + fixture.weights[row];
      right[row] = (row ? right[row - 1] : 0) + fixture.weights[rowCount - 1 - row];
    }
    for (const [name, data] of [
      ['left', left],
      ['right', right]
    ] as const) {
      const binding = termRecord.bindings.find(binding => binding.name === name);
      if (!binding) throw new Error(`Missing ${name} binding`);
      memory.write(binding.view, data);
    }
    const rebind = (record: typeof termRecord, parameters: readonly number[]) =>
      createWGSLKernelNode(graph, {
        id: `${record.id}-arithmetic`,
        operation: record.operation,
        bindings: [...record.bindings],
        invocationCount: record.invocationCount,
        declarations: `${record.declarations ?? ''}
fn flushSubnormal(value: f32) -> f32 {
  return select(value, 0.0, (bitcast<u32>(value) & 0x7fffffffu) < 0x00800000u);
}`,
        body: parameters.reduce(
          (body, value, index) =>
            body.replaceAll(
              `getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + index}u)`,
              `${value}u`
            ),
          record.body.replace(/let gap = (.*);/, 'let gap = flushSubnormal($1);')
        )
      });
    const rowExponents = getWGSLKernelNodeRecord(graph, 'gap-income-row-exponents');
    const incomeTerms = getWGSLKernelNodeRecord(graph, 'gap-income-terms');
    if (!rowExponents || !incomeTerms) throw new Error('Missing scaled-income kernels');
    const termsBinding = termRecord.bindings.find(binding => binding.name === 'terms');
    const exponentBinding = incomeTerms.bindings.find(binding => binding.name === 'exponents');
    const totalsBinding = termRecord.bindings.find(binding => binding.name === 'totals');
    if (!termsBinding || !exponentBinding || !totalsBinding)
      throw new Error('Missing scaled-income bindings');
    executeCommandNodesOnCPU([rebind(rowExponents, [1])], memory);
    const incomeExponent = Math.max(...memory.readFloat32(termsBinding.view));
    memory.write(exponentBinding.view, Float32Array.of(incomeExponent));
    executeCommandNodesOnCPU([rebind(incomeTerms, [1])], memory);
    const scaledIncome = memory
      .readFloat32(termsBinding.view)
      .reduce((total, term) => Math.fround(total + term), 0);
    memory.write(
      totalsBinding.view,
      Float32Array.of(left[rowCount - 1], scaledIncome, incomeExponent)
    );
    expect(nodes.some(node => node.id === 'gap-sum')).toBe(true);
    executeCommandNodesOnCPU([rebind(termRecord, [rowCount, 1])], memory);
    const sumsBinding = publishRecord.bindings.find(binding => binding.name === 'sums');
    if (!termsBinding || !sumsBinding) throw new Error('Missing gap reduction bindings');
    // Cooperative segment reductions also need a real GPU; sum these nonnegative row terms
    // on the host so this test isolates the actual WGSL contribution and publication formulas.
    const sum = memory
      .readFloat32(termsBinding.view)
      .reduce((total, term) => Math.fround(total + term), 0);
    memory.write(sumsBinding.view, Float32Array.of(sum));
    executeCommandNodesOnCPU([rebind(publishRecord, [1, 0])], memory);
    const actual = memory.readFloat32(output)[0];
    // Two-row pair difference: w0*w1*abs(x1-x0)/((w0+w1)*(w0*x0+w1*x1)).
    // Equal incomes have zero pair differences. These are independent literal f32 roundings.
    if (Number.isNaN(fixture.expected)) {
      expect(actual).toBeNaN();
    } else {
      expect(Number.isFinite(actual)).toBe(true);
      expect(Math.abs(actual - fixture.expected)).toBeLessThanOrEqual(fixture.expected * 2e-5);
    }
  } finally {
    device.destroy();
  }
});
