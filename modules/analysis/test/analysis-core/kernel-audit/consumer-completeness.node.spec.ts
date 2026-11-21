// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {MOB10_FLOW_MODELS_AUDIT_CASES} from './cases-mob10-flow-models';
import {QUANTILE_LISA_AUDIT_CASES} from './cases-quantile-lisa';
import {auditKernelCase} from './kernel-audit-runner';
import {GPUKernelFlowSmoothing} from '../../../src/gpu-network/flow-aggregation/gpu-kernel-flow-smoothing';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  type KernelAuditCase
} from './kernel-audit-context';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-quantile-lisa';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

for (const mode of ['producer', 'no-candidate-word', 'legacy-import'] as const) {
  it(`CPL-09 publishes completeness from ${mode} flow evidence`, () => {
    const context = createKernelAuditContext('flow-completeness');
    const {view, graph} = context;
    try {
      const count = view('uint32', 1);
      const overflow = view('uint32', 1);
      const candidateOverflow = view('uint32', 1);
      const complete = view('uint32', 1);
      const contributor = new GPUKernelFlowSmoothing({
        positions: view('float32x2', 2),
        locationWeights: view('float32', 2),
        parameters: view('float32', 4),
        flows: {
          kind: 'pairs',
          originZoneIds: view('uint32', 2),
          destinationZoneIds: view('uint32', 2),
          weights: view('float32', 2),
          ...(mode === 'legacy-import'
            ? {count, completeness: 'known-complete' as const}
            : {status: {count, overflow, ...(mode === 'producer' ? {candidateOverflow} : {})}})
        },
        output: {smoothed: view('float32', 4), status: {complete}}
      });
      const nodes = contributor
        .getCommandNodes(graph)
        .filter(node => node.id.endsWith('-completeness'));
      expect(nodes).toHaveLength(1);
      const memory = createCPUGraphMemory();
      for (const [outputLoss, candidateLoss, expected] of [
        [0, 0, mode === 'no-candidate-word' ? 0 : 1],
        [1, 0, mode === 'legacy-import' ? 1 : 0],
        [0, 1, mode === 'legacy-import' ? 1 : 0]
      ]) {
        memory.write(count, Uint32Array.of(2));
        memory.write(overflow, Uint32Array.of(outputLoss));
        memory.write(candidateOverflow, Uint32Array.of(candidateLoss));
        memory.write(complete, Uint32Array.of(99));
        executeCommandNodesOnCPU(nodes, memory);
        // A certificate is true exactly when both producer loss words are zero;
        // imported legacy rows carry a separate explicit completeness guarantee.
        expect(Array.from(memory.readUint32(complete))).toEqual([expected]);
      }
    } finally {
      context.device.destroy();
    }
  });
}

for (const auditCase of MOB10_FLOW_MODELS_AUDIT_CASES.filter(
  auditCase => auditCase.exportName === 'GPUKernelFlowSmoothing'
)) {
  it(`CPL-09 records all flow modules: ${auditCase.name}`, () => {
    const result = auditKernelCase(auditCase);
    expect(result.findings).toEqual([]);
    expect(result.moduleCount).toBeGreaterThan(0);
  });
}

function expectPinnedInferenceRule(auditCase: KernelAuditCase): void {
  const context = createKernelAuditContext('quantile-inference-variant');
  try {
    auditCase.build(context);
    const finalizers = (
      getDeclaredNodes(context) as {id: string; workload?: {variant?: string}}[]
    ).filter(node => node.id.endsWith('-geoda-tail'));
    expect(finalizers).toHaveLength(1);
    expect(finalizers[0].workload?.variant).toBe('strict-complement-tail');
  } finally {
    context.device.destroy();
  }
  const result = auditKernelCase(auditCase);
  expect(result.findings).toEqual([]);
  expect(result.moduleCount).toBeGreaterThan(0);
}

it('DEF-01 pins the quantile-LISA audit to its original explicit inference rule', () => {
  expectPinnedInferenceRule(QUANTILE_LISA_AUDIT_CASES[0]);
});

it('review B9 refuses a greater-tail mutation even though its modules are valid', () => {
  const original = QUANTILE_LISA_AUDIT_CASES[0];
  const changedRule: KernelAuditCase = {
    ...original,
    build(context) {
      const add = context.graph.add.bind(context.graph);
      context.graph.add = contributor => {
        if (contributor instanceof GPUQuantileLISA)
          add(new GPUQuantileLISA({...contributor.props, alternative: 'greater'}));
        else add(contributor);
      };
      original.build(context);
    }
  };
  expect(() => expectPinnedInferenceRule(changedRule)).toThrow(/strict-complement-tail/);
});
