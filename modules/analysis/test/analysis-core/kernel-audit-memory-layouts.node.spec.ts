// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it, vi} from 'vitest';
import {KERNEL_AUDIT_CASES} from './kernel-audit/kernel-audit-cases';
import {
  auditKernelCase,
  formatKernelAuditFindings,
  getKernelAuditResultFailures
} from './kernel-audit/kernel-audit-runner';

vi.setConfig({testTimeout: 120_000});

describe('registered kernel memory layouts', () => {
  it('has a nonempty case registry in every area', () => {
    expect(Object.keys(KERNEL_AUDIT_CASES).length).toBeGreaterThan(0);
    for (const [area, cases] of Object.entries(KERNEL_AUDIT_CASES)) {
      expect(cases.length, area).toBeGreaterThan(0);
    }
  });

  it('keeps a resolved workgroup-budget finding of a case (CORE-114)', () => {
    // 8,192 u32 words are 32,768 bytes of workgroup storage; core WebGPU guarantees 16,384.
    const result = auditKernelCase({
      exportName: 'GPUFittingGradientSums',
      name: 'over-budget-workgroup-regression',
      build: ({graph, view}) => {
        const values = view('uint32', 1);
        graph.add({
          id: 'over-budget-workgroup-node',
          type: 'compute',
          resources: [{buffer: values, usage: 'storage-read'}],
          compile: ({device}) => {
            const shader = device.createShader({
              stage: 'compute',
              source: [
                'const N = 8192u;',
                'var<workgroup> scratch: array<u32, N>;',
                '@group(0) @binding(0) var<storage, read> values: array<u32>;',
                '@compute @workgroup_size(1) fn main() { scratch[0] = values[0]; }'
              ].join('\n')
            });
            device.createComputePipeline({
              shader,
              shaderLayout: {
                bindings: [{name: 'values', type: 'read-only-storage', group: 0, location: 0}]
              }
            });
            return {encode: () => {}, destroy: () => shader.destroy()};
          }
        });
      }
    });
    expect(getKernelAuditResultFailures(result)).toEqual([
      {
        check: 'workgroup bytes',
        detail: '32768 > 16384',
        location: expect.stringMatching(/^over-budget-workgroup-node /)
      }
    ]);
  });

  // Enumerate the registry directly: a separately maintained area-spec list or contributor
  // filter must not silently remove a registered configuration from memory-layout coverage.
  for (const [area, cases] of Object.entries(KERNEL_AUDIT_CASES)) {
    for (const auditCase of cases) {
      it(`${area}: ${auditCase.exportName} ${auditCase.name}`, () => {
        const failures = getKernelAuditResultFailures(auditKernelCase(auditCase));
        expect(failures, formatKernelAuditFindings(failures)).toEqual([]);
      });
    }
  }
});
