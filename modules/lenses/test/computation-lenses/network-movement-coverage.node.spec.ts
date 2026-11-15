// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUComputationCapture} from '@luma-contributors/analysis/computation-record';
import {validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {
  describeNetworkOptimizationCoverage,
  createNetworkOptimizationScenario,
  type NetworkOptimizationKind
} from '../../src/computation-lenses/scenarios/gpu-network/optimization-coverage.lens';
import {COVERAGE_NETWORK_MOVEMENT_CASES} from '@luma-contributors/analysis/test/analysis-core/kernel-audit/cases-coverage-network-movement';
import {auditKernelCase} from '@luma-contributors/analysis/test/analysis-core/kernel-audit/kernel-audit-runner';

for (const auditCase of COVERAGE_NETWORK_MOVEMENT_CASES) {
  it(`bounded coverage maximal shader audit: ${auditCase.exportName} ${auditCase.name}`, () => {
    const result = auditKernelCase(auditCase);
    expect(result.findings).toEqual([]);
    expect(result.nodeCount).toBeGreaterThan(0);
  });
}
for (const kind of [
  'point-routing',
  'flow',
  'louvain',
  'louvain-sparse',
  'facility-optimization',
  'time-routing',
  'rational-access'
] as const satisfies readonly NetworkOptimizationKind[]) {
  it(`network ${kind} lens builds a valid executable graph description`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const record = await describeNetworkOptimizationCoverage(device, kind);
      expect(validateGPUComputationRecord(record)).toEqual({valid: true, issues: []});
      expect(record.graphs.length).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
}

for (const kind of ['facility-optimization', 'rational-access'] as const) {
  it(`network ${kind} lens retains and validates the renamed allocation summary`, () => {
    const scenario = createNetworkOptimizationScenario(kind);
    expect(scenario.retain).toContain('summary');
    expect(scenario.retain).not.toContain('status');
    const summary: GPUComputationCapture = {
      id: 'summary',
      name: 'summary',
      elementType: 'u32',
      values: Uint32Array.of(1, 1, 1, 4),
      provenance: 'gpu-readback'
    };
    const final = (name: string) => {
      expect(name).toBe('summary');
      return summary;
    };
    // Capture does not depend on execution context; the reader supplies the completed readback.
    const captures = scenario.capture(final, {} as Parameters<typeof scenario.capture>[1], [
      summary
    ]);
    expect(captures).toEqual([{...summary, keySpace: 'entry'}]);
    for (const values of [Uint32Array.of(0, 0, 0, 0), Uint32Array.of(1, 0, 0, 4)]) {
      expect(() =>
        scenario.capture(
          () => ({...summary, values}),
          {} as Parameters<typeof scenario.capture>[1],
          [summary]
        )
      ).toThrow('Network allocation example did not produce a valid result');
    }
  });
}
