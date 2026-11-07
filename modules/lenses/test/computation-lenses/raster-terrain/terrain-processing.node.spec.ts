// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '@luma-contributors/graph/testing';
import {auditWGSLModule} from '@luma-contributors/graph/test/analysis-core/kernel-audit/kernel-audit-wgsl';
import {buildTerrainProcessingScenario} from '../../../src/computation-lenses/scenarios/gpu-terrain/terrain-processing-runner';
import {TERRAIN_PROCESSING_SCENARIOS} from '../../../src/computation-lenses/scenarios/gpu-terrain/terrain-processing.scenarios';
import {createCoverageTrancheOneHarness} from './coverage-tranche-one-harness';

it.each(
  TERRAIN_PROCESSING_SCENARIOS
)('$element builds its actual public example and result captures', async ({element}) => {
  const device = createNullWebGPUDevice();
  const harness = createCoverageTrancheOneHarness(device, element);
  try {
    await buildTerrainProcessingScenario(harness.session, element);
    const outputs = [...harness.captures.values()].filter(
      capture => capture.spec.role === 'output'
    );
    expect(outputs.length).toBeGreaterThan(0);
    for (const output of outputs) expect(output.spec.meaning?.length).toBeGreaterThan(10);
    const kernels = getParsedKernelSources((harness.graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.size).toBeGreaterThan(0);
    for (const [name, source] of kernels) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(source), name).toEqual([]);
    }
  } finally {
    harness.destroy();
    device.destroy();
  }
});
