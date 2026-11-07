// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '@luma-contributors/analysis/testing';
import {auditWGSLModule} from '@luma-contributors/analysis/test/analysis-core/kernel-audit/kernel-audit-wgsl';
import {buildRemoteSensingScenario} from '../../../src/computation-lenses/scenarios/gpu-raster/remote-sensing-runner';
import {REMOTE_SENSING_SCENARIOS} from '../../../src/computation-lenses/scenarios/gpu-raster/remote-sensing.scenarios';
import {createCoverageTrancheOneHarness} from './coverage-tranche-one-harness';

it.each(
  REMOTE_SENSING_SCENARIOS
)('$element lens constructs real graph kernels and semantic outputs', async ({element}) => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
  });
  const harness = createCoverageTrancheOneHarness(device, element, 16);
  try {
    await buildRemoteSensingScenario(harness.session, element);
    const outputs = [...harness.captures.values()].filter(
      capture => capture.spec.role === 'output'
    );
    expect(outputs.length).toBeGreaterThan(0);
    for (const output of outputs) expect(output.spec.meaning?.length).toBeGreaterThan(10);
    const kernels = getParsedKernelSources((harness.graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.size).toBeGreaterThan(0);
    for (const source of kernels.values()) {
      expect(auditWGSLModule(source)).toEqual([]);
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    }
    const compiled = harness.graph.compile();
    try {
      for (const {view} of harness.captures.values())
        expect(view.buffer.byteLength).toBeGreaterThanOrEqual(view.length * 4);
    } finally {
      compiled.destroy();
    }
  } finally {
    harness.destroy();
    device.destroy();
  }
});
