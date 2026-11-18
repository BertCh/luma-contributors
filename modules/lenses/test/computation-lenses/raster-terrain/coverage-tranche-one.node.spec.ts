// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '@luma-contributors/analysis/testing';
import {buildCoverageTrancheOneScenario} from '../../../src/computation-lenses/scenarios/gpu-raster/coverage-tranche-one-runner';
import {COVERAGE_TRANCHE_ONE_SCENARIOS} from '../../../src/computation-lenses/scenarios/gpu-raster/coverage-tranche-one.scenarios';
import {createCoverageTrancheOneHarness} from './coverage-tranche-one-harness';

it.each(
  COVERAGE_TRANCHE_ONE_SCENARIOS
)('$element builds its real example with semantic captures and core-limit kernels', async scenario => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  // NullDevice lacks compute pipelines; stub the GPU resource while preserving graph validation.
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
  });
  const harness = createCoverageTrancheOneHarness(device, scenario.element);
  try {
    await buildCoverageTrancheOneScenario(harness.session, scenario.element);
    const outputs = [...harness.captures.values()].filter(
      capture => capture.spec.role === 'output'
    );
    expect(outputs.length).toBeGreaterThan(0);
    for (const output of outputs) expect(output.spec.meaning?.length).toBeGreaterThan(10);
    const kernels = getParsedKernelSources((harness.graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.size).toBeGreaterThan(0);
    for (const {id: name, source} of kernels.configurations)
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    const compiled = harness.graph.compile();
    try {
      for (const {view} of harness.captures.values()) {
        expect(view.buffer.byteLength).toBeGreaterThanOrEqual(view.length * 4);
      }
    } finally {
      compiled.destroy();
    }
    expect(scenario.domain).toBe(
      scenario.element === 'addDasymetricRedistributionRecipe'
        ? 'gpu-spatial-analysis'
        : 'gpu-raster'
    );
  } finally {
    harness.destroy();
    device.destroy();
  }
});
