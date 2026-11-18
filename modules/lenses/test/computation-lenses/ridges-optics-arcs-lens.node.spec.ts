// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice, getParsedKernelSources} from '@luma-contributors/analysis/testing';
import {auditWGSLModule} from '@luma-contributors/analysis/test/analysis-core/kernel-audit/kernel-audit-wgsl';
import {buildScalarFieldRidgesScenario} from '../../src/computation-lenses/scenarios/gpu-raster/scalar-field-ridges.lens';
import {createCoverageTrancheOneHarness} from './raster-terrain/coverage-tranche-one-harness';
import {getRasterOperatorEntries} from '../../src/gpu-raster/computation-pack/pack';
import {LINES_FORMULAS} from '../../src/gpu-spatial-analysis/computation-pack/operators/lines-formulas';
import {POINT_PROCESS_OPERATORS} from '../../src/gpu-spatial-analysis/point-process/point-process-operators';
import {listLensScenarios} from '../../src/computation-lenses';
import '../../src/computation-lenses/scenarios/gpu-raster/index';

it('covers scalar field ridges with a formula and registered raster scenario', () => {
  const entry = getRasterOperatorEntries()['GPUScalarFieldRidges'];
  expect(entry).toBeDefined();
  expect(entry.neighbourhood).toBe('stencil-3x3');
  expect('tex' in entry.formula).toBe(true);
  expect(
    listLensScenarios().find(scenario => scenario.element === 'GPUScalarFieldRidges')
  ).toMatchObject({id: 'gpu-raster.scalar-field-ridges', entry: 'gpu-raster'});
});

it('describes the indexed and parallel OPTICS ordering limits', () => {
  const entry = POINT_PROCESS_OPERATORS['GPUOPTICS'];
  expect(entry.metadata).toMatchObject({maximumEvents: 65536, maximumBruteForceEvents: 4096});
  expect(entry.choices).toContainEqual({
    name: 'orderingMethod',
    stage: 'operator',
    alternatives: ['sequential', 'parallel'],
    sweepable: false
  });
});

it('builds the ridge lens through its public example with typed, meaningful captures', async () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy() {}})
  });
  const harness = createCoverageTrancheOneHarness(device, 'scalar-field-ridges', 49);
  try {
    await buildScalarFieldRidgesScenario(harness.session);
    expect(harness.inputs.get('scalarFieldValues')?.values[24]).toBe(19.75);
    expect(harness.inputs.get('scalarFieldValidity')?.values[40]).toBe(0);
    for (const [name, format] of [
      ['ridgeMask', 'uint32'],
      ['ridgeValidity', 'uint32'],
      ['ridgeStrength', 'float32']
    ]) {
      const capture = harness.captures.get(name);
      expect(capture?.view).toMatchObject({format, length: 49});
      expect(capture?.spec.meaning?.length).toBeGreaterThan(10);
    }
    const kernels = getParsedKernelSources((harness.graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.size).toBeGreaterThan(0);
    for (const {source, constants} of kernels.configurations)
      expect(auditWGSLModule(source, constants)).toEqual([]);
    const compiled = harness.graph.compile();
    compiled.destroy();
  } finally {
    harness.destroy();
    device.destroy();
  }
});

it('keeps the inhomogeneous point-pattern limit separate from indexed OPTICS', () => {
  const metadata = POINT_PROCESS_OPERATORS['GPUInhomogeneousPointPattern'].metadata;
  expect(metadata).toMatchObject({maximumEvents: 4096});
  expect(metadata).not.toHaveProperty('maximumBruteForceEvents');
});

it('describes arc interpolation without promising whole-arc antipodal rejection', () => {
  expect(LINES_FORMULAS.arcs.ascii).toContain('No whole-arc rejection is performed');
  expect(LINES_FORMULAS.arcs.ascii).not.toContain('rejects the whole arc');
});
