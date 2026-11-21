// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {validateCPUHostOffsets} from '../../src/analysis-core/operations/compact-host-offsets';
import {
  createWGSLKernelNode,
  getCompiledWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {auditKernelCase, getKernelAuditResultFailures} from './kernel-audit/kernel-audit-runner';
import {createKernelAuditContext} from './kernel-audit/kernel-audit-context';
import {collectClassParameters, renderClassParameters} from './doc-parameters';

it('forward compact offsets read each indexed value once plus the initial-offset check', () => {
  let reads = 0;
  const offsets = new Proxy(Uint32Array.of(0, 1, 2, 3), {
    get(target, property) {
      if (typeof property === 'string' && /^\d+$/.test(property)) reads++;
      return Reflect.get(target, property, target);
    }
  });
  validateCPUHostOffsets('compact', 'offsets', offsets, 3);
  // Four loop reads plus one initial zero check; diagnostics must reuse the captured value.
  expect(reads).toBe(5);
});

it.each([
  'before',
  'after'
] as const)('forward selected kernel excludes independent inspection %s compile', order => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  Object.defineProperty(device, 'createComputePipeline', {
    value: () => ({id: '', hash: '', destroy: () => {}})
  });
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const output = createTransientView(graph, 'output', 'uint32', 3);
  const makeNode = (value: number, invocationCount: number) =>
    createWGSLKernelNode(graph, {
      id: 'step',
      operation: 'recording',
      invocationCount,
      workgroupSize: 2,
      bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
      body: `output[outputOffset + index] = ${value}u;`
    });
  const selected = makeNode(3, 3);
  const spare = makeNode(7, 1);
  graph.add(selected);
  if (order === 'before') {
    getGPUCommandNodeKernelSources([spare]);
    expect(getCompiledWGSLKernelNodeRecord(graph, 'step')).toBeUndefined();
  }
  const compiled = graph.compile();
  try {
    if (order === 'after') getGPUCommandNodeKernelSources([spare]);
    const record = getCompiledWGSLKernelNodeRecord(graph, 'step');
    expect(record?.body).toContain('= 3u;');
    // ceil(3 invocations / 2 lanes) = 2 workgroups.
    expect(record?.workgroups).toEqual([2, 1, 1]);
    expect(record?.bindings[0].view).toBe(output);
  } finally {
    compiled.destroy();
    device.destroy();
  }
});

it('forward limit constants resolve through the public spatial entry', async () => {
  const spatial = await import('../../src/gpu-spatial-analysis/index');
  expect(spatial['MAXIMUM_GPU_GRID_CENTER_AXIS_CELL_COUNT']).toBe(8_388_608);
  expect(spatial['MAXIMUM_PROBIT_CONVERGENCE_TOLERANCE']).toBe(0.00001);
});

it('forward Kalman maximum reserves its 16-byte metadata header', async () => {
  const {COVERAGE_NETWORK_MOVEMENT_CASES} = await import(
    './kernel-audit/cases-coverage-network-movement'
  );
  const fixture = COVERAGE_NETWORK_MOVEMENT_CASES.find(
    entry =>
      entry.exportName === 'GPUTrajectoryKalman' &&
      entry.name === 'binding-limited observations,filter and smoother'
  )!;
  const result = auditKernelCase(fixture);
  expect(getKernelAuditResultFailures(result)).toEqual([]);
});

it('forward Web Mercator horizon fixtures use southward rows', async () => {
  const {TERRAIN_CASES} = await import('./kernel-audit/cases-terrain');
  const fixtures = TERRAIN_CASES.filter(
    entry =>
      ['GPUPointHorizonProfile', 'GPUPointHorizonVisibility'].includes(entry.exportName) &&
      /^web-mercator, (pyramid|march),/.test(entry.name)
  );
  expect(fixtures).toHaveLength(6);
  for (const fixture of fixtures) {
    const context = createKernelAuditContext('forward-horizon');
    try {
      expect(() => fixture.build(context), fixture.name).not.toThrow();
    } finally {
      context.device.destroy();
    }
  }
});

it('forward exact trace accepted maxima obey the total command ceiling', async () => {
  const {STATISTICS_CASES} = await import('./kernel-audit/cases-statistics');
  const fixtures = STATISTICS_CASES.filter(
    entry =>
      entry.exportName === 'GPUSpatialWeightsTraces' &&
      entry.name.includes('exact') &&
      entry.name.includes('rows at')
  );
  expect(fixtures).toHaveLength(2);
  for (const fixture of fixtures) {
    const context = createKernelAuditContext('forward-traces');
    try {
      expect(() => fixture.build(context), fixture.name).not.toThrow();
    } finally {
      context.device.destroy();
    }
  }
});

it('forward exact trace next batches and binding-only maxima refuse before registration', async () => {
  const {STATISTICS_CASES} = await import('./kernel-audit/cases-statistics');
  const fixtures = STATISTICS_CASES.filter(
    entry =>
      entry.exportName === 'GPUSpatialWeightsTraces' &&
      /next batch|binding-limited rows/.test(entry.name)
  );
  expect(fixtures).toHaveLength(4);
  for (const fixture of fixtures) {
    const result = auditKernelCase(fixture);
    expect(getKernelAuditResultFailures(result), fixture.name).toEqual([]);
    expect(result.rejected).toBe(true);
  }
});

it('forward terrain horizon parameter summary includes its three conditional lengths', () => {
  const entry = collectClassParameters().find(entry => entry.className === 'GPUTerrainHorizon')!;
  const rendered = renderClassParameters(entry, 3);
  for (const length of ['(8)', '(12)', '(24)']) expect(rendered).toContain(length);
  expect(rendered).toContain('base settings');
  expect(rendered).toContain('explicit sweep settings');
});
