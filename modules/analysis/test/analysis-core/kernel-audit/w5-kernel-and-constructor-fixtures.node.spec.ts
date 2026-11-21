// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {RECIPES_CASES} from './cases-recipes';
import {POINTCLOUD_AND_DATUM_CASES} from './cases-pointcloud-and-datum';
import {TERRAIN_CASES} from './cases-terrain';
import {GEOMETRY_CASES} from './cases-geometry';
import {CELLS_AND_JOINS_CASES} from './cases-cells-and-joins';
import {RTS_TIME_SERIES_AUDIT_CASES} from './cases-rts-time-series';
import {GPUTrackSimilarity} from '../../../src/gpu-spatial-analysis/index';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/analysis-validation-error';
import {
  createKernelAuditContext,
  getDeclaredNodes,
  type KernelAuditCase,
  type KernelAuditContext
} from './kernel-audit-context';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes,
  type WGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  auditKernelCase,
  getExpectedRejection,
  getKernelAuditResultFailures
} from './kernel-audit-runner';

function withRecordedCase(
  auditCase: KernelAuditCase,
  check: (
    context: KernelAuditContext,
    kernels: WGSLKernelNodeRecord[],
    views: ReadonlyMap<string, GraphDataView>
  ) => void
): void {
  const context = createKernelAuditContext('review-round-one');
  const views = new Map<string, GraphDataView>();
  try {
    recordWGSLKernelNodes(context.graph);
    auditCase.build({
      ...context,
      view: (format, length, name) => {
        const view = context.view(format, length, name);
        if (name) views.set(name, view);
        return view;
      }
    });
    const kernels = (getDeclaredNodes(context) as {id: string}[]).flatMap(node => {
      const kernel = getWGSLKernelNodeRecord(context.graph, node.id);
      return kernel ? [kernel] : [];
    });
    check(context, kernels, views);
  } finally {
    context.device.destroy();
  }
}

function assertCoverageOutputs(kernel: WGSLKernelNodeRecord): void {
  expect(kernel.source).toContain('(coverage & 1u) != 0u');
  expect(kernel.source).toContain('(coverage & 2u) != 0u');
  const values: Record<string, Uint32Array> = {
    coverage: Uint32Array.of(0, 1, 2, 3),
    surfaceValidity: new Uint32Array(4).fill(99),
    terrainValidity: new Uint32Array(4).fill(99),
    canopyValidity: new Uint32Array(4).fill(99)
  };
  const bindings: Record<number, Uint32Array> = {};
  for (const binding of new WgslReflect(kernel.source).storage) {
    const value = values[binding.name];
    if (!value) throw new Error(`unexpected coverage binding ${binding.name}`);
    bindings[binding.binding] = value;
  }
  // Execute the actual generated module over four rows. The independent truth table is the
  // four possible two-bit words: bit 0 means surface and bit 1 means ground (terrain/canopy).
  new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
    kernel.entryPoint,
    [1, 1, 1],
    {0: bindings},
    {constants: {...kernel.constants, INVOCATION_COUNT: 4}}
  );
  expect(Array.from(values['surfaceValidity'])).toEqual([0, 1, 0, 1]);
  expect(Array.from(values['terrainValidity'])).toEqual([0, 0, 1, 1]);
  expect(Array.from(values['canopyValidity'])).toEqual([0, 0, 1, 1]);
}

function assertStopCubeInputs(
  kernels: readonly WGSLKernelNodeRecord[],
  views: ReadonlyMap<string, GraphDataView>,
  time: 'start' | 'finish'
): void {
  const fixtureView = (name: string) => {
    const view = views.get(name);
    if (!view) throw new Error(`missing stop-cube fixture view ${name}`);
    return view;
  };
  const prefix = `stop-cube-${time}`;
  const timeKernel = kernels.find(kernel => kernel.id === `${prefix}-stop-time`);
  if (!timeKernel) throw new Error(`missing ${time} stop-time kernel`);
  expect(timeKernel.bindings.find(binding => binding.name === 'rows')?.view).toBe(
    fixtureView(`${time}-rows`)
  );
  expect(timeKernel.bindings.find(binding => binding.name === 'count')?.view).toBe(
    fixtureView('live-stop-count')
  );
  for (const [status, suffix] of [
    ['overflow', 'overflow'],
    ['candidateOverflow', 'candidate-overflow']
  ] as const) {
    const lossReads = kernels
      .filter(kernel => kernel.id.startsWith(`${prefix}-${status}-`))
      .flatMap(kernel => kernel.bindings.filter(binding => binding.access === 'read'));
    expect(lossReads, `${time} ${status} source count`).toHaveLength(3);
    for (const producer of ['stop', 'slice', 'weights']) {
      const view = fixtureView(`${producer}-${suffix}`);
      expect(
        lossReads.some(binding => binding.view === view),
        `${time} ${status} must read ${producer} loss`
      ).toBe(true);
    }
  }
}

function assertDispatchCoverage(
  kernel: WGSLKernelNodeRecord,
  maximumWorkgroupsPerDimension: number
): void {
  for (const dimension of kernel.workgroups) {
    expect(Number.isInteger(dimension), 'integer dispatch dimension').toBe(true);
    expect(dimension, 'positive dispatch dimension').toBeGreaterThan(0);
    expect(dimension, 'bounded dispatch dimension').toBeLessThanOrEqual(
      maximumWorkgroupsPerDimension
    );
  }
  expect(
    kernel.workgroups.reduce((product, count) => product * count, 1) * kernel.workgroupSize[0]
  ).toBeGreaterThanOrEqual(kernel.invocationCount);
}

it('B1 rejects a coverage shader with swapped surface and ground masks', () => {
  const auditCase = POINTCLOUD_AND_DATUM_CASES.find(
    entry => entry.exportName === 'GPUPointCloudCoverageValidity' && entry.name === '257 cells'
  );
  if (!auditCase) throw new Error('missing coverage validity case');
  withRecordedCase(auditCase, (_context, kernels) => {
    const kernel = kernels[0];
    const swapped = {
      ...kernel,
      source: kernel.source.replace(
        /\(coverage & ([12])u\)/g,
        (_match, bit: string) => `(coverage & ${bit === '1' ? '2' : '1'}u)`
      )
    };
    expect(swapped.source).not.toBe(kernel.source);
    expect(() => assertCoverageOutputs(swapped)).toThrow();
  });
});

it('B2 rejects a finish fixture wired to start rows or missing source loss', () => {
  const auditCase = RECIPES_CASES.find(
    entry => entry.exportName === 'addStopValueCubeRecipe' && entry.name.startsWith('finish ')
  );
  if (!auditCase) throw new Error('missing finish stop-cube case');
  withRecordedCase(auditCase, (_context, kernels, views) => {
    const startRows = views.get('start-rows');
    if (!startRows) throw new Error('missing start-row fixture');
    const wrongRows = kernels.map(kernel => ({
      ...kernel,
      bindings: kernel.bindings.map(binding =>
        kernel.id.endsWith('-stop-time') && binding.name === 'rows'
          ? {...binding, view: startRows}
          : binding
      )
    }));
    expect(() => assertStopCubeInputs(wrongRows, views, 'finish')).toThrow();
    const wrongCount = kernels.map(kernel => ({
      ...kernel,
      bindings: kernel.bindings.map(binding =>
        kernel.id.endsWith('-stop-time') && binding.name === 'count'
          ? {...binding, view: startRows}
          : binding
      )
    }));
    expect(() => assertStopCubeInputs(wrongCount, views, 'finish')).toThrow();
    for (const name of [
      'stop-overflow',
      'stop-candidate-overflow',
      'slice-overflow',
      'slice-candidate-overflow',
      'weights-overflow',
      'weights-candidate-overflow'
    ]) {
      const missingView = views.get(name);
      if (!missingView) throw new Error(`missing source loss fixture ${name}`);
      const missingLoss = kernels.map(kernel => ({
        ...kernel,
        bindings: kernel.bindings.filter(binding => binding.view !== missingView)
      }));
      expect(() => assertStopCubeInputs(missingLoss, views, 'finish'), name).toThrow();
    }
  });
});

it('S1 rejects row-covering dispatches that exceed the per-dimension device limit', () => {
  const auditCase = POINTCLOUD_AND_DATUM_CASES.find(
    entry =>
      entry.exportName === 'GPUPointCloudCoverageValidity' &&
      entry.name === 'cells at the binding limit'
  );
  if (!auditCase) throw new Error('missing binding-limit coverage validity case');
  withRecordedCase(auditCase, (context, kernels) => {
    for (const dimension of [0, 1, 2]) {
      for (const invalidCount of [131072, 0, 1.5, NaN]) {
        const workgroups: [number, number, number] = [...kernels[0].workgroups];
        workgroups[dimension] = invalidCount;
        expect(() =>
          assertDispatchCoverage(
            {...kernels[0], workgroups},
            context.device.limits.maxComputeWorkgroupsPerDimension
          )
        ).toThrow();
      }
    }
  });
});

it('CMP-12 registers and compiles both signed-time stop-cube selections', () => {
  const cases = RECIPES_CASES.filter(entry => entry.exportName === 'addStopValueCubeRecipe');
  expect(cases.map(entry => entry.name)).toEqual([
    'start time, live stops, dictionary and source loss',
    'finish time, live stops, dictionary and source loss'
  ]);
  for (const auditCase of cases) {
    withRecordedCase(auditCase, (_context, kernels, views) => {
      const time = auditCase.name.startsWith('start ') ? 'start' : 'finish';
      assertStopCubeInputs(kernels, views, time);
    });
    const result = auditKernelCase(auditCase);
    expect(getKernelAuditResultFailures(result)).toEqual([]);
    expect(result.rejected).toBe(false);
    expect(result.nodeCount).toBeGreaterThan(0);
    expect(result.moduleCount).toBeGreaterThan(0);
  }
});

it('keeps the newly public named tessellation routes in the strict inventory', () => {
  for (const exportName of [
    'addBuildingTessellationReferenceRecipe',
    'addLocalBuildingTessellationRecipe'
  ]) {
    const cases = RECIPES_CASES.filter(entry => entry.exportName === exportName);
    expect(cases, exportName).toHaveLength(1);
    const result = auditKernelCase(cases[0]);
    expect(getKernelAuditResultFailures(result), exportName).toEqual([]);
    expect(result.nodeCount).toBeGreaterThan(0);
  }
});

it('W5-13 and TSC-900 compile explicit validity and rectangle-clip output shapes', () => {
  const cases = GEOMETRY_CASES.filter(
    entry =>
      (entry.exportName === 'GPUGeometryValidity' && ['lines', 'points'].includes(entry.name)) ||
      (entry.exportName === 'GPURectangleClip' &&
        ['lines, every output', 'lines at the binding limit'].includes(entry.name))
  );
  expect(cases).toHaveLength(4);
  for (const auditCase of cases) {
    const result = auditKernelCase(auditCase);
    expect(
      getKernelAuditResultFailures(result),
      `${auditCase.exportName} ${auditCase.name}`
    ).toEqual([]);
    expect(result.nodeCount).toBeGreaterThan(0);
  }
});

it('W5-14 accepts zero track pairs without declaring work', () => {
  const context = createKernelAuditContext('zero-track-pairs');
  const {graph, view, construct} = context;
  try {
    const contributor = construct(GPUTrackSimilarity, {
      linesA: {
        kind: 'lines',
        positions: view('float32x2', 2),
        lineOffsets: view('uint32', 2)
      },
      pairA: view('uint32', 0),
      pairB: view('uint32', 0),
      output: {hausdorff: view('float32', 0), pairStatus: view('uint32', 0)}
    });
    expect(contributor.props.pairA.length).toBe(0);
    expect(contributor.props.output.hausdorff?.length).toBe(0);
    expect(contributor.getCommandNodes(graph)).toEqual([]);
    graph.add(contributor);
    expect(getDeclaredNodes(context)).toHaveLength(0);
  } finally {
    context.device.destroy();
  }
});

it('W5-14 registers and compiles the one-slice rolling-window lower endpoint', () => {
  const cases = RTS_TIME_SERIES_AUDIT_CASES.filter(
    entry => entry.exportName === 'GPURasterRollingMean' && entry.name === 'rts06-minimum-window'
  );
  expect(cases).toHaveLength(1);
  const result = auditKernelCase(cases[0]);
  expect(getKernelAuditResultFailures(result)).toEqual([]);
  expect(result.nodeCount).toBeGreaterThan(0);
});

it('W5-14 pins the migrated numeric ranges, construction phase and refusal atomicity', () => {
  // Independent endpoints from the public contracts: zero pairs are valid; a one-slice mean
  // is the sample itself. The upper-bound probes supply maximum+1, without allocating GPU data.
  for (const {cases, exportName, name, path, received, minimum, maximum} of [
    {
      cases: CELLS_AND_JOINS_CASES,
      exportName: 'GPUTrackSimilarity',
      name: 'pairs above the maximum',
      path: 'pairA.length',
      received: 16777216,
      minimum: 0,
      maximum: 16777215
    },
    {
      cases: RTS_TIME_SERIES_AUDIT_CASES,
      exportName: 'GPURasterRollingMean',
      name: 'rts06-window-above-maximum',
      path: 'windowLength',
      received: 129,
      minimum: 1,
      maximum: 128
    }
  ]) {
    const auditCase = cases.find(entry => entry.exportName === exportName && entry.name === name);
    if (!auditCase) throw new Error(`missing upper-bound fixture ${exportName} ${name}`);
    const rejection = getExpectedRejection(auditCase);
    expect(rejection).toMatchObject({
      limit: maximum,
      phase: 'construction',
      validation: {path, received, expected: {minimum, maximum}}
    });
    const result = auditKernelCase(auditCase);
    expect(getKernelAuditResultFailures(result), exportName).toEqual([]);
    expect(result.rejected).toBe(true);
    expect(result.moduleCount).toBe(0);
    const observation = result.rejectionObservation;
    expect(observation?.error).toBeInstanceOf(AnalysisValidationError);
    expect(observation?.error).toMatchObject({
      code: 'INTEGER_RANGE',
      path,
      received,
      expected: {minimum, maximum}
    });
    expect(observation?.phase).toBe('construction');
    expect(observation?.after).toEqual(observation?.before);
  }
});

it('W5-12 binds every selected relief output for all three shapes on both grid sizes', () => {
  const cases = TERRAIN_CASES.filter(entry => entry.exportName === 'GPUSimpleLocalRelief');
  expect(cases).toHaveLength(6);
  for (const auditCase of cases) {
    const context = createKernelAuditContext('simple-relief-fixture');
    try {
      recordWGSLKernelNodes(context.graph);
      auditCase.build(context);
      // The upstream elevation-copy node has no analysis kernel record.
      const kernels = (getDeclaredNodes(context) as {id: string}[]).flatMap(node => {
        const kernel = getWGSLKernelNodeRecord(context.graph, node.id);
        return kernel ? [kernel] : [];
      });
      expect(kernels.length).toBeGreaterThan(0);
      const writers = kernels.flatMap(kernel =>
        kernel.bindings.filter(binding => binding.access === 'read_write')
      );
      const reliefWriters = writers.filter(binding =>
        binding.view.buffer.id.startsWith('relief-output-')
      );
      const validityWriters = writers.filter(binding =>
        binding.view.buffer.id.startsWith('validity-output-')
      );
      expect(reliefWriters, auditCase.name).toHaveLength(
        auditCase.name.startsWith('validity only') ? 0 : 1
      );
      expect(validityWriters, auditCase.name).toHaveLength(
        auditCase.name.startsWith('relief only') ? 0 : 1
      );
      for (const binding of reliefWriters) {
        expect(binding.name).toBe('deviation');
        expect(binding.view.format).toBe('float32');
      }
      for (const binding of validityWriters) {
        expect(binding.name).toBe('validityOutput');
        expect(binding.view.format).toBe('uint32');
      }
    } finally {
      context.device.destroy();
    }
    const result = auditKernelCase(auditCase);
    expect(getKernelAuditResultFailures(result), auditCase.name).toEqual([]);
    expect(result.nodeCount).toBeGreaterThan(0);
  }
});

it('UNI-01 registers coverage-to-validity inputs, ground bit, writers and dispatch', () => {
  const cases = POINTCLOUD_AND_DATUM_CASES.filter(
    entry => entry.exportName === 'GPUPointCloudCoverageValidity'
  );
  expect(cases.map(entry => entry.name)).toEqual(['257 cells', 'cells at the binding limit']);
  for (const auditCase of cases) {
    const context = createKernelAuditContext('coverage-validity-fixture');
    try {
      recordWGSLKernelNodes(context.graph);
      auditCase.build(context);
      expect(getDeclaredNodes(context)).toHaveLength(1);
      const kernel = getWGSLKernelNodeRecord(context.graph, 'coverage-validity-audit');
      expect(kernel).toBeDefined();
      if (!kernel) throw new Error('missing coverage validity kernel');
      const rows = auditCase.name === '257 cells' ? 257 : context.maximumRows(4);
      expect(kernel.invocationCount).toBe(rows);
      expect(
        kernel.bindings.map(({name, access, view}) => [name, access, view.format, view.length])
      ).toEqual([
        ['coverage', 'read', 'uint32', rows],
        ['surfaceValidity', 'read_write', 'uint32', rows],
        ['terrainValidity', 'read_write', 'uint32', rows],
        ['canopyValidity', 'read_write', 'uint32', rows]
      ]);
      // Coverage flags are independent: surface=1, ground=2. Both terrain and canopy use ground.
      expect(kernel.source).toContain('let coverage = readCoverage(index)');
      expect(kernel.source).toContain('(coverage & 1u) != 0u');
      expect(kernel.source).toContain('(coverage & 2u) != 0u');
      expect(kernel.source).toContain('writeSurfaceValidity(index, select(0u, 1u,');
      expect(kernel.source).toContain('writeTerrainValidity(index, ground)');
      expect(kernel.source).toContain('writeCanopyValidity(index, ground)');
      assertCoverageOutputs(kernel);
      expect(kernel.body).toBe('runPointCloudRow(index, localInvocationIndex);');
      expect(kernel.workgroupSize).toEqual([256, 1, 1]);
      if (rows === 257) expect(kernel.workgroups).toEqual([2, 1, 1]);
      assertDispatchCoverage(kernel, context.device.limits.maxComputeWorkgroupsPerDimension);
    } finally {
      context.device.destroy();
    }
    const result = auditKernelCase(auditCase);
    expect(getKernelAuditResultFailures(result)).toEqual([]);
    expect(result.nodeCount).toBe(1);
    expect(result.moduleCount).toBe(1);
  }
});
