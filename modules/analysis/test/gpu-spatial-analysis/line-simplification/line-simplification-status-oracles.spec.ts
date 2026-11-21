// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineSimplificationOperation} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification-operation';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {addOracleOperation, createOracleFixture} from '../global-coordinates/output-oracle-fixture';

it.each([
  'endpoints',
  'douglas-peucker',
  'visvalingam',
  'visvalingam complete'
] as const)('simplification contributor and operation overwrite full hand-derived status: %s', async scene => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  // Endpoints have no undecided interior. DP chooses root row 1 (tie with row 3),
  // leaving child [1,4] with two interiors beyond finishSpanLimit 1. Visvalingam
  // removes row 1 at area 1/2, leaving row 2 undecided. Thus completion is 1/0/0
  // and iterations exactly 1, independently of the production stopping algorithm.
  const coordinates =
    scene === 'endpoints'
      ? [0, 0, 1, 0]
      : scene === 'douglas-peucker'
        ? [0, 0, 1, 1, 2, 0, 3, 1, 4, 0]
        : [0, 0, 1, 1, 2, 1, 3, 0];
  const rows = coordinates.length / 2;
  const operation = new GPULineSimplificationOperation({
    positions: 'positions',
    lineOffsets: 'offsets',
    parameters: 'parameters',
    importance: 'importance',
    rowCount: rows,
    lineCount: 1,
    capacity: rows,
    method: scene.startsWith('visvalingam') ? 'visvalingam' : 'douglas-peucker',
    maximumIterations: scene === 'visvalingam complete' ? 3 : 1,
    finishSpanLimit: 1,
    output: {
      ids: 'ids',
      lineCounts: 'lineCounts',
      status: {
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        converged: 'converged',
        iterationCount: 'iterationCount'
      }
    }
  });
  const outputs = addOracleOperation(fixture, operation, {
    positions: fixture.input('float32x2', coordinates),
    lineOffsets: fixture.input('uint32', [0, rows]),
    parameters: fixture.input(
      'float32',
      Array.from(getGPULineSimplificationParameterValues({tolerance: 0}))
    )
  });
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of Object.values(outputs)) output.poison();
      submitGraph(device, compiled, undefined);
      // At tolerance zero all positive/endpoint/undecided importances are kept.
      expect(await outputs['count'].read()).toEqual([rows]);
      expect(await outputs['requiredCount'].read()).toEqual([rows]);
      expect(await outputs['lineCounts'].read()).toEqual([rows]);
      expect(await outputs['overflow'].read()).toEqual([0]);
      expect(await outputs['converged'].read()).toEqual([
        Number(scene === 'endpoints' || scene === 'visvalingam complete')
      ]);
      expect(await outputs['iterationCount'].read()).toEqual([
        scene === 'visvalingam complete' ? 2 : 1
      ]);
      if (scene === 'visvalingam complete') {
        // The sole second interior's triangle has area 3/2; no third removal exists.
        expect(await outputs['importance'].read()).toEqual([Infinity, 0.5, 1.5, Infinity]);
      }
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('zero-capacity endpoint selection clears a poisoned count word', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const count = fixture.output('uint32', 1);
  const requiredCount = fixture.output('uint32', 1);
  const overflow = fixture.output('uint32', 1);
  const lineCounts = fixture.output('uint32', 1);
  const paddedIds = fixture.output('uint32', 1);
  // A zero-length logical IDs view still has valid four-byte backing storage.
  // This exercises the contributor's zero-capacity publisher rather than a graph
  // executor replacing its empty output. Two mandatory endpoints cannot fit.
  const views = {
    positions: fixture.input('float32x2', [0, 0, 1, 0]),
    lineOffsets: fixture.input('uint32', [0, 2]),
    parameters: fixture.input(
      'float32',
      Array.from(getGPULineSimplificationParameterValues({tolerance: 0}))
    ),
    importance: fixture.output('float32', 2).view,
    ids: fixture.graph.createDataView(paddedIds.view.buffer, {format: 'uint32', length: 0}),
    count: count.view,
    requiredCount: requiredCount.view,
    overflow: overflow.view,
    lineCounts: lineCounts.view
  };
  const operation = new GPULineSimplificationOperation({
    positions: 'positions',
    lineOffsets: 'lineOffsets',
    parameters: 'parameters',
    importance: 'importance',
    rowCount: 2,
    lineCount: 1,
    capacity: 0,
    maximumIterations: 1,
    output: {
      ids: 'ids',
      lineCounts: 'lineCounts',
      status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
    }
  });
  fixture.graph.add(
    operation.createContributor(name => views[name as keyof typeof views] as never)
  );
  const compiled = fixture.graph.compile();
  try {
    for (let submission = 0; submission < 2; submission++) {
      for (const output of [count, requiredCount, overflow, lineCounts]) output.poison();
      submitGraph(device, compiled, undefined);
      expect(await count.read()).toEqual([0]);
      expect(await requiredCount.read()).toEqual([2]);
      expect(await overflow.read()).toEqual([1]);
      expect(await lineCounts.read()).toEqual([2]);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
