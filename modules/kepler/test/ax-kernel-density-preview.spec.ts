// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {test, vi} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  AnalysisPreviewSession,
  KeplerDatasetRegistry,
  getCatalogBinding,
  runCatalogOperation,
  type CatalogRunResult
} from '../src/index';

function check(condition: unknown, message: string, evidence: unknown): asserts condition {
  if (!condition)
    throw new Error(`Retained kernel density: ${message}\n${JSON.stringify(evidence, null, 2)}`);
}

test('kernel density reuses its compiled graph for bandwidth changes, matches the CPU and stays admitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const binding = getCatalogBinding('kernel-density-grid')!;
  const inputs = {
    points: {
      type: 'FeatureCollection' as const,
      features: [
        {geometry: {type: 'Point' as const, coordinates: [0, 0]}, properties: {weight: 1}},
        {geometry: {type: 'Point' as const, coordinates: [10, 10]}, properties: {weight: 2}},
        {geometry: {type: 'Point' as const, coordinates: [5, 3]}, properties: {weight: 3}}
      ]
    }
  };
  const parameters = {
    coordinateSystem: 'planar',
    cellSize: 2,
    bandwidth: 1,
    padding: 0,
    kernel: 'gaussian',
    weightProperty: 'weight',
    includeEmptyCells: true
  };
  const plan = binding.planPreview!(inputs, parameters);
  const registry = new KeplerDatasetRegistry({
    memoryBudgetBytes: 3 * (plan.gpuByteLength + plan.cpuByteLength)
  });
  const session = new AnalysisPreviewSession({device, registry, binding});
  const compile = vi.spyOn(GPUCommandGraph.prototype, 'compileAsync');
  const gpuDevice = device.handle as GPUDevice;
  const createBuffer = gpuDevice.createBuffer.bind(gpuDevice);
  let liveBytes = 0;
  let peakBytes = 0;
  const destruction: {mockRestore(): void}[] = [];
  const allocation = vi.spyOn(gpuDevice, 'createBuffer').mockImplementation(properties => {
    const buffer = createBuffer(properties);
    liveBytes += buffer.size;
    peakBytes = Math.max(peakBytes, liveBytes);
    const destroy = buffer.destroy.bind(buffer);
    let released = false;
    destruction.push(
      vi.spyOn(buffer, 'destroy').mockImplementation(() => {
        if (!released) liveBytes -= buffer.size;
        released = true;
        destroy();
      })
    );
    check(
      liveBytes <= registry.memoryUsage.reservedBytes.gpu,
      'device allocation exceeded admission',
      {liveBytes, peakBytes, memory: registry.memoryUsage, buffer: properties}
    );
    return buffer;
  });
  let displayed: CatalogRunResult | undefined;
  let firstValues: readonly (number | string | boolean | null)[] | undefined;
  try {
    for (const kernel of ['gaussian', 'quartic']) {
      for (const bandwidth of [1, 3]) {
        const values = {...parameters, kernel, bandwidth};
        const committed = await session.request({
          inputs,
          parameters: values,
          revisionKey: 'fixture-1',
          isCurrent: () => true,
          commit: result => {
            displayed = result;
          }
        });
        check(committed && displayed, 'no accepted preview', {kernel, bandwidth});
        const expected = runCatalogOperation(binding, values, inputs).tables[0];
        const actual = displayed.tables[0];
        check(actual.rowCount === expected.rowCount, 'row count differs', {
          kernel,
          bandwidth,
          actual,
          expected
        });
        const differences = expected.columns['density'].flatMap((value, row) => {
          const received = actual.columns['density'][row];
          return typeof value !== 'number' ||
            typeof received !== 'number' ||
            Math.abs(value - received) > 1e-7 + Math.abs(value) * 2e-5
            ? [{row, value, received}]
            : [];
        });
        check(differences.length === 0, 'CPU/GPU density mismatch', {
          kernel,
          bandwidth,
          differences
        });
        check(
          JSON.stringify(actual.columns['cell_id']) === JSON.stringify(expected.columns['cell_id']),
          'cell identities differ',
          {actual, expected}
        );
        if (kernel === 'gaussian' && bandwidth === 1) firstValues = actual.columns['density'];
        if (kernel === 'gaussian' && bandwidth === 3)
          check(
            JSON.stringify(firstValues) !== JSON.stringify(actual.columns['density']),
            'bandwidth did not change the surface',
            {actual}
          );
      }
      check(
        compile.mock.calls.length === (kernel === 'gaussian' ? 1 : 2),
        'live bandwidth recompiled or kernel failed to rebuild',
        {kernel, compilations: compile.mock.calls.length}
      );
    }
  } finally {
    await session.destroy();
    allocation.mockRestore();
    compile.mockRestore();
    for (const spy of destruction) spy.mockRestore();
  }
  check(registry.memoryUsage.totalBytes === 0 && liveBytes === 0, 'cleanup retained bytes', {
    liveBytes,
    peakBytes,
    memory: registry.memoryUsage
  });
});
