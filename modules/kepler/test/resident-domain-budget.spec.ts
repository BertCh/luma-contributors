// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {ResidentScopeBinding} from '../../../examples/deck/kepler-compute/src/resident-scope-binding';

test('domain inspection reserves per-column scratch for the default 100,000-row sample', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 128 * 1024 * 1024});
  const names = ['longitude', 'latitude', 'value', 'time', 'targetLongitude', 'targetLatitude'];
  const dataset = session.importArrowDataset({
    id: 'sample',
    table: arrow.tableFromArrays(
      Object.fromEntries(names.map(name => [name, new Float32Array(100000)]))
    )
  });
  const props = {
    session,
    datasetId: dataset.id,
    fields: Object.fromEntries(
      names.map(name => [name, {kind: 'range' as const, dimension: name, domain: [0, 1]}])
    ),
    onUpdate() {},
    onUnsupported() {}
  };
  try {
    const baseline = session.registry.memoryUsage.totalBytes;
    const binding = new ResidentScopeBinding(props);
    await binding.destroy(Promise.resolve());
    expect(session.registry.memoryUsage.totalBytes).toBe(baseline);
    expect(
      () => new ResidentScopeBinding({...props, inspectorWorkingMemoryBudgetBytes: 1024})
    ).toThrow(/working memory reservation/);
    await expect.poll(() => session.registry.memoryUsage.totalBytes).toBe(baseline);
  } finally {
    await session.destroy(Promise.resolve());
  }
});
