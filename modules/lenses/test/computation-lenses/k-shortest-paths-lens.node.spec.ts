// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import {describeExpansion} from '../../src/computation-lenses/scenarios/gpu-network/expansion.lens';

it('k-shortest paths lens constructs a valid graph with the plural counts output', async () => {
  const device = createNullWebGPUDevice();
  try {
    const record = await describeExpansion(device, 'k-shortest-paths');
    expect(validateGPUComputationRecord(record)).toEqual({valid: true, issues: []});
    expect(JSON.stringify(record.graphs)).toContain('counts');
    expect(record.graphs[0].nodes.length).toBeGreaterThan(0);
  } finally {
    device.destroy();
  }
});
