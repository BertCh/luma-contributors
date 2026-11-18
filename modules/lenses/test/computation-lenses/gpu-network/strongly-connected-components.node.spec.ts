// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test, vi} from 'vitest';
import * as lensViewSnapshot from '../../../src/computation-lenses/lens-view-snapshot';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {validateGPUComputationRecord} from '@luma-contributors/analysis/computation-record';
import '../../../src/computation-lenses/scenarios/gpu-network/index';
import {getLensScenario} from '../../../src/computation-lenses/registry-implementation';
import {
  extractLensExample,
  findNonPublicImports
} from '../../../src/computation-lenses/lens-example';
import {NETWORK_FORMULAS} from '../../../src/gpu-network/computation-pack/network-formulas';

const SCENARIO_ID = 'gpu-network.network-strongly-connected-components';

test('directed SCC lens registers synthetic provenance and exposes the algorithm graph', async () => {
  const scenario = getLensScenario(SCENARIO_ID)!;
  expect(scenario.element).toBe('GPUNetworkStronglyConnectedComponents');
  expect(scenario.dataset?.id).toBe('synthetic-four-node-directed-components-v1');
  expect(scenario.rungs).toEqual(['result', 'datum', 'kernel', 'graph']);
  expect(scenario.defaultRung).toBe('result');
  expect(NETWORK_FORMULAS['GPUNetworkStronglyConnectedComponents']).toMatchObject({
    formula: 'none'
  });
  const device = createNullWebGPUDevice();
  const snapshotSpy = vi.spyOn(lensViewSnapshot, 'addLensViewSnapshot');
  try {
    const loaded = await scenario.load();
    const record = await loaded.describeStructure!(device);
    expect(validateGPUComputationRecord(record).issues).toEqual([]);
    expect(snapshotSpy).toHaveBeenCalledTimes(1);
    // Snapshot names come from the view-map keys, not the uploaded buffer IDs.
    expect(Object.keys(snapshotSpy.mock.calls[0][2].views)).toEqual([
      'offsets',
      'neighbors',
      'strongComponents',
      'converged',
      'iterationCount',
      'iterationLimitReached'
    ]);
    const nodes = record.graphs[0].nodes.filter(node => node.kind === 'compute');
    expect(nodes.filter(node => node.op === 'GPUNetworkStronglyConnectedComponents')).toHaveLength(
      194
    );
    for (const variant of [
      'trim-mark',
      'trim-remove',
      'color',
      'back',
      'finalize',
      'controller',
      'publish'
    ]) {
      expect(
        nodes.some(node => node.id.endsWith(`-${variant}`)),
        variant
      ).toBe(true);
    }
  } finally {
    snapshotSpy.mockRestore();
    device.destroy();
  }
});

test('directed SCC example constructs the public contributor and returns caller-owned outputs', () => {
  const source = readFileSync(
    new URL(
      '../../../src/computation-lenses/scenarios/gpu-network/network-strongly-connected-components.example.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(findNonPublicImports(source)).toEqual([]);
  const example = extractLensExample(source);
  expect(example.lineCount).toBeLessThanOrEqual(25);
  expect(example.code).toContain('new GPUNetworkStronglyConnectedComponents(properties)');
  expect(example.code).toMatch(/return properties.output;$/);
});
