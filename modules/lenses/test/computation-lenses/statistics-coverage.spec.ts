// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {listLensScenarios} from '../../src/computation-lenses/registry-implementation';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/statistics-coverage.lens';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {
  getStatisticsCaptureFailures,
  STATISTICS_STATUS_CAPTURES
} from './statistics-capture-expectations';

const scenarios = listLensScenarios().filter(
  scenario => scenario.dataset?.id === 'statistics-coverage-illustrative/1'
);

test('statistics scenarios cover the independently enumerated capture contracts', () => {
  expect(scenarios.map(scenario => scenario.element).sort()).toEqual(
    Object.keys(STATISTICS_STATUS_CAPTURES).sort()
  );
});

test.each(scenarios)('$element executes and captures its synthetic result', async scenario => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await (await scenario.load()).run(device);
  const snapshots = result.captures?.snapshots ?? [];
  expect(snapshots.length).toBeGreaterThan(0);
  const outputs = snapshots.filter(snapshot => !snapshot.name.startsWith('input-'));
  expect(outputs.length).toBeGreaterThan(0);
  for (const snapshot of outputs) {
    expect(snapshot.values.length, snapshot.name).toBeGreaterThan(0);
  }
  expect(getStatisticsCaptureFailures(scenario.element, snapshots)).toEqual([]);
}, 120000);
