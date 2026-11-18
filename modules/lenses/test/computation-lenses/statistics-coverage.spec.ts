// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {listLensScenarios} from '../../src/computation-lenses/registry-implementation';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/statistics-coverage.lens';
import {requireTestDevice} from '@luma-contributors/analysis/testing';

const scenarios = listLensScenarios().filter(
  scenario => scenario.dataset?.id === 'statistics-coverage-illustrative/1'
);

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
  // Every synthetic fit has sufficient observations, finite support and a valid design.
  for (const success of outputs.filter(snapshot =>
    ['status', 'status-code', 'folds-status', 'fitStatus', 'metricStatus'].includes(snapshot.name)
  )) {
    expect(Array.from(success.values), `${scenario.element}: ${success.name}`).toEqual(
      Array(success.values.length).fill(0)
    );
  }
}, 120000);
