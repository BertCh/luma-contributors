// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {expect, test} from 'vitest';
import {listLensScenarios} from '../../src/computation-lenses/registry-implementation';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/analysis-spacing.lens';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/analysis-contiguity-statistics.lens';

const scenarios = listLensScenarios().filter(
  scenario => scenario.dataset?.id === 'synthetic-analysis-profile/1'
);

test.each(
  scenarios
)('$element records its real graph and hand-computed profile', async scenario => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The runner throws a readable report if spacing or contiguity disagrees with its oracle.
  const record = await (await scenario.load()).run(device);
  const snapshots = record.captures?.snapshots ?? [];
  expect(
    snapshots.some(capture => capture.role === 'output'),
    scenario.element
  ).toBe(true);
  expect(record.graphs.length, scenario.element).toBeGreaterThan(0);
  expect(
    record.refs.formulas.some(formula => formula.id.startsWith('analysis-')),
    scenario.element
  ).toBe(true);
}, 120000);
