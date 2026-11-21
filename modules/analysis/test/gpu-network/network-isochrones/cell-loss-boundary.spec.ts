// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createCellLossBoundaryFixture} from './cell-loss-boundary-fixture';
import {createIsochroneOracleHarness, expectOracleValues} from './isochrones-output-oracle-utils';

it.each([
  false,
  true
])('propagates cell loss to assembled geometry and recovers (candidateOnly=%s, CPL-12)', async candidateOnly => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const fixture = createCellLossBoundaryFixture(harness, candidateOnly);
  const result = await harness.run({getCommandNodes: () => fixture.nodes}, first => {
    expectOracleValues(first['outlineOverflow'], [0]);
    expectOracleValues(first['outlineCandidateOverflow'], [1]);
    expectOracleValues(first['ringCount'], [0]);
    expectOracleValues(first['ringOverflow'], [0]);
    expectOracleValues(first['ringCandidateOverflow'], [1]);
    expectOracleValues(first['ringComplete'], [0]);
    harness.write(fixture.table.status.overflow, [0]);
    harness.write(fixture.table.status.candidateOverflow, [0]);
  });
  // The same declaration now receives one complete square: four edges, one closed ring.
  expectOracleValues(result['outlineCount'], [4]);
  expectOracleValues(result['outlineOverflow'], [0]);
  expectOracleValues(result['outlineCandidateOverflow'], [0]);
  expectOracleValues(result['ringCount'], [1]);
  expectOracleValues(result['ringOverflow'], [0]);
  expectOracleValues(result['ringCandidateOverflow'], [0]);
  expectOracleValues(result['ringComplete'], [1]);
});

it.each([
  'count',
  'candidateOverflow'
] as const)('replays disjoint cell status slices in one allocation (%s, B6)', async destination => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const fixture = createCellLossBoundaryFixture(harness, false, destination);
  const initial = new Array(128).fill(0);
  initial[0] = 1;
  harness.write(fixture.sharedStatusBacking!, initial);
  const result = await harness.run({getCommandNodes: () => fixture.nodes}, first => {
    expectOracleValues([first['sharedStatus'][0]], [1]);
    expectOracleValues([first['sharedStatus'][64]], [destination === 'count' ? 4 : 1]);
    expectOracleValues(first['ringCount'], [0]);
    expectOracleValues(first['ringCandidateOverflow'], [1]);
    expectOracleValues(first['ringComplete'], [0]);
    harness.write(fixture.sharedStatusBacking!, new Array(128).fill(0));
    harness.write(fixture.table.status.candidateOverflow, [0]);
  });
  expectOracleValues([result['sharedStatus'][0]], [0]);
  expectOracleValues([result['sharedStatus'][64]], [destination === 'count' ? 4 : 0]);
  expectOracleValues(result['ringCount'], [1]);
  expectOracleValues(result['ringCandidateOverflow'], [0]);
  expectOracleValues(result['ringComplete'], [1]);
});
