// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createLongRoadSamplingFixture} from './long-road-sampling-fixture';
import {createIsochroneOracleHarness, expectOracleValues} from './isochrones-output-oracle-utils';

it('keeps the midpoint when a long road reaches the three-sample cap (WGS-10)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const fixture = createLongRoadSamplingFixture(harness);
  const result = await harness.run({getCommandNodes: () => fixture.nodes});
  // Independent hand derivation and row-walking reference are pinned in the Node case.
  expectOracleValues(result['values'], [1, 1, 1]);
});
