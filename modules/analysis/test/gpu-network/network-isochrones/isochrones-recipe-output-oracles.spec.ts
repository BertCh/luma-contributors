// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createIsochroneOracleHarness, expectOracleValues} from './isochrones-output-oracle-utils';
import {createRecipeSquareOracle} from './isochrones-recipe-output-oracle-fixture';

it('R1 recipe square publishes intermediate edge loss and clears it on a fitting replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const {recipe, output, edgeOverflow, isochroneCostBreaks} = createRecipeSquareOracle(harness);
  const stage = recipe.status.stages.find(entry => entry.stage === 'isochrone-band-rings');
  // Read the exact views exposed by the recipe stage on both submissions.
  expect(stage?.status.candidateOverflow === edgeOverflow).toBe(true);
  expect(stage?.status.overflow === output.status.overflow).toBe(true);
  const result = await harness.run(undefined, first => {
    // Four edges are needed; the first two form an open chain and no closed ring.
    expectOracleValues(first['classes'], [0, 0, 0, 0]);
    expectOracleValues(first['ringCount'], [0]);
    expectOracleValues(first['ringOverflow'], [0]);
    expectOracleValues(first['ringEdgeOverflow'], [1]);
    // Exclude the zero-cost square from band 0. The empty selected band fits capacity 2.
    harness.write(isochroneCostBreaks, [-1]);
  });
  expectOracleValues(result['classes'], [1, 1, 1, 1]);
  expectOracleValues(result['ringCount'], [0]);
  expectOracleValues(result['ringOverflow'], [0]);
  expectOracleValues(result['ringEdgeOverflow'], [0]);
});
