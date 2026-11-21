// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createIsochroneOracleHarness} from './isochrones-output-oracle-utils';
import {createRecipeSquareOracle} from './isochrones-recipe-output-oracle-fixture';

it('R1 recipe ring status preserves the intermediate edge-loss view', () => {
  const device = createNullWebGPUDevice();
  const harness = createIsochroneOracleHarness(device);
  try {
    const {recipe, output, edgeOverflow} = createRecipeSquareOracle(harness);
    const stage = recipe.status.stages.find(entry => entry.stage === 'isochrone-band-rings');
    expect(stage?.status.count).toBe(output.status.count);
    expect(stage?.status.overflow).toBe(output.status.overflow);
    expect(stage?.status.candidateOverflow === edgeOverflow).toBe(true);
  } finally {
    harness.destroy();
    device.destroy();
  }
});
