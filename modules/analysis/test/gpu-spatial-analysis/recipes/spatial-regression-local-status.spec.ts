// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {getGPUGeographicallyWeightedRegressionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regression/index';
import {RecipeTestFixture} from './recipe-harness';

it('RCP-104 exposes successful and singular local fit row codes through the GWR stage', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'gwr-stage-codes');
  try {
    const codes = fixture.output('codes', 'uint32', 10);
    codes.buffer.write(new Uint32Array(10).fill(77));
    const localStatus = fixture.graph.createDataView(codes.view.buffer, {
      format: 'uint32',
      length: 8,
      byteOffset: 8
    });
    const olsCode = fixture.output('ols-code', 'uint32', 3);
    const diagnosticCode = fixture.output('diagnostic-code', 'uint32', 2);
    olsCode.buffer.write(Uint32Array.of(77, 77, 77));
    diagnosticCode.buffer.write(Uint32Array.of(77, 77));
    const localParameters = getGPUGeographicallyWeightedRegressionParameterValues(
      {kernel: 'bisquare', bandwidthMode: 'fixed', bandwidths: [3]},
      1
    );
    const recipe = addSpatialRegressionRecipe(fixture.graph, {
      predictors: fixture.input('predictors', Float32Array.of(0, 1, 2, 3), 'float32', 4),
      response: fixture.input('response', Float32Array.of(1, 3, 5, 7), 'float32', 4),
      predictorCount: 1,
      weights: {
        offsets: fixture.input('offsets', Uint32Array.of(0, 1, 2, 3, 4), 'uint32', 5),
        neighbors: fixture.input('neighbors', Uint32Array.of(1, 0, 1, 2), 'uint32', 4),
        weights: fixture.input('weights', Float32Array.of(1, 1, 1, 1), 'float32', 4)
      },
      parameters: fixture.parameterBuffer('parameters', 'float32', 64),
      localFits: {
        positions: fixture.input(
          'positions',
          Float32Array.of(0, 0, 1, 0, 0, 1, 100, 100),
          'float32x2',
          4
        ),
        parameters: fixture.input(
          'gwr-parameters',
          localParameters,
          'float32',
          localParameters.length
        ),
        maximumBandwidthCount: 1,
        maximumNeighborCount: 4,
        refinement: false
      },
      output: {
        localFits: {localStatus},
        ols: {status: {code: olsCode.view}},
        diagnostics: {status: {code: diagnosticCode.view}}
      }
    });
    const localStage = recipe.status.stages.find(
      stage => stage.stage === 'geographically-weighted-regression'
    )!;
    expect(localStage.rowCodes?.length).toBe(4);
    expect(localStage.rowCodes?.byteOffset).toBe(8);
    expect(recipe.localFits?.localStatus === localStatus).toBe(true);
    for (const stage of recipe.status.stages.filter(stage => stage.code))
      expect(stage.code?.length).toBe(1);
    fixture.run();
    // The first three locations see three distinct x values (rank two for [1,x]).
    // The distant location sees itself alone: its local design is rank one.
    expect(await fixture.readUint32(codes, 10)).toEqual([77, 77, 0, 0, 0, 1, 77, 77, 77, 77]);
    expect((await fixture.readUint32(olsCode, 3)).slice(1)).toEqual([77, 77]);
    expect((await fixture.readUint32(diagnosticCode, 2)).slice(1)).toEqual([77]);
  } finally {
    fixture.destroy();
  }
});
