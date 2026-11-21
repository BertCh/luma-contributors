// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {addKeplerPointQuantileLISA} from '../src/analysis/point-quantile-lisa';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import {getKeplerPointWeightSpatialContext} from '../src/analysis/point-weight-coordinates';
import type {KeplerPointWeightsBinding} from '../src/analysis/point-weights';

test.each([
  ['greater', false, false],
  ['two-sided', false, false],
  ['geoda-legacy', false, false],
  ['greater', true, false],
  ['two-sided', true, false],
  ['geoda-legacy', true, false],
  ['greater', true, true]
] as const)('public point inference ports for %s candidate loss=%s recipe=%s', async (alternative, candidateLoss, recipe) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new CatalogGpuSession(
    device,
    `point-ports-${alternative}-${candidateLoss}-${recipe}`
  );
  let result: {destroyParameters(): void} | undefined;
  try {
    const source = {datasetId: 'three-selected-points', revision: 1};
    const weightsSettings = {
      kind: 'knn' as const,
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      k: 1,
      coordinateSpace: 'longitude-latitude' as const,
      metric: 'great-circle' as const,
      units: 'meters' as const
    };
    const status = {
      overflow: session.input('weights-overflow', Uint32Array.of(0), 'uint32'),
      requiredCount: session.input('weights-required', Uint32Array.of(6), 'uint32'),
      invalidCount: session.input('weights-invalid', Uint32Array.of(0), 'uint32')
    };
    const mask = session.input('mask', new Uint32Array(6).fill(1), 'uint32');
    const pointWeights: KeplerPointWeightsBinding = {
      source,
      settings: weightsSettings,
      status,
      coordinateMask: mask,
      weights: {
        offsets: session.input('offsets', Uint32Array.of(0, 1, 2, 3, 4, 5, 6), 'uint32'),
        neighbors: session.input('neighbors', Uint32Array.of(1, 0, 3, 2, 5, 4), 'uint32'),
        weights: session.input('weights', new Float32Array(6).fill(1), 'float32'),
        status: {
          ...status,
          candidateOverflow: session.input(
            'candidate-overflow',
            Uint32Array.of(Number(candidateLoss)),
            'uint32'
          )
        },
        spatialContext: getKeplerPointWeightSpatialContext(weightsSettings),
        information: {
          rowCount: 6,
          transform: 'binary',
          hasSelfWeights: false,
          symmetric: true,
          provenance: {source: 'GPUNeighborSearch', criterion: 'knn', k: 1}
        }
      }
    };
    const settings = {
      quantileCount: 2,
      selectedQuantile: 2,
      seed: 7,
      permutationCount: 39,
      significanceLevel: 0.05,
      multipleTesting: 'none' as const,
      islands: 'exclude' as const,
      quantileMethod: 'type-5' as const,
      alternative
    };
    const tested = session.output('tested', 'uint32', 6);
    const analysisCount = session.output('analysis-count', 'uint32', 1);
    const pseudoPValues = session.output('p-values', 'float32', 6);
    const overflow = session.output('overflow', 'uint32', 1);
    const outputs = {
      tested: tested.view,
      analysisCount: analysisCount.view,
      pseudoPValues: pseudoPValues.view,
      status: {overflow: overflow.view}
    };
    const values = session.input('values', Float32Array.of(1, 2, 3, 4, 5, 6), 'float32');
    result = recipe
      ? addKeplerAnalysisRecipe(
          session.graph,
          {
            schemaVersion: 1,
            algorithmVersion: 1,
            id: 'point-recipe-loss',
            source,
            population: 'all',
            backendPolicy: 'require-gpu',
            operation: {
              kind: 'quantile-lisa-points',
              column: 'value',
              weights: weightsSettings,
              ...settings
            }
          },
          {source, columns: {value: values}, pointWeights, outputs: {quantileLisa: outputs}}
        )
      : addKeplerPointQuantileLISA(session.graph, {
          source,
          population: 'all',
          values,
          mask,
          pointWeights,
          settings,
          outputs
        });
    session.run();
    // Type-5 median is 3.5. Rows 3,4,5 are members, but only rows 4 and 5 have
    // selected neighbors and positive joins. All six rows are analyzed non-islands.
    expect(Array.from(await session.readUint32(tested))).toEqual(
      candidateLoss ? [0, 0, 0, 0, 0, 0] : [0, 0, 0, 0, 1, 1]
    );
    expect(await session.readScalar(overflow)).toBe(Number(candidateLoss));
    const probabilities = await session.readFloat32(pseudoPValues);
    expect(probabilities).toHaveLength(6);
    for (const row of [0, 1, 2, 3]) expect(probabilities[row]).toBeNaN();
    for (const row of [4, 5]) {
      if (candidateLoss) expect(probabilities[row]).toBeNaN();
      else expect(Number.isFinite(probabilities[row])).toBe(true);
    }
    if (!candidateLoss) expect(await session.readScalar(analysisCount)).toBe(6);
  } finally {
    await session.dispose();
    result?.destroyParameters();
  }
});
