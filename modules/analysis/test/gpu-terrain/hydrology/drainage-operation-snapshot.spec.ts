// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUTerrainWatershedsOperation,
  type TerrainWatershedsOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds-operation';
import {createTerrainReviewFixture} from '../terrain-processing/terrain-review-fixture';

it('R10/R11 inherited watershed budget and destinations survive mutation before GPU lowering', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createTerrainReviewFixture(device);
  let budget = 1,
    destination = 'rounds';
  const grid = {
    width: 5,
    height: 1,
    affine: [1, 0, 0, 0, -1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const props: TerrainWatershedsOperationProps = Object.create({
    grid,
    flowDirections: 'directions',
    get maximumIterations() {
      return budget;
    },
    output: Object.create({
      labels: 'labels',
      status: Object.create({
        converged: 'done',
        iterationLimitReached: 'limited',
        get iterationCount() {
          return destination;
        }
      })
    })
  });
  const operation = new GPUTerrainWatershedsOperation(props);
  budget = 1025;
  destination = 'changed';
  grid.width = 1;
  const directions = fixture.input('uint32', [1, 1, 1, 1, 0]);
  const labels = fixture.output('uint32', 5);
  const done = fixture.output('uint32', 1),
    limited = fixture.output('uint32', 1),
    rounds = fixture.output('uint32', 1);
  const views = {
    flowDirections: directions.view,
    labels: labels.view,
    converged: done.view,
    iterationLimitReached: limited.view,
    iterationCount: rounds.view
  };
  fixture.graph.add(
    operation.createContributor(
      port => views[port as keyof typeof views] as never
    ) as GPUCommandNodeProducer
  );
  const compiled = fixture.graph.compile();
  try {
    fixture.submit(compiled);
    // One synchronous round doubles [1,2,3,4,4] to [2,3,4,4,4]; the first two
    // pointers are intermediate, and a changed round cannot certify convergence.
    expect(await labels.read()).toEqual([0xfffffffe, 0xfffffffe, 4, 4, 4]);
    expect(await done.read()).toEqual([0]);
    expect(await limited.read()).toEqual([1]);
    expect(await rounds.read()).toEqual([1]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
