// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {declareLandscapeDenseLabelsFixture} from './landscape-dense-labels-test-utils';
import {RecipeTestFixture} from '../recipes/recipe-harness';

it.each([
  ['metrics', false],
  ['metrics', true],
  ['reference', false],
  ['reference', true],
  ['boundary', false],
  ['boundary', true]
] as const)('UCT-01 %s reconcile=%s measures dense [0,1,1] as two foreground cells and suppresses unfinished labels', async (route, reconcilePatches) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `dense-${route}`);
  try {
    const results = declareLandscapeDenseLabelsFixture(fixture, route, reconcilePatches);
    fixture.run();
    for (const {result, expected, boundary, reconciliation} of results) {
      const values = await fixture.readFloat32(result, 4);
      expect(values).toHaveLength(4);
      const actual = values[0];
      if (Number.isNaN(expected)) expect(Number.isNaN(actual)).toBe(true);
      else {
        expect(Number.isFinite(actual)).toBe(true);
        expect(actual).toBe(expected);
      }
      if (reconciliation && Number.isNaN(expected)) {
        expect(await fixture.readUint32(reconciliation.patches, 3)).toEqual([
          0xffffffff, 0xffffffff, 0xffffffff
        ]);
        expect(await fixture.readUint32(reconciliation.parents, 1)).toEqual([0xffffffff]);
      }
      if (boundary && Number.isNaN(expected)) {
        expect(await fixture.readUint32(boundary.validity, 1)).toEqual([0]);
        expect(await fixture.readUint32(boundary.count, 1)).toEqual([0]);
        expect(await fixture.readUint32(boundary.candidateOverflow, 1)).toEqual([0]);
        expect(await fixture.readUint32(boundary.overflow, 1)).toEqual([0]);
      }
    }
  } finally {
    fixture.destroy();
  }
});
