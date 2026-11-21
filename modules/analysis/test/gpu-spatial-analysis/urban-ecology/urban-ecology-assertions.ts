// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import type {RecipeTestFixture} from '../recipes/recipe-harness';

export async function expectUrbanFloat32Output(
  fixture: Pick<RecipeTestFixture, 'readFloat32'>,
  output: {buffer: Buffer; view: GraphDataView},
  expected: number[],
  tolerance = 1e-4
): Promise<void> {
  const scalarCount =
    (output.view.length * output.view.rowByteLength) / Float32Array.BYTES_PER_ELEMENT;
  expect(scalarCount).toBe(expected.length);
  const actual = await fixture.readFloat32(output, scalarCount);
  expect(actual).toHaveLength(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) expect(actual[index]).toBeNaN();
    else {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
        tolerance * (1 + Math.abs(value))
      );
    }
  });
}

export function expectUrbanMissingValues(actual: number[], count: number): void {
  expect(actual).toEqual(new Array<number>(count).fill(NaN));
}
