// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {TRANSLATED_DENSITY_FIXTURE as fixture} from './density-translated-fixture';

it('B5 translated fixture distinguishes an incorrect point bin and raster axis/row swaps', () => {
  // Independently model the shader's candidate rectangle, including its one-cell expansion.
  // Corrupting the point's bin to the opposite corner must alter at least one known nonzero cell.
  const visibleWhenMisindexed = fixture.values.map((value, cell) => {
    const column = cell % fixture.width;
    const row = Math.floor(cell / fixture.width);
    const highColumn = Math.min(
      fixture.width - 1,
      Math.floor(column + 0.5 + fixture.bandwidth) + 1
    );
    const highRow = Math.min(fixture.height - 1, Math.floor(row + 0.5 + fixture.bandwidth) + 1);
    return highColumn >= fixture.width - 1 && highRow >= fixture.height - 1 ? value : 0;
  });
  expect(visibleWhenMisindexed).not.toEqual(fixture.values);
  const verticallyReversed = fixture.values.map(
    (_, cell) =>
      fixture.values[
        (fixture.height - 1 - Math.floor(cell / fixture.width)) * fixture.width +
          (cell % fixture.width)
      ]
  );
  expect(verticallyReversed).not.toEqual(fixture.values);
  const axesSwapped = fixture.values.map((_, cell) => {
    const column = cell % fixture.width;
    const row = Math.floor(cell / fixture.width);
    return column < fixture.height ? fixture.values[column * fixture.width + row] : 0;
  });
  expect(axesSwapped).not.toEqual(fixture.values);
  expect(fixture.values).toContain(0);
});
