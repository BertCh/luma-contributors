// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getFloat32GridCoordinate,
  GRID_COORDINATE_CASES,
  WIDE_GRID_COORDINATE_CASES
} from './grid-coordinate-bump-fixtures';

it.each([
  ...GRID_COORDINATE_CASES,
  ...WIDE_GRID_COORDINATE_CASES
])('rectangular grid CPU coordinate: $name', ({value, minimum, maximum, size, cell}) => {
  expect(getFloat32GridCoordinate(value, minimum, maximum, size)).toBe(cell);
});
