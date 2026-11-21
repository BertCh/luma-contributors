// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  readPointCloudAttribute,
  writePointCloudAttribute
} from '../../src/pointcloud-analysis/operations/pointcloud-attribute';

// DPC-107: compound row and value expressions keep their meaning. Row `index + 1u` of a
// three-component attribute starts at word `(index + 1u) * 3u`; `index + 1u * 3u` would address
// `index + 3u`. The first component of `value + bias` is `(value + bias).x`, not `value + bias.x`.
it('readPointCloudAttribute and writePointCloudAttribute parenthesise compound arguments', () => {
  expect(readPointCloudAttribute('Values', 'index + 1u', 3)).toBe(
    'vec3f(readValues((index + 1u) * 3u + 0u), readValues((index + 1u) * 3u + 1u), readValues((index + 1u) * 3u + 2u))'
  );
  expect(readPointCloudAttribute('Values', 'index + 1u', 1)).toBe(
    'readValues((index + 1u) * 1u + 0u)'
  );
  expect(writePointCloudAttribute('Out', 'index + 1u', 'value + bias', 2)).toBe(
    'writeOut((index + 1u) * 2u + 0u, (value + bias).x);\nwriteOut((index + 1u) * 2u + 1u, (value + bias).y);'
  );
  expect(writePointCloudAttribute('Out', 'row', 'a * b', 1)).toBe(
    'writeOut(row * 1u + 0u, (a * b));'
  );
});

it('readPointCloudAttribute and writePointCloudAttribute leave plain identifiers unchanged', () => {
  // The text GPUPointCloudVoxelMerge emits today stays byte-identical.
  expect(readPointCloudAttribute('FirstMeans', 'index', 2)).toBe(
    'vec2f(readFirstMeans(index * 2u + 0u), readFirstMeans(index * 2u + 1u))'
  );
  expect(writePointCloudAttribute('NextAttributeMean', 'index', 'mean', 2)).toBe(
    'writeNextAttributeMean(index * 2u + 0u, mean.x);\nwriteNextAttributeMean(index * 2u + 1u, mean.y);'
  );
});
