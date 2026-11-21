// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {computeSkaterRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {getGPUSkaterRegionsParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {expect, it} from 'vitest';
import {compactRegionLabels} from '../../../src/gpu-spatial-analysis/spatial-regionalization/compact-region-labels';

it('compacts source-row labels while preserving their join keys and missing rows', () => {
  const input = Uint32Array.of(0, 0, 2, 2, 0xffffffff);
  const result = compactRegionLabels(input);
  expect(Array.from(result.labels)).toEqual([0, 0, 1, 1, 0xffffffff]);
  expect(Array.from(result.sourceLabels)).toEqual([0, 2]);
  expect(result.count).toBe(2);
  expect(Array.from(input)).toEqual([0, 0, 2, 2, 0xffffffff]);
  // A read can have a nonzero byte offset; bytes outside the view are not labels.
  expect(compactRegionLabels(new Uint8Array(input.buffer, 8, 8)).sourceLabels).toEqual(
    Uint32Array.of(2)
  );
  expect(() => compactRegionLabels(new Uint8Array(3))).toThrow(/whole/);
});

it('preserves the SKATER four-row source labels and makes their region count a safe capacity', () => {
  const result = computeSkaterRegionsOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 3, 5, 6),
      neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2),
      weights: new Float32Array(6).fill(1),
      treeEdgeFlags: new Uint32Array(6).fill(1),
      componentLabels: new Uint32Array(4),
      values: Float32Array.of(0, 0, 10, 10),
      parameters: getGPUSkaterRegionsParameterValues({regionCount: 2})
    },
    {maximumRegionCount: 2}
  );
  // Cutting the middle edge uniquely removes all within-region SSD.
  expect(Array.from(result.labels)).toEqual([0, 0, 2, 2]);
  expect(Array.from(result.regionCount)).toEqual([2]);
  expect(Array.from(compactRegionLabels(result.labels).labels)).toEqual([0, 0, 1, 1]);
});
