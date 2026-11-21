// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {computeRasterRegionGraphOnCPU} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';

const guide = readFileSync(
  new URL('../../../../../docs/api/gpu-raster/remote-sensing.md', import.meta.url),
  'utf8'
);

it('WGS-02 documents upward-rounded float32 bin boundaries and both hand-worked contrasts', () => {
  expect(guide.includes('floor((value - low) / (high - low) * levels)')).toBe(false);
  for (const phrase of [
    'captured float32 endpoints',
    'rounded upward',
    'equality enters the upper bin',
    '[0, 25]',
    '[0, 5]',
    'contrast 1',
    '[10000000.49, 10000002.51]',
    'contrast 4'
  ]) {
    expect(guide.includes(phrase), phrase).toBe(true);
  }
});

it('CPL-05 documents execution provenance, full recipe admission and the pure CPU certificate contract', () => {
  for (const phrase of [
    'getExecutionPlan(device).selectedSearchStructure',
    'estimatedRecipeCommandNodeCount',
    'mandatory input',
    'overlap',
    'padding beyond the grid',
    'Uint32Array.of(0)',
    'provisional.complete',
    'omitted.complete'
  ]) {
    expect(guide.includes(phrase), phrase).toBe(true);
  }
  const options = {
    grid: {
      width: 3,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    },
    regionCount: 3,
    searchStructure: 'dense' as const,
    mergeThreshold: 0,
    maximumIterations: 1
  };
  const labels = Uint32Array.of(0, 1, 2);
  const values = Float32Array.of(0, 0, 0);
  // One supplied zero word certifies loss; omission promises completeness, any nonzero word is true.
  const provisional = computeRasterRegionGraphOnCPU(labels, values, {
    ...options,
    labelsComplete: Uint32Array.of(0)
  });
  const omitted = computeRasterRegionGraphOnCPU(labels, values, options);
  expect(Array.from(provisional.complete)).toEqual([0]);
  expect(Array.from(provisional.iterationLimitReached)).toEqual([1]);
  expect(Array.from(omitted.complete)).toEqual([1]);
  expect(Array.from(omitted.iterationLimitReached)).toEqual([1]);
  // One synchronous dense round reads original labels: 0<-0, 1<-0, 2<-1.
  expect(Array.from(omitted.mapping)).toEqual([0, 0, 1]);
  const certified = computeRasterRegionGraphOnCPU(labels, values, {
    ...options,
    labelsComplete: Uint32Array.of(7)
  });
  expect(Array.from(certified.complete)).toEqual([1]);
  expect(Array.from(certified.iterationLimitReached)).toEqual([1]);
  expect(() =>
    computeRasterRegionGraphOnCPU(labels, values, {...options, labelsComplete: new Uint32Array(2)})
  ).toThrow(/exactly one word/);
});
