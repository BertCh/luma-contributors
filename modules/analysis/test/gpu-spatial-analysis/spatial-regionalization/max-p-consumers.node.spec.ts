// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {computeMaxPRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {createMaxPRefinementInput} from './regionalization-operation-test-utils';
import {formatMaxPImprovementRounds} from '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/statistics/county-regionalization-summary';

it('labels two independent row moves in one Max-p round as successful improvement rounds', async () => {
  const input = createMaxPRefinementInput(2);
  const rows = input.values.length;
  const combined = {
    ...input,
    offsets: Uint32Array.from([
      ...input.offsets,
      ...Array.from(input.offsets.slice(1), offset => offset + input.neighbors.length)
    ]),
    neighbors: Uint32Array.from([
      ...input.neighbors,
      ...Array.from(input.neighbors, row => row + rows)
    ]),
    weights: Float32Array.from([...input.weights, ...input.weights]),
    values: Float32Array.from([...input.values, ...input.values]),
    extensiveValues: Float32Array.from([...input.extensiveValues, ...input.extensiveValues])
  };
  const before = computeMaxPRegionsOnCPU(combined, {standardize: false, startCount: 1});
  const result = computeMaxPRegionsOnCPU(combined, {
    standardize: false,
    startCount: 1,
    maximumIterations: 1
  });
  expect(result.labels.filter((label, row) => label !== before.labels[row]).length).toBe(2);
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/statistics/county-regionalization.compute.ts',
    'utf8'
  );
  expect(source.includes('formatMaxPImprovementRounds(status)')).toBe(true);
  expect(formatMaxPImprovementRounds(result.status)).toBe('successful improvement rounds 1');
  for (const path of [
    'examples/deck/spatial-analysis-showcase/src/scenes/statistics/county-regionalization.scene.ts',
    'modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/max-p-regions.lens.ts'
  ])
    expect(readFileSync(path, 'utf8').toLowerCase()).toContain('successful improvement rounds');
});
