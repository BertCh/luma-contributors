// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMapColoringOnCPU} from '../../../src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation';
import {MAP_COLORING_STATUS_CASES} from './map-coloring-status-oracles-fixture';

it.each(MAP_COLORING_STATUS_CASES)('CPU coloring hand-derived status: $name', scene => {
  const result = computeMapColoringOnCPU(
    Uint32Array.from(scene.offsets),
    Uint32Array.from(scene.neighbors),
    Uint32Array.of(0),
    'largest-degree-first',
    scene.maximumIterations
  );
  expect(Array.from(result.colors)).toEqual(scene.colors);
  for (const port of ['colorCount', 'conflictCount', 'converged', 'iterationCount'] as const)
    expect(Array.from(result[port])).toEqual([scene[port]]);
});
