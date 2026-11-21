// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {snapshotSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-preflight';
import type {GPUSpaceTimeHotSpotsRecipeProps} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {addSpaceTimeHotSpotsRecipe} from '../../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('cell-ID preflight preserves distinct completeness and required-count views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = () => createTransientView(graph, `certificate-${serial++}`, 'uint32', 1);
    const complete = view();
    const requiredCount = view();
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 64
    });
    const props: GPUSpaceTimeHotSpotsRecipeProps = {
      timestamps: createTransientView(graph, 'timestamps', 'uint32x2', 1),
      calendarParameters: createTransientView(graph, 'calendar', 'sint32', 8),
      parameters,
      cells: {
        kind: 'ids',
        cellCount: 1,
        cellIds: view(),
        weights: {
          offsets: createTransientView(graph, 'offsets', 'uint32', 2),
          neighbors: view(),
          weights: createTransientView(graph, 'weights', 'float32', 1),
          status: {complete, requiredCount}
        }
      },
      slices: {step: 'year', interval: 1, origin: {year: 2020}, count: 2},
      settings: {temporalWindow: 1}
    };
    const captured = snapshotSpaceTimeHotSpotsRecipe(props);
    if (captured.cells.kind !== 'ids') throw new Error('expected ID route');
    expect(captured.cells.weights.status?.complete).toBe(complete);
    expect(captured.cells.weights.status?.requiredCount).toBe(requiredCount);
    // The current emerging-hot-spots consumer explicitly refuses positive certificates.
    // Preserving the view through preflight makes that refusal visible instead of dropping it.
    expect(() => addSpaceTimeHotSpotsRecipe(graph, props)).toThrow(
      /complete requires a completeness-aware consumer/
    );
    parameters.destroy();
  } finally {
    device.destroy();
  }
});
