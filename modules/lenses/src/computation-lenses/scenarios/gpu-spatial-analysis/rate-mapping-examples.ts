// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addExcessRiskRecipe,
  addSpatialRateRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';

/** Unequal populations make pooling visibly different from averaging individual rates. */
export function buildRateMappingScenario(
  element: string,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
): Record<string, S2CaptureSpec> {
  const events = uploads.view('events', new Float32Array([20, 2, 10]), 'float32');
  const populations = uploads.view('populations', new Float32Array([100, 20, 100]), 'float32');
  const output = uploads.view('rates', new Float32Array(3), 'float32');
  if (element === 'addExcessRiskRecipe') {
    addExcessRiskRecipe(graph, {events, populations, output});
  } else {
    addSpatialRateRecipe(graph, {
      events,
      populations,
      output,
      scale: 1000,
      weights: {
        offsets: uploads.view('offsets', new Uint32Array([0, 1, 3, 4]), 'uint32'),
        neighbors: uploads.view('neighbors', new Uint32Array([1, 0, 2, 1]), 'uint32'),
        weights: uploads.view('weights', new Float32Array([1, 1, 1, 1]), 'float32')
      }
    });
  }
  return {
    events: {view: events, role: 'input'},
    populations: {view: populations, role: 'input'},
    rates: {
      view: output,
      role: 'output',
      meaning:
        element === 'addExcessRiskRecipe'
          ? 'Dimensionless pooled excess risk'
          : 'Pooled neighborhood rate per thousand'
    }
  };
}
