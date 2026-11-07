// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {registerLensScenarios} from '../../define-lens-scenarios';
import type {LensScenarioDescriptor} from '../../types';
import type {S2CaptureSpec, S2Uploads} from './s2-lens-run';

/** Builds one named element's graph and returns the views to capture. */
export type S2ScenarioBuilder = (
  element: string,
  graph: GPUCommandGraph<undefined>,
  uploads: S2Uploads
) => Promise<Record<string, S2CaptureSpec>> | Record<string, S2CaptureSpec>;

/**
 * Registers scenarios that run through `runS2Lens` with one shared example builder and dataset.
 * The GPU runner and builder modules load lazily, only when a scenario is run.
 */
export function registerS2LensScenarios(
  defaults: {
    dataset: NonNullable<LensScenarioDescriptor['dataset']>;
    loadBuilder: () => Promise<S2ScenarioBuilder>;
  },
  entries: readonly Pick<LensScenarioDescriptor, 'id' | 'element' | 'title'>[]
): void {
  const {dataset, loadBuilder} = defaults;
  registerLensScenarios(
    {entry: 'gpu-spatial-analysis', dataset},
    entries.map(({id, element, title}) => ({
      id,
      element,
      title,
      async load() {
        const [{runS2Lens}, buildScenario] = await Promise.all([
          import('./s2-lens-run'),
          loadBuilder()
        ]);
        return {
          run: device =>
            runS2Lens(device, {
              scenarioId: id,
              element,
              datasets: [dataset.id],
              build: (graph, uploads) => buildScenario(element, graph, uploads)
            })
        };
      }
    }))
  );
}
