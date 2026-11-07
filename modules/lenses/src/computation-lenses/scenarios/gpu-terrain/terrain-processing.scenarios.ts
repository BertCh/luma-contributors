// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

/** Executable, repository-authored bounded examples for terrain processing. */
export const TERRAIN_PROCESSING_SCENARIOS = [
  {
    element: 'GPUTerrainBreach',
    slug: 'terrain-breach',
    title: 'Breach a bounded embankment and inspect cut provenance'
  },
  {
    element: 'GPUTerrainBranchingFlow',
    slug: 'terrain-branching-flow',
    title: 'Conserve injected mass through a branching diamond'
  },
  {
    element: 'GPUTerrainFlowFractions',
    slug: 'terrain-flow-fractions',
    title: 'Materialize MFD receivers and ground distances'
  },
  {
    element: 'GPUTerrainDepressionHierarchy',
    slug: 'terrain-depression-hierarchy',
    title: 'Inspect nested sink ownership and stage storage'
  },
  {
    element: 'GPUTerrainCatchmentSummary',
    slug: 'terrain-catchment-summary',
    title: 'Compare basin hypsometry and stream profiles'
  },
  {
    element: 'GPUTerrainDEMCoregistration',
    slug: 'terrain-dem-coregistration',
    title: 'Remove stable-ground vertical bias and propagate uncertainty'
  },
  {
    element: 'GPUTerrainTiledFlow',
    slug: 'terrain-tiled-flow',
    title: 'Exchange fractional flow across owned tile seams'
  }
] as const;

for (const {element, slug, title} of TERRAIN_PROCESSING_SCENARIOS) {
  const id = `gpu-terrain.${slug}`;
  defineLensScenario({
    id,
    entry: 'gpu-terrain',
    element,
    title,
    audience: 'analyst',
    rungs: ['result', 'graph', 'table', 'kernel'],
    defaultRung: 'result',
    example: {path: `scenarios/gpu-terrain/${slug}.example.ts`},
    dataset: {
      id: 'terrain-processing-synthetic/1',
      licence: 'MIT (repository-authored synthetic inputs)',
      source:
        'Explicit 3 by 3 planar grids with bounded pits, fractional routing and stable-ground masks.'
    },
    load: async () => ({
      run: device =>
        import('./terrain-processing-runner').then(module =>
          module.runTerrainProcessing(device, element, id)
        )
    })
  });
}
