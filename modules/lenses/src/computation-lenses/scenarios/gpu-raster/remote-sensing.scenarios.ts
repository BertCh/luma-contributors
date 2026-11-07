// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

export const REMOTE_SENSING_SCENARIOS = [
  {
    element: 'GPURasterConservativeRemapping',
    slug: 'conservative-remapping',
    title: 'Half-cell conservative remapping with visible support'
  },
  {
    element: 'GPURasterTexture',
    slug: 'texture',
    title: 'Checkerboard pair counts and Haralick texture'
  },
  {
    element: 'GPURasterTemporal',
    slug: 'temporal',
    title: 'Quadratic smoothing through an acquisition QA gap'
  },
  {
    element: 'GPURasterRegistration',
    slug: 'registration',
    title: 'Masked linear alignment with accepted overlap'
  },
  {
    element: 'GPURasterRegionGraph',
    slug: 'region-graph',
    title: 'Tiled region assembly and adaptive mean updates'
  },
  {
    element: 'GPURasterSpectralUnmixing',
    slug: 'spectral-unmixing',
    title: 'Supplied endmembers recover simplex abundances'
  },
  {
    element: 'GPURasterSeasonalBreak',
    slug: 'seasonal-break',
    title: 'Dummy-season historical fitting and persistent monitoring alarms'
  }
] as const;

for (const {element, slug, title} of REMOTE_SENSING_SCENARIOS) {
  const id = `gpu-raster.raster-${slug}`;
  defineLensScenario({
    id,
    entry: 'gpu-raster',
    element,
    title,
    audience: 'analyst',
    rungs: ['result', 'graph', 'table', 'kernel'],
    defaultRung: 'result',
    example: {path: `scenarios/gpu-raster/remote-sensing-${slug}.example.ts`},
    dataset: {
      id: 'remote-sensing-synthetic/1',
      licence: 'MIT (repository-authored synthetic inputs)',
      source:
        'Explicit 4 by 4 Cartesian grids, supplied endmembers, regular-time polynomial/harmonic observations and separate QA masks.'
    },
    load: async () => ({
      run: device =>
        import('./remote-sensing-runner').then(module =>
          module.runRemoteSensingLens(device, element, id)
        )
    })
  });
}
