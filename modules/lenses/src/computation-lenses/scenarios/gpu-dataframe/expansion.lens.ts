// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenarios} from '../../define-lens-scenarios';

registerLensScenarios(
  {
    entry: 'gpu-dataframe',
    audience: 'analyst',
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
    defaultRung: 'result',
    dataset: {
      id: 'synthetic-four-point-pair-statistics-v1',
      source: 'Deterministic example constructed in expansion.run.ts',
      licence: 'MIT'
    }
  },
  [
    {
      id: 'gpu-dataframe.cross-variogram',
      element: 'GPUCrossVariogram',
      title: 'Synthetic projected observations: cross-variogram',
      load: async () => {
        const module = await import('./expansion.run');
        return {run: device => module.runExpansion(device, 'cross-variogram')};
      }
    },
    {
      id: 'gpu-dataframe.space-time-variogram',
      element: 'GPUSpaceTimeVariogram',
      title: 'Synthetic projected observations: space-time-variogram',
      load: async () => {
        const module = await import('./expansion.run');
        return {run: device => module.runExpansion(device, 'space-time-variogram')};
      }
    },
    {
      id: 'gpu-dataframe.marked-point-statistics',
      element: 'GPUMarkedPointStatistics',
      title: 'Synthetic projected observations: marked-point-statistics',
      load: async () => {
        const module = await import('./expansion.run');
        return {run: device => module.runExpansion(device, 'marked-point-statistics')};
      }
    }
  ]
);
