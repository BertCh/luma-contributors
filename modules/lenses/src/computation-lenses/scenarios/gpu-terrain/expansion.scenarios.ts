// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-terrain.terrain-flow-distance',
  element: 'GPUTerrainFlowDistance',
  title: 'GPUTerrainFlowDistance',
  rungs: ['result', 'graph', 'table', 'kernel'],
  dataset: {
    id: 'bounded-raster-expansion',
    licence: 'MIT',
    source: 'Synthetic explicit 3 by 3 reference grid'
  },
  load: async () => ({
    run: device =>
      import('../gpu-raster/expansion-runner').then(module =>
        module.runExpansion(
          device,
          'GPUTerrainFlowDistance',
          'gpu-terrain.terrain-flow-distance',
          'gpu-terrain'
        )
      )
  })
});
defineLensScenario({
  id: 'gpu-terrain.terrain-stream-network',
  element: 'GPUTerrainStreamNetwork',
  title: 'GPUTerrainStreamNetwork',
  rungs: ['result', 'graph', 'table', 'kernel'],
  dataset: {
    id: 'bounded-raster-expansion',
    licence: 'MIT',
    source: 'Synthetic explicit 3 by 3 reference grid'
  },
  load: async () => ({
    run: device =>
      import('../gpu-raster/expansion-runner').then(module =>
        module.runExpansion(
          device,
          'GPUTerrainStreamNetwork',
          'gpu-terrain.terrain-stream-network',
          'gpu-terrain'
        )
      )
  })
});
