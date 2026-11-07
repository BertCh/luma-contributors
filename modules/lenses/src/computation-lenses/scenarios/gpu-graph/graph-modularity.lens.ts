// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-graph.graph-modularity',
  element: 'GPUGraphAlgorithmsModularity',
  title: 'Modularity score of a rail-graph partition',
  audience: 'engineer',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula'],
  dataset: {
    id: 'g-nl-rail-graph',
    licence: 'CC0',
    source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)'
  },
  load: () => import('./graph-modularity.run')
});
