// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-graph.graph-topology',
  element: 'GPUGraphAlgorithmsTopology',
  title: 'Reusable adjacency for the Dutch rail graph',
  audience: 'engineer',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula'],
  dataset: {
    id: 'g-nl-rail-graph',
    licence: 'CC0',
    source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)'
  },
  load: () => import('./graph-topology.run')
});
