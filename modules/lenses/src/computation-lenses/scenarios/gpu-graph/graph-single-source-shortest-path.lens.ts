// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineLensScenario} from '../../define-lens-scenarios';

defineLensScenario({
  id: 'gpu-graph.graph-single-source-shortest-path',
  element: 'GPUGraphAlgorithmsSingleSourceShortestPath',
  title: 'Rail travel time from Utrecht Centraal',
  audience: 'engineer',
  rungs: ['result', 'kernel', 'graph', 'datum', 'formula', 'iterations'],
  dataset: {
    id: 'g-nl-rail-graph',
    licence: 'CC0',
    source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)'
  },
  load: () => import('./graph-single-source-shortest-path.run')
});
