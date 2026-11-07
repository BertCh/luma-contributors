// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUConstrainedDelaunayTriangulation starts from an ear-clipped triangulation and flips edges
// toward the Delaunay criterion while keeping the polygon edges. `maximumFlips` caps the work: a
// small cap leaves slivers, a large one converges to well-shaped triangles.

import {createTriangulationRun} from './shared/triangulation';
import {parameters} from './constrained-delaunay-triangulation.parameters';

export default createTriangulationRun({
  parameters,
  constrained: true,
  caption: 'Edge flips turn the ear-clipped slivers into well-shaped triangles.'
});
