// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonTriangulation cuts polygons with holes into triangles (GPU ear clipping). The fill is
// the triangle list the operation wrote, colored by polygon; the blue outline is the input rings.

import {createTriangulationRun} from './shared/triangulation';
import {parameters} from './polygon-triangulation.parameters';

export default createTriangulationRun({
  parameters,
  constrained: false,
  caption: 'Concave polygons (with holes) triangulated on the GPU; fill is the output triangles.'
});
