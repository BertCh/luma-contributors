// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSpatialAnalysisPackedKernelNodes} from '../geographic-distribution/spatial-analysis-packed-kernel';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUPolygonTriangulation} from '../../analysis-core/polygon-triangulation/gpu-polygon-triangulation';

/** Triangulates one bounded enclosure, including holes, once per encoding. @internal */
export function addBuildingTessellationEnclosure<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    ringOffsets?: GraphDataView<'uint32'>;
    triangleCapacity: number;
  }
): GraphDataView<'float32x2'> {
  const {id, positions, triangleCapacity} = props;
  const featureOffsets = createTransientView(graph, `${id}-feature-offsets`, 'uint32', 2);
  const polygonOffsets = createTransientView(graph, `${id}-polygon-offsets`, 'uint32', 2);
  const ringOffsets =
    props.ringOffsets ?? createTransientView(graph, `${id}-ring-offsets`, 'uint32', 2);
  const indices = createTransientView(graph, `${id}-indices`, 'uint32', 3 * triangleCapacity);
  const valid = createTransientView(graph, `${id}-valid`, 'uint32', 1);
  const data = createTransientView(graph, `${id}-triangles`, 'float32x2', 1 + 3 * triangleCapacity);
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-layout`,
      operation: 'BuildingTessellationEnclosure',
      invocationCount: 1,
      bindings: [
        {name: 'features', view: featureOffsets, type: 'u32', access: 'read_write'},
        {name: 'polygons', view: polygonOffsets, type: 'u32', access: 'read_write'},
        ...(!props.ringOffsets
          ? [
              {
                name: 'rings',
                view: ringOffsets,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      body: `features[featuresOffset]=0u;features[featuresOffset+1u]=1u;
polygons[polygonsOffset]=0u;polygons[polygonsOffset+1u]=${ringOffsets.length - 1}u;
${!props.ringOffsets ? `rings[ringsOffset]=0u;rings[ringsOffset+1u]=${positions.length}u;` : ''}`
    })
  );
  graph.add(
    new GPUPolygonTriangulation({
      id: `${id}-triangulation`,
      positions,
      featureOffsets,
      polygonOffsets,
      ringOffsets,
      output: {indices, valid},
      useZOrderHash: false,
      largePolygonMethod: 'none'
    }).getCommandNodes(graph)
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-pack`,
      operation: 'BuildingTessellationEnclosure',
      invocationCount: triangleCapacity,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'indices', view: indices, type: 'u32', access: 'read'},
        {name: 'validity', view: valid, type: 'u32', access: 'read'},
        {name: 'triangles', view: data, type: 'f32', access: 'read_write'}
      ],
      body: `if(index==0u){triangles[trianglesOffset]=f32(validity[validityOffset]);triangles[trianglesOffset+1u]=0.0;}
for(var vertex=0u;vertex<3u;vertex++){
 let source=indices[indicesOffset+3u*index+vertex];
 let destination=trianglesOffset+2u*(1u+3u*index+vertex);
 triangles[destination]=positions[positionsOffset+2u*source];triangles[destination+1u]=positions[positionsOffset+2u*source+1u];
}`
    })
  );
  return data;
}
