// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VertexFormat} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';
import {
  GPUPolygonOverlay,
  planGPUPolygonOverlay,
  type GPUPolygonOverlayOutput
} from '../polygon-overlay/index';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateProvidedRecipeViews} from '../landscape-ecology/urban-landscape-recipe-validation';

/**
 * Most buildings with structures: every building adds a layout, a pack, a `GPUPolygonOverlay`
 * dissolve and a measure to the graph, so the node count grows with the building count.
 */
export const BUILDING_STRUCTURES_MAXIMUM_BUILDING_COUNT = 64;

export type GPUBuildingStructureGeometry = {
  /** One feature per building; counterclockwise shells, clockwise holes. */
  geometry: GPUPackedGeometryPort<GPUPolygonGeometryPort>;
  vertexTolerance: number;
  capacity: {intersections: number; nodedSegments: number; vertices: number; rings: number};
};
/**
 * Every structures rejection that depends on the caller, without touching the graph, so
 * `addBuildingAdjacencyRecipe` can run it before its first node. The per-building overlays apply
 * the same vertexTolerance and capacity rules at construction. @internal
 */
export function validateBuildingStructures<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  count: number,
  props: GPUBuildingStructureGeometry,
  perimetersOutput?: GraphDataView<'float32'>
): void {
  validateRequirement(
    id,
    count <= BUILDING_STRUCTURES_MAXIMUM_BUILDING_COUNT,
    `structures support at most ${BUILDING_STRUCTURES_MAXIMUM_BUILDING_COUNT} buildings`
  );
  validateRequirement(
    id,
    props.geometry.featureOffsets.length === count + 1,
    'structures.geometry.featureOffsets must have one row per building plus one'
  );
  validateRequirement(
    id,
    Number.isFinite(props.vertexTolerance) &&
      props.vertexTolerance > 0 &&
      Math.fround(props.vertexTolerance) >= 2 ** -126 &&
      Math.fround(props.vertexTolerance) <= 2 ** 127,
    'structures.vertexTolerance must be a positive finite float32 number'
  );
  validateRequirement(
    id,
    Object.values(props.capacity).every(
      value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffffff
    ),
    'structures.capacity entries (intersections, nodedSegments, vertices, rings) must be integers in [1, 4294967295]'
  );
  validateGraphViewsBelongToGraph(id, graph, [
    props.geometry.positions,
    props.geometry.featureOffsets,
    props.geometry.polygonOffsets,
    props.geometry.ringOffsets
  ]);
  validateProvidedRecipeViews(graph, [
    [`${id}-structure-perimeters`, 'float32', count, perimetersOutput]
  ]);
  for (let building = 0; building < count; building++)
    planGPUPolygonOverlay(graph.device, {
      id: `${id}-structure-${building}-union`,
      operation: 'dissolve',
      fillRule: 'positive',
      left: {
        vertexCount: props.geometry.positions.length,
        ringCount: props.geometry.ringOffsets.length - 1
      },
      capacity: props.capacity,
      vertexTolerance: props.vertexTolerance,
      output: {
        vertexCapacity: props.capacity.vertices,
        ringCapacity: props.capacity.rings,
        boundaryCapacity: props.capacity.nodedSegments
      }
    });
}

/** True component union, then exterior-only perimeter, including courtyard holes and overlaps. @internal */
export function measureBuildingStructures<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  components: GraphDataView<'uint32'>,
  props: GPUBuildingStructureGeometry,
  perimetersOutput?: GraphDataView<'float32'>
) {
  const count = components.length;
  validateBuildingStructures(graph, id, count, props, perimetersOutput);
  const view = <Format extends VertexFormat>(name: string, format: Format, length: number) =>
    getRecipeView(graph, `${id}-${name}`, format, length);
  const perimeters = getRecipeView(
    graph,
    `${id}-structure-perimeters`,
    'float32',
    count,
    perimetersOutput
  );
  const ringCount = props.geometry.ringOffsets.length - 1;
  const owners = view('ring-owners', 'uint32', ringCount);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-ring-owners`,
      operation: id,
      invocationCount: ringCount,
      bindings: [
        {name: 'features', view: props.geometry.featureOffsets, type: 'u32', access: 'read'},
        {name: 'polygons', view: props.geometry.polygonOffsets, type: 'u32', access: 'read'},
        {name: 'owners', view: owners, type: 'u32', access: 'read_write'}
      ],
      body: `var owner=0xffffffffu;for(var feature=0u;feature<${count}u;feature++){let first=features[featuresOffset+feature];let last=features[featuresOffset+feature+1u];if(first<=last&&last<${props.geometry.polygonOffsets.length}u&&index>=polygons[polygonsOffset+first]&&index<polygons[polygonsOffset+last]){owner=feature;}}owners[ownersOffset+index]=owner;`
    })
  );
  const overlays: GPUPolygonOverlayOutput[] = [];
  for (let building = 0; building < count; building++) {
    const prefix = `structure-${building}`;
    const positions = view(
      `${prefix}-input-positions`,
      'float32x2',
      props.geometry.positions.length
    );
    const ringOffsets = view(`${prefix}-input-rings`, 'uint32', ringCount + 1);
    const polygonOffsets = view(`${prefix}-input-polygons`, 'uint32', 2);
    const featureOffsets = view(`${prefix}-input-features`, 'uint32', 2);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-${prefix}-layout`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'polygons', view: polygonOffsets, type: 'u32', access: 'read_write'},
          {name: 'features', view: featureOffsets, type: 'u32', access: 'read_write'}
        ],
        body: `polygons[polygonsOffset]=0u;polygons[polygonsOffset+1u]=${ringCount}u;features[featuresOffset]=0u;features[featuresOffset+1u]=1u;`
      })
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-${prefix}-pack`,
        operation: id,
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          {name: 'input', view: props.geometry.positions, type: 'f32', access: 'read'},
          {name: 'rings', view: props.geometry.ringOffsets, type: 'u32', access: 'read'},
          {name: 'owners', view: owners, type: 'u32', access: 'read'},
          {name: 'components', view: components, type: 'u32', access: 'read'},
          {name: 'output', view: positions, type: 'f32', access: 'read_write'},
          {name: 'offsets', view: ringOffsets, type: 'u32', access: 'read_write'}
        ],
        body: `var written=0u;offsets[offsetsOffset]=0u;for(var ring=0u;ring<${ringCount}u;ring++){let owner=owners[ownersOffset+ring];let first=rings[ringsOffset+ring];let last=rings[ringsOffset+ring+1u];if(owner<${count}u&&components[componentsOffset+${building}u]!=0xffffffffu&&components[componentsOffset+owner]==components[componentsOffset+${building}u]&&first<=last&&last<=${props.geometry.positions.length}u){for(var vertex=first;vertex<last;vertex++){output[outputOffset+2u*written]=input[inputOffset+2u*vertex];output[outputOffset+2u*written+1u]=input[inputOffset+2u*vertex+1u];written++;}}offsets[offsetsOffset+ring+1u]=written;}`
      })
    );
    const status = (name: string) => ({
      count: view(`${prefix}-${name}-count`, 'uint32', 1),
      overflow: view(`${prefix}-${name}-overflow`, 'uint32', 1),
      candidateOverflow: view(`${prefix}-${name}-candidate-overflow`, 'uint32', 1)
    });
    // The overlay geometry has no feature requirement (contract rule 8): its unclamped ring count
    // sizes `capacity.rings`. The boundary requirement is `requiredCount`.
    const geometryStatus = (name: string) => ({
      ...status(name),
      requiredRingCount: view(`${prefix}-${name}-required`, 'uint32', 1)
    });
    const boundaryStatus = (name: string) => ({
      ...status(name),
      requiredCount: view(`${prefix}-${name}-required`, 'uint32', 1)
    });
    const overlay: GPUPolygonOverlayOutput = {
      geometry: {
        kind: 'polygons',
        positions: view(`${prefix}-positions`, 'float32x2', props.capacity.vertices),
        featureOffsets: view(`${prefix}-features`, 'uint32', props.capacity.rings + 1),
        polygonOffsets: view(`${prefix}-polygons`, 'uint32', props.capacity.rings + 1),
        ringOffsets: view(`${prefix}-rings`, 'uint32', props.capacity.rings + 1)
      },
      status: geometryStatus('geometry'),
      boundary: {
        endpoints: view(`${prefix}-boundary`, 'float32x4', props.capacity.nodedSegments),
        operandIds: view(`${prefix}-operands`, 'uint32', props.capacity.nodedSegments),
        sourceFeatureIds: view(
          `${prefix}-boundary-sources`,
          'uint32',
          props.capacity.nodedSegments
        ),
        status: boundaryStatus('boundary')
      }
    };
    const dissolve = new GPUPolygonOverlay({
      id: `${id}-${prefix}-union`,
      left: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},
      operation: 'dissolve',
      fillRule: 'positive',
      capacity: props.capacity,
      vertexTolerance: props.vertexTolerance,
      output: overlay
    });
    dissolve.preflight(graph);
    graph.add(dissolve);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-${prefix}-exterior-measure`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'positions', view: overlay.geometry.positions, type: 'f32', access: 'read'},
          {name: 'features', view: overlay.geometry.featureOffsets, type: 'u32', access: 'read'},
          {name: 'polygons', view: overlay.geometry.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'rings', view: overlay.geometry.ringOffsets, type: 'u32', access: 'read'},
          {name: 'count', view: overlay.status.count, type: 'u32', access: 'read'},
          {name: 'overflow', view: overlay.status.overflow!, type: 'u32', access: 'read'},
          {name: 'candidates', view: overlay.status.candidateOverflow, type: 'u32', access: 'read'},
          {name: 'output', view: perimeters, type: 'f32', access: 'read_write'}
        ],
        body: `var perimeter=0.0;let countValue=count[countOffset];var valid=countValue>0u&&countValue<=${props.capacity.rings}u&&overflow[overflowOffset]==0u&&candidates[candidatesOffset]==0u;if(valid){let polygonCount=features[featuresOffset+countValue];valid=polygonCount<=${props.capacity.rings}u;for(var polygon=0u;polygon<min(polygonCount,${props.capacity.rings}u);polygon++){let ring=polygons[polygonsOffset+polygon];let first=rings[ringsOffset+ring];let last=rings[ringsOffset+ring+1u];valid=valid&&first<last&&last<=${props.capacity.vertices}u;for(var vertex=first;vertex<min(last,${props.capacity.vertices}u);vertex++){let next=select(vertex+1u,first,vertex+1u==last);let left=vec2<f32>(positions[positionsOffset+2u*vertex],positions[positionsOffset+2u*vertex+1u]);let right=vec2<f32>(positions[positionsOffset+2u*next],positions[positionsOffset+2u*next+1u]);perimeter+=distance(left,right);}}}let missing=bitcast<f32>(0x7fc00000u | (index & 0u));output[outputOffset+${building}u]=select(missing,perimeter,valid);`
      })
    );
    overlays.push(overlay);
  }
  return {perimeters, overlays};
}
