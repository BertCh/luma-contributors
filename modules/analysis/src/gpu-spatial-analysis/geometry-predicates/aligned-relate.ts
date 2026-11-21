// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSpatialRelateWGSL} from '../spatial-join/spatial-relate-wgsl';
import {
  getMinimumRingVertices,
  type SpatialPredicateSide
} from '../spatial-join/spatial-predicate-wgsl';
import {
  getPredicateRelatePatterns,
  getRelateMatchesWGSL,
  getSpatialKindDimension
} from '../spatial-join/spatial-relate-types';
import type {GPUGeometryPredicatesProps} from './gpu-geometry-predicates';

// The join suppresses empty candidates because its broad phase cannot enumerate them.
// Aligned pairs do enumerate them: only the exterior cells can be nonempty. Reuse the
// join's accessors and mod-2 boundary predicate while distinguishing invalid from empty.
function getAlignedFeatureDimensions(side: SpatialPredicateSide): string {
  const prefix = side.prefix;
  return `fn ${prefix}AlignedDimensions(feature: u32) -> vec3u {
    ${side.partCount === undefined ? '' : `let firstPart=${prefix}FeatureOffsets[${prefix}FeatureOffsetsOffset+feature];let endPart=${prefix}FeatureOffsets[${prefix}FeatureOffsetsOffset+feature+1u];if(firstPart>endPart || endPart>${side.partCount}u){return vec3u(0u);}`}
    var nonempty=false;
    var boundary=0u;
    for(var part=${prefix}RingStart(feature);part<${prefix}RingEnd(feature);part++) {
      let start=${prefix}RingVertexStart(part);
      let end=${prefix}RingVertexEnd(part);
      if(start>end || end>${side.vertexCount}u){return vec3u(0u);}
      if(start==end){continue;}
      if(end-start<${getMinimumRingVertices(side.kind)}u){return vec3u(0u);}
      nonempty=true;
      ${side.kind === 'lines' ? `if(${prefix}IsBoundary(${prefix}Vertex(start),feature) || ${prefix}IsBoundary(${prefix}Vertex(end-1u),feature)){boundary=1u;}` : ''}
    }
    if(!nonempty){return vec3u(1u,0u,0u);}
    if(${prefix}First(feature).z==0.0){return vec3u(0u);}
    return vec3u(1u,${getSpatialKindDimension(side.kind) + 1}u,${side.kind === 'polygons' ? '2u' : side.kind === 'points' ? '0u' : 'boundary'});
  }`;
}
/** @internal Aligned pairs use precisely the join's feature-level relate implementation. */
export function createAlignedRelateNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUGeometryPredicatesProps,
  featureCount: number
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const bindings: WGSLKernelBinding[] = [];
  const describe = (
    prefix: 'left' | 'right',
    geometry: GPUGeometryPredicatesProps['geometry']
  ): SpatialPredicateSide => {
    const positions = geometry.positions.length
      ? geometry.positions
      : createTransientView(graph, `${id}-${prefix}-empty-positions`, 'float32x2', 1);
    bindings.push({name: `${prefix}Positions`, view: positions, type: 'f32', access: 'read'});
    if (geometry.kind === 'lines')
      bindings.push({
        name: `${prefix}RingOffsets`,
        view: geometry.lineOffsets,
        type: 'u32',
        access: 'read'
      });
    if (geometry.kind === 'polygons') {
      const featureRings = createTransientView(
        graph,
        `${id}-${prefix}-feature-rings`,
        'uint32',
        featureCount * 2
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${prefix}-feature-rings`,
          operation: 'GPUGeometryPredicates',
          invocationCount: featureCount,
          bindings: [
            {name: 'features', view: geometry.featureOffsets, type: 'u32', access: 'read'},
            {name: 'polygons', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
            {name: 'output', view: featureRings, type: 'u32', access: 'read_write'}
          ],
          body: 'output[outputOffset+index*2u]=polygons[polygonsOffset+features[featuresOffset+index]];output[outputOffset+index*2u+1u]=polygons[polygonsOffset+features[featuresOffset+index+1u]];'
        })
      );
      bindings.push(
        {name: `${prefix}FeatureRings`, view: featureRings, type: 'u32', access: 'read'},
        {name: `${prefix}RingOffsets`, view: geometry.ringOffsets, type: 'u32', access: 'read'}
      );
    } else if (geometry.featureOffsets)
      bindings.push({
        name: `${prefix}FeatureOffsets`,
        view: geometry.featureOffsets,
        type: 'u32',
        access: 'read'
      });
    return {
      prefix,
      kind: geometry.kind,
      vertexCount: geometry.positions.length,
      ...(geometry.kind !== 'polygons' && geometry.featureOffsets
        ? {
            partCount:
              geometry.kind === 'points'
                ? geometry.positions.length
                : geometry.lineOffsets.length - 1
          }
        : {})
    };
  };
  const left = describe('left', props.geometry);
  const right = describe('right', props.other!);
  const matrix =
    props.output.relate ??
    createTransientView(graph, `${id}-aligned-matrix`, 'uint32', featureCount);
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-aligned-relate`,
      operation: 'GPUGeometryPredicates',
      invocationCount: featureCount,
      bindings: [...bindings, {name: 'output', view: matrix, type: 'u32', access: 'read_write'}],
      declarations: `${getSpatialRelateWGSL(left, right)}\n${getAlignedFeatureDimensions(left)}\n${getAlignedFeatureDimensions(right)}`,
      body: `var matrix=pairRelate(index,index);
      if(matrix==0u){
        let leftDimensions=leftAlignedDimensions(index);
        let rightDimensions=rightAlignedDimensions(index);
        if(leftDimensions.x!=0u && rightDimensions.x!=0u && (leftDimensions.y==0u || rightDimensions.y==0u)) {
          // EE has dimension 2. IE/BE and EI/EB inherit the nonempty operand's dimensions.
          matrix=196608u | (leftDimensions.y<<4u) | (leftDimensions.z<<10u) | (rightDimensions.y<<12u) | (rightDimensions.z<<14u);
        }
      }
      output[outputOffset+index]=matrix;`
    })
  );
  for (const [name, output] of Object.entries(props.output.topological ?? {}))
    if (output)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-aligned-${name}`,
          operation: 'GPUGeometryPredicates',
          invocationCount: featureCount,
          bindings: [
            {name: 'matrix', view: matrix, type: 'u32', access: 'read'},
            {name: 'output', view: output, type: 'u32', access: 'read_write'}
          ],
          declarations: getRelateMatchesWGSL(
            getPredicateRelatePatterns(
              name as keyof NonNullable<GPUGeometryPredicatesProps['output']['topological']>,
              getSpatialKindDimension(left.kind),
              getSpatialKindDimension(right.kind)
            )
          ),
          body: `let value=matrix[matrixOffset+index];output[outputOffset+index]=select(0u,1u,value!=0u && (value & 0x80000000u)==0u && (relateMatches(value)${name === 'equals' ? ' || value==196608u' : ''}));`
        })
      );
  return nodes;
}
