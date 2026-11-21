// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPULineFeatureGeometry} from '../../analysis-core/contracts/line-feature-provenance';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** @internal Sequence metrics require a caller-defined traversal of disconnected parts. */
export function validateMultipartTrackMeasures(
  id: string,
  grouped: boolean,
  measures: readonly string[]
): void {
  validateRequirement(
    id,
    !grouped ||
      measures.every(measure =>
        ['hausdorff', 'maximumDistance', 'symmetricSegmentPathDistance', 'pairStatus'].includes(
          measure
        )
      ),
    'linesA.featureOffsets / linesB.featureOffsets cannot define an ordered traversal for sequence measures; select hausdorff, maximumDistance or symmetricSegmentPathDistance, or supply explicit single-part tracks for Frechet, DTW, EDR, ERP and LCSS'
  );
}

/** @internal Feature vertex ranges followed by original part offsets, sharing one binding. */
export function createMultipartTrackLayout<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  geometry: GPULineFeatureGeometry
) {
  const features = geometry.featureOffsets;
  const featureCount = (features ?? geometry.lineOffsets).length - 1;
  if (!features)
    return {
      offsets: geometry.lineOffsets,
      featureCount,
      nodes: [],
      segmentHelpers: (side: string) =>
        `fn segmentEnd${side}(row:u32)->vec2f{return load${side}(row+1u);}`
    };
  const offsets = createTransientView(
    graph,
    `${id}-layout`,
    'uint32',
    features.length + geometry.lineOffsets.length
  );
  const node = createWGSLKernelNode(graph, {
    id: `${id}-layout`,
    operation: 'GPUTrackSimilarity',
    invocationCount: offsets.length,
    bindings: [
      {name: 'features', view: features, type: 'u32', access: 'read'},
      {name: 'parts', view: geometry.lineOffsets, type: 'u32', access: 'read'},
      {name: 'output', view: offsets, type: 'u32', access: 'read_write'}
    ],
    body: `if(index<${features.length}u){output[outputOffset+index]=parts[partsOffset+features[featuresOffset+index]];}else{output[outputOffset+index]=parts[partsOffset+index-${features.length}u];}`
  });
  return {
    offsets,
    featureCount,
    nodes: [node],
    segmentHelpers: (side: string) => `fn segmentEnd${side}(row:u32)->vec2f{
 var low=0u;var high=${geometry.lineOffsets.length - 1}u;
 while(high-low>1u){let middle=(low+high)/2u;if(offsets${side}[offsets${side}Offset+${features.length}u+middle]<=row){low=middle;}else{high=middle;}}
 let end=offsets${side}[offsets${side}Offset+${features.length}u+low+1u];
 return load${side}(select(row,row+1u,row+1u<end));
}`
  };
}
