// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../kernels/construction-snapshot';
import {createWGSLKernelNode} from '../kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs
} from '../kernels/gpu-contributor-utils';
import {
  validateGPUMultipartGeometryPort,
  type GPULineGeometryPort,
  type GPUMultipartLineGeometryPort,
  type GPUPackedGeometryPort
} from './ports';
import type {GPUPointGeometryPort, GPUMultipartPointGeometryPort} from './ports';

/** @internal Packed line parts with an optional logical-feature level. */
export type GPULineFeatureGeometry = GPUPackedGeometryPort<
  GPULineGeometryPort | GPUMultipartLineGeometryPort
>;

/**
 * CPU consumers accept generated capacity padding after the active part prefix. Terminal-only
 * featureOffsets `[0]` denotes zero features; `[0, 0]` denotes one empty feature.
 * @internal
 */
export function validateCPULineFeatureLayout(
  id: string,
  positions: Float32Array,
  lineOffsets: Uint32Array,
  featureOffsets?: Uint32Array,
  name = 'lines',
  allowNoParts = false
): number {
  const activeParts = featureOffsets?.[featureOffsets.length - 1] ?? lineOffsets.length - 1;
  validateRequirement(
    id,
    positions.length % 2 === 0,
    `${name}.positions must contain float32x2 rows`
  );
  validateRequirement(
    id,
    lineOffsets.length >= (featureOffsets || allowNoParts ? 1 : 2),
    `${name}.lineOffsets requires part offsets and a terminal entry`
  );
  if (featureOffsets) {
    validateRequirement(
      id,
      lineOffsets[0] === 0,
      `${name}.lineOffsets must start at zero with featureOffsets`
    );
    validateRequirement(
      id,
      featureOffsets.length >= 1 && featureOffsets[0] === 0 && activeParts < lineOffsets.length,
      `${name}.featureOffsets must span an active prefix within lineOffsets capacity`
    );
    for (let row = 1; row < featureOffsets.length; row++)
      validateRequirement(
        id,
        featureOffsets[row] >= featureOffsets[row - 1],
        `${name}.featureOffsets must be nondecreasing`
      );
  }
  for (let row = 0; row < lineOffsets.length; row++)
    validateRequirement(
      id,
      lineOffsets[row] <= positions.length / 2 &&
        (row === 0 || lineOffsets[row] >= lineOffsets[row - 1]) &&
        (row <= activeParts || lineOffsets[row] === lineOffsets[activeParts]),
      `${name}.lineOffsets must be nondecreasing within positions and repeat the active terminal offset in capacity padding`
    );
  return activeParts;
}

/** @internal Packed points with optional logical-feature grouping. */
export type GPUPointFeatureGeometry = GPUPackedGeometryPort<
  GPUPointGeometryPort | GPUMultipartPointGeometryPort
>;
/** @internal Capture point grouping before a part-oriented kernel is constructed. */
export function snapshotPointFeatureGeometry(
  id: string,
  geometry: GPUPointFeatureGeometry
): GPUPointFeatureGeometry {
  const snapshot = snapshotConstructionFields(geometry, [
    'kind',
    'positions',
    'featureOffsets',
    'sourceIds'
  ]);
  if (snapshot.featureOffsets) {
    validateGPUMultipartGeometryPort(id, {...snapshot, featureOffsets: snapshot.featureOffsets});
    validateSinglePackedView(snapshot.featureOffsets, ['uint32'], `${id} points.featureOffsets`);
  }
  return snapshot;
}

/** @internal Capture the complete descriptor before a part-oriented delegate is constructed. */
export function snapshotLineFeatureGeometry(
  id: string,
  geometry: GPULineFeatureGeometry
): GPULineFeatureGeometry {
  const snapshot = snapshotConstructionFields(geometry, [
    'kind',
    'positions',
    'lineOffsets',
    'featureOffsets',
    'sourceIds'
  ]);
  if (snapshot.featureOffsets) {
    validateGPUMultipartGeometryPort(id, {...snapshot, featureOffsets: snapshot.featureOffsets});
    validateSinglePackedView(snapshot.featureOffsets, ['uint32'], `${id} lines.featureOffsets`);
  }
  if (snapshot.sourceIds) {
    validateSinglePackedView(snapshot.sourceIds, ['uint32'], `${id} lines.sourceIds`);
    validateRequirement(
      id,
      snapshot.sourceIds.length === (snapshot.featureOffsets ?? snapshot.lineOffsets).length - 1,
      'lines.sourceIds length must equal the logical feature count'
    );
  }
  return snapshot;
}

/** @internal Logical feature row of a line-part row; offset contents follow the port contract. */
export function getLineFeatureLookupWGSL(
  geometry: {featureOffsets?: GraphDataView<'uint32'>},
  name = 'lineFeature',
  binding = 'features'
): string {
  if (!geometry.featureOffsets) return `fn ${name}(part: u32) -> u32 { return part; }`;
  return `fn ${name}(part: u32) -> u32 {
  var low = 0u; var high = ${geometry.featureOffsets.length - 1}u;
  if (high == 0u || part < ${binding}[${binding}Offset] || part >= ${binding}[${binding}Offset + high]) { return 0xffffffffu; }
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (${binding}[${binding}Offset + middle] <= part) { low = middle; } else { high = middle; }
  }
  return low;
}`;
}

/** @internal Restore feature identities after a delegate that intentionally operates on part rows. */
export function createLineFeatureProvenanceNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    geometry: GPULineFeatureGeometry;
    partRows: GraphDataView<'uint32'>;
    featureIds: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
  }
) {
  const {geometry} = props;
  validateGraphOutputsDisjointFromInputs(
    props.id,
    [props.featureIds],
    [props.partRows, props.count, geometry.featureOffsets, geometry.sourceIds]
  );
  return createWGSLKernelNode(graph, {
    id: props.id,
    operation: 'LineFeatureProvenance',
    invocationCount: props.partRows.length,
    bindings: [
      {name: 'parts', view: props.partRows, type: 'u32', access: 'read'},
      {name: 'count', view: props.count, type: 'u32', access: 'read'},
      {name: 'output', view: props.featureIds, type: 'u32', access: 'read_write'},
      ...(geometry.featureOffsets
        ? [{name: 'features', view: geometry.featureOffsets, type: 'u32', access: 'read'} as const]
        : []),
      ...(geometry.sourceIds
        ? [{name: 'identities', view: geometry.sourceIds, type: 'u32', access: 'read'} as const]
        : [])
    ],
    declarations: getLineFeatureLookupWGSL(geometry),
    body: `output[outputOffset + index] = 0xffffffffu;
  if (index >= count[countOffset]) { return; }
  let feature = lineFeature(parts[partsOffset + index]);
  output[outputOffset + index] = ${geometry.sourceIds ? 'identities[identitiesOffset + feature]' : 'feature'};`
  });
}

/** @internal Publish explicit row columns before converting the legacy identity buffer to IDs. */
export function createLinePieceMetadataNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    geometry: GPULineFeatureGeometry;
    sourceIds: GraphDataView<'uint32'>;
    sourcePartRows?: GraphDataView<'uint32'>;
    sourceFeatureRows?: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, geometry, sourceIds, sourcePartRows, sourceFeatureRows, count} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (sourcePartRows) {
    validateGraphOutputsDisjointFromInputs(
      id,
      [sourcePartRows],
      [sourceIds, count, sourceFeatureRows]
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-part-rows`,
        operation: 'LinePieceMetadata',
        invocationCount: sourceIds.length,
        bindings: [
          {name: 'input', view: sourceIds, type: 'u32', access: 'read'},
          {name: 'count', view: count, type: 'u32', access: 'read'},
          {name: 'output', view: sourcePartRows, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset+index]=select(0xffffffffu,input[inputOffset+index],index<count[countOffset]);'
      })
    );
  }
  if (sourceFeatureRows)
    nodes.push(
      createLineFeatureProvenanceNode(graph, {
        id: `${id}-feature-rows`,
        geometry: {...geometry, sourceIds: undefined},
        partRows: sourceIds,
        featureIds: sourceFeatureRows,
        count
      })
    );
  if (geometry.featureOffsets || geometry.sourceIds)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-source-identities`,
        operation: 'LinePieceMetadata',
        invocationCount: sourceIds.length,
        bindings: [
          {name: 'output', view: sourceIds, type: 'u32', access: 'read_write'},
          {name: 'count', view: count, type: 'u32', access: 'read'},
          ...(geometry.featureOffsets
            ? [
                {
                  name: 'features',
                  view: geometry.featureOffsets,
                  type: 'u32',
                  access: 'read'
                } as const
              ]
            : []),
          ...(geometry.sourceIds
            ? [{name: 'identities', view: geometry.sourceIds, type: 'u32', access: 'read'} as const]
            : [])
        ],
        declarations: getLineFeatureLookupWGSL(geometry),
        body: `if(index>=count[countOffset]){output[outputOffset+index]=0xffffffffu;return;}let feature=lineFeature(output[outputOffset+index]);output[outputOffset+index]=${geometry.sourceIds ? 'identities[identitiesOffset+feature]' : 'feature'};`
      })
    );
  return nodes;
}
