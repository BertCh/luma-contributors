// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateCellDispatchBindings} from '../cell-indexing/cell-dispatch-validation';
import {S2_GEOMETRY_WGSL} from '../cell-indexing/gpu-cell-geometry';
import {S2_ANALYSIS_WGSL} from '../cell-topology/s2-analysis-wgsl';
import {
  S2_COVERING_ROUND_SLOT_WORDS,
  createS2CoveringRoundParameters,
  getScanCommandNodeCount,
  validateS2CoveringRoundParameters
} from '../cell-topology/s2-covering-support';
import {POLYGON_WGSL} from './cell-cover-kernels';
import type {GPUCellCoverProps} from './gpu-cell-cover';

/** f32 spherical corner/center bounds retain useful margin through S2 level 15. */
export const GPU_CELL_COVER_S2_MAXIMUM_LEVEL = 15;
/** A split count is four; its whole frontier prefix must fit uint32 without wrapping. */
export const GPU_CELL_COVER_S2_MAXIMUM_FRONTIER = 0x3fffffff;
/** Root frontier enumerates six face cells per feature with uint32 indices. */
export const GPU_CELL_COVER_S2_MAXIMUM_FEATURE_COUNT = Math.floor(0xffffffff / 6);
/** Conservative angular slack for f32 cell-cap bounds; it changes pruning work, never containment. */
const S2_COVER_BOUND_MARGIN_DEGREES = 0.001;
/** Feature offsets, four bound words and a source ID share one storage binding. */
const S2_COVER_FEATURE_WORDS = 6;

export function getS2CellCoverCommandNodeCount(props: GPUCellCoverProps): number {
  return (
    5 +
    Number(Boolean(props.output.status.invalidCount)) +
    Number(Boolean(props.output.status.uncertainCount)) +
    (props.resolution + 1) * (1 + getScanCommandNodeCount(props.candidateCapacity + 1)) +
    props.resolution
  );
}

/** Bound every S2 frontier, metadata plane and runtime slot before declaring graph resources. @internal */
export function validateS2CellCoverBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUCellCoverProps
): void {
  const features = props.featureOffsets.length - 1;
  const frontier = props.candidateCapacity;
  validateS2CoveringRoundParameters(graph, id, props.resolution + 1);
  validateCellDispatchBindings(
    graph,
    id,
    {
      positions: props.positions,
      featureOffsets: props.featureOffsets,
      polygonOffsets: props.polygonOffsets,
      ringOffsets: props.ringOffsets,
      featureIds: props.featureIds,
      outputFeatureIds: props.output.featureIds,
      outputCells: props.output.cells,
      outputCore: props.output.core,
      ...props.output.status
    },
    [
      ['S2 frontier cells', frontier * 8],
      ['S2 frontier feature rows', frontier * 4],
      ['S2 split scan', (frontier + 1) * 4],
      ['S2 feature metadata', (features * S2_COVER_FEATURE_WORDS + 1) * 4],
      ['S2 control', 8],
      ['S2 round parameters', (props.resolution + 1) * S2_COVERING_ROUND_SLOT_WORDS * 4]
    ]
  );
}

/**
 * Parallel center cover of planar lon/lat polygons. A conservative spherical cap bounds every
 * S2 cell; only cells whose cap bounds overlap the feature bounds are split. The target-level
 * center uses the same exact planar ring test as H3 and Quadbin. No spherical polygon semantics
 * are substituted. A truncated intermediate frontier always reports candidateOverflow.
 * @internal
 */
export function getS2CellCoverNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUCellCoverProps
): GPUCommandNode<Parameters>[] {
  const {output, candidateCapacity: capacity, resolution} = props;
  const featureCount = props.featureOffsets.length - 1;
  const roundCount = resolution + 1;
  const parameters = createS2CoveringRoundParameters(graph, id, roundCount);
  const roundSlots = parameters.slots.map(slot =>
    graph.createDataView(slot.buffer, {format: 'uint32', length: 2, byteOffset: slot.byteOffset})
  );
  const wholeParameters = graph.createDataView(parameters.slots[0].buffer, {
    format: 'uint32',
    length: roundCount * S2_COVERING_ROUND_SLOT_WORDS
  });
  const cells = [
    createTransientView(graph, `${id}-s2-cells-a`, 'uint32x2', capacity),
    createTransientView(graph, `${id}-s2-cells-b`, 'uint32x2', capacity)
  ];
  const features = [
    createTransientView(graph, `${id}-s2-features-a`, 'uint32', capacity),
    createTransientView(graph, `${id}-s2-features-b`, 'uint32', capacity)
  ];
  const metadata = createTransientView(
    graph,
    `${id}-s2-feature-metadata`,
    'uint32',
    S2_COVER_FEATURE_WORDS * featureCount + 1
  );
  const counts = createTransientView(graph, `${id}-s2-counts`, 'uint32', capacity + 1);
  const offsets = createTransientView(graph, `${id}-s2-offsets`, 'uint32', capacity + 1);
  const control = createTransientView(graph, `${id}-s2-control`, 'uint32', 2);
  const nodes: GPUCommandNode<Parameters>[] = [...parameters.nodes];
  for (const name of ['invalidCount', 'uncertainCount'] as const) {
    const view = output.status[name];
    if (view)
      nodes.push(
        createFillNode(graph, {
          id: `${id}-s2-clear-${name}`,
          operation: 'GPUCellCover',
          view,
          type: 'u32',
          value: '0u',
          componentCount: 1
        })
      );
  }
  const polygonBindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
    {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
    {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'}
  ];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-s2-features`,
      operation: 'GPUCellCover',
      variant: 's2-features',
      invocationCount: featureCount + 1,
      bindings: [
        ...polygonBindings,
        {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'},
        ...(props.featureIds
          ? [{name: 'sourceIds', view: props.featureIds, type: 'u32', access: 'read'} as const]
          : []),
        ...(output.status.invalidCount
          ? [
              {
                name: 'invalidCount',
                view: output.status.invalidCount,
                type: 'atomic<u32>',
                access: 'read_write'
              } as const
            ]
          : [])
      ],
      declarations: POLYGON_WGSL,
      body: `metadata[metadataOffset + index] = featureOffsets[featureOffsetsOffset + index];
  if (index < ${featureCount}u) {
    let info = coverGetFeatureInfo(index);
    let valid = !info.hasCrossingEdge && !info.hasNonFinite && info.bounds.x <= info.bounds.z && info.bounds.y <= info.bounds.w;
    let bounds = select(vec4f(1.0, 1.0, -1.0, -1.0), info.bounds, valid);
    let base = metadataOffset + ${featureCount + 1}u + 5u * index;
    for (var axis = 0u; axis < 4u; axis++) { metadata[base + axis] = bitcast<u32>(bounds[axis]); }
    metadata[base + 4u] = ${props.featureIds ? 'sourceIds[sourceIdsOffset + index]' : 'index'};
    ${output.status.invalidCount ? 'if (info.hasCrossingEdge) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }' : ''}
  }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-s2-start`,
      operation: 'GPUCellCover',
      variant: 's2-start',
      invocationCount: capacity,
      bindings: [
        {name: 'cells', view: cells[0], type: 'u32', access: 'read_write'},
        {name: 'features', view: features[0], type: 'u32', access: 'read_write'},
        {name: 'parameters', view: wholeParameters, type: 'u32', access: 'read_write'},
        {name: 'control', view: control, type: 'u32', access: 'read_write'}
      ],
      body: `if (index == 0u) { parameters[parametersOffset + 1u] = ${Math.min(featureCount * 6, capacity)}u;
    control[controlOffset] = ${Number(featureCount * 6 > capacity)}u; control[controlOffset + 1u] = 0u; }
  if (index < ${Math.min(featureCount * 6, capacity)}u) {
    cells[cellsOffset + 2u * index] = 0u;
    cells[cellsOffset + 2u * index + 1u] = (2u * (index % 6u) + 1u) << 28u;
    features[featuresOffset + index] = index / 6u;
  }`
    })
  );
  const declarations = `${dggs.source}\n${S2_GEOMETRY_WGSL}\n${S2_ANALYSIS_WGSL}\n${POLYGON_WGSL}
fn s2CoverCellBounds(cell: vec2u) -> vec4f {
  let surface = dggs_s2_ij_to_st(dggs_s2_get_ij(cell), dggs_s2_get_level(cell), vec2f(0.5));
  let center = normalize(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(cell), dggs_s2_st_to_uv(surface)));
  var radius = 0.0;
  for (var vertex = 0u; vertex < 4u; vertex++) { radius = max(radius, s2AnalysisAngle(center, s2AnalysisVertex(cell, vertex))); }
  let angularMargin = ${S2_COVER_BOUND_MARGIN_DEGREES} * 0.017453292519943295;
  radius += angularMargin;
  let longitudeLatitude = geometryS2XyzToLngLat(center);
  let latitudeMargin = radius * 57.29577951308232;
  var lowerLongitude = -180.0; var upperLongitude = 180.0;
  if (abs(longitudeLatitude.y) + latitudeMargin < 90.0) {
    let width = asin(clamp(sin(radius) / cos(radians(longitudeLatitude.y)), 0.0, 1.0)) * 57.29577951308232 + ${S2_COVER_BOUND_MARGIN_DEGREES};
    if (longitudeLatitude.x - width > -180.0 && longitudeLatitude.x + width < 180.0) {
      lowerLongitude = longitudeLatitude.x - width; upperLongitude = longitudeLatitude.x + width;
    }
  }
  return vec4f(lowerLongitude, max(-90.0, longitudeLatitude.y - latitudeMargin), upperLongitude, min(90.0, longitudeLatitude.y + latitudeMargin));
}`;
  for (let round = 0; round < roundCount; round++) {
    const sourceCells = cells[round % 2];
    const sourceFeatures = features[round % 2];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-s2-classify-${round}`,
        operation: 'GPUCellCover',
        variant: 's2-classify',
        invocationCount: capacity + 1,
        bindings: [
          polygonBindings[0],
          {name: 'featureOffsets', view: metadata, type: 'u32', access: 'read'},
          polygonBindings[2],
          polygonBindings[3],
          {name: 'cells', view: sourceCells, type: 'u32', access: 'read'},
          {name: 'features', view: sourceFeatures, type: 'u32', access: 'read_write'},
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
          {name: 'parameters', view: roundSlots[round], type: 'u32', access: 'read'}
        ],
        declarations,
        body: `var count = 0u;
  if (index < min(parameters[parametersOffset + 1u], ${capacity}u)) {
    let feature = features[featuresOffset + index];
    let base = featureOffsetsOffset + ${featureCount + 1}u + 5u * feature;
    let bounds = vec4f(bitcast<f32>(featureOffsets[base]), bitcast<f32>(featureOffsets[base + 1u]), bitcast<f32>(featureOffsets[base + 2u]), bitcast<f32>(featureOffsets[base + 3u]));
    let cell = vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]);
    let cellBounds = s2CoverCellBounds(cell);
    if (bounds.x <= bounds.z && all(cellBounds.xy <= bounds.zw) && all(bounds.xy <= cellBounds.zw)) {
      if (parameters[parametersOffset] < ${resolution}u) { count = 4u; }
      else if (coverContains(feature, geometryGetCenter(cell))) {
        count = 1u;
        ${output.core ? 'if (!coverEdgesHitRect(feature, cellBounds.xy, cellBounds.zw)) { features[featuresOffset + index] = feature | 0x80000000u; }' : ''}
      }
    }
  }
  counts[countsOffset + index] = count;`
      }),
      ...new GPUScan({
        id: `${id}-s2-scan-${round}`,
        input: counts,
        output: offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    if (round < resolution)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-s2-write-${round}`,
          operation: 'GPUCellCover',
          variant: 's2-write',
          invocationCount: capacity,
          bindings: [
            {name: 'cells', view: sourceCells, type: 'u32', access: 'read'},
            {name: 'features', view: sourceFeatures, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'nextCells', view: cells[(round + 1) % 2], type: 'u32', access: 'read_write'},
            {
              name: 'nextFeatures',
              view: features[(round + 1) % 2],
              type: 'u32',
              access: 'read_write'
            },
            {name: 'parameters', view: wholeParameters, type: 'u32', access: 'read_write'},
            {name: 'control', view: control, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations: dggs.source,
          body: `if (index == 0u) {
    let required = offsets[offsetsOffset + ${capacity}u];
    let roundIndex = atomicAdd(&control[controlOffset + 1u], 1u);
    parameters[parametersOffset + (roundIndex + 1u) * ${S2_COVERING_ROUND_SLOT_WORDS}u + 1u] = min(required, ${capacity}u);
    if (required > ${capacity}u) { atomicOr(&control[controlOffset], 1u); }
  }
  if (counts[countsOffset + index] != 0u) {
    let cell = vec2u(cells[cellsOffset + 2u * index + 1u], cells[cellsOffset + 2u * index]);
    let level = dggs_s2_get_level(cell);
    let marker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - level));
    let childMarker = dggs_u64_shift_right(marker, 2u);
    var child = dggs_u64_add(dggs_u64_subtract(cell, marker), childMarker);
    for (var slot = 0u; slot < 4u; slot++) {
      let destination = offsets[offsetsOffset + index] + slot;
      if (destination < ${capacity}u) { nextCells[nextCellsOffset + 2u * destination] = child.y; nextCells[nextCellsOffset + 2u * destination + 1u] = child.x; nextFeatures[nextFeaturesOffset + destination] = features[featuresOffset + index]; }
      child = dggs_u64_add(child, dggs_u64_shift_left(childMarker, 1u));
    }
  }`
        })
      );
  }
  const finalCells = cells[resolution % 2];
  const finalFeatures = features[resolution % 2];
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-s2-emit`,
      operation: 'GPUCellCover',
      variant: 's2-emit',
      invocationCount: capacity,
      bindings: [
        {name: 'cells', view: finalCells, type: 'u32', access: 'read'},
        {name: 'features', view: finalFeatures, type: 'u32', access: 'read'},
        {name: 'counts', view: counts, type: 'u32', access: 'read'},
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'metadata', view: metadata, type: 'u32', access: 'read'},
        {name: 'outputCells', view: output.cells, type: 'u32', access: 'read_write'},
        {name: 'outputIds', view: output.featureIds, type: 'u32', access: 'read_write'},
        ...(output.core
          ? [{name: 'core', view: output.core, type: 'u32', access: 'read_write'} as const]
          : [])
      ],
      body: `let destination = offsets[offsetsOffset + index];
  if (counts[countsOffset + index] != 0u && destination < ${output.cells.length}u) {
    let featureWord = features[featuresOffset + index];
    let feature = featureWord & 0x7fffffffu;
    outputCells[outputCellsOffset + 2u * destination] = cells[cellsOffset + 2u * index];
    outputCells[outputCellsOffset + 2u * destination + 1u] = cells[cellsOffset + 2u * index + 1u];
    outputIds[outputIdsOffset + destination] = metadata[metadataOffset + ${featureCount + 1}u + 5u * feature + 4u];
    ${output.core ? 'core[coreOffset + destination] = featureWord >> 31u;' : ''}
  }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-s2-publish`,
      operation: 'GPUCellCover',
      variant: 's2-publish',
      invocationCount: 1,
      bindings: [
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'control', view: control, type: 'u32', access: 'read'},
        {name: 'parameters', view: roundSlots[resolution], type: 'u32', access: 'read'},
        {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
        {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
        {
          name: 'candidateOverflow',
          view: output.status.candidateOverflow,
          type: 'u32',
          access: 'read_write'
        },
        ...(output.status.requiredCount
          ? [
              {
                name: 'requiredCount',
                view: output.status.requiredCount,
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : []),
        ...(output.status.candidateCount
          ? [
              {
                name: 'candidateCount',
                view: output.status.candidateCount,
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : [])
      ],
      body: `let required = offsets[offsetsOffset + ${capacity}u];
  count[countOffset] = min(required, ${output.cells.length}u); overflow[overflowOffset] = select(0u, 1u, required > ${output.cells.length}u);
  candidateOverflow[candidateOverflowOffset] = control[controlOffset];
  ${output.status.requiredCount ? 'requiredCount[requiredCountOffset] = required;' : ''}
  ${output.status.candidateCount ? 'candidateCount[candidateCountOffset] = parameters[parametersOffset + 1u];' : ''}`
    })
  );
  return nodes;
}
