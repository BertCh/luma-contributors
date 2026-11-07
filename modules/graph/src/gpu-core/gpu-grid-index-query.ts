// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type Binding, Buffer, type Device} from '@luma.gl/core';
import {Kernel} from '../execution/compute/kernel';
import {createGPUComputeCommandNode, type GPUCommandNode} from './gpu-command-node';
import {
  GPUCommandGraph,
  type GraphBufferHandle,
  type GraphDataView,
  type GraphVectorView
} from './gpu-command-graph';
import type {GPUGridIndexBounds, GPUGridIndexSize} from './gpu-grid-index';
import {
  doGraphDataViewsOverlap,
  createTransientView,
  getViewBinding,
  getViewElementOffset,
  validatePackedUint32View,
  validatePackedView
} from './graph-data-view-utils';

import {createChunkNode, getGraphDataRange, validateChunkViews} from './gpu-chunk-utils';
import {getBoundedDispatchOverrides} from './gpu-dispatch-utils';
import {
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBindingList
} from './gpu-kernel-node';
import {alignGraphVectorViews, getGraphVectorData} from './graph-vector-view-utils';
import {
  getGPUGridIndexBoundsSource,
  getGPUGridIndexDispatchLayout,
  getGPUGridIndexInvocationIndexSource,
  isOrderedFiniteBounds
} from './gpu-grid-index-internals';
import {GPUScatter} from './gpu-scatter';

const GRID_QUERY_WORKGROUP_SIZE = 256;
const GRID_QUERY_STATE_LENGTH = 12;

/** Storage and domain contract consumed by {@link GPUGridIndexQuery}. */
export type GPUGridIndexView = {
  gridSize: GPUGridIndexSize;
  /** Literal domain; with `boundsBuffer` it only fixes the dimension. */
  bounds: GPUGridIndexBounds;
  /** Optional run-time domain overriding `bounds`, commonly the index's own `boundsBuffer`. */
  boundsBuffer?: GraphDataView<'float32'>;
  cellOffsets: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  objectIds: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/** Coarse spatial query evaluated against grid cells. */
export type GPUGridIndexQueryKind = 'point' | 'bounds' | 'radius';

/** Properties for one grid-index candidate query. */
export type GPUGridIndexQueryProps = {
  /** Prefix for generated graph node IDs. */
  id?: string;
  /** Grid storage and domain, commonly a `GPUGridIndex` instance. */
  index: GPUGridIndexView;
  /** Cell-selection rule. */
  kind: GPUGridIndexQueryKind;
  /** Packed query scalars: point, minima/maxima, or center/radius. */
  query: GraphDataView<'float32'>;
  /** Caller-owned capacity-bounded candidate IDs. */
  output: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /** Caller-owned row receiving the stored-index candidate count. */
  count: GraphDataView<'uint32'>;
  /** Caller-owned row receiving index or candidate-output overflow. */
  overflow: GraphDataView<'uint32'>;
  /** Optional source-ID-addressed candidate mask, cleared on every encoding. */
  outputMask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

/**
 * Queries a flat grid index for IDs in cells intersecting a point, bounds, or radius.
 *
 * Results are conservative cell candidates. Bounds and radius queries may include objects outside
 * the exact geometry, and point queries return every object in the containing cell. Output order is
 * unspecified because candidates append atomically.
 */
export class GPUGridIndexQuery {
  readonly id: string;
  readonly index: GPUGridIndexView;
  readonly kind: GPUGridIndexQueryKind;
  readonly query: GraphDataView<'float32'>;
  readonly output: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly count: GraphDataView<'uint32'>;
  readonly overflow: GraphDataView<'uint32'>;
  readonly outputMask?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  readonly dimension: 2 | 3;

  constructor(props: GPUGridIndexQueryProps) {
    this.id = props.id ?? 'gpu-grid-index-query';
    this.index = props.index;
    this.kind = props.kind;
    this.query = props.query;
    this.output = props.output;
    this.count = props.count;
    this.overflow = props.overflow;
    this.outputMask = props.outputMask;
    this.dimension = this.index.gridSize.length === 2 ? 2 : 3;

    validateIndexView(this.id, this.index, this.dimension);
    validatePackedView(this.query, ['float32'], `${this.id} query`);
    for (const chunk of getGraphVectorData(this.output))
      validatePackedUint32View(chunk, `${this.id} output`);
    validatePackedUint32View(this.count, `${this.id} count`);
    validatePackedUint32View(this.overflow, `${this.id} overflow`);
    if (this.outputMask)
      for (const chunk of getGraphVectorData(this.outputMask))
        validatePackedUint32View(chunk, `${this.id} outputMask`);
    if (this.count.length < 1 || this.overflow.length < 1) {
      throw new Error(`${this.id} count and overflow must each contain one uint32 row`);
    }
    const expectedQueryLength =
      this.kind === 'point'
        ? this.dimension
        : this.kind === 'bounds'
          ? this.dimension * 2
          : this.dimension + 1;
    if (this.query.length !== expectedQueryLength) {
      throw new Error(`${this.id} ${this.kind} query must contain ${expectedQueryLength} floats`);
    }
    validateDisjointQueryViews(this.id, this.index, this.query, [
      this.output,
      this.count,
      this.overflow,
      ...(this.outputMask ? [this.outputMask] : [])
    ]);
  }

  /** Adds output initialization and candidate collection without submitting or reading back work. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateChunkViews(
      graph,
      [
        this.index.cellOffsets,
        this.index.objectIds,
        this.index.count,
        this.index.overflow,
        this.query,
        ...(this.index.boundsBuffer ? [this.index.boundsBuffer] : [])
      ],
      [this.output, this.count, this.overflow, ...(this.outputMask ? [this.outputMask] : [])]
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const maximum = graph.device.limits.maxComputeWorkgroupsPerDimension;
    nodes.push(
      createChunkNode(graph, {
        id: `${this.id}-initialize`,
        inputs: {indexOverflow: this.index.overflow},
        outputs: {count: this.count, overflow: this.overflow},
        dispatch: {x: 1, y: 1, z: 1},
        workgroupSize: 1,
        constants: {
          COUNT_OFFSET: getViewElementOffset(this.count),
          OVERFLOW_OFFSET: getViewElementOffset(this.overflow),
          INDEX_OVERFLOW_OFFSET: getViewElementOffset(this.index.overflow)
        },
        shaderBody: `
@compute @workgroup_size(1)
fn main() {
  count[COUNT_OFFSET] = 0u;
  overflow[OVERFLOW_OFFSET] = min(indexOverflow[INDEX_OVERFLOW_OFFSET], 1u);
}`
      })
    );
    for (const [chunkIndex, mask] of (this.outputMask
      ? getGraphVectorData(this.outputMask)
      : []
    ).entries()) {
      if (!mask.length) continue;
      const dispatch = getGPUGridIndexDispatchLayout(mask.length, maximum);
      nodes.push(
        createChunkNode(graph, {
          id: `${this.id}-clear-mask-${chunkIndex}`,
          outputs: {mask},
          dispatch,
          constants: {
            MASK_LENGTH: mask.length,
            MASK_OFFSET: getViewElementOffset(mask),
            ...getBoundedDispatchOverrides(dispatch)
          },
          shaderBody: `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource()}
  if (index < MASK_LENGTH) { mask[MASK_OFFSET + index] = 0u; }
}`
        })
      );
    }
    const queryState = createTransientView(
      graph,
      `${this.id}-query-state`,
      'uint32',
      GRID_QUERY_STATE_LENGTH,
      Buffer.STORAGE | Buffer.INDIRECT
    );
    nodes.push(...addQueryPreparationPass(graph, this, queryState, maximum));

    const cellCount = this.index.cellOffsets.length - 1;
    const cells = alignGraphVectorViews(graph, [
      getGraphDataRange(graph, this.index.cellOffsets, 0, cellCount),
      getGraphDataRange(graph, this.index.cellOffsets, 1, cellCount)
    ]);
    let objectStart = 0;
    for (const [objectChunkIndex, objectIds] of getGraphVectorData(
      this.index.objectIds
    ).entries()) {
      if (!objectIds.length) continue;
      const ranks = createTransientView(
        graph,
        `${this.id}-ranks-${objectChunkIndex}`,
        'uint32',
        objectIds.length
      );
      const selectedSlots = createTransientView(
        graph,
        `${this.id}-selected-slots-${objectChunkIndex}`,
        'uint32',
        objectIds.length
      );
      const selectedState = createTransientView(
        graph,
        `${this.id}-selected-state-${objectChunkIndex}`,
        'uint32',
        4,
        Buffer.STORAGE | Buffer.INDIRECT
      );
      const dispatch = getGPUGridIndexDispatchLayout(objectIds.length, maximum);
      nodes.push(
        createChunkNode(graph, {
          id: `${this.id}-clear-ranks-${objectChunkIndex}`,
          outputs: {ranks, selectedState},
          dispatch,
          constants: {OBJECT_COUNT: objectIds.length, ...getBoundedDispatchOverrides(dispatch)},
          shaderBody: `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource()}
  if (index < OBJECT_COUNT) { ranks[index] = 0xffffffffu; }
  if (index < 4u) { selectedState[index] = 0u; }
}`
        })
      );
      let cellStart = 0;
      for (const [cellChunkIndex, [starts, ends]] of cells.entries()) {
        nodes.push(
          createIndirectQueryNode(graph, {
            id: `${this.id}-query-${objectChunkIndex}-${cellChunkIndex}`,
            views: {
              starts,
              ends,
              indexCount: this.index.count,
              queryValues: this.query,
              queryState,
              ...(this.index.boundsBuffer ? {boundsValues: this.index.boundsBuffer} : {}),
              selectedSlots,
              selectedState,
              outputOverflow: this.overflow
            },
            resources: [
              {buffer: starts, usage: 'storage-read'},
              {buffer: ends, usage: 'storage-read'},
              {buffer: this.index.count, usage: 'storage-read'},
              {buffer: this.query, usage: 'storage-read'},
              {buffer: queryState, usage: 'storage-read'},
              ...(this.index.boundsBuffer
                ? ([
                    {buffer: this.index.boundsBuffer, usage: 'storage-read'}
                  ] as QueryPassResource[])
                : []),
              {buffer: selectedSlots, usage: 'storage-write'},
              {buffer: selectedState, usage: 'storage-read-write'},
              {buffer: this.overflow, usage: 'storage-write'}
            ],
            dispatchBuffer: queryState.buffer,
            atomic: ['selectedState', 'outputOverflow'],
            ...makeCellGatherPass(this, {
              starts,
              ends,
              queryState,
              selectedSlots,
              selectedState,
              cellStart,
              objectStart,
              objectCount: objectIds.length
            })
          })
        );
        cellStart += starts.length;
      }
      nodes.push(
        createChunkNode(graph, {
          id: `${this.id}-publish-${objectChunkIndex}`,
          outputs: {selectedState},
          dispatch: {x: 1, y: 1, z: 1},
          workgroupSize: 1,
          constants: {OBJECT_COUNT: objectIds.length},
          shaderBody: `
@compute @workgroup_size(1)
fn main() {
  let count = min(selectedState[0], OBJECT_COUNT);
  selectedState[1] = count / ${GRID_QUERY_WORKGROUP_SIZE}u + select(0u, 1u, count % ${GRID_QUERY_WORKGROUP_SIZE}u != 0u);
  selectedState[2] = 1u;
  selectedState[3] = 1u;
}`
        })
      );
      nodes.push(
        createIndirectQueryNode(graph, {
          id: `${this.id}-rank-${objectChunkIndex}`,
          views: {
            selectedSlots,
            selectedState,
            ranks,
            outputCount: this.count,
            outputOverflow: this.overflow
          },
          resources: [
            {buffer: selectedSlots, usage: 'storage-read'},
            {buffer: selectedState, usage: 'storage-read'},
            {buffer: ranks, usage: 'storage-write'},
            {buffer: this.count, usage: 'storage-read-write'},
            {buffer: this.overflow, usage: 'storage-write'}
          ],
          dispatchBuffer: selectedState.buffer,
          dispatchByteOffset: selectedState.byteOffset + Uint32Array.BYTES_PER_ELEMENT,
          atomic: ['outputCount', 'outputOverflow'],
          constants: {
            OBJECT_COUNT: objectIds.length,
            OUTPUT_CAPACITY: this.output.length,
            COUNT_OFFSET: getViewElementOffset(this.count),
            OVERFLOW_OFFSET: getViewElementOffset(this.overflow)
          },
          shaderBody: `
@compute @workgroup_size(${GRID_QUERY_WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) globalId: vec3u) {
  if (globalId.x >= min(selectedState[0], OBJECT_COUNT)) { return; }
  let localObjectIndex = selectedSlots[globalId.x];
  let destination = atomicAdd(&outputCount[COUNT_OFFSET], 1u);
  ranks[localObjectIndex] = destination;
  if (destination >= OUTPUT_CAPACITY) {
    atomicStore(&outputOverflow[OVERFLOW_OFFSET], 1u);
  }
}`
        })
      );
      nodes.push(
        ...new GPUScatter({
          id: `${this.id}-scatter-${objectChunkIndex}`,
          source: objectIds,
          indices: ranks,
          output: this.output
        }).getCommandNodes(graph)
      );
      let maskStart = 0;
      for (const [maskChunkIndex, mask] of (this.outputMask
        ? getGraphVectorData(this.outputMask)
        : []
      ).entries()) {
        if (mask.length)
          nodes.push(
            createChunkNode(graph, {
              id: `${this.id}-mask-${objectChunkIndex}-${maskChunkIndex}`,
              inputs: {objectIds, ranks},
              outputs: {mask},
              dispatch,
              atomic: ['mask'],
              constants: {
                OBJECT_COUNT: objectIds.length,
                OBJECT_IDS_OFFSET: getViewElementOffset(objectIds),
                MASK_START: maskStart,
                MASK_LENGTH: mask.length,
                MASK_OFFSET: getViewElementOffset(mask),
                ...getBoundedDispatchOverrides(dispatch)
              },
              shaderBody: `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource()}
  if (index >= OBJECT_COUNT || ranks[index] == 0xffffffffu) { return; }
  let objectId = objectIds[OBJECT_IDS_OFFSET + index];
  if (objectId >= MASK_START && objectId - MASK_START < MASK_LENGTH) {
    atomicStore(&mask[MASK_OFFSET + objectId - MASK_START], 1u);
  }
}`
            })
          );
        maskStart += mask.length;
      }
      objectStart += objectIds.length;
    }
    return nodes;
  }
}

type QueryPassResource = {
  buffer: GraphDataView;
  usage: 'storage-read' | 'storage-write' | 'storage-read-write';
};

function addQueryPreparationPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  query: GPUGridIndexQuery,
  queryState: GraphDataView<'uint32'>,
  maximum: number
): readonly GPUCommandNode<Parameters>[] {
  const bounds = getGPUGridIndexBoundsSource(
    query.index.bounds,
    query.index.boundsBuffer,
    query.dimension
  );
  const inputs = {
    queryValues: query.query,
    ...(query.index.boundsBuffer ? {boundsValues: query.index.boundsBuffer} : {})
  };
  const constants: Record<string, number> = {
    QUERY_OFFSET: getViewElementOffset(query.query),
    STATE_OFFSET: getViewElementOffset(queryState),
    WIDTH: query.index.gridSize[0],
    HEIGHT: query.index.gridSize[1],
    DEPTH: query.index.gridSize[2] ?? 1,
    MAXIMUM_DISPATCH: maximum,
    ...bounds.constants
  };
  const shaderBody = /* wgsl */ `
fn finite(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823466e+38;
}

fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (maximum == minimum || value <= minimum) { return 0u; }
  if (value >= maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    let scaledValue = value / scale;
    let scaledMinimum = minimum / scale;
    let scaledMaximum = maximum / scale;
    return min(u32((scaledValue - scaledMinimum) / (scaledMaximum - scaledMinimum) * f32(size)), size - 1u);
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}

fn divideRoundUp(value: u32, divisor: u32) -> u32 {
  return value / divisor + select(0u, 1u, value % divisor != 0u);
}

@compute @workgroup_size(1)
fn main() {
  ${makeQueryCellEnvelope(query, bounds)}
  var minimumCell = vec3u(0u);
  var cellExtent = vec3u(1u);
  var workgroupCount = 0u;
  if (valid) {
    let mappedMinimum = vec3u(
      getCoordinate(envelopeMinimum.x, ${bounds.minimum[0]}, ${bounds.maximum[0]}, WIDTH),
      getCoordinate(envelopeMinimum.y, ${bounds.minimum[1]}, ${bounds.maximum[1]}, HEIGHT),
      ${query.dimension === 3 ? `getCoordinate(envelopeMinimum.z, ${bounds.minimum[2]}, ${bounds.maximum[2]}, DEPTH)` : '0u'}
    );
    let mappedMaximum = vec3u(
      getCoordinate(envelopeMaximum.x, ${bounds.minimum[0]}, ${bounds.maximum[0]}, WIDTH),
      getCoordinate(envelopeMaximum.y, ${bounds.minimum[1]}, ${bounds.maximum[1]}, HEIGHT),
      ${query.dimension === 3 ? `getCoordinate(envelopeMaximum.z, ${bounds.minimum[2]}, ${bounds.maximum[2]}, DEPTH)` : '0u'}
    );
    let gridMaximum = vec3u(WIDTH - 1u, HEIGHT - 1u, DEPTH - 1u);
    minimumCell = mappedMinimum - min(mappedMinimum, vec3u(1u));
    let maximumCell = min(mappedMaximum + vec3u(1u), gridMaximum);
    cellExtent = maximumCell - minimumCell + vec3u(1u);
    workgroupCount = cellExtent.x * cellExtent.y * cellExtent.z;
  }
  var dispatchSize = vec3u(0u, 1u, 1u);
  if (workgroupCount > 0u) {
    dispatchSize.x = min(workgroupCount, MAXIMUM_DISPATCH);
    let remaining = divideRoundUp(workgroupCount, dispatchSize.x);
    dispatchSize.y = min(remaining, MAXIMUM_DISPATCH);
    dispatchSize.z = divideRoundUp(remaining, dispatchSize.y);
  }
  queryState[STATE_OFFSET] = dispatchSize.x;
  queryState[STATE_OFFSET + 1u] = dispatchSize.y;
  queryState[STATE_OFFSET + 2u] = dispatchSize.z;
  queryState[STATE_OFFSET + 3u] = minimumCell.x;
  queryState[STATE_OFFSET + 4u] = minimumCell.y;
  queryState[STATE_OFFSET + 5u] = minimumCell.z;
  queryState[STATE_OFFSET + 6u] = cellExtent.x;
  queryState[STATE_OFFSET + 7u] = cellExtent.y;
  queryState[STATE_OFFSET + 8u] = cellExtent.z;
  queryState[STATE_OFFSET + 9u] = workgroupCount;
  queryState[STATE_OFFSET + 10u] = dispatchSize.x;
  queryState[STATE_OFFSET + 11u] = dispatchSize.y;
}`;
  return [
    createChunkNode(graph, {
      id: `${query.id}-prepare-query-cells`,
      inputs,
      outputs: {queryState},
      dispatch: {x: 1, y: 1, z: 1},
      workgroupSize: 1,
      constants,
      constantTypes: bounds.constantTypes,
      elementTypes: {queryValues: 'f32', boundsValues: 'f32'},
      shaderBody
    })
  ];
}

function makeQueryCellEnvelope(
  query: GPUGridIndexQuery,
  bounds: ReturnType<typeof getGPUGridIndexBoundsSource>
): string {
  const components = query.dimension === 2 ? ['x', 'y'] : ['x', 'y', 'z'];
  const domainOverlap = components
    .map(
      (component, axis) =>
        `envelopeMaximum.${component} >= ${bounds.minimum[axis]} && envelopeMinimum.${component} <= ${bounds.maximum[axis]}`
    )
    .join(' && ');
  if (query.kind === 'point') {
    const reads = components
      .map(
        (component, axis) =>
          `let query${component.toUpperCase()} = queryValues[QUERY_OFFSET + ${axis}u];`
      )
      .join('\n  ');
    const finiteValues = components
      .map(component => `finite(query${component.toUpperCase()})`)
      .join(' && ');
    return `${reads}
  let envelopeMinimum = vec3f(queryX, queryY, ${query.dimension === 3 ? 'queryZ' : '0.0'});
  let envelopeMaximum = envelopeMinimum;
  let valid = ${bounds.validity}${finiteValues} && ${domainOverlap};`;
  }
  if (query.kind === 'radius') {
    const reads = components
      .map(
        (component, axis) =>
          `let query${component.toUpperCase()} = queryValues[QUERY_OFFSET + ${axis}u];`
      )
      .join('\n  ');
    const finiteValues = components
      .map(component => `finite(query${component.toUpperCase()})`)
      .join(' && ');
    return `${reads}
  let radius = queryValues[QUERY_OFFSET + ${query.dimension}u];
  let center = vec3f(queryX, queryY, ${query.dimension === 3 ? 'queryZ' : '0.0'});
  let envelopeMinimum = center - vec3f(radius);
  let envelopeMaximum = center + vec3f(radius);
  let valid = ${bounds.validity}${finiteValues} && finite(radius) && radius >= 0.0 && ${domainOverlap};`;
  }
  const minimumReads = components
    .map(
      (component, axis) =>
        `let minimum${component.toUpperCase()} = queryValues[QUERY_OFFSET + ${axis}u];`
    )
    .join('\n  ');
  const maximumReads = components
    .map(
      (component, axis) =>
        `let maximum${component.toUpperCase()} = queryValues[QUERY_OFFSET + ${axis + query.dimension}u];`
    )
    .join('\n  ');
  const ordered = components
    .map(component => {
      const upper = component.toUpperCase();
      return `finite(minimum${upper}) && finite(maximum${upper}) && minimum${upper} <= maximum${upper}`;
    })
    .join(' && ');
  return `${minimumReads}
  ${maximumReads}
  let envelopeMinimum = vec3f(minimumX, minimumY, ${query.dimension === 3 ? 'minimumZ' : '0.0'});
  let envelopeMaximum = vec3f(maximumX, maximumY, ${query.dimension === 3 ? 'maximumZ' : '0.0'});
  let valid = ${bounds.validity}${ordered} && ${domainOverlap};`;
}

/** Shader body and pipeline constants of one cell-gather pass over a slice of the index cells. */
function makeCellGatherPass(
  query: GPUGridIndexQuery,
  props: {
    starts: GraphDataView<'uint32'>;
    ends: GraphDataView<'uint32'>;
    queryState: GraphDataView<'uint32'>;
    selectedSlots: GraphDataView<'uint32'>;
    selectedState: GraphDataView<'uint32'>;
    cellStart: number;
    objectStart: number;
    objectCount: number;
  }
): {
  shaderBody: string;
  constants: Record<string, number>;
  constantTypes: Record<string, 'u32' | 'i32' | 'f32'>;
  elementTypes: Record<string, 'u32' | 'i32' | 'f32'>;
} {
  const bounds = getGPUGridIndexBoundsSource(
    query.index.bounds,
    query.index.boundsBuffer,
    query.dimension
  );
  const constants: Record<string, number> = {
    WIDTH: query.index.gridSize[0],
    HEIGHT: query.index.gridSize[1],
    DEPTH: query.index.gridSize[2] ?? 1,
    QUERY_OFFSET: getViewElementOffset(query.query),
    STATE_OFFSET: getViewElementOffset(props.queryState),
    OBJECT_START: props.objectStart,
    OBJECT_END: props.objectStart + props.objectCount,
    CELL_START: props.cellStart,
    CELL_LENGTH: props.starts.length,
    OBJECT_COUNT: props.objectCount,
    OBJECT_CAPACITY: query.index.objectIds.length,
    INDEX_COUNT_OFFSET: getViewElementOffset(query.index.count),
    STARTS_OFFSET: getViewElementOffset(props.starts),
    ENDS_OFFSET: getViewElementOffset(props.ends),
    SELECTED_SLOTS_OFFSET: getViewElementOffset(props.selectedSlots),
    SELECTED_STATE_OFFSET: getViewElementOffset(props.selectedState),
    OVERFLOW_OFFSET: getViewElementOffset(query.overflow),
    ...bounds.constants
  };
  const shaderBody = /* wgsl */ `
fn finite(value: f32) -> bool { return value == value && abs(value) <= 3.402823466e+38; }
fn getCoordinate(value: f32, minimum: f32, maximum: f32, size: u32) -> u32 {
  if (!finite(value) || maximum == minimum || value <= minimum) { return 0u; }
  if (value >= maximum) { return size - 1u; }
  if (minimum < 0.0 && maximum > 0.0) {
    let scale = max(abs(minimum), abs(maximum));
    return min(u32(((value / scale) - (minimum / scale)) / ((maximum / scale) - (minimum / scale)) * f32(size)), size - 1u);
  }
  return min(u32((value - minimum) / (maximum - minimum) * f32(size)), size - 1u);
}
fn cellMinimum(coordinate: u32, size: u32, minimum: f32, maximum: f32) -> f32 {
  let ratio = f32(coordinate) / f32(size);
  return minimum * (1.0 - ratio) + maximum * ratio;
}
fn cellMaximum(coordinate: u32, size: u32, minimum: f32, maximum: f32) -> f32 {
  if (coordinate + 1u == size) { return maximum; }
  let ratio = f32(coordinate + 1u) / f32(size);
  return minimum * (1.0 - ratio) + maximum * ratio;
}

@compute @workgroup_size(${GRID_QUERY_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localId: u32) {
  let dispatchWidth = queryState[STATE_OFFSET + 10u];
  let workgroupCount = queryState[STATE_OFFSET + 9u];
  let dispatchRow = workgroupId.z * queryState[STATE_OFFSET + 11u] + workgroupId.y;
  let cellOrdinal = dispatchRow * dispatchWidth + workgroupId.x;
  if (cellOrdinal >= workgroupCount) { return; }
  let extentX = queryState[STATE_OFFSET + 6u];
  let extentY = queryState[STATE_OFFSET + 7u];
  let column = queryState[STATE_OFFSET + 3u] + cellOrdinal % extentX;
  let row = queryState[STATE_OFFSET + 4u] + (cellOrdinal / extentX) % extentY;
  let layer = queryState[STATE_OFFSET + 5u] + cellOrdinal / (extentX * extentY);
  let cellIndex = (layer * HEIGHT + row) * WIDTH + column;
  if (cellIndex < CELL_START || cellIndex - CELL_START >= CELL_LENGTH) { return; }
  ${makeCellSelection(query, bounds)}
  if (!selected) { return; }
  let localCell = cellIndex - CELL_START;
  let storedCount = min(indexCount[INDEX_COUNT_OFFSET], OBJECT_CAPACITY);
  let firstObject = max(min(starts[STARTS_OFFSET + localCell], storedCount), OBJECT_START);
  let endObject = min(min(ends[ENDS_OFFSET + localCell], storedCount), OBJECT_END);
  for (var objectIndex = firstObject + localId; objectIndex < endObject; objectIndex += ${GRID_QUERY_WORKGROUP_SIZE}u) {
    let slot = atomicAdd(&selectedState[SELECTED_STATE_OFFSET], 1u);
    if (slot < OBJECT_COUNT) {
      selectedSlots[SELECTED_SLOTS_OFFSET + slot] = objectIndex - OBJECT_START;
    } else {
      atomicStore(&outputOverflow[OVERFLOW_OFFSET], 1u);
    }
  }
}`;
  return {
    shaderBody,
    constants,
    constantTypes: bounds.constantTypes,
    elementTypes: {queryValues: 'f32', boundsValues: 'f32'}
  };
}

function createIndirectQueryNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    views: Record<string, GraphDataView>;
    resources: QueryPassResource[];
    dispatchBuffer: GraphBufferHandle;
    dispatchByteOffset?: number;
    /** Pipeline override constants referenced by the shader. */
    constants: Readonly<Record<string, number>>;
    /** Types of constants that are not `u32`. */
    constantTypes?: Readonly<Record<string, 'u32' | 'i32' | 'f32'>>;
    /** Views declared as `array<atomic<...>>`. */
    atomic: readonly string[];
    /** WGSL element types by view name; defaults come from the view format. */
    elementTypes?: Readonly<Record<string, 'u32' | 'i32' | 'f32'>>;
    /** WGSL after the generated declarations. */
    shaderBody: string;
  }
): GPUCommandNode<Parameters> {
  // Access follows the declared resource usage, so declarations and graph hazards agree.
  const bindings: GPUKernelBindingList = Object.entries(props.views).map(([name, view]) => {
    const usage = props.resources.find(resource => resource.buffer === view)?.usage;
    return {
      name,
      view,
      access:
        usage === 'storage-read'
          ? ('read' as const)
          : usage === 'storage-read-write'
            ? ('read_write' as const)
            : ('write' as const),
      atomic: props.atomic.includes(name),
      elementType: props.elementTypes?.[name]
    };
  });
  const source = `${getGPUKernelOverrideDeclarations(props.constants, props.constantTypes)}
${getGPUKernelBindingDeclarations(bindings)}
${props.shaderBody}`;
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources: [...props.resources, {buffer: props.dispatchBuffer, usage: 'indirect'}],
    compile: ({device}) => {
      const kernel = makeQueryKernel(device, props.id, source, props.views, props.constants);
      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {};
          for (const [name, view] of Object.entries(props.views)) {
            bindings[name] = getViewBinding(view, getBuffer);
          }
          kernel.dispatchIndirect(computePass, {
            bindings,
            indirectBuffer: getBuffer(props.dispatchBuffer),
            indirectOffset: props.dispatchByteOffset ?? 0
          });
        },
        destroy: () => kernel.destroy()
      };
    }
  });
}

function makeQueryKernel(
  device: Device,
  id: string,
  source: string,
  views: Record<string, GraphDataView>,
  constants: Readonly<Record<string, number>>
): Kernel {
  return new Kernel(device, {
    id,
    source,
    constants: {...constants},
    shaderLayout: {
      bindings: Object.keys(views).map((name, location) => ({
        name,
        type: 'storage' as const,
        group: 0,
        location
      }))
    }
  });
}

function makeCellSelection(
  query: GPUGridIndexQuery,
  bounds: ReturnType<typeof getGPUGridIndexBoundsSource>
): string {
  const dimension = query.dimension;
  const axes = [
    {
      name: 'x',
      coordinate: 'column',
      size: 'WIDTH',
      minimum: bounds.minimum[0]!,
      maximum: bounds.maximum[0]!
    },
    {
      name: 'y',
      coordinate: 'row',
      size: 'HEIGHT',
      minimum: bounds.minimum[1]!,
      maximum: bounds.maximum[1]!
    },
    ...(dimension === 3
      ? [
          {
            name: 'z',
            coordinate: 'layer',
            size: 'DEPTH',
            minimum: bounds.minimum[2]!,
            maximum: bounds.maximum[2]!
          }
        ]
      : [])
  ];
  const cellDeclarations = axes
    .map(
      axis => `let cellMin${axis.name.toUpperCase()} = cellMinimum(${axis.coordinate}, ${axis.size}, ${axis.minimum}, ${axis.maximum});
  let cellMax${axis.name.toUpperCase()} = cellMaximum(${axis.coordinate}, ${axis.size}, ${axis.minimum}, ${axis.maximum});`
    )
    .join('\n  ');

  if (query.kind === 'point') {
    const values = axes
      .map(
        (axis, axisIndex) =>
          `let query${axis.name.toUpperCase()} = queryValues[QUERY_OFFSET + ${axisIndex}u];`
      )
      .join('\n  ');
    const valid = axes
      .map(
        axis =>
          `finite(query${axis.name.toUpperCase()}) && query${axis.name.toUpperCase()} >= ${axis.minimum} && query${axis.name.toUpperCase()} <= ${axis.maximum}`
      )
      .join(' && ');
    const queryCoordinates = axes
      .map(
        axis =>
          `let query${axis.name.toUpperCase()}Coordinate = getCoordinate(query${axis.name.toUpperCase()}, ${axis.minimum}, ${axis.maximum}, ${axis.size});`
      )
      .join('\n  ');
    const queryCell =
      dimension === 2
        ? 'queryYCoordinate * WIDTH + queryXCoordinate'
        : '(queryZCoordinate * HEIGHT + queryYCoordinate) * WIDTH + queryXCoordinate';
    return `${values}
  ${queryCoordinates}
  let selected = ${bounds.validity}${valid} && cellIndex == ${queryCell};`;
  }

  if (query.kind === 'bounds') {
    const values = axes
      .map(
        (axis, axisIndex) =>
          `let queryMin${axis.name.toUpperCase()} = queryValues[QUERY_OFFSET + ${axisIndex}u];
  let queryMax${axis.name.toUpperCase()} = queryValues[QUERY_OFFSET + ${axisIndex + dimension}u];`
      )
      .join('\n  ');
    const valid = axes
      .map(
        axis =>
          `finite(queryMin${axis.name.toUpperCase()}) && finite(queryMax${axis.name.toUpperCase()}) && queryMin${axis.name.toUpperCase()} <= queryMax${axis.name.toUpperCase()}`
      )
      .join(' && ');
    const selected = axes
      .map(
        axis =>
          `cellMax${axis.name.toUpperCase()} >= queryMin${axis.name.toUpperCase()} && cellMin${axis.name.toUpperCase()} <= queryMax${axis.name.toUpperCase()}`
      )
      .join(' && ');
    return `${cellDeclarations}
  ${values}
  let selected = ${bounds.validity}${valid} && ${selected};`;
  }

  const values = axes
    .map(
      (axis, axisIndex) =>
        `let query${axis.name.toUpperCase()} = queryValues[QUERY_OFFSET + ${axisIndex}u];`
    )
    .join('\n  ');
  const validCenter = axes.map(axis => `finite(query${axis.name.toUpperCase()})`).join(' && ');
  const closestPoints = axes
    .map(
      axis =>
        `let closest${axis.name.toUpperCase()} = clamp(query${axis.name.toUpperCase()}, cellMin${axis.name.toUpperCase()}, cellMax${axis.name.toUpperCase()});`
    )
    .join('\n  ');
  const scale = makeNestedMaximum([
    'radius',
    ...axes.flatMap(axis => [
      `abs(query${axis.name.toUpperCase()})`,
      `abs(closest${axis.name.toUpperCase()})`
    ])
  ]);
  const squaredDistance = axes
    .map(
      axis =>
        `(query${axis.name.toUpperCase()} / scale - closest${axis.name.toUpperCase()} / scale) * (query${axis.name.toUpperCase()} / scale - closest${axis.name.toUpperCase()} / scale)`
    )
    .join(' + ');
  return `${cellDeclarations}
  ${values}
  let radius = queryValues[QUERY_OFFSET + ${dimension}u];
  ${closestPoints}
  let scale = ${scale};
  let selected = ${bounds.validity}${validCenter} && finite(radius) && radius >= 0.0 && (scale == 0.0 || ${squaredDistance} <= (radius / scale) * (radius / scale));`;
}

function makeNestedMaximum(values: string[]): string {
  return values.slice(1).reduce((maximum, value) => `max(${maximum}, ${value})`, values[0]);
}

function validateDisjointQueryViews(
  id: string,
  index: GPUGridIndexView,
  query: GraphDataView<'float32'>,
  outputs: (GraphDataView<'uint32'> | GraphVectorView<'uint32'>)[]
): void {
  const inputs: [string, GraphDataView | GraphVectorView][] = [
    ['index cellOffsets', index.cellOffsets],
    ['index objectIds', index.objectIds],
    ['index count', index.count],
    ['index overflow', index.overflow],
    ['query', query],
    ...(index.boundsBuffer
      ? [['index boundsBuffer', index.boundsBuffer] as [string, GraphDataView]]
      : [])
  ];
  const outputNames = ['output', 'count', 'overflow', 'outputMask'];
  for (let outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
    const output = outputs[outputIndex];
    for (const [inputName, input] of inputs) {
      if (
        getGraphVectorData(output).some(write =>
          getGraphVectorData(input).some(read => doGraphDataViewsOverlap(write, read))
        )
      ) {
        throw new Error(`${id} ${outputNames[outputIndex]} and ${inputName} must not overlap`);
      }
    }
    for (let previousIndex = 0; previousIndex < outputIndex; previousIndex++) {
      if (
        getGraphVectorData(output).some(write =>
          getGraphVectorData(outputs[previousIndex]).some(previous =>
            doGraphDataViewsOverlap(write, previous)
          )
        )
      ) {
        throw new Error(
          `${id} ${outputNames[outputIndex]} and ${outputNames[previousIndex]} must not overlap`
        );
      }
    }
  }
}

function validateIndexView(id: string, index: GPUGridIndexView, dimension: 2 | 3): void {
  if (index.boundsBuffer) {
    validatePackedView(index.boundsBuffer, ['float32'], `${id} index boundsBuffer`);
    if (index.boundsBuffer.length !== dimension * 2) {
      throw new Error(`${id} index boundsBuffer must hold ${dimension * 2} float32 values`);
    }
  }
  if (index.bounds.length !== dimension * 2) {
    throw new Error(`${id} index gridSize and bounds must have matching dimensions`);
  }
  if (index.gridSize.some(size => !Number.isSafeInteger(size) || size <= 0)) {
    throw new Error(`${id} index gridSize must contain positive integers`);
  }
  if (!isOrderedFiniteBounds(index.bounds)) {
    throw new Error(`${id} index bounds must contain finite ordered minima and maxima`);
  }
  const cellCount = index.gridSize.reduce((product, size) => product * size, 1);
  if (!Number.isSafeInteger(cellCount) || cellCount > 0xffffffff) {
    throw new Error(`${id} index gridSize product must fit in uint32`);
  }
  for (const [name, view] of [
    ['cellOffsets', index.cellOffsets],
    ['objectIds', index.objectIds],
    ['count', index.count],
    ['overflow', index.overflow]
  ] as const) {
    for (const chunk of getGraphVectorData(view))
      validatePackedUint32View(chunk, `${id} index ${name}`);
  }
  if (index.cellOffsets.length !== cellCount + 1) {
    throw new Error(`${id} index cellOffsets.length must equal cellCount + 1`);
  }
  if (index.count.length < 1 || index.overflow.length < 1) {
    throw new Error(`${id} index count and overflow must each contain one uint32 row`);
  }
}
