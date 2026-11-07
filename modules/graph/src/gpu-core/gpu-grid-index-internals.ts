// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandNode} from './gpu-command-node';
import {type GPUCommandGraph, type GraphDataView} from './gpu-command-graph';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource,
  type GPUBoundedDispatchLayout
} from './gpu-dispatch-utils';
import type {GPUGridIndex, GPUGridIndexBounds} from './gpu-grid-index';
import {GPUScan, getGPUScanCommandNodesWithDispatchLimit} from './gpu-scan';
import {GPUScatter, getGPUScatterCommandNodesWithDispatchLimit} from './gpu-scatter';
import {
  createTransientView,
  createTransientVectorView,
  getViewElementOffset
} from './graph-data-view-utils';
import {getGraphVectorData} from './graph-vector-view-utils';
import {getGraphDataRange, createChunkNode, validateChunkViews} from './gpu-chunk-utils';
import {GPU_GRID_COORDINATE_WGSL, isFiniteFloat32} from './gpu-spatial-utils';

const GRID_INDEX_WORKGROUP_SIZE = 256;

type ChunkNodeProps = Parameters<typeof createChunkNode>[1];

/** Rebuilds counts and global destinations while retaining every source and destination chunk. @internal */
export function getGPUGridIndexCommandNodesWithDispatchLimit<Parameters>(
  index: GPUGridIndex,
  graph: GPUCommandGraph<Parameters>,
  maximum: number
): readonly GPUCommandNode<Parameters>[] {
  validateChunkViews(
    graph,
    [
      index.positions,
      ...(index.sourceIds ? [index.sourceIds] : []),
      ...(index.boundsBuffer ? [index.boundsBuffer] : [])
    ],
    [index.cellOffsets, index.objectIds, index.count, index.overflow]
  );
  const nodes: GPUCommandNode<Parameters>[] = [];
  const offsets = getGraphDataRange(graph, index.cellOffsets, 0, index.cellCount);
  const counts = createTransientVectorView(graph, `${index.id}-counts`, offsets);
  const cursors = createTransientVectorView(graph, `${index.id}-cursors`, offsets);
  const positions = getGraphVectorData(index.positions);
  const ranks = positions.map((chunk, chunkIndex) =>
    createTransientView(graph, `${index.id}-ranks-${chunkIndex}`, 'uint32', chunk.length)
  );
  const generatedIds = index.sourceIds
    ? undefined
    : positions.map((chunk, chunkIndex) =>
        createTransientView(graph, `${index.id}-ids-${chunkIndex}`, 'uint32', chunk.length)
      );
  let positionStart = index.firstSourceIndex;
  for (const [chunkIndex, chunk] of positions.entries()) {
    if (chunk.length) {
      const rank = ranks[chunkIndex];
      const ids = generatedIds?.[chunkIndex];
      const dispatch = getGPUGridIndexDispatchLayout(chunk.length, maximum);
      nodes.push(
        createIndexNode(graph, index, {
          id: `${index.id}-initialize-ranks-${chunkIndex}`,
          outputs: {ranks: rank, ...(ids ? {ids} : {})},
          dispatch,
          invocationCount: chunk.length,
          constants: {
            ELEMENT_COUNT: chunk.length,
            ...(ids && {FIRST_SOURCE_INDEX: positionStart}),
            ...getBoundedDispatchOverrides(dispatch)
          },
          shaderBody: `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource()}
  if (index >= ELEMENT_COUNT) { return; }
  ranks[index] = 0xffffffffu;
  ${ids ? 'ids[index] = FIRST_SOURCE_INDEX + index;' : ''}
}`
        })
      );
    }
    positionStart += chunk.length;
  }
  let cellStart = 0;
  for (const [cellChunkIndex, cellCounts] of counts.data.entries()) {
    const cellCursors = cursors.data[cellChunkIndex];
    const dispatch = getGPUGridIndexDispatchLayout(cellCounts.length, maximum);
    nodes.push(
      createIndexNode(graph, index, {
        id: `${index.id}-initialize-${cellChunkIndex}`,
        outputs: {cellCounts, cellCursors},
        dispatch,
        invocationCount: cellCounts.length,
        constants: {ELEMENT_COUNT: cellCounts.length, ...getBoundedDispatchOverrides(dispatch)},
        shaderBody: `
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource()}
  if (index >= ELEMENT_COUNT) { return; }
  cellCounts[index] = 0u;
  cellCursors[index] = 0u;
}`
      })
    );
    for (const [positionChunkIndex, chunk] of positions.entries()) {
      if (!chunk.length) continue;
      const dispatch = getGPUGridIndexDispatchLayout(chunk.length, maximum);
      nodes.push(
        createIndexNode(graph, index, {
          id: `${index.id}-count-${cellChunkIndex}-${positionChunkIndex}`,
          inputs: {positions: chunk, ...getBoundsInput(index)},
          outputs: {cellCounts},
          dispatch,
          invocationCount: chunk.length,
          atomic: ['cellCounts'],
          ...makePositionPass(index, chunk, dispatch, {
            constants: {CELL_START: cellStart, CELL_LENGTH: cellCounts.length},
            body: `
  if (accepted && cellIndex >= CELL_START && cellIndex - CELL_START < CELL_LENGTH) {
    atomicAdd(&cellCounts[cellIndex - CELL_START], 1u);
  }`
          })
        })
      );
    }
    cellStart += cellCounts.length;
  }
  nodes.push(
    ...getGPUScanCommandNodesWithDispatchLimit(
      new GPUScan({id: `${index.id}-scan`, input: counts, output: offsets}),
      graph,
      maximum
    )
  );
  const lastCounts = counts.data[counts.data.length - 1];
  const lastOffsets = offsets.data[offsets.data.length - 1];
  const terminal = getGraphDataRange(graph, index.cellOffsets, index.cellCount, 1).data[0];
  nodes.push(
    createIndexNode(graph, index, {
      id: `${index.id}-finalize`,
      inputs: {counts: lastCounts, offsets: lastOffsets},
      outputs: {count: index.count, overflow: index.overflow},
      dispatch: {x: 1, y: 1, z: 1},
      invocationCount: 1,
      workgroupSize: 1,
      constants: {
        LAST_COUNT_INDEX: getViewElementOffset(lastCounts) + lastCounts.length - 1,
        LAST_OFFSET_INDEX: getViewElementOffset(lastOffsets) + lastOffsets.length - 1,
        COUNT_OFFSET: getViewElementOffset(index.count),
        OVERFLOW_OFFSET: getViewElementOffset(index.overflow),
        OBJECT_CAPACITY: index.objectIds.length
      },
      shaderBody: `
@compute @workgroup_size(1)
fn main() {
  let total = counts[LAST_COUNT_INDEX] + offsets[LAST_OFFSET_INDEX];
  count[COUNT_OFFSET] = total;
  overflow[OVERFLOW_OFFSET] = select(0u, 1u, total > OBJECT_CAPACITY);
}`
    })
  );
  nodes.push(
    createIndexNode(graph, index, {
      id: `${index.id}-terminal`,
      inputs: {count: index.count},
      outputs: {terminal},
      dispatch: {x: 1, y: 1, z: 1},
      invocationCount: 1,
      workgroupSize: 1,
      constants: {
        TERMINAL_OFFSET: getViewElementOffset(terminal),
        COUNT_OFFSET: getViewElementOffset(index.count)
      },
      shaderBody: `
@compute @workgroup_size(1)
fn main() {
  terminal[TERMINAL_OFFSET] = count[COUNT_OFFSET];
}`
    })
  );
  cellStart = 0;
  for (const [cellChunkIndex, cellOffsets] of offsets.data.entries()) {
    const cellCursors = cursors.data[cellChunkIndex];
    for (const [positionChunkIndex, chunk] of positions.entries()) {
      if (!chunk.length || !index.objectIds.length) continue;
      const dispatch = getGPUGridIndexDispatchLayout(chunk.length, maximum);
      nodes.push(
        createIndexNode(graph, index, {
          id: `${index.id}-rank-${cellChunkIndex}-${positionChunkIndex}`,
          inputs: {positions: chunk, ...getBoundsInput(index), cellOffsets},
          outputs: {cellCursors, ranks: ranks[positionChunkIndex]},
          dispatch,
          invocationCount: chunk.length,
          atomic: ['cellCursors'],
          ...makePositionPass(index, chunk, dispatch, {
            constants: {
              CELL_START: cellStart,
              CELL_LENGTH: cellOffsets.length,
              CELL_OFFSETS_OFFSET: getViewElementOffset(cellOffsets)
            },
            body: `
  if (accepted && cellIndex >= CELL_START && cellIndex - CELL_START < CELL_LENGTH) {
    let cell = cellIndex - CELL_START;
    ranks[index] = cellOffsets[CELL_OFFSETS_OFFSET + cell] + atomicAdd(&cellCursors[cell], 1u);
  }`
          })
        })
      );
    }
    cellStart += cellOffsets.length;
  }
  let sourceStart = 0;
  for (const [chunkIndex, chunk] of positions.entries()) {
    if (chunk.length && index.objectIds.length)
      nodes.push(
        ...getGPUScatterCommandNodesWithDispatchLimit(
          new GPUScatter({
            id: `${index.id}-scatter-${chunkIndex}`,
            source: index.sourceIds
              ? getGraphDataRange(graph, index.sourceIds, sourceStart, chunk.length)
              : generatedIds![chunkIndex],
            indices: ranks[chunkIndex],
            output: index.objectIds
          }),
          graph,
          maximum
        )
      );
    sourceStart += chunk.length;
  }
  if (graph.recording) {
    for (const node of nodes) {
      node.workload = {
        ...node.workload,
        operation: 'GPUGridIndex',
        variant: node.id.slice(index.id.length + 1).replace(/-\d.*$/, ''),
        readByteLength: (node.resources ?? []).reduce(
          (total, resource) =>
            total +
            ('buffer' in resource &&
            (resource.usage === 'storage-read' || resource.usage === 'storage-read-write')
              ? 'length' in resource.buffer
                ? resource.buffer.length * resource.buffer.rowByteLength
                : resource.buffer.byteLength
              : 0),
          0
        ),
        writeByteLength: (node.resources ?? []).reduce(
          (total, resource) =>
            total +
            ('buffer' in resource &&
            (resource.usage === 'storage-write' || resource.usage === 'storage-read-write')
              ? 'length' in resource.buffer
                ? resource.buffer.length * resource.buffer.rowByteLength
                : resource.buffer.byteLength
              : 0),
          0
        )
      };
    }
  }
  return nodes;
}

/**
 * Creates one chunk node whose recording annotation carries its kernel facts. The module id is the
 * node id without the index prefix and chunk numbers, such as `count`.
 */
function createIndexNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  index: GPUGridIndex,
  props: ChunkNodeProps & {invocationCount: number}
): GPUCommandNode<Parameters> {
  return createChunkNode(graph, {
    ...props,
    operation: 'GPUGridIndex',
    moduleId: `GPUGridIndex/${props.id.slice(index.id.length + 1).replace(/-\d.*$/, '')}`,
    workgroupSize: props.workgroupSize ?? GRID_INDEX_WORKGROUP_SIZE
  });
}

/**
 * Shader body and constants of a pass that maps every position to its grid cell. The grid size,
 * bounds, offsets, and the caller's constants are pipeline constants, so one shader serves every
 * grid and input length with the same dimension.
 */
function makePositionPass(
  props: GPUGridIndex,
  positions: GraphDataView,
  dispatch: GPUBoundedDispatchLayout,
  pass: {constants: Record<string, number>; body: string}
): {
  shaderBody: string;
  constants: Record<string, number>;
  constantTypes: Record<string, 'u32' | 'i32' | 'f32'>;
  elementTypes: Record<string, 'u32' | 'i32' | 'f32'>;
} {
  const bounds = getGPUGridIndexBoundsSource(props.bounds, props.boundsBuffer, props.dimension);
  const body = pass.body;
  const width = props.gridSize[0];
  const height = props.gridSize[1];
  const depth = props.dimension === 3 ? props.gridSize[2]! : 1;
  const zDeclarations =
    props.dimension === 3
      ? `let z = positions[POSITIONS_OFFSET + index * 3u + 2u];
    let finiteZ = z == z && abs(z) <= 3.402823466e+38;
    let inZ = z >= ${bounds.minimum[2]} && z <= ${bounds.maximum[2]};`
      : 'let finiteZ = true;\n    let inZ = true;';
  const zCoordinate =
    props.dimension === 3
      ? `let layer = getGridCoordinate(z, ${bounds.minimum[2]}, ${bounds.maximum[2]}, DEPTH);`
      : 'let layer = 0u;';
  const shaderBody = /* wgsl */ `
${GPU_GRID_COORDINATE_WGSL}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${getGPUGridIndexInvocationIndexSource()}
  var accepted = false;
  var cellIndex = 0u;
  if (index < ELEMENT_COUNT) {
    let x = positions[POSITIONS_OFFSET + index * ${props.dimension}u];
    let y = positions[POSITIONS_OFFSET + index * ${props.dimension}u + 1u];
    ${zDeclarations}
    let finiteXY = x == x && y == y && abs(x) <= 3.402823466e+38 && abs(y) <= 3.402823466e+38;
    let inX = x >= ${bounds.minimum[0]} && x <= ${bounds.maximum[0]};
    let inY = y >= ${bounds.minimum[1]} && y <= ${bounds.maximum[1]};
    if (${bounds.validity}finiteXY && finiteZ && inX && inY && inZ) {
      accepted = true;
      let column = getGridCoordinate(x, ${bounds.minimum[0]}, ${bounds.maximum[0]}, WIDTH);
      let row = getGridCoordinate(y, ${bounds.minimum[1]}, ${bounds.maximum[1]}, HEIGHT);
      ${zCoordinate}
      cellIndex = (layer * HEIGHT + row) * WIDTH + column;
    }
  }
  ${body}
}`;
  return {
    shaderBody,
    constants: {
      ELEMENT_COUNT: positions.length,
      POSITIONS_OFFSET: getViewElementOffset(positions),
      WIDTH: width,
      HEIGHT: height,
      DEPTH: depth,
      ...bounds.constants,
      ...pass.constants,
      ...getBoundedDispatchOverrides(dispatch)
    },
    constantTypes: bounds.constantTypes,
    elementTypes: {positions: 'f32', boundsValues: 'f32'}
  };
}

/** Plans bounded 3D dispatches for index kernels. @internal */
export function getGPUGridIndexDispatchLayout(
  count: number,
  maximum: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout('GPUGridIndex', count, GRID_INDEX_WORKGROUP_SIZE, maximum);
}

/** Maps a bounded dispatch to logical rows. @internal */
export function getGPUGridIndexInvocationIndexSource(): string {
  return getBoundedInvocationIndexOverrideSource(GRID_INDEX_WORKGROUP_SIZE).body;
}

/** Returns the optional bounds buffer input for a kernel, bound directly after `positions`. @internal */
function getBoundsInput(index: GPUGridIndex): {boundsValues?: GraphDataView<'float32'>} {
  return index.boundsBuffer ? {boundsValues: index.boundsBuffer} : {};
}

/** Returns whether literal bounds are finite with minima not above maxima. @internal */
export function isOrderedFiniteBounds(bounds: GPUGridIndexBounds): boolean {
  const dimension = bounds.length / 2;
  return (
    bounds.every(isFiniteFloat32) &&
    Array.from({length: dimension}, (_, axis) => axis).every(
      axis => bounds[axis]! <= bounds[axis + dimension]!
    )
  );
}

/**
 * WGSL for an index domain: per-axis minimum and maximum expressions, pipeline constants, and a
 * leading `&&` clause rejecting invalid run-time bounds.
 *
 * Literal bounds become `f32` pipeline constants (`BOUNDS_MINIMUM_0`, `BOUNDS_MAXIMUM_0`, ...), so a
 * shader is reused across bounds. With `boundsBuffer`, values are read through the `boundsValues`
 * storage binding at `BOUNDS_OFFSET`, which the caller must bind as an input and declare as `f32`.
 * Merge `constants` into the pass constants and `constantTypes` into the declared types. @internal
 */
export function getGPUGridIndexBoundsSource(
  bounds: GPUGridIndexBounds,
  boundsBuffer: GraphDataView<'float32'> | undefined,
  dimension: 2 | 3
): {
  minimum: string[];
  maximum: string[];
  validity: string;
  constants: Record<string, number>;
  constantTypes: Record<string, 'u32' | 'i32' | 'f32'>;
} {
  if (!boundsBuffer) {
    const axes = Array.from({length: dimension}, (_, axis) => axis);
    return {
      minimum: axes.map(axis => `BOUNDS_MINIMUM_${axis}`),
      maximum: axes.map(axis => `BOUNDS_MAXIMUM_${axis}`),
      validity: '',
      constants: Object.fromEntries(
        axes.flatMap(axis => [
          [`BOUNDS_MINIMUM_${axis}`, Math.fround(bounds[axis]!)],
          [`BOUNDS_MAXIMUM_${axis}`, Math.fround(bounds[axis + dimension]!)]
        ])
      ),
      constantTypes: Object.fromEntries(
        axes.flatMap(axis => [
          [`BOUNDS_MINIMUM_${axis}`, 'f32'],
          [`BOUNDS_MAXIMUM_${axis}`, 'f32']
        ])
      )
    };
  }
  const read = (component: number) => `boundsValues[BOUNDS_OFFSET + ${component}u]`;
  const minimum = Array.from({length: dimension}, (_, axis) => read(axis));
  const maximum = Array.from({length: dimension}, (_, axis) => read(axis + dimension));
  const validity = minimum.map(
    (value, axis) =>
      `${value} == ${value} && ${maximum[axis]} == ${maximum[axis]} && abs(${value}) <= 3.402823466e+38 && abs(${maximum[axis]}) <= 3.402823466e+38 && ${value} <= ${maximum[axis]}`
  );
  return {
    minimum,
    maximum,
    validity: `${validity.join(' && ')} && `,
    constants: {BOUNDS_OFFSET: getViewElementOffset(boundsBuffer)},
    constantTypes: {}
  };
}
