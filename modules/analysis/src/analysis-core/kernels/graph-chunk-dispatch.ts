// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  getViewBindingRange,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from './gpu-contributor-utils';
import {createSpatialStatisticsParameterNodes} from './spatial-statistics-parameter-nodes';
import {getCoreDefaultBuffer} from '../network-topology';

/** Row bases, lengths and totals are stored in uint32 shader parameters without wrapping. */
export const GRAPH_CHUNK_MAXIMUM_ROW_COUNT = 0xffffffff;
/** Each slot starts at WebGPU's 256-byte storage binding alignment, regardless of chunk index. */
export const GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT = 64;
/** Only rowBase, length and totalLength are bound; slot padding is never shader-addressable. */
export const GRAPH_CHUNK_PARAMETER_WORD_COUNT = 3;
/** WebGPU requires every storage binding size to be positive and a multiple of four bytes. */
const GRAPH_CHUNK_STORAGE_BINDING_BYTE_ALIGNMENT = 4;

/** Named fields, including optional inputs or outputs, to validate together. */
export type GraphChunkDispatchViews = Readonly<
  Record<string, GraphDataView | GraphVectorView | undefined>
>;

/** One source-ordered dispatch, including empty chunks (which callers skip). */
export type GraphChunkDispatch = {
  readonly chunkIndex: number;
  readonly rowBase: number;
  readonly length: number;
  /** Same field names as the plan inputs; absent optional fields are omitted. */
  readonly views: Readonly<Record<string, GraphDataView>>;
};

/** Construction-time metadata only: no resources, repacking, nodes or GPU allocation. */
export type GraphChunkDispatchPlan = {
  readonly chunks: readonly GraphChunkDispatch[];
  readonly totalLength: number;
  readonly nonemptyChunkCount: number;
};

/**
 * Plans row-aligned packed analysis fields at contributor construction. The first defined field
 * determines the topology; every other field must have identical ordered chunk lengths. Packed
 * and singleton-vector fields may be mixed. Formats may differ; contributors still validate their
 * accepted formats. Empty chunks retain their indices and row bases. The total is bounded before
 * conversion to uint32, so no prefix can wrap. At least one field must be defined.
 *
 * This aligns logical rows, not byte offsets: uint32-aligned packed subviews are accepted. When
 * using `createWGSLKernelNode`, keep all binding element offsets identical between dispatches
 * (normally use 256-byte-aligned views). Otherwise pass offsets through runtime parameters too.
 * Never interpolate chunk indices, lengths or row bases into WGSL or pipeline overrides.
 */
export function planGraphChunkDispatch(
  id: string,
  views: GraphChunkDispatchViews
): GraphChunkDispatchPlan {
  const fields = Object.entries(views).filter(
    (entry): entry is [string, GraphDataView | GraphVectorView] => entry[1] !== undefined
  );
  validateRequirement(id, fields.length > 0, 'views must contain at least one defined field');
  const [templateName, template] = fields[0];
  const templateChunks = getGraphViewChunks(template);
  validateCapacity(
    id,
    `${templateName} total rows`,
    template.length,
    0,
    GRAPH_CHUNK_MAXIMUM_ROW_COUNT
  );
  const fieldsWithChunks = fields.map(([name, view]) => {
    const fieldChunks = getGraphViewChunks(view);
    validateRequirement(
      id,
      view.length === template.length && fieldChunks.length === templateChunks.length,
      `${name} must match ${templateName} row count and chunk topology`
    );
    for (const [chunkIndex, chunk] of fieldChunks.entries()) {
      validateCapacity(
        id,
        `${name} chunk ${chunkIndex} rows`,
        chunk.length,
        0,
        GRAPH_CHUNK_MAXIMUM_ROW_COUNT
      );
      validatePackedView(chunk, [view.format], `${id} ${name} chunk ${chunkIndex}`);
      validateRequirement(
        id,
        chunk.length === templateChunks[chunkIndex].length,
        `${name} chunk ${chunkIndex} length must match ${templateName} chunk topology`
      );
    }
    return {name, chunks: fieldChunks};
  });
  let totalLength = 0;
  const plannedChunks = templateChunks.map((chunk, chunkIndex) => {
    const rowBase = totalLength;
    totalLength += chunk.length;
    validateCapacity(
      id,
      `${templateName} total rows`,
      totalLength,
      0,
      GRAPH_CHUNK_MAXIMUM_ROW_COUNT
    );
    return {
      chunkIndex,
      rowBase,
      length: chunk.length,
      views: Object.fromEntries(
        fieldsWithChunks.map(field => [field.name, field.chunks[chunkIndex]])
      )
    };
  });
  validateRequirement(
    id,
    totalLength === template.length,
    `${templateName} length must equal its chunk total`
  );
  return {
    chunks: plannedChunks,
    totalLength,
    nonemptyChunkCount: plannedChunks.filter(chunk => chunk.length > 0).length
  };
}

type WritableByteRange = {name: string; offset: number; end: number};
type WritableChunkRanges = {
  /** Packed fields are bound in every chunk dispatch. */
  chunkIndex: number | null;
  aligned: WritableByteRange;
  logical?: WritableByteRange;
};

/**
 * Validates writable separation at contributor construction for one dispatch per chunk index.
 * Different output fields at the same vector chunk index must have disjoint aligned binding
 * ranges, including prefixes. Empty vector chunks are excluded: their dispatches are skipped,
 * so they may reuse backing views within and across fields. A packed GraphDataView is treated as bound
 * in every dispatch, so its aligned binding must be disjoint from every other output binding.
 * A singleton GraphVectorView participates only at chunk index zero.
 *
 * Different vector chunk indices are written by separate dispatches: only their logical byte
 * ranges must be disjoint, and empty chunks have no logical writes. Consecutive chunks within
 * one alignment window are therefore allowed. Pass `{strictAlignedBindings: true}` when several
 * chunks are bound together in one dispatch; that checks every aligned binding against every
 * other binding, including empty backing rows, regardless of chunk index. Existing two-argument
 * calls retain the chunk default. Strict mode is for views that are actually bound together;
 * omit unused empty views from that call if the dispatch does not bind them.
 * With `separateDispatches`, only logical ranges are compared globally. Call again for each
 * co-bound subset without that option to validate its aligned bindings as well.
 * `strictAlignedBindings` takes precedence if both options are supplied.
 *
 * Imported handles are grouped by their current underlying core buffer, unwrapping DynamicBuffer
 * defaults with the shared analysis helper. Transient and unbound handles retain graph identity;
 * allocations or rebindings chosen later must preserve separation. Sorting per buffer costs
 * O(C log C), without an all-pairs scan or repeating packed fields once per chunk. No resources
 * or nodes are declared.
 */
export function validateDisjointWritableViews(
  id: string,
  outputs: GraphChunkDispatchViews,
  options: {strictAlignedBindings?: boolean; separateDispatches?: boolean} = {}
): void {
  const rangesByBuffer = new Map<Buffer | GraphDataView['buffer'], WritableChunkRanges[]>();
  for (const [name, output] of Object.entries(outputs)) {
    for (const [chunkIndex, view] of output ? getGraphViewChunks(output).entries() : []) {
      if (
        output instanceof GraphVectorView &&
        view.length === 0 &&
        !options.strictAlignedBindings
      ) {
        continue;
      }
      const buffer = view.buffer.defaultBuffer
        ? getCoreDefaultBuffer(view.buffer.defaultBuffer)
        : view.buffer;
      const range = getViewBindingRange(view);
      const rangeName = `${name} chunk ${chunkIndex}`;
      const ranges = rangesByBuffer.get(buffer) ?? [];
      ranges.push({
        chunkIndex: output instanceof GraphVectorView ? chunkIndex : null,
        aligned: {name: rangeName, offset: range.offset, end: range.offset + range.size},
        logical:
          view.length > 0
            ? {
                name: rangeName,
                offset: view.byteOffset,
                end: view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength
              }
            : undefined
      });
      rangesByBuffer.set(buffer, ranges);
    }
  }
  for (const ranges of rangesByBuffer.values()) {
    if (options.separateDispatches && !options.strictAlignedBindings) {
      validateDisjointByteRanges(
        id,
        ranges.flatMap(range => (range.logical ? [range.logical] : [])),
        'logical'
      );
      continue;
    }
    if (options.strictAlignedBindings) {
      validateDisjointByteRanges(
        id,
        ranges.map(range => range.aligned),
        'aligned binding'
      );
      continue;
    }
    const packedRanges: WritableByteRange[] = [];
    const chunkRanges: WritableByteRange[] = [];
    const rangesByChunkIndex = new Map<number, WritableByteRange[]>();
    for (const range of ranges) {
      if (range.chunkIndex === null) {
        packedRanges.push(range.aligned);
      } else {
        const sameIndexRanges = rangesByChunkIndex.get(range.chunkIndex) ?? [];
        sameIndexRanges.push(range.aligned);
        rangesByChunkIndex.set(range.chunkIndex, sameIndexRanges);
        chunkRanges.push(range.aligned);
      }
    }
    const orderedPackedRanges = validateDisjointByteRanges(id, packedRanges, 'aligned binding');
    for (const sameIndexRanges of rangesByChunkIndex.values()) {
      validateDisjointByteRanges(id, sameIndexRanges, 'aligned binding');
    }
    // Packed bindings coexist with every chunk, but chunks at different indices may share prefixes.
    // A two-pointer sweep checks those cross-group conflicts without copying packed fields C times.
    if (orderedPackedRanges.length > 0) {
      const orderedChunkRanges = chunkRanges.sort((first, second) => first.offset - second.offset);
      let packedIndex = 0;
      let chunkIndex = 0;
      while (packedIndex < orderedPackedRanges.length && chunkIndex < orderedChunkRanges.length) {
        const packed = orderedPackedRanges[packedIndex];
        const chunk = orderedChunkRanges[chunkIndex];
        if (packed.end <= chunk.offset) {
          packedIndex++;
        } else if (chunk.end <= packed.offset) {
          chunkIndex++;
        } else {
          validateRequirement(
            id,
            false,
            `${packed.name} and ${chunk.name} aligned binding byte ranges must not overlap`
          );
        }
      }
    }
    validateDisjointByteRanges(
      id,
      ranges.flatMap(range => (range.logical ? [range.logical] : [])),
      'logical'
    );
  }
}

/** Returns sorted, separated intervals for one physical buffer or co-bound subset. */
function validateDisjointByteRanges(
  id: string,
  ranges: readonly WritableByteRange[],
  kind: 'aligned binding' | 'logical'
): WritableByteRange[] {
  const orderedRanges = [...ranges].sort((first, second) => first.offset - second.offset);
  for (let rangeIndex = 1; rangeIndex < orderedRanges.length; rangeIndex++) {
    const previous = orderedRanges[rangeIndex - 1];
    const current = orderedRanges[rangeIndex];
    validateRequirement(
      id,
      previous.end <= current.offset,
      `${previous.name} and ${current.name} ${kind} byte ranges must not overlap`
    );
  }
  return orderedRanges;
}

/**
 * Pre-checks graph ownership and the actual aligned storage binding of every named view/chunk,
 * including empty views' backing row and prefixes of non-256-byte-aligned subviews. Each buffer
 * must include Buffer.STORAGE, and each binding size must be a positive multiple of four bytes.
 * A subword format is accepted only when its complete aligned binding satisfies that requirement.
 * Call at the start of `getCommandNodes`, before declaring ANY resources or nodes. Include packed
 * side inputs,
 * status fields and already declared scratch too. Validate derived scratch byte lengths separately
 * with `validateGPUStorageBindingByteLength` before allocating it. This helper never mutates graph.
 */
export function validateGraphChunkBindingRanges<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: GraphChunkDispatchViews
): void {
  validateNamedChunkBindingRanges(
    graph,
    id,
    Object.entries(views).flatMap(([name, view]) =>
      view
        ? getGraphViewChunks(view).map(
            (chunk, chunkIndex) => [`${name} chunk ${chunkIndex}`, chunk] as const
          )
        : []
    )
  );
}

/**
 * Pre-checks all plan fields plus parameter scratch, without declaring anything. The buffer may
 * exceed maxStorageBufferBindingSize: each dispatch binds only three words at an aligned slot.
 * Its full allocation must fit maxBufferSize. Call before any other contributor allocations.
 */
export function validateGraphChunkDispatchBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  plan: GraphChunkDispatchPlan
): void {
  validateNamedChunkBindingRanges(
    graph,
    id,
    plan.chunks.flatMap(chunk =>
      Object.entries(chunk.views).map(
        ([name, view]) => [`${name} chunk ${chunk.chunkIndex}`, view] as const
      )
    )
  );
  if (plan.chunks.length === 0) return;
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'chunk parameter slot',
    GRAPH_CHUNK_PARAMETER_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT,
    'use a device that supports the chunk parameter slot'
  );
  validateCapacity(
    id,
    'chunk parameter buffer byteLength',
    plan.chunks.length * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT,
    0,
    graph.device.limits.maxBufferSize
  );
}

/**
 * Allocates one graph-owned uint32 parameter buffer with 256-byte slots. `slots[i]` binds exactly
 * `[rowBase, length, totalLength]` for `plan.chunks[i]`, always at element offset zero, so every
 * chunk can share one pipeline. Append the returned upload node BEFORE dispatch nodes; it copies
 * constants on every encoding, preserving transient aliasing. The compiled upload node owns its
 * staging buffer and destroys it. No slots or resources are allocated for a zero-chunk plan.
 *
 * Run `validateGraphChunkDispatchBindings` and all other binding/scratch pre-checks at the start
 * of `getCommandNodes`. This function repeats the plan check before allocation. Include one upload
 * node plus your nonempty chunk dispatches in the contributor's construction-time node budget.
 * Dispatch bounds must read slot word 1 (or `arrayLength`), not vary pipeline constants. Use a
 * published `Kernel` with each node's workgroup count computed from its own chunk length by
 * `getBoundedDispatchLayout`. Read `num_workgroups` to flatten the bounded dispatch in fixed WGSL.
 * Do not pad every dispatch to the largest chunk to keep its pipeline constants fixed.
 */
export function createGraphChunkDispatchParameters<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; plan: GraphChunkDispatchPlan}
): {slots: readonly GraphDataView<'uint32'>[]; nodes: readonly GPUCommandNode<Parameters>[]} {
  const {id, plan} = props;
  validateGraphChunkDispatchBindings(graph, id, plan);
  if (plan.chunks.length === 0) return {slots: [], nodes: []};
  const values = new Uint32Array(plan.chunks.length * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT);
  for (const [chunkIndex, chunk] of plan.chunks.entries()) {
    values.set(
      [chunk.rowBase, chunk.length, plan.totalLength],
      chunkIndex * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT
    );
  }
  const {parameters, nodes} = createSpatialStatisticsParameterNodes(graph, id, values);
  const slots = plan.chunks.map((_, chunkIndex) =>
    graph.createDataView(parameters.buffer, {
      format: 'uint32',
      length: GRAPH_CHUNK_PARAMETER_WORD_COUNT,
      byteOffset: chunkIndex * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT
    })
  );
  return {slots, nodes};
}

/** Checks pre-labelled chunk ranges without losing the original plan's chunk indices. */
function validateNamedChunkBindingRanges<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  chunks: readonly (readonly [string, GraphDataView])[]
): void {
  validateGraphViewsBelongToGraph(
    id,
    graph,
    chunks.map(([, view]) => view)
  );
  for (const [name, chunk] of chunks) {
    const range = getViewBindingRange(chunk);
    validateRequirement(
      id,
      (chunk.buffer.usage & Buffer.STORAGE) !== 0,
      `${name} buffer usage must include Buffer.STORAGE`
    );
    validateRequirement(
      id,
      range.size > 0 && range.size % GRAPH_CHUNK_STORAGE_BINDING_BYTE_ALIGNMENT === 0,
      `${name} aligned binding size (${range.size}) must be positive and a multiple of GRAPH_CHUNK_STORAGE_BINDING_BYTE_ALIGNMENT (${GRAPH_CHUNK_STORAGE_BINDING_BYTE_ALIGNMENT})`
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      range.size,
      'reduce the chunk row count or output capacity'
    );
    validateRequirement(
      id,
      range.offset + range.size <= chunk.buffer.byteLength,
      `${name} aligned binding must fit its buffer byteLength (${chunk.buffer.byteLength})`
    );
  }
}
