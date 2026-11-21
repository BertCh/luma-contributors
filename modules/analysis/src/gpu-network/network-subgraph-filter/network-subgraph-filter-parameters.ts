// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getNetworkReadbackWords,
  getNetworkReadbackBytes,
  type NetworkReadbackBytes
} from '../network-statistics/network-packed-readback';

/** Floats per predicate record in the {@link GPUNetworkSubgraphFilter} `parameters` view. */
export const GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE = 4;

/** Word indices of the optional `counts` output. */
export const GPU_NETWORK_SUBGRAPH_FILTER_COUNT_WORD = {
  /** Nodes with a nonzero `nodeMask`. */
  liveNodeCount: 0,
  /** CSR slots with a nonzero `edgeMask`. */
  liveSlotCount: 1,
  /** Live slots u -> u. */
  selfLoopSlotCount: 2,
  /**
   * Directed: `liveSlotCount`. Undirected: live non-self-loop slots / 2 plus live self-loop slots
   * (the same rule as `GPUNetworkStatistics`).
   */
  liveEdgeCount: 3
} as const;

/** Number of words of the `counts` output. */
export const GPU_NETWORK_SUBGRAPH_FILTER_COUNT_LENGTH = 4;

/**
 * Per-frame description of the f32 predicates of one {@link GPUNetworkSubgraphFilter}.
 *
 * Every entry is one `[min, max]` pair, or `null` / `undefined` to disable that predicate. Attribute
 * ranges are half-open: a row passes when `min <= value < max` (the `GPUAttributeCrossfilter` brush
 * rule; use `Infinity` as the maximum to include everything above the minimum). A NaN value never
 * passes an enabled predicate. The optional edge time window is closed on both ends, like the
 * `GPUTimeWindowFilter` windows: `start <= time <= end`.
 */
export type GPUNetworkSubgraphFilterState = {
  /** Half-open ranges in each node column's units, in column order. */
  nodeAttributeRanges?: readonly (readonly [number, number] | null | undefined)[];
  /** Half-open ranges in each edge column's units, in column order. */
  edgeAttributeRanges?: readonly (readonly [number, number] | null | undefined)[];
  /** Closed window in the units of the float32 edgeTimes column. */
  closedEdgeTimeWindow?: readonly [number, number] | null;
};

/** Shape needed to size and pack the parameter view. */
export type GPUNetworkSubgraphFilterParameterLayout = {
  /** Number of node columns. */
  nodeColumnCount: number;
  /** Number of edge columns. */
  edgeColumnCount: number;
  /** Whether an f32 edge time column exists. */
  hasEdgeTimes: boolean;
};

/** Returns the minimum `parameters` view length (float32 rows) for a layout. */
export function getGPUNetworkSubgraphFilterParameterLength(
  layout: GPUNetworkSubgraphFilterParameterLayout
): number {
  return (
    (layout.nodeColumnCount + layout.edgeColumnCount + (layout.hasEdgeTimes ? 1 : 0)) *
    GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE
  );
}

/**
 * Packs ranges into the float32 layout the contributor reads.
 *
 * Records of {@link GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE} floats `[min, max, enabled, 0]`:
 * node columns first, then edge columns, then the f32 edge time window. Write the result into the
 * caller-owned parameter buffer between encodings; nothing recompiles.
 *
 * Missing, `null` and `undefined` entries are disabled predicates. Attribute endpoints round
 * upward; closed time starts round upward and ends downward. Thus float32 rows obey the
 * original JavaScript inequalities. Infinite endpoints are preserved.
 *
 * @throws If `target` is too short.
 */
export function getGPUNetworkSubgraphFilterParameterValues(
  layout: GPUNetworkSubgraphFilterParameterLayout,
  state: GPUNetworkSubgraphFilterState = {},
  target: Float32Array = new Float32Array(getGPUNetworkSubgraphFilterParameterLength(layout))
): Float32Array {
  if (target.length < getGPUNetworkSubgraphFilterParameterLength(layout)) {
    throw new Error('GPUNetworkSubgraphFilter parameter target is too short');
  }
  const stride = GPU_NETWORK_SUBGRAPH_FILTER_PARAMETER_STRIDE;
  const id = 'GPUNetworkSubgraphFilter';
  for (const [removed, replacement] of [
    ['nodeRanges', 'nodeAttributeRanges'],
    ['edgeRanges', 'edgeAttributeRanges'],
    ['edgeTimeWindow', 'closedEdgeTimeWindow']
  ] as const) {
    validateRequirement(id, !(removed in state), `${removed} was removed; use ${replacement}`);
  }
  const {nodeAttributeRanges, edgeAttributeRanges, closedEdgeTimeWindow} = state;
  validateRanges('nodeAttributeRanges', nodeAttributeRanges, layout.nodeColumnCount);
  validateRanges('edgeAttributeRanges', edgeAttributeRanges, layout.edgeColumnCount);
  if (layout.hasEdgeTimes) validateRanges('closedEdgeTimeWindow', [closedEdgeTimeWindow], 1);
  const records: (readonly [number, number] | null | undefined)[] = [];
  for (let column = 0; column < layout.nodeColumnCount; column++)
    records.push(nodeAttributeRanges?.[column]);
  for (let column = 0; column < layout.edgeColumnCount; column++)
    records.push(edgeAttributeRanges?.[column]);
  if (layout.hasEdgeTimes) records.push(closedEdgeTimeWindow);
  target.fill(0, 0, getGPUNetworkSubgraphFilterParameterLength(layout));
  for (const [recordIndex, range] of records.entries()) {
    if (!range) continue;
    target[recordIndex * stride] = roundRangeBound(range[0], true);
    target[recordIndex * stride + 1] = roundRangeBound(
      range[1],
      recordIndex < layout.nodeColumnCount + layout.edgeColumnCount
    );
    target[recordIndex * stride + 2] = 1;
  }
  return target;
}

type Ranges = readonly (readonly [number, number] | null | undefined)[];

function validateRanges(name: string, ranges: Ranges | undefined, activeCount: number): void {
  for (let index = 0; index < activeCount; index++) {
    const range = ranges?.[index];
    if (!range) continue;
    for (const [endpoint, value] of range.entries()) {
      validateRequirement(
        'GPUNetworkSubgraphFilter',
        typeof value === 'number' && !Number.isNaN(value),
        `${name}[${index}][${endpoint}] must be a number; received ${value}`
      );
    }
  }
}

/** Directed IEEE binary32 rounding preserves comparisons against representable column values. */
function roundRangeBound(value: number, upward: boolean): number {
  const rounded = Math.fround(value);
  if (upward ? rounded >= value : rounded <= value) return rounded;
  const words = new Uint32Array(1);
  const floats = new Float32Array(words.buffer);
  floats[0] = rounded;
  if (rounded === 0) return upward ? 2 ** -149 : -(2 ** -149);
  words[0] += rounded > 0 === upward ? 1 : -1;
  return floats[0];
}

/** Decoded `counts` output. */
export type GPUNetworkSubgraphFilterCounts = {
  liveNodeCount: number;
  liveSlotCount: number;
  selfLoopSlotCount: number;
  liveEdgeCount: number;
};

/** Checked numeric-word reader for the 4-word record. */
export function decodeGPUNetworkSubgraphFilterCountsWords(
  values: Uint32Array | readonly number[]
): GPUNetworkSubgraphFilterCounts {
  const words = getNetworkReadbackWords('decodeGPUNetworkSubgraphFilterCountsWords', values, 4);
  const word = GPU_NETWORK_SUBGRAPH_FILTER_COUNT_WORD;
  return {
    liveNodeCount: words[word.liveNodeCount],
    liveSlotCount: words[word.liveSlotCount],
    selfLoopSlotCount: words[word.selfLoopSlotCount],
    liveEdgeCount: words[word.liveEdgeCount]
  };
}

/** Little-endian byte reader, accepting `{bytes: await buffer.readAsync()}`. */
export function decodeGPUNetworkSubgraphFilterCountsFromBytes(
  values: NetworkReadbackBytes
): GPUNetworkSubgraphFilterCounts {
  return decodeGPUNetworkSubgraphFilterCountsWords(
    getNetworkReadbackBytes('decodeGPUNetworkSubgraphFilterCountsFromBytes', values, 4)
  );
}
