// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGPUTemporalContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/index';
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUSpaceTimeRelativeTimes,
  type GPUSpaceTimeRelativeTimes
} from './space-time-parameters';

/** Largest number of bands along one axis of the near-repeat table. */
export const GPU_NEAR_REPEAT_MAXIMUM_BANDS = 16;

/** Largest number of cells (distance bands times time bands) of the near-repeat table. */
export const GPU_NEAR_REPEAT_MAXIMUM_CELLS = 256;

/**
 * The float32 spacing of relative times may be at most this fraction of the narrowest time band or
 * time threshold. Float32 rounding of two times moves their difference by up to one spacing, so a
 * pair can change band only when it lies within that spacing of an edge.
 */
export const GPU_SPACE_TIME_BAND_RESOLUTION_FRACTION = 0.25;

/** Value of a band lookup outside every band. @internal */
export const SPACE_TIME_NO_BAND_WGSL = '0xffffffffu';

/**
 * Validates band edges: finite, strictly ascending, non-negative, at most
 * {@link GPU_NEAR_REPEAT_MAXIMUM_BANDS} bands. @internal
 */
export function validateSpaceTimeBandEdges(
  id: string,
  name: string,
  edges: readonly number[]
): void {
  if (edges.length < 2 || edges.length > GPU_NEAR_REPEAT_MAXIMUM_BANDS + 1) {
    throw new Error(`${id} ${name} needs 2 to ${GPU_NEAR_REPEAT_MAXIMUM_BANDS + 1} edges`);
  }
  edges.forEach((edge, index) => {
    // Edges compare as float32 on the GPU: two edges that round together would make an empty band.
    if (!Number.isFinite(Math.fround(edge)) || edge < 0) {
      throw new Error(`${id} ${name} edges must be finite and non-negative`);
    }
    if (index > 0 && Math.fround(edge) <= Math.fround(edges[index - 1])) {
      throw new Error(`${id} ${name} edges must be strictly ascending as float32`);
    }
  });
}

/** Formats a float32 value as a WGSL `f32` literal. */
function getFloat32Literal(value: number): string {
  const text = String(Math.fround(value));
  return /[.e]/.test(text) ? text : `${text}.0`;
}

/**
 * WGSL `get<Name>Band(value: f32) -> u32`: the band of `value` for `edges`, or `NO_BAND`. Band `k`
 * is `(edges[k], edges[k + 1]]` and band 0 also holds `edges[0]` itself, the convention of the
 * `<=` thresholds of `GPUKnoxTest` and of pointpats. Unrolled over the edges.
 * @internal
 */
export function getSpaceTimeBandWGSL(name: string, edges: readonly number[]): string {
  const lines = edges
    .slice(1, -1)
    .map((edge, index) => `  if (value > ${getFloat32Literal(edge)}) { band = ${index + 1}u; }`);
  return /* wgsl */ `
fn get${name}Band(value: f32) -> u32 {
  if (value < ${getFloat32Literal(edges[0])} || value > ${getFloat32Literal(edges[edges.length - 1])}) {
    return ${SPACE_TIME_NO_BAND_WGSL};
  }
  var band = 0u;
${lines.join('\n')}
  return band;
}
`;
}

/**
 * Validates an optional `temporalContext` of the space-time tests: the times are `float32`
 * ticks counted from the context epoch, which must be declared (`float32-relative` rejects
 * `'unix'`). Returns the context unchanged.
 * @internal
 */
export function resolveGPUSpaceTimeTemporalContext(
  id: string,
  context: GPUTemporalContext | undefined
): GPUTemporalContext | undefined {
  return context && resolveGPUTemporalContext(id, context, {encodings: ['float32-relative']});
}

/** Options of {@link getGPUSpaceTimeGuardedRelativeTimes}. */
export type GPUSpaceTimeGuardedRelativeTimesOptions = {
  /**
   * The float64 origin to subtract, in ticks. Defaults to the minimum time, rounded down to a whole
   * tick when a `temporalContext` is given, so the result can declare an integer epoch.
   */
  origin?: number;
  /**
   * What the input numbers mean: ticks of `unit` after `epoch`. With it the result carries the
   * context of the float32 times (`float32-relative`, the epoch moved to the origin).
   */
  temporalContext?: Pick<GPUTemporalContext, 'unit' | 'epoch' | 'timeZone'>;
};

/** Result of {@link getGPUSpaceTimeGuardedRelativeTimes}. */
export type GPUSpaceTimeGuardedRelativeTimes = GPUSpaceTimeRelativeTimes & {
  /** The context of `times`, present when the options gave a `temporalContext`. */
  temporalContext?: GPUTemporalContext;
};

/**
 * {@link getGPUSpaceTimeRelativeTimes} with the float32 guard of the band tests: throws when the
 * float32 spacing of the relative times exceeds {@link GPU_SPACE_TIME_BAND_RESOLUTION_FRACTION} of
 * `narrowestTimeWidth`, the narrowest time band (or the time threshold) in the unit of the times,
 * so a lost resolution is never silent.
 *
 * @param times Event times in ticks, finite.
 * @param narrowestTimeWidth Narrowest time band or time threshold, in ticks; positive.
 * @param options Origin and the context of the input ticks.
 */
export function getGPUSpaceTimeGuardedRelativeTimes(
  times: ArrayLike<number>,
  narrowestTimeWidth: number,
  options: GPUSpaceTimeGuardedRelativeTimesOptions = {}
): GPUSpaceTimeGuardedRelativeTimes {
  if (!(narrowestTimeWidth > 0) || !Number.isFinite(narrowestTimeWidth)) {
    throw new Error('space-time narrowest time width must be positive and finite');
  }
  const {temporalContext} = options;
  let {origin} = options;
  if (temporalContext && origin === undefined) {
    let minimum = Infinity;
    for (let index = 0; index < times.length; index++) {
      minimum = Math.min(minimum, times[index]);
    }
    origin = Number.isFinite(minimum) ? Math.floor(minimum) : undefined;
  }
  const relative = getGPUSpaceTimeRelativeTimes(
    times,
    origin,
    narrowestTimeWidth * GPU_SPACE_TIME_BAND_RESOLUTION_FRACTION
  );
  if (!temporalContext) {
    return relative;
  }
  if (!Number.isSafeInteger(relative.origin)) {
    throw new Error('space-time origin must be a whole number of ticks to declare an epoch');
  }
  const {unit, epoch, timeZone} = temporalContext;
  return {
    ...relative,
    temporalContext: {
      unit,
      epoch: (epoch === 'unix' ? 0n : epoch) + BigInt(relative.origin),
      encoding: 'float32-relative',
      ...(timeZone === undefined ? {} : {timeZone}),
      axis: {kind: 'irregular'}
    }
  };
}

/**
 * Creates the `entry-mask` node: writes 1 into `entryMask[entry]` for every CSR entry that is the
 * first occurrence of a valid, non-self neighbour in its row, and 0 otherwise, so a row's own
 * neighbours are visited once each however the list repeats them. Unlike the pair mask it does not
 * merge the two directions of a pair.
 *
 * @internal
 */
export function createSpaceTimeEntryMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rows: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    entryMask: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-entry-mask`,
    operation: props.operation,
    variant: 'entry-mask',
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'entryMask', view: props.entryMask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.rows,
    declarations: `const ROWS: u32 = ${props.rows}u;`,
    body: `let start = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var entry = start; entry < end; entry++) {
    let other = neighbors[neighborsOffset + entry];
    var counted = other < ROWS && other != index;
    if (counted) {
      for (var earlier = start; earlier < entry; earlier++) {
        if (neighbors[neighborsOffset + earlier] == other) {
          counted = false;
          break;
        }
      }
    }
    entryMask[entryMaskOffset + entry] = select(0u, 1u, counted);
  }`
  });
}
