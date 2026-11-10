// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUSpatialWeights} from './spatial-weights';
import {createCSRStages} from './spatial-weights-algebra-wgsl';
import type {RowLocalStageBase} from './spatial-weights-row-local-stages';

/** Cell and period order of the rows of a space-time weights matrix. */
export type GPUSpaceTimeWeightsLayout = 'cell-major' | 'period-major';

/** One temporal lag of a space-time weights matrix. */
export type GPUSpatialWeightsSpaceTimeLag = {
  /**
   * Integer lag in periods. Row (cell `i`, period `t`) lists period `t - lag`, so a positive lag
   * looks into the past, `0` is the same period and a negative lag looks ahead. Entries whose
   * period falls outside `[0, periodCount)` are dropped (no wrap-around).
   */
  lag: number;
  /** Temporal weight `T[t, t - lag]`, positive and finite. Defaults to 1. */
  weight?: number;
  /**
   * Also list the same cell at the lagged period with the unit spatial weight 1 (the identity
   * part of `I + W`), in place of a diagonal `w_ii` slot of the spatial weights. Not allowed at
   * lag 0, where it would be the row itself.
   */
  includeSelf?: boolean;
};

/** Builds the CSR stages of `T (x) W`. @internal */
export function getSpaceTimeStages<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  base: RowLocalStageBase,
  props: {
    weights: GPUSpatialWeights;
    output: GPUSpatialWeights;
    cellCount: number;
    periodCount: number;
    layout: GPUSpaceTimeWeightsLayout;
    lags: readonly GPUSpatialWeightsSpaceTimeLag[];
  }
): GPUCommandNode<Parameters>[] {
  const {weights, output, cellCount, periodCount, layout} = props;
  // Ascending source period (descending lag) keeps the neighbor IDs ascending within a block.
  const lags = [...props.lags].sort((left, right) => right.lag - left.lag);
  const hasSelf = lags.some(entry => entry.includeSelf);
  const rowId = (cell: string, source: string) =>
    layout === 'period-major' ? `(${source}) * CELLS + ${cell}` : `${cell} * PERIODS + (${source})`;

  // Cell-major: emits, for one slot, every lag in ascending source period.
  const emitCandidate = (onItem: string) =>
    lags
      .map((entry, position) => {
        const source = `sourceOf${position}`;
        const temporalWeight = getWGSLFloatLiteral(entry.weight ?? 1);
        const spatial = entry.includeSelf
          ? 'select(spatialWeight, 1.0, neighbor == cell)'
          : 'spatialWeight';
        return `if (${source} >= 0 && ${source} < i32(PERIODS)) {
        let id = ${rowId('neighbor', `u32(${source})`)};
        let weightValue = ${temporalWeight} * ${spatial};
        ${onItem}
      }`;
      })
      .join('\n      ');

  const emitSelfOnly = (onItem: string) =>
    lags
      .map((entry, position) => {
        if (!entry.includeSelf) {
          return '';
        }
        const source = `sourceOf${position}`;
        return `if (${source} >= 0 && ${source} < i32(PERIODS)) {
        let id = ${rowId('cell', `u32(${source})`)};
        let weightValue = ${getWGSLFloatLiteral(entry.weight ?? 1)};
        ${onItem}
      }`;
      })
      .join('\n      ');

  const sourceDeclarations = lags
    .map((entry, position) => `let sourceOf${position}: i32 = i32(period) - (${entry.lag});`)
    .join('\n  ');

  // Period-major: one block per lag, each walking the slots once. Cell-major: one walk over the
  // slots, each slot expanded over every lag.
  const visit = (onItem: string) => {
    const locate =
      layout === 'period-major'
        ? `let period = index / CELLS;
  let cell = index % CELLS;`
        : `let cell = index / PERIODS;
  let period = index % PERIODS;`;
    if (layout === 'cell-major') {
      return `${locate}
  ${sourceDeclarations}
  let spatialBegin = min(baseOffsets[baseOffsetsOffset + cell], BASE_CAPACITY);
  let spatialEnd = min(baseOffsets[baseOffsetsOffset + cell + 1u], BASE_CAPACITY);
  var selfPending = ${hasSelf ? 'true' : 'false'};
  for (var slot = spatialBegin; slot < spatialEnd; slot++) {
    let neighbor = baseNeighbors[baseNeighborsOffset + slot];
    if (neighbor >= CELLS) {
      continue;
    }
    if (selfPending && neighbor >= cell) {
      selfPending = false;
      if (neighbor > cell) {
        ${emitSelfOnly(onItem)}
      }
    }
    let spatialWeight = baseWeights[baseWeightsOffset + slot];
    ${emitCandidate(onItem)}
  }
  if (selfPending) {
    ${emitSelfOnly(onItem)}
  }`;
    }
    // Period-major.
    const blocks = lags
      .map((entry, position) => {
        const source = `sourceOf${position}`;
        const temporalWeight = getWGSLFloatLiteral(entry.weight ?? 1);
        const itemFor = (candidate: string, weightExpression: string) => `{
          let id = ${rowId(candidate, `u32(${source})`)};
          let weightValue = ${temporalWeight} * ${weightExpression};
          ${onItem}
        }`;
        return `if (${source} >= 0 && ${source} < i32(PERIODS)) {
    var selfPending = ${entry.includeSelf ? 'true' : 'false'};
    for (var slot = spatialBegin; slot < spatialEnd; slot++) {
      let neighbor = baseNeighbors[baseNeighborsOffset + slot];
      if (neighbor >= CELLS) {
        continue;
      }
      if (selfPending && neighbor >= cell) {
        selfPending = false;
        ${itemFor('cell', '1.0')}
        if (neighbor == cell) {
          continue;
        }
      }
      ${itemFor('neighbor', 'baseWeights[baseWeightsOffset + slot]')}
    }
    if (selfPending) {
      ${itemFor('cell', '1.0')}
    }
  }`;
      })
      .join('\n  ');
    return `${locate}
  ${sourceDeclarations}
  let spatialBegin = min(baseOffsets[baseOffsetsOffset + cell], BASE_CAPACITY);
  let spatialEnd = min(baseOffsets[baseOffsetsOffset + cell + 1u], BASE_CAPACITY);
  ${blocks}`;
  };

  return createCSRStages(graph, {
    ...base,
    bindings: [
      {name: 'baseOffsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'baseNeighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'baseWeights', view: weights.weights, type: 'f32', access: 'read'}
    ],
    declarations: `const CELLS: u32 = ${cellCount}u;
const PERIODS: u32 = ${periodCount}u;
const BASE_CAPACITY: u32 = ${weights.neighbors.length}u;`,
    visit,
    attributes: [
      {name: 'neighbors', output: output.neighbors, type: 'u32', value: 'id'},
      {name: 'weights', output: output.weights, type: 'f32', value: 'weightValue'}
    ]
  });
}
