// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPURankMarkov} from './gpu-rank-markov';

/** Shared graph-value properties. All properties are compile-time; vector contents are per-frame. */
export type RankMarkovOperationProps = {
  id?: string;
  values: string;
  rowCount: number;
  periodCount: number;
  mask?: string;
  mode?: 'full-rank' | 'geographic-rank';
  tieOrder?: 'giddy' | 'row-id';
  output: {
    ranks: string;
    counts: string;
    probabilities?: string;
    rowTotals?: string;
    ignored?: string;
    mobility?: string;
    steadyStates?: string;
    stateSparsity?: string;
    status?: {converged: string};
  };
};

/**
 * Vector operation form of GPURankMarkov. Compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * Homogeneity is unavailable: the contributor has one unconditional chain.
 */
export class GPURankMarkovOperation extends GPUContributorOperation {
  constructor(props: RankMarkovOperationProps) {
    const output = props.output;
    const ports = {
      ranks: {
        value: output.ranks,
        format: 'uint32' as const,
        length: props.rowCount * props.periodCount
      },
      counts: {value: output.counts, format: 'uint32' as const, length: props.rowCount ** 2},
      ...(output.probabilities !== undefined
        ? {
            probabilities: {
              value: output.probabilities,
              format: 'float32' as const,
              length: props.rowCount ** 2
            }
          }
        : {}),
      ...(output.rowTotals !== undefined
        ? {rowTotals: {value: output.rowTotals, format: 'uint32' as const, length: props.rowCount}}
        : {}),
      ...(output.ignored !== undefined
        ? {ignored: {value: output.ignored, format: 'uint32' as const, length: 1}}
        : {}),
      ...(output.mobility !== undefined
        ? {
            mobility: {
              value: output.mobility,
              format: 'float32' as const,
              length: props.periodCount - 1
            }
          }
        : {}),
      ...(output.steadyStates !== undefined
        ? {
            steadyStates: {
              value: output.steadyStates,
              format: 'float32' as const,
              length: props.rowCount
            }
          }
        : {}),
      ...(output.stateSparsity !== undefined
        ? {stateSparsity: {value: output.stateSparsity, format: 'uint32' as const, length: 2}}
        : {}),
      ...(output.status !== undefined
        ? {converged: {value: output.status.converged, format: 'uint32' as const, length: 1}}
        : {})
    };
    super({
      id: props.id ?? 'rank-markov',
      inputs: {values: props.values, ...(props.mask !== undefined ? {mask: props.mask} : {})},
      outputs: ports,
      createContributor: getView =>
        new GPURankMarkov({
          id: props.id,
          values: getView<'float32'>('values'),
          periodCount: props.periodCount,
          mask: props.mask === undefined ? undefined : getView<'uint32'>('mask'),
          mode: props.mode,
          tieOrder: props.tieOrder,
          output: {
            ranks: getView<'uint32'>('ranks'),
            counts: getView<'uint32'>('counts'),
            probabilities:
              output.probabilities === undefined ? undefined : getView<'float32'>('probabilities'),
            rowTotals: output.rowTotals === undefined ? undefined : getView<'uint32'>('rowTotals'),
            ignored: output.ignored === undefined ? undefined : getView<'uint32'>('ignored'),
            mobility: output.mobility === undefined ? undefined : getView<'float32'>('mobility'),
            steadyStates:
              output.steadyStates === undefined ? undefined : getView<'float32'>('steadyStates'),
            stateSparsity:
              output.stateSparsity === undefined ? undefined : getView<'uint32'>('stateSparsity'),
            status:
              output.status === undefined ? undefined : {converged: getView<'uint32'>('converged')}
          }
        })
    });
  }
}
