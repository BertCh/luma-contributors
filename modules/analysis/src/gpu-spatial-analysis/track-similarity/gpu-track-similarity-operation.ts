// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrackSimilarity, type GPUTrackSimilarityProps} from './gpu-track-similarity';

/** Graph-value properties shared by the GPU and CPU track similarity operations. */
export type TrackSimilarityOperationProps = Omit<
  GPUTrackSimilarityProps,
  'linesA' | 'linesB' | 'pairA' | 'pairB' | 'activePairCount' | 'frechetEpsilon' | 'output'
> & {
  linesA: {positions: string; lineOffsets: string};
  linesB?: {positions: string; lineOffsets: string};
  pairA: string;
  pairB: string;
  /** Explicit pair capacity, inferred from pairA.length by the contributor. */
  pairCount: number;
  activePairCount?: string;
  frechetEpsilon?: string;
  output: {[Port in keyof GPUTrackSimilarityProps['output']]?: string};
};

/**
 * Vector operation form of GPUTrackSimilarity; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING.
 * All props are compile-time; input vector contents can change on each run. Contributor validation
 * runs when the graph resolves its views. CPUTrackSimilarityOperation is the uncapped CPU twin.
 */
export class GPUTrackSimilarityOperation extends GPUContributorOperation {
  constructor(props: TrackSimilarityOperationProps) {
    const inputs: Record<string, string> = {
      positionsA: props.linesA.positions,
      offsetsA: props.linesA.lineOffsets,
      pairA: props.pairA,
      pairB: props.pairB
    };
    if (props.linesB) {
      inputs['positionsB'] = props.linesB.positions;
      inputs['offsetsB'] = props.linesB.lineOffsets;
    }
    if (props.activePairCount !== undefined) inputs['activePairCount'] = props.activePairCount;
    if (props.frechetEpsilon !== undefined) inputs['frechetEpsilon'] = props.frechetEpsilon;
    const outputs: Record<string, GPUContributorOperationOutput> = {};
    for (const [port, value] of Object.entries(props.output)) {
      if (value !== undefined) {
        outputs[port] = {
          value,
          format: port === 'pairStatus' || port === 'frechetWithin' ? 'uint32' : 'float32',
          length: props.pairCount
        };
      }
    }
    super({
      id: props.id ?? 'track-similarity',
      inputs,
      outputs,
      createContributor: getView =>
        new GPUTrackSimilarity({
          ...props,
          linesA: {
            kind: 'lines',
            positions: getView<'float32x2'>('positionsA'),
            lineOffsets: getView<'uint32'>('offsetsA')
          },
          linesB: props.linesB
            ? {
                kind: 'lines',
                positions: getView<'float32x2'>('positionsB'),
                lineOffsets: getView<'uint32'>('offsetsB')
              }
            : undefined,
          pairA: getView<'uint32'>('pairA'),
          pairB: getView<'uint32'>('pairB'),
          activePairCount:
            props.activePairCount === undefined ? undefined : getView<'uint32'>('activePairCount'),
          frechetEpsilon:
            props.frechetEpsilon === undefined ? undefined : getView<'float32'>('frechetEpsilon'),
          output: Object.fromEntries(Object.keys(outputs).map(port => [port, getView(port)]))
        })
    });
  }
}
