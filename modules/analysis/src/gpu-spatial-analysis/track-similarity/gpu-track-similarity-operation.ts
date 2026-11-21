// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUContributorOperation,
  type GPUContributorOperationOutput
} from '../../analysis-core/operations/gpu-contributor-operation';
import {GPUTrackSimilarity, type GPUTrackSimilarityProps} from './gpu-track-similarity';
import {validateMultipartTrackMeasures} from './multipart-track-layout';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Graph-value properties shared by the GPU and CPU track similarity operations. */
export type TrackSimilarityOperationProps = Omit<
  GPUTrackSimilarityProps,
  'linesA' | 'linesB' | 'pairA' | 'pairB' | 'activePairCount' | 'frechetEpsilon' | 'output'
> & {
  linesA: {positions: string; lineOffsets: string; featureOffsets?: string};
  linesB?: {positions: string; lineOffsets: string; featureOffsets?: string};
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
    props = snapshotTrackSimilarityOperationProps(props);
    validateMultipartTrackMeasures(
      props.id ?? 'track-similarity',
      Boolean(props.linesA.featureOffsets || props.linesB?.featureOffsets),
      Object.keys(props.output).filter(
        key => props.output[key as keyof typeof props.output] !== undefined
      )
    );
    const inputs: Record<string, string> = {
      positionsA: props.linesA.positions,
      offsetsA: props.linesA.lineOffsets,
      pairA: props.pairA,
      pairB: props.pairB
    };
    if (props.linesA.featureOffsets) inputs['featuresA'] = props.linesA.featureOffsets;
    if (props.linesB) {
      inputs['positionsB'] = props.linesB.positions;
      inputs['offsetsB'] = props.linesB.lineOffsets;
      if (props.linesB.featureOffsets) inputs['featuresB'] = props.linesB.featureOffsets;
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
      emptyOutputPorts: Object.keys(outputs),
      inputs,
      outputs,
      createContributor: getView =>
        new GPUTrackSimilarity({
          ...props,
          linesA: {
            kind: 'lines',
            positions: getView<'float32x2'>('positionsA'),
            lineOffsets: getView<'uint32'>('offsetsA'),
            ...(props.linesA.featureOffsets ? {featureOffsets: getView<'uint32'>('featuresA')} : {})
          },
          linesB: props.linesB
            ? {
                kind: 'lines',
                positions: getView<'float32x2'>('positionsB'),
                lineOffsets: getView<'uint32'>('offsetsB'),
                ...(props.linesB.featureOffsets
                  ? {featureOffsets: getView<'uint32'>('featuresB')}
                  : {})
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

/** @internal Shared descriptor snapshot for both operation forms. */
export function snapshotTrackSimilarityOperationProps(
  props: TrackSimilarityOperationProps
): TrackSimilarityOperationProps {
  return Object.freeze({
    ...props,
    linesA: snapshotConstructionFields(props.linesA, [
      'positions',
      'lineOffsets',
      'featureOffsets'
    ]),
    linesB:
      props.linesB &&
      snapshotConstructionFields(props.linesB, ['positions', 'lineOffsets', 'featureOffsets']),
    output: Object.freeze({...props.output}),
    editDistanceGapPoint:
      props.editDistanceGapPoint &&
      (Object.freeze([...props.editDistanceGapPoint]) as readonly [number, number])
  });
}
