// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSamplingSeed} from './sampling-seed';
import {snapshotContinuousSampleFrame} from './continuous-sample-frame-construction';
import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUContinuousSpatialSample,
  type GPUContinuousSpatialSampleProps
} from './gpu-continuous-spatial-sample';

/** Graph-value names replace views; frameRowCount and legacySiteCount make their lengths explicit. */
export type ContinuousSpatialSampleOperationProps = Pick<
  GPUContinuousSpatialSampleProps,
  | 'id'
  | 'spatialContext'
  | 'spacing'
  | 'seed'
  | 'replicate'
  | 'waves'
  | 'capacity'
  | 'maximumCandidates'
> & {
  frameRowCount: number;
  legacySiteCount?: number;
  frame: {positions: string; legacySites?: string} & (
    | {type: 'line'}
    | {type: 'area'; bounds: readonly [number, number, number, number]}
  );
  parameters?: string;
  output: {positions: string; designWeights: string; provenance: string; status: string};
};

/** @internal Capture schema and population settings for both operation backends. */
export function snapshotContinuousSampleOperation(
  props: ContinuousSpatialSampleOperationProps
): ContinuousSpatialSampleOperationProps {
  const snapshot = snapshotConstructionFields(props, [
    'id',
    'spatialContext',
    'spacing',
    'seed',
    'replicate',
    'waves',
    'capacity',
    'maximumCandidates',
    'frameRowCount',
    'legacySiteCount',
    'frame',
    'parameters',
    'output'
  ]);
  return Object.freeze({
    ...snapshot,
    frame: snapshotContinuousSampleFrame(
      snapshot.id ?? 'continuous-spatial-sample',
      snapshot.frame
    ),
    output: snapshotConstructionFields(snapshot.output, [
      'positions',
      'designWeights',
      'provenance',
      'status'
    ])
  });
}

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUContinuousSpatialSampleOperation extends GPUContributorOperation {
  constructor(props: ContinuousSpatialSampleOperationProps) {
    props = snapshotContinuousSampleOperation(props);
    const id = props.id ?? 'continuous-spatial-sample';
    validateRequirement(
      id,
      ['line', 'area'].includes(props.frame.type),
      `frame.type must be line or area; received ${props.frame.type}`
    );
    validateSamplingSeed(id, props.seed, props.parameters !== undefined);
    const frameRowCount = props.frameRowCount;
    const legacySiteCount = props.legacySiteCount ?? 0;
    const waves = props.waves ?? 1;
    const length = waves * props.capacity;
    super({
      id: props.id ?? 'continuous-spatial-sample',
      inputs: {
        positions: props.frame.positions,
        ...(props.frame.legacySites !== undefined ? {legacySites: props.frame.legacySites} : {}),
        ...(props.parameters !== undefined ? {parameters: props.parameters} : {})
      },
      outputs: {
        positionsOutput: {value: props.output.positions, format: 'float32x2', length},
        designWeights: {value: props.output.designWeights, format: 'float32', length},
        provenance: {value: props.output.provenance, format: 'uint32x4', length},
        status: {value: props.output.status, format: 'uint32x4', length: waves}
      },
      createContributor: getView => {
        const positions = getView<'float32x2'>('positions');
        validateViewLength(id, 'frame.positions (frameRowCount)', positions, frameRowCount);
        const legacySites =
          props.frame.legacySites === undefined ? undefined : getView('legacySites');
        validateRequirement(
          id,
          (legacySites?.length ?? 0) === legacySiteCount,
          'frame.legacySites length must equal legacySiteCount; supply matching declared counts and views'
        );
        return new GPUContinuousSpatialSample({
          id: props.id,
          spatialContext: props.spatialContext,
          frame:
            props.frame.type === 'line'
              ? {
                  type: 'line',
                  positions,
                  legacySites: legacySites as GraphDataView<'float32'> | undefined
                }
              : {
                  type: 'area',
                  positions,
                  bounds: props.frame.bounds,
                  legacySites: legacySites as GraphDataView<'float32x2'> | undefined
                },
          spacing: props.spacing,
          seed: props.seed,
          replicate: props.replicate,
          waves: props.waves,
          capacity: props.capacity,
          maximumCandidates: props.maximumCandidates,
          parameters: props.parameters !== undefined ? getView<'uint32'>('parameters') : undefined,
          output: {
            positions: getView<'float32x2'>('positionsOutput'),
            designWeights: getView<'float32'>('designWeights'),
            provenance: getView<'uint32x4'>('provenance'),
            status: getView<'uint32x4'>('status')
          }
        });
      }
    });
  }
}
