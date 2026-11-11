// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
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

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUContinuousSpatialSampleOperation extends GPUContributorOperation {
  constructor(props: ContinuousSpatialSampleOperationProps) {
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
      createContributor: getView =>
        new GPUContinuousSpatialSample({
          id: props.id,
          spatialContext: props.spatialContext,
          frame:
            props.frame.type === 'line'
              ? {
                  type: 'line',
                  positions: getView<'float32x2'>('positions'),
                  legacySites:
                    props.frame.legacySites !== undefined
                      ? getView<'float32'>('legacySites')
                      : undefined
                }
              : {
                  type: 'area',
                  positions: getView<'float32x2'>('positions'),
                  bounds: props.frame.bounds,
                  legacySites:
                    props.frame.legacySites !== undefined
                      ? getView<'float32x2'>('legacySites')
                      : undefined
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
        })
    });
  }
}
