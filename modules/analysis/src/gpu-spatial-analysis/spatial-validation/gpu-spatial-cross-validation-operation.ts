// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUContributorOperation} from '../../analysis-core/operations/gpu-contributor-operation';
import {
  GPUSpatialCrossValidation,
  type GPUSpatialCrossValidationProps
} from './gpu-spatial-cross-validation';
import type {GPUSpatialValidationDesign} from './spatial-validation-design';

type OperationDesign<Design = GPUSpatialValidationDesign> =
  Design extends GPUSpatialValidationDesign
    ? Omit<Design, 'features' | 'predictionPositions' | 'candidates' | 'diagnostics'> & {
        diagnostics?: string;
      } & (Design extends {mode: 'environmental'} ? {features: string} : {}) &
        (Design extends {mode: 'nndm' | 'knndm'}
          ? {predictionPositions: string; predictionRowCount: number}
          : {}) &
        (Design extends {mode: 'knndm'} ? {candidates?: string} : {})
    : never;

/** Graph-value names replace views, with explicit row counts for shape inference. */
export type SpatialCrossValidationOperationProps = Pick<
  GPUSpatialCrossValidationProps,
  'id' | 'spatialContext' | 'foldCount' | 'seed' | 'bufferDistance'
> & {
  rowCount: number;
  positions: string;
  blocks: string;
  mask?: string;
  design?: OperationDesign;
  output: {folds: string; training: string; testing: string; counts: string; status: string};
};

/** Vector operation form; compile with GPU_CONTRIBUTOR_OPERATION_LOWERING. */
export class GPUSpatialCrossValidationOperation extends GPUContributorOperation {
  constructor(props: SpatialCrossValidationOperationProps) {
    const {design, rowCount, foldCount, output} = props;
    super({
      id: props.id ?? 'spatial-cross-validation',
      inputs: {
        positions: props.positions,
        blocks: props.blocks,
        ...(props.mask !== undefined ? {mask: props.mask} : {}),
        ...(design?.mode === 'environmental' ? {features: design.features} : {}),
        ...(design?.mode === 'nndm' || design?.mode === 'knndm'
          ? {predictionPositions: design.predictionPositions}
          : {}),
        ...(design?.mode === 'knndm' && design.candidates !== undefined
          ? {candidates: design.candidates}
          : {})
      },
      outputs: {
        folds: {value: output.folds, format: 'uint32', length: rowCount},
        training: {value: output.training, format: 'uint32', length: rowCount * foldCount},
        testing: {value: output.testing, format: 'uint32', length: rowCount * foldCount},
        counts: {value: output.counts, format: 'uint32', length: foldCount * 3},
        status: {value: output.status, format: 'uint32', length: foldCount},
        ...(design?.diagnostics !== undefined
          ? {
              diagnostics: {
                value: design.diagnostics,
                format: 'float32' as const,
                length: 4 + (design.mode === 'nndm' ? rowCount : 0)
              }
            }
          : {})
      },
      createContributor: getView => {
        const diagnostics =
          design?.diagnostics !== undefined ? getView<'float32'>('diagnostics') : undefined;
        return new GPUSpatialCrossValidation({
          id: props.id,
          spatialContext: props.spatialContext,
          positions: getView<'float32x2'>('positions'),
          blocks: getView<'uint32'>('blocks'),
          mask: props.mask !== undefined ? getView<'uint32'>('mask') : undefined,
          foldCount,
          seed: props.seed,
          bufferDistance: props.bufferDistance,
          design: !design
            ? undefined
            : design.mode === 'groups'
              ? {...design, diagnostics}
              : design.mode === 'environmental'
                ? {...design, diagnostics, features: getView<'float32'>('features')}
                : design.mode === 'nndm'
                  ? {
                      ...design,
                      diagnostics,
                      predictionPositions: getView<'float32x2'>('predictionPositions')
                    }
                  : {
                      ...design,
                      diagnostics,
                      predictionPositions: getView<'float32x2'>('predictionPositions'),
                      candidates:
                        design.candidates !== undefined
                          ? getView<'uint32'>('candidates')
                          : undefined
                    },
          output: {
            folds: getView<'uint32'>('folds'),
            training: getView<'uint32'>('training'),
            testing: getView<'uint32'>('testing'),
            counts: getView<'uint32'>('counts'),
            status: getView<'uint32'>('status')
          }
        });
      }
    });
  }
}
