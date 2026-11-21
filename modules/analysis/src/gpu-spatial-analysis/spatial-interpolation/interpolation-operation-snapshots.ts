// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotInterpolationConstructionProps} from './interpolation-construction-snapshot';
import type {VecchiaLogLikelihoodOperationProps} from './gpu-vecchia-log-likelihood-operation';
import type {VecchiaFisherScoringOperationProps} from './gpu-vecchia-fisher-scoring-operation';
import type {VecchiaSpaceTimeLogLikelihoodOperationProps} from './gpu-vecchia-space-time-log-likelihood-operation';
import type {ConditionalGaussianSimulationOperationProps} from './gpu-conditional-gaussian-simulation-operation';

const VECCHIA_OPERATION_FIELDS = [
  'id',
  'neighborCount',
  'neighborhoodOffsets',
  'neighborhoods',
  'observations',
  'output',
  'parameters',
  'positions',
  'rowCount',
  'spatialContext'
] as const;

/** Complete construction capture shared by the spatial GPU operation and its CPU twin. @internal */
export function snapshotVecchiaLogLikelihoodOperationProps(
  props: VecchiaLogLikelihoodOperationProps
): VecchiaLogLikelihoodOperationProps {
  return snapshotInterpolationConstructionProps(props, VECCHIA_OPERATION_FIELDS, [
    'conditionals',
    'conditionalStatus',
    'siteTerms',
    'siteStatus',
    'logLikelihood',
    'status'
  ]);
}

/** Complete construction capture shared by the Fisher GPU operation and its CPU twin. @internal */
export function snapshotVecchiaFisherScoringOperationProps(
  props: VecchiaFisherScoringOperationProps
): VecchiaFisherScoringOperationProps {
  return snapshotInterpolationConstructionProps(props, VECCHIA_OPERATION_FIELDS, [
    'siteTerms',
    'siteStatus',
    'totals',
    'status'
  ]);
}

/** Complete construction capture shared by the space-time GPU operation and its CPU twin. @internal */
export function snapshotVecchiaSpaceTimeLogLikelihoodOperationProps(
  props: VecchiaSpaceTimeLogLikelihoodOperationProps
): VecchiaSpaceTimeLogLikelihoodOperationProps {
  return snapshotInterpolationConstructionProps(
    props,
    [...VECCHIA_OPERATION_FIELDS, 'times', 'temporalContext'],
    ['conditionals', 'conditionalStatus', 'siteTerms', 'siteStatus', 'logLikelihood', 'status']
  );
}

/** Complete construction capture shared by the simulation GPU operation and its CPU twin. @internal */
export function snapshotConditionalGaussianSimulationOperationProps(
  props: ConditionalGaussianSimulationOperationProps
): ConditionalGaussianSimulationOperationProps {
  validateRequirement(
    props.id ?? 'conditional-gaussian-simulation',
    !('reusePreparation' in props),
    'reusePreparation was removed from the operation form; use GPUConditionalGaussianSimulation'
  );
  return snapshotInterpolationConstructionProps(
    props,
    [
      'id',
      'spatialContext',
      'covariance',
      'mean',
      'seed',
      'replicateCount',
      'replicateOffset',
      'observationCount',
      'queryCount',
      'positions',
      'observations',
      'queries',
      'output'
    ],
    ['values', 'moments', 'status']
  );
}
