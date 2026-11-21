// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUInequalityOutput} from './gpu-inequality';

const INEQUALITY_OPERATION_OUTPUT_NAMES = [
  'gini',
  'theilT',
  'theilL',
  'atkinson',
  'hoover',
  'palma',
  'mean',
  'counts',
  'lorenzKnots',
  'globalSummary'
] as const satisfies readonly (keyof GPUInequalityOutput)[];
const SUPPORTED_INEQUALITY_OPERATION_OUTPUT_NAMES = new Set<string>(
  INEQUALITY_OPERATION_OUTPUT_NAMES
);

/** Reject undeclared logical ports before CPU/GPU forms declare their destinations. @internal */
export function validateInequalityOperationOutputNames(
  id: string,
  output: Readonly<Record<string, unknown>>
): void {
  for (const name of Object.keys(output)) {
    validateRequirement(
      id,
      name !== 'weightedGini',
      'output.weightedGini is a host result alias; select output.gini for an operation instead'
    );
    validateRequirement(
      id,
      SUPPORTED_INEQUALITY_OPERATION_OUTPUT_NAMES.has(name),
      `output.${name} must be a supported output port (${INEQUALITY_OPERATION_OUTPUT_NAMES.join(', ')}); received ${name}`
    );
  }
}
