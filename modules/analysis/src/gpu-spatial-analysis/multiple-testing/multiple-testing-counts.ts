// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Logical count ports shared by adjustment contributors and their operation forms. @internal */
export const MULTIPLE_TESTING_COUNT_OUTPUT_NAMES = [
  'finiteTestCount',
  'effectiveFamilySize'
] as const;

/** Report removed count ports before construction snapshots discard unknown keys. @internal */
export function validateMultipleTestingCountPorts(id: string, output: object): void {
  for (const [name, replacement] of [
    ['observedTestCount', 'finiteTestCount'],
    ['familySize', 'effectiveFamilySize'],
    ['testedCount', 'effectiveFamilySize']
  ]) {
    validateRequirement(
      id,
      !(name in output),
      `output.${name} was removed. Use output.${replacement}`
    );
  }
}
