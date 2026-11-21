// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {InequalityOperationProps} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality-operation';

/** Keep a definite named terminal in every all/alone/omitted output fixture. */
export function getInequalityOperationTestOutput(
  selected: readonly (keyof InequalityOperationProps['output'])[]
): InequalityOperationProps['output'] {
  const siblings = Object.fromEntries(selected.map(name => [name, name]));
  switch (selected[0]) {
    case 'gini':
      return {...siblings, gini: 'gini'};
    case 'theilT':
      return {...siblings, theilT: 'theilT'};
    case 'theilL':
      return {...siblings, theilL: 'theilL'};
    case 'atkinson':
      return {...siblings, atkinson: 'atkinson'};
    case 'hoover':
      return {...siblings, hoover: 'hoover'};
    case 'palma':
      return {...siblings, palma: 'palma'};
    case 'mean':
      return {...siblings, mean: 'mean'};
    case 'counts':
      return {...siblings, counts: 'counts'};
    case 'lorenzKnots':
      return {...siblings, lorenzKnots: 'lorenzKnots'};
    case 'globalSummary':
      return {...siblings, globalSummary: 'globalSummary'};
    default:
      throw new Error('Expected at least one named inequality terminal');
  }
}
