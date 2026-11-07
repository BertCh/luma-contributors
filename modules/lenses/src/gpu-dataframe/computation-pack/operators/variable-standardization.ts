// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {DataFrameFamilyOperators} from '../types';

/** Named GeoDa variable transformations composed from shared scalar reductions. */
export const VARIABLE_STANDARDIZATION_OPERATORS: DataFrameFamilyOperators = {
  GPUVariableStandardization: {
    family: 'classification',
    descriptor: {scope: 'global', scopeSecondary: 'local', reduction: 'sum'},
    formula: {
      id: 'gpu-dataframe:GPUVariableStandardization',
      derivation: 'curated',
      tex: String.raw`\term{output}{z_i}=\frac{\term{values}{x_i}-\term{mean}{\bar{x}}}{\term{scale}{s}},\quad \term{scale}{s}=\sqrt{\frac{\sum_i(\term{values}{x_i}-\term{mean}{\bar{x}})^2}{\term{count}{n}-1}}`,
      ascii:
        'Default sample z-score: output=(value-mean)/sqrt(sum((value-mean)^2)/(n-1)). MAD uses mean absolute deviation / n; range-adjust=value/range; range-standardize=(value-min)/range; deviation=value-mean. Only finite unmasked rows contribute.',
      symbols: [
        {id: 'output', role: 'output', binding: 'wgsl:output'},
        {id: 'values', role: 'input', binding: 'wgsl:values'},
        {id: 'mean', role: 'intermediate', binding: 'wgsl:mean'},
        {id: 'scale', role: 'intermediate', binding: 'wgsl:scale'},
        {id: 'count', role: 'intermediate', binding: 'wgsl:countValue'}
      ]
    },
    stages: ['filter', 'summarise', 'derive'],
    datumKinds: [{kind: 'row', label: 'Column rows', indexSpace: '1d'}],
    guards: ['count-vs-rate'],
    choices: ['method', 'variance'],
    variables: {
      values: {level: 'ratio', extent: 'intensive'},
      output: {level: 'interval', extent: 'intensive'}
    },
    sourceFiles: [
      'modules/graph/src/gpu-dataframe/variable-standardization/gpu-variable-standardization.ts'
    ]
  }
};
