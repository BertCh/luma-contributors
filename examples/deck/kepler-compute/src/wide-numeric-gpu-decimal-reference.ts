// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compatibility entry: the implementation belongs to the public analysis package.
export {
  DECIMAL128_MAXIMUM,
  DECIMAL128_MINIMUM,
  splitDecimal128ToWords,
  joinWordsToDecimal128,
  combineDecimal128SumAccumulators,
  summarizeDecimal128Reference,
  formatDecimal128,
  normalizeDecimal128Comparison,
  compareDecimal128Reference
} from '@luma-contributors/analysis/dataframe-analysis';
export type {
  Decimal128Words,
  Decimal128CompareOperator,
  Decimal128Summary,
  NormalizedDecimal128Comparison
} from '@luma-contributors/analysis/dataframe-analysis';
