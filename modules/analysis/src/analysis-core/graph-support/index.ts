// vis.gl analysis
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  MAXIMUM_ARROW_ANALYTICS_SCALAR_ROW_COUNT,
  MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT,
  MAXIMUM_ARROW_ANALYTICS_DICTIONARY_BYTE_LENGTH,
  prepareArrowBatchesForGPUAnalytics,
  prepareArrowKeyPairForGPUAnalytics,
  prepareArrowMeasureColumnsForGPUAnalytics,
  prepareArrowTableForGPUAnalytics,
  type ArrowAnalyticsColumnConversion,
  type ArrowAnalyticsFloat64Policy,
  type ArrowAnalyticsTimeUnit,
  type PrepareArrowTableForGPUAnalyticsOptions,
  type PreparedArrowAnalyticsKeyColumn,
  type PreparedArrowAnalyticsKeyPair,
  type PreparedArrowAnalyticsMeasureColumn,
  type PreparedArrowAnalyticsTable,
  type PreparedArrowAnalyticsTimeColumn
} from './arrow-analytics-normalization';
export {
  getGPUTemporalContextFromPreparedArrowColumn,
  makeGPUTimeVectorFromPreparedArrowColumn,
  type GPUTimeVectorFromPreparedArrowColumn,
  type GPUTimeVectorWithTemporalContext
} from './arrow-analytics-time-upload';
export {
  convertWideNumericValue,
  convertWideNumericVector,
  isWideNumericType
} from './arrow-analytics-wide-numeric';
