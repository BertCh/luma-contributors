// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TopicPageData} from '../../topic-types';

// Cell aggregation's kernels come from building-blocks.json: its view model folds them into groups.
export const SCENARIOS = {
  timeWindow: 'gpu-dataframe.time-window-filter',
  kernelDensity: 'gpu-spatial-analysis.kernel-density'
} as const;

export const pageData: TopicPageData = {
  scenarios: Object.values(SCENARIOS),
  examples: {}
};
