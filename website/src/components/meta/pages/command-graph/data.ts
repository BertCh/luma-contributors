// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import localMoranChain from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/local-moran.example.ts?raw';
import {FEATURED_LENS_SCENARIOS} from '../../../../lens/featured';
import type {TopicPageData} from '../../topic-types';
import declareKernels from './declare-kernels.example.ts?raw';

/** Recorded runs this page draws. Each hazard kind is shown on a record that has it uncollapsed. */
export const COMMAND_GRAPH_SCENARIOS = {
  timeWindowFilter: 'gpu-dataframe.time-window-filter',
  classBreaks: 'gpu-dataframe.class-breaks',
  hilbertKeys: 'gpu-spatial-analysis.hilbert-keys',
  localMoran: FEATURED_LENS_SCENARIOS.localMoran
} as const;

export const pageData: TopicPageData = {
  scenarios: Object.values(COMMAND_GRAPH_SCENARIOS),
  examples: {declareKernels, localMoranChain}
};
