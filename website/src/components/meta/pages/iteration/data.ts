// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The `// #region example` block of this lens scenario is the code that produced the recorded run.
import reachabilityScenario from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-network/network-reachability.example.ts?raw';
import type {TopicPageData} from '../../topic-types';

export const pageData: TopicPageData = {
  // The figures draw from reachability-search.json and gated-records.json (generate-data.ts).
  scenarios: [],
  examples: {reachability: reachabilityScenario}
};
