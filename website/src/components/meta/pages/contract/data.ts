// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TopicPageData} from '../../topic-types';
// The code that produced the recorded network-service-areas run; its `#region example` is shown.
import serviceAreas from '../../../../../../modules/lenses/src/computation-lenses/scenarios/gpu-network/network-service-areas.example.ts?raw';

export const pageData: TopicPageData = {
  scenarios: [],
  examples: {serviceAreas}
};
