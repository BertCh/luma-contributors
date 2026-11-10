// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {FEATURED_LENS_SCENARIOS} from '../../../../lens/featured';
import type {TopicPageData} from '../../topic-types';
import kernelDensity from './kernel-density.example.ts?raw';

export const pageData: TopicPageData = {
  scenarios: [FEATURED_LENS_SCENARIOS.kernelDensity],
  examples: {kernelDensity}
};
