// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateOption} from '@luma-contributors/analysis/analysis-core';
import type {GPUSimilarLocationsStandardization} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {Standardization} from './similar-places.stats';

/** Resolve the canonical standardization to the CPU profile key. */
export function resolveProfileStandardization(
  value: GPUSimilarLocationsStandardization
): Standardization {
  validateOption('similar-places', 'standardization', value, ['z-score', 'rank']);
  return value === 'z-score' ? 'zscore' : value;
}
