// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma-contributors/analysis/computation-record';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';

export type SpatialReplayTag = {keySpace: string; meaning: string; infoCapture: string};

/** Publishes recording-only replay arrays separately from ordinary GPU readbacks. */
export function createSpatialReplayCaptures(
  captures: readonly GPUComputationCapture[],
  tags: Readonly<Record<string, SpatialReplayTag>>
): GPUComputationCapture[] {
  return captures
    .filter(capture => Object.hasOwn(tags, capture.name))
    .flatMap(capture => {
      const tag = tags[capture.name];
      const count = captures.find(candidate => candidate.name === tag.infoCapture)?.values[1];
      validateRequirement(
        'spatial-replay-capture',
        count !== undefined &&
          Number.isInteger(count) &&
          count >= 0 &&
          count * (capture.components ?? 1) <= capture.values.length,
        `${capture.name} replay count must be an integer within its captured capacity`
      );
      if (!count) return [];
      return [
        {
          ...capture,
          keySpace: tag.keySpace,
          meaning: tag.meaning,
          values: Array.prototype.slice.call(capture.values, 0, count * (capture.components ?? 1)),
          role: 'intermediate' as const,
          provenance: 'gpu-replay' as const
        }
      ];
    });
}
