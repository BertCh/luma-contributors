// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUClassBreaks} from '../../dataframe-analysis/column-classification/gpu-class-breaks';
import {
  CLASS_BREAKS_STATE_LENGTH,
  getClassBreaksMomentTileCount,
  MOMENT_STRIDE,
  NATURAL_BREAKS_CELL_COUNT,
  NATURAL_BREAKS_ELEMENT_STRIDE
} from '../../dataframe-analysis/column-classification/class-breaks-kernels';
import {getNaturalBreaksDispatchWordCount} from '../../dataframe-analysis/column-classification/natural-breaks-nodes';
import {validateClassificationScratch} from '../../dataframe-analysis/column-classification/column-classification-rows';

/** GPUClassBreaks stores four control words before its head-tail break values. */
const MARKOV_HEAD_TAIL_CONTROL_WORDS = 4;
/** Published auto sort uses local bitonic through 256 rows, then 16-bucket radix histograms. */
const MARKOV_SORT_WORKGROUP_SIZE = 256;
const MARKOV_SORT_BUCKET_COUNT = 16;

/** Pooled lag classification scratch, checked before the parent declares its period buffers.
 * Mirrors GPUClassBreaks' approximate-mode layout using that instance's resolved topology.
 * Spatial Markov does not expose exact/monotone natural-breaks modes. @internal */
export function validateMarkovClassBreakScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  classifier: GPUClassBreaks
): void {
  const {methods, quantileCount, naturalBreaksBinCount, props} = classifier;
  const count = props.values.length;
  const classCount = props.maximumClassCount;
  const quantiles = methods.includes('quantile') || methods.includes('box-plot');
  const moments = methods.includes('standard-deviation') || methods.includes('head-tail');
  const natural = methods.includes('natural-breaks');
  validateClassificationScratch(graph, id, {
    'lag classifier state': CLASS_BREAKS_STATE_LENGTH,
    'lag classifier edges': classCount + 1,
    'lag classifier extremes': 4,
    'lag classifier quantile histogram': quantiles ? 2 * quantileCount * 256 : 0,
    'lag classifier quantile state': quantiles ? 8 * quantileCount + 2 : 0,
    'lag classifier quantile parameters': quantiles ? 4 + quantileCount : 0,
    'lag classifier moments': moments ? 2 * MOMENT_STRIDE : 0,
    'lag classifier moment partials': moments
      ? getClassBreaksMomentTileCount(count) * MOMENT_STRIDE
      : 0,
    'lag classifier head-tail state': methods.includes('head-tail')
      ? MARKOV_HEAD_TAIL_CONTROL_WORDS + Math.max(classCount - 1, 1)
      : 0,
    'lag classifier head-tail dispatch': methods.includes('head-tail') ? 6 : 0,
    'lag classifier natural histogram': natural ? 5 * (naturalBreaksBinCount + 1) : 0,
    'lag classifier natural DP': natural ? classCount * naturalBreaksBinCount : 0,
    'lag classifier natural refinement': natural
      ? Math.max(classCount - 1, 1) *
        (NATURAL_BREAKS_CELL_COUNT + 2) *
        NATURAL_BREAKS_ELEMENT_STRIDE
      : 0,
    'lag classifier natural dispatch': natural ? getNaturalBreaksDispatchWordCount(1) : 0,
    'lag classifier sort histogram':
      methods.includes('maximum-breaks') && count > MARKOV_SORT_WORKGROUP_SIZE
        ? Math.ceil(count / MARKOV_SORT_WORKGROUP_SIZE) * MARKOV_SORT_BUCKET_COUNT
        : 0
  });
}
