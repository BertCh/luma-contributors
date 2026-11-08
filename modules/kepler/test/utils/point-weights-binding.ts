// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {KeplerPointWeightsStatus} from '../../src/analysis/point-weights';

/**
 * `addKeplerPointWeights` types its status as the published neighbor-search status, whose
 * `requiredCount` and `invalidCount` ports are optional. Without a caller-owned `output`, the
 * builder always creates all three, so a test that passes the builder result on as a
 * `KeplerPointWeightsBinding` asserts that here instead of casting.
 */
export function requireCompletePointWeightsStatus<
  Producer extends {
    status: {
      overflow: GraphDataView<'uint32'>;
      requiredCount?: GraphDataView<'uint32'>;
      invalidCount?: GraphDataView<'uint32'>;
    };
  }
>(producer: Producer): Omit<Producer, 'status'> & {status: KeplerPointWeightsStatus} {
  const {overflow, requiredCount, invalidCount} = producer.status;
  if (!requiredCount || !invalidCount) {
    throw new Error('Expected the point weights builder to create every status port');
  }
  // Keep the producer object (and its destroyParameters method) and narrow only its status.
  return Object.assign(producer, {status: {overflow, requiredCount, invalidCount}});
}
