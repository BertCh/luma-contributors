// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUCardinalityPort} from '../../analysis-core/contracts/status';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Exact cardinality of the input population used by a statistic, distinct from output status.
 * `includedCount` is the preferred analytical output. The inherited optional `count` port is
 * retained solely for source compatibility with the first version of the exact-count API.
 * Supply one spelling; they address the same result and never create separate GPU bindings.
 * @internal
 */
export type SpatialStatisticsPopulationOutput = Partial<Pick<GPUCardinalityPort, 'count'>> & {
  /** Exact number of included rows or bins. Do not also supply the legacy `count` alias. */
  includedCount?: GraphDataView<'uint32'>;
};

/** Resolves the compatibility alias once, at contributor construction. @internal */
export function resolveStatisticsPopulationCount(
  id: string,
  output: SpatialStatisticsPopulationOutput
): GraphDataView<'uint32'> | undefined {
  validateRequirement(
    id,
    !output.count || !output.includedCount,
    'output.includedCount and legacy output.count are aliases; provide only one'
  );
  return output.includedCount ?? output.count;
}
