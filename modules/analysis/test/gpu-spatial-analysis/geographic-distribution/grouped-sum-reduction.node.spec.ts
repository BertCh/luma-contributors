// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  getGroupedSumNodes,
  GROUPED_SUM_MAXIMUM_GROUPS,
  type GroupedSumProps
} from '../../../src/gpu-spatial-analysis/geographic-distribution/grouped-sum-reduction';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function getNodes(overrides: Partial<GroupedSumProps>, outputCount = 1) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  try {
    return getGroupedSumNodes(graph, {
      id: 'grouped-sum',
      operation: 'grouped-sum',
      rows: 8,
      groupCount: 2,
      bindings: [],
      rowSnippet: 'group = 0u;',
      outputs: Array.from({length: outputCount}, () => undefined as never),
      ...overrides
    } as GroupedSumProps);
  } finally {
    graph.device.destroy();
  }
}

describe('getGroupedSumNodes', () => {
  it('rejects a group count outside the chunked limit', () => {
    expect(() => getNodes({groupCount: 0})).toThrow(
      `grouped-sum groupCount must be an integer in [1, ${GROUPED_SUM_MAXIMUM_GROUPS}]`
    );
    expect(() => getNodes({groupCount: GROUPED_SUM_MAXIMUM_GROUPS + 1})).toThrow(
      /^grouped-sum groupCount must be an integer in \[1, 256\]/
    );
  });

  it('rejects more outputs than the shared-memory tile holds', () => {
    expect(() => getNodes({}, 5)).toThrow('grouped-sum output count must be an integer in [1, 4]');
    expect(() => getNodes({}, 0)).toThrow('grouped-sum output count must be an integer in [1, 4]');
  });
});
