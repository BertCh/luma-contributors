// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getSummaryReductionPrefixCount} from './summary-reduction-prefix';

it('B2 submitted prefix reaches the scheduled summary merge and leaves publication pending', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const preparation = new GPUStudyWindowPreparation({
    positions: createTransientView(graph, 'positions', 'float32x2', 2052),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 514),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 514),
    samplingGrid: [2, 2],
    output: {summary: createTransientView(graph, 'output-summary', 'float32', 8)}
  });
  const nodes = preparation.getCommandNodes(graph);
  // Preserve all real resource hazards and workload metadata, but compile inert callbacks:
  // this regression inspects the public host schedule and never submits GPU work.
  for (const node of nodes) graph.add({...node, compile: () => ({encode: () => {}})});
  const compiled = graph.compile();
  try {
    const order = compiled.stats.nodeOrder;
    const prefixCount = getSummaryReductionPrefixCount(order);
    expect(order[prefixCount - 1]).toBe('study-window-summary-reduce-4');
    const completed = order.slice(0, prefixCount);
    expect(completed).toContain('study-window-summary-partials');
    expect(completed).toContain('study-window-summary-reduce-2');
    expect(completed).not.toContain('study-window-summarize');
    expect(completed).not.toContain('study-window-snapshot-commit');
  } finally {
    compiled.destroy();
    preparation.destroy();
    device.destroy();
  }
});
