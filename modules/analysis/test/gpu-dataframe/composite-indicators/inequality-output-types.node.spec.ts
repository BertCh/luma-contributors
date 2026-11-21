// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUInequality,
  type GPUInequalityOutput
} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('requires an actual inequality terminal and accepts each minimal output selection', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 2);
  const zoneIds = createTransientView(graph, 'zoneIds', 'uint32', 2);
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const terminal = createTransientView(graph, 'terminal', 'float32', 11);
  const counts = createTransientView(graph, 'counts', 'uint32', 1);
  try {
    const selections: GPUInequalityOutput[] = [
      {gini: terminal},
      {theilT: terminal},
      {theilL: terminal},
      {atkinson: terminal},
      {hoover: terminal},
      {palma: terminal},
      {mean: terminal},
      {counts},
      {lorenzKnots: terminal},
      {globalSummary: terminal}
    ];
    for (const output of selections) {
      const contributor = new GPUInequality({
        id: `inequality-${Object.keys(output)[0]}`,
        values,
        zoneIds,
        parameters,
        zoneCount: 1,
        output
      });
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
    }
    // @ts-expect-error An empty top-level selection has no terminal output.
    const empty: GPUInequalityOutput = {};
    // @ts-expect-error An optional undefined view does not select an output.
    const undefinedTerminal: GPUInequalityOutput = {gini: undefined};
    // @ts-expect-error An empty nested object is not a terminal view.
    const nested: GPUInequalityOutput = {globalSummary: {}};
    expect(
      () => new GPUInequality({values, zoneIds, parameters, zoneCount: 1, output: empty})
    ).toThrow(/at least one output/);
    expect([undefinedTerminal, nested]).toHaveLength(2);
  } finally {
    device.destroy();
  }
});
