// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeLineSimplificationOnCPU} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';

// GEO-131: keep these caller-constructor guards when kernel planning is split into modules.
for (const maximumIterations of [0, 1025]) {
  it(`coverage rejects iteration budget ${maximumIterations} in the caller constructor`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const positions = createTransientView(graph, 'positions', 'float32x2', 3);
      const ringOffsets = createTransientView(graph, 'rings', 'uint32', 2);
      const polygonOffsets = createTransientView(graph, 'polygons', 'uint32', 2);
      const featureOffsets = createTransientView(graph, 'features', 'uint32', 2);
      const parameters = createTransientView(graph, 'parameters', 'float32', 4);
      expect(
        () =>
          new GPUCoverageSimplification({
            id: 'audit-coverage',
            positions,
            ringOffsets,
            polygonOffsets,
            featureOffsets,
            parameters,
            maximumIterations,
            output: {
              positions: createTransientView(graph, 'result', 'float32x2', 3),
              ringOffsets: createTransientView(graph, 'result-rings', 'uint32', 2),
              status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
            }
          })
      ).toThrow(/^audit-coverage maximumIterations must be an integer in \[1, 1024\]/);
    } finally {
      device.destroy();
    }
  });
}

it('the repeated 72-corner coverage arc still has unresolved spans after one split', () => {
  const ring = Float32Array.from(
    Array.from({length: 72}, (_, row) => {
      const angle = (2 * Math.PI * row) / 72;
      return [Math.cos(angle), Math.sin(angle)];
    }).flat()
  );
  const positions = new Float32Array(146);
  positions.set(ring);
  positions.set(ring.subarray(0, 2), 144);
  const observed: number[] = [];
  for (const maximumIterations of [1, 16]) {
    const result = computeLineSimplificationOnCPU(
      {
        positions,
        lineOffsets: Uint32Array.of(0, 73),
        parameters: getGPULineSimplificationParameterValues({tolerance: 0})
      },
      {maximumIterations, finishSpanLimit: 32}
    );
    observed.push(result.converged[0]);
  }
  // Initial endpoints coincide. The farthest row is opposite (36), leaving rows 1..35
  // and 37..71. Both unfinished spans exceed the 32-row finishing threshold after round 1.
  expect(observed).toEqual([0, 1]);
});

for (const changedBudget of [0, 16]) {
  it(`coverage captures the validated budget before caller changes it to ${changedBudget}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let maximumIterations = 1;
      let budgetReads = 0;
      const props = {
        id: 'audit-coverage',
        positions: createTransientView(graph, 'positions', 'float32x2', 72),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
        polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
        featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
        parameters: createTransientView(graph, 'parameters', 'float32', 4),
        get maximumIterations() {
          budgetReads++;
          return maximumIterations;
        },
        topologyIterationCount: 0,
        output: {
          positions: createTransientView(graph, 'result', 'float32x2', 72),
          ringOffsets: createTransientView(graph, 'result-rings', 'uint32', 2),
          status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
        }
      };
      const contributor = new GPUCoverageSimplification(props);
      expect(budgetReads).toBe(1);
      maximumIterations = changedBudget;
      const nodes = contributor.getCommandNodes(graph);
      expect(budgetReads).toBe(1);
      expect(contributor.props.maximumIterations).toBe(1);
      // One validated round, independent of subsequent caller properties. There is no second
      // split round or delayed refusal under the nested arcs ID.
      const rounds = new Set(
        nodes.flatMap(node => {
          const match = /-arcs-round-(\d+)-/.exec(node.id);
          return match ? [match[1]] : [];
        })
      );
      expect([...rounds]).toEqual(['0']);
    } finally {
      device.destroy();
    }
  });
}
