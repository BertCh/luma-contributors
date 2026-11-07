// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {InterpolationBenchmarkScope} from './interpolation-performance-harness';
import {calculateConditionalSimulationReference} from './conditional-simulation-reference';
import {requireTestDevice} from '../../utils/require-test-device';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const [observationCount, queryCount, reusePreparation] of [
  [129, 129, false],
  [129, 129, true],
  [128, 384, true]
] as const) {
  it(`blocked simulation ${observationCount}+${queryCount} retained=${reusePreparation} preserves the full conditional covariance`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scope = new InterpolationBenchmarkScope(device, 'blocked-simulation-covariance');
    const capacity = observationCount + queryCount;
    const positions = Float32Array.from({length: observationCount * 2}, (_, component) =>
      component % 2 === 0 ? (Math.floor(component / 2) % 16) / 4 : Math.floor(component / 32) / 4
    );
    const observations = Float32Array.from({length: observationCount * 4}, (_, component) =>
      component % 4 === 0
        ? 1 + Math.sin(Math.floor(component / 4))
        : component % 4 === 1
          ? 0.1
          : component % 4 === 3
            ? 1
            : 0
    );
    observations[1] = 0;
    const queries = Float32Array.from({length: queryCount * 2}, (_, component) =>
      component % 2 === 0
        ? ((Math.floor(component / 2) % 16) + 0.3) / 4
        : (Math.floor(component / 32) + 0.2) / 4
    );
    // A noiseless observation hit and duplicate queries exercise zero conditional pivots.
    queries[0] = positions[0];
    queries[1] = positions[1];
    queries[32] = queries[30];
    queries[33] = queries[31];
    const covariance = {model: 'exponential' as const, range: 1.3, sill: 1};
    const contributor = new GPUConditionalGaussianSimulation({
      positions: scope.input('positions', positions, 'float32x2'),
      observations: scope.input('observations', observations, 'float32x4'),
      queries: scope.input('queries', queries, 'float32x2'),
      covariance,
      mean: 0.7,
      seed: 1307,
      reusePreparation,
      replicateCount: 19,
      output: {
        values: scope.output('values', queryCount * 19, 'float32'),
        moments: scope.output('moments', queryCount, 'float32x2'),
        status: scope.output('status', 19, 'uint32')
      }
    });
    const nodes = contributor.getCommandNodes(scope.graph);
    scope.graph.add(nodes);
    let factor: GraphDataView | undefined;
    for (const resource of nodes[0].resources ?? []) {
      if (
        'buffer' in resource &&
        resource.buffer instanceof GraphDataView &&
        resource.buffer.buffer.id === `${contributor.id}-factor`
      )
        factor = resource.buffer;
    }
    if (!factor) throw new Error('Missing conditional factor');
    const captured = scope.output('factor', 2 + capacity * capacity, 'float32');
    scope.graph.add(
      createWGSLKernelNode(scope.graph, {
        id: 'capture-blocked-factor',
        operation: 'SimulationFactorDiagnostic',
        variant: 'copy',
        invocationCount: captured.length,
        bindings: [
          {name: 'source', view: factor, type: 'f32', access: 'read'},
          {name: 'output', view: captured, type: 'f32', access: 'read_write'}
        ],
        body: 'output[outputOffset + index] = source[sourceOffset + index];'
      })
    );
    const compiled = scope.graph.compile();
    try {
      for (const write of scope.initialWrites) write.buffer.write(write.values);
      // Changing the active observation count moves the query block across diagonal tile edges.
      for (const activeObservationCount of [observationCount, 127, 0]) {
        for (let row = 0; row < observationCount; row++)
          observations[row * 4 + 3] = Number(row < activeObservationCount);
        scope.initialWrites[1].buffer.write(observations);
        if (reusePreparation) contributor.invalidateObservations();
        submitGraph(device, compiled, undefined);
        const actual = await scope.read();
        expect(actual['status']).toEqual(Array(19).fill(0));
        expect(actual['values'].every(Number.isFinite)).toBe(true);
        const reference = calculateConditionalSimulationReference(
          positions,
          observations,
          queries,
          covariance,
          0.7
        );
        expect(actual['factor'][1]).toBe(activeObservationCount);
        let maximumMeanError = 0;
        let maximumCovarianceError = 0;
        for (let row = 0; row < queryCount; row++) {
          maximumMeanError = Math.max(
            maximumMeanError,
            Math.abs(actual['moments'][row * 2] - reference.means[row])
          );
          expect(actual['moments'][row * 2 + 1]).toBeCloseTo(reference.covariance[row][row], 4);
          for (let column = 0; column <= row; column++) {
            let reconstructed = 0;
            for (let component = 0; component <= column; component++) {
              reconstructed +=
                actual['factor'][
                  2 + (activeObservationCount + row) * capacity + activeObservationCount + component
                ] *
                actual['factor'][
                  2 +
                    (activeObservationCount + column) * capacity +
                    activeObservationCount +
                    component
                ];
            }
            maximumCovarianceError = Math.max(
              maximumCovarianceError,
              Math.abs(reconstructed - reference.covariance[row][column])
            );
          }
        }
        expect(maximumMeanError).toBeLessThan(1e-4);
        expect(maximumCovarianceError).toBeLessThan(1e-4);
        if (activeObservationCount > 0)
          for (let replicate = 0; replicate < 19; replicate++)
            expect(actual['values'][replicate * queryCount]).toBe(observations[0]);
        for (let replicate = 0; replicate < 19; replicate++)
          expect(actual['values'][replicate * queryCount + 16]).toBe(
            actual['values'][replicate * queryCount + 15]
          );
      }
    } finally {
      compiled.destroy();
      contributor.destroy();
      scope.destroy();
    }
  });
}
