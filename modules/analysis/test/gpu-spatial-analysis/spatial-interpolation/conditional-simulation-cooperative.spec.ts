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

for (const [model, maximum] of [
  ['exponential', false],
  ['spherical', false],
  ['gaussian', false],
  ['exponential', true]
] as const) {
  for (const reusePreparation of [false, true]) {
    it(`cooperative ${model} simulation retained=${reusePreparation} ${maximum ? 'maximum shape' : 'replay'} preserves full conditional covariance`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scope = new InterpolationBenchmarkScope(device, `simulation-${model}-${maximum}`);
      const observedCapacity = maximum ? 128 : 4;
      const queryCount = maximum ? 128 : 5;
      const capacity = observedCapacity + queryCount;
      const positions = maximum
        ? Float32Array.from({length: observedCapacity * 2}, (_, index) =>
            index % 2 === 0 ? (Math.floor(index / 2) % 16) / 16 : Math.floor(index / 32) / 8
          )
        : new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
      const queries = maximum
        ? Float32Array.from({length: queryCount * 2}, (_, index) =>
            index % 2 === 0
              ? ((Math.floor(index / 2) % 16) + 0.3) / 16
              : (Math.floor(index / 32) + 0.2) / 8
          )
        : new Float32Array([0, 0, 0.2, 0.4, 0.2, 0.4, 0.8, 0.7, 2, 0]);
      const observations = Float32Array.from({length: observedCapacity * 4}, (_, index) => {
        switch (index % 4) {
          case 0:
            return 1 + Math.sin(Math.floor(index / 4));
          case 1:
            return maximum || index > 1 ? 0.1 : 0;
          case 3:
            return 1;
          default:
            return 0;
        }
      });
      const covariance = {model, range: 1.3, sill: 1};
      const contributor = new GPUConditionalGaussianSimulation({
        positions: scope.input('positions', positions, 'float32x2'),
        observations: scope.input('observations', observations, 'float32x4'),
        queries: scope.input('queries', queries, 'float32x2'),
        covariance,
        mean: 0.7,
        seed: 1307,
        reusePreparation,
        replicateCount: 8,
        output: {
          values: scope.output('values', queryCount * 8, 'float32'),
          moments: scope.output('moments', queryCount, 'float32x2'),
          status: scope.output('status', 8, 'uint32')
        }
      });
      const nodes = contributor.getCommandNodes(scope.graph);
      scope.graph.add(nodes);
      // Capture internal factors only in this diagnostic graph, never in measured production runs.
      let factorView: GraphDataView | undefined;
      for (const resource of nodes[0].resources ?? []) {
        if (
          'buffer' in resource &&
          resource.buffer instanceof GraphDataView &&
          resource.buffer.buffer.id === `${contributor.id}-factor`
        )
          factorView = resource.buffer;
      }
      if (!factorView) throw new Error('Missing simulation factor in diagnostic graph');
      const capture = scope.output('factor', 2 + capacity * capacity, 'float32');
      scope.graph.add(
        createWGSLKernelNode(scope.graph, {
          id: 'capture-factor',
          operation: 'SimulationFactorDiagnostic',
          variant: 'copy',
          invocationCount: capture.length,
          bindings: [
            {name: 'source', view: factorView, type: 'f32', access: 'read'},
            {name: 'output', view: capture, type: 'f32', access: 'read_write'}
          ],
          body: 'output[outputOffset + index] = source[sourceOffset + index];'
        })
      );
      const compiled = scope.graph.compile();
      let initialDraws: number[] | undefined;
      try {
        const scenarios = maximum
          ? ['initial', 'changed', 'initial']
          : [
              'initial',
              'invalid-query',
              'invalid-observation',
              'singular',
              'below-pivot',
              'above-pivot',
              'masked',
              'none',
              'changed',
              'initial'
            ];
        for (const scenario of scenarios) {
          const currentPositions = positions.slice();
          const currentQueries = queries.slice();
          const currentObservations = observations.slice();
          if (scenario === 'invalid-query') currentQueries[2] = NaN;
          if (scenario === 'invalid-observation') currentObservations[0] = NaN;
          if (scenario === 'masked') {
            currentObservations[0] = NaN;
            currentObservations[3] = 0;
          }
          if (scenario === 'none')
            for (let row = 0; row < observedCapacity; row++) currentObservations[row * 4 + 3] = 0;
          if (scenario === 'changed')
            for (let row = 0; row < observedCapacity; row++) currentObservations[row * 4] += 0.5;
          if (['singular', 'below-pivot', 'above-pivot'].includes(scenario)) {
            currentPositions[2] = currentPositions[0];
            currentPositions[3] = currentPositions[1];
            currentObservations[1] = currentObservations[5] =
              scenario === 'above-pivot' ? 1e-4 : scenario === 'below-pivot' ? 5e-8 : 0;
          }
          scope.initialWrites[0].buffer.write(currentPositions);
          scope.initialWrites[1].buffer.write(currentObservations);
          scope.initialWrites[2].buffer.write(currentQueries);
          if (reusePreparation) contributor.invalidatePreparation();
          submitGraph(device, compiled, undefined);
          const actual = await scope.read();
          const status = scenario.startsWith('invalid')
            ? 1
            : ['singular', 'below-pivot'].includes(scenario)
              ? 2
              : 0;
          expect(actual['status'], scenario).toEqual(Array(8).fill(status));
          if (status !== 0) {
            expect(actual['moments'].every(Number.isNaN)).toBe(true);
            expect(actual['values'].every(Number.isNaN)).toBe(true);
            continue;
          }
          expect(actual['values'].every(Number.isFinite)).toBe(true);
          expect(actual['moments'].every(Number.isFinite)).toBe(true);
          // This near-threshold case tests the failure classification, not an ill-conditioned accuracy bound.
          if (scenario === 'above-pivot') continue;
          const reference = calculateConditionalSimulationReference(
            currentPositions,
            currentObservations,
            currentQueries,
            covariance,
            0.7
          );
          const observedCount = actual['factor'][1];
          for (let row = 0; row < queryCount; row++) {
            expect(actual['moments'][row * 2], `${scenario} mean ${row}`).toBeCloseTo(
              reference.means[row],
              4
            );
            expect(actual['moments'][row * 2 + 1], `${scenario} variance ${row}`).toBeCloseTo(
              reference.covariance[row][row],
              4
            );
            for (let column = 0; column < queryCount; column++) {
              let value = 0;
              for (let component = 0; component <= Math.min(row, column); component++) {
                value +=
                  actual['factor'][
                    2 + (observedCount + row) * capacity + observedCount + component
                  ] *
                  actual['factor'][
                    2 + (observedCount + column) * capacity + observedCount + component
                  ];
              }
              expect(value, `${scenario} covariance ${row}/${column}`).toBeCloseTo(
                reference.covariance[row][column],
                4
              );
            }
          }
          if (scenario === 'initial') {
            if (initialDraws) expect(actual['values']).toEqual(initialDraws);
            else initialDraws = actual['values'];
            if (!maximum) {
              expect(actual['moments'][1]).toBe(0);
              for (let replicate = 0; replicate < 8; replicate++)
                expect(actual['values'][replicate * queryCount]).toBe(observations[0]);
            }
          }
        }
      } finally {
        compiled.destroy();
        contributor.destroy();
        scope.destroy();
      }
    });
  }
}
