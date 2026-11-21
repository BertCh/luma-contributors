// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {computeSpatialWeightsAlgebraOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {GPUChunkedSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-algebra';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {AUDIT_ALGEBRA_CASES} from './audit-weights-a-algebra-fixture';

for (const fixture of AUDIT_ALGEBRA_CASES) {
  it(`audits CPU ${fixture.name} against literal dense-set/shortest-path/Kronecker values`, () => {
    for (const capacity of [2, 64]) {
      const actual = computeSpatialWeightsAlgebraOnCPU(fixture.input, {
        ...fixture.options,
        capacity
      });
      const used = Math.min(capacity, fixture.neighbors.length);
      expect(Array.from(actual.offsets)).toEqual(
        fixture.offsets.map(value => Math.min(capacity, value))
      );
      expect(Array.from(actual.neighbors.slice(0, used))).toEqual(fixture.neighbors.slice(0, used));
      expect(Array.from(actual.weights.slice(0, used))).toEqual(fixture.weights.slice(0, used));
      if (fixture.distances)
        expect(Array.from(actual.distances!.slice(0, used))).toEqual(
          fixture.distances.slice(0, used)
        );
      expect(Array.from(actual.status.requiredCount)).toEqual([fixture.neighbors.length]);
      expect(Array.from(actual.status.overflow)).toEqual([
        Number(capacity < fixture.neighbors.length)
      ]);
      expect(Array.from(actual.status.candidateOverflow)).toEqual([0]);
    }
  });
}

it('preserves possible diagonal slots in whole and chunked binary algebra metadata', () => {
  // Smallest witness: a one-row left CSR with one self slot and an empty right row.
  // L union empty, L minus empty, and L xor empty all retain that self slot.
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `audit-${serial++}`, format, length);
  const weights = (hasSelfWeights: boolean) => ({
    offsets: view('uint32', 2),
    neighbors: view('uint32', 1),
    weights: view('float32', 1),
    information: {rowCount: 1, transform: 'raw' as const, symmetric: true, hasSelfWeights}
  });
  const output = () => ({...weights(false), status: {overflow: view('uint32', 1)}});
  try {
    for (const operation of [
      'union',
      'intersection',
      'difference',
      'symmetric-difference'
    ] as const) {
      const left = weights(true);
      const right = weights(operation === 'intersection');
      const whole = new GPUSpatialWeightsAlgebra({operation, left, right, output: output()});
      expect(whole.outputWeights.information?.hasSelfWeights, `whole ${operation}`).toBe(true);
      const chunked = new GPUChunkedSpatialWeightsAlgebra({
        operation,
        left: {
          rowCount: 1,
          information: left.information,
          chunks: [{rowStart: 0, rowCount: 1, weights: left}]
        },
        right: {
          rowCount: 1,
          information: right.information,
          chunks: [{rowStart: 0, rowCount: 1, weights: right}]
        },
        output: {chunks: [{rowStart: 0, rowCount: 1, weights: output()}]}
      });
      expect(chunked.outputWeights.information?.hasSelfWeights, `chunked ${operation}`).toBe(true);
      expect(chunked.outputWeights.chunks[0].weights.information?.hasSelfWeights).toBe(true);
    }
  } finally {
    device.destroy();
  }
});
