// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeContiguityWeightsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {computeSpatialWeightsAlgebraOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {computeSpatialWeightsTracesOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';

const topology = {
  // Two triangles share exactly the edge (0,0)-(1,0), hence W=[[0,1],[1,0]], W²=I.
  positions: Float32Array.of(0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, -1),
  ringOffsets: Uint32Array.of(0, 3, 6),
  polygonOffsets: Uint32Array.of(0, 1, 2),
  featureOffsets: Uint32Array.of(0, 1, 2)
};
it('STA-01 composes intact CPU results and honors each named loss cause', () => {
  for (const pairCapacity of [8, 1, 8]) {
    const weights = computeContiguityWeightsOnCPU(topology, {
      criterion: 'rook',
      capacity: 2,
      pairCapacity
    });
    const loss = Number(pairCapacity === 1);
    expect(weights.status.candidateOverflow[0]).toBe(loss);
    expect(weights).not.toHaveProperty('candidateOverflow');
    const first = computeSpatialWeightsAlgebraOnCPU(
      {weights},
      {operation: 'selfWeight', selfWeight: 0, capacity: 6}
    );
    const second = computeSpatialWeightsAlgebraOnCPU(
      {weights: first},
      {operation: 'selfWeight', selfWeight: 0, capacity: 6}
    );
    expect(first.status.candidateOverflow[0]).toBe(loss);
    expect(second.status.candidateOverflow[0]).toBe(loss);
    if (loss)
      expect(() =>
        computeSpatialWeightsTracesOnCPU({weights}, {mode: 'exact', maximumPower: 2})
      ).toThrow(/incomplete/);
    else
      expect(
        Array.from(
          computeSpatialWeightsTracesOnCPU({weights}, {mode: 'exact', maximumPower: 2}).powerTraces
        )
      ).toEqual([0, 2]);
  }
  const clean = computeContiguityWeightsOnCPU(topology, {criterion: 'rook', capacity: 2});
  for (const status of [
    {overflow: Uint32Array.of(1)},
    {candidateOverflow: Uint32Array.of(1)},
    {complete: Uint32Array.of(0)},
    {requiredCount: Uint32Array.of(3)}
  ]) {
    const weights = {...clean, status: {...clean.status, ...status}};
    expect(
      computeSpatialWeightsAlgebraOnCPU(
        {weights},
        {operation: 'selfWeight', selfWeight: 0, capacity: 6}
      ).status.candidateOverflow[0]
    ).toBe(1);
    expect(() => computeSpatialWeightsTracesOnCPU({weights}, {maximumPower: 2})).toThrow(
      /incomplete/
    );
  }
  for (const field of ['candidateOverflow', 'weightsCandidateOverflow'] as const) {
    // Old JavaScript inputs must fail closed, even with clear named status.
    const weights = {...clean, [field]: Uint32Array.of(1)};
    expect(() =>
      computeSpatialWeightsAlgebraOnCPU(
        {weights},
        {operation: 'selfWeight', selfWeight: 0, capacity: 6}
      )
    ).toThrow(
      new RegExp(`weights\\.${field} was removed; use weights\\.status\\.candidateOverflow`)
    );
  }
});

import {getWeightsLossSources} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-loss';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
it('CPL-03 refuses negative-only reads and separately collects positive certificates', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const complete = createTransientView(graph, 'complete', 'uint32', 1);
    const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
    const candidateOverflow = createTransientView(graph, 'candidate', 'uint32', 1);
    expect(() => getWeightsLossSources([{status: {overflow, complete}} as never])).toThrow(
      /candidateOverflow/
    );
    expect(() =>
      getWeightsLossSources([{status: {overflow, complete, candidateOverflow}}])
    ).toThrow(/completeness-aware/);
    expect(
      getWeightsLossSources(
        [{status: {overflow, complete, candidateOverflow}} as never],
        'aware',
        true
      )
    ).toEqual({overflow: [overflow], candidateOverflow: [candidateOverflow], complete: [complete]});
  } finally {
    device.destroy();
  }
});

it('NAM-09 computes symmetric difference and rejects the historical spelling', () => {
  const rows = {
    offsets: Uint32Array.of(0, 1, 2),
    neighbors: Uint32Array.of(1, 0),
    weights: Float32Array.of(1, 1)
  };
  const options = {operation: 'symmetric-difference', capacity: 2} as const;
  const result = computeSpatialWeightsAlgebraOnCPU({left: rows, right: rows}, options);
  // A xor A is empty, independent of either weights implementation.
  expect([...result.offsets]).toEqual([0, 0, 0]);
  // The right-only edge (1,0) survives xor but cannot survive ordinary left-minus-right difference.
  const left = {
    offsets: Uint32Array.of(0, 1, 1),
    neighbors: Uint32Array.of(1),
    weights: Float32Array.of(1)
  };
  const unequal = computeSpatialWeightsAlgebraOnCPU({left, right: rows}, options);
  expect([...unequal.offsets]).toEqual([0, 0, 1]);
  expect([...unequal.neighbors.slice(0, 1)]).toEqual([0]);
  expect([...unequal.weights.slice(0, 1)]).toEqual([1]);
  for (const operation of ['Union', 'symmetricDifference']) {
    expect(() =>
      computeSpatialWeightsAlgebraOnCPU({left: rows, right: rows}, {...options, operation} as never)
    ).toThrow(/operation.*union.*intersection.*symmetric-difference/);
  }
});
