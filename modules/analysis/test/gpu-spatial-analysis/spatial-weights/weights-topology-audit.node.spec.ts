// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computePolygonContiguityOnCPU} from '../../../src/gpu-spatial-analysis/spatial-weights/prepare-polygon-contiguity-topology';
import {getIslandAttachmentFallbackBatchCount} from '../../../src/gpu-spatial-analysis/spatial-weights/island-attachment-fallback';
import {getKeyGroupTransientWordCount} from '../../../src/gpu-spatial-analysis/spatial-weights/key-pair-grouping';
import {getGPUSpatialWeightsCapacityPlan} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';

it('capacity headroom saturates at uint32 and an unrepresentable requirement is explicitly limited', () => {
  for (const requiredCount of [0xffffffff, 0x100000000]) {
    const plan = getGPUSpatialWeightsCapacityPlan({rowCount: 8, capacity: 1, requiredCount});
    expect(plan.capacity).toBe(0xffffffff);
    expect(plan.limited).toBe(requiredCount > 0xffffffff);
    expect(plan.estimatedRequiredCount).toBe(requiredCount);
    const retry = getGPUSpatialWeightsCapacityPlan({
      rowCount: 8,
      capacity: plan.capacity,
      requiredCount
    });
    expect(retry.capacity).toBe(0xffffffff);
    expect(retry.limited).toBe(plan.limited);
  }
});
import {
  getGPUSpatialWeightsSummaryParameterValues,
  getGPUSpatialWeightsTransposeParameterValues,
  getGPUSpatialWeightsTransformParameterValues,
  getGPULatticeWeightsParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights-parameters';
import {
  createWeightsTopologyAuditFixture,
  createWeightsCornerContactAuditFixture,
  CORNER_CONTACT_EXPECTED,
  TOPOLOGY_AUDIT_EXPECTED as EXPECTED
} from './weights-topology-audit-fixture';

it('topology oracle distinguishes queen from rook through a corner-only contact', () => {
  for (const [shift, scale] of [
    [0, 1],
    [100, 2]
  ])
    for (const criterion of ['rook', 'queen'] as const) {
      const topology = createWeightsCornerContactAuditFixture(shift, scale);
      const expected = CORNER_CONTACT_EXPECTED[criterion];
      const result = computePolygonContiguityOnCPU(topology, {criterion, capacity: 4});
      expect(Array.from(result.offsets)).toEqual(expected.offsets);
      expect(Array.from(result.neighbors)).toEqual([
        ...expected.neighbors,
        ...Array(4 - expected.neighbors.length).fill(0)
      ]);
      expect(Array.from(result.weights)).toEqual([
        ...expected.weights,
        ...Array(4 - expected.weights.length).fill(0)
      ]);
      expect(Array.from(result.status.requiredCount)).toEqual([expected.neighbors.length]);
      expect(Array.from(result.status.overflow)).toEqual([0]);
      expect(Array.from(result.status.candidateOverflow)).toEqual([0]);
    }
});

it('exact-source topology preserves every source row and reports every preparation output', () => {
  for (const [shift, scale] of [
    [0, 1],
    [100, 2]
  ]) {
    const topology = createWeightsTopologyAuditFixture(shift, scale);
    // First-seen equality classes, derived by listing the six squares' corners:
    // 0 and 1 share two corners, 10 and 11 share two; the other corners differ.
    expect(Array.from(topology.positions)).toEqual([
      0, 0, 1, 0, 2, 0, 3, 0, 1, 0, 4, 0, 5, 0, 2, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0,
      13, 0, 11, 0, 14, 0, 15, 0, 12, 0, 16, 0, 17, 0, 18, 0, 19, 0
    ]);
    expect(Array.from(topology.ringOffsets)).toEqual([0, 4, 8, 12, 16, 20, 24]);
    expect(Array.from(topology.polygonOffsets)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(Array.from(topology.featureOffsets)).toEqual([0, 1, 2, 3, 3, 3, 3, 4, 6]);
    expect(Array.from(topology.geometryMask)).toEqual([1, 1, 1, 0, 0, 0, 1, 1]);
    expect(topology.invalidRows).toEqual([
      {sourceRowIndex: 4, reason: 'source-invalid'},
      {sourceRowIndex: 5, reason: 'invalid-topology'}
    ]);
    expect(topology.batches).toEqual([
      {id: 'first', rowCount: 4, sourceRowOffset: 0},
      {id: 'second', rowCount: 4, sourceRowOffset: 4}
    ]);
    // 24 XY tokens (192B), 7+7+9+8 uint32 words (124B), batch
    // manifests 2*64+2*(5+6)=150B, and two 64B diagnostics: 594B.
    expect(topology.cpuByteLength).toBe(594);
    expect(topology.rowCount).toBe(8);
    expect(topology.vertexCount).toBe(24);
    expect(topology.uniqueVertexCount).toBe(20);
    expect(topology.topologyEncoding).toBe('source-xy-key-ids');
    expect(topology.equality).toBe('exact-source-xy');
    expect(topology.ignoresZ).toBe(true);
    for (const criterion of ['queen', 'rook'] as const) {
      const result = computePolygonContiguityOnCPU(topology, {criterion, capacity: 6});
      expect(Array.from(result.offsets)).toEqual(EXPECTED.offsets);
      expect(Array.from(result.neighbors)).toEqual(EXPECTED.neighbors);
      expect(Array.from(result.weights)).toEqual(EXPECTED.weights);
      expect(Array.from(result.status.requiredCount)).toEqual([4]);
      expect(Array.from(result.status.overflow)).toEqual([0]);
      expect(Array.from(result.status.candidateOverflow)).toEqual([0]);
    }
  }
});

it('internal scratch and fallback counts match closed-form small and rollover cases', () => {
  // Run flags N + group indices N + boundary array N+1 = 3N+1.
  expect(getKeyGroupTransientWordCount(8)).toBe(25);
  expect(getKeyGroupTransientWordCount(0)).toBe(1);
  // Cartesian product of query batches and representative batches.
  expect(getIslandAttachmentFallbackBatchCount(8, 3, 3)).toBe(9);
  expect(getIslandAttachmentFallbackBatchCount(33, 3, 7)).toBe(55);
  expect(getIslandAttachmentFallbackBatchCount(65537)).toBe(2);
  expect(getIslandAttachmentFallbackBatchCount(16777216)).toBe(512);
});

it('weights capacity planning reports every field from the closed-form allocation sizes', () => {
  // ceil(17 * 1.25) = 22; 9 offsets cost 36 bytes; each slot is
  // 8 bytes, or 12 with distances. These are independent allocation counts.
  expect(getGPUSpatialWeightsCapacityPlan({rowCount: 8, capacity: 11, requiredCount: 17})).toEqual({
    capacity: 22,
    estimatedRequiredCount: 17,
    requirement: {kind: 'exact', value: 17},
    retryStage: 'final',
    limited: false,
    offsetsByteLength: 36,
    estimatedOutputBytes: 212,
    estimatedPeakTransientBytes: 0
  });
  expect(
    getGPUSpatialWeightsCapacityPlan({
      rowCount: 8,
      capacity: 11,
      requiredCount: 17,
      distances: true
    })
  ).toEqual({
    capacity: 22,
    estimatedRequiredCount: 17,
    requirement: {kind: 'exact', value: 17},
    retryStage: 'final',
    limited: false,
    offsetsByteLength: 36,
    estimatedOutputBytes: 300,
    estimatedPeakTransientBytes: 0
  });
  expect(getGPUSpatialWeightsCapacityPlan({rowCount: 8, capacity: 11, requiredCount: 8})).toEqual({
    capacity: 11,
    estimatedRequiredCount: 8,
    requirement: {kind: 'exact', value: 8},
    retryStage: 'none',
    limited: false,
    offsetsByteLength: 36,
    estimatedOutputBytes: 124,
    estimatedPeakTransientBytes: 0
  });
  for (const headroom of [NaN, Infinity, -Infinity]) {
    expect(() =>
      getGPUSpatialWeightsCapacityPlan({rowCount: 8, capacity: 11, requiredCount: 17, headroom})
    ).toThrow(/headroom/);
  }
  expect(
    getGPUSpatialWeightsCapacityPlan({
      rowCount: 8,
      capacity: 11,
      requiredCount: 17,
      headroom: 1e308
    })
  ).toMatchObject({capacity: 0xffffffff, limited: false});
});

it('every exported weights parameter packer preserves the target tail and IEEE float32 values', () => {
  const expected = 0.10000000149011612; // nearest binary32 to decimal 0.1
  for (const pack of [
    () =>
      getGPUSpatialWeightsSummaryParameterValues({symmetryTolerance: 0.1}, Float32Array.of(7, 99)),
    () =>
      getGPUSpatialWeightsTransposeParameterValues(
        {symmetryTolerance: 0.1},
        Float32Array.of(7, 99)
      ),
    () => getGPUSpatialWeightsTransformParameterValues({bandwidth: 0.1}, Float32Array.of(7, 99))
  ])
    expect(Array.from(pack())).toEqual([expected, 99]);
  expect(
    Array.from(getGPULatticeWeightsParameterValues({cellSize: [0.1, 2]}, Float32Array.of(7, 8, 99)))
  ).toEqual([expected, 2, 99]);
  expect(Array.from(getGPULatticeWeightsParameterValues())).toEqual([1, 1]);
  expect(Array.from(getGPUSpatialWeightsSummaryParameterValues())).toEqual([0]);
  expect(Array.from(getGPUSpatialWeightsTransposeParameterValues())).toEqual([0]);
});
