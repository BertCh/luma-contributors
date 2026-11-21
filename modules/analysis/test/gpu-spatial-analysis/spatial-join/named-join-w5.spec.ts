// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createSpatialPredicateJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {createPointInPolygonJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join-operation';
import {createBufferSelectionOperation} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection-operation';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createPoisonedOracleGraph,
  runPoisonedOracleGraph,
  type OracleInput,
  type OraclePort
} from '../spatial-clustering/output-oracle-test-utils';
import {createJoinReview2Cases} from './join-review2-fixture';

const descriptor = (value: string, length: number) => ({value, length});
const positions = (values: number[]): OracleInput => ({
  format: 'float32x2',
  length: values.length / 2,
  values: Float32Array.from(values)
});
const integers = (values: number[]): OracleInput => ({
  format: 'uint32',
  length: values.length,
  values: Uint32Array.from(values)
});
function wire(
  operation: GPUContributorOperation,
  harness: ReturnType<typeof createPoisonedOracleGraph>
) {
  harness.graph.add(
    operation.createContributor(port =>
      harness.getView(operation.inputs[port] ?? operation.outputs[port].value)
    )
  );
}
function outputs(operation: GPUContributorOperation): Record<string, OraclePort> {
  return Object.fromEntries(Object.values(operation.outputs).map(output => [output.value, output]));
}
const status = {
  count: 'count',
  requiredCount: 'requiredCount',
  overflow: 'overflow',
  candidateOverflow: 'candidateOverflow'
};

it('UGE-02 named polygon live prefixes shrink, retain an empty feature, propagate loss and replay cleanly', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = createSpatialPredicateJoinOperation({
    left: {
      kind: 'polygons',
      positions: descriptor('vertices', 8),
      featureOffsets: descriptor('features', 4),
      polygonOffsets: descriptor('polygons', 3),
      ringOffsets: descriptor('rings', 3)
    },
    right: {kind: 'points', positions: descriptor('target', 1)},
    leftFeatureCount: 'live',
    leftStatus: {overflow: 'loss', candidateOverflow: 'candidateLoss', complete: 'complete'},
    predicate: 'intersects',
    how: 'anti',
    candidateCapacity: 3,
    pairCapacity: 3,
    output: {unmatchedRows: 'unmatchedRows', ...status}
  });
  const harness = createPoisonedOracleGraph(
    device,
    {
      vertices: positions([0, 0, 2, 0, 2, 2, 0, 2, 8, 0, 10, 0, 10, 2, 8, 2]),
      features: integers([0, 1, 1, 2]),
      polygons: integers([0, 1, 2]),
      rings: integers([0, 4, 8]),
      target: positions([1, 1]),
      live: integers([3]),
      loss: integers([0]),
      candidateLoss: integers([0]),
      complete: integers([1])
    },
    outputs(operation)
  );
  wire(operation, harness);
  const compiled = harness.graph.compile();
  try {
    // The middle feature owns no polygon but is a genuine live empty. Spare rows never count.
    for (const [live, loss, candidateLoss, complete, expected] of [
      [3, 0, 0, 1, [1, 2]],
      [1, 0, 0, 1, []],
      [2, 0, 0, 1, [1]],
      [2, 1, 0, 1, []],
      [2, 0, 1, 1, []],
      [2, 0, 0, 0, []],
      [4, 0, 0, 1, []],
      [2, 0, 0, 1, [1]],
      [0, 0, 0, 1, []]
    ] as const) {
      harness.buffers['live'].write(Uint32Array.of(live));
      harness.buffers['loss'].write(Uint32Array.of(loss));
      harness.buffers['candidateLoss'].write(Uint32Array.of(candidateLoss));
      harness.buffers['complete'].write(Uint32Array.of(complete));
      harness.poison();
      submitGraph(device, compiled, undefined);
      const result = await harness.read();
      expect(result['count']).toEqual([expected.length]);
      expect(result['unmatchedRows'].slice(0, expected.length)).toEqual(expected);
      expect(result['candidateOverflow']).toEqual([
        Number(Boolean(loss || candidateLoss || !complete || live > 3))
      ]);
    }
  } finally {
    compiled.destroy();
    operation.destroy();
    harness.destroy();
  }
});

it('B1 named inner oversized live count suppresses absence and recovers on valid replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = createSpatialPredicateJoinOperation({
    left: {kind: 'points', positions: descriptor('left', 2)},
    right: {kind: 'points', positions: descriptor('right', 1)},
    leftFeatureCount: 'live',
    predicate: 'intersects',
    candidateCapacity: 2,
    pairCapacity: 2,
    output: {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      ...status,
      unmatchedRows: 'unmatchedRows',
      unmatchedCount: 'unmatchedCount',
      unmatchedOverflow: 'unmatchedOverflow',
      unmatchedCandidateOverflow: 'unmatchedCandidateOverflow'
    }
  });
  const harness = createPoisonedOracleGraph(
    device,
    {left: positions([0, 0, 1, 1]), right: positions([9, 9]), live: integers([3])},
    outputs(operation)
  );
  wire(operation, harness);
  const compiled = harness.graph.compile();
  try {
    for (const live of [3, 1, 2]) {
      harness.buffers['live'].write(Uint32Array.of(live));
      harness.poison();
      submitGraph(device, compiled, undefined);
      const result = await harness.read();
      expect(result['count']).toEqual([0]);
      expect(result['unmatchedCount']).toEqual([live > 2 ? 0 : live]);
      expect(result['unmatchedCandidateOverflow']).toEqual([Number(live > 2)]);
      if (live <= 2)
        expect(result['unmatchedRows'].slice(0, live)).toEqual(live === 1 ? [0] : [0, 1]);
    }
  } finally {
    compiled.destroy();
    operation.destroy();
    harness.destroy();
  }
});

it('UGE-12 named inner absence excludes truncated matches and candidate unknowns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const candidateCapacity of [1, 6]) {
    const operation = createSpatialPredicateJoinOperation({
      left: {kind: 'points', positions: descriptor('points', 3)},
      right: {kind: 'points', positions: descriptor('targets', 2)},
      predicate: 'intersects',
      candidateCapacity,
      pairCapacity: 1,
      output: {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        ...status,
        unmatchedRows: 'unmatchedRows',
        unmatchedCount: 'unmatchedCount',
        unmatchedRequiredCount: 'unmatchedRequiredCount',
        unmatchedOverflow: 'unmatchedOverflow',
        unmatchedCandidateOverflow: 'unmatchedCandidateOverflow'
      }
    });
    const harness = createPoisonedOracleGraph(
      device,
      {points: positions([0, 0, 0, 0, 9, 9]), targets: positions([0, 0, 0, 0])},
      outputs(operation)
    );
    wire(operation, harness);
    try {
      for (const result of await runPoisonedOracleGraph(harness, device)) {
        expect(result['leftRows']).toEqual([0]);
        expect(result['unmatchedRows']).toEqual([2]);
        expect(result['unmatchedCount']).toEqual([1]);
        expect(result['unmatchedRequiredCount']).toEqual([1]);
        expect(result['unmatchedOverflow']).toEqual([0]);
        expect(result['unmatchedCandidateOverflow']).toEqual([Number(candidateCapacity === 1)]);
      }
    } finally {
      operation.destroy();
    }
  }
});

it('UGE-12 named inner uncertainty withholds a true miss until a certified replay', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const large = 2 ** 120,
    delta = 2 ** 98,
    small = 2 ** -100;
  const operation = createSpatialPredicateJoinOperation({
    left: {
      kind: 'lines',
      positions: descriptor('left', 4),
      lineOffsets: descriptor('leftLines', 3)
    },
    right: {
      kind: 'lines',
      positions: descriptor('right', 2),
      lineOffsets: descriptor('rightLines', 2)
    },
    predicate: 'dwithin',
    distance: 0,
    engine: 'relate',
    candidateCapacity: 2,
    pairCapacity: 1,
    output: {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      ...status,
      uncertainCount: 'uncertainCount',
      unmatchedRows: 'unmatchedRows',
      unmatchedCount: 'unmatchedCount',
      unmatchedRequiredCount: 'unmatchedRequiredCount',
      unmatchedOverflow: 'unmatchedOverflow',
      unmatchedCandidateOverflow: 'unmatchedCandidateOverflow'
    }
  });
  const harness = createPoisonedOracleGraph(
    device,
    {
      left: positions([large / 4, large / 4 + delta, large / 2, large / 2 + delta, -2, -2, -1, -1]),
      right: positions([small, small, large, large]),
      leftLines: integers([0, 2, 4]),
      rightLines: integers([0, 2])
    },
    outputs(operation)
  );
  wire(operation, harness);
  const compiled = harness.graph.compile();
  try {
    // The original 220-bit-span pair is uncertifiable. The separate negative line is a
    // true miss, but global uncertainty conservatively withholds every absence claim.
    for (const uncertain of [true, false, true]) {
      harness.buffers['right'].write(
        Float32Array.from(uncertain ? [small, small, large, large] : [0, 0, 4, 4])
      );
      harness.poison();
      submitGraph(device, compiled, undefined);
      const result = await harness.read();
      expect(result['uncertainCount']).toEqual([Number(uncertain)]);
      expect(result['unmatchedCount']).toEqual([uncertain ? 0 : 1]);
      expect(result['unmatchedRequiredCount']).toEqual([uncertain ? 0 : 2]);
      expect(result['unmatchedOverflow']).toEqual([uncertain ? 0 : 1]);
      if (!uncertain) expect(result['unmatchedRows']).toEqual([0]);
      expect(result['unmatchedCandidateOverflow']).toEqual([0]);
    }
  } finally {
    compiled.destroy();
    operation.destroy();
    harness.destroy();
  }
});

it('UGE-01 named point polygon outputs publish row and ID separately and both polygon counts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = createPointInPolygonJoinOperation({
    points: {kind: 'points', positions: descriptor('points', 1)},
    polygons: {
      kind: 'polygons',
      positions: descriptor('vertices', 8),
      featureOffsets: descriptor('features', 3),
      polygonOffsets: descriptor('polygons', 3),
      ringOffsets: descriptor('rings', 3),
      sourceIds: descriptor('sourceIds', 2)
    },
    candidateCapacity: 4,
    matchCapacity: 1,
    output: {
      pointFeatureIds: 'pointFeatureIds',
      pointFeatureRows: 'pointFeatureRows',
      polygonAssignmentCounts: 'polygonAssignmentCounts',
      polygonMembershipCounts: 'polygonMembershipCounts',
      ids: 'ids',
      ...status,
      matchOverflow: 'matchOverflow',
      matchCandidateOverflow: 'matchCandidateOverflow',
      candidateTruncated: 'candidateTruncated',
      candidateCount: 'candidateCount',
      uncertainCount: 'uncertainCount'
    }
  });
  const harness = createPoisonedOracleGraph(
    device,
    {
      points: positions([0.5, 0.5]),
      vertices: positions([0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]),
      features: integers([0, 1, 2]),
      polygons: integers([0, 1, 2]),
      rings: integers([0, 4, 8]),
      sourceIds: integers([99, 1])
    },
    outputs(operation)
  );
  wire(operation, harness);
  try {
    for (const result of await runPoisonedOracleGraph(harness, device)) {
      expect(result['pointFeatureRows']).toEqual([0]);
      expect(result['pointFeatureIds']).toEqual([99]);
      expect(result['polygonAssignmentCounts']).toEqual([1, 0]);
      expect(result['polygonMembershipCounts']).toEqual([1, 1]);
    }
  } finally {
    operation.destroy();
  }
});

it('UGE-06 named radius-only selection retains the distance input key and inclusive result', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const operation = createBufferSelectionOperation({
    points: {kind: 'points', positions: descriptor('points', 1)},
    features: {kind: 'points', positions: descriptor('features', 1)},
    radius: descriptor('radius', 1),
    candidateCapacity: 1,
    matchCapacity: 1,
    output: {
      outputMask: 'outputMask',
      distances: 'distances',
      nearestFeatureRows: 'nearestFeatureRows',
      candidateTruncated: 'candidateTruncated',
      ids: 'ids',
      ...status,
      matchOverflow: 'matchOverflow',
      matchCandidateOverflow: 'matchCandidateOverflow',
      drawInstanceCount: 'drawInstanceCount'
    }
  });
  const harness = createPoisonedOracleGraph(
    device,
    {
      points: positions([0, 0]),
      features: positions([1, 0]),
      radius: {format: 'float32', length: 1, values: Float32Array.of(1)}
    },
    outputs(operation)
  );
  wire(operation, harness);
  try {
    for (const result of await runPoisonedOracleGraph(harness, device)) {
      expect(result['outputMask']).toEqual([1]);
      expect(result['ids']).toEqual([0]);
      expect(result['count']).toEqual([1]);
      expect(result['distances']).toEqual([1]);
      expect(result['candidateOverflow']).toEqual([0]);
    }
  } finally {
    operation.destroy();
  }
});

for (const engine of ['fast', 'relate'] as const) {
  it(`KI-rr-joins-R5 ${engine} zero-radius exact contact and single pair uncertainty`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const operation = createSpatialPredicateJoinOperation({
      left: {kind: 'points', positions: descriptor('left', 1)},
      right: {kind: 'points', positions: descriptor('right', 1)},
      predicate: 'dwithin',
      distance: 0,
      engine,
      candidateCapacity: 1,
      pairCapacity: 1,
      output: {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        ...status,
        uncertainCount: 'uncertainCount'
      }
    });
    const harness = createPoisonedOracleGraph(
      device,
      {left: positions([1, 1]), right: positions([1, 1])},
      outputs(operation)
    );
    wire(operation, harness);
    try {
      for (const result of await runPoisonedOracleGraph(harness, device)) {
        expect(result['count']).toEqual([1]);
        expect(result['uncertainCount']).toEqual([0]);
      }
    } finally {
      operation.destroy();
    }
    for (const entry of createJoinReview2Cases().filter(
      entry => entry.name.startsWith('R5') && entry.name.endsWith(`engine=${engine}`)
    )) {
      const inputs = Object.fromEntries(
        Object.entries(entry.inputs).map(([name, port]) => [
          name,
          {...port, length: port.values.length / (port.format === 'float32x2' ? 2 : 1)}
        ])
      );
      const test = createPoisonedOracleGraph(device, inputs, entry.outputs);
      test.graph.add({
        getCommandNodes: graph =>
          entry.declare(
            graph,
            Object.fromEntries(Object.keys(inputs).map(name => [name, test.getView(name)])),
            Object.fromEntries(Object.keys(entry.outputs).map(name => [name, test.getView(name)]))
          )
      });
      for (const result of await runPoisonedOracleGraph(test, device))
        expect(result['uncertainCount']).toEqual([1]);
    }
  });
}
