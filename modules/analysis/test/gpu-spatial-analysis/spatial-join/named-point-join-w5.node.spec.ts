// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createPointInPolygonJoinOperation,
  type PointInPolygonJoinOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join-operation';
import {createCPUPointInPolygonJoinOperation} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-point-in-polygon-join-operation';
import {
  createBufferSelectionOperation,
  type BufferSelectionOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection-operation';
import {createCPUBufferSelectionOperation} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-buffer-selection-operation';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUPointInPolygonJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';

const descriptor = (value: string, length: number) => ({value, length});
function rows(values: Float32Array | Uint32Array, components = 1) {
  const length = values.length / components;
  return {
    format:
      components === 2
        ? ('float32x2' as const)
        : values instanceof Float32Array
          ? ('float32' as const)
          : ('uint32' as const),
    length,
    chunks: [{values, length}],
    pack: () => values
  };
}
function pipProps(): PointInPolygonJoinOperationProps {
  return {
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
      candidateTruncated: 'candidateTruncated',
      ids: 'ids',
      count: 'count',
      requiredCount: 'requiredCount',
      matchOverflow: 'matchOverflow',
      matchCandidateOverflow: 'matchCandidateOverflow',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      candidateCount: 'candidateCount',
      uncertainCount: 'uncertainCount'
    }
  };
}
function resolve(
  operation: GPUContributorOperation,
  inputLengths: Record<string, [number, 'float32' | 'float32x2' | 'uint32']>
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const views = Object.fromEntries(
    Object.entries(operation.outputs).map(([port, output]) => [
      port,
      createTransientView(graph, port, output.format, output.length)
    ])
  );
  for (const [port, [length, format]] of Object.entries(inputLengths))
    views[port] = createTransientView(graph, port, format, length);
  try {
    return operation.createContributor(port => views[port] as never);
  } finally {
    device.destroy();
  }
}

it('captures the same point-in-polygon output map it checks for retired keys', () => {
  const input = pipProps();
  let operationReads = 0;
  const operation = createPointInPolygonJoinOperation({
    ...input,
    get output() {
      return ++operationReads === 1
        ? input.output
        : {...input.output, polygonAssignmentCounts: 'different-counts'};
    }
  });
  expect(operationReads).toBe(1);
  expect(operation.outputs['polygonAssignmentCounts'].value).toBe('polygonAssignmentCounts');
});

it('UGE-01 named GPU wiring and CPU outputs distinguish polygon rows, IDs, assignments and memberships', async () => {
  const input = pipProps();
  const operation = createPointInPolygonJoinOperation(input);
  expect(operation.outputs['pointFeatureRows'].length).toBe(1);
  expect(operation.outputs['polygonAssignmentCounts'].length).toBe(2);
  expect(operation.outputs['polygonMembershipCounts'].length).toBe(2);
  const contributor = resolve(operation, {
    pointspositions: [1, 'float32x2'],
    polygonspositions: [8, 'float32x2'],
    polygonsfeatureOffsets: [3, 'uint32'],
    polygonspolygonOffsets: [3, 'uint32'],
    polygonsringOffsets: [3, 'uint32'],
    polygonssourceIds: [2, 'uint32']
  });
  if (!(contributor instanceof GPUPointInPolygonJoin))
    throw new Error('expected point-in-polygon contributor');
  expect(contributor.props.output).toHaveProperty('pointFeatureRows');
  expect(contributor.props.output).toHaveProperty('polygonAssignmentCounts');
  expect(contributor.props.output).toHaveProperty('polygonMembershipCounts');
  const result = await createCPUPointInPolygonJoinOperation(input).execute({
    inputs: {
      pointspositions: rows(Float32Array.of(0.5, 0.5), 2),
      polygonspositions: rows(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1), 2),
      polygonsfeatureOffsets: rows(Uint32Array.of(0, 1, 2)),
      polygonspolygonOffsets: rows(Uint32Array.of(0, 1, 2)),
      polygonsringOffsets: rows(Uint32Array.of(0, 4, 8)),
      polygonssourceIds: rows(Uint32Array.of(99, 1))
    }
  });
  // Both identical unit squares contain the point; lowest row wins, independent of IDs.
  for (const [port, expected] of Object.entries({
    pointFeatureRows: [0],
    pointFeatureIds: [99],
    polygonAssignmentCounts: [1, 0],
    polygonMembershipCounts: [1, 1]
  }))
    expect(Array.from(result[port] as Uint32Array)).toEqual(expected);
  const legacy = {...input, output: {...input.output, featureCounts: 'counts'}};
  for (const factory of [createPointInPolygonJoinOperation, createCPUPointInPolygonJoinOperation]) {
    expect(() => factory(legacy)).toThrow(
      /featureCounts.*polygonAssignmentCounts.*polygonMembershipCounts/
    );
  }
  const empty = pipProps();
  empty.points = {kind: 'points', positions: descriptor('points', 0)};
  empty.polygons = {
    kind: 'polygons',
    positions: descriptor('positions', 0),
    featureOffsets: descriptor('features', 1),
    polygonOffsets: descriptor('polygons', 1),
    ringOffsets: descriptor('rings', 1)
  };
  empty.matchCapacity = 0;
  expect(() => createPointInPolygonJoinOperation(empty)).not.toThrow();
});

function selectionProps(): BufferSelectionOperationProps {
  return {
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
      count: 'count',
      requiredCount: 'requiredCount',
      matchOverflow: 'matchOverflow',
      matchCandidateOverflow: 'matchCandidateOverflow',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      drawInstanceCount: 'drawInstanceCount'
    }
  };
}
for (const factory of [createBufferSelectionOperation, createCPUBufferSelectionOperation]) {
  it(`N2 ${factory.name} checks each bound segment endpoint against its captured descriptor length`, async () => {
    const input: BufferSelectionOperationProps = {
      ...selectionProps(),
      features: {kind: 'segments', starts: descriptor('starts', 1), ends: descriptor('ends', 1)}
    };
    for (const [starts, ends] of [
      [2, 2],
      [2, 1],
      [1, 2]
    ]) {
      const operation = factory(input);
      if ('createContributor' in operation) {
        expect(() =>
          resolve(operation, {
            pointspositions: [1, 'float32x2'],
            featureStarts: [starts, 'float32x2'],
            featureEnds: [ends, 'float32x2'],
            distance: [1, 'float32']
          })
        ).toThrow(/featureStarts|featureEnds/);
      } else {
        expect(() =>
          operation.execute({
            inputs: {
              pointspositions: rows(Float32Array.of(0, 0), 2),
              featureStarts: rows(new Float32Array(starts * 2), 2),
              featureEnds: rows(new Float32Array(ends * 2), 2),
              distance: rows(Float32Array.of(1))
            }
          })
        ).toThrow(/featureStarts and featureEnds.*declared lengths/);
      }
    }
  });
  it(`UGE-06 ${factory.name} captures inherited radius descriptors once and rejects distance`, async () => {
    const reads = {radius: 0, value: 0, length: 0};
    const radius = Object.create({
      get value() {
        reads.value++;
        return 'radius';
      },
      get length() {
        reads.length++;
        return 1;
      }
    });
    const input = {...selectionProps()} as Partial<BufferSelectionOperationProps>;
    delete input.radius;
    Object.setPrototypeOf(input, {
      get radius() {
        reads.radius++;
        return radius;
      }
    });
    const operation = factory(input as ReturnType<typeof selectionProps>);
    expect(reads).toEqual({radius: 1, value: 1, length: 1});
    expect(operation.inputs['distance']).toBe('radius');
    const legacy = {...selectionProps(), distance: descriptor('radius', 1)};
    expect(() => factory(legacy)).toThrow(/distance.*radius/);
    if ('createContributor' in operation) {
      expect(() =>
        resolve(operation, {
          pointspositions: [1, 'float32x2'],
          featurespositions: [1, 'float32x2'],
          distance: [1, 'float32']
        })
      ).not.toThrow();
    } else {
      const result = await operation.execute({
        inputs: {
          pointspositions: rows(Float32Array.of(0, 0), 2),
          featurespositions: rows(Float32Array.of(1, 0), 2),
          distance: rows(Float32Array.of(1))
        }
      });
      expect(Array.from(result['outputMask'] as Uint32Array)).toEqual([1]);
    }
  });
}
