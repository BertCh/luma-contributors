// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {
  createSpatialPredicateJoinOperation,
  createNearestFeatureJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import type {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {createJoinReviewCases, type JoinReviewCase} from './join-review-response-fixture';

export type Review2Case = JoinReviewCase & {cooperative?: boolean};
const positions = (values: number[]) => ({
  format: 'float32x2' as const,
  values: Float32Array.from(values)
});
const integers = (values: number[]) => ({
  format: 'uint32' as const,
  values: Uint32Array.from(values)
});
const scalar = (format: 'uint32' | 'float32' = 'uint32') => ({format, length: 1});
export function createJoinReview2Cases(): Review2Case[] {
  const cases: Review2Case[] = [];
  for (const neighbors of [false, true])
    for (const reverse of [false, true])
      for (const radius of [13, 12.999999046325684]) {
        cases.push({
          ...createNearestCase(
            `R1 boundary neighbors=${neighbors} reverse=${reverse} radius=${radius}`,
            neighbors,
            [1, 13],
            reverse ? [895, 0] : [0, 0],
            reverse ? [0, 0] : [895, 0],
            radius,
            radius === 13 ? 33 : 0xffffffff,
            radius === 13 ? 13 : -1
          ),
          // Publication takes an f32 square root. At 13, two representable steps are
          // 2 * 2^(3 - 23) = 2^-19. This does not relax ID/count/status comparisons or
          // the adjacent-smaller-radius miss, whose distance remains exactly -1.
          tolerance: radius === 13 ? 2 ** -19 : 0
        });
      }
  for (const neighbors of [false, true])
    for (const reverse of [false, true]) {
      const first = reverse ? [Math.fround(1.1), 2.5] : [-5, -10],
        second = reverse ? [-5, -10] : [Math.fround(1.1), 2.5];
      // Three f32 operations through the squared rank and sqrt allow at most 2e-8 absolute
      // distance error. Integer identity assertions remain exact.
      cases.push({
        ...createNearestCase(
          `R2 fractional tie neighbors=${neighbors} reverse=${reverse}`,
          neighbors,
          [0, 0],
          [...first, ...second],
          [...second, ...first],
          10,
          33,
          Math.fround(0.10784390321559371)
        ),
        tolerance: 2e-8
      });
    }
  for (const reverse of [false, true]) {
    const first = reverse ? [Math.fround(1.1), 2.5] : [-5, -10],
      second = reverse ? [-5, -10] : [Math.fround(1.1), 2.5];
    const output = {
      neighborRows: 'neighborRows',
      neighborIds: 'neighborIds',
      neighborCounts: 'neighborCounts',
      neighborDistances: 'neighborDistances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    };
    cases.push({
      name: `R2 fractional tie operation reverse=${reverse}`,
      tolerance: 2e-8,
      inputs: {
        queriesPositions: positions([0, 0]),
        featuresPositions: positions([...first, ...second, ...second, ...first]),
        featuresLineOffsets: integers([0, 2, 4]),
        featuresSourceIds: integers([33, 42])
      },
      outputs: {
        ...Object.fromEntries(Object.keys(output).map(name => [name, scalar()])),
        neighborDistances: scalar('float32')
      },
      expected: {
        neighborRows: [0],
        neighborIds: [33],
        neighborCounts: [1],
        neighborDistances: [Math.fround(0.10784390321559371)],
        overflow: [0],
        candidateOverflow: [0]
      },
      declare(graph, input, views) {
        const operation = createNearestFeatureJoinOperation({
          queries: {kind: 'points', positions: {value: 'queriesPositions', length: 1}},
          features: {
            kind: 'lines',
            positions: {value: 'featuresPositions', length: 4},
            lineOffsets: {value: 'featuresLineOffsets', length: 3},
            sourceIds: {value: 'featuresSourceIds', length: 2}
          },
          output
        });
        return (
          operation.createContributor(
            port => (views[port] ?? input[port]) as never
          ) as GPUNearestFeatureJoin
        ).getCommandNodes(graph);
      }
    });
  }
  for (const reverse of [false, true])
    cases.push(
      createNearestCase(
        `R4 distant horizontal foot reverse=${reverse}`,
        true,
        [1, 8192],
        reverse ? [2, 2 ** -13] : [0, 2 ** -13],
        reverse ? [0, 2 ** -13] : [2, 2 ** -13],
        10000,
        33,
        8192,
        [1, 2 ** -13]
      )
    );
  cases.push(
    ...createJoinReviewCases({inheritPolygonOffsets: true}).filter(entry =>
      entry.name.startsWith('R3')
    )
  );
  for (const engine of ['fast', 'relate'] as const)
    cases.push(
      ...createJoinReviewCases({dwithinEngine: engine})
        .filter(entry => entry.name.startsWith('R5'))
        .map(entry => ({
          ...entry,
          name: `${entry.name} dwithin engine=${engine}`,
          cooperative: engine === 'relate'
        }))
    );
  for (const engine of ['fast', 'relate'] as const)
    for (const reverse of [false, true]) {
      const road = reverse
        ? [2 ** -10, 2 ** -9, -65536, -65536]
        : [-65536, -65536, 2 ** -10, 2 ** -9];
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        uncertainCount: 'uncertainCount'
      };
      cases.push({
        name: `R9 original-coordinate gap engine=${engine} reverse=${reverse}`,
        cooperative: engine === 'relate',
        inputs: {
          leftPositions: positions([0, 0]),
          rightPositions: positions(road),
          rightLineOffsets: integers([0, 2])
        },
        outputs: Object.fromEntries(Object.keys(output).map(name => [name, scalar()])),
        expected: {
          count: [0],
          requiredCount: [0],
          overflow: [0],
          candidateOverflow: [0],
          uncertainCount: [0]
        },
        declare(graph, input, views) {
          const operation = createSpatialPredicateJoinOperation({
            left: {kind: 'points', positions: {value: 'leftPositions', length: 1}},
            right: {
              kind: 'lines',
              positions: {value: 'rightPositions', length: 2},
              lineOffsets: {value: 'rightLineOffsets', length: 2}
            },
            predicate: 'dwithin',
            distance: 2 ** -11,
            engine,
            candidateCapacity: 1,
            pairCapacity: 1,
            output
          });
          return (
            operation.createContributor(
              port => (views[port] ?? input[port]) as never
            ) as GPUSpatialPredicateJoin
          ).getCommandNodes(graph);
        }
      });
    }
  return cases;
}

function createNearestCase(
  name: string,
  neighbors: boolean,
  query: number[],
  starts: number[],
  ends: number[],
  radius: number,
  expectedId: number,
  expectedDistance: number,
  foot?: number[]
): Review2Case {
  const count = starts.length / 2;
  return {
    name,
    inputs: {
      query: positions(query),
      starts: positions(starts),
      ends: positions(ends),
      sourceIds: integers(count === 1 ? [33] : [33, 42]),
      radius: {format: 'float32', values: Float32Array.of(radius)}
    },
    outputs: {
      ids: scalar(),
      distances: scalar('float32'),
      overflow: scalar(),
      ...(neighbors ? {counts: scalar()} : {}),
      ...(foot
        ? {
            foot: {format: 'float32x2' as const, length: 1},
            queryPoint: {format: 'float32x2' as const, length: 1}
          }
        : {})
    },
    expected: {
      ids: [expectedId],
      distances: [expectedDistance],
      overflow: [0],
      ...(neighbors ? {counts: [Number(expectedId !== 0xffffffff)]} : {}),
      ...(foot ? {foot, queryPoint: query} : {})
    },
    declare(graph, input, output) {
      const integer = (name: string) => output[name] as GraphDataView<'uint32'>;
      return new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: input['query'] as GraphDataView<'float32x2'>},
        features: {
          kind: 'segments',
          starts: input['starts'] as GraphDataView<'float32x2'>,
          ends: input['ends'] as GraphDataView<'float32x2'>,
          sourceIds: input['sourceIds'] as GraphDataView<'uint32'>
        },
        radius: input['radius'] as GraphDataView<'float32'>,
        candidateCapacity: 8,
        spatialSort: false,
        querySort: false,
        output: {
          status: {overflow: integer('overflow')},
          ...(neighbors
            ? {
                neighborIds: integer('ids'),
                neighborCounts: integer('counts'),
                neighborDistances: output['distances'] as GraphDataView<'float32'>
              }
            : {
                nearestFeatureIds: integer('ids'),
                nearestDistances: output['distances'] as GraphDataView<'float32'>
              }),
          ...(foot
            ? {
                neighborFootPoints: output['foot'] as GraphDataView<'float32x2'>,
                neighborQueryPoints: output['queryPoint'] as GraphDataView<'float32x2'>
              }
            : {})
        }
      }).getCommandNodes(graph);
    }
  };
}
