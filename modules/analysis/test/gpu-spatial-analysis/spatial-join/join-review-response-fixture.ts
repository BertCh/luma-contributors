// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {captureGraphCommandNodes} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import {
  createSpatialPredicateJoinOperation,
  createNearestFeatureJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import type {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import type {FixtureInput, FixtureOutput} from '../outline-geometry/geometry-fixture';

type Views = Record<string, GraphDataView>;
export type JoinReviewCase = {
  name: string;
  inputs: Record<string, FixtureInput>;
  outputs: Record<string, FixtureOutput>;
  expected: Record<string, number[]>;
  tolerance?: number;
  declare(graph: GPUCommandGraph, input: Views, output: Views): readonly GPUCommandNode[];
};
const positions = (values: number[]): FixtureInput => ({
  format: 'float32x2',
  values: Float32Array.from(values)
});
const integers = (values: number[]): FixtureInput => ({
  format: 'uint32',
  values: Uint32Array.from(values)
});
const floats = (values: number[]): FixtureInput => ({
  format: 'float32',
  values: Float32Array.from(values)
});
const scalar = (format: 'uint32' | 'float32' = 'uint32'): FixtureOutput => ({format, length: 1});
const view = <Format extends GraphDataView['format']>(views: Views, name: string) =>
  views[name] as GraphDataView<Format>;

export function createJoinReviewCases(
  options: {inheritPolygonOffsets?: boolean; dwithinEngine?: 'fast' | 'relate'} = {}
): JoinReviewCase[] {
  const cases: JoinReviewCase[] = [];
  for (const reverse of [false, true])
    for (const hit of [false, true]) {
      cases.push({
        name: `R1 nearest-feature road reverse=${reverse} hit=${hit}`,
        inputs: {
          queries: positions([0, 0]),
          starts: positions([...(reverse ? [1, 1] : [-8191, -8190]), ...(hit ? [0, 0] : [])]),
          ends: positions([...(reverse ? [-8191, -8190] : [1, 1]), ...(hit ? [1, 0] : [])]),
          ids: integers(hit ? [33, 42] : [33]),
          radius: floats([0])
        },
        outputs: {ids: scalar(), distances: scalar('float32'), overflow: scalar()},
        expected: {ids: [hit ? 42 : 0xffffffff], distances: [hit ? 0 : -1], overflow: [0]},
        declare: (graph, input, output) =>
          new GPUNearestFeatureJoin({
            queries: {kind: 'points', positions: view<'float32x2'>(input, 'queries')},
            features: {
              kind: 'segments',
              starts: view<'float32x2'>(input, 'starts'),
              ends: view<'float32x2'>(input, 'ends'),
              sourceIds: view<'uint32'>(input, 'ids')
            },
            radius: view<'float32'>(input, 'radius'),
            candidateCapacity: 8,
            spatialSort: false,
            querySort: false,
            output: {
              nearestFeatureIds: view<'uint32'>(output, 'ids'),
              nearestDistances: view<'float32'>(output, 'distances'),
              status: {overflow: view<'uint32'>(output, 'overflow')}
            }
          }).getCommandNodes(graph)
      });
    }
  cases.push({
    name: 'R1 nearest-feature tiny point outranks coincident point',
    inputs: {
      queries: positions([0, 0]),
      targets: positions([1e-24, 0, 0, 0]),
      ids: integers([33, 42]),
      radius: floats([1])
    },
    outputs: {ids: scalar(), distances: scalar('float32'), overflow: scalar()},
    expected: {ids: [42], distances: [0], overflow: [0]},
    declare: (graph, input, output) =>
      new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: view<'float32x2'>(input, 'queries')},
        features: {
          kind: 'points',
          positions: view<'float32x2'>(input, 'targets'),
          sourceIds: view<'uint32'>(input, 'ids')
        },
        radius: view<'float32'>(input, 'radius'),
        candidateCapacity: 8,
        spatialSort: false,
        querySort: false,
        output: {
          nearestFeatureIds: view<'uint32'>(output, 'ids'),
          nearestDistances: view<'float32'>(output, 'distances'),
          status: {overflow: view<'uint32'>(output, 'overflow')}
        }
      }).getCommandNodes(graph)
  });
  for (const reverse of [false, true])
    cases.push({
      name: `R4 nearby perpendicular foot reverse=${reverse}`,
      inputs: {
        queries: positions([0, 0]),
        targets: positions(reverse ? [1, 1, -8191, -8190] : [-8191, -8190, 1, 1]),
        lines: integers([0, 2])
      },
      outputs: {
        ids: scalar(),
        rows: scalar(),
        counts: scalar(),
        distances: scalar('float32'),
        foot: {format: 'float32x2', length: 1},
        query: {format: 'float32x2', length: 1},
        overflow: scalar()
      },
      // Original-coordinate determinant is 1; squared length = 8192^2 + 8191^2.
      expected: {
        ids: [0],
        rows: [0],
        counts: [1],
        distances: [0.0000863220142671425],
        foot: [-0.00006103515579519714, 0.00006104260728534427],
        query: [0, 0],
        overflow: [0]
      },
      // Three float32 steps at this scale cover the scaled products/division/conversion.
      tolerance: 2e-11,
      declare: (graph, input, output) =>
        new GPUNearestFeatureJoin({
          queries: {kind: 'points', positions: view<'float32x2'>(input, 'queries')},
          features: {
            kind: 'lines',
            positions: view<'float32x2'>(input, 'targets'),
            lineOffsets: view<'uint32'>(input, 'lines')
          },
          spatialSort: false,
          output: {
            neighborIds: view<'uint32'>(output, 'ids'),
            neighborRows: view<'uint32'>(output, 'rows'),
            neighborCounts: view<'uint32'>(output, 'counts'),
            neighborDistances: view<'float32'>(output, 'distances'),
            neighborFootPoints: view<'float32x2'>(output, 'foot'),
            neighborQueryPoints: view<'float32x2'>(output, 'query'),
            status: {overflow: view<'uint32'>(output, 'overflow')}
          }
        }).getCommandNodes(graph)
    });
  for (const backend of ['zonal', 'group'] as const)
    cases.push({
      name: `R3 inherited grouping ${backend}`,
      inputs: {
        points: positions([1, 1, 2, 2]),
        pointOffsets: integers([0, 0, 2]),
        polygon: positions([0, 0, 3, 0, 3, 3, 0, 3]),
        features: integers([0, 1]),
        polygons: integers([0, 1]),
        rings: integers([0, 4])
      },
      outputs: {counts: scalar(), overflow: scalar(), uncertain: scalar()},
      expected: {counts: [1], overflow: [0], uncertain: [0]},
      declare: (graph, input, output) =>
        captureGraphCommandNodes(graph, () =>
          addPointsInPolygonsChoroplethRecipe(graph, {
            backend,
            points: Object.assign(
              Object.create({
                get featureOffsets() {
                  return input['pointOffsets'];
                }
              }),
              {kind: 'points', positions: input['points']}
            ),
            polygons: Object.assign(
              Object.create(
                options.inheritPolygonOffsets
                  ? {
                      get featureOffsets() {
                        return input['features'];
                      }
                    }
                  : null
              ),
              {
                kind: 'polygons',
                positions: view<'float32x2'>(input, 'polygon'),
                ...(!options.inheritPolygonOffsets
                  ? {featureOffsets: view<'uint32'>(input, 'features')}
                  : {}),
                polygonOffsets: view<'uint32'>(input, 'polygons'),
                ringOffsets: view<'uint32'>(input, 'rings'),
                candidateCapacity: 8
              }
            ),
            output: {
              counts: view<'uint32'>(output, 'counts'),
              status: {
                overflow: view<'uint32'>(output, 'overflow'),
                uncertainCount: view<'uint32'>(output, 'uncertain')
              }
            }
          })
        )
    });
  for (const grouped of [false, true]) {
    const large = 2 ** 120,
      delta = 2 ** 98,
      small = 2 ** -100;
    cases.push({
      name: `R5 operation uncertainty grouped=${grouped}`,
      inputs: {
        leftPositions: positions([large / 4, large / 4 + delta, large / 2, large / 2 + delta]),
        rightPositions: positions([small, small, large, large]),
        leftLineOffsets: integers([0, 2]),
        rightLineOffsets: integers([0, 2]),
        leftFeatureOffsets: integers([0, 1]),
        ...(grouped ? {rightFeatureOffsets: integers([0, 1])} : {})
      },
      outputs: {
        leftRows: scalar(),
        rightRows: scalar(),
        count: scalar(),
        requiredCount: scalar(),
        overflow: scalar(),
        candidateOverflow: scalar(),
        candidateTruncated: scalar(),
        uncertainCount: scalar()
      },
      expected: {
        leftRows: [0],
        rightRows: [0],
        count: [1],
        requiredCount: [1],
        overflow: [0],
        candidateOverflow: [0],
        candidateTruncated: [0],
        uncertainCount: [1]
      },
      declare(graph, input, output) {
        const destinations = Object.fromEntries(
          Object.keys(this.outputs).map(name => [name, name])
        ) as {
          leftRows: string;
          rightRows: string;
          count: string;
          requiredCount: string;
          overflow: string;
          candidateOverflow: string;
          uncertainCount: string;
        };
        const operation = createSpatialPredicateJoinOperation({
          left: {
            kind: 'lines',
            positions: {value: 'leftPositions', length: 2},
            lineOffsets: {value: 'leftLineOffsets', length: 2},
            featureOffsets: {value: 'leftFeatureOffsets', length: 2}
          },
          right: {
            kind: 'lines',
            positions: {value: 'rightPositions', length: 2},
            lineOffsets: {value: 'rightLineOffsets', length: 2},
            ...(grouped ? {featureOffsets: {value: 'rightFeatureOffsets', length: 2}} : {})
          },
          predicate: options.dwithinEngine ? 'dwithin' : 'intersects',
          ...(options.dwithinEngine ? {engine: options.dwithinEngine, distance: 0} : {}),
          candidateCapacity: 8,
          pairCapacity: 1,
          output: destinations
        });
        return (
          operation.createContributor(
            port => (output[port] ?? input[port]) as never
          ) as GPUSpatialPredicateJoin
        ).getCommandNodes(graph);
      }
    });
  }
  for (const reverse of [false, true]) {
    const road = reverse ? [-65535, -65534, 1, 1] : [1, 1, -65535, -65534];
    cases.push({
      name: `R2 neighbors operation duplicate-road exact tie reverse=${reverse}`,
      inputs: {
        queriesPositions: positions([0, 0]),
        featuresPositions: positions([...road, road[2], road[3], road[0], road[1]]),
        featuresLineOffsets: integers([0, 2, 4]),
        featuresSourceIds: integers(reverse ? [42, 33] : [33, 42])
      },
      outputs: {
        neighborRows: scalar(),
        neighborIds: scalar(),
        neighborCounts: scalar(),
        neighborDistances: scalar('float32'),
        overflow: scalar(),
        candidateOverflow: scalar()
      },
      expected: {
        neighborRows: [0],
        neighborIds: [reverse ? 42 : 33],
        neighborCounts: [1],
        neighborDistances: [0.00001078967579815071],
        overflow: [0],
        candidateOverflow: [0]
      },
      tolerance: 2e-12,
      declare(graph, input, output) {
        const operation = createNearestFeatureJoinOperation({
          queries: {kind: 'points', positions: {value: 'queriesPositions', length: 1}},
          features: {
            kind: 'lines',
            positions: {value: 'featuresPositions', length: 4},
            lineOffsets: {value: 'featuresLineOffsets', length: 3},
            sourceIds: {value: 'featuresSourceIds', length: 2}
          },
          output: {
            neighborRows: 'neighborRows',
            neighborIds: 'neighborIds',
            neighborCounts: 'neighborCounts',
            neighborDistances: 'neighborDistances',
            overflow: 'overflow',
            candidateOverflow: 'candidateOverflow'
          }
        });
        return (
          operation.createContributor(
            port => (output[port] ?? input[port]) as never
          ) as GPUNearestFeatureJoin
        ).getCommandNodes(graph);
      }
    });
  }
  return cases;
}
