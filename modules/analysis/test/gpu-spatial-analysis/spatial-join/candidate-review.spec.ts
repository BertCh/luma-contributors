// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {getSpatialJoinCandidateNodes} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-candidates';
import {
  QUANTIZE_AXIS_WGSL,
  SPATIAL_JOIN_WGSL_HELPERS
} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-passes';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {requireTestDevice} from '../../utils/require-test-device';
import {getCandidateRankDeclarations} from './candidate-rank-test-utils';
import {createPolygonJoinRun, createSquareFeature} from './spatial-join-sort-utils';

it('quantizes half, quarter and three-quarter extents at extreme float32 scales', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      values: {
        format: 'float32',
        values: Float32Array.from([
          2 ** 126,
          0,
          2 ** 127,
          2 ** -102,
          0,
          2 ** -100,
          3 * 2 ** -102,
          0,
          2 ** -100,
          0,
          -(2 ** 126),
          2 ** 126
        ])
      }
    },
    outputs: {cells: {format: 'uint32', length: 4}},
    create: ({inputs, outputs}) => ({
      getCommandNodes(graph) {
        return [
          createWGSLKernelNode(graph, {
            id: 'extreme-join-quantization',
            operation: 'SpatialJoinQuantization',
            invocationCount: 4,
            bindings: [
              {name: 'values', view: inputs['values'], type: 'f32', access: 'read'},
              {name: 'cells', view: outputs['cells'], type: 'u32', access: 'read_write'}
            ],
            declarations: `${SPATIAL_JOIN_WGSL_HELPERS}\n${QUANTIZE_AXIS_WGSL}`,
            body: `let row = valuesOffset + index * 3u;
  cells[cellsOffset + index] = quantizeAxis(values[row], values[row + 1u], values[row + 2u]);`
          })
        ];
      }
    })
  });
  try {
    expect((await fixture.run())['cells']).toEqual([32768, 16384, 49151, 32768]);
  } finally {
    fixture.destroy();
  }
});

for (const radius of [2 ** 127, 2 ** -100]) {
  it(`retains the 20-degree geodesic candidate for a half-radian runtime margin at scale ${radius}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        leftMinima: {format: 'float32x2', values: new Float32Array(2)},
        leftMaxima: {format: 'float32x2', values: new Float32Array(2)},
        nodeMinima: {format: 'float32x2', values: Float32Array.from([20, 0, 20, 0, 40, 0])},
        nodeMaxima: {format: 'float32x2', values: Float32Array.from([40, 0, 20, 0, 40, 0])},
        leafIds: {format: 'uint32', values: Uint32Array.from([0, 1])},
        margin: {format: 'float32', values: Float32Array.of(radius / 2)}
      },
      outputs: {state: {format: 'uint32', length: 4}, pairs: {format: 'uint32', length: 4}},
      create: ({inputs, outputs}) => ({
        getCommandNodes(graph) {
          return getSpatialJoinCandidateNodes(graph, {
            id: 'extreme-geodesic-candidates',
            leftCount: 1,
            rightCount: 2,
            leftMinima: inputs['leftMinima'] as GraphDataView<'float32x2'>,
            leftMaxima: inputs['leftMaxima'] as GraphDataView<'float32x2'>,
            tree: {
              nodeMinima: inputs['nodeMinima'] as GraphDataView<'float32x2'>,
              nodeMaxima: inputs['nodeMaxima'] as GraphDataView<'float32x2'>,
              leafIds: inputs['leafIds'] as GraphDataView<'uint32'>,
              internalNodeCount: 1
            },
            candidateCapacity: 2,
            sphereRadius: radius,
            margin: inputs['margin'] as GraphDataView<'float32'>,
            state: outputs['state'] as GraphDataView<'uint32'>,
            candidatePairs: graph.createDataView(outputs['pairs'].buffer, {
              format: 'uint32x2',
              length: 2
            })
          });
        }
      })
    });
    try {
      const result = await fixture.run();
      // Half a radian reaches past 20 degrees but not 40 degrees at the equator.
      expect(result['state'][0]).toBe(1);
      expect(result['pairs'].slice(0, 2)).toEqual([0, 0]);
      fixture.writeInput('margin', Float32Array.of(0));
      expect((await fixture.run())['state'][0]).toBe(0);
      fixture.writeInput('margin', Float32Array.of(radius / 2));
      expect((await fixture.run())['state'][0]).toBe(1);
    } finally {
      fixture.destroy();
    }
  });
}

it('benchmarks production rank selection over 32768 overlapping sorted leaves', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rightCount = 32768;
  const measured: number[] = [];
  const elapsed: number[] = [];
  for (const legacy of [true, false]) {
    const declarations = getCandidateRankDeclarations(rightCount, legacy);
    const fixture = createGeometryFixture(device, {
      inputs: {
        leftMinima: {values: new Float32Array(2), format: 'float32x2'},
        leftMaxima: {values: new Float32Array(2), format: 'float32x2'},
        nodeMinima: {values: new Float32Array((rightCount * 2 - 1) * 2), format: 'float32x2'},
        nodeMaxima: {values: new Float32Array((rightCount * 2 - 1) * 2), format: 'float32x2'},
        leafIds: {
          values: Uint32Array.from({length: rightCount}, (_, index) => rightCount - index - 1),
          format: 'uint32'
        }
      },
      outputs: {probeWork: {format: 'uint32', length: 1}, result: {format: 'uint32', length: 1}},
      create: ({inputs, outputs}) => ({
        getCommandNodes(graph) {
          return [
            createWGSLKernelNode(graph, {
              id: 'rank-probe-work',
              operation: 'CandidateRankBenchmark',
              invocationCount: 1,
              bindings: [
                ...Object.entries(inputs).map<WGSLKernelBinding>(([name, view]) => ({
                  name,
                  view,
                  type: name === 'leafIds' ? 'u32' : 'f32',
                  access: 'read' as const
                })),
                {name: 'probeWork', view: outputs['probeWork'], type: 'u32', access: 'read_write'},
                {name: 'result', view: outputs['result'], type: 'u32', access: 'read_write'}
              ],
              declarations,
              body: 'probeWork[probeWorkOffset] = 0u; result[resultOffset] = findFirstPageRight(0u, 0u);'
            })
          ];
        }
      })
    });
    try {
      const start = performance.now();
      const result = await fixture.run();
      elapsed.push(performance.now() - start);
      measured.push(result['probeWork'][0]);
      expect(result['result']).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  }
  // The production traversal visits every leaf at each of 15 rank bisections before the fix.
  expect(measured).toEqual([15 * rightCount, 0]);
  console.log('CANDIDATE-RANK-WORK', {
    before: measured[0],
    after: measured[1],
    milliseconds: elapsed
  });
});

it('benchmarks shuffled point querySort on/off with one and multiple candidate pages', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pointCount = 4096;
  const features = Array.from({length: 64}, (_, index) => createSquareFeature(index * 2, 0, 1));
  const rows = Array.from({length: pointCount}, (_, index) => (index * 73) % pointCount);
  const points: [number, number][] = rows.map(row => [(row % 64) * 2 + 0.5, 0.5]);
  for (const candidateTileCapacity of [pointCount, 257]) {
    for (const querySort of [false, true]) {
      const run = createPolygonJoinRun(device, features, points, {
        spatialSort: true,
        querySort,
        candidateCapacity: pointCount,
        candidateTileCapacity
      });
      try {
        await run.encodeAndWait();
        const start = performance.now();
        for (let encoding = 0; encoding < 3; encoding++) await run.encodeAndWait();
        const milliseconds = (performance.now() - start) / 3;
        const result = await run.readResult();
        expect(result.featureRows).toEqual(rows.map(row => row % 64));
        expect(result.counts).toEqual(new Array(64).fill(pointCount / 64));
        expect(result.candidateCount).toBe(pointCount);
        expect(result.overflow).toBe(0);
        console.log('PAGED-QUERY-SORT', {candidateTileCapacity, querySort, milliseconds});
      } finally {
        run.destroy();
      }
    }
  }
});

it('predicate pages above 256 slots publish CSR, distances and matrices through repeated encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const leftCount = 521;
  const pairCapacity = 513;
  // Every point is strictly interior to two coincident unit squares: a Cartesian product oracle.
  const positions = Float32Array.from(
    Array.from({length: leftCount}, (_, index) => [
      (((index * 73) % leftCount) + 0.5) / 1024,
      0.5
    ]).flat()
  );
  const fixture = createGeometryFixture(device, {
    inputs: {
      points: {values: positions, format: 'float32x2'},
      vertices: {
        values: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]),
        format: 'float32x2'
      },
      features: {values: new Uint32Array([0, 1, 2]), format: 'uint32'},
      polygons: {values: new Uint32Array([0, 1, 2]), format: 'uint32'},
      rings: {values: new Uint32Array([0, 4, 8]), format: 'uint32'}
    },
    outputs: {
      leftIds: {format: 'uint32', length: pairCapacity},
      rightIds: {format: 'uint32', length: pairCapacity},
      offsets: {format: 'uint32', length: leftCount + 1},
      neighbors: {format: 'uint32', length: pairCapacity},
      weights: {format: 'float32', length: pairCapacity},
      distances: {format: 'float32', length: pairCapacity},
      matrices: {format: 'uint32', length: pairCapacity},
      count: {format: 'uint32', length: 1},
      required: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidates: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) => {
      const words = (name: string) => outputs[name] as GraphDataView<'uint32'>;
      return new GPUSpatialPredicateJoin({
        left: {kind: 'points', positions: inputs['points'] as GraphDataView<'float32x2'>},
        right: {
          kind: 'polygons',
          positions: inputs['vertices'] as GraphDataView<'float32x2'>,
          featureOffsets: inputs['features'] as GraphDataView<'uint32'>,
          polygonOffsets: inputs['polygons'] as GraphDataView<'uint32'>,
          ringOffsets: inputs['rings'] as GraphDataView<'uint32'>
        },
        predicate: 'coveredBy',
        spatialContext: {coordinateSpace: 'planar', metric: 'native'},
        candidateCapacity: leftCount * 2,
        candidateTileCapacity: 257,
        output: {
          pairs: {
            leftIds: words('leftIds'),
            rightIds: words('rightIds'),
            status: {
              count: words('count'),
              requiredCount: words('required'),
              overflow: words('overflow')
            }
          },
          weights: {
            offsets: words('offsets'),
            neighbors: words('neighbors'),
            weights: outputs['weights'] as GraphDataView<'float32'>,
            distances: outputs['distances'] as GraphDataView<'float32'>
          },
          relate: words('matrices'),
          status: {candidateCount: words('candidates')}
        }
      });
    }
  });
  try {
    for (const populated of [true, false, true]) {
      fixture.writeInput('points', populated ? positions : new Float32Array(leftCount * 2).fill(5));
      const result = await fixture.run();
      expect(result['count']).toEqual([populated ? pairCapacity : 0]);
      expect(result['required']).toEqual([populated ? leftCount * 2 : 0]);
      expect(result['candidates']).toEqual([populated ? leftCount * 2 : 0]);
      expect(result['overflow']).toEqual([populated ? 1 : 0]);
      expect(result['offsets']).toEqual(
        Array.from({length: leftCount + 1}, (_, row) =>
          populated ? Math.min(row * 2, pairCapacity) : 0
        )
      );
      if (populated) {
        expect(result['leftIds']).toEqual(
          Array.from({length: pairCapacity}, (_, slot) => Math.floor(slot / 2))
        );
        expect(result['rightIds']).toEqual(
          Array.from({length: pairCapacity}, (_, slot) => slot % 2)
        );
        expect(result['neighbors']).toEqual(
          Array.from({length: pairCapacity}, (_, slot) => slot % 2)
        );
        expect(result['weights']).toEqual(new Array(pairCapacity).fill(1));
        expect(result['distances']).toEqual(new Array(pairCapacity).fill(0));
        // II is a point, EI an area, EB a line, EE an area; the point has no boundary.
        expect(result['matrices'].map(formatGPUSpatialRelate)).toEqual(
          new Array(pairCapacity).fill('0FFFFF212')
        );
        expect(result['matrices'].every(value => (value & 0x80000000) === 0)).toBe(true);
      }
    }
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});
