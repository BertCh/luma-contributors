// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPairGather} from '../../../src/gpu-spatial-analysis/pair-gather/index';
import {
  GPUSpatialPredicateJoin,
  GPUNearestFeatureJoin
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {runExample} from '../../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example';
import {getPairGatherCatalogProps} from '../../utils/join-example-constructors';

it('final2 G2/G1 catalog gather composes recorded candidate status and inherited requiredCount', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'final2-catalog-gather');
  try {
    const polygon = (name: string, count: number) => ({
      kind: 'polygons' as const,
      positions: fixture.input(
        `${name}-positions`,
        Float32Array.from(Array.from({length: count}, () => [0, 0, 1, 0, 1, 1, 0, 1]).flat()),
        'float32x2',
        count * 4
      ),
      featureOffsets: fixture.input(
        `${name}-features`,
        Uint32Array.from({length: count + 1}, (_, index) => index),
        'uint32',
        count + 1
      ),
      polygonOffsets: fixture.input(
        `${name}-polygons`,
        Uint32Array.from({length: count + 1}, (_, index) => index),
        'uint32',
        count + 1
      ),
      ringOffsets: fixture.input(
        `${name}-rings`,
        Uint32Array.from({length: count + 1}, (_, index) => index * 4),
        'uint32',
        count + 1
      )
    });
    const requiredCount = fixture.output('pair-required', 'uint32', 1);
    requiredCount.buffer.write(Uint32Array.of(77));
    const pairs = {
      leftRows: fixture.output('pair-left', 'uint32', 2).view,
      rightRows: fixture.output('pair-right', 'uint32', 2).view,
      status: Object.assign(
        Object.create({
          get requiredCount() {
            return requiredCount.view;
          }
        }),
        {
          count: fixture.output('pair-count', 'uint32', 1).view,
          overflow: fixture.output('pair-overflow', 'uint32', 1).view,
          candidateOverflow: fixture.output('pair-candidate', 'uint32', 1).view
        }
      )
    };
    await runExample(fixture.graph, {
      parks: polygon('parks', 1),
      tracts: polygon('tracts', 2),
      pairs,
      parkAreas: fixture.input('areas', Float32Array.of(5), 'float32', 1),
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      candidateCapacity: 1
    });
    const catalogOutputs = new Map<string, {buffer: Buffer}>();
    const props = getPairGatherCatalogProps({
      pairs,
      capacity: 2,
      incomeData: fixture.input('income-source', Float32Array.of(10, 20), 'float32', 2),
      output: (id: string, format: 'uint32' | 'float32', length: number) => {
        const value = fixture.output(`catalog-${id}`, format, length);
        catalogOutputs.set(id, value);
        return value.view;
      }
    });
    fixture.graph.add(new GPUPairGather({...props, id: 'catalog-gather'}));
    fixture.run();
    expect(await fixture.readUint32(requiredCount, 1)).toEqual([1]);
    expect(await fixture.readUint32(catalogOutputs.get('count')!, 1)).toEqual([1]);
    expect(await fixture.readUint32(catalogOutputs.get('overflow')!, 1)).toEqual([0]);
    expect(await fixture.readUint32(catalogOutputs.get('candidate-overflow')!, 1)).toEqual([1]);
    expect((await fixture.readFloat32(catalogOutputs.get('income')!, 2))[0]).toBe(10);
  } finally {
    fixture.destroy();
  }
});

it('final2 G1 inherited left mode emits one unmatched row with zero neighbors', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'final2-left-gather');
  try {
    let mode = 'left';
    let reads = 0;
    const neighbors = {
      rows: fixture.input('neighbor-rows', Uint32Array.of(0xffffffff), 'uint32', 1),
      counts: fixture.input('neighbor-counts', Uint32Array.of(0), 'uint32', 1),
      capacity: 1
    };
    const leftRows = fixture.output('left-rows', 'uint32', 1);
    const rightRows = fixture.output('right-rows', 'uint32', 1);
    const rowCount = fixture.output('row-count', 'uint32', 1);
    const overflow = fixture.output('overflow', 'uint32', 1);
    const value = fixture.output('value', 'float32', 1);
    const props = Object.assign(
      Object.create({
        get how() {
          reads++;
          return mode;
        }
      }),
      {
        neighbors,
        leftColumns: [
          {
            source: fixture.input('attribute', Float32Array.of(42), 'float32', 1),
            output: value.view
          }
        ],
        output: {
          leftRows: leftRows.view,
          rightRows: rightRows.view,
          status: {count: rowCount.view, overflow: overflow.view}
        }
      }
    );
    const gather = new GPUPairGather(props);
    mode = 'inner';
    neighbors.capacity = 0;
    fixture.graph.add(gather);
    fixture.run();
    expect(reads).toBe(1);
    expect(await fixture.readUint32(rowCount, 1)).toEqual([1]);
    expect(await fixture.readUint32(leftRows, 1)).toEqual([0]);
    expect(await fixture.readUint32(rightRows, 1)).toEqual([0xffffffff]);
    expect(await fixture.readUint32(overflow, 1)).toEqual([0]);
    expect(await fixture.readFloat32(value, 1)).toEqual([42]);
  } finally {
    fixture.destroy();
  }
});

it('F4 public example allocates an omitted candidate flag for two true matches and one candidate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, 'example-missing-candidate');
  try {
    const polygon = (name: string, count: number) => ({
      kind: 'polygons' as const,
      positions: fixture.input(
        `${name}-positions`,
        new Float32Array(Array.from({length: count}, () => [0, 0, 1, 0, 1, 1, 0, 1]).flat()),
        'float32x2',
        count * 4
      ),
      featureOffsets: fixture.input(
        `${name}-features`,
        Uint32Array.from({length: count + 1}, (_, index) => index),
        'uint32',
        count + 1
      ),
      polygonOffsets: fixture.input(
        `${name}-polygons`,
        Uint32Array.from({length: count + 1}, (_, index) => index),
        'uint32',
        count + 1
      ),
      ringOffsets: fixture.input(
        `${name}-rings`,
        Uint32Array.from({length: count + 1}, (_, index) => index * 4),
        'uint32',
        count + 1
      )
    });
    const requiredCount = fixture.output('pair-required', 'uint32', 1);
    requiredCount.buffer.write(new Uint32Array([77]));
    const result = await runExample(fixture.graph, {
      parks: polygon('parks', 1),
      tracts: polygon('tracts', 2),
      pairs: {
        leftRows: fixture.output('left', 'uint32', 2).view,
        rightRows: fixture.output('right', 'uint32', 2).view,
        status: {
          count: fixture.output('count', 'uint32', 1).view,
          overflow: fixture.output('pair-overflow', 'uint32', 1).view,
          get requiredCount() {
            return requiredCount.view;
          }
        }
      },
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      parkAreas: fixture.input('areas', new Float32Array([1]), 'float32', 1),
      candidateCapacity: 1
    });
    const status = fixture.output('status', 'uint32', 3);
    fixture.graph.add(
      createWGSLKernelNode(fixture.graph, {
        id: 'capture-example-status',
        operation: 'review',
        invocationCount: 1,
        bindings: [
          {name: 'count', view: result.rowCount, type: 'u32', access: 'read'},
          {name: 'overflow', view: result.overflow, type: 'u32', access: 'read'},
          {name: 'candidate', view: result.candidateOverflow, type: 'u32', access: 'read'},
          {name: 'status', view: status.view, type: 'u32', access: 'read_write'}
        ],
        body: 'status[statusOffset] = count[countOffset]; status[statusOffset + 1u] = overflow[overflowOffset]; status[statusOffset + 2u] = candidate[candidateOffset];'
      })
    );
    fixture.run();
    expect(await fixture.readUint32(status, 3)).toEqual([1, 0, 1]);
    expect(await fixture.readUint32(requiredCount, 1)).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});

for (const matchCount of [2, 9]) {
  it(`join to gather preserves candidate loss and clears it on reuse (${matchCount} matches)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        leftPositions: {values: new Float32Array([0, 0]), format: 'float32x2'},
        rightPositions: {values: new Float32Array(matchCount * 2), format: 'float32x2'}
      },
      outputs: {
        leftRows: {format: 'uint32', length: matchCount},
        rightRows: {format: 'uint32', length: matchCount},
        rowCount: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        candidateOverflow: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) => ({
        getCommandNodes(graph) {
          const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
          const pairs = {
            leftRows: createTransientView(graph, 'pair-left', 'uint32', matchCount),
            rightRows: createTransientView(graph, 'pair-right', 'uint32', matchCount),
            status: {
              count: scalar('pair-count'),
              overflow: scalar('pair-overflow'),
              candidateOverflow: scalar('pair-candidate-overflow')
            }
          };
          const join = new GPUSpatialPredicateJoin({
            spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'},
            left: {kind: 'points', positions: inputs['leftPositions'] as never},
            right: {kind: 'points', positions: inputs['rightPositions'] as never},
            predicate: 'intersects',
            ...(matchCount === 2 ? {candidateCapacity: 1} : {}),
            output: {pairs}
          });
          const gather = new GPUPairGather({
            pairs,
            output: {
              leftRows: outputs['leftRows'] as never,
              rightRows: outputs['rightRows'] as never,
              status: {
                count: outputs['rowCount'] as never,
                overflow: outputs['overflow'] as never,
                candidateOverflow: outputs['candidateOverflow'] as never
              }
            }
          });
          return [...join.getCommandNodes(graph), ...gather.getCommandNodes(graph)];
        }
      })
    });
    try {
      const incomplete = await fixture.run();
      const retainedCount = matchCount === 2 ? 1 : 8;
      expect(incomplete['rowCount']).toEqual([retainedCount]);
      expect(incomplete['leftRows'].slice(0, retainedCount)).toEqual(Array(retainedCount).fill(0));
      expect(incomplete['rightRows'].slice(0, retainedCount)).toEqual(
        Array.from({length: retainedCount}, (_, index) => index)
      );
      expect(incomplete['overflow']).toEqual([0]);
      expect(incomplete['candidateOverflow']).toEqual([1]);
      const oneMatch = new Float32Array(matchCount * 2).fill(100);
      oneMatch[0] = 0;
      oneMatch[1] = 0;
      fixture.writeInput('rightPositions', oneMatch);
      const complete = await fixture.run();
      expect(complete['rowCount']).toEqual([1]);
      expect(complete['leftRows'][0]).toBe(0);
      expect(complete['rightRows'][0]).toBe(0);
      expect(complete['overflow']).toEqual([0]);
      expect(complete['candidateOverflow']).toEqual([0]);
      expect(fixture.getCompileCount()).toBe(0);
    } finally {
      fixture.destroy();
    }
  });
}

for (const perQuery of [true, false]) {
  it(`GEO-113 synthetic selective candidate-only loss withholds uncertain unmatched rows and clears on reuse (${perQuery})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        rows: {values: Uint32Array.of(0xffffffff, 0xffffffff), format: 'uint32'},
        counts: {values: Uint32Array.of(0, 0), format: 'uint32'},
        candidate: {values: Uint32Array.of(1), format: 'uint32'},
        producerOverflow: {values: Uint32Array.of(0), format: 'uint32'},
        truncated: {values: Uint32Array.of(1, 0), format: 'uint32'}
      },
      outputs: {
        left: {format: 'uint32', length: 2},
        right: {format: 'uint32', length: 2},
        count: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        candidate: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) =>
        new GPUPairGather({
          how: 'left',
          neighbors: {
            rows: inputs['rows'] as never,
            counts: inputs['counts'] as never,
            capacity: 1,
            status: {
              overflow: inputs['producerOverflow'] as never,
              candidateOverflow: inputs['candidate'] as never
            },
            ...(perQuery ? {candidateTruncated: inputs['truncated'] as never} : {})
          },
          output: {
            leftRows: outputs['left'] as never,
            rightRows: outputs['right'] as never,
            status: {
              count: outputs['count'] as never,
              overflow: outputs['overflow'] as never,
              candidateOverflow: outputs['candidate'] as never
            }
          }
        })
    });
    try {
      // Synthetic gather-port fixture: selective candidate-only loss is not emitted by nearest
      // neighbors mode, whose candidate table is absent and whose BVH loss flags every query.
      // Hand-worked: both counts are zero, but query 0 lost candidates. Only query 1 has
      // established absence; without row flags neither query has established absence.
      const incomplete = await fixture.run();
      expect(incomplete['count']).toEqual([perQuery ? 1 : 0]);
      expect(incomplete['left']).toEqual(perQuery ? [1, 0xffffffff] : [0xffffffff, 0xffffffff]);
      expect(incomplete['right']).toEqual([0xffffffff, 0xffffffff]);
      expect(incomplete['overflow']).toEqual([0]);
      expect(incomplete['candidate']).toEqual([1]);
      fixture.writeInput('candidate', Uint32Array.of(0));
      fixture.writeInput('truncated', Uint32Array.of(0, 0));
      const complete = await fixture.run();
      expect(complete['count']).toEqual([2]);
      expect(complete['left']).toEqual([0, 1]);
      expect(complete['right']).toEqual([0xffffffff, 0xffffffff]);
      expect(complete['overflow']).toEqual([0]);
      expect(complete['candidate']).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  });
}

for (const scenario of [
  {kind: 'bvh', perQuery: true},
  {kind: 'ties', perQuery: true},
  {kind: 'ties', perQuery: false}
] as const) {
  it(`S1 nearest ${scenario.kind} loss reaches gather and exact rows refresh (${scenario.perQuery})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        queries: {format: 'float32x2', values: Float32Array.of(0, 0, 100, 0)},
        features: {
          format: 'float32x2',
          values:
            scenario.kind === 'bvh' ? Float32Array.of(0, 0, 50, 0) : Float32Array.of(-1, 0, 1, 0)
        },
        radius: {format: 'float32', values: Float32Array.of(2)}
      },
      outputs: {
        nearestRows: {format: 'uint32', length: 2},
        nearestCounts: {format: 'uint32', length: 2},
        truncated: {format: 'uint32', length: 2},
        producerOverflow: {format: 'uint32', length: 1},
        producerCandidate: {format: 'uint32', length: 1},
        left: {format: 'uint32', length: 2},
        right: {format: 'uint32', length: 2},
        count: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        candidate: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs}) => ({
        getCommandNodes(graph) {
          const status = {
            overflow: outputs['producerOverflow'] as never,
            candidateOverflow: outputs['producerCandidate'] as never
          };
          const join = new GPUNearestFeatureJoin({
            queries: {kind: 'points', positions: inputs['queries'] as never},
            features: {kind: 'points', positions: inputs['features'] as never},
            radius: inputs['radius'] as never,
            leafCapacity: scenario.kind === 'bvh' ? 1 : 2,
            spatialSort: false,
            neighborCount: 1,
            neighborCapacity: 1,
            ties: 'all',
            output: {
              neighborIds: createTransientView(graph, 'nearest-ids', 'uint32', 2),
              neighborRows: outputs['nearestRows'] as never,
              neighborCounts: outputs['nearestCounts'] as never,
              candidateTruncated: outputs['truncated'] as never,
              status
            }
          });
          const gather = new GPUPairGather({
            how: 'left',
            neighbors: {
              rows: outputs['nearestRows'] as never,
              counts: outputs['nearestCounts'] as never,
              capacity: 1,
              status,
              ...(scenario.perQuery ? {candidateTruncated: outputs['truncated'] as never} : {})
            },
            output: {
              leftRows: outputs['left'] as never,
              rightRows: outputs['right'] as never,
              status: {
                count: outputs['count'] as never,
                overflow: outputs['overflow'] as never,
                candidateOverflow: outputs['candidate'] as never
              }
            }
          });
          return [...join.getCommandNodes(graph), ...gather.getCommandNodes(graph)];
        }
      })
    });
    try {
      const first = await fixture.run();
      // BVH retains only feature 0. Tie mode has equal unit distances to rows 0 and 1;
      // capacity 1 retains row 0. Query 1 is 100 units away, beyond the radius in either case.
      expect(first['nearestRows']).toEqual([0, 0xffffffff]);
      expect(first['nearestCounts']).toEqual([1, 0]);
      expect(first['producerOverflow']).toEqual([1]);
      expect(first['producerCandidate']).toEqual([scenario.kind === 'bvh' ? 1 : 0]);
      expect(first['truncated']).toEqual(scenario.kind === 'bvh' ? [1, 1] : [0, 0]);
      const withUnmatched = scenario.kind === 'ties' && scenario.perQuery;
      expect(first['count']).toEqual([withUnmatched ? 2 : 1]);
      expect(first['left']).toEqual([0, withUnmatched ? 1 : 0xffffffff]);
      expect(first['right']).toEqual([0, 0xffffffff]);
      expect(first['overflow']).toEqual([0]);
      expect(first['candidate']).toEqual([1]);
      if (scenario.kind === 'bvh') {
        fixture.writeInput('queries', Float32Array.of(100, 0, 0, 0));
        const reused = await fixture.run();
        expect(reused['nearestRows']).toEqual([0xffffffff, 0]);
        expect(reused['nearestCounts']).toEqual([0, 1]);
        expect(reused['truncated']).toEqual([1, 1]);
        expect(reused['left']).toEqual([1, 0xffffffff]);
        expect(reused['right']).toEqual([0, 0xffffffff]);
        expect(reused['count']).toEqual([1]);
        expect(reused['producerOverflow']).toEqual([1]);
        expect(reused['producerCandidate']).toEqual([1]);
        expect(reused['overflow']).toEqual([0]);
        expect(reused['candidate']).toEqual([1]);
      } else {
        fixture.writeInput('features', Float32Array.of(-1, 0, 5, 0));
        const complete = await fixture.run();
        expect(complete['nearestRows']).toEqual([0, 0xffffffff]);
        expect(complete['nearestCounts']).toEqual([1, 0]);
        expect(complete['truncated']).toEqual([0, 0]);
        expect(complete['left']).toEqual([0, 1]);
        expect(complete['right']).toEqual([0, 0xffffffff]);
        expect(complete['count']).toEqual([2]);
        expect(complete['producerOverflow']).toEqual([0]);
        expect(complete['producerCandidate']).toEqual([0]);
        expect(complete['overflow']).toEqual([0]);
        expect(complete['candidate']).toEqual([0]);
      }
      expect(fixture.getCompileCount()).toBe(0);
    } finally {
      fixture.destroy();
    }
  });
}

it('S2 truncation-only maximum preserves noncanonical nonzero loss and clears on reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      rows: {format: 'uint32', values: Uint32Array.of(0xffffffff, 0xffffffff)},
      counts: {format: 'uint32', values: Uint32Array.of(0, 0)},
      truncated: {format: 'uint32', values: Uint32Array.of(0, 7)}
    },
    outputs: {
      left: {format: 'uint32', length: 2},
      right: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidate: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs}) =>
      new GPUPairGather({
        how: 'left',
        neighbors: {
          rows: inputs['rows'] as never,
          counts: inputs['counts'] as never,
          capacity: 1,
          candidateTruncated: inputs['truncated'] as never
        },
        output: {
          leftRows: outputs['left'] as never,
          rightRows: outputs['right'] as never,
          status: {
            count: outputs['count'] as never,
            overflow: outputs['overflow'] as never,
            candidateOverflow: outputs['candidate'] as never
          }
        }
      })
  });
  try {
    // Flag 7 still means incomplete: query 0 is established absent, query 1 is unknown.
    const incomplete = await fixture.run();
    expect(incomplete['left']).toEqual([0, 0xffffffff]);
    expect(incomplete['right']).toEqual([0xffffffff, 0xffffffff]);
    expect(incomplete['count']).toEqual([1]);
    expect(incomplete['overflow']).toEqual([0]);
    expect(incomplete['candidate']).toEqual([1]);
    fixture.writeInput('truncated', Uint32Array.of(0, 0));
    const complete = await fixture.run();
    expect(complete['left']).toEqual([0, 1]);
    expect(complete['right']).toEqual([0xffffffff, 0xffffffff]);
    expect(complete['count']).toEqual([2]);
    expect(complete['overflow']).toEqual([0]);
    expect(complete['candidate']).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
