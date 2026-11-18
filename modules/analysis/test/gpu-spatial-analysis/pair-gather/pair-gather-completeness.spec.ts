// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUPairGather} from '../../../src/gpu-spatial-analysis/pair-gather/index';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/index';
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
          rowCount: rowCount.view,
          status: {overflow: overflow.view}
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
              rowCount: outputs['rowCount'] as never,
              status: {
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
