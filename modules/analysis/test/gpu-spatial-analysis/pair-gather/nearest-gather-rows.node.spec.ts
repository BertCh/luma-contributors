// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {getStorageTopKNodeRecord} from '../../../src/analysis-core/kernels/storage-backed-top-k';
import {
  GPUPairGather,
  type GPUPairGatherNeighbors
} from '../../../src/gpu-spatial-analysis/pair-gather/index';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const sourceIds of [undefined, [1, 0], [100, 200]]) {
  it(`nearest publisher to gather keeps rows separate from IDs ${sourceIds}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const integers = (name: string, length = 1) =>
      createTransientView(graph, name, 'uint32', length);
    const neighborIds = integers('neighbor-ids', 3);
    const neighborRows = integers('neighbor-rows', 3);
    const neighborCounts = integers('neighbor-counts', 3);
    const neighborDistances = createTransientView(graph, 'distances', 'float32', 3);
    const join = new GPUNearestFeatureJoin({
      spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'},
      queries: {kind: 'points', positions: createTransientView(graph, 'queries', 'float32x2', 3)},
      features: {
        kind: 'points',
        positions: createTransientView(graph, 'features', 'float32x2', 2),
        ...(sourceIds ? {sourceIds: integers('feature-ids', 2)} : {})
      },
      output: {
        neighborIds,
        neighborRows,
        neighborCounts,
        neighborDistances,
        status: {overflow: integers('join-overflow')}
      }
    });
    try {
      const columnNode = join.getCommandNodes(graph).find(node => node.id.endsWith('-columns'))!;
      const record = getStorageTopKNodeRecord(columnNode)!;
      const stride = Number(/const STRIDE: u32 = (\d+)u/.exec(record.source)![1]);
      const topk = new Uint32Array(stride * 3 + 3);
      // Hand-computed traversal results: query 0 wins feature row 0, query 1 wins row 1,
      // query 2 has no match. This executes the production publisher, not a CPU rewrite.
      topk[0] = 1;
      topk[3] = 0;
      topk[stride] = 1;
      topk[stride + 3] = 1;
      topk[stride * 3 + 2] = 3;
      const buffers: Record<string, Uint32Array | Float32Array> = {
        topk,
        featureIds: Uint32Array.from(sourceIds ?? [0, 1]),
        neighborIds: new Uint32Array(3).fill(37),
        neighborRows: new Uint32Array(3).fill(37),
        neighborCounts: new Uint32Array(3).fill(37),
        neighborDistances: new Float32Array(3).fill(37),
        overflow: new Uint32Array(1)
      };
      // The interpreter needs scalar storage for a uniform vec4 array. Only the layout changes;
      // every production publisher statement and dispatch setting is retained.
      const source = record.source
        .replace(/^\s*_ = &\w+;$/gm, '')
        .replace(
          'var<uniform> selectionSettings: array<vec4<u32>, 3>',
          'var<storage, read> selectionSettings: array<u32>'
        )
        .replace(
          /selectionSettings\[(\d+)\]\.([xyzw])/g,
          (_, row, component) => `selectionSettings[${Number(row) * 4 + 'xyzw'.indexOf(component)}]`
        );
      const bindings = Object.fromEntries(
        record.bindings.map((binding, index) => [index, buffers[binding.name]])
      );
      bindings[record.bindings.length] = record.settings;
      new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
        0: bindings
      });

      const gather = new GPUPairGather({
        neighbors: {rows: neighborRows, counts: neighborCounts, capacity: 1},
        rightColumns: [{source: integers('attributes', 2), output: integers('gathered', 3)}],
        output: {
          leftRows: integers('left-rows', 3),
          rightRows: integers('right-rows', 3),
          status: {count: integers('row-count'), overflow: integers('gather-overflow')}
        }
      });
      const gatherNodes = gather.getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(
        gatherNodes.filter(node => /-(rows|right-column-0)$/.test(node.id))
      );
      const gatherBuffers: Record<string, Uint32Array | Float32Array> = {
        matchedIn: Uint32Array.of(2),
        slotIds: Uint32Array.of(0, 1, 0),
        neighborIds: buffers['neighborIds'],
        neighborRows: buffers['neighborRows'],
        leftOut: new Uint32Array(3),
        rightOut: new Uint32Array(3),
        sourceIn: Uint32Array.of(10, 20),
        columnOut: new Uint32Array(3)
      };
      gatherBuffers['rowsIn'] = gatherBuffers['rightOut'];
      for (const kernel of kernels) {
        const kernelBindings = Object.fromEntries(
          new WgslReflect(kernel.source).storage.map(declaration => [
            declaration.binding,
            gatherBuffers[declaration.name]
          ])
        );
        new WgslExec(
          new WgslParser().parse(kernel.source.replace(/^\s*_ = &\w+;$/gm, ''))
        ).dispatchWorkgroups('main', [1, 1, 1], {0: kernelBindings}, {constants: kernel.constants});
      }
      expect(Array.from(gatherBuffers['columnOut'])).toEqual([10, 20, 0xffffffff]);
      expect(Array.from(gatherBuffers['rightOut'])).toEqual([0, 1, 0xffffffff]);
      expect(Array.from(buffers['neighborIds'])).toEqual([...(sourceIds ?? [0, 1]), 0xffffffff]);
      expect(Array.from(buffers['neighborRows'])).toEqual([0, 1, 0xffffffff]);
      expect(Array.from(buffers['neighborCounts'])).toEqual([1, 1, 0]);
      expect(Array.from(buffers['neighborDistances'])).toEqual([0, 0, -1]);
    } finally {
      device.destroy();
    }
  });
}

it('nearest row outputs validate length, aliasing and neighbors mode at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const integers = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const props = {
    id: 'nearest-row-validation',
    spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'} as const,
    queries: {
      kind: 'points' as const,
      positions: createTransientView(graph, 'queries', 'float32x2', 2)
    },
    features: {
      kind: 'points' as const,
      positions: createTransientView(graph, 'features', 'float32x2', 2)
    },
    output: {
      neighborIds: integers('ids', 2),
      neighborRows: integers('rows', 2),
      neighborCounts: integers('counts', 2),
      status: {overflow: integers('overflow')}
    }
  };
  try {
    expect(() => new GPUNearestFeatureJoin(props)).not.toThrow();
    expect(
      () =>
        new GPUNearestFeatureJoin({
          ...props,
          output: {...props.output, neighborRows: integers('short', 1)}
        })
    ).toThrow(/nearest-row-validation.*output.neighborRows needs 2 rows/);
    expect(
      () =>
        new GPUNearestFeatureJoin({
          ...props,
          output: {...props.output, neighborRows: props.output.neighborIds}
        })
    ).toThrow(/nearest-row-validation.*overlap|nearest-row-validation.*disjoint/);
    expect(
      () =>
        new GPUNearestFeatureJoin({
          ...props,
          output: {...props.output, neighborIds: undefined}
        })
    ).toThrow(/nearest-row-validation.*output.neighborRows requires output.neighborIds/);
  } finally {
    device.destroy();
  }
});

it('pair gather refuses ambiguous legacy IDs even through a spread or cast', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const integers = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
  const rows = integers('neighbors', 2);
  const counts = integers('counts', 2);
  const output = {
    leftRows: integers('left', 2),
    rightRows: integers('right', 2),
    status: {count: integers('count'), overflow: integers('overflow')}
  };
  try {
    for (const neighbors of [
      {ids: rows, counts, capacity: 1},
      {ids: rows, rows, counts, capacity: 1}
    ]) {
      expect(
        () =>
          new GPUPairGather({
            id: 'gather-row-validation',
            neighbors: neighbors as GPUPairGatherNeighbors,
            output
          })
      ).toThrow(
        /gather-row-validation.*neighbors.rows is required and neighbors.ids is unsupported/
      );
    }
    expect(
      () =>
        new GPUPairGather({
          id: 'gather-row-validation',
          neighbors: {rows, counts, capacity: 2},
          output
        })
    ).toThrow(/gather-row-validation.*neighbors.rows length must equal counts length \* capacity/);
  } finally {
    device.destroy();
  }
});
