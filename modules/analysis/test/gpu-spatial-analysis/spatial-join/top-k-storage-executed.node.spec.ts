// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, getViewBinding} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {createGPUComputeCommandNode} from '@luma.gl/graph/gpu-core';
import {getStorageTopKNodeRecord} from '../../../src/analysis-core/kernels/storage-backed-top-k';
import {describe, expect, it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUInverseDistanceWeighting} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {getNearestNeighborNodes} from '../../../src/gpu-spatial-analysis/spatial-join/nearest-nodes';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';
import {assertValidCrossWeights} from './nearest-feature-weights-oracle';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

/** Supply the same settings through scalar storage for the interpreter's uniform/vector limitation.
 * The generated source, comparisons and dispatch remain unchanged apart from this layout adapter.
 */
function executeTopKNodes(
  nodes: readonly object[],
  memory: ReturnType<typeof createCPUGraphMemory>
) {
  for (const node of nodes) {
    const record = getStorageTopKNodeRecord(node);
    if (!record) {
      executeCommandNodesOnCPU([node], memory);
      continue;
    }
    const graph = record.bindings[0].view.buffer.graph as GPUCommandGraph;
    const settings = createTransientView(
      graph,
      `${(node as {id: string}).id}-interpreter-settings`,
      'uint32',
      12
    );
    const parameterWords = record.settings.slice();
    // wgsl_reflect loses nonzero typed-array byteOffsets. Supply equivalent logical read views
    // at offset zero; recording tests separately verify the real runtime offsets and pipelines.
    const recordedBindings = record.bindings.map((binding, bindingIndex) => {
      if (binding.access !== 'read' || binding.view.byteOffset === 0) return binding;
      const copy = createTransientView(
        graph,
        `${(node as {id: string}).id}-read-${bindingIndex}`,
        'uint32',
        (binding.view.length * binding.view.rowByteLength) / 4
      );
      memory.write(copy, memory.readUint32(binding.view));
      const buffer = memory.getBuffer(copy.buffer);
      buffer.backing = buffer.backing.slice(0, buffer.byteLength);
      parameterWords[bindingIndex + 1] = 0;
      return {...binding, view: copy};
    });
    memory.write(settings, parameterWords);
    const source = record.source
      .replace(
        'var<uniform> selectionSettings: array<vec4<u32>, 3>',
        'var<storage, read> selectionSettings: array<u32>'
      )
      .replace(
        /selectionSettings\[(\d+)\]\.([xyzw])/g,
        (_, row, component) => `selectionSettings[${Number(row) * 4 + 'xyzw'.indexOf(component)}]`
      );
    const bindings = [
      ...recordedBindings,
      {name: 'selectionSettings', view: settings, type: 'u32' as const, access: 'read' as const}
    ];
    const executable = createGPUComputeCommandNode({
      id: (node as {id: string}).id,
      resources: bindings.map(binding => ({
        buffer: binding.view,
        usage:
          binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
      })),
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          source,
          shaderLayout: {
            bindings: bindings.map((binding, location) => ({
              name: binding.name,
              type:
                binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
              group: 0,
              location
            }))
          }
        });
        return {
          encode: ({computePass, getBuffer}) =>
            kernel.dispatch(computePass, {
              bindings: Object.fromEntries(
                bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
              ),
              x: record.layout.x,
              y: record.layout.y,
              z: record.layout.z
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
    executeCommandNodesOnCPU([executable], memory);
  }
}

describe('storage top-k kernels executed on the WGSL interpreter', () => {
  useCPUExecutorTestHooks();

  it.each([
    false,
    true
  ])('nearest preserves row bases and sorted chunk slices (querySort=%s)', querySort => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `executed-nearest-${serial++}`, format, length);
    const queryCount = 66;
    const capacity = 65;
    const chunks = [view('float32x2', 65), view('float32x2', 1)];
    const positions = view('float32x2', 5);
    const ids = view('uint32', queryCount * capacity);
    const counts = view('uint32', queryCount);
    const distances = view('float32', queryCount * capacity);
    const foot = view('float32x2', queryCount * capacity);
    const overflow = view('uint32', 1);
    const nodes = getNearestNeighborNodes(graph, {
      id: 'executed-nearest',
      operation: 'GPUNearestFeatureJoin',
      queries: {kind: 'points', points: createVectorView('executed-queries', 'float32x2', chunks)},
      queryCount,
      features: {kind: 'points', positions},
      featureCount: 5,
      k: capacity,
      capacity,
      leafCapacity: 8,
      spatialSort: false,
      querySort,
      scratchChunkQueryCount: 64,
      neighborIds: ids,
      neighborCounts: counts,
      neighborDistances: distances,
      neighborFootPoints: foot,
      overflow
    }).nodes;
    const memory = createCPUGraphMemory();
    const features = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
      [3, 0]
    ];
    const queries = Array.from({length: queryCount}, (_, row) => [row % 11, 0]);
    memory.write(positions, Float32Array.from(features.flat()));
    memory.write(chunks[0], Float32Array.from(queries.slice(0, 65).flat()));
    memory.write(chunks[1], Float32Array.from(queries[65]));
    // The interpreter's arrayLength reads the whole backing ArrayBuffer, including the executor's
    // 16 guard bytes. Trim those read-only query buffers to their actual WebGPU binding length.
    for (const chunk of chunks) {
      const buffer = memory.getBuffer(chunk.buffer);
      buffer.backing = buffer.backing.slice(0, buffer.byteLength);
    }
    executeTopKNodes(nodes, memory);
    const actualIds = memory.readUint32(ids);
    const actualDistances = memory.readFloat32(distances);
    const actualFoot = memory.readFloat32(foot);
    expect(Array.from(memory.readUint32(counts))).toEqual(Array(queryCount).fill(features.length));
    for (const [row, query] of queries.entries()) {
      const expected = features
        .map((point, featureRow) => ({
          point,
          featureRow,
          distance: Math.hypot(point[0] - query[0], point[1] - query[1])
        }))
        .sort(
          (first, second) =>
            first.distance - second.distance || first.featureRow - second.featureRow
        );
      for (let slot = 0; slot < capacity; slot++) {
        const offset = row * capacity + slot;
        expect(actualIds[offset]).toBe(expected[slot]?.featureRow ?? 0xffffffff);
        expect(actualDistances[offset]).toBeCloseTo(expected[slot]?.distance ?? -1, 5);
        if (slot < expected.length) {
          expect(Array.from(actualFoot.slice(offset * 2, offset * 2 + 2))).toEqual(
            expected[slot].point
          );
        }
      }
    }
    expect(memory.readUint32(overflow)[0]).toBe(0);
    device.destroy();
  }, 30000);

  it('IDW storage gather reuses two scratch rows over five output cells against a full-sort mean', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `executed-idw-${serial++}`, format, length);
    const positions = view('float32x2', 70);
    const values = view('float32', 70);
    const parameters = view('float32', 9);
    const output = {values: view('float32', 5), counts: view('uint32', 5)};
    const nodes = new GPUInverseDistanceWeighting({
      positions,
      values,
      parameters,
      maximumNeighborCount: 65,
      invocationTileSize: 2,
      grid: {width: 5, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
      indexGridSize: [1, 1],
      indexBounds: [0, 0, 32, 32],
      output
    }).getCommandNodes(graph);
    const gatherNodes = nodes.filter(node =>
      /-(reset-cursor|gather-\d+|publish-\d+|advance-\d+)$/.test(node.id)
    );
    const findView = (suffix: string) => {
      const resource = gatherNodes
        .flatMap(node => node.resources ?? [])
        .find(
          use => 'buffer' in use && 'buffer' in use.buffer && use.buffer.buffer.id.endsWith(suffix)
        );
      if (!resource || !('buffer' in resource)) throw new Error(`Missing ${suffix}`);
      return resource.buffer as GraphDataView;
    };
    const memory = createCPUGraphMemory();
    const points = Array.from({length: 70}, (_, row) => [
      10 + (row % 10),
      10 + Math.floor(row / 10)
    ]);
    memory.write(positions, Float32Array.from(points.flat()));
    memory.write(
      values,
      Float32Array.from({length: 70}, (_, row) => row)
    );
    // Power zero gives the exact arithmetic mean of the 65 nearest rows.
    memory.write(parameters, Float32Array.of(0, 0, 5, 1, Infinity, 0, 65, 1, 0));
    memory.write(findView('-cell-offsets'), Uint32Array.of(0, 70));
    memory.write(
      findView('-sorted-rows'),
      Uint32Array.from({length: 70}, (_, row) => row)
    );
    executeTopKNodes(gatherNodes, memory);
    const expected = Array.from(
      {length: 5},
      (_, column) =>
        points
          .map((point, row) => ({
            row,
            distanceSquared: (point[0] - column - 0.5) ** 2 + (point[1] - 0.5) ** 2
          }))
          .sort(
            (first, second) =>
              first.distanceSquared - second.distanceSquared || first.row - second.row
          )
          .slice(0, 65)
          .reduce((sum, entry) => sum + entry.row, 0) / 65
    );
    expect(Array.from(memory.readUint32(output.counts))).toEqual([65, 65, 65, 65, 65]);
    memory
      .readFloat32(output.values)
      .forEach((value, row) => expect(value).toBeCloseTo(expected[row], 4));
    device.destroy();
  }, 30000);

  it('weights sorts the full kept set before truncating the output prefix', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 524});
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `executed-weights-${serial++}`, format, length);
    const ids = view('uint32', 130);
    const counts = view('uint32', 2);
    const distances = view('float32', 130);
    const output = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 70),
      weights: view('float32', 70),
      distances: view('float32', 70),
      status: {overflow: view('uint32', 1)}
    };
    const contributor = new GPUNearestFeatureWeights({
      slotCapacity: 65,
      neighborIds: ids,
      neighborCounts: counts,
      neighborDistances: distances,
      weightType: 'inverse-distance',
      output
    });
    const memory = createCPUGraphMemory();
    const inputIds = Uint32Array.from({length: 130}, (_, slot) => 65 - (slot % 65));
    memory.write(ids, inputIds);
    memory.write(
      distances,
      Float32Array.from(inputIds, (neighbor, slot) => neighbor + (slot < 65 ? 10 : 20))
    );
    memory.write(counts, Uint32Array.of(65, 65));
    executeTopKNodes(contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(output.offsets))).toEqual([0, 65, 70]);
    expect(Array.from(memory.readUint32(output.neighbors))).toEqual([
      ...Array.from({length: 65}, (_, row) => row + 1),
      1,
      2,
      3,
      4,
      5
    ]);
    expect(Array.from(memory.readFloat32(output.distances))).toEqual([
      ...Array.from({length: 65}, (_, row) => row + 11),
      21,
      22,
      23,
      24,
      25
    ]);
    memory
      .readFloat32(output.weights)
      .forEach((weight, slot) =>
        expect(weight).toBeCloseTo(1 / (slot < 65 ? slot + 11 : slot - 65 + 21), 6)
      );
    expect(memory.readUint32(output.status.overflow)[0]).toBe(1);
    assertValidCrossWeights(
      {
        offsets: Array.from(memory.readUint32(output.offsets)),
        neighbors: Array.from(memory.readUint32(output.neighbors)),
        distances: Array.from(memory.readFloat32(output.distances)),
        weights: Array.from(memory.readFloat32(output.weights))
      },
      2,
      66,
      'executed-cross-weights'
    );
    device.destroy();
  }, 30000);
});
