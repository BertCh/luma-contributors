// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {addPatchProximityRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/patch-proximity';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {expectSpatialAnalysisWGSL} from '../geographic-distribution/spatial-analysis-wgsl-test-utils';
import {PATCH_PROXIMITY_DISTANCE_WGSL} from '../../../src/gpu-spatial-analysis/landscape-ecology/patch-proximity-distance';

function context(patchCount = 8193) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    mode: 'boundary' as const,
    patches: createTransientView(graph, 'patches', 'uint32', 2 * patchCount - 1),
    patchClasses: createTransientView(graph, 'classes', 'uint32', patchCount),
    width: 2 * patchCount - 1,
    height: 1,
    cellSize: [1, 1] as const,
    edgeDepth: 0,
    searchRadius: 1
  };
  return {device, graph, props};
}

it('8,193 patches use bounded boundary scratch, audited identifiers and no patch-by-cell or pair matrix', () => {
  const {device, graph, props} = context();
  try {
    const result = addPatchProximityRecipe(graph, props);
    const state = graph as unknown as {
      nodes: {id: string}[];
      buffers: Map<string, {byteLength: number}>;
    };
    expect(state.nodes.length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    expect('fields' in result.intermediates).toBe(false);
    expect('distances' in result.outputs).toBe(false);
    const maximum = Math.max(...[...state.buffers.values()].map(buffer => buffer.byteLength));
    expect(maximum).toBeLessThanOrEqual(props.patches.length * 8 * 12);
    for (const kernel of getGPUCommandNodeKernelSources(state.nodes)) {
      expectSpatialAnalysisWGSL(kernel.source, kernel.id);
      expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
    const terms = getGPUCommandNodeKernelSources(state.nodes).find(kernel =>
      kernel.id.endsWith('-pair-terms')
    )!;
    expect(terms.source).toContain('getScaledQuotient(area,distance,distance,0)');
    expect(PATCH_PROXIMITY_DISTANCE_WGSL).not.toContain('displacement/scale');
    expect(PATCH_PROXIMITY_DISTANCE_WGSL).toContain(
      'getScaledQuotient(displacement.x,scale,1.0,0)'
    );
  } finally {
    device.destroy();
  }
});

it('reference proximity audits finite literals and preflights its dense nested scratch atomically', () => {
  const {device, graph, props} = context(3);
  try {
    const {mode: unusedMode, ...reference} = props;
    void unusedMode;
    const state = graph as unknown as {nodes: {id: string}[]; buffers: Map<string, unknown>};
    const before = state.buffers.size;
    for (const overrides of [{edgeDepth: 1e100}, {cellSize: [1e100, 1] as const}]) {
      expect(() => addPatchProximityRecipe(graph, {...reference, ...overrides})).toThrow(/finite/);
      expect(state.buffers.size).toBe(before);
      expect(state.nodes).toHaveLength(0);
    }
    Object.assign(device.limits, {maxStorageBufferBindingSize: 48});
    expect(() => addPatchProximityRecipe(graph, reference)).toThrow(
      /distance fields needs 60 bytes/
    );
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 134217728});
    addPatchProximityRecipe(graph, {...reference, edgeDepth: 1e12, coreModel: 'raster-erosion'});
    for (const kernel of getGPUCommandNodeKernelSources(state.nodes)) {
      expectSpatialAnalysisWGSL(kernel.source, kernel.id);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'boundary',
  'reference'
] as const)('%s reconciliation rejects aligned relabel/core aliasing before allocation', mode => {
  const {device, graph, props} = context(2);
  try {
    const buffer = graph.createTransientBuffer({
      id: 'aligned-reconciled',
      byteLength: 256,
      usage: Buffer.STORAGE
    });
    const core = graph.createDataView(buffer, {format: 'uint32', length: 3});
    const reconciledPatches = graph.createDataView(buffer, {
      format: 'uint32',
      byteOffset: 12,
      length: 3
    });
    const {mode: unusedMode, ...common} = props;
    void unusedMode;
    const state = graph as unknown as {nodes: unknown[]; buffers: Map<string, unknown>};
    const before = state.buffers.size;
    const run = () =>
      mode === 'boundary'
        ? addPatchProximityRecipe(graph, {
            ...common,
            mode,
            reconcilePatches: true,
            output: {core, reconciledPatches}
          })
        : addPatchProximityRecipe(graph, {
            ...common,
            mode,
            reconcilePatches: true,
            output: {core, reconciledPatches}
          });
    expect(run).toThrow(/aligned binding byte ranges must not overlap/);
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('boundary proximity rejects every bad shape, numeric bound and STORAGE view before allocation', () => {
  const {device, graph, props} = context(3);
  try {
    const bad = graph.createTransientBuffer({id: 'bad', byteLength: 256, usage: Buffer.COPY_DST});
    const before = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
    for (const overrides of [
      {candidateCapacity: 0},
      {searchRadius: 1e100},
      {cellSize: [Number.MIN_VALUE, 1] as const},
      {candidateRadius: 0},
      {edgeDepth: 1e100},
      {output: {metrics: createTransientView(graph, 'short-metrics', 'float32x4', 1)}},
      {patches: graph.createDataView(bad, {format: 'uint32', length: props.patches.length})}
    ]) {
      // The short caller view is the only resource deliberately declared by the test itself.
      const resourceCount = (graph as unknown as {buffers: Map<string, unknown>}).buffers.size;
      expect(() => addPatchProximityRecipe(graph, {...props, ...overrides})).toThrow();
      expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(
        resourceCount
      );
      expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
    }
    expect((graph as unknown as {buffers: Map<string, unknown>}).buffers.size).toBe(before + 1);
  } finally {
    device.destroy();
  }
});

it('boundary proximity permits packed outputs of separate dispatches and ignores unused reconciliation outputs', () => {
  const {device, graph, props} = context(2);
  try {
    const storage = graph.createTransientBuffer({
      id: 'packed-output',
      byteLength: 2048,
      usage: Buffer.STORAGE
    });
    const words = (byteOffset: number, length = 1) =>
      graph.createDataView(storage, {format: 'uint32', byteOffset, length});
    const unused = graph.createTransientBuffer({
      id: 'unused',
      byteLength: 256,
      usage: Buffer.COPY_DST
    });
    expect(() =>
      addPatchProximityRecipe(graph, {
        ...props,
        output: {
          core: words(0, 3),
          metrics: graph.createDataView(storage, {format: 'float32x4', byteOffset: 12, length: 2}),
          validity: words(256, 2),
          status: {
            count: words(512),
            requiredCount: words(516),
            overflow: words(768),
            candidateOverflow: words(520)
          },
          patchParents: graph.createDataView(unused, {format: 'uint32', length: 2})
        }
      })
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
