// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createGPUCopyCommandNode,
  createTransientView,
  GPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {
  GPUHuffTradeAreas,
  getGPUHuffTradeAreasParameterValues
} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {GPUGeographicDistribution} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {expectSpatialAnalysisWGSL} from './spatial-analysis-wgsl-test-utils';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/network-topology';
import {getSpatialAnalysisPackedKernelNodes} from '../../../src/gpu-spatial-analysis/geographic-distribution/spatial-analysis-packed-kernel';
import type {GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {addPatchProximityRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/patch-proximity';
import {addBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {addFiniteScaleAnalysisFixture} from '../catchment-accessibility/analysis-finite-ratios-test-fixture';

// NullDevice does not enforce WebGPU's buffer-wide storage usage compatibility. Model that
// validation explicitly, independently of numerical results or the interpreter's WGSL support.
function expectCompatibleStorageUses(graph: GPUCommandGraph, nodes: readonly GPUCommandNode[]) {
  for (const node of nodes) {
    const record = getWGSLKernelNodeRecord(graph, node.id);
    if (!record) continue;
    const modes = new Map<unknown, Set<string>>();
    for (const binding of record.bindings) {
      const buffer = binding.view.buffer.defaultBuffer
        ? getCoreDefaultBuffer(binding.view.buffer.defaultBuffer)
        : binding.view.buffer;
      const uses = modes.get(buffer) ?? new Set<string>();
      uses.add(binding.access);
      modes.set(buffer, uses);
    }
    for (const uses of modes.values()) expect(uses.size, node.id).toBe(1);
    if (/\b(?:workgroupBarrier|storageBarrier)\s*\(/.test(record.source)) {
      // Cooperative kernels guard row accesses inside the body; a generated per-lane return
      // before their barriers is invalid even when the last workgroup is only partly occupied.
      expect(record.source, node.id).not.toContain('if (index >= INVOCATION_COUNT)');
    }
    expectSpatialAnalysisWGSL(record.source, node.id);
  }
}

function context() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const scalar = (length = 2) =>
    createTransientView(graph, `scalar-${serial++}`, 'float32', length);
  const words = (length = 2) => createTransientView(graph, `words-${serial++}`, 'uint32', length);
  const weights = {offsets: words(3), neighbors: words(4), weights: scalar(4)};
  const buffer = graph.createTransientBuffer({
    id: 'packed',
    byteLength: 2048,
    usage: Buffer.STORAGE
  });
  const packedScalar = (byteOffset: number, length = 2) =>
    graph.createDataView(buffer, {format: 'float32', byteOffset, length});
  const packedWords = (byteOffset: number, length = 2) =>
    graph.createDataView(buffer, {format: 'uint32', byteOffset, length});
  return {device, graph, scalar, words, weights, buffer, packedScalar, packedWords};
}

it.each([
  2 ** 127,
  2 ** -126
])('constructs the exact finite-scale headless graph without caller/scratch ID collisions (%s)', scale => {
  const device = createNullWebGPUDevice();
  const fixture = new RecipeTestFixture(device, 'analysis-finite-ratios');
  recordWGSLKernelNodes(fixture.graph);
  try {
    const {nodes} = addFiniteScaleAnalysisFixture(fixture, scale);
    expect(nodes.length).toBeGreaterThan(0);
    expectCompatibleStorageUses(fixture.graph, nodes);
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it.each([
  false,
  true
])('published graph encoding rejects duplicate writable physical imports (duplicate=%s)', duplicate => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({
    byteLength: 1024,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    const first = graph.importBuffer({id: 'source', byteLength: 1024, usage: buffer.usage}, buffer);
    const second = duplicate
      ? graph.importBuffer(
          {id: 'same-physical-buffer', byteLength: 1024, usage: buffer.usage},
          buffer
        )
      : first;
    let encodings = 0;
    graph.add(
      createGPUCopyCommandNode({
        id: 'import-resolution-regression',
        resources: [
          {buffer: first, usage: 'copy-source'},
          {buffer: second, usage: 'copy-destination'}
        ],
        // A recording copy avoids GPU compilation while exercising the real graph import resolver.
        compile: () => ({
          encode: () => {
            encodings++;
          }
        })
      })
    );
    compiled = graph.compile();
    const encode = () => compiled!.encode(device.createCommandEncoder(), {parameters: undefined});
    if (duplicate) {
      expect(encode).toThrow(
        /imported buffers "source" and "same-physical-buffer" resolve to the same physical buffer/
      );
      expect(encodings).toBe(0);
    } else {
      expect(encode).not.toThrow();
      expect(encodings).toBe(1);
    }
  } finally {
    compiled?.destroy();
    buffer.destroy();
    device.destroy();
  }
});

it.each([
  'chunked',
  'sorted'
] as const)('packed geographic %s spread and polygon outputs never mix storage access modes', reduction => {
  const {device, graph, buffer, scalar, packedScalar} = context();
  try {
    const points = (byteOffset: number, length: number) =>
      graph.createDataView(buffer, {format: 'float32x2', byteOffset, length});
    const nodes = new GPUGeographicDistribution({
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      parameters: scalar(8),
      reduction,
      polygonVertexCount: 4,
      output: {
        meanCenters: points(0, 1),
        standardDistances: packedScalar(8, 1),
        ellipses: packedScalar(256, 3),
        ellipseVertices: points(512, 4),
        circleVertices: points(768, 4)
      }
    }).getCommandNodes(graph);
    expectCompatibleStorageUses(graph, nodes);
    expect(
      nodes.filter(node => node.id.includes('-packed-') && node.id.endsWith('-publish'))
    ).toHaveLength(2);
  } finally {
    device.destroy();
  }
});

it('packed publication preserves overlapping read-only prefixes and shader/pipeline reuse across queries', () => {
  const {device, graph} = context();
  const buffer = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer(
      {id: 'read-physical', byteLength: 1024, usage: Buffer.STORAGE},
      buffer
    );
    const incoming = graph.createDataView(first, {format: 'float32', byteOffset: 4, length: 2});
    const weights = graph.createDataView(first, {format: 'float32', byteOffset: 12, length: 2});
    const outgoing = graph.createDataView(first, {format: 'float32', byteOffset: 260, length: 2});
    const create = (query: number, destination: GraphDataView = outgoing) =>
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `packed-query-${query}`,
        operation: 'PackedSpatialAnalysisRegression',
        invocationCount: 2,
        bindings: [
          {name: 'values', view: incoming, type: 'f32', access: 'read'},
          {name: 'weights', view: weights, type: 'f32', access: 'read'},
          {name: 'result', view: destination, type: 'f32', access: 'read_write'}
        ],
        // Include an initial output read: the adapter must preserve existing contents as well.
        body: 'result[resultOffset+index]+=values[valuesOffset+index]*weights[weightsOffset+index];'
      });
    const single = create(0);
    const many = Array.from({length: 8}, (_, query) => create(query + 1)).flat();
    expect(single).toHaveLength(3);
    expectCompatibleStorageUses(graph, [...single, ...many]);
    expect(getGPUCommandNodePipelineKeys(many).size).toBe(
      getGPUCommandNodePipelineKeys(single).size
    );
    expect(new Set(getGPUCommandNodeKernelSources(many).map(kernel => kernel.source)).size).toBe(
      new Set(getGPUCommandNodeKernelSources(single).map(kernel => kernel.source)).size
    );
    const separate = createTransientView(graph, 'separate', 'float32', 3);
    const original = create(
      10,
      graph.createDataView(separate.buffer, {format: 'float32', byteOffset: 4, length: 2})
    )[0];
    const staged = getWGSLKernelNodeRecord(graph, 'packed-query-0')!;
    // Same element prefix: scratch must not change the analysis shader or its invocation count.
    expect(staged.bindings[2].view.byteOffset).toBe(4);
    expect(staged.invocationCount).toBe(2);
    expect(getWGSLKernelNodeRecord(graph, original.id)!.source).toBe(staged.source);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('packed publication checks all aligned scratch sizes before declaring its first replacement', () => {
  const {device, graph, buffer} = context();
  try {
    Object.assign(device.limits, {maxStorageBufferBindingSize: 256});
    const input = graph.createDataView(buffer, {format: 'uint32', length: 2});
    const first = graph.createDataView(buffer, {format: 'uint32', byteOffset: 256, length: 2});
    const second = graph.createDataView(buffer, {format: 'uint32', byteOffset: 512, length: 65});
    const state = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = state.buffers.size;
    expect(() =>
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: 'oversized-packed',
        operation: 'PackedSpatialAnalysisRegression',
        invocationCount: 2,
        bindings: [
          {name: 'inputValues', view: input, type: 'u32', access: 'read'},
          {name: 'first', view: first, type: 'u32', access: 'read_write'},
          {name: 'second', view: second, type: 'u32', access: 'read_write'}
        ],
        body: 'first[firstOffset+index]=inputValues[inputValuesOffset+index];second[secondOffset+index]=0u;'
      })
    ).toThrow(/second.*needs 260 bytes/);
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 1024, maxBufferSize: 4});
    expect(() =>
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: 'oversized-allocation',
        operation: 'PackedSpatialAnalysisRegression',
        invocationCount: 2,
        bindings: [
          {name: 'inputValues', view: input, type: 'u32', access: 'read'},
          {name: 'first', view: first, type: 'u32', access: 'read_write'}
        ],
        body: 'first[firstOffset+index]=inputValues[inputValuesOffset+index];'
      })
    ).toThrow(/packed publication allocation byteLength \(maxBufferSize\)/);
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('catchment preflights packed publication allocations before declaring selection or ratio scratch', () => {
  const {device, graph, scalar, weights, packedScalar} = context();
  try {
    const contributor = new GPUCatchmentAccessibility({
      method: '3sfca',
      facilityWeights: weights,
      supply: scalar(),
      demand: scalar(),
      output: {accessibility: packedScalar(4), ratios: packedScalar(256)}
    });
    Object.assign(device.limits, {maxBufferSize: 10});
    const state = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = state.buffers.size;
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /packed output allocation byteLength \(maxBufferSize\)/
    );
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('packed reference proximity and local tessellation also have compatible physical storage uses', () => {
  const {device, graph, words} = context();
  try {
    const storage = graph.createTransientBuffer({
      id: 'recipe-packed',
      byteLength: 16384,
      usage: Buffer.STORAGE
    });
    const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x4' | 'float32x4'>(
      format: Format,
      byteOffset: number,
      length: number
    ) => graph.createDataView(storage, {format, byteOffset, length});
    addPatchProximityRecipe(graph, {
      id: 'reference-packed',
      width: 3,
      height: 1,
      patches: words(3),
      patchClasses: words(2),
      cellSize: [1, 1],
      edgeDepth: 0,
      searchRadius: 1,
      output: {
        core: view('uint32', 0, 3),
        horizontal: view('float32', 256, 6),
        fields: view('float32', 512, 6),
        distances: view('float32', 768, 4),
        metrics: view('float32x4', 1024, 2)
      }
    });
    const result = addBuildingTessellationRecipe(graph, {
      id: 'tessellation-packed',
      segments: createTransientView(graph, 'segments', 'float32x4', 8),
      segmentBuildingIds: words(8),
      enclosure: createTransientView(graph, 'enclosure', 'float32x2', 4),
      buildingCount: 2,
      sampleCapacity: 8,
      sampleSpacing: 1,
      candidateRadius: 32,
      vertexTolerance: 1 / 1024,
      parameters: view('float32', 5888, 3),
      output: {
        positions: view('float32x2', 2048, 192),
        cells: view('uint32x4', 4096, 16),
        buildingStatus: view('uint32x4', 4352, 2),
        samplingStatus: view('uint32x4', 4384, 1)
      }
    });
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
    expectCompatibleStorageUses(graph, nodes);
    expect(
      nodes.filter(node => node.id.startsWith('tessellation-packed')).length
    ).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
    expect(
      nodes.some(node => node.id === 'reference-packed-patch-summary-packed-output-publish')
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('boundary proximity keeps packed reconciliation inputs and outputs compatible with WebGPU', () => {
  const {device, graph, buffer, packedWords} = context();
  try {
    const result = addPatchProximityRecipe(graph, {
      mode: 'boundary',
      reconcilePatches: true,
      patches: packedWords(0, 3),
      patchClasses: packedWords(256, 2),
      width: 3,
      height: 1,
      cellSize: [1, 1],
      edgeDepth: 0,
      searchRadius: 1,
      output: {
        patchParents: packedWords(512, 2),
        reconciledPatches: packedWords(768, 3),
        metrics: graph.createDataView(buffer, {format: 'float32x4', byteOffset: 1024, length: 2}),
        core: packedWords(1280, 3),
        validity: packedWords(1536, 2)
      }
    });
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
    expectCompatibleStorageUses(graph, nodes);
    expect(nodes.length).toBeLessThanOrEqual(result.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});

it.each([
  'catchment',
  'huff',
  'geographic'
] as const)('%s accepts valid packed outputs and rejects missing STORAGE before allocation', kind => {
  const contextValue = context();
  const {device, graph, scalar, weights, packedScalar, packedWords} = contextValue;
  try {
    const badBuffer = graph.createTransientBuffer({
      id: 'bad',
      byteLength: 256,
      usage: Buffer.COPY_DST
    });
    const bad = graph.createDataView(badBuffer, {format: 'float32', length: 2});
    const create = (badInput: boolean) => {
      if (kind === 'catchment')
        return new GPUCatchmentAccessibility({
          method: '3sfca',
          facilityWeights: weights,
          supply: badInput ? bad : scalar(),
          demand: scalar(),
          output: {
            accessibility: packedScalar(256),
            ratios: packedScalar(512),
            validity: packedWords(520),
            reachableFacilities: packedWords(528)
          }
        });
      if (kind === 'huff')
        return new GPUHuffTradeAreas({
          attractiveness: badInput ? bad : scalar(),
          demandWeights: weights,
          output: {
            tradeArea: packedWords(256),
            tradeAreaProbability: packedScalar(512),
            probabilities: packedScalar(520, 4)
          }
        });
      const positions = createTransientView(graph, `positions-${badInput}`, 'float32x2', 2);
      return new GPUGeographicDistribution({
        positions,
        parameters: badInput
          ? graph.createDataView(badBuffer, {format: 'float32', length: 8})
          : scalar(8),
        output: {
          meanCenters: graph.createDataView(contextValue.buffer, {
            format: 'float32x2',
            byteOffset: 256,
            length: 1
          }),
          standardDistances: packedScalar(264, 1),
          medianCenters: graph.createDataView(contextValue.buffer, {
            format: 'float32x2',
            byteOffset: 512,
            length: 1
          })
        },
        maximumMedianIterations: 2
      });
    };
    const contributor = create(false);
    const nodes = contributor.getCommandNodes(graph);
    expectCompatibleStorageUses(graph, nodes);
    expect(
      getGPUCommandNodeKernelSources(nodes).some(kernel =>
        kernel.source.includes('getScaledQuotient(value,denominator,1.0,0)')
      )
    ).toBe(true);
    const invalid = create(true);
    const state = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = state.buffers.size;
    expect(() => invalid.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it.each([
  'catchment',
  'huff',
  'geographic'
] as const)('%s rejects physical input/output aliases imported through distinct handles', kind => {
  const {device, graph, scalar, weights} = context();
  const buffer = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer(
      {id: 'read-alias', byteLength: 1024, usage: Buffer.STORAGE},
      buffer
    );
    const second = graph.importBuffer(
      {id: 'write-alias', byteLength: 1024, usage: Buffer.STORAGE},
      buffer
    );
    const input = graph.createDataView(first, {format: 'float32', length: 2});
    const output = graph.createDataView(second, {format: 'float32', length: 2});
    expect(() => {
      if (kind === 'catchment')
        return new GPUCatchmentAccessibility({
          method: '2sfca',
          supply: input,
          demand: scalar(),
          facilityWeights: weights,
          output: {accessibility: output}
        });
      if (kind === 'huff')
        return new GPUHuffTradeAreas({
          attractiveness: input,
          demandWeights: weights,
          output: {tradeArea: graph.createDataView(second, {format: 'uint32', length: 2})}
        });
      return new GPUGeographicDistribution({
        positions: graph.createDataView(first, {format: 'float32x2', length: 2}),
        parameters: scalar(8),
        output: {meanCenters: graph.createDataView(second, {format: 'float32x2', length: 1})}
      });
    }).toThrow(/logical byte ranges must not overlap/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it.each([
  'chunked',
  'sorted'
] as const)('geographic %s median kernels have a constant pipeline count as rounds grow and use no reserved identifiers', reduction => {
  const sizes = [1, 8].map(maximumMedianIterations => {
    const {device, graph, scalar} = context();
    try {
      const contributor = new GPUGeographicDistribution({
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        parameters: scalar(8),
        reduction,
        maximumMedianIterations,
        output: {medianCenters: createTransientView(graph, 'median', 'float32x2', 1)}
      });
      const nodes = contributor.getCommandNodes(graph);
      expectCompatibleStorageUses(graph, nodes);
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expectSpatialAnalysisWGSL(kernel.source, kernel.id);
      return getGPUCommandNodePipelineKeys(nodes).size;
    } finally {
      device.destroy();
    }
  });
  expect(sizes[1]).toBe(sizes[0]);
});

it('Huff ignores unused facility and demand bindings and rejects float32 parameter overflow', () => {
  const {device, graph, scalar, weights, packedWords} = context();
  try {
    const unused = graph.createTransientBuffer({
      id: 'unused',
      byteLength: 256,
      usage: Buffer.COPY_DST
    });
    const demand = graph.createDataView(unused, {format: 'float32', length: 2});
    expect(() =>
      new GPUHuffTradeAreas({
        attractiveness: scalar(),
        demandWeights: weights,
        facilityWeights: {...weights, weights: demand},
        demand,
        output: {tradeArea: packedWords(0)}
      }).getCommandNodes(graph)
    ).not.toThrow();
    for (const exponent of [1e100, Number.MIN_VALUE])
      expect(() => getGPUHuffTradeAreasParameterValues({exponent})).toThrow(/positive and finite/);
  } finally {
    device.destroy();
  }
});

it('checks nested grouped scratch before declaring anything, even when all caller bindings fit', () => {
  const {device, graph, scalar, words} = context();
  try {
    Object.assign(device.limits, {maxStorageBufferBindingSize: 4096, maxBufferSize: 8192});
    const contributor = new GPUGeographicDistribution({
      positions: createTransientView(graph, 'positions', 'float32x2', 512),
      groupIds: words(512),
      groupCount: 256,
      parameters: scalar(8),
      reduction: 'chunked',
      output: {meanCenters: createTransientView(graph, 'means', 'float32x2', 256)}
    });
    const state = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const before = state.buffers.size;
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /grouped partial sums scratch needs 6144 bytes/
    );
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it.each([
  'chunked',
  'sorted'
] as const)('audits every geographic %s output kernel and its node estimate', reduction => {
  const {device, graph, scalar, words} = context();
  try {
    const contributor = new GPUGeographicDistribution({
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      lineEnds: createTransientView(graph, 'ends', 'float32x2', 2),
      parameters: scalar(8),
      reduction,
      maximumMedianIterations: 2,
      polygonVertexCount: 4,
      output: {
        counts: words(1),
        weightSums: scalar(1),
        meanCenters: createTransientView(graph, 'mean', 'float32x2', 1),
        medianCenters: createTransientView(graph, 'median', 'float32x2', 1),
        medianConverged: words(1),
        standardDistances: scalar(1),
        ellipses: scalar(3),
        directionalMeans: scalar(3),
        ellipseVertices: createTransientView(graph, 'ellipse', 'float32x2', 4),
        circleVertices: createTransientView(graph, 'circle', 'float32x2', 4)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expectCompatibleStorageUses(graph, nodes);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expectSpatialAnalysisWGSL(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});
