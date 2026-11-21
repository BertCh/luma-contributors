// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {afterAll, afterEach, describe, expect, it, vi} from 'vitest';
import {
  GPUArealInterpolation,
  type GPUArealInterpolationProps
} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation';
import {
  GPUPycnophylacticInterpolation,
  type GPUPycnophylacticInterpolationProps
} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-pycnophylactic';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {getZoneAnchorNodes} from '../../../src/gpu-spatial-analysis/areal-interpolation/zone-anchor-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {addChangeOfSupportRecipe} from '../../../src/gpu-spatial-analysis/recipes/change-of-support-recipe';
import {createArealRecipeRectangleFixture} from './areal-interpolation-output-oracle-fixtures';
import {
  GPUPolygonRasterization,
  getGPUPolygonRasterizationParameterValues
} from '../../../src/raster-analysis/polygon-rasterization/index';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord,
  createWGSLKernelNode
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

describe('areal rectangle recipe raster oracle', () => {
  useCPUExecutorTestHooks();
  it('places the recipe rectangle raster over both positive-y source polygons', () => {
    const {grid} = createArealRecipeRectangleFixture();
    // North-up rows must have centres at y=1.5 and .5, with bottom edge zero.
    expect(Array.from(getGPUPolygonRasterizationParameterValues(grid))).toEqual([0, 0, 1, -1]);
  });
  for (const [capacity, sourceCrossingCapacity] of [
    [1, 8],
    [2, 8],
    [1, 4]
  ] as const) {
    it(`rasterizes recipe rectangles with pair capacity ${capacity} and source crossing capacity ${sourceCrossingCapacity}`, () => {
      const rectangle = createArealRecipeRectangleFixture();
      rectangle.source.crossingCapacity = sourceCrossingCapacity;
      const graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      recordWGSLKernelNodes(graph);
      const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
      const parameters = createTransientView(graph, 'parameters', 'float32', 4);
      memory.write(parameters, getGPUPolygonRasterizationParameterValues(rectangle.grid));
      const rasters = [rectangle.source, rectangle.target].map((system, index) => {
        const prefix = `raster-${index}`;
        const positions = createTransientView(
          graph,
          `${prefix}-positions`,
          'float32x2',
          system.positions.length / 2
        );
        const featureOffsets = createTransientView(
          graph,
          `${prefix}-features`,
          'uint32',
          system.featureOffsets.length
        );
        const polygonOffsets = createTransientView(
          graph,
          `${prefix}-polygons`,
          'uint32',
          system.polygonOffsets.length
        );
        const ringOffsets = createTransientView(
          graph,
          `${prefix}-rings`,
          'uint32',
          system.ringOffsets.length
        );
        const zones = createTransientView(graph, `${prefix}-zones`, 'uint32', 8);
        const overflow = scalar(`${prefix}-overflow`),
          crossingCount = scalar(`${prefix}-crossings`);
        memory.write(positions, system.positions);
        memory.write(featureOffsets, system.featureOffsets);
        memory.write(polygonOffsets, system.polygonOffsets);
        memory.write(ringOffsets, system.ringOffsets);
        const nodes = new GPUPolygonRasterization({
          id: prefix,
          grid: rectangle.grid,
          parameters,
          positions,
          featureOffsets,
          polygonOffsets,
          ringOffsets,
          crossingCapacity: system.crossingCapacity,
          output: {zones, status: {overflow, crossingCount}}
        }).getCommandNodes(graph);
        const interpretedNodes = nodes.map(node => {
          const record = getWGSLKernelNodeRecord(graph, node.id);
          if (!record?.declarations?.includes('struct RasterExtent')) return node;
          // The shared interpreter fixes bool struct layout only for one member per line.
          // Reformat this inline declaration without changing WGSL tokens or GPU behavior.
          const declarations = record.declarations.replace(
            'struct RasterExtent { origin: vec2<f32>, cellSize: vec2<f32>, valid: bool, flipped: bool };',
            'struct RasterExtent {\n  origin: vec2<f32>,\n  cellSize: vec2<f32>,\n  valid: bool,\n  flipped: bool\n};'
          );
          return createWGSLKernelNode(graph, {
            id: `${node.id}-cpu-format`,
            operation: record.operation,
            bindings: record.bindings,
            invocationCount: record.invocationCount,
            workgroupSize: record.workgroupSize[0],
            declarations,
            body: record.body
          });
        });
        executeCommandNodesOnCPU(interpretedNodes, memory);
        // Each rectangle has two vertical edges crossing each of two rows: 4 records per polygon.
        expect(memory.readUint32(crossingCount)[0]).toBe(index === 0 ? 8 : 4);
        const expectedOverflow = index === 0 && sourceCrossingCapacity === 4 ? 1 : 0;
        expect(memory.readUint32(overflow)[0]).toBe(expectedOverflow);
        expect(Array.from(memory.readUint32(zones))).toEqual(
          expectedOverflow
            ? Array(8).fill(0xffffffff)
            : index === 0
              ? [0, 0, 1, 1, 0, 0, 1, 1]
              : [0xffffffff, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0xffffffff]
        );
        return zones;
      });
      if (sourceCrossingCapacity === 4) return; // The negative control intentionally leaves no source zones.
      const overflow = scalar('pair-overflow'),
        requiredPairCount = scalar('pair-count');
      const weights = {
        offsets: createTransientView(graph, 'offsets', 'uint32', 2),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', capacity),
        weights: createTransientView(graph, 'weights', 'float32', capacity),
        status: {overflow}
      };
      executeCommandNodesOnCPU(
        new GPUArealInterpolation({
          grid: rectangle.grid,
          sourceZones: rasters[0],
          targetZones: rasters[1],
          sourceZoneCount: 2,
          targetZoneCount: 1,
          output: {weights, status: {overflow, requiredPairCount}}
        }).getCommandNodes(graph),
        memory
      );
      expect(memory.readUint32(requiredPairCount)[0]).toBe(2);
      expect(memory.readUint32(overflow)[0]).toBe(capacity === 1 ? 1 : 0);
      expect(Array.from(memory.readFloat32(weights.weights))).toEqual(Array(capacity).fill(0.5));
    });
  }
});

it('keeps the recipe caller nested overflow word and refuses conflicting overrides', () => {
  const graph = new GPUCommandGraph(device);
  const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const zones = {
    kind: 'polygons' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    crossingCapacity: 4
  };
  const overflow = scalar('caller-overflow');
  const props = {
    grid: makeGPURasterMetadata({width: 4, height: 2}),
    source: zones,
    target: zones,
    pairCapacity: 1,
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {extensiveWeights: {status: {overflow}}}
  };
  const result = addChangeOfSupportRecipe(graph, props);
  expect(result.status.overflow).toBe(overflow);
  expect(result.extensiveWeights.status.overflow).toBe(overflow);
  expect(() =>
    addChangeOfSupportRecipe(graph, {
      ...props,
      id: 'conflicting',
      output: {
        ...props.output,
        status: {overflow: scalar('other-overflow')}
      }
    })
  ).toThrow(/overflow.*same|overflow.*reference/);
});

it('validates the first areal overflow descriptor once and retains prototype ports', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const canonical = props.output.status.overflow;
  const independent = createTransientView(graph, 'independent', 'uint32', 1);
  props.output.weights.status = {overflow: canonical};
  let reads = 0;
  Object.defineProperty(props.output.status, 'overflow', {
    enumerable: true,
    get: () => (reads++ === 0 ? independent : canonical)
  });
  expect(() => new GPUArealInterpolation(props)).toThrow(/weights.status.overflow/);
  expect(reads).toBe(1);
  const stable = createArealProps(new GPUCommandGraph(device));
  const overflow = stable.output.status.overflow;
  stable.output.status = Object.create({
    get overflow() {
      return overflow;
    }
  });
  const contributor = new GPUArealInterpolation(stable);
  expect(contributor.props.output.status.overflow).toBe(overflow);
});

const device = createNullWebGPUDevice();

it('rejects an independent nested overflow flag before an areal CSR reaches a consumer', () => {
  const graph = new GPUCommandGraph(device);
  const overflow = createTransientView(graph, 'outer-overflow', 'uint32', 1);
  const props = {
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    sourceZones: createTransientView(graph, 'source', 'uint32', 4),
    targetZones: createTransientView(graph, 'target', 'uint32', 4),
    sourceZoneCount: 2,
    targetZoneCount: 1,
    output: {
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 2),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 1),
        weights: createTransientView(graph, 'weights', 'float32', 1),
        status: {overflow: createTransientView(graph, 'stale-overflow', 'uint32', 1)}
      },
      status: {overflow}
    }
  };
  expect(() => new GPUArealInterpolation(props)).toThrow(
    /weights.status.overflow.*output.status.overflow/
  );
  props.output.weights.status.overflow = overflow;
  expect(() => new GPUArealInterpolation(props)).not.toThrow();
});
afterEach(() => vi.restoreAllMocks());
afterAll(() => device.destroy());

it('uses a fixed set of valid anchor kernels as the zone count grows', () => {
  const counts: number[] = [];
  for (const zoneCount of [10, 100, 1000]) {
    const graph = new GPUCommandGraph(device);
    const nodes = getZoneAnchorNodes(graph, {
      id: 'anchors',
      operation: 'GPUArealInterpolation',
      zoneCount,
      zones: createTransientView(graph, 'zones', 'uint32', 64),
      anchorCells: createTransientView(graph, 'anchor-cells', 'uint32', zoneCount)
    }).nodes.filter(node => node.workload?.variant?.startsWith('anchor-'));
    counts.push(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      expectValidQueryShaderLiterals(source);
      const words =
        source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      expect(source).not.toMatch(/for\s*\(var zone\s*=/);
    }
  }
  expect(new Set(counts).size).toBe(1);
});

it('declares the returned anchor raster for storage and copy-source composition', () => {
  const graph = new GPUCommandGraph(device);
  const anchored = getZoneAnchorNodes(graph, {
    id: 'copyable-anchors',
    operation: 'GPUArealInterpolation',
    zoneCount: 10,
    zones: createTransientView(graph, 'zones', 'uint32', 6),
    anchorCells: createTransientView(graph, 'anchors', 'uint32', 10)
  });
  graph.add(anchored.nodes);
  expect(anchored.zones.buffer.usage & Buffer.STORAGE).toBe(Buffer.STORAGE);
  expect(anchored.zones.buffer.usage & Buffer.COPY_SRC).toBe(Buffer.COPY_SRC);
  const readback = createTransientView(
    graph,
    'readback',
    'uint32',
    6,
    Buffer.STORAGE | Buffer.COPY_DST
  );
  expect(() =>
    graph.addCopyPass({
      id: 'anchor-readback',
      resources: [
        {buffer: anchored.zones, usage: 'copy-source'},
        {buffer: readback, usage: 'copy-destination'}
      ],
      compile: () => ({
        encode: ({commandEncoder, getBuffer}) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: getBuffer(anchored.zones),
            sourceOffset: anchored.zones.byteOffset,
            destinationBuffer: getBuffer(readback),
            destinationOffset: readback.byteOffset,
            size: 24
          })
      })
    })
  ).not.toThrow();
});

it('rejects missing STORAGE usage in anchor inputs before declaring nested scratch', () => {
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 24, usage: Buffer.COPY_SRC | Buffer.COPY_DST});
  try {
    const zones = importGraphBuffer(graph, 'non-storage-zones', buffer, 'uint32', 6);
    const anchorCells = createTransientView(graph, 'anchors', 'uint32', 10);
    expectNoDeclarations(
      graph,
      () =>
        getZoneAnchorNodes(graph, {
          id: 'anchors',
          operation: 'GPUArealInterpolation',
          zoneCount: 10,
          zones,
          anchorCells
        }),
      /zones.*Buffer.STORAGE/
    );
  } finally {
    buffer.destroy();
  }
});

it('rejects an unrepresentable anchor sort sentinel before declaring resources', () => {
  const graph = new GPUCommandGraph(device);
  const zones = createTransientView(graph, 'zones', 'uint32', 6);
  const anchorCells = createTransientView(graph, 'anchors', 'uint32', 1);
  expectNoDeclarations(
    graph,
    () =>
      getZoneAnchorNodes(graph, {
        id: 'anchors',
        operation: 'GPUArealInterpolation',
        zoneCount: 0xffffffff,
        zones,
        anchorCells
      }),
    /anchors zoneCount.*4294967294/
  );
});

function createArealProps(graph: GPUCommandGraph): GPUArealInterpolationProps {
  return {
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    sourceZones: createTransientView(graph, 'source-zones', 'uint32', 4),
    targetZones: createTransientView(graph, 'target-zones', 'uint32', 4),
    sourceZoneCount: 2,
    targetZoneCount: 1,
    output: {
      weights: {
        offsets: createTransientView(graph, 'offsets', 'uint32', 2),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 4),
        weights: createTransientView(graph, 'weights', 'float32', 4)
      },
      status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
    }
  };
}

describe('areal nested status writers', () => {
  useCPUExecutorTestHooks();
  it.each([
    'count',
    'requiredCount'
  ] as const)('rejects same-buffer read/write copies for disjoint nested %s', field => {
    const graph = new GPUCommandGraph(device);
    const props = createArealProps(graph);
    const backing = createTransientView(graph, 'packed-status', 'uint32', 65);
    const source = graph.createDataView(backing.buffer, {
      format: 'uint32',
      length: field === 'count' ? 2 : 1
    });
    const destination = graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 256,
      length: 1
    });
    if (field === 'count') props.output.weights.offsets = source;
    else props.output.status.requiredPairCount = source;
    props.output.weights.status = {[field]: destination};
    expect(() => new GPUArealInterpolation(props)).toThrow(
      /storage-write.*storage-read.*physical buffer/
    );
  });
  it('checks inherited unsupported destinations before narrowing and captures supported accessors once', () => {
    const graph = new GPUCommandGraph(device);
    const props = createArealProps(graph);
    props.output.weights.status = Object.create({
      complete: createTransientView(graph, 'complete', 'uint32', 1)
    });
    expect(() => new GPUArealInterpolation(props)).toThrow('output.weights.status.complete');
    const count = createTransientView(graph, 'count', 'uint32', 1);
    let reads = 0;
    props.output.weights.status = Object.create({
      get count() {
        reads++;
        return count;
      }
    });
    expect(new GPUArealInterpolation(props).props.output.weights.status?.count).toBe(count);
    expect(reads).toBe(1);
  });
  it.each([
    true,
    false
  ])('publishes clamped count and unclamped aliased counts (shared=%s)', shared => {
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    const props = createArealProps(graph);
    props.output.weights.neighbors = createTransientView(graph, 'neighbors-small', 'uint32', 1);
    props.output.weights.weights = createTransientView(graph, 'weights-small', 'float32', 1);
    const count = createTransientView(graph, 'count', 'uint32', 1);
    const requiredPairCount = createTransientView(graph, 'required-pair-count', 'uint32', 1);
    const requiredCount = shared
      ? requiredPairCount
      : createTransientView(graph, 'required-count', 'uint32', 1);
    props.output.status.requiredPairCount = requiredPairCount;
    props.output.weights.status = {count, requiredCount, overflow: props.output.status.overflow};
    memory.write(props.sourceZones, Uint32Array.of(0, 1, 0, 1));
    memory.write(props.targetZones, new Uint32Array(4));
    executeCommandNodesOnCPU(new GPUArealInterpolation(props).getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(count))).toEqual([1]);
    expect(Array.from(memory.readUint32(requiredCount))).toEqual([2]);
    expect(Array.from(memory.readUint32(requiredPairCount))).toEqual([2]);
    expect(Array.from(memory.readUint32(props.output.status.overflow))).toEqual([1]);
  });
  for (const unweightedFastPath of [true]) {
    it(`resets poisoned nested loss and count aliases on reuse (fast=${unweightedFastPath})`, () => {
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      recordWGSLKernelNodes(graph);
      const props = createArealProps(graph);
      props.unweightedFastPath = unweightedFastPath;
      const count = createTransientView(graph, 'nested-count', 'uint32', 1);
      const requiredCount = createTransientView(graph, 'nested-required', 'uint32', 1);
      const candidateOverflow = createTransientView(graph, 'nested-candidate', 'uint32', 1);
      props.output.weights.status = {
        overflow: props.output.status.overflow,
        count,
        requiredCount,
        candidateOverflow
      };
      memory.write(props.sourceZones, Uint32Array.of(0, 1, 0, 1));
      memory.write(props.targetZones, new Uint32Array(4));
      const nodes = new GPUArealInterpolation(props).getCommandNodes(graph);
      for (const poisoned of [1, 99]) {
        for (const destination of [count, requiredCount, candidateOverflow])
          memory.write(destination, Uint32Array.of(poisoned));
        executeCommandNodesOnCPU(nodes, memory);
        // Two distinct target/source pairs; each source occupies two of the four cells.
        expect(Array.from(memory.readUint32(count))).toEqual([2]);
        expect(Array.from(memory.readUint32(requiredCount))).toEqual([2]);
        expect(Array.from(memory.readUint32(candidateOverflow))).toEqual([0]);
      }
    });
  }
  it.each([
    'complete',
    'unresolvedCount',
    'uncertainCount',
    'invalidCount',
    'iterationCount',
    'converged',
    'iterationLimitReached',
    'approximation'
  ] as const)('rejects unsupported output.weights.status.%s at construction', field => {
    const graph = new GPUCommandGraph(device);
    const props = createArealProps(graph);
    props.output.weights.status = {
      [field]: createTransientView(graph, `unsupported-${field}`, 'uint32', 1)
    };
    expect(() => new GPUArealInterpolation(props)).toThrow(`output.weights.status.${field}`);
  });
});

function createPycnophylacticProps(graph: GPUCommandGraph): GPUPycnophylacticInterpolationProps {
  return {
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    zones: createTransientView(graph, 'zones', 'uint32', 4),
    totals: createTransientView(graph, 'totals', 'float32', 2),
    zoneCount: 2,
    maximumIterations: 1,
    output: {mass: createTransientView(graph, 'mass', 'float32', 4)}
  };
}

function expectNoDeclarations(graph: GPUCommandGraph, declare: () => unknown, message: RegExp) {
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(declare).toThrow(message);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
}

it('checks aligned writable ranges across every areal output field', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const backing = createTransientView(graph, 'shared-output', 'uint32', 68);
  props.output.weights.neighbors = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 4
  });
  props.output.areas = graph.createDataView(backing.buffer, {
    format: 'float32',
    byteOffset: 16,
    length: 4
  });
  // Logical bytes do not overlap, but both views bind from aligned byte zero in the same node.
  expect(() => new GPUArealInterpolation(props)).toThrow(/must not overlap/);
  props.output.areas = graph.createDataView(backing.buffer, {
    format: 'float32',
    byteOffset: 256,
    length: 4
  });
  expect(() => new GPUArealInterpolation(props)).not.toThrow();
  props.output.weights.weights = props.output.areas;
  expect(() => new GPUArealInterpolation(props)).toThrow(/must not overlap/);
});

it('permits consecutive neighbor/status bytes in distinct passes and ignores unused distances', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const backing = createTransientView(graph, 'shared', 'uint32', 5);
  props.output.weights.neighbors = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 4
  });
  props.output.status.overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 16,
    length: 1
  });
  const distanceBuffer = device.createBuffer({
    byteLength: 16,
    usage: Buffer.COPY_SRC | Buffer.COPY_DST
  });
  try {
    props.output.weights.distances = importGraphBuffer(
      graph,
      'unused-distances',
      distanceBuffer,
      'float32',
      4
    );
    expect(() => graph.add(new GPUArealInterpolation(props))).not.toThrow();
    props.output.weights.distances = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: 4
    });
    expect(() => new GPUArealInterpolation(props)).not.toThrow();
    props.output.weights.distances = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: 3
    });
    expect(() => new GPUArealInterpolation(props)).toThrow(/distances length/);
  } finally {
    distanceBuffer.destroy();
  }
});

it('rejects physical input/output aliases through offset views of one import', () => {
  const graph = new GPUCommandGraph(device);
  const props = createPycnophylacticProps(graph);
  const buffer = device.createBuffer({
    byteLength: 16,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  try {
    props.zones = importGraphBuffer(graph, 'zones-import', buffer, 'uint32', 4);
    props.output.mass = graph.createDataView(props.zones.buffer, {format: 'float32', length: 4});
    expect(() => new GPUPycnophylacticInterpolation(props)).toThrow(
      /outputs must not share buffers with inputs/
    );
  } finally {
    buffer.destroy();
  }
});

it('rejects pycnophylactic mass/density and scalar status aliases at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createPycnophylacticProps(graph);
  expect(
    () =>
      new GPUPycnophylacticInterpolation({
        ...props,
        output: {...props.output, density: props.output.mass}
      })
  ).toThrow(/must not overlap/);
  const status = createTransientView(graph, 'shared-status', 'uint32', 1);
  expect(
    () =>
      new GPUPycnophylacticInterpolation({
        ...props,
        tolerance: createTransientView(graph, 'tolerance', 'float32', 1),
        output: {...props.output, status: {converged: status, invalidCount: status}}
      })
  ).toThrow(/must not overlap/);
});

it('accepts packed pycnophylactic mass/status outputs used in separate passes', () => {
  const graph = new GPUCommandGraph(device);
  const props = createPycnophylacticProps(graph);
  const backing = createTransientView(graph, 'packed-mass-status', 'uint32', 5);
  props.output.mass = graph.createDataView(backing.buffer, {format: 'float32', length: 4});
  props.output.status = {
    invalidCount: graph.createDataView(backing.buffer, {
      format: 'uint32',
      byteOffset: 16,
      length: 1
    })
  };
  expect(() => graph.add(new GPUPycnophylacticInterpolation(props))).not.toThrow();
});

it('rejects a non-scalar outer loss view before declaring any resources', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const backing = createTransientView(graph, 'oversized-status', 'uint32', 33_554_433);
  props.output.status.overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 33_554_432
  });
  expectNoDeclarations(
    graph,
    () => new GPUArealInterpolation(props),
    /overflow must contain one uint32 row/
  );
});

it('prechecks the extra source offset of weighted zone totals before declarations', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUArealInterpolation({
    ...createArealProps(graph),
    sourceZoneCount: 33_554_432,
    unweightedFastPath: false
  });
  expectNoDeclarations(
    graph,
    () => contributor.getCommandNodes(graph),
    /sourceSegmentOffsets.*134217732 bytes/
  );
});

it('prechecks pycnophylactic aligned inputs and derived trailing offsets', () => {
  const graph = new GPUCommandGraph(device);
  const props = createPycnophylacticProps(graph);
  const backing = createTransientView(graph, 'oversized-tolerance', 'float32', 33_554_433);
  const tolerance = graph.createDataView(backing.buffer, {
    format: 'float32',
    byteOffset: 4,
    length: 33_554_432
  });
  const contributor = new GPUPycnophylacticInterpolation({...props, tolerance});
  expectNoDeclarations(
    graph,
    () => contributor.getCommandNodes(graph),
    /tolerance.*134217732 bytes/
  );
  vi.restoreAllMocks();
  const large = new GPUPycnophylacticInterpolation({
    ...props,
    zoneCount: 33_554_432,
    totals: createTransientView(graph, 'large-totals', 'float32', 33_554_432)
  });
  expectNoDeclarations(
    graph,
    () => large.getCommandNodes(graph),
    /segmentOffsets.*134217732 bytes/
  );
});

it('reduces categories in parallel and permits category outputs beside retained neighbors', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const backing = createTransientView(graph, 'packed-neighbors-categories', 'uint32', 6);
  props.output.weights.neighbors = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 4
  });
  props.output.categoryShares = graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 2,
    byteOffset: 16
  });
  props.categories = {
    sourceCategories: createTransientView(graph, 'source-categories', 'uint32', 2),
    categoryCount: 2
  };
  const nodes = new GPUArealInterpolation(props).getCommandNodes(graph);
  expect(() => graph.add(nodes)).not.toThrow();
  const categoryNodes = nodes.filter(node => node.workload?.variant?.startsWith('category-'));
  expect(categoryNodes.map(node => node.workload?.variant)).toEqual([
    'category-keys',
    'category-shares'
  ]);
  for (const {source} of getGPUCommandNodeKernelSources(categoryNodes)) {
    expect(source).not.toMatch(/for\s*\(/);
    expectValidQueryShaderLiterals(source);
    const words =
      source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
    expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
  }
});

it('prechecks category reduction trailing offsets before declaring any scratch', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  props.categories = {
    sourceCategories: createTransientView(graph, 'source-categories', 'uint32', 2),
    categoryCount: 33_554_432
  };
  props.output.categoryShares = createTransientView(graph, 'shares', 'float32', 33_554_432);
  const contributor = new GPUArealInterpolation(props);
  expectNoDeclarations(
    graph,
    () => contributor.getCommandNodes(graph),
    /categorySegmentOffsets.*134217732 bytes/
  );
});

it('rejects unrepresentable category segment counts at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  props.categories = {
    sourceCategories: createTransientView(graph, 'source-categories', 'uint32', 2),
    categoryCount: 0xffffffff
  };
  expect(() => new GPUArealInterpolation(props)).toThrow(/categories.categoryCount.*4294967294/);
});

it('rejects aligned areal neighbors/weights sharing read/write storage scope', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const backing = createTransientView(graph, 'read-write-alias', 'uint32', 68);
  props.output.weights.neighbors = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 4
  });
  props.output.weights.weights = graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 4,
    byteOffset: 256
  });
  expect(() => new GPUArealInterpolation(props)).toThrow(
    /storage-write.*storage-read.*physical buffer/
  );
});

it.each([
  true,
  false
])('publishes mandatory outer areal loss without a nested alias (fast=%s)', unweightedFastPath => {
  const graph = new GPUCommandGraph(device);
  const props = {...createArealProps(graph), unweightedFastPath};
  const outer = props.output.status.overflow;
  const producer = new GPUArealInterpolation(props);
  expect(producer.props.output.weights.status?.overflow === outer).toBe(true);
  const conflicting = createTransientView(graph, 'conflicting-overflow', 'uint32', 1);
  props.output.weights.status = {overflow: conflicting};
  expect(() => new GPUArealInterpolation(props)).toThrow(/status.overflow.*reference/);
});

it('validates a supplied nested overflow scalar before replacing its alias', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const backing = createTransientView(graph, 'malformed-alias', 'uint32', 2);
  props.output.status.overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1
  });
  props.output.weights.status = {overflow: backing};
  expect(() => new GPUArealInterpolation(props)).toThrow(
    /output.weights.status overflow must contain one uint32 row/
  );
});
