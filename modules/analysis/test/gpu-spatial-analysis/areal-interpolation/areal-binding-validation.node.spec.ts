// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  GPUArealInterpolation,
  type GPUArealInterpolationProps
} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation';
import {
  GPUPycnophylactic,
  type GPUPycnophylacticProps
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

const device = createNullWebGPUDevice();
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

function createPycnophylacticProps(graph: GPUCommandGraph): GPUPycnophylacticProps {
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
    expect(() => new GPUPycnophylactic(props)).toThrow(
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
      new GPUPycnophylactic({
        ...props,
        output: {...props.output, density: props.output.mass}
      })
  ).toThrow(/must not overlap/);
  const status = createTransientView(graph, 'shared-status', 'uint32', 1);
  expect(
    () =>
      new GPUPycnophylactic({
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
  expect(() => graph.add(new GPUPycnophylactic(props))).not.toThrow();
});

it('prechecks aligned areal output ranges before declaring any resources', () => {
  const graph = new GPUCommandGraph(device);
  const props = createArealProps(graph);
  const backing = createTransientView(graph, 'oversized-status', 'uint32', 33_554_433);
  props.output.status.overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    byteOffset: 4,
    length: 33_554_432
  });
  const contributor = new GPUArealInterpolation(props);
  expectNoDeclarations(
    graph,
    () => contributor.getCommandNodes(graph),
    /output.status.overflow.*134217732 bytes/
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
  const contributor = new GPUPycnophylactic({...props, tolerance});
  expectNoDeclarations(
    graph,
    () => contributor.getCommandNodes(graph),
    /tolerance.*134217732 bytes/
  );
  vi.restoreAllMocks();
  const large = new GPUPycnophylactic({
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
