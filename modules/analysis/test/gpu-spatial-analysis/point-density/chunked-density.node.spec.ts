// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  GraphDataView,
  type GPUCommandNode,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {
  GPUKernelDensity,
  type GPUKernelDensityProps
} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {
  GPUPointDensity,
  type GPUPointDensityProps
} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {createDensityPointPass} from '../../../src/gpu-spatial-analysis/point-density/density-point-passes';
import {getDensityReservedWords} from './density-wgsl-reserved-words';
import {GPU_POINT_DENSITY_MAXIMUM_POINT_COUNT} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {getFixedPointHeadroomBits} from '../../../src/analysis-core/kernels/fixed-point-accumulator';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/network-topology';
import {createPointDensityFinalizeNode} from '../../../src/gpu-spatial-analysis/point-density/point-density-kernels';
import {getSharedOutputDensityFixture} from './shared-output-density-fixture';
import {withDensityTestBindingLimit} from './density-test-device';
import {
  findConstantNonFiniteBitcasts,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('split density totals construct NaN and both infinities from runtime integer bits', () => {
  const {device, graph, props} = createFixture([256, 256, 256, 251]);
  try {
    device.limits.maxStorageBufferBindingSize = 2048;
    device.limits.maxBufferSize = 2048;
    const nodes = new GPUKernelDensity({...props, binnedPadding: 0}).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    const finishes = sources.filter(({id}) => id.endsWith('-nonfinite-finish'));
    expect(finishes.length).toBeGreaterThan(0);
    for (const {source, id} of sources) {
      expect(findConstantNonFiniteBitcasts(stripWGSLComments(source)), id).toEqual([]);
      expect(getDensityReservedWords(source), id).toEqual([]);
    }
    for (const {source, id} of finishes) {
      expect(source, id).toContain('var nonfiniteBits = 0u;');
      expect(source, id).toContain('bitcast<f32>(nonfiniteBits)');
    }
    for (const bits of ['0x7fc00000u', '0x7f800000u', '0xff800000u']) {
      expect(findConstantNonFiniteBitcasts(`let missing = bitcast<f32>(${bits});`)).toHaveLength(1);
    }
  } finally {
    device.destroy();
  }
});

it('synthetic density limits support native read-only attributes and restore their descriptor', () => {
  const nativeLimits = Object.create({
    get maxStorageBufferBindingSize() {
      return 65536;
    },
    get maxBufferSize() {
      return 262144;
    },
    get maxUniformBufferBindingSize() {
      return 65536;
    }
  });
  const device = {limits: nativeLimits} as ReturnType<typeof createNullWebGPUDevice>;
  const descriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  expect(() => {
    device.limits.maxBufferSize = 1;
  }).toThrow();
  expect(withDensityTestBindingLimit(device, undefined, () => device.limits)).toBe(nativeLimits);
  expect(() =>
    withDensityTestBindingLimit(device, 16384, () => {
      expect(device.limits.maxStorageBufferBindingSize).toBe(16384);
      expect(device.limits.maxBufferSize).toBe(16384);
      expect(device.limits.maxUniformBufferBindingSize).toBe(65536);
      throw new Error('construction failed');
    })
  ).toThrow('construction failed');
  expect(Object.getOwnPropertyDescriptor(device, 'limits')).toEqual(descriptor);
  expect(device.limits).toBe(nativeLimits);
});

it('density skips empty end-of-buffer row chunks while checking actual empty kernel bindings', () => {
  for (const kind of ['grid', 'hexagon', 'kde'] as const) {
    for (const scratchTileSize of kind === 'kde' ? [undefined] : [undefined, 256]) {
      const {device, graph, props} = createFixture([1031]);
      try {
        const createSubvector = <Format extends 'float32x2' | 'float32'>(
          name: string,
          format: Format,
          lengths: number[]
        ) => {
          const backing = createTransientView(graph, name, format, 1031);
          let rowOffset = 0;
          return createVectorView(
            name,
            format,
            lengths.map(length => {
              const chunk = graph.createDataView(backing.buffer, {
                format,
                length,
                byteOffset: rowOffset * backing.rowByteLength
              });
              rowOffset += length;
              return chunk;
            })
          );
        };
        const positions = createSubvector('shared-positions', 'float32x2', [0, 1, 256, 0, 774, 0]);
        const weights = createSubvector(
          'shared-weights',
          'float32',
          kind === 'kde' ? [0, 1, 256, 0, 774, 0] : [2, 0, 17, 1012, 0]
        );
        const density =
          kind === 'kde'
            ? new GPUKernelDensity({...props, positions, weights})
            : new GPUPointDensity({
                positions,
                weights,
                scratchTileSize,
                binning: kind,
                hexagonRadius: kind === 'hexagon' ? 1 : undefined,
                bounds: [0, 0, 8, 4],
                gridSize: [8, 4],
                statistic: 'sum',
                output: {values: props.output.values}
              });
        const nodes = density.getCommandNodes(graph);
        for (const node of nodes) {
          for (const resource of node.resources ?? []) {
            if ('buffer' in resource && resource.buffer instanceof GraphDataView) {
              const binding = getViewBindingRange(resource.buffer);
              expect(binding.offset + binding.size, node.id).toBeLessThanOrEqual(
                resource.buffer.buffer.byteLength
              );
            }
          }
        }
        expect(() =>
          createDensityPointPass(graph, {
            id: 'actually-bound-empty-view',
            operation: 'GPUPointDensity',
            invocationCount: 0,
            bindings: [{name: 'source', view: positions.data.at(-1)!, access: 'read', type: 'f32'}],
            body: ''
          })
        ).toThrow(/aligned binding must fit its buffer/);
      } finally {
        device.destroy();
      }
    }
  }
});

it('shared-output KDE expectations include the default uniform edge correction', () => {
  // Independent SciPy Gaussian sums and normal-CDF window integrals for the four cell centers.
  const expectedUncorrected = [
    0.11301202722345664, 0.1862683390450718, 0.25952465086668697, 0.33278096268830215
  ];
  const expectedUniform = [
    0.15321195615185323, 0.25252654337244423, 0.35184113059303523, 0.4511557178136263
  ];
  for (const method of ['exact', 'auto', 'binned'] as const) {
    for (const edgeCorrection of [undefined, 'none'] as const) {
      const fixture = getSharedOutputDensityFixture(method, edgeCorrection);
      const {device, graph, props} = createFixture([5]);
      try {
        const density = new GPUKernelDensity({
          ...props,
          ...fixture.settings,
          positions: method === 'binned' ? props.positions : props.positions.data[0],
          weights: method === 'binned' ? props.weights : props.weights.data[0],
          grid: fixture.grid,
          indexGridSize: [2, 2],
          indexBounds: [0, 0, 2, 2],
          normalization: 'probability'
        });
        expect(density.edgeCorrection).toBe(fixture.resolvedEdgeCorrection);
        expect(fixture.bandwidth).toBeCloseTo(0.4638694609608677, 14);
        const nodes = density.getCommandNodes(graph);
        const sources = getGPUCommandNodeKernelSources(nodes);
        for (const kernel of sources) {
          expect(getDensityReservedWords(kernel.source), kernel.id).toEqual([]);
          expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        }
        const finish = sources.find(
          kernel => kernel.id === `kernel-density-${method === 'binned' ? 'finish' : 'gather'}`
        )!;
        if (edgeCorrection === undefined) {
          expect(finish.source).toContain('total / mass');
        } else {
          expect(finish.source).not.toContain('total / mass');
        }
        if (method === 'binned') {
          expect(fixture.expectedValues[0]).toBeCloseTo(
            edgeCorrection === undefined ? 0.14283840180108498 : 0.10536029797113192,
            12
          );
        } else {
          const expected = edgeCorrection === undefined ? expectedUniform : expectedUncorrected;
          for (const [cell, value] of fixture.expectedValues.entries()) {
            expect(value).toBeCloseTo(expected[cell], 12);
          }
        }
      } finally {
        device.destroy();
      }
    }
  }
});

/** WebGPU storage usage compatibility is per physical buffer, regardless of binding ranges. */
function getMixedStorageUsageNodes<Parameters>(
  nodes: readonly GPUCommandNode<Parameters>[]
): string[] {
  return nodes
    .filter(node => {
      const usagesByBuffer = new Map<unknown, Set<string>>();
      for (const resource of node.resources ?? []) {
        if (!('buffer' in resource) || !resource.usage.startsWith('storage-')) {
          continue;
        }
        const handle =
          resource.buffer instanceof GraphDataView ? resource.buffer.buffer : resource.buffer;
        const buffer = handle.defaultBuffer ? getCoreDefaultBuffer(handle.defaultBuffer) : handle;
        const usages = usagesByBuffer.get(buffer) ?? new Set<string>();
        usages.add(resource.usage);
        usagesByBuffer.set(buffer, usages);
      }
      return [...usagesByBuffer.values()].some(
        usages =>
          usages.has('storage-read') &&
          (usages.has('storage-write') || usages.has('storage-read-write'))
      );
    })
    .map(node => node.id);
}

function createFixture(lengths: number[] = [7, 0, 264, 748], inputOffsets = false) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createVectorView(
    'positions',
    'float32x2',
    lengths.map((length, index) => {
      const prefixWords = inputOffsets ? index + 1 : 0;
      const whole = createTransientView(
        graph,
        `positions-${index}`,
        'float32x2',
        length + prefixWords
      );
      return graph.createDataView(whole.buffer, {
        format: 'float32x2',
        length,
        byteOffset: prefixWords * 4
      });
    })
  );
  const weights = createVectorView(
    'weights',
    'float32',
    lengths.map((length, index) => {
      const prefixWords = inputOffsets ? index + 1 : 0;
      const whole = createTransientView(
        graph,
        `weights-${index}`,
        'float32',
        Math.max(length, 1) + prefixWords
      );
      return graph.createDataView(whole.buffer, {
        format: 'float32',
        length,
        byteOffset: prefixWords * 4
      });
    })
  );
  const props = {
    positions,
    weights,
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: makeExtentGrid(8, 4, [0, 0, 8, 4]),
    indexGridSize: [4, 2] as const,
    indexBounds: [0, 0, 8, 4] as const,
    method: 'binned' as const,
    binnedPadding: 8,
    bandwidthRule: 'scott' as const,
    edgeCorrection: 'diggle' as const,
    normalization: 'probability' as const,
    output: {
      values: createTransientView(graph, 'values', 'float32', 32),
      weightSum: createTransientView(graph, 'weight-sum', 'float32', 1),
      status: {invalidCount: createTransientView(graph, 'invalid-count', 'uint32', 1)}
    }
  };
  return {device, graph, props};
}

it('binned KDE uses global row offsets, one scale and total-row headroom across uneven chunks', () => {
  const {device, graph, props} = createFixture();
  try {
    const density = new GPUKernelDensity(props);
    expect(props.positions.length).toBe(1019);
    const nodes = density.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(density.estimatedCommandNodeCount!);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const chunkIndex of [0, 2, 3]) {
      for (const stage of ['bandwidth-terms', 'bandwidth-spread', 'point-weights', 'bin']) {
        const kernel = sources.find(entry => entry.id === `kernel-density-${stage}-${chunkIndex}`)!;
        expect(kernel, stage).toBeDefined();
        expect(kernel.source).not.toContain('ROW_START');
        expect(kernel.source).toContain('var<uniform> densityDispatch');
        expect(getDensityReservedWords(kernel.source)).toEqual([]);
        expect(() => new WgslReflect(kernel.source)).not.toThrow();
        expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
        if (stage === 'bin') {
          expect(kernel.constants['FIXED_POINT_HEADROOM_BITS']).toBe(
            getFixedPointHeadroomBits(1019)
          );
          expect(kernel.source).toContain('pointWeights[densityDispatch.words[');
        }
      }
    }
    expect(
      nodes.some(node => /-(bin|point-weights|bandwidth-terms|bandwidth-spread)-1$/.test(node.id))
    ).toBe(false);
    expect(nodes.filter(node => node.id === 'kernel-density-bin-scale-clear')).toHaveLength(1);
    expect(nodes.filter(node => node.id === 'kernel-density-bin-clear')).toHaveLength(1);
    expect(nodes.filter(node => node.id === 'kernel-density-bin-decode')).toHaveLength(1);
    expect(nodes.some(node => node.id.includes('-index'))).toBe(false);
    expect(new Set(nodes.map(node => node.id)).size).toBe(nodes.length);
  } finally {
    device.destroy();
  }
});

it('binned KDE validates chunk format, alignment, graph ownership and packed-only routes', () => {
  const {device, graph, props} = createFixture();
  try {
    for (const method of ['exact', 'auto'] as const) {
      expect(() => new GPUKernelDensity({...props, method})).toThrow(
        /kernel-density positions must be one packed view/
      );
    }
    expect(
      () =>
        new GPUKernelDensity({
          ...props,
          weights: createVectorView('unaligned', 'float32', [
            createTransientView(graph, 'unaligned-first', 'float32', 8),
            createTransientView(graph, 'unaligned-last', 'float32', 1011)
          ])
        })
    ).toThrow(/kernel-density weights must match the source topology/);
    const invalidPositions = createVectorView('invalid', 'float32', [
      createTransientView(graph, 'invalid-position', 'float32', 1019)
    ]);
    expect(() => new GPUKernelDensity({...props, positions: invalidPositions as never})).toThrow(
      /kernel-density positions/
    );
    const foreign = new GPUCommandGraph(device);
    const foreignWeights = createVectorView(
      'foreign',
      'float32',
      [7, 0, 264, 748].map((length, index) =>
        createTransientView(foreign, `foreign-${index}`, 'float32', length)
      )
    );
    expect(() =>
      new GPUKernelDensity({...props, weights: foreignWeights}).getCommandNodes(graph)
    ).toThrow(/belong to the target graph/);
    const unweighted = new GPUKernelDensity({...props, weights: undefined});
    expect(() => unweighted.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('density binding checks reject packed scratch before allocating nodes or scratch views', () => {
  for (const scratch of ['point', 'grid', 'fixed-point', 'mask', 'binned-grid'] as const) {
    const {device, graph, props} = createFixture();
    try {
      const density =
        scratch === 'binned-grid'
          ? new GPUKernelDensity({...props, bandwidthRule: undefined})
          : new GPUPointDensity({
              positions: props.positions,
              weights: scratch === 'fixed-point' ? props.weights : undefined,
              statistic: scratch === 'fixed-point' ? 'sum' : 'count',
              binning: scratch === 'point' ? 'hexagon' : 'grid',
              hexagonRadius: scratch === 'point' ? 1 : undefined,
              mask:
                scratch === 'mask' ? createTransientView(graph, 'mask', 'uint32', 1019) : undefined,
              bounds: [0, 0, 8, 4],
              gridSize: [8, 4],
              output: {values: props.output.values}
            });
      const limit =
        scratch === 'point'
          ? 4075
          : scratch === 'mask'
            ? 4096
            : scratch === 'grid'
              ? 127
              : scratch === 'fixed-point'
                ? 383
                : 6000;
      device.limits.maxStorageBufferBindingSize = limit;
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => density.getCommandNodes(graph)).toThrow(/needs .* bytes in one binding/);
      expect(allocate).not.toHaveBeenCalled();
      allocate.mockRestore();
    } finally {
      device.destroy();
    }
  }
});

it('binned KDE splits point scratch when the total exceeds one deliberately small binding', () => {
  const {device, graph, props} = createFixture([256, 256, 256, 251]);
  try {
    const density = new GPUKernelDensity({...props, binnedPadding: 0});
    device.limits.maxStorageBufferBindingSize = 2048;
    device.limits.maxBufferSize = 2048;
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const nodes = density.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(density.estimatedCommandNodeCount!);
    expect(allocate).toHaveBeenCalled();
    for (const [allocation] of allocate.mock.calls) {
      expect(allocation.byteLength, allocation.id).toBeLessThanOrEqual(2048);
    }
    const weights = nodes.filter(node => /point-weights-\d+$/.test(node.id));
    expect(weights).toHaveLength(4);
    expect(weights.map(node => node.workload?.maximumInvocationCount)).toEqual([
      256, 256, 256, 251
    ]);
    allocate.mockRestore();
  } finally {
    device.destroy();
  }
});

it('chunk-capable density routes budget graph growth at construction', () => {
  const {device, props} = createFixture(new Array(8192).fill(0).concat(1));
  try {
    expect(() => new GPUKernelDensity(props)).toThrow(/estimated command node count/);
    expect(
      () =>
        new GPUPointDensity({
          positions: props.positions,
          bounds: [0, 0, 8, 4],
          gridSize: [8, 4],
          output: {values: props.output.values}
        })
    ).toThrow(/estimated command node count/);
  } finally {
    device.destroy();
  }
});

it('point density keeps its existing per-chunk fixed-point path and sums the logical rows', () => {
  const {device, graph, props} = createFixture();
  try {
    const density = new GPUPointDensity({
      positions: props.positions,
      weights: props.weights,
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      statistic: 'sum',
      output: {values: props.output.values}
    });
    const nodes = density.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(density.estimatedCommandNodeCount);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const chunkIndex of [0, 1, 2]) {
      const kernel = sources.find(
        entry => entry.id === `point-density-sums-workgroup-${chunkIndex}`
      )!;
      expect(kernel.source).not.toContain('KEY_START');
      expect(kernel.source).toContain('var<uniform> densityDispatch');
      expect(kernel.constants['HEADROOM_BITS']).toBe(getFixedPointHeadroomBits(1019));
    }
  } finally {
    device.destroy();
  }
});

it('packed KDE keeps exact as the default and preserves both gated auto routes', () => {
  const {device, graph, props} = createFixture();
  try {
    const packed = {
      ...props,
      positions: createTransientView(graph, 'packed-positions', 'float32x2', 1019),
      weights: createTransientView(graph, 'packed-weights', 'float32', 1019)
    };
    for (const method of [undefined, 'exact', 'auto', 'binned'] as const) {
      const id = `packed-${method ?? 'default'}`;
      const density = new GPUKernelDensity({...packed, id, method});
      expect(density.method).toBe(method ?? 'exact');
      const nodes = density.getCommandNodes(graph);
      const ids = nodes.map(node => node.id);
      expect(ids).toContain(`${id}-point-weights`);
      expect(ids.includes(`${id}-gather`)).toBe(method !== 'binned');
      expect(ids.includes(`${id}-bin`)).toBe(method === 'auto' || method === 'binned');
      if (method === 'auto') {
        expect(nodes.find(node => node.id === `${id}-gather`)?.condition).toBeDefined();
        expect(nodes.find(node => node.id === `${id}-bin`)?.condition).toBeDefined();
      }
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(getDensityReservedWords(kernel.source), kernel.id).toEqual([]);
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      }
    }
  } finally {
    device.destroy();
  }
});

it('thirty million chunked rows fit the packed key and corrected-weight scratch layouts', () => {
  const {device, graph, props} = createFixture([9_000_007, 10_000_003, 10_999_990]);
  try {
    expect(props.positions.length).toBe(30_000_000);
    const kernelDensity = new GPUKernelDensity(props);
    expect(kernelDensity.getCommandNodes(graph).length).toBeLessThanOrEqual(
      kernelDensity.estimatedCommandNodeCount!
    );
    const pointDensity = new GPUPointDensity({
      positions: props.positions,
      weights: props.weights,
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      statistic: 'sum',
      output: {values: props.output.values}
    });
    expect(pointDensity.getCommandNodes(graph).length).toBeLessThanOrEqual(
      pointDensity.estimatedCommandNodeCount
    );
  } finally {
    device.destroy();
  }
});

it('point density rejects 2^32 logical rows without allocating their data', () => {
  const {device, graph, props} = createFixture([4_194_304]);
  try {
    const positions = createVectorView(
      'repeated-positions',
      'float32x2',
      new Array(1024).fill(props.positions.data[0])
    );
    expect(positions.length).toBe(2 ** 32);
    expect(
      () =>
        new GPUPointDensity({
          positions,
          bounds: [0, 0, 1, 1],
          gridSize: [1, 1],
          output: {values: createTransientView(graph, 'one-cell', 'float32', 1)}
        })
    ).toThrow(
      `point-density positions.length must be at most ${GPU_POINT_DENSITY_MAXIMUM_POINT_COUNT}`
    );
  } finally {
    device.destroy();
  }
});

it('density point shaders and pipelines stay constant as uneven, unaligned chunks multiply', () => {
  const collect = (lengths: number[], kind: 'kde' | 'hexagon' | 'grid') => {
    const {device, graph, props} = createFixture(lengths, true);
    try {
      const density =
        kind === 'kde'
          ? new GPUKernelDensity({...props, binnedPadding: 0})
          : new GPUPointDensity({
              positions: props.positions,
              weights: props.weights,
              binning: kind,
              hexagonRadius: kind === 'hexagon' ? 1 : undefined,
              mask:
                kind === 'grid'
                  ? createTransientView(graph, 'mask', 'uint32', props.positions.length)
                  : undefined,
              statistic: 'sum',
              bounds: [0, 0, 8, 4],
              gridSize: [8, 4],
              output: {values: props.output.values}
            });
      if (kind === 'kde') device.limits.maxStorageBufferBindingSize = 9000;
      const nodes = density.getCommandNodes(graph);
      const sources = getGPUCommandNodeKernelSources(nodes);
      for (const kernel of sources) {
        expect(getDensityReservedWords(kernel.source), kernel.id).toEqual([]);
      }
      return {
        sources: new Set(sources.map(kernel => kernel.source)),
        keys: getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true})
      };
    } finally {
      device.destroy();
    }
  };
  for (const kind of ['kde', 'hexagon', 'grid'] as const) {
    const fewer = collect([1024, 1024, 1024], kind);
    const equalChunks = collect(new Array(12).fill(256), kind);
    expect([...equalChunks.sources].sort()).toEqual([...fewer.sources].sort());
    expect([...equalChunks.keys].sort()).toEqual([...fewer.keys].sort());
    const more = collect([63, 0, 64, 65, 255, 0, 256, 257, 400, 500, 600, 612], kind);
    expect([...more.sources].sort()).toEqual([...fewer.sources].sort());
    expect([...more.keys].sort()).toEqual([...fewer.keys].sort());
  }
});

it('density prechecks aligned input/output ranges before declaring any resources', () => {
  for (const kind of ['point', 'kde'] as const) {
    for (const field of ['positions', 'weights', 'parameters', 'output'] as const) {
      const {device, graph, props} = createFixture([8]);
      try {
        const format = field === 'positions' ? 'float32x2' : 'float32';
        const length = field === 'positions' ? 8 : field === 'output' ? 32 : 9;
        const buffer = graph.createTransientBuffer({id: 'prefix', byteLength: 512, usage: 128});
        const prefixed = graph.createDataView(buffer, {format, length, byteOffset: 252});
        const positions = field === 'positions' ? (prefixed as never) : props.positions;
        const weights =
          field === 'weights'
            ? createVectorView('prefixed-weight', 'float32', [
                graph.createDataView(buffer, {format: 'float32', length: 8, byteOffset: 252})
              ])
            : props.weights;
        const parameters =
          field === 'parameters' ? (prefixed as GraphDataView<'float32'>) : props.parameters;
        const output =
          field === 'output'
            ? {values: prefixed as GraphDataView<'float32'>}
            : {values: props.output.values};
        const density =
          kind === 'kde'
            ? new GPUKernelDensity({
                ...props,
                positions:
                  field === 'positions'
                    ? createVectorView('prefixed-position', 'float32x2', [prefixed as never])
                    : positions,
                weights,
                parameters,
                output
              })
            : new GPUPointDensity({
                positions,
                weights,
                bounds:
                  field === 'parameters'
                    ? graph.createDataView(buffer, {format: 'float32', length: 4, byteOffset: 252})
                    : [0, 0, 8, 4],
                gridSize: [8, 4],
                output
              });
        device.limits.maxStorageBufferBindingSize = 256;
        const allocate = vi.spyOn(graph, 'createTransientBuffer');
        expect(() => density.getCommandNodes(graph)).toThrow(/needs .* bytes in one binding/);
        expect(allocate).not.toHaveBeenCalled();
        allocate.mockRestore();
      } finally {
        device.destroy();
      }
    }
  }
});

it('density constructors reject overlapping writable fields, including scalar status outputs', () => {
  const {device, graph, props} = createFixture([8]);
  try {
    const shared = graph.createTransientBuffer({
      id: 'shared-outputs',
      byteLength: 1024,
      usage: 128
    });
    const values = graph.createDataView(shared, {format: 'float32', length: 32});
    for (const field of ['counts', 'sums', 'means', 'extent', 'histogram', 'overflow'] as const) {
      const scalar = field === 'overflow';
      const format = field === 'counts' || field === 'histogram' || scalar ? 'uint32' : 'float32';
      const length = field === 'extent' ? 2 : field === 'histogram' ? 4 : scalar ? 1 : 32;
      const overlap = graph.createDataView(shared, {format, length, byteOffset: 4});
      const output =
        field === 'overflow' ? {values, status: {overflow: overlap}} : {values, [field]: overlap};
      expect(
        () =>
          new GPUPointDensity({
            positions: props.positions,
            weights: props.weights,
            bounds: [0, 0, 8, 4],
            gridSize: [8, 4],
            output: output as never
          })
      ).toThrow(/must not overlap/);
    }
    for (const field of [
      'weightSum',
      'bandwidth',
      'invalidCount',
      'insufficientPadding'
    ] as const) {
      const format =
        field === 'invalidCount' || field === 'insufficientPadding' ? 'uint32' : 'float32';
      const overlap = graph.createDataView(shared, {format, length: 1, byteOffset: 4});
      const output =
        field === 'invalidCount' || field === 'insufficientPadding'
          ? {values, status: {[field]: overlap}}
          : {values, [field]: overlap};
      expect(() => new GPUKernelDensity({...props, output: output as never})).toThrow(
        /must not overlap/
      );
    }
    const scalar = graph.createDataView(shared, {format: 'float32', length: 1, byteOffset: 256});
    expect(
      () => new GPUKernelDensity({...props, output: {values, weightSum: scalar, bandwidth: scalar}})
    ).toThrow(/must not overlap/);
  } finally {
    device.destroy();
  }
});

it('density rejects a writable field aliasing any input chunk', () => {
  const {device, graph, props} = createFixture();
  try {
    const values = graph.createDataView(props.positions.data[2].buffer, {
      format: 'float32',
      length: 32,
      byteOffset: 4
    });
    expect(() => new GPUKernelDensity({...props, output: {values}})).toThrow(/share buffers/);
    expect(
      () =>
        new GPUPointDensity({
          positions: props.positions,
          weights: props.weights,
          bounds: [0, 0, 8, 4],
          gridSize: [8, 4],
          output: {values}
        })
    ).toThrow(/separate buffers/);
  } finally {
    device.destroy();
  }
});

it('smoothing FFT scratch is preflighted through the published convolution support plan', () => {
  const {device, graph, props} = createFixture([1]);
  try {
    const density = new GPUPointDensity({
      positions: props.positions,
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      smoothing: {
        kernel: createTransientView(graph, 'smoothing-kernel', 'float32', 9),
        kernelWidth: 3,
        kernelHeight: 3,
        strategy: 'fft'
      },
      output: {values: props.output.values}
    });
    device.limits.maxStorageBufferBindingSize = 256;
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => density.getCommandNodes(graph)).toThrow(/smoothing FFT scratch/);
    expect(allocate).not.toHaveBeenCalled();
    allocate.mockRestore();
  } finally {
    device.destroy();
  }
});

it('density total-row bounds accept UINT32_MAX and reject UINT32_MAX + 1 at construction', () => {
  const {device, graph, props} = createFixture([4_194_304]);
  try {
    const whole = props.positions.data[0];
    const last = graph.createDataView(whole.buffer, {format: 'float32x2', length: 4_194_303});
    const maximum = createVectorView('maximum-positions', 'float32x2', [
      ...new Array(1023).fill(whole),
      last
    ]);
    expect(maximum.length).toBe(GPU_POINT_DENSITY_MAXIMUM_POINT_COUNT);
    expect(
      () =>
        new GPUPointDensity({
          positions: maximum,
          bounds: [0, 0, 8, 4],
          gridSize: [8, 4],
          output: {values: props.output.values}
        })
    ).not.toThrow();
    expect(
      () => new GPUKernelDensity({...props, positions: maximum, weights: undefined})
    ).not.toThrow();
    const overflow = createVectorView(
      'overflow-positions',
      'float32x2',
      new Array(1024).fill(whole)
    );
    expect(() => new GPUKernelDensity({...props, positions: overflow, weights: undefined})).toThrow(
      /kernel-density positions.length must be at most 4294967295/
    );
  } finally {
    device.destroy();
  }
});

it('polygon KDE point kernels parse and avoid the full WGSL reserved-word list at eight storage bindings', () => {
  const {device, graph, props} = createFixture();
  try {
    const studyWindow = new GPUStudyWindowPreparation({
      positions: createTransientView(graph, 'window-positions', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'window-features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'window-polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'window-rings', 'uint32', 2),
      samplingGrid: [2, 2]
    });
    graph.add(studyWindow);
    for (const kernel of getGPUCommandNodeKernelSources(
      new GPUKernelDensity({...props, studyWindow}).getCommandNodes(graph)
    )) {
      expect(getDensityReservedWords(kernel.source), kernel.id).toEqual([]);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
    }
  } finally {
    device.destroy();
  }
});

it('packed defaults and inferred packed/chunked inputs preserve packed grid output types', () => {
  const {device, props} = createFixture([8]);
  try {
    const packed = {...props, positions: props.positions.data[0], weights: props.weights.data[0]};
    const kernelDensity = new GPUKernelDensity(packed);
    expectTypeOf<GPUKernelDensityProps['positions']>().toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf<GPUPointDensityProps['positions']>().toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf(kernelDensity.props.positions).toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf(kernelDensity.props.weights).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
    expectTypeOf(kernelDensity.props.output.values).toEqualTypeOf<GraphDataView<'float32'>>();
    expectTypeOf(kernelDensity.props.output.weightSum).toEqualTypeOf<
      GraphDataView<'float32'> | undefined
    >();
    const pointDensity = new GPUPointDensity({
      positions: packed.positions,
      weights: packed.weights,
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      output: {values: packed.output.values}
    });
    expectTypeOf(pointDensity.props.output.values).toEqualTypeOf<GraphDataView<'float32'>>();
    expectTypeOf(pointDensity.props.positions).toEqualTypeOf<GraphDataView<'float32x2'>>();
    expectTypeOf(pointDensity.props.weights).toEqualTypeOf<GraphDataView<'float32'> | undefined>();
    expectTypeOf(pointDensity.props.output.counts).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    expect(kernelDensity.props.output.values.buffer).toBe(props.output.values.buffer);
    expect(pointDensity.props.output.values.buffer).toBe(props.output.values.buffer);
    const chunkedKernelDensity = new GPUKernelDensity(props);
    const chunkedPointDensity = new GPUPointDensity({
      ...pointDensity.props,
      positions: props.positions,
      weights: props.weights
    });
    expectTypeOf(chunkedKernelDensity.props.positions).toEqualTypeOf<
      GraphVectorView<'float32x2'>
    >();
    expectTypeOf(chunkedKernelDensity.props.weights).toEqualTypeOf<
      GraphVectorView<'float32'> | undefined
    >();
    expectTypeOf(chunkedPointDensity.props.positions).toEqualTypeOf<GraphVectorView<'float32x2'>>();
    expectTypeOf(chunkedPointDensity.props.weights).toEqualTypeOf<
      GraphVectorView<'float32'> | undefined
    >();
    // Grid rows describe cells rather than input points, so outputs stay packed for both kinds.
    expectTypeOf(chunkedKernelDensity.props.output.values).toEqualTypeOf<
      GraphDataView<'float32'>
    >();
    expectTypeOf(chunkedPointDensity.props.output.values).toEqualTypeOf<GraphDataView<'float32'>>();
  } finally {
    device.destroy();
  }
});

it('density shared-buffer outputs avoid mixed storage usage and accept separate dispatch binding prefixes', () => {
  const {device, graph, props} = createFixture([8]);
  try {
    const shared = graph.createTransientBuffer({
      id: 'aligned-output-storage',
      byteLength: 1792,
      usage: 128
    });
    const values = graph.createDataView(shared, {format: 'float32', length: 32});
    const counts = graph.createDataView(shared, {format: 'uint32', length: 32, byteOffset: 256});
    const sums = graph.createDataView(shared, {format: 'float32', length: 32, byteOffset: 512});
    const means = graph.createDataView(shared, {format: 'float32', length: 32, byteOffset: 768});
    const extent = graph.createDataView(shared, {format: 'float32', length: 2, byteOffset: 1024});
    const histogram = graph.createDataView(shared, {format: 'uint32', length: 4, byteOffset: 1280});
    const weightSum = graph.createDataView(shared, {format: 'float32', length: 1, byteOffset: 256});
    const bandwidth = graph.createDataView(shared, {format: 'float32', length: 1, byteOffset: 512});
    const invalidCount = graph.createDataView(shared, {
      format: 'uint32',
      length: 1,
      byteOffset: 768
    });
    const insufficientPadding = graph.createDataView(shared, {
      format: 'uint32',
      length: 1,
      byteOffset: 1024
    });
    const pointDensity = new GPUPointDensity({
      positions: props.positions,
      weights: props.weights,
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      statistic: 'sum',
      output: {values, counts, sums, means, extent, histogram}
    });
    const pointNodes = pointDensity.getCommandNodes(graph);
    // This is the pre-fix finalize dispatch. Its logical ranges are disjoint, but Metal/WebGPU
    // invalidates the whole command buffer because one buffer has mixed storage usages.
    const invalidFinalize = createPointDensityFinalizeNode(graph, {
      id: 'pre-fix-shared-finalize',
      statistic: 'sum',
      values,
      counts,
      sums,
      means
    });
    expect(getMixedStorageUsageNodes([invalidFinalize])).toEqual(['pre-fix-shared-finalize']);
    expect(getMixedStorageUsageNodes(pointNodes)).toEqual([]);
    expect(
      getGPUCommandNodePipelineKeys(pointNodes, {
        idPattern: /-publish-/,
        includeShaderLayout: true
      }).size
    ).toBe(1);
    for (const name of ['values', 'counts', 'sums', 'means', 'extent', 'histogram']) {
      const publish = pointNodes.find(node => node.id === `point-density-publish-${name}`)!;
      expect(publish, name).toBeDefined();
      const output = publish.resources!.find(resource => resource.usage === 'storage-read-write')!;
      expect(
        'buffer' in output && output.buffer instanceof GraphDataView && output.buffer.buffer
      ).toBe(shared);
    }
    expect(pointNodes.map(node => node.id)).toContain('point-density-publish-extent');
    expect(pointNodes.map(node => node.id)).toContain('point-density-publish-histogram');
    expect(pointNodes.length).toBeLessThanOrEqual(pointDensity.estimatedCommandNodeCount);
    for (const kernel of getGPUCommandNodeKernelSources(pointNodes)) {
      expect(getDensityReservedWords(kernel.source), kernel.id).toEqual([]);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
    for (const method of ['binned', 'exact', 'auto'] as const) {
      const density = new GPUKernelDensity({
        ...props,
        id: `kernel-density-${method}`,
        positions: props.positions.data[0],
        weights: props.weights.data[0],
        method,
        output: {
          values,
          weightSum,
          bandwidth,
          status: {
            invalidCount,
            insufficientPadding: method === 'exact' ? undefined : insufficientPadding
          }
        }
      });
      const nodes = density.getCommandNodes(graph);
      expect(getMixedStorageUsageNodes(nodes), method).toEqual([]);
      expect(
        getGPUCommandNodePipelineKeys(nodes, {
          idPattern: /-publish-/,
          includeShaderLayout: true
        }).size
      ).toBe(1);
      if (density.estimatedCommandNodeCount !== undefined) {
        expect(nodes.length).toBeLessThanOrEqual(density.estimatedCommandNodeCount);
      }
      // The exact gather must write its staged field too, rather than props.output.values.
      for (const node of nodes.filter(candidate => !candidate.id.includes('-publish-'))) {
        for (const resource of node.resources ?? []) {
          if ('buffer' in resource && resource.buffer instanceof GraphDataView) {
            expect(resource.buffer.buffer, node.id).not.toBe(shared);
          }
        }
      }
      for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
        expect(getDensityReservedWords(kernel.source), kernel.id).toEqual([]);
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      }
    }
    // Logical bytes [0, 128) and [128, 132) are disjoint; their aligned bindings overlap.
    const prefixedScalar = graph.createDataView(shared, {
      format: 'float32',
      length: 1,
      byteOffset: 128
    });
    const packedKernel = new GPUKernelDensity({
      ...props,
      id: 'packed-prefix-kernel',
      output: {values, weightSum: prefixedScalar}
    });
    expect(getMixedStorageUsageNodes(packedKernel.getCommandNodes(graph))).toEqual([]);
    const prefixedCounts = graph.createDataView(shared, {
      format: 'uint32',
      length: 32,
      byteOffset: 128
    });
    const packedPoint = new GPUPointDensity({
      id: 'packed-prefix-point',
      positions: props.positions,
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      output: {values, counts: prefixedCounts}
    });
    expect(getMixedStorageUsageNodes(packedPoint.getCommandNodes(graph))).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('chunk point passes use each chunk workgroup count without size overrides or largest-chunk padding', () => {
  const {device, graph, props} = createFixture([1, 65, 4097, 4097, 4097]);
  try {
    device.limits.maxStorageBufferBindingSize = 32776;
    const nodes = new GPUKernelDensity(props).getCommandNodes(graph);
    const sources = getGPUCommandNodeKernelSources(nodes);
    for (const [chunkIndex, length] of [1, 65, 4097, 4097, 4097].entries()) {
      for (const stage of ['bandwidth-terms', 'bandwidth-spread', 'point-weights', 'bin']) {
        const id = `kernel-density-${stage}-${chunkIndex}`;
        const node = nodes.find(candidate => candidate.id === id)!;
        expect(node.workload?.maximumInvocationCount, id).toBe(length);
        expect(node.workload?.maximumWorkgroupCount, id).toBe(Math.ceil(length / 64));
        const kernel = sources.find(candidate => candidate.id === id)!;
        expect(kernel.constants, id).not.toHaveProperty('INVOCATION_COUNT');
        expect(kernel.constants, id).not.toHaveProperty('WGSL_KERNEL_DISPATCH_X');
        expect(kernel.constants, id).not.toHaveProperty('ROW_START');
      }
    }
    for (const [chunkIndex, length] of [1, 65, 4097, 4097, 4097].entries()) {
      for (const stage of [
        'bandwidth-sum-weight-accumulate',
        'weight-sum-accumulate',
        'invalid-count'
      ]) {
        const node = nodes.find(
          candidate => candidate.id === `kernel-density-${stage}-${chunkIndex}`
        )!;
        expect(node.workload?.maximumInvocationCount).toBe(length);
        expect(node.workload?.maximumWorkgroupCount).toBe(Math.ceil(length / 256));
      }
    }
    expect(getDensityReservedWords('fn sample() { let value = 1u; }')).toEqual([]);
    expect(
      getDensityReservedWords(
        'fn target() { let active = 1u; let filter = active; let module = filter; let meta = module; }'
      )
    ).toEqual(['target', 'active', 'filter', 'module', 'meta']);
  } finally {
    device.destroy();
  }
});

it('density imports shared physical output storage once and publishes only written rows', () => {
  const {device, graph, props} = createFixture([8]);
  const allocation = device.createBuffer({byteLength: 1024, usage: 128});
  try {
    const first = graph.importBuffer(
      {id: 'first-output-handle', byteLength: 1024, usage: 128},
      allocation
    );
    // One physical buffer is imported once; all output fields are views of that handle.
    const second = first;
    const values = graph.createDataView(first, {format: 'float32', length: 32});
    const counts = graph.createDataView(second, {format: 'uint32', length: 32, byteOffset: 256});
    const pointNodes = new GPUPointDensity({
      positions: props.positions,
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      output: {values, counts}
    }).getCommandNodes(graph);
    expect(getMixedStorageUsageNodes(pointNodes)).toEqual([]);
    expect(pointNodes.filter(node => node.id.includes('-publish-'))).toHaveLength(2);
    const kernelNodes = new GPUKernelDensity({
      ...props,
      output: {
        values: graph.createDataView(first, {format: 'float32', length: 40}),
        weightSum: graph.createDataView(second, {format: 'float32', length: 1, byteOffset: 256})
      }
    }).getCommandNodes(graph);
    expect(getMixedStorageUsageNodes(kernelNodes)).toEqual([]);
    const publish = kernelNodes.find(node => node.id === 'kernel-density-publish-values')!;
    expect(publish.workload?.maximumInvocationCount).toBe(32);
    expect(publish.workload?.maximumWorkgroupCount).toBe(1);
    const source = publish.resources!.find(resource => resource.usage === 'storage-read')!;
    expect(
      'buffer' in source && source.buffer instanceof GraphDataView && source.buffer.length
    ).toBe(32);
  } finally {
    allocation.destroy();
    device.destroy();
  }
});

it('density rejects non-representable shader literals at construction', () => {
  const {device, graph, props} = createFixture([8]);
  try {
    for (const value of [Number.NaN, Infinity, -Infinity, 1e39, -1e39]) {
      expect(() => new GPUKernelDensity({...props, indexBounds: [value, 0, 8, 4]})).toThrow(
        /kernel-density indexBounds\[0\] must be a finite number/
      );
    }
    for (const value of [0x100000000, Number.MAX_SAFE_INTEGER, Infinity, Number.NaN]) {
      expect(() => new GPUKernelDensity({...props, indexGridSize: [value, 2]})).toThrow(
        /kernel-density indexGridSize\[0\] must be an integer in \[1, 4294967295\]/
      );
    }
    const pointProps = {
      positions: props.positions,
      bounds: [0, 0, 8, 4] as const,
      gridSize: [8, 4] as const,
      output: {values: props.output.values}
    };
    for (const value of [Number.NaN, Infinity, 1e39, 1e-50]) {
      expect(
        () => new GPUPointDensity({...pointProps, binning: 'hexagon', hexagonRadius: value})
      ).toThrow(/point-density hexagonRadius/);
    }
    expect(
      () =>
        new GPUPointDensity({
          ...pointProps,
          smoothing: {
            kernel: createTransientView(graph, 'kernel', 'float32', 9),
            kernelWidth: 0x100000001,
            kernelHeight: 1
          }
        })
    ).toThrow(/point-density smoothing.kernelWidth must be an integer in \[1, 4294967295\]/);
    const literalNodes = new GPUKernelDensity({
      ...props,
      indexBounds: [-1e30, -1e30, 1e30, 1e30]
    }).getCommandNodes(graph);
    for (const kernel of getGPUCommandNodeKernelSources(literalNodes)) {
      expect(getDensityReservedWords(kernel.source), kernel.id).toEqual([]);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('KDE refuses an oversized index by its named capacity before declaring scratch', () => {
  for (const kernel of ['gaussian', 'epanechnikov'] as const) {
    const {device, graph, props} = createFixture([8]);
    try {
      const density = new GPUKernelDensity({
        ...props,
        positions: props.positions.data[0],
        weights: props.weights.data[0],
        method: 'exact',
        kernel,
        indexGridSize: [8192, 4096]
      });
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      try {
        // 33,554,432 cells require 134,217,732 offset bytes, four above the binding limit.
        expect(() => density.getCommandNodes(graph)).toThrow(
          /^kernel-density indexGridSize cells must be at most 33554431$/
        );
        expect(allocate).not.toHaveBeenCalled();
      } finally {
        allocate.mockRestore();
      }
    } finally {
      device.destroy();
    }
  }
});

it('unmasked grid density splits deterministic keys and borrows unequal weight spans within a small binding', () => {
  const {device, graph, props} = createFixture([256, 256, 256, 256]);
  try {
    const weights = createVectorView(
      'different-weight-chunks',
      'float32',
      [3, 509, 512].map((length, index) =>
        createTransientView(graph, `different-weight-${index}`, 'float32', length)
      )
    );
    const density = new GPUPointDensity({
      positions: props.positions,
      weights,
      statistic: 'sum',
      bounds: [0, 0, 8, 4],
      gridSize: [8, 4],
      output: {values: props.output.values}
    });
    device.limits.maxStorageBufferBindingSize = 2048;
    device.limits.maxBufferSize = 2048;
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const nodes = density.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(density.estimatedCommandNodeCount);
    for (const [allocation] of allocate.mock.calls)
      expect(allocation.byteLength, allocation.id).toBeLessThanOrEqual(2048);
    const sums = nodes.filter(node => /sums-workgroup-/.test(node.id));
    expect(sums.map(node => node.workload?.maximumInvocationCount)).toEqual([
      3, 253, 256, 256, 256
    ]);
    for (const kernel of getGPUCommandNodeKernelSources(sums)) {
      expect(getDensityReservedWords(kernel.source)).toEqual([]);
      expect(() => new WgslReflect(kernel.source)).not.toThrow();
    }
    expect(getGPUCommandNodePipelineKeys(sums, {includeShaderLayout: true}).size).toBe(1);
    allocate.mockRestore();
  } finally {
    device.destroy();
  }
});

it('density rejects maxBufferSize and dispatch-uniform failures without graph mutation', () => {
  for (const invalidLimit of ['buffer', 'uniform'] as const) {
    const {device, graph, props} = createFixture([256, 256]);
    try {
      const density = new GPUKernelDensity({...props, binnedPadding: 0});
      if (invalidLimit === 'buffer') device.limits.maxBufferSize = 2047;
      else device.limits.maxUniformBufferBindingSize = 63;
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      const add = vi.spyOn(graph, 'add');
      expect(() => density.getCommandNodes(graph)).toThrow(
        invalidLimit === 'buffer' ? /buffer byteLength/ : /maxUniformBufferBindingSize/
      );
      expect(allocate).not.toHaveBeenCalled();
      expect(add).not.toHaveBeenCalled();
      allocate.mockRestore();
      add.mockRestore();
    } finally {
      device.destroy();
    }
  }
});

it('density preflight rejects missing STORAGE usage and nested binned scratch buffer limits atomically', () => {
  for (const failure of ['storage', 'scratch'] as const) {
    const {device, graph, props} = createFixture([8]);
    try {
      const values =
        failure === 'storage'
          ? graph.createDataView(
              graph.createTransientBuffer({id: 'non-storage', usage: 8, byteLength: 128}),
              {format: 'float32', length: 32}
            )
          : props.output.values;
      const density = new GPUKernelDensity({...props, output: {values}, binnedPadding: 12});
      if (failure === 'scratch') device.limits.maxBufferSize = 8192;
      const allocate = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => density.getCommandNodes(graph)).toThrow(
        failure === 'storage' ? /Buffer.STORAGE/ : /binned fixed-point grid scratch byteLength/
      );
      expect(allocate).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

for (const binning of ['grid', 'hexagon'] as const)
  it(`fixed density tiles keep all shaders and pipelines constant for ${binning}`, () => {
    const sets = [130, 177, 225].map(rows => {
      const {device, graph, props} = createFixture([3, 0, rows - 3]);
      try {
        const density = new GPUPointDensity({
          positions: props.positions,
          weights: props.weights,
          scratchTileSize: 16,
          binning,
          hexagonRadius: binning === 'hexagon' ? 1 : undefined,
          mask: binning === 'grid' ? createTransientView(graph, 'mask', 'uint32', rows) : undefined,
          bounds: [0, 0, 8, 4],
          gridSize: [8, 4],
          statistic: 'sum',
          output: {values: props.output.values}
        });
        const nodes = density.getCommandNodes(graph);
        expect(nodes.length).toBeLessThanOrEqual(density.estimatedCommandNodeCount);
        for (const {source, id} of getGPUCommandNodeKernelSources(nodes)) {
          expect(getDensityReservedWords(source), id).toEqual([]);
          expect(() => new WgslReflect(source), id).not.toThrow();
        }
        const gathers = nodes.filter(node => node.id.includes('-gather-'));
        expect(gathers.some(node => node.workload?.maximumInvocationCount === 3)).toBe(true);
        return getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true});
      } finally {
        device.destroy();
      }
    });
    expect([...sets[0]].sort()).toEqual([...sets[1]].sort());
    expect([...sets[0]].sort()).toEqual([...sets[2]].sort());
  });

it('density rejects padded workgroups before a uint32 row index can wrap', () => {
  const {device, graph} = createFixture([1]);
  try {
    const node = createDensityPointPass(graph, {
      id: 'uint32-dispatch',
      operation: 'GPUPointDensity',
      invocationCount: 0xffffffff,
      bindings: [],
      body: '_ = index;'
    });
    const [kernel] = getGPUCommandNodeKernelSources([node]);
    expect(node.workload?.maximumWorkgroupCount).toBeGreaterThan(0x4000000);
    expect(kernel.source).toContain('if (workgroupIndex > 67108863u) { return; }');
    expect(getDensityReservedWords(kernel.source)).toEqual([]);
    expect(() => new WgslReflect(kernel.source)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('density tile widths reject non-finite, unaddressable and non-power-of-two sizes at construction', () => {
  const {device, props} = createFixture([1]);
  try {
    for (const scratchTileSize of [NaN, Infinity, -1, 0, 3, 2 ** 31])
      expect(
        () =>
          new GPUPointDensity({
            positions: props.positions,
            scratchTileSize,
            bounds: [0, 0, 8, 4],
            gridSize: [8, 4],
            output: {values: props.output.values}
          })
      ).toThrow(/scratchTileSize/);
  } finally {
    device.destroy();
  }
});
