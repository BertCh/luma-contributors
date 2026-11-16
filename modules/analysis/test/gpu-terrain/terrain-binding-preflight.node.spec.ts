// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  getViewBindingRange,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import {TERRAIN_CASES} from '../analysis-core/kernel-audit/cases-terrain';
import {
  createKernelAuditContext,
  type KernelAuditCase
} from '../analysis-core/kernel-audit/kernel-audit-context';
import {
  getTerrainBindingViews,
  validateTerrainBindings
} from '../../src/gpu-terrain/terrain-processing/terrain-binding-preflight';
import {getGPURasterExtremaPyramidLayout} from '../../src/raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid';
import {GPUTerrainViewshed} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed';
import {GPUTerrainLineOfSight} from '../../src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight';
import {GPUTerrainSummits} from '../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {GPUTerrainFlowDistance} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainHorizon} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {GPUTerrainFlow} from '../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUReliefShading} from '../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {GPUSolarIrradiance} from '../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {GPUTerrainStreamNetwork} from '../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';

const CONTRIBUTORS = [
  'GPUGeomorphons',
  'GPUTerrainFlow',
  'GPUTerrainFlowDistance',
  'GPUTerrainHydrologicIndices',
  'GPUTerrainStreamNetwork',
  'GPUTerrainStreamOrder',
  'GPUTerrainWatersheds',
  'GPUPointHorizonProfile',
  'GPUPointHorizonVisibility',
  'GPULocalDominance',
  'GPUMultiScaleRelief',
  'GPUReliefBlend',
  'GPUSimpleLocalRelief',
  'GPUTerrainContours',
  'GPUTerrainCumulativeViewshed',
  'GPUTerrainDerivatives',
  'GPUTerrainLineOfSight',
  'GPUTerrainViewshed',
  'GPUTerrainCurvature',
  'GPUTerrainRGBDecode',
  'GPUProfilePeaks',
  'GPUTerrainCriticalPoints',
  'GPUTerrainSummits',
  'GPUTerrainFlowField',
  'GPUReliefShading',
  'GPUSolarIrradiance',
  'GPUSolarPosition',
  'GPUSolarShadowMask',
  'GPUTerrainCastShadow',
  'GPUTerrainHorizon',
  'GPUTextureShading',
  'GPUTerrainRuggedness',
  'GPUTerrainVectorRuggedness'
] as const;

const GRID = {
  width: 4,
  height: 4,
  affine: [1, 0, 0, 0, -1, 0] as const,
  pixelInterpretation: 'area' as const
};
const EXTRA_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUTerrainFlowDistance',
    name: 'pointer state',
    build: ({graph, view}) => {
      graph.add(
        new GPUTerrainFlowDistance({
          grid: GRID,
          flowDirections: view('uint32', 16),
          output: {
            distance: view('float32', 16),
            validity: view('uint32', 16),
            terminalCells: view('uint32', 16)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainStreamNetwork',
    name: 'pointer and compaction state',
    build: ({graph, view}) => {
      graph.add(
        new GPUTerrainStreamNetwork({
          grid: GRID,
          flowDirections: view('uint32', 16),
          streams: view('uint32', 16),
          output: {
            sourceCells: view('uint32', 16),
            targetCells: view('uint32', 16),
            lengths: view('float32', 16),
            positions: view('float32x2', 32),
            vertexOffsets: view('uint32', 17),
            reachCount: view('uint32', 1),
            vertexCount: view('uint32', 1),
            status: {overflow: view('uint32', 1)},
            unresolvedCount: view('uint32', 1)
          }
        })
      );
    }
  }
];

type Contributor = GPUCommandNodeProducer & {readonly props: unknown};

function createFixture(name: string, selectedCase?: KernelAuditCase) {
  const context = createKernelAuditContext(`terrain-preflight-${name}`);
  // Real, valid prefix ranges: every view starts one format-sized row into its backing buffer.
  const originalView = context.view;
  context.view = (format, length, label) => {
    const storage = originalView(format, length + 1, label);
    return context.graph.createDataView(storage.buffer, {
      format,
      length,
      byteOffset: storage.rowByteLength
    });
  };
  let contributor: Contributor | undefined;
  const declaration = vi.spyOn(context.graph, 'add').mockImplementation(node => {
    if ('getCommandNodes' in node) contributor = node as Contributor;
  });
  const auditCase =
    selectedCase ??
    [...EXTRA_CASES, ...TERRAIN_CASES].find(candidate => candidate.exportName === name)!;
  try {
    auditCase.build(context);
  } finally {
    declaration.mockRestore();
  }
  expect(contributor, name).toBeDefined();
  const views = Object.values(getTerrainBindingViews(contributor!.props)) as GraphDataView[];
  return {context, contributor: contributor!, views};
}

function watchDeclarations(context: ReturnType<typeof createKernelAuditContext>) {
  const buffers = vi.spyOn(context.graph, 'createTransientBuffer');
  const textures = vi.spyOn(context.graph, 'createTransientTexture');
  const nodes = vi.spyOn(context.graph, 'add');
  return {
    expectUnchanged() {
      expect(buffers).not.toHaveBeenCalled();
      expect(textures).not.toHaveBeenCalled();
      expect(nodes).not.toHaveBeenCalled();
    },
    restore() {
      buffers.mockRestore();
      textures.mockRestore();
      nodes.mockRestore();
    }
  };
}

for (const name of CONTRIBUTORS) {
  describe(name, () => {
    it('rejects the actual aligned caller binding before any declaration', () => {
      const {context, contributor, views} = createFixture(name);
      try {
        const largestBinding = Math.max(...views.map(view => getViewBindingRange(view).size));
        const largestLogicalRange = Math.max(
          ...views.map(view => (view.length - 1) * view.byteStride + view.rowByteLength)
        );
        expect(largestBinding).toBeGreaterThan(largestLogicalRange);
        Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {
          value: largestBinding - 1,
          configurable: true
        });
        const declarations = watchDeclarations(context);
        try {
          expect(() => contributor.getCommandNodes(context.graph)).toThrow(
            /bytes in one binding \(limit/
          );
          declarations.expectUnchanged();
        } finally {
          declarations.restore();
        }
      } finally {
        context.device.destroy();
      }
    });

    it('rejects every caller buffer missing STORAGE before any declaration', () => {
      const {context, contributor, views} = createFixture(name);
      try {
        for (const buffer of new Set(views.map(view => view.buffer))) {
          const usage = buffer.usage;
          Object.defineProperty(buffer, 'usage', {
            value: usage & ~Buffer.STORAGE,
            configurable: true
          });
          const declarations = watchDeclarations(context);
          try {
            expect(() => contributor.getCommandNodes(context.graph), buffer.id).toThrow(/STORAGE/);
            declarations.expectUnchanged();
          } finally {
            declarations.restore();
            Object.defineProperty(buffer, 'usage', {value: usage, configurable: true});
          }
        }
      } finally {
        context.device.destroy();
      }
    });

    it('accepts valid prefixed views and preflights the largest nested allocation', () => {
      const {context, contributor} = createFixture(name);
      try {
        const allocations = vi.spyOn(context.graph, 'createTransientBuffer');
        expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
        const largestScratch = Math.max(
          0,
          ...allocations.mock.calls.map(([descriptor]) => descriptor.byteLength)
        );
        for (const [descriptor] of allocations.mock.calls)
          expect(descriptor.usage & Buffer.STORAGE).not.toBe(0);
        allocations.mockRestore();
        if (!largestScratch) return;
        // Keep the route selection unchanged while denying its largest allocation. Caller-owned
        // buffers already exist; no resources from this second declaration may be added.
        Object.defineProperty(context.device.limits, 'maxBufferSize', {
          value: largestScratch - 1,
          configurable: true
        });
        const declarations = watchDeclarations(context);
        try {
          expect(() => contributor.getCommandNodes(context.graph)).toThrow(/allocation byteLength/);
          declarations.expectUnchanged();
        } finally {
          declarations.restore();
        }
      } finally {
        context.device.destroy();
      }
    });
  });
}

it('accepts storage-aligned writable ranges sharing one imported buffer', () => {
  const context = createKernelAuditContext('terrain-packed-bindings');
  const buffer = context.device.createBuffer({
    byteLength: 512,
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  try {
    const handle = context.graph.importBuffer(
      {id: 'packed-once', byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    const fields = {
      first: context.graph.createDataView(handle, {format: 'uint32', length: 1}),
      second: context.graph.createDataView(handle, {format: 'float32', length: 1, byteOffset: 256})
    };
    expect(() => validateTerrainBindings(context.graph, 'packed', fields)).not.toThrow();
  } finally {
    buffer.destroy();
    context.device.destroy();
  }
});

// Exercise nested allocation routes as well as the first configuration of each contributor.
for (const auditCase of TERRAIN_CASES.filter(
  candidate =>
    [
      'GPUTerrainFlow',
      'GPUPointHorizonProfile',
      'GPUPointHorizonVisibility',
      'GPUTerrainDerivatives',
      'GPUReliefShading',
      'GPUTerrainHorizon',
      'GPUTerrainLineOfSight',
      'GPUTerrainViewshed',
      'GPUTerrainCumulativeViewshed',
      'GPUTerrainSummits'
    ].includes(candidate.exportName) &&
    !/maximum|binding limit|large|at the limit/.test(candidate.name)
)) {
  it(`preflights nested scratch: ${auditCase.exportName}, ${auditCase.name}`, () => {
    const {context, contributor} = createFixture(auditCase.exportName, auditCase);
    try {
      const allocations = vi.spyOn(context.graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
      const maximumScratch = Math.max(
        ...allocations.mock.calls.map(([descriptor]) => descriptor.byteLength)
      );
      allocations.mockRestore();
      expect(maximumScratch).toBeGreaterThan(0);
      Object.defineProperty(context.device.limits, 'maxBufferSize', {
        value: maximumScratch - 1,
        configurable: true
      });
      const declarations = watchDeclarations(context);
      try {
        expect(() => contributor.getCommandNodes(context.graph)).toThrow(/allocation byteLength/);
        declarations.expectUnchanged();
      } finally {
        declarations.restore();
      }
    } finally {
      context.device.destroy();
    }
  });
}

it('rejects a derived horizon map binding before declaring its target or canonical band', () => {
  const context = createKernelAuditContext('horizon-derived-map');
  try {
    const elevation = context.view('float32', 16);
    const contributor = new GPUTerrainHorizon({
      grid: GRID,
      elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values: elevation}},
      parameters: context.view('float32', 8),
      directionCount: 16,
      maximumRadius: 2,
      output: {irradianceSkyViewFactor: context.view('float32', 16)}
    });
    Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {
      value: 1020,
      configurable: true
    });
    const declarations = watchDeclarations(context);
    try {
      expect(() => contributor.getCommandNodes(context.graph)).toThrow(
        /horizon map needs 1024 bytes/
      );
      declarations.expectUnchanged();
    } finally {
      declarations.restore();
    }
  } finally {
    context.device.destroy();
  }
});

it('does not impose storage requirements on unused optional views', () => {
  const context = createKernelAuditContext('unused-views');
  const unused = (length: number) => {
    const handle = context.graph.createTransientBuffer({
      id: `unused-${length}-${serial++}`,
      byteLength: length * 4,
      usage: Buffer.COPY_DST
    });
    return context.graph.createDataView(handle, {format: 'float32', length});
  };
  let serial = 0;
  try {
    const band = () => ({
      id: 'elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: context.view('float32', 16)}
    });
    const flow = new GPUTerrainFlow({
      grid: GRID,
      elevation: band(),
      parameters: context.view('float32', 8),
      runoff: unused(16),
      output: {flowDirections: context.view('uint32', 16)}
    });
    const relief = new GPUReliefShading({
      grid: GRID,
      elevation: band(),
      parameters: context.view('float32', 80),
      skyViewFactor: unused(16),
      textureShade: unused(16),
      curvature: unused(16),
      output: {hillshade: context.view('float32', 16)}
    });
    const irradiance = new GPUSolarIrradiance({
      grid: GRID,
      horizon: context.view('float32', 64),
      directionCount: 4,
      sampleCapacity: 1,
      sunTable: unused(4),
      parameters: unused(8),
      output: {outputValidity: context.view('uint32', 16)}
    });
    for (const contributor of [flow, relief, irradiance])
      expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
  } finally {
    context.device.destroy();
  }
});

for (const kind of ['viewshed', 'line-of-sight', 'summits'] as const) {
  for (const limit of ['binding', 'allocation'] as const) {
    it(`preflights the nested pyramid parameter ${limit} before declarations (${kind})`, () => {
      const context = createKernelAuditContext(`pyramid-parameters-${kind}-${limit}`);
      try {
        const grid = {
          ...GRID,
          width: limit === 'binding' ? 5 : 9,
          height: limit === 'binding' ? 2 : 7
        };
        const count = grid.width * grid.height;
        const common = {
          grid,
          elevation: {
            id: 'elevation',
            format: 'float32' as const,
            storage: {kind: 'buffer' as const, values: context.view('float32', count)}
          },
          parameters: context.view('float32', 12)
        };
        const contributor =
          kind === 'viewshed'
            ? new GPUTerrainViewshed({
                ...common,
                traversal: 'pyramid',
                output: {visibility: context.view('uint32', count)}
              })
            : kind === 'line-of-sight'
              ? new GPUTerrainLineOfSight({
                  ...common,
                  traversal: 'pyramid',
                  pairs: context.view('float32x4', 1),
                  output: {visibility: context.view('uint32', 1)}
                })
              : new GPUTerrainSummits({
                  ...common,
                  output: {summitMask: context.view('uint32', count)}
                });
        Object.defineProperty(
          context.device.limits,
          limit === 'binding' ? 'maxStorageBufferBindingSize' : 'maxBufferSize',
          {value: limit === 'binding' ? 64 : 255, configurable: true}
        );
        const declarations = watchDeclarations(context);
        try {
          expect(() => contributor.getCommandNodes(context.graph)).toThrow(/pyramid.*parameter/);
          declarations.expectUnchanged();
        } finally {
          declarations.restore();
        }
        // Bind only the eighteen parameter words, while allocating a 256-byte aligned slot.
        Object.defineProperty(
          context.device.limits,
          limit === 'binding' ? 'maxStorageBufferBindingSize' : 'maxBufferSize',
          {value: limit === 'binding' ? 72 : 256, configurable: true}
        );
        expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
      } finally {
        context.device.destroy();
      }
    });
  }
}

for (const kind of ['viewshed', 'line-of-sight', 'summits'] as const) {
  it(`does not preflight an unused parameter upload with a shared pyramid (${kind})`, () => {
    const context = createKernelAuditContext(`shared-pyramid-parameters-${kind}`);
    try {
      const grid = {...GRID, width: 5, height: 2};
      const layout = getGPURasterExtremaPyramidLayout(grid, {
        firstBlockSize: kind === 'summits' ? 2 : 4,
        footprint: kind === 'summits' ? 'cell' : 'bilinear'
      });
      const common = {
        grid,
        elevation: {
          id: 'elevation',
          format: 'float32' as const,
          storage: {kind: 'buffer' as const, values: context.view('float32', 10)}
        },
        parameters: context.view('float32', 12),
        pyramid: {layout, combined: context.view('float32', layout.length * 2)}
      };
      const contributor =
        kind === 'viewshed'
          ? new GPUTerrainViewshed({
              ...common,
              traversal: 'pyramid',
              output: {visibility: context.view('uint32', 10)}
            })
          : kind === 'line-of-sight'
            ? new GPUTerrainLineOfSight({
                ...common,
                traversal: 'pyramid',
                pairs: context.view('float32x4', 1),
                output: {visibility: context.view('uint32', 1)}
              })
            : new GPUTerrainSummits({
                ...common,
                incompleteNeighborhood: 'ignore',
                output: {summitMask: context.view('uint32', 10)}
              });
      Object.defineProperty(context.device.limits, 'maxStorageBufferBindingSize', {
        value: 64,
        configurable: true
      });
      Object.defineProperty(context.device.limits, 'maxBufferSize', {
        value: 255,
        configurable: true
      });
      expect(() => contributor.getCommandNodes(context.graph)).not.toThrow();
    } finally {
      context.device.destroy();
    }
  });
}
