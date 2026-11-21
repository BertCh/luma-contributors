// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  getViewBindingRange,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPURasterArithmetic} from '../../../src/raster-analysis/raster-algebra/gpu-raster-arithmetic';
import {GPURasterConditional} from '../../../src/raster-analysis/raster-algebra/gpu-raster-conditional';
import {GPURasterReclassify} from '../../../src/raster-analysis/raster-algebra/gpu-raster-reclassify';
import {GPURasterCellStatistics} from '../../../src/raster-analysis/raster-algebra/gpu-raster-cell-statistics';
import {GPUWeightedOverlay} from '../../../src/raster-analysis/raster-algebra/gpu-weighted-overlay';
import {GPURasterSampling} from '../../../src/raster-analysis/raster-sampling/gpu-raster-sampling';
import {GPURasterFillNoData} from '../../../src/raster-analysis/gpu-raster-fill-nodata';
import {GPURasterRankFilter} from '../../../src/raster-analysis/gpu-raster-rank-filter';
import {
  GPURasterFillHoles,
  GPURasterReconstruction
} from '../../../src/raster-analysis/gpu-raster-reconstruction';
import {GPURasterStretch} from '../../../src/raster-analysis/raster-stretch/gpu-raster-stretch';
import {
  GPURasterSkeleton,
  GPURasterMedialAxis
} from '../../../src/raster-analysis/gpu-raster-skeleton';
import {GPUIsolines} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {GPUIsobandRings} from '../../../src/raster-analysis/isolines/gpu-isoband-rings';
import {GPURasterProfile} from '../../../src/raster-analysis/raster-sampling/gpu-raster-profile';
import {GPURasterJoin} from '../../../src/raster-analysis/polygon-rasterization/gpu-raster-join';
import {GPURasterWalkingCost} from '../../../src/raster-analysis/gpu-raster-walking-cost';
import {GPURasterWatershedSegmentation} from '../../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {GPURasterTileReprojection} from '../../../src/raster-analysis/gpu-raster-tile-reprojection';
import {GPURasterReprojection} from '../../../src/raster-analysis/gpu-raster-reprojection';
import {GPUCostDistancePath} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path';
import {GPURasterFocalPairs} from '../../../src/raster-analysis/focal-pairs/gpu-raster-focal-pairs';
import {GPULineIntegralConvolution} from '../../../src/raster-analysis/flow-texture/gpu-line-integral-convolution';
import {GPUStreamlines} from '../../../src/raster-analysis/flow-texture/gpu-streamlines';
import {GPUParticleAdvection} from '../../../src/raster-analysis/particle-advection/gpu-particle-advection';
import {GPURasterPatchMetrics} from '../../../src/raster-analysis/raster-patches/gpu-raster-patch-metrics';
import {GPURasterSieve} from '../../../src/raster-analysis/raster-patches/gpu-raster-sieve';
import {GPUIsobands} from '../../../src/raster-analysis/isolines/gpu-isobands';
import {GPUPolygonRasterization} from '../../../src/raster-analysis/polygon-rasterization/gpu-polygon-rasterization';
import {GPUDistanceField} from '../../../src/raster-analysis/distance-field/gpu-distance-field';
import {validateRasterStorageBindings} from '../../../src/raster-analysis/raster-grid/raster-storage-preflight';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture(badUsage = false, width = 8, height = 8) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const count = width * height;
  const grid = {
    width,
    height,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const views: GraphDataView[] = [];
  const view = <Format extends 'float32' | 'uint32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length = count
  ) => {
    const rowByteLength = format === 'float32x4' ? 16 : format === 'float32x2' ? 8 : 4;
    const buffer = graph.createTransientBuffer({
      id: `caller-${views.length}`,
      byteLength: Math.max(length, 1) * rowByteLength,
      usage: badUsage ? Buffer.COPY_DST : Buffer.STORAGE
    });
    const result = graph.createDataView(buffer, {format, length});
    views.push(result);
    return result;
  };
  const band = () => ({
    id: `band-${views.length}`,
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: view('float32')}
  });
  const status = () => ({converged: view('uint32', 1)});
  return {device, graph, grid, count, view, band, status, views};
}

type Fixture = ReturnType<typeof createFixture>;
type Contributor = GPUCommandNodeProducer & {readonly id: string};
type Case = {name: string; create: (fixture: Fixture) => Contributor};
const cases: Case[] = [
  {
    name: 'arithmetic',
    create: ({view, band, grid, count}) =>
      new GPURasterArithmetic({
        grid,
        cellCount: count,
        a: band(),
        parameters: view('float32', 8),
        output: {values: view('float32')}
      })
  },
  {
    name: 'conditional',
    create: ({view, band, grid, count}) =>
      new GPURasterConditional({
        grid,
        cellCount: count,
        conditionValues: band(),
        a: band(),
        parameters: view('float32', 8),
        output: {values: view('float32')}
      })
  },
  {
    name: 'reclassify',
    create: ({view, band, grid}) =>
      new GPURasterReclassify({
        grid,
        values: band(),
        breaks: view('float32', 2),
        parameters: view('float32', 4),
        output: {classes: view('uint32')}
      })
  },
  {
    name: 'cell-statistics',
    create: ({view, band, grid, count}) =>
      new GPURasterCellStatistics({
        grid,
        cellCount: count,
        layerCount: 2,
        stack: [band(), band()],
        parameters: view('float32', 4),
        output: {mean: view('float32')}
      })
  },
  {
    name: 'weighted-overlay',
    create: ({view, band, grid, count}) =>
      new GPUWeightedOverlay({
        grid,
        cellCount: count,
        layerCount: 2,
        stack: [band(), band()],
        parameters: view('float32', 24),
        output: {score: view('float32')}
      })
  },
  {
    name: 'sampling',
    create: ({view, band, grid}) =>
      new GPURasterSampling({
        grid,
        values: band(),
        positions: view('float32x2', 2),
        parameters: view('float32', 12),
        output: {values: view('float32', 2)}
      })
  },
  {
    name: 'fill-nodata',
    create: ({view, band, grid}) =>
      new GPURasterFillNoData({
        grid,
        input: band(),
        maximumDistance: 1,
        output: {
          values: view('float32'),
          validity: view('uint32'),
          filled: view('uint32'),
          status: {invalidCount: view('uint32', 1)}
        }
      })
  },
  {
    name: 'rank-filter',
    create: ({view, band, grid}) =>
      new GPURasterRankFilter({
        grid,
        input: band(),
        radius: 1,
        output: {values: view('float32'), validity: view('uint32')}
      })
  },
  {
    name: 'fill-holes',
    create: ({view, status, grid}) =>
      new GPURasterFillHoles({
        grid,
        input: view('uint32'),
        maximumIterations: 4,
        output: {values: view('uint32'), status: status()}
      })
  },
  {
    name: 'skeleton',
    create: ({view, status, grid}) =>
      new GPURasterSkeleton({
        grid,
        input: view('uint32'),
        maximumIterations: 4,
        output: {skeleton: view('uint32'), status: status()}
      })
  },
  {
    name: 'walking-cost',
    create: ({view, band, status, grid}) =>
      new GPURasterWalkingCost({
        grid,
        elevation: band(),
        friction: band(),
        sources: view('uint32'),
        parameters: view('float32', 16),
        maximumIterations: 4,
        output: {costs: view('float32'), validity: view('uint32'), status: status()}
      })
  },
  {
    name: 'watershed-fifo',
    create: ({view, band, status, grid}) =>
      new GPURasterWatershedSegmentation({
        grid,
        input: band(),
        markers: view('uint32'),
        output: {labels: view('uint32'), status: status()}
      })
  },
  {
    name: 'watershed-forest',
    create: ({view, band, status, grid}) =>
      new GPURasterWatershedSegmentation({
        grid,
        input: band(),
        markers: view('uint32'),
        method: 'minimum-spanning-forest',
        output: {labels: view('uint32'), status: status()}
      })
  },
  {
    name: 'tile-reprojection',
    create: ({view, band, grid}) =>
      new GPURasterTileReprojection({
        sourceGrid: grid,
        destinationGrid: grid,
        sources: [{pixelBounds: [0, 0, grid.width, grid.height], input: band()}],
        parameters: view('float32', 20),
        output: {values: view('float32'), validity: view('uint32')}
      })
  },
  {
    name: 'reprojection',
    create: ({view, band, grid}) =>
      new GPURasterReprojection({
        sourceGrid: grid,
        destinationGrid: grid,
        input: band(),
        parameters: view('float32', 20),
        output: {values: view('float32'), validity: view('uint32')}
      })
  },
  {
    name: 'cost-path',
    create: ({view, grid}) =>
      new GPUCostDistancePath({
        grid,
        backLinks: view('uint32'),
        target: view('uint32', 1),
        output: {
          path: {
            ids: view('uint32', 8),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        }
      })
  },
  {
    name: 'focal-pairs',
    create: ({view, band, grid}) =>
      new GPURasterFocalPairs({
        grid,
        predictor: band(),
        response: band(),
        radius: 1,
        output: {pairedCount: view('uint32'), status: view('uint32'), covariance: view('float32')}
      })
  },
  {
    name: 'lic',
    create: ({view, grid}) =>
      new GPULineIntegralConvolution({
        grid,
        velocities: view('float32x2', 16),
        fieldWidth: 4,
        fieldHeight: 4,
        parameters: view('float32', 16),
        uint32Parameters: view('uint32', 4),
        output: {values: view('float32')}
      })
  },
  {
    name: 'streamlines',
    create: ({view}) =>
      new GPUStreamlines({
        velocities: view('float32x2', 16),
        fieldWidth: 4,
        fieldHeight: 4,
        gridWidth: 8,
        gridHeight: 8,
        seedColumns: 3,
        seedRows: 2,
        stepsPerDirection: 5,
        maximumIterations: 4,
        parameters: view('float32', 12),
        uint32Parameters: view('uint32', 4),
        output: {
          lines: {
            ids: view('uint32', 6),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          },
          lineOffsets: view('uint32', 7),
          positions: view('float32x2', 1),
          positionCount: view('uint32', 1)
        }
      })
  },
  {
    name: 'particles',
    create: ({view}) =>
      new GPUParticleAdvection({
        velocities: view('float32x2', 16),
        fieldWidth: 4,
        fieldHeight: 4,
        parameters: view('float32', 12),
        uint32Parameters: view('uint32', 4),
        state: {
          positions: view('float32x2', 2),
          ages: view('uint32', 2),
          generations: view('uint32', 2)
        },
        output: {trails: {positions: view('float32x2', 6), length: 3}}
      })
  },
  {
    name: 'patch-metrics',
    create: ({view, grid}) =>
      new GPURasterPatchMetrics({
        grid,
        labels: view('uint32'),
        output: {pixelCounts: view('uint32', 3), perimeters: view('float32', 3)}
      })
  },
  {
    name: 'sieve',
    create: ({view, grid}) =>
      new GPURasterSieve({
        grid,
        labels: view('uint32'),
        patchCapacity: 3,
        parameters: view('uint32', 4),
        output: {labels: view('uint32')}
      })
  },
  {
    name: 'isobands',
    create: ({view, band, grid}) =>
      new GPUIsobands({
        grid,
        values: band(),
        breaks: view('float32', 2),
        parameters: view('float32', 12),
        output: {bandClasses: view('uint32')}
      })
  },
  {
    name: 'polygon-rasterization',
    create: ({view, grid}) =>
      new GPUPolygonRasterization({
        grid,
        parameters: view('float32', 4),
        positions: view('float32x2', 8),
        featureOffsets: view('uint32', 2),
        polygonOffsets: view('uint32', 2),
        ringOffsets: view('uint32', 2),
        crossingCapacity: 65,
        output: {zones: view('uint32'), status: {overflow: view('uint32', 1)}}
      })
  },
  {
    name: 'distance-exact',
    create: ({view, grid}) =>
      new GPUDistanceField({
        grid,
        parameters: view('float32', 8),
        seedMask: view('uint32'),
        output: {distances: view('float32')}
      })
  },
  {
    name: 'distance-jump-flood',
    create: ({view, grid}) =>
      new GPUDistanceField({
        grid,
        mode: 'jump-flood',
        parameters: view('float32', 8),
        seedMask: view('uint32'),
        output: {distances: view('float32')}
      })
  }
];

cases.push(
  {
    name: 'medial-axis',
    create: ({view, grid}) =>
      new GPURasterMedialAxis({
        grid,
        input: view('uint32'),
        output: {medialAxis: view('uint32'), radius: view('float32')}
      })
  },
  {
    name: 'isolines',
    create: ({view, band, grid}) =>
      new GPUIsolines({
        grid,
        values: band(),
        levels: view('float32', 2),
        parameters: view('float32', 8),
        output: {
          segments: view('float32x4', 8),
          segmentLevels: view('uint32', 8),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        }
      })
  },
  {
    name: 'isoband-rings',
    create: ({view, band, grid}) =>
      new GPUIsobandRings({
        grid,
        values: band(),
        breaks: view('float32', 2),
        parameters: view('float32', 12),
        edgeCapacity: 8,
        output: {
          positions: view('float32x2', 32),
          ringOffsets: view('uint32', 3),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            edgeOverflow: view('uint32', 1)
          }
        }
      })
  },
  {
    name: 'profile',
    create: ({view, band, grid}) =>
      new GPURasterProfile({
        grid,
        values: band(),
        positions: view('float32x2', 2),
        lineOffsets: view('uint32', 2),
        sampleCapacity: 1000,
        parameters: view('float32', 16),
        output: {
          lineMinimums: view('float32', 1),
          status: {count: view('uint32', 1), overflow: view('uint32', 1)}
        }
      })
  },
  {
    name: 'raster-join',
    create: ({view, grid}) =>
      new GPURasterJoin({
        grid,
        parameters: view('float32', 4),
        zones: view('uint32'),
        zoneCount: 3,
        points: view('float32x2', 2),
        values: view('float32', 2),
        output: {sums: view('float32', 3)}
      })
  }
);

it('status-only profile accepts an unused non-storage raster with a 256-byte limit and seven count-route nodes', () => {
  const fixture = createFixture(false, 8, 1);
  try {
    const values = fixture.band();
    Object.defineProperty(values.storage.values.buffer, 'usage', {value: Buffer.COPY_DST});
    const contributor = new GPURasterProfile({
      grid: fixture.grid,
      values,
      positions: fixture.view('float32x2', 2),
      lineOffsets: fixture.view('uint32', 2),
      sampleCapacity: 1000,
      parameters: fixture.view('float32', 16),
      output: {status: {count: fixture.view('uint32', 1), overflow: fixture.view('uint32', 1)}}
    });
    Object.defineProperty(fixture.device, 'limits', {
      value: {...fixture.device.limits, maxStorageBufferBindingSize: 256}
    });
    const nodes = contributor.getCommandNodes(fixture.graph);
    expect(nodes).toHaveLength(7);
    expect(nodes.map(node => node.id)).not.toContain('raster-profile-locate');
    expect(nodes.map(node => node.id)).not.toContain('raster-profile-sample');
    for (const node of nodes) fixture.graph.add(node);
  } finally {
    fixture.device.destroy();
  }
});

function expectNoDeclarations(graph: GPUCommandGraph, run: () => unknown, pattern: RegExp): void {
  const declarations = [
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'addComputePass'),
    vi.spyOn(graph, 'importBuffer'),
    vi.spyOn(graph.device, 'createBuffer')
  ];
  expect(run).toThrow(pattern);
  for (const declaration of declarations) expect(declaration).not.toHaveBeenCalled();
}

for (const entry of cases) {
  it(`${entry.name} rejects missing STORAGE before any graph or device declaration`, () => {
    const fixture = createFixture(true);
    try {
      const contributor = entry.create(fixture);
      expectNoDeclarations(
        fixture.graph,
        () => contributor.getCommandNodes(fixture.graph),
        /STORAGE/
      );
    } finally {
      fixture.device.destroy();
    }
  });
  it(`${entry.name} checks every caller port individually before declaring anything`, () => {
    const fixture = createFixture();
    try {
      const contributor = entry.create(fixture);
      for (const [index, caller] of fixture.views.entries()) {
        const usage = caller.buffer.usage;
        Object.defineProperty(caller.buffer, 'usage', {value: Buffer.COPY_DST, configurable: true});
        try {
          expectNoDeclarations(
            fixture.graph,
            () => contributor.getCommandNodes(fixture.graph),
            /STORAGE/
          );
        } catch (error) {
          throw new Error(
            `${entry.name} caller ${index} (${caller.buffer.id}) was not preflighted`,
            {cause: error}
          );
        } finally {
          Object.defineProperty(caller.buffer, 'usage', {value: usage, configurable: true});
        }
      }
    } finally {
      fixture.device.destroy();
    }
  });
  it(`${entry.name} still builds its valid route`, () => {
    const fixture = createFixture();
    try {
      const nodes = entry.create(fixture).getCommandNodes(fixture.graph);
      expect(nodes.length).toBeGreaterThan(0);
      for (const node of nodes) fixture.graph.add(node);
    } finally {
      fixture.device.destroy();
    }
  });
}

for (const name of [
  'fill-holes',
  'walking-cost',
  'focal-pairs',
  'streamlines',
  'distance-exact',
  'distance-jump-flood',
  'cell-statistics',
  'weighted-overlay',
  'profile'
]) {
  it(`${name} preflights derived scratch with valid caller bindings`, () => {
    const fixture = createFixture(false, 64, 1);
    try {
      const contributor = cases.find(entry => entry.name === name)!.create(fixture);
      const largestCallerRange = Math.max(
        ...fixture.views.map(view => getViewBindingRange(view).size)
      );
      Object.defineProperty(fixture.device, 'limits', {
        value: {...fixture.device.limits, maxStorageBufferBindingSize: largestCallerRange}
      });
      expectNoDeclarations(
        fixture.graph,
        () => contributor.getCommandNodes(fixture.graph),
        /scratch needs .*one binding \(limit/
      );
    } finally {
      fixture.device.destroy();
    }
  });
}

it('validates actual aligned ranges and backing allocation without restricting valid packed layouts', () => {
  const {graph, device} = createFixture();
  try {
    const packed = graph.createTransientBuffer({
      id: 'packed',
      byteLength: 512,
      usage: Buffer.STORAGE
    });
    const first = graph.createDataView(packed, {format: 'uint32', byteOffset: 0, length: 1});
    const second = graph.createDataView(packed, {format: 'float32', byteOffset: 4, length: 1});
    expect(() => validateRasterStorageBindings(graph, 'packed', [first, second])).not.toThrow();
    const empty = graph.createDataView(packed, {format: 'uint32', byteOffset: 512, length: 0});
    expectNoDeclarations(
      graph,
      () => validateRasterStorageBindings(graph, 'empty', [empty]),
      /binding must fit/
    );
  } finally {
    device.destroy();
  }
});

it('checks a backing allocation separately from a small aligned binding', () => {
  const {graph, view, device} = createFixture();
  try {
    const caller = view('uint32', 256);
    const prefix = graph.createDataView(caller.buffer, {format: 'uint32', length: 1});
    Object.defineProperty(device, 'limits', {value: {...device.limits, maxBufferSize: 512}});
    expectNoDeclarations(
      graph,
      () => validateRasterStorageBindings(graph, 'backing', [prefix]),
      /maxBufferSize/
    );
  } finally {
    device.destroy();
  }
});

it('uses the aligned prefix in preflight rather than the logical row byte length', () => {
  const {graph, device} = createFixture();
  try {
    const buffer = graph.createTransientBuffer({
      id: 'alignment-prefix',
      byteLength: 256,
      usage: Buffer.STORAGE
    });
    const caller = graph.createDataView(buffer, {format: 'uint32', byteOffset: 4, length: 1});
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 4}
    });
    expectNoDeclarations(
      graph,
      () => validateRasterStorageBindings(graph, 'aligned', [caller]),
      /needs 8 bytes.*limit 4/
    );
  } finally {
    device.destroy();
  }
});

it('binds only the grid prefix of padded native bands', () => {
  const fixture = createFixture(false, 1, 1);
  try {
    const {view, grid, graph, device} = fixture;
    const input = {
      id: 'padded',
      format: 'uint32' as const,
      storage: {kind: 'buffer' as const, values: view('uint32', 256)},
      validity: view('uint32', 256)
    };
    const contributor = new GPURasterArithmetic({
      grid,
      cellCount: 1,
      a: input,
      parameters: view('float32', 8),
      output: {values: view('float32', 1)}
    });
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 32}
    });
    expect(() => graph.add(contributor)).not.toThrow();
  } finally {
    fixture.device.destroy();
  }
});

it('ignores optional geometry parameters when only patch counts are requested', () => {
  const {graph, grid, view, device} = createFixture();
  try {
    const buffer = graph.createTransientBuffer({
      id: 'unused-georeference',
      byteLength: 32,
      usage: Buffer.COPY_DST
    });
    const parameters = graph.createDataView(buffer, {format: 'float32', length: 8});
    const contributor = new GPURasterPatchMetrics({
      grid,
      labels: view('uint32'),
      parameters,
      output: {pixelCounts: view('uint32', 3)}
    });
    expect(() => graph.add(contributor)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('checks a FIFO queue only when that route allocates storage scratch', () => {
  for (const workgroupQueue of [true, false]) {
    const fixture = createFixture();
    try {
      const contributor = cases.find(entry => entry.name === 'watershed-fifo')!.create(fixture);
      Object.defineProperty(fixture.device, 'limits', {
        value: {
          ...fixture.device.limits,
          maxStorageBufferBindingSize: 256,
          ...(workgroupQueue ? {} : {maxComputeWorkgroupStorageSize: 0})
        }
      });
      if (workgroupQueue) expect(() => fixture.graph.add(contributor)).not.toThrow();
      else
        expectNoDeclarations(
          fixture.graph,
          () => contributor.getCommandNodes(fixture.graph),
          /fifo queue scratch needs 768 bytes/
        );
    } finally {
      fixture.device.destroy();
    }
  }
});

it('preflights nested ring shell BVH padding before declaring boundary edges', () => {
  const {graph, grid, view, band, device} = createFixture(false, 2, 2);
  try {
    const contributor = new GPUIsobandRings({
      grid,
      values: band(),
      breaks: view('float32', 1),
      parameters: view('float32', 12),
      edgeCapacity: 1,
      output: {
        positions: view('float32x2', 8),
        ringOffsets: view('uint32', 6),
        ringShells: view('uint32', 5),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          edgeOverflow: view('uint32', 1)
        }
      }
    });
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBufferBindingSize: 100}
    });
    expectNoDeclarations(
      graph,
      () => contributor.getCommandNodes(graph),
      /shell BVH nodes scratch needs 120 bytes/
    );
  } finally {
    device.destroy();
  }
});

it('permits outputs in unused padded band tails after a separate conversion dispatch', () => {
  for (const route of ['reconstruction', 'stretch']) {
    const {graph, grid, view, band, status, device} = createFixture(false, 1, 1);
    try {
      const native = view('uint32', 1024);
      const input = {
        id: 'padded-input',
        format: 'uint32' as const,
        storage: {kind: 'buffer' as const, values: native},
        validity: view('uint32', 1024)
      };
      const output = graph.createDataView(native.buffer, {
        format: 'float32',
        byteOffset: 4,
        length: 1
      });
      const contributor =
        route === 'reconstruction'
          ? new GPURasterReconstruction({
              grid,
              marker: input,
              mask: band(),
              maximumIterations: 1,
              output: {values: output, validity: view('uint32'), status: status()}
            })
          : new GPURasterStretch({
              grid,
              values: input,
              parameters: view('float32', 16),
              binCount: 16,
              output: {stretched: output}
            });
      Object.defineProperty(device, 'limits', {
        value: {
          ...device.limits,
          maxStorageBufferBindingSize: route === 'reconstruction' ? 768 : 64
        }
      });
      expect(() => graph.add(contributor)).not.toThrow();
    } finally {
      device.destroy();
    }
  }
});
