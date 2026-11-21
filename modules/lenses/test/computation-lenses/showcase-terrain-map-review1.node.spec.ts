// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {
  createSourceFile,
  forEachChild,
  isPropertyAssignment,
  transpileModule,
  ScriptTarget,
  type Node
} from 'typescript';
import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  GPUCompaction,
  GPUElementwise
} from '@luma.gl/graph/gpu-core';
import {GPUPointCloudVoxelDownsample} from '@luma.gl/graph/gpu-pointcloud';
import * as pointcloud from '../../../analysis/src/pointcloud-analysis/operations/index';
import {GPUPolygonRasterization} from '../../../analysis/src/raster-analysis/polygon-rasterization/gpu-polygon-rasterization';
import {GPURasterZonalStatistics} from '../../../analysis/src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../analysis/src/raster-analysis/raster-grid/gpu-raster-grid';
import {GPUZonalStatistics} from '../../../analysis/src/gpu-spatial-analysis/zonal-statistics/gpu-zonal-statistics';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {makeSpatialFixture} from '../../../analysis/test/spatial-test-utils';
import * as groundSupport from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-ground-support';
import {importGraphBuffer} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/graph-buffers';

vi.mock(
  '@luma-contributors/analysis/pointcloud-analysis',
  async () => import('../../../analysis/src/pointcloud-analysis/operations/index')
);
const scenes = 'examples/deck/spatial-analysis-showcase/src/scenes/';
function evaluate(source: string, bindings: Record<string, unknown>, expression: string) {
  const javascript = transpileModule(source, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  return new Function(...Object.keys(bindings), `${javascript}; return ${expression};`)(
    ...Object.values(bindings)
  );
}
function displayed(path: string, state: Record<string, unknown> = {}) {
  const source = createSourceFile(path, readFileSync(path, 'utf8'), ScriptTarget.Latest, true);
  let initializer = '';
  function visit(node: Node) {
    if (isPropertyAssignment(node) && node.name.getText(source) === 'snippet')
      initializer = node.initializer.getText(source);
    else forEachChild(node, visit);
  }
  visit(source);
  expect(initializer).not.toBe('');
  return evaluate(`const render = ${initializer};`, {}, 'render')(state) as string;
}
function statement(source: string, marker: string) {
  const start = source.indexOf(marker);
  expect(start).toBeGreaterThanOrEqual(0);
  return source.slice(start, source.indexOf('}));', start) + 4);
}

it('B1 all certified-bounds routes declare on one null graph and use the descending eigenvalue oracle', () => {
  const source = readFileSync(
    'modules/lenses/test/computation-lenses/showcase-terrain-map-evidence.spec.ts',
    'utf8'
  );
  const start = source.indexOf('    const positions =');
  const end = source.indexOf('    await fixture.run();', start);
  const device = createNullWebGPUDevice();
  const fixture = makeSpatialFixture(device);
  try {
    expect(() =>
      evaluate(source.slice(start, end), {fixture, ...pointcloud, ...groundSupport}, 'results')
    ).not.toThrow();
    // diag(.5,.5,0) sorted in the producer's documented descending order.
    expect(source).toContain('toEqual([0.5, 0.5, 0])');
  } finally {
    fixture.destroy();
    device.destroy();
  }
});

it('B2 the whole displayed LiDAR chain declares on one graph with distinct seed/final producers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const floats = (name: string, length = 4) => createTransientView(graph, name, 'float32', length);
  const words = (name: string, length = 4) => createTransientView(graph, name, 'uint32', length);
  const vectors = (name: string, length = 5) =>
    createTransientView(graph, name, 'float32x3', length);
  const positions = vectors('positions');
  const bindings: Record<string, unknown> = {
    graph,
    ...pointcloud,
    ...groundSupport,
    makeGPURasterBand,
    GPUPointCloudVoxelDownsample,
    GPUCompaction,
    GPUElementwise,
    positions,
    grid: {origin: [-1, -1], cellSize: 1, width: 2, height: 2},
    neighborhoodGrid: {origin: [-5, -5, -5], cellSize: 2.5, dimensions: [4, 4, 4]}
  };
  for (const name of [
    'classifications',
    'counts',
    'neighborhoodStatus',
    'inliers',
    'groundMask',
    'outputClassifications',
    'groundStatus',
    'supportedClassifications',
    'groundDisplay',
    'heightStatus',
    'featureStatus'
  ])
    bindings[name] = words(name, 5);
  bindings['indices'] = words('indices', 80);
  bindings['squaredDistances'] = floats('distances', 80);
  bindings['meanDistances'] = floats('mean-distances', 5);
  bindings['heights'] = floats('heights', 5);
  bindings['thresholdComplete'] = words('threshold-complete', 1);
  for (const name of [
    'minimumSurface',
    'seedSurface',
    'seedCanopy',
    'terrain',
    'surface',
    'canopy'
  ])
    bindings[name] = floats(name);
  for (const name of [
    'seedCoverage',
    'seedSurfaceValidity',
    'groundSeedCoverage',
    'seedCanopyValidity',
    'coverage',
    'surfaceValidity',
    'terrainValidity',
    'canopyValidity'
  ])
    bindings[name] = words(name);
  for (const name of ['normals', 'eigenvalues', 'shapes']) bindings[name] = vectors(name);
  const voxelCount = 110 * 110 * 45;
  const representatives = words('representatives', voxelCount),
    voxelCounts = words('voxel-counts', voxelCount);
  Object.assign(bindings, {
    voxelBuffers: {
      positions,
      origin: [-110, -110, -5],
      representatives,
      counts: voxelCounts,
      output: vectors('voxel-positions', voxelCount)
    },
    representatives,
    voxelCounts,
    ones: words('ones', voxelCount),
    occupiedFlags: words('occupied', voxelCount),
    compactIndices: words('compact', voxelCount),
    drawCount: words('draw-count', 1)
  });
  const source = displayed(`${scenes}terrain/lidar-ground.scene.ts`, {
    view: 'features',
    pointSize: 1
  }).replace(/^import[\s\S]*?;\n/gm, '');
  try {
    expect(() => evaluate(source, bindings, 'undefined')).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('B3 retained-neighbour threshold labels are distinct from hidden suppressed classifications', () => {
  const source = readFileSync(`${scenes}terrain/lidar-ground.compute.ts`, 'utf8');
  expect(source).not.toContain('No reported support loss');
  const first = source.indexOf('    getLayers(): Layer[] {');
  const last = source.indexOf('    destroy:', first);
  const state = evaluate(
    `let filterThresholdComplete = false; const instance = {${source.slice(first, last)}};`,
    {
      ctx: {options: {view: 'outliers', pointSize: 1}},
      origin: [0, 0, 0],
      count: 2,
      position2DBuffer: {},
      inlierBuffer: {},
      SpatialAnalysisPointLayer: class {
        constructor(props: object) {
          return props;
        }
      }
    },
    '({layers:()=>instance.getLayers(),set:(value)=>{filterThresholdComplete=value;}})'
  );
  expect(state.layers()).toEqual([]);
  state.set(true);
  expect(state.layers()).toHaveLength(1);
  const scene = readFileSync(`${scenes}terrain/lidar-ground.scene.ts`, 'utf8');
  expect(scene).toContain('Sparse outlier (retained neighbours)');
  expect(scene).toContain('Typical retained support');
  expect(groundSupport.getLidarFilterSupportText(true)).toMatch(/retained nearest neighbours/);
  expect(groundSupport.getLidarFilterSupportText(false)).toMatch(/suppressed \(hidden\)/);
});

it('S1 LiDAR summary copies five reduced words without a CPU row loop', () => {
  const source = readFileSync(`${scenes}terrain/lidar-ground.compute.ts`, 'utf8');
  const summary = source.slice(
    source.indexOf('  const summaryReader ='),
    source.indexOf("  ctx.setReadout('samplePoints'")
  );
  expect(summary).toContain('{buffer: featureSummaryBuffer, size: 20}');
  expect(summary).not.toContain('featureStatusBuffer');
  expect(summary).not.toMatch(/for\s*\(/);
  expect(summary).not.toContain('decodeGPUPointCloudFeaturesStatus');
  expect(source).toContain('addLidarFeatureSummary(');
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(() =>
      groundSupport.addLidarFeatureSummary(
        graph,
        createTransientView(graph, 'status', 'uint32', 8),
        createTransientView(graph, 'summary', 'uint32', 5)
      )
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('B4 wildfire polygon grid uses the live south-increasing raster frame at 239 metres', () => {
  const source = displayed(`${scenes}raster/wildfire-terrain.scene.ts`, {
    distanceMode: 'exact',
    sumOrder: 'atomic',
    borderMode: 'clamp'
  });
  const first = source.indexOf('const rasterGrid =');
  const last = source.indexOf('\nextentParameters.write', first);
  const grid = evaluate(
    source.slice(first, last),
    {width: 2, height: 2, cellSize: 239, extent: [0, 0, 239, 239], makeGPURasterMetadata},
    'rasterGrid'
  );
  expect(grid.affine).toEqual([239, 0, 0, 0, 239, 0]);
});

it('S3 displayed wildfire rasterizer and zonal reducer share the actual loss view on a null graph', () => {
  expect(
    readFileSync(
      'modules/lenses/test/computation-lenses/showcase-raster-review.node.spec.ts',
      'utf8'
    )
  ).not.toContain("'fast'");
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const source = displayed(`${scenes}raster/wildfire-terrain.scene.ts`, {
    distanceMode: 'exact',
    sumOrder: 'atomic',
    borderMode: 'clamp'
  });
  const grid = makeGPURasterMetadata({
    width: 2,
    height: 2,
    cellSize: 239,
    origin: [0, 0],
    rowDirection: 'north'
  });
  const words = (name: string, length = 4) => createTransientView(graph, name, 'uint32', length);
  const floats = (name: string, length = 4) => createTransientView(graph, name, 'float32', length);
  const overflow = words('overflow', 1),
    incomplete = words('incomplete', 1),
    zones = words('zones'),
    zoneFire = words('zone-fire');
  const producers: GPUPolygonRasterization[] = [],
    consumers: GPURasterZonalStatistics[] = [];
  const bindings = {
    graph,
    GPUPolygonRasterization: class extends GPUPolygonRasterization {
      constructor(props: ConstructorParameters<typeof GPUPolygonRasterization>[0]) {
        super(props);
        producers.push(this);
      }
    },
    GPURasterZonalStatistics: class extends GPURasterZonalStatistics {
      constructor(props: ConstructorParameters<typeof GPURasterZonalStatistics>[0]) {
        super(props);
        consumers.push(this);
      }
    },
    rasterGrid: grid,
    mercatorGrid: grid,
    extentParameters: {importToGraph: () => floats('extent')},
    polygonPositions: createTransientView(graph, 'polygon-positions', 'float32x2', 4),
    featureOffsets: words('feature-offsets', 2),
    polygonOffsets: words('polygon-offsets', 2),
    ringOffsets: words('ring-offsets', 2),
    crossingCapacity: 8,
    zones,
    boundary: words('boundary'),
    overflow,
    crossingCount: words('crossing-count', 1),
    zoneFire,
    fireCount: 1,
    slopeBand: makeGPURasterBand(floats('slope'), grid),
    incomplete,
    cellCounts: words('cell-counts', 2),
    valueCounts: words('value-counts', 2),
    sums: floats('sums', 2),
    maximums: floats('maximums', 2)
  };
  try {
    expect(source).not.toContain("sumOrder: 'fast'");
    expect(() =>
      evaluate(
        statement(source, 'graph.add(new GPUPolygonRasterization({') +
          '\n' +
          statement(source, 'graph.add(new GPURasterZonalStatistics({'),
        bindings,
        'undefined'
      )
    ).not.toThrow();
    expect(consumers[0].props.producerOverflow).toBe(producers[0].props.output.status?.overflow);
    expect(consumers[0].props.producerOverflow).toBe(overflow);
  } finally {
    device.destroy();
  }
});

it('B5 wildfire summary captures status and columns in one submission before overlapping readbacks resolve', async () => {
  const source = displayed(scenes + 'raster/wildfire-terrain.scene.ts', {sumOrder: 'atomic'});
  const first = source.indexOf('const slopeSummaryReader =');
  expect(first).toBeGreaterThan(0);
  const published: number[][] = [],
    messages: string[] = [],
    pending: Array<() => void> = [];
  let current = {loss: 0, count: 1, sum: 0};
  class ReadbackRing {
    tryAcquire() {
      const buffer = new Uint8Array(20);
      return {
        buffer,
        markEncoded() {},
        read: () => new Promise<Uint8Array>(resolve => pending.push(() => resolve(buffer)))
      };
    }
  }
  // Execute the real SummaryReader implementation with only its device ring replaced.
  // The fake encoder queues buffer copies, then executes them together at submit().
  const shared = readFileSync('examples/deck/spatial-analysis-shared/summary-reader.ts', 'utf8');
  const SummaryReader = evaluate(
    shared.slice(shared.indexOf('export class SummaryReader')).replace('export ', ''),
    {GPUReadbackRing: ReadbackRing},
    'SummaryReader'
  );
  const makeEncoder = () => {
    const copies: Array<() => void> = [];
    return {
      copyBufferToBuffer({
        sourceBuffer,
        destinationBuffer,
        destinationOffset
      }: {
        sourceBuffer: string;
        destinationBuffer: Uint8Array;
        destinationOffset: number;
      }) {
        copies.push(() => {
          const values =
            sourceBuffer === 'loss'
              ? Uint32Array.of(current.loss)
              : sourceBuffer === 'counts'
                ? Uint32Array.of(current.count, 0)
                : Float32Array.of(current.sum, 0);
          destinationBuffer.set(new Uint8Array(values.buffer), destinationOffset);
        });
      },
      submit() {
        for (const copy of copies) copy();
      }
    };
  };
  const tick = () => new Promise<void>(resolve => setTimeout(resolve, 0));
  const run = evaluate(
    source.slice(first),
    {
      SummaryReader,
      resources: {device: {}, track: (resource: unknown) => resource},
      incompleteBuffer: 'loss',
      valueCountsBuffer: 'counts',
      sumsBuffer: 'sums',
      fireCount: 1,
      statsCompiled: {encode() {}},
      clearTerrainSummary: (message: string) => messages.push(message),
      publishTerrainSummary: (counts: Uint32Array, sums: Float32Array) =>
        published.push([counts[0], sums[0]])
    },
    '({encode:encodeSlopeSummary,reader:slopeSummaryReader})'
  );
  const submissionA = makeEncoder();
  run.encode(submissionA);
  submissionA.submit();
  await tick();
  current = {loss: 1, count: 0xffffffff, sum: 99};
  const submissionB = makeEncoder();
  run.encode(submissionB);
  submissionB.submit();
  // A's delayed status is clean; B has already overwritten all producer columns.
  pending[0]();
  await tick();
  expect(published).toEqual([[1, 0]]);
  const subsequentFrame = makeEncoder();
  run.reader.flush(subsequentFrame);
  subsequentFrame.submit();
  await tick();
  pending[1]();
  await tick();
  expect(published).toEqual([[1, 0]]); // B's poison never publishes under A's clean status.
  expect(messages).toContain('unavailable: rasterizer loss');
});

it('B6 static and live storm membership views share only their backing buffer and invoke static readback after submit', () => {
  const device = createNullWebGPUDevice();
  const staticGraph = new GPUCommandGraph(device),
    graph = new GPUCommandGraph(device);
  const source = displayed(`${scenes}earth/storm-outage-exposure.scene.ts`, {
    sumOrder: 'atomic',
    exposureWindow: 'so-far'
  });
  const owned: Buffer[] = [];
  const resources = {
    createBuffer: (id: string, byteLength: number) => {
      const buffer = device.createBuffer({
        id,
        byteLength,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      owned.push(buffer);
      return buffer;
    }
  };
  const words = (target: GPUCommandGraph, name: string, length = 1) =>
    createTransientView(target, name, 'uint32', length);
  const floats = (target: GPUCommandGraph, name: string, length = 2) =>
    createTransientView(target, name, 'float32', length);
  const first = source.indexOf('// Once: join');
  const end = source.indexOf('// Once: the peak', first);
  const bindings: Record<string, unknown> = {
    resources,
    importGraphBuffer,
    staticGraph,
    graph,
    GPUZonalStatistics,
    flashCount: 2,
    countyCount: 1,
    STORM_FRAME: {spatialContext: {coordinateSpace: 'planar', metric: 'native'}},
    countyPositions: createTransientView(staticGraph, 'positions', 'float32x2', 4),
    featureOffsets: words(staticGraph, 'features', 2),
    polygonOffsets: words(staticGraph, 'polygons', 2),
    ringOffsets: words(staticGraph, 'rings', 2),
    candidateCapacity: 8,
    flashMeters: createTransientView(staticGraph, 'flashes', 'float32x2', 2),
    flashTotals: words(staticGraph, 'totals'),
    pointFeatureRows: words(staticGraph, 'rows', 2),
    ones: floats(graph, 'ones'),
    flashWeights: floats(graph, 'weights'),
    flashSums: floats(graph, 'sums', 1)
  };
  try {
    const prefix = source.slice(first, end);
    const live = statement(source, 'graph.add(new GPUZonalStatistics({');
    const imports = new Map<unknown, Buffer>();
    bindings['importGraphBuffer'] = (...parameters: Parameters<typeof importGraphBuffer>) => {
      const view = importGraphBuffer(...parameters);
      imports.set(view, parameters[2]);
      return view;
    };
    let returned: {staticRows: unknown; liveRows: unknown; foreignError: string} | undefined;
    const foreign = live.replace(
      'pointFeatureRows: livePointFeatureRows',
      'pointFeatureRows: staticPointFeatureRows'
    );
    const declarations = `${prefix}\nlet foreignError='';try {${foreign}} catch(error) {foreignError=String(error);}\n${live}`;
    expect(() => {
      returned = evaluate(
        declarations,
        bindings,
        '({staticRows:staticPointFeatureRows,liveRows:livePointFeatureRows,foreignError})'
      );
    }).not.toThrow();
    expect(returned).toBeDefined();
    expect(returned!.foreignError).toMatch(/views must belong to the target graph/);
    expect(returned!.staticRows).not.toBe(returned!.liveRows);
    expect(imports.get(returned!.staticRows)).toBeDefined();
    expect(imports.get(returned!.staticRows)).toBe(imports.get(returned!.liveRows));
    expect(source).toMatch(/device.submit\([\s\S]*?void readStaticMembership\(\)/);
  } finally {
    for (const buffer of owned) buffer.destroy();
    device.destroy();
  }
});

it('earlier-fix note storm producers use independent live capacity words', () => {
  const source = displayed(`${scenes}earth/storm-outage-exposure.scene.ts`, {sumOrder: 'atomic'});
  for (const name of ['windowOverflow', 'lineOverflow', 'outageOverflow'])
    expect(source).toContain(`overflow: ${name}`);
});

it('S2 quake device evidence includes the original-frame translated boundary counterexample', () => {
  const source = readFileSync(
    'modules/lenses/test/computation-lenses/showcase-terrain-map-evidence.spec.ts',
    'utf8'
  );
  expect(source).toContain('99.875');
  expect(source).toContain('100.4749984741211');
  const step = Math.fround(0.3),
    position = Math.fround(100.4749984741211),
    origin = Math.fround(99.875);
  const relative = Math.fround(position - origin);
  expect(relative).toBeLessThan(Math.fround(2 * step));
  expect(position).toBe(Math.fround(origin + Math.fround(2 * step)));
});
