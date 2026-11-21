// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, it, vi} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../../analysis/src/testing/gpu-contributor-test-utils';
import {
  getRasterMetadataFromPointCloudGrid,
  GPUPointCloudCoverageValidity
} from '@luma-contributors/analysis/pointcloud-analysis';
import {
  createGroundSeedCoverageNode,
  getLidarFeatureDisplayClass
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-ground-support';
// Exercise this wave's source without rebuilding shared developer outputs.
vi.mock(
  '@luma-contributors/analysis/pointcloud-analysis',
  async () => import('../../../analysis/src/pointcloud-analysis/index')
);
const showcase = 'examples/deck/spatial-analysis-showcase/src/scenes/';
const explorer = 'examples/deck/spatial-analysis-explorer/modes/raster-join-mode.ts';
function code(path: string, start: string, end: string, after = 0) {
  const source = readFileSync(path, 'utf8'),
    first = source.indexOf(start, after);
  expect(first).toBeGreaterThanOrEqual(0);
  const last = source.indexOf(end, first);
  expect(last).toBeGreaterThan(first);
  return source.slice(first, last);
}
function evaluate(source: string, bindings: Record<string, unknown>, expression: string) {
  const javascript = transpileModule(source, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  return new Function(...Object.keys(bindings), `${javascript}; return ${expression};`)(
    ...Object.values(bindings)
  );
}
// Production constructor statements run with inert contributors; every import creates a distinct
// graph handle, just as the installed resolver does, even if physical buffers are identical.
it.each([
  'explorer',
  'showcase',
  'counts-only'
])('B1 %s reuses the exact producer view identity', route => {
  const path = route === 'explorer' ? explorer : `${showcase}joins/raster-join.compute.ts`;
  const source = readFileSync(path, 'utf8');
  const after =
    route === 'counts-only' ? source.indexOf('  async function buildRasterOnlyGraph(') : 0;
  const graphName = route === 'explorer' ? 'rasterGraph' : 'graph';
  const start = source.indexOf(`    ${graphName}.add(\n      new GPUPolygonRasterization`, after);
  const end = source.indexOf(`    ${graphName}.add(\n      new GPURasterJoin`, start);
  const last = source.indexOf('\n    );', end) + 7;
  const contributors: Array<{producerOverflow?: unknown; output: {status: {overflow: unknown}}}> =
    [];
  const graph = {add: (value: (typeof contributors)[number]) => contributors.push(value)};
  const bindings: Record<string, unknown> = {
    graph,
    rasterGraph: graph,
    GPUPolygonRasterization: class {
      constructor(props: object) {
        return props;
      }
    },
    GPURasterJoin: class {
      constructor(props: object) {
        return props;
      }
    },
    importGraphBuffer: () => ({buffer: {}}),
    estimateCrossings: () => 1,
    Math
  };
  const declaration = source
    .slice(after, start)
    .match(/const rasterOverflowView = importGraphBuffer\([\s\S]*?\);/);
  const scope = new Proxy(bindings, {
    has: () => true,
    get: (object, key: string | symbol) =>
      typeof key === 'symbol'
        ? undefined
        : key in object
          ? object[key]
          : key in globalThis
            ? (globalThis as Record<string, unknown>)[key]
            : {positions: {}, featureOffsets: {}, polygonOffsets: {}, ringOffsets: {}}
  });
  const javascript = transpileModule(`${declaration?.[0] ?? ''}\n${source.slice(start, last)}`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  new Function('scope', `with(scope) {${javascript}}`)(scope);
  expect(contributors).toHaveLength(2);
  expect(contributors[1].producerOverflow).toBe(contributors[0].output.status.overflow);
});

it('B3 scenario refuses failed records but accepts genuinely outside assignments', async () => {
  const path = 'modules/lenses/src/computation-lenses/scenarios/gpu-raster/raster-join.lens.ts';
  const source = code(path, 'export async function run(', '\ndefineLensScenario(')
    .replace('export ', '')
    .replace(/import\('([^']+)'\)/g, "loadModule('$1')");
  for (const incomplete of [1, 0]) {
    const result = {
      captures: {
        snapshots: [
          {name: 'joinIncomplete', values: Uint32Array.of(incomplete)},
          {name: 'joinPointZones', values: Uint32Array.of(0xffffffff, 0xffffffff)}
        ]
      }
    };
    const dependencies = {
      runExample() {},
      runRasterLens: async () => result,
      getRasterLensGeometry: () => ({}),
      loadDixiePerimeter: async () => ({
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0]},
        rings: []
      }),
      loadDixieBands: async () => ({
        noData: -999,
        bands: {nirBefore: [1, 1], swirBefore: [0, 0], nirAfter: [1, 1], swirAfter: [0, 0]}
      }),
      getRasterOperatorEntries: () => ({GPURasterJoin: {formula: {}}})
    };
    const run = evaluate(
      source,
      {
        SCENARIO_ID: 'gpu-raster.raster-join',
        BURN_DNBR_THRESHOLD: 0.27,
        T_DIXIE_PERIMETER: {id: 'fixture'},
        loadModule: async () => dependencies
      },
      'run'
    );
    if (incomplete) await expect(run({})).rejects.toThrow(/incomplete/);
    else expect(await run({})).toBe(result);
  }
});

it('B6 storm diagnostics distinguish pending, line loss and flash membership loss', () => {
  const values = new Map();
  const source = code(
    `${showcase}earth/storm-outage-exposure.compute.ts`,
    '  function updateOverflowReadout()',
    '  function updateValueRange()'
  );
  const state = evaluate(
    `let staticOverflowed=false, exposureOverflowed=false, staticUncertainCount=null, staticFlashOverflow=null; ${source}`,
    {ctx: {setReadout: (key: string, value: unknown) => values.set(key, value)}},
    '({run:updateOverflowReadout,set:(uncertain,flash,line)=>{staticUncertainCount=uncertain;staticFlashOverflow=flash;exposureOverflowed=line;}})'
  );
  state.run();
  expect(values.get('overflow')).toMatch(/pending/);
  state.set(0, false, true);
  state.run();
  expect(values.get('overflow')).not.toMatch(/flash totals are not exact/);
  state.set(1, false, false);
  state.run();
  expect(values.get('overflow')).toMatch(/1 uncertain.*not exact/);
  state.set(0, true, false);
  state.run();
  expect(values.get('overflow')).toMatch(/flash.*not exact/);
});

it('B5 explorer preserves raster totals but suppresses comparisons on reference loss', () => {
  const values = new Map();
  const names = [
    'crossing',
    'inside',
    'boundary',
    'exact',
    'boundHolds',
    'worstShare',
    'worstDifference'
  ];
  const bindings: Record<string, unknown> = {
    zoneCount: 1,
    pointCount: 2,
    RASTER_WIDTH: 1,
    RASTER_HEIGHT: 1,
    CROSSING_CAPACITY: 10,
    GPU_POLYGON_RASTERIZATION_NO_ZONE: 0xffffffff,
    parking: {positions: [0.25, 0.25, 0.75, 0.75]},
    formatCount: String,
    getZoneName: () => 'zone'
  };
  for (const name of names)
    bindings[`${name}Readout`] = {setValue: (value: unknown) => values.set(name, value)};
  const run = evaluate(
    code(explorer, '    function refreshReadouts(', '    const readSummary ='),
    bindings,
    'refreshReadouts'
  );
  const snapshot = {
    extent: [0, 0, 1, 1],
    counts: [2],
    sums: [2],
    boundaryCounts: [0],
    pointBoundaryMask: [0, 0],
    pointZones: [0, 0],
    exactZones: [0, 0xffffffff],
    outsideCount: 0,
    unassignedBoundaryCount: 0,
    crossingCount: 2,
    overflow: 0,
    incomplete: 0,
    exactOverflow: 1,
    exactUncertainCount: 0
  };
  for (const cause of ['exactOverflow', 'exactUncertainCount']) {
    snapshot.exactOverflow = Number(cause === 'exactOverflow');
    snapshot.exactUncertainCount = Number(cause === 'exactUncertainCount');
    run(snapshot);
    expect(values.get('inside')).toMatch(/^2 of 2/);
    expect(values.get('exact')).toMatch(/unavailable/);
    expect(values.get('boundHolds')).toMatch(/unavailable/);
  }
});

it.each([
  'showcase',
  'explorer'
])('B2 %s point layers disappear on loss and return on recovery', route => {
  const path = route === 'showcase' ? `${showcase}joins/raster-join.compute.ts` : explorer;
  const source = code(
    path,
    route === 'showcase' ? '    getLayers() {' : '      getLayers() {',
    route === 'showcase' ? '    destroy()' : '      getTooltip:'
  );
  const method = source.slice(0, source.lastIndexOf('},') + 2);
  const layer = class {
    props: Record<string, unknown>;
    constructor(props: Record<string, unknown>) {
      this.props = props;
    }
  };
  const bindings: Record<string, unknown> = {
    ctx: {
      options: {layer: 'choropleth', points: 'mismatch', showBoundary: false, metric: 'count'},
      theme: () => 'dark',
      requestLayers() {}
    },
    origin: [0, 0],
    variant: {width: 1, height: 1},
    areaSet: {outline: []},
    polygons: {},
    rasterCounts: null,
    exactCounts: null,
    zips: {origin: [0, 0], outlineSegments: []},
    snapshot: null,
    SpatialAnalysisPointLayer: layer,
    SpatialAnalysisSegmentLayer: layer,
    SpatialAnalysisRasterLayer: layer,
    ZoneFillLayer: layer,
    pointCount: 2,
    displayKind: 'choropleth',
    showBoundary: false,
    RASTER_WIDTH: 1,
    RASTER_HEIGHT: 1,
    RASTER_JOIN_HIDDEN: -1
  };
  const scope = new Proxy(bindings, {
    has: () => true,
    get: (object, key: string | symbol) =>
      typeof key === 'symbol'
        ? undefined
        : key in object
          ? object[key]
          : key in globalThis
            ? (globalThis as Record<string, unknown>)[key]
            : {}
  });
  const encode = code(
    path,
    route === 'showcase'
      ? '    encode(commandEncoder, frame) {'
      : '      encode(commandEncoder: CommandEncoder, frame: SpatialAnalysisFrame) {',
    route === 'showcase' ? '    getLayers()' : '      getLayers()'
  );
  Object.assign(bindings, {
    writeExtent: () => false,
    frames: 0,
    encodedFrames: 0,
    dirty: true,
    settle: 0,
    SETTLE_FRAMES: 0,
    extentMode: 'city',
    context: {updateLayers() {}},
    rasterCompiled: {encode() {}},
    readbackPending: true
  });
  bindings['variant'] = {width: 1, height: 1, compiled: {encode() {}}, reader: {flush() {}}};
  const javascript = transpileModule(`const instance={${encode}${method}}; return instance;`, {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const instance = new Function('scope', `with(scope){${javascript}}`)(scope);
  const points = () =>
    instance
      .getLayers()
      .filter((entry: {props: Record<string, unknown>}) =>
        String(entry.props['id']).includes('points')
      );
  expect(points()).toHaveLength(0);
  bindings['rasterCounts'] = Uint32Array.of(2);
  bindings['exactCounts'] = Uint32Array.of(2);
  bindings['snapshot'] = {overflow: 0, incomplete: 0, exactOverflow: 0, exactUncertainCount: 0};
  expect(points()).toHaveLength(1);
  instance.encode({}, {frameIndex: 1});
  expect(points()).toHaveLength(0);
  bindings['rasterCounts'] = null;
  bindings['exactCounts'] = null;
  bindings['snapshot'] = {overflow: 1, incomplete: 1};
  expect(points()).toHaveLength(0);
});

it('B4 counts-only timestamp measurement reads loss and withholds normal timing', async () => {
  const values = new Map();
  let reads = 0;
  const loss = {
    readAsync: async () => {
      reads++;
      return new Uint8Array(Uint32Array.of(1).buffer);
    }
  };
  const clean = {
    readAsync: async () => {
      reads++;
      return new Uint8Array(4);
    }
  };
  const resources = {destroy() {}};
  const source = code(
    `${showcase}joins/raster-join.compute.ts`,
    '  async function measureAll()',
    '  async function buildExactGraph()'
  );
  const run = evaluate(
    `let measuring=false,destroyed=false,dirty=false; ${source}`,
    {
      ctx: {
        options: {resolutionLevel: 0},
        setReadout: (key: string, value: unknown) => values.set(key, value)
      },
      device: {},
      getResolution: () => 1,
      exactOverflow: clean,
      exactUncertainCount: clean,
      buildVariant: async () => ({
        compiled: {destroy() {}},
        reader: {stop() {}},
        resources,
        rasterOverflow: clean,
        incomplete: clean,
        exactUncertainCount: clean
      }),
      buildExactGraph: async () => ({
        compiled: {destroy() {}},
        resources,
        overflow: clean,
        uncertainCount: clean
      }),
      buildRasterOnlyGraph: async () => ({compiled: {destroy() {}}, resources, overflow: loss}),
      measureCompiledGraph: async () => ({milliseconds: 1}),
      formatCompiledGraphTiming: () => '1 ms'
    },
    'measureAll'
  );
  await run();
  expect(reads).toBe(6);
  expect(values.get('timing')).toMatch(/incomplete|unavailable/);
});

it('S1 snippet forwards producer status and checks before publication; earlier zone recipe has one output map', () => {
  const join = readFileSync(`${showcase}joins/raster-join.scene.ts`, 'utf8');
  expect(join).toContain('producerOverflow: overflow');
  expect(join).toContain('status: {incomplete}');
  expect(join).toMatch(/incomplete.*readAsync|readAsync.*incomplete/);
  const scene = readFileSync(`${showcase}movement/zone-dwell.scene.ts`, 'utf8');
  const recipe = scene.slice(
    scene.indexOf('const events = `'),
    scene.indexOf('const eventsCompiled')
  );
  expect(recipe.match(/\boutput:/g)).toHaveLength(1);
});

it('S5 formulas distinguish exact -2^63 from the negative saturation replacement', () => {
  const source = readFileSync(
    'modules/lenses/src/gpu-spatial-analysis/computation-pack/operators/cells-formulas.ts',
    'utf8'
  );
  expect(source).toContain('[-2^63, 2^63-1]');
  expect(source).toContain('negative overflow');
  expect(source).toContain('-(2^63-1)');
});

it('S2 GPU regression reuses one compiled graph across failure and both recovery cases', () => {
  const source = code(
    'modules/lenses/test/computation-lenses/showcase-join-readouts.spec.ts',
    "it('COMMON-W3",
    '\n});'
  );
  expect(source).not.toContain('for (const capacity');
  expect(source).toContain('fixture.extent.write(');
  expect(source.match(/graph.compile\(/g)).toHaveLength(1);
});

it('earlier viewport fix compares the same extent and B5 suppresses uncertain reference only', () => {
  const values = new Map();
  const run = evaluate(
    `const zoneCount=1,pointCount=2; let rasterCounts,exactCounts,valueRange; const zoneValues={write(){}}; const formatInteger=String; ${code(`${showcase}joins/raster-join.compute.ts`, '  function handleSummary(', '  /** Makes a built variant')}`,
    {
      ctx: {
        options: {metric: 'count'},
        setReadout: (key: string, value: unknown) => values.set(key, value),
        requestLayers() {},
        setLegendExtent() {}
      }
    },
    'handleSummary'
  );
  // One of two polygon members is outside the 1x1 raster. Citywide reference count 2 must
  // become 1 for the comparison; both assignments are certain, so the boundary bound holds.
  const words = new Uint32Array(23);
  words[0] = 1;
  words[2] = 2;
  words[5] = 1;
  words.set([0, 0], 15); // Exact zone IDs: both points match zone 0.
  words.set([1, 0], 17); // Extent membership: only the first point is in the raster.
  new Float32Array(words.buffer).set([0, 0, 1, 1], 19);
  run(words.buffer, 1, 1, 10);
  expect(values.get('bound')).toMatch(/^1 of 1/);
  expect(values.get('joined')).toBe('1 by raster, 1 exact, 1 outside the raster');
  for (const word of [8, 10]) {
    words[word] = 1;
    run(words.buffer, 1, 1, 10);
    expect(values.get('joined')).toBe('1 by raster; exact reference unavailable');
    expect(values.get('bound')).toMatch(/unavailable/);
    words[word] = 0;
  }
});

it('B6 actual live callback arriving first keeps static flash diagnostics pending', () => {
  const values = new Map();
  const path = `${showcase}earth/storm-outage-exposure.compute.ts`;
  const format = code(path, '  function updateOverflowReadout()', '  function updateValueRange()');
  const readers = code(path, '  const staticReader =', '  function exposureOf(');
  const bindings: Record<string, unknown> = {
    ctx: {setReadout: (key: string, value: unknown) => values.set(key, value)},
    resources: {},
    countyCount: 1,
    population: [1],
    inRegion: [1],
    countyStatic: [0, 1, 0, 0],
    flashScale: 1,
    fixedValueRanges: {},
    updateValueRange() {},
    refreshStatistics() {},
    formatInteger: String,
    SummaryReader: class {
      callback: (bytes: ArrayBuffer) => void;
      constructor(_r: unknown, _n: string, _s: unknown, callback: (bytes: ArrayBuffer) => void) {
        this.callback = callback;
      }
    }
  };
  for (const name of [
    'peakOutage',
    'flashTotals',
    'fullEventTrackLengths',
    'zonalOverflow',
    'peakOverflow',
    'staticLineOverflow',
    'zonalUncertainCount',
    'countyValues',
    'outageNow',
    'trackLengths',
    'flashSums',
    'flashSelected',
    'lineOverflow',
    'snapshotTimeBuffer'
  ])
    bindings[name] = {};
  const state = evaluate(
    `let staticOverflowed=false, exposureOverflowed=false, staticUncertainCount=null, staticFlashOverflow=null, destroyed=false, latest,peakPerThousand; ${format} ${readers}`,
    bindings,
    '({live:exposureReader.callback,static:staticReader.callback})'
  );
  const live = new ArrayBuffer(28);
  new Uint32Array(live)[5] = 1;
  state.live(live);
  expect(values.get('overflow')).toMatch(/pending/);
  expect(values.get('overflow')).not.toMatch(/no uncertain/);
  const fixed = new ArrayBuffer(28);
  state.static(fixed);
  expect(values.get('overflow')).toMatch(/no uncertain/);
  expect(values.get('overflow')).not.toMatch(/flash totals are not exact/);
});

it('B4 explorer timestamp path reads both producers before publishing speedup', async () => {
  let reads = 0;
  const values = new Map();
  const failed = {
      readAsync: async () => {
        reads++;
        return new Uint8Array(Uint32Array.of(1).buffer);
      }
    },
    clear = {
      readAsync: async () => {
        reads++;
        return new Uint8Array(4);
      }
    };
  const run = evaluate(
    `let measuring=false,destroyed=false;${code(explorer, '    const measureGraphs =', '    // --- Tooltip')}`,
    {
      device: {},
      context: {signal: null},
      rasterCompiled: {},
      exactCompiled: {},
      overflow: failed,
      incomplete: failed,
      exactOverflow: clear,
      exactUncertainCount: clear,
      measureCompiledGraph: async () => ({milliseconds: 1}),
      formatCompiledGraphTiming: () => '1 ms',
      formatSpeedup: () => '2x',
      rasterTimingReadout: {setValue: (value: unknown) => values.set('raster', value)},
      exactTimingReadout: {setValue: (value: unknown) => values.set('exact', value)},
      speedupReadout: {setValue: (value: unknown) => values.set('speedup', value)}
    },
    'measureGraphs'
  );
  await run();
  expect(reads).toBe(3);
  expect(values.get('speedup')).toMatch(/unavailable/);
});

it('STA-04 LiDAR binds row-aligned status to both consumers and keeps the sample uncertified', () => {
  const source = readFileSync(`${showcase}terrain/lidar-ground.compute.ts`, 'utf8');
  for (const name of ['GPUPointCloudStatisticalFilter', 'GPUPointCloudFeatures']) {
    const construction = source.slice(source.indexOf(`new ${name}({`)).split('\n    })')[0];
    expect(construction).toContain('neighborhoodStatus,');
  }
  const neighborhoods = source
    .slice(source.indexOf('new GPUPointCloudNeighborhoods({'))
    .split('\n    })')[0];
  expect(neighborhoods).not.toContain('supportBounds:');
  expect(source).toMatch(/ctx\.setReadout\(\s*'featureSupport',/);
  const scene = readFileSync(`${showcase}terrain/lidar-ground.scene.ts`, 'utf8');
  expect(scene).toContain('counts, neighborhoodStatus, capacity: 16');
  expect(scene).toContain('Halo completeness is not certified');
  expect(scene).toContain('eigensolve');
});

it('STA-04 display separates usable incomplete support, truncation, degeneracy and failed eigensolve', () => {
  // Documented feature bits: 8 support loss, 2 truncation, 1 degeneracy, 4 failed solve.
  expect([0, 8, 2, 1, 4, 10, 12].map(getLidarFeatureDisplayClass)).toEqual([0, 1, 2, 3, 4, 2, 4]);
});

it('UNI-01 seed conversion uses the public producer and the southward affine starts at the northern edge', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const coverage = createTransientView(graph, 'coverage', 'uint32', 4);
    const output = createTransientView(graph, 'terrain-validity', 'uint32', 4);
    const producer = createGroundSeedCoverageNode(graph, coverage, output);
    expect(producer).toBeInstanceOf(GPUPointCloudCoverageValidity);
    expect(producer.props.coverage).toBe(coverage);
    expect(producer.props.output.terrainValidity).toBe(output);
    expect(producer.props.output.surfaceValidity).not.toBe(output);
    expect(producer.props.output.canopyValidity).not.toBe(output);
    // 2 rows of 10 m extend north from 200 to 220; row zero begins at 220.
    expect(
      getRasterMetadataFromPointCloudGrid({
        origin: [100, 200],
        cellSize: 10,
        width: 2,
        height: 2,
        rowDirection: 'south'
      }).affine
    ).toEqual([10, 0, 100, 0, -10, 220]);
  } finally {
    device.destroy();
  }
});

it('UNI-01 LiDAR uses public coverage conversion and an explicit raster-band handoff', () => {
  const helper = readFileSync(`${showcase}terrain/point-cloud-ground-support.ts`, 'utf8');
  expect(helper).toContain('new GPUPointCloudCoverageValidity(');
  expect(helper).not.toContain('(coverage[coverageOffset + index] & 2u)');
  const source = readFileSync(`${showcase}terrain/lidar-ground.compute.ts`, 'utf8');
  expect(source).toContain(
    'makeGPURasterBand(terrain, getRasterMetadataFromPointCloudGrid(RASTER_GRID)'
  );
  expect(source).toContain('validity: terrainValidity');
  const scene = readFileSync(`${showcase}terrain/lidar-ground.scene.ts`, 'utf8');
  expect(scene).toContain('GPUPointCloudCoverageValidity');
  expect(scene).toContain('getRasterMetadataFromPointCloudGrid(grid)');
  expect(scene).toContain('createSupportedGroundClassesNode');
  expect(scene).toContain('createGroundDisplayClassesNode');
});

it('W5-29 wildfire snippet refuses publication on rasterizer loss and keeps independent ring zones', async () => {
  const scene = readFileSync(`${showcase}raster/wildfire-terrain.scene.ts`, 'utf8');
  const start = scene.indexOf('graph.add(new GPURasterZonalStatistics({');
  const finish = scene.indexOf('}));', start) + 4;
  const statement = scene.slice(start, finish).replace(/\$\{state.sumOrder\}/g, 'atomic');
  const overflow = {},
    incomplete = {},
    zoneFire = {};
  const entries: Array<Record<string, unknown>> = [];
  evaluate(
    statement,
    {
      graph: {add: (entry: Record<string, unknown>) => entries.push(entry)},
      GPURasterZonalStatistics: class {
        constructor(props: object) {
          return props;
        }
      },
      mercatorGrid: {},
      zoneFire,
      zoneId: {},
      slopeBand: {},
      fireCount: 1,
      overflow,
      incomplete,
      cellCounts: {},
      valueCounts: {},
      sums: {},
      maximums: {}
    },
    'undefined'
  );
  expect(entries[0]['producerOverflow']).toBe(overflow);
  expect(entries[0]['zones']).toBe(zoneFire);
  expect(entries[0]['output']).toMatchObject({status: {incomplete}});
  expect(scene).toContain('view: zoneFire');
  expect(scene).toContain('let nearest = alloc[allocOffset + index]');
  expect(scene).toContain('slopeSummaryReader.request(commandEncoder)');
  expect(scene).toContain('publishTerrainSummary');
  const publication = scene.slice(
    scene.indexOf('const slopeSummaryReader ='),
    scene.indexOf('} // Submit this encoder') + 1
  );
  for (const loss of [1, 0]) {
    const published: number[][] = [];
    const messages: string[] = [];
    const run = evaluate(
      publication,
      {
        resources: {},
        fireCount: 1,
        statsCompiled: {encode() {}},
        incompleteBuffer: {},
        valueCountsBuffer: {},
        sumsBuffer: {},
        SummaryReader: class {
          constructor(
            _resources: unknown,
            _id: string,
            _sources: unknown,
            readonly callback: (bytes: ArrayBuffer) => void
          ) {}
          request() {
            const snapshot = new ArrayBuffer(20);
            new Uint32Array(snapshot).set([loss, 1, 0]);
            this.callback(snapshot);
          }
        },
        clearTerrainSummary: (message: string) => messages.push(message),
        publishTerrainSummary: (counts: Uint32Array, sums: Float32Array) =>
          published.push([...counts, ...sums])
      },
      'encodeSlopeSummary'
    );
    await run({});
    expect(published).toEqual(loss ? [] : [[1, 0, 0, 0]]);
    expect(messages).toEqual(loss ? ['pending', 'unavailable: rasterizer loss'] : ['pending']);
  }
});

it('W5-30 displayed static membership diagnostics stay pending until their own readback', async () => {
  const scene = readFileSync(`${showcase}earth/storm-outage-exposure.scene.ts`, 'utf8');
  expect(scene).toContain('uncertainCount: staticUncertainCount');
  const start = scene.indexOf('// Static membership diagnostics accompany');
  expect(start).toBeGreaterThan(0);
  const end = scene.indexOf('// When the playhead', start);
  const published: Array<string> = [];
  let release: ((bytes: Uint8Array) => void) | undefined;
  const pending = new Promise<Uint8Array>(resolve => {
    release = resolve;
  });
  const run = evaluate(
    scene.slice(start, end),
    {
      staticOverflowBuffer: {readAsync: () => pending},
      staticUncertainCountBuffer: {readAsync: async () => new Uint8Array(Uint32Array.of(2).buffer)},
      setFlashMembershipStatus: (value: string) => published.push(value)
    },
    'readStaticMembership'
  );
  expect(published).toEqual(['static flash membership diagnostics pending']);
  const reading = run();
  expect(published).toHaveLength(1);
  release!(new Uint8Array(Uint32Array.of(0).buffer));
  await reading;
  expect(published[1]).toMatch(/2 uncertain.*not exact/);
});
