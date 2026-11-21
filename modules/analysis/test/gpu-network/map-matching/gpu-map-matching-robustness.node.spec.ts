// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_MAP_MATCHING_PARAMETER_LENGTH,
  GPUMapMatching,
  getGPUMapMatchingMaximumPointCount,
  getGPUMapMatchingMaximumEdgeCount,
  getGPUMapMatchingParameterValues,
  type GPUMapMatchingProps
} from '../../../src/gpu-network/map-matching/index';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  createNullWebGPUDevice,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

it('measures map-matching edge lengths from geometry rather than unused routing weights', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 2);
  recordWGSLKernelNodes(graph);
  new GPUMapMatching({
    ...props,
    topology: {
      ...props.topology,
      weights: createTransientView(graph, 'unused-weights', 'float32', 8)
    },
    candidateCount: 1
  }).getCommandNodes(graph);
  const record = getWGSLKernelNodeRecord(graph, 'map-matching-edge-table')!;
  expect(auditWGSLModule(record.source)).toEqual([]);
  expect(record.bindings.map(binding => binding.name)).not.toContain('weights');
  const edgeTable = new Uint32Array(8 * 3);
  const data: Record<string, Uint32Array | Float32Array> = {
    adjacency: Uint32Array.of(0, 1, 1, 1, 1),
    edgeTargets: new Uint32Array(8).fill(1),
    nodePositions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0),
    edgeTable
  };
  new WgslExec(new WgslParser().parse(record.source)).dispatchWorkgroups(
    'main',
    [1, 1, 1],
    {0: Object.fromEntries(record.bindings.map((binding, index) => [index, data[binding.name]]))},
    {constants: record.constants}
  );
  expect(Array.from(edgeTable.slice(0, 2))).toEqual([0, 1]);
  expect(new Float32Array(edgeTable.buffer)[2]).toBe(1);
});

it('evaluates near-maximum beta against geometric and genuinely large route distances', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  new GPUMapMatching({...createProps(graph, 2), candidateCount: 1}).getCommandNodes(graph);
  const record = getWGSLKernelNodeRecord(graph, 'map-matching-forward')!;
  expect(auditWGSLModule(record.source)).toEqual([]);
  expect(record.source).toContain('getScaledQuotient(abs(straight - route), beta, 1.0, 0)');
  const beta = Math.fround(1e38);
  for (const route of [0.5, Math.fround(5e37)]) {
    const scores = new Float32Array(4);
    const data: Record<string, Uint32Array | Float32Array> = {
      trackOffsets: Uint32Array.of(0, 2),
      points: Float32Array.of(0.25, 0, 0.75, 0),
      candidates: new Uint32Array(Float32Array.of(0, 0.25, 0, 0, 0.75, 0).buffer),
      parameters: getGPUMapMatchingParameterValues({sigma: 1, beta, searchRadius: 0.1}),
      routes: Float32Array.of(-1, 0, route, 0),
      scores,
      backPointers: new Uint32Array(2)
    };
    // The node interpreter does not implement frexp/ldexp. Supply their IEEE normal-f32
    // definitions for these finite normal fixtures; retain the production quotient and model.
    const source = record.source
      .replace(/^\s*_\s*=[^;]*;/gm, '')
      .replace(/\bfrexp\(/g, 'testFrexp(')
      .replace(/\bldexp\(/g, 'testLdexp(');
    const builtins = `
struct TestFrexpResult { fract: f32, exp: i32 }
fn testFrexp(value: f32) -> TestFrexpResult {
  let bits = bitcast<u32>(value);
  return TestFrexpResult(bitcast<f32>((bits & 0x807fffffu) | 0x3f000000u),
    i32((bits >> 23u) & 255u) - 126);
}
fn testLdexp(value: f32, exponent: i32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) + (bitcast<u32>(exponent) << 23u));
}`;
    expect(auditWGSLModule(builtins + source)).toEqual([]);
    new WgslExec(new WgslParser().parse(builtins + source)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: Object.fromEntries(record.bindings.map((binding, index) => [index, data[binding.name]]))},
      {constants: record.constants}
    );
    // One candidate: normalized scores are zero; the two offsets sum to the likelihood.
    expect([scores[0], scores[2]]).toEqual([0, 0]);
    const expected = -Math.log(2 * Math.PI) - Math.log(beta) - Math.abs(0.5 - route) / beta;
    expect(Math.abs(scores[1] + scores[3] - expected), `route ${route}`).toBeLessThan(2e-4);
  }
});

it('scales geographic candidate radii before dividing by a near-maximum sphere radius', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const nodes = new GPUMapMatching({
    ...createProps(graph, 2),
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      sphereRadius: 1e38
    },
    bounds: {minimum: [-1, -44.99], maximum: [91, 11]},
    cellSize: 45,
    candidateCount: 1
  }).getCommandNodes(graph);
  for (const {source, constants} of getParsedKernelSources(nodes).getConfigurations(
    'map-matching-candidates'
  )) {
    expect(auditWGSLModule(source, constants)).toEqual([]);
    expect(source).toContain('getScaledQuotient(radius, NETWORK_EARTH_RADIUS, 1.0, 0)');
    expect(source).not.toContain('radius / NETWORK_EARTH_RADIUS');
  }
});

it('settles a route-search key exactly at a bound where adding one rounds away', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  new GPUMapMatching({...createProps(graph, 2), candidateCount: 1}).getCommandNodes(graph);
  const record = getWGSLKernelNodeRecord(graph, 'map-matching-routes')!;
  expect(auditWGSLModule(record.source)).toEqual([]);
  const edgeTable = new Uint32Array(24);
  const edgeLengths = new Float32Array(edgeTable.buffer);
  // CSR edges: destination 0 -> 3, origin 1 -> 2, connecting edge 2 -> 0.
  edgeTable.set([0, 3, 0, 1, 2, 0, 2, 0, 0]);
  edgeLengths[2] = 2;
  edgeLengths[5] = 1;
  edgeLengths[8] = 16777216;
  const candidates = Uint32Array.of(1, 0, 0, 0, 0, 0);
  new Float32Array(candidates.buffer)[1] = 1;
  const routes = new Float32Array(4);
  const data: Record<string, Uint32Array | Float32Array> = {
    pointStarts: Uint32Array.of(1, 0),
    points: Float32Array.of(1, 0, 16777218, 0),
    nodePositions: Float32Array.of(16777218, 0, 0, 0, 1, 0, 16777220, 0),
    adjacency: Uint32Array.of(0, 1, 2, 3, 3),
    edgeTable,
    candidates,
    parameters: getGPUMapMatchingParameterValues({
      sigma: 1,
      beta: 1,
      searchRadius: 1,
      routeFactor: 0,
      routeSlack: 16777216
    }),
    routes
  };
  const source = record.source.replace(/^\s*_\s*=[^;]*;/gm, '');
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
    'main',
    [1, 1, 1],
    {0: Object.fromEntries(record.bindings.map((binding, index) => [index, data[binding.name]]))},
    {constants: record.constants}
  );
  expect(routes[2]).toBe(16777216);
  expect(routes[3]).toBe(0);
});

it('uses the shared wide likelihood sum with headroom for every term in a track', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph, 3);
  const likelihood = createTransientView(graph, 'likelihood', 'float32', 1);
  recordWGSLKernelNodes(graph);
  new GPUMapMatching({
    ...props,
    candidateCount: 1,
    output: {...props.output, trackLogLikelihoods: likelihood}
  }).getCommandNodes(graph);
  const source = getWGSLKernelNodeRecord(graph, 'map-matching-backward')!.source;
  expect(auditWGSLModule(source)).toEqual([]);
  expect(source).toContain('likelihoodSum: vec3<u32>');
  expect(source).toContain('getFixedPointUnitExponent(maximumExponent, 33u)');
  expect(source).toContain('getFixedPointFloat(likelihoodSum, likelihoodUnitExponent)');
  expect(source).not.toContain('sumWhole');
});

it('rejects non-STORAGE caller views before creating map-matching scratch', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  const buffer = graph.createTransientBuffer({
    id: 'copy-only',
    byteLength: 48,
    usage: Buffer.COPY_DST
  });
  const positions = graph.createDataView(buffer, {format: 'float32x2', length: 6});
  const contributor = new GPUMapMatching({...props, positions});
  const create = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => graph.add(contributor)).toThrow(/map-matching.*STORAGE/);
  expect(create).not.toHaveBeenCalled();
});

it('rejects nonfinite float32 shader literals and cell sizes that round to zero', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  for (const cellSize of [Number.MIN_VALUE, 1e100, Infinity, NaN]) {
    expect(() => new GPUMapMatching({...props, cellSize})).toThrow(/map-matching cellSize/);
  }
  for (const coordinate of [Infinity, NaN, 1e100]) {
    expect(
      () => new GPUMapMatching({...props, bounds: {minimum: [0, 0], maximum: [coordinate, 200]}})
    ).toThrow(/map-matching bounds/);
  }
  expect(() => new GPUMapMatching({...props, minimumFixSpacingSigmas: 1e100})).toThrow(
    /map-matching minimumFixSpacingSigmas/
  );
  for (const sphereRadius of [1e100, Number.MIN_VALUE]) {
    expect(
      () =>
        new GPUMapMatching({
          ...props,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            sphereRadius
          }
        })
    ).toThrow(/map-matching spatialContext.sphereRadius must be a finite number other than 0/);
  }
});

it('bounds grid prefix totals at construction even on larger-binding devices', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 2 ** 30}
  });
  expect(getGPUMapMatchingMaximumEdgeCount(device)).toBe(16777215);
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const neighbors = createTransientView(graph, 'too-many-grid-insertions', 'uint32', 16777216);
  expect(() => new GPUMapMatching({...props, topology: {...props.topology, neighbors}})).toThrow(
    /map-matching topology edges.*16777215/
  );
});

it('preflights aligned input bindings and dense grid scratch before declaring resources', () => {
  for (const oversizedGrid of [true, false]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = createProps(graph);
    const backing = createTransientView(graph, 'aligned-points', 'uint32', 64).buffer;
    const positions = graph.createDataView(backing, {
      format: 'float32x2',
      byteOffset: 208,
      length: 6
    });
    const contributor = new GPUMapMatching({
      ...props,
      positions: oversizedGrid ? props.positions : positions,
      candidateCount: 1,
      entryCapacity: 1,
      ...(oversizedGrid ? {cellSize: 1} : {})
    });
    Object.defineProperty(graph.device, 'limits', {
      value: {...graph.device.limits, maxStorageBufferBindingSize: 200}
    });
    const create = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      oversizedGrid ? /map-matching grid.*limit 200/ : /map-matching view.*256.*limit 200/
    );
    expect(create).not.toHaveBeenCalled();
  }
});

it('does not allocate an unused hop table for one-candidate large batches', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUMapMatching({...createProps(graph, 7000000), candidateCount: 1});
  const create = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  expect(create.mock.calls.some(([descriptor]) => descriptor.id === 'map-matching-hops')).toBe(
    false
  );
});

it('uses atomic shared overflow flags and runtime hash initialization in audited WGSL', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  const edges = createTransientView(graph, 'traversed-edges', 'uint32', 1);
  const counts = createTransientView(graph, 'traversed-counts', 'uint32', 1);
  const nodes = new GPUMapMatching({
    ...props,
    routeNodeBudget: 128,
    traversedEdgeCapacity: 1,
    output: {...props.output, traversedEdges: {edges, counts}}
  }).getCommandNodes(graph);
  const sources = getParsedKernelSources(nodes);
  for (const {id: name, source, constants} of sources.configurations)
    expect(auditWGSLModule(source, constants), name).toEqual([]);
  for (const name of ['cell-fill', 'route-status', 'route-edges']) {
    expect(sources.get(`map-matching-${name}`)).toContain('atomicStore(');
  }
  expect(sources.get('map-matching-routes')).toContain('min(arrayLength(&adjacency), 1u)');
});

let serial = 0;

function createProps(graph: GPUCommandGraph, pointCount = 6, trackCount = 1): GPUMapMatchingProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `map-matching-robustness-view-${serial++}`, format, length);
  return {
    id: 'map-matching',
    positions: view('float32x2', pointCount),
    trackOffsets: view('uint32', trackCount + 1),
    topology: {
      nodePositions: view('float32x2', 4),
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8)
    },
    parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
    cellSize: 50,
    bounds: {minimum: [0, 0], maximum: [200, 200]},
    output: {matchedEdges: view('uint32', pointCount)}
  };
}

it('GPUMapMatching checks its per-point scratch against the storage binding limit', () => {
  const device = createNullWebGPUDevice();
  // The routes table takes 8 * 9 * 4 = 288 bytes per point at 8 candidates: 128 MiB / 288.
  const maximum = getGPUMapMatchingMaximumPointCount(device);
  expect(maximum).toBe(466033);
  expect(getGPUMapMatchingMaximumPointCount(device, 4)).toBe(Math.floor(134217728 / 80));
  // One candidate: the 20-byte hop table binds tighter than the 12-byte candidate table.
  expect(getGPUMapMatchingMaximumPointCount(device, 1, true)).toBe(Math.floor(134217728 / 20));
  const getNodes = (pointCount: number, candidateCount?: number) => {
    const graph = new GPUCommandGraph(device, {id: `map-matching-scratch-${serial++}`});
    const props = createProps(graph, pointCount);
    return () => new GPUMapMatching({...props, candidateCount}).getCommandNodes(graph);
  };
  expect(getNodes(maximum)).not.toThrow();
  expect(getNodes(maximum + 1)).toThrow(/positions rows must be at most 466033/);
  expect(getNodes(maximum + 1, 4)).not.toThrow();
});

it('GPUMapMatching validates minimumFixSpacingSigmas and traversedEdges', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-validation'});
  const base = createProps(graph);
  const view = <Format extends 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `map-matching-validation-${serial++}`, format, length);
  expect(() => new GPUMapMatching({...base, minimumFixSpacingSigmas: -1})).toThrow(
    /minimumFixSpacingSigmas must be a non-negative number/
  );
  const traversedEdges = {edges: view('uint32', 8), counts: view('uint32', 1)};
  const withOutput = (overrides: Partial<GPUMapMatchingProps>) => () =>
    new GPUMapMatching({
      ...base,
      output: {...base.output, traversedEdges},
      ...overrides
    });
  expect(withOutput({})).toThrow(/traversedEdgeCapacity must be an integer/);
  expect(withOutput({traversedEdgeCapacity: 4})).toThrow(
    /output\.traversedEdges\.edges must contain trackCount \* traversedEdgeCapacity rows/
  );
  expect(() => withOutput({traversedEdgeCapacity: 8})()).not.toThrow();
  expect(
    () =>
      new GPUMapMatching({
        ...base,
        traversedEdgeCapacity: 8,
        output: {
          ...base.output,
          traversedEdges: {...traversedEdges, counts: view('uint32', 2)}
        }
      })
  ).toThrow(/output\.traversedEdges\.counts must contain 1 rows/);
  expect(
    () =>
      new GPUMapMatching({
        ...base,
        traversedEdgeCapacity: 8,
        output: {
          ...base.output,
          traversedEdges: {...traversedEdges, status: {overflow: view('uint32', 2)}}
        }
      })
  ).toThrow(/output\.traversedEdges\.status\.overflow must contain 1 rows/);
});

it('GPUMapMatching without the new options keeps its node list', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-plain-nodes'});
  const ids = new GPUMapMatching(createProps(graph)).getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => /fix-filter|predecessors|hops|route-edges/.test(id))).toBe(false);
  expect(ids).toContain('map-matching-clear-starts');
});

it('GPUMapMatching fix filter and traversed route kernels parse and stay within the limits', () => {
  for (const withTable of [false, true]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
      id: `map-matching-new-kernels-${withTable}`
    });
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `map-matching-new-kernels-${serial++}`, format, length);
    const base = createProps(graph);
    const nodes = new GPUMapMatching({
      ...base,
      minimumFixSpacingSigmas: 2,
      traversedEdgeCapacity: 16,
      routeNodeBudget: 128,
      ...(withTable
        ? {
            routeTable: {
              offsets: view('uint32', 5),
              targets: view('uint32', 16),
              costs: view('float32', 16),
              costLimit: view('float32', 1),
              status: {complete: view('uint32', 1)}
            }
          }
        : {}),
      output: {
        ...base.output,
        trackLogLikelihoods: view('float32', 1),
        breaks: view('uint32', 6),
        traversedEdges: {edges: view('uint32', 16), counts: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const ids = nodes.map(node => node.id);
    expect(ids).toContain('map-matching-fix-filter');
    expect(ids).not.toContain('map-matching-starts');
    expect(ids.indexOf('map-matching-hops')).toBeLessThan(ids.indexOf('map-matching-route-edges'));
    const sources = getParsedKernelSources(nodes);
    for (const {id, source} of sources.configurations) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }
    expect(sources.get('map-matching-routes')).toContain(
      'predecessors[predecessorsOffset + point]'
    );
    expect(sources.get('map-matching-routes')).not.toContain('pointStarts');
    expect(sources.get('map-matching-forward')).toContain('DROPPED_FIX');
    expect(sources.get('map-matching-route-edges')).toContain('tableParent');
    for (const id of ['map-matching-route-edges', 'map-matching-routes']) {
      const source = sources.get(id);
      expect(source, id).toBeDefined();
      for (const kernel of sources.getConfigurations(id)) {
        expect(getPrivateMemoryBytes(kernel.source, kernel.constants), id).toBeLessThanOrEqual(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES
        );
      }
    }
  }
});
