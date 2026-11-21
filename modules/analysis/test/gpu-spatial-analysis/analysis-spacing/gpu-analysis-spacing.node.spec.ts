// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUAnalysisSpacing,
  GPU_ANALYSIS_SPACING_MAXIMUM_POINT_COUNT,
  GPU_ANALYSIS_SPACING_SUMMARY_LENGTH,
  decodeGPUAnalysisSpacingStatistics,
  type GPUAnalysisSpacingProps
} from '../../../src/gpu-spatial-analysis/analysis-spacing/index';
import {computeGPUAnalysisSpacingStatistics} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/index';
import {NEIGHBOR_SEARCH_CANONICAL_LONGITUDE_WGSL} from '../../../src/gpu-spatial-analysis/neighbor-search/neighbor-search-wgsl';
import {POINTS_AND_REGIONS_CASES} from '../../analysis-core/kernel-audit/cases-points-and-regions';
import {registerKernelAudit} from '../../analysis-core/kernel-audit/register-kernel-audit';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount,
  getPrivateMemoryBytes,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

registerKernelAudit(
  'analysis-spacing',
  POINTS_AND_REGIONS_CASES.filter(({exportName}) => exportName === 'GPUAnalysisSpacing')
);

let serial = 0;
function createProps(graph: GPUCommandGraph, length = 16): GPUAnalysisSpacingProps {
  return {
    id: `spacing-${serial++}`,
    positions: createTransientView(graph, `positions-${serial++}`, 'float32x2', length),
    spatialContext: {coordinateSpace: 'planar', metric: 'native'},
    gridSize: [4, 4],
    output: {
      summary: createTransientView(
        graph,
        `summary-${serial++}`,
        'uint32',
        GPU_ANALYSIS_SPACING_SUMMARY_LENGTH
      )
    }
  };
}

it('validates declared contexts, capacities, rank, packed inputs and output ownership', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props = createProps(graph);
  expect(() => new GPUAnalysisSpacing(props)).not.toThrow();
  const stridedPositions = graph.createDataView(props.positions.buffer, {
    format: 'float32x2',
    length: 8,
    byteStride: 16
  });
  expect(() => new GPUAnalysisSpacing({...props, positions: stridedPositions})).toThrow(
    /positions.*packed/
  );
  for (const length of [0, GPU_ANALYSIS_SPACING_MAXIMUM_POINT_COUNT + 1]) {
    expect(() => new GPUAnalysisSpacing(createProps(graph, length))).toThrow(
      /spacing-.*positions.length.*1000000/
    );
  }
  for (const rank of [0, 65, 2.5]) {
    expect(() => new GPUAnalysisSpacing({...props, neighborRank: rank})).toThrow(
      /neighborRank.*64/
    );
  }
  expect(() => new GPUAnalysisSpacing({...props, spatialContext: undefined!})).toThrow(
    /spatialContext.*declared/
  );
  expect(
    () =>
      new GPUAnalysisSpacing({
        ...props,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'kilometers'
        }
      })
  ).toThrow(/spatialContext.*meters/);
  expect(
    () =>
      new GPUAnalysisSpacing({
        ...props,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: 42
        }
      })
  ).toThrow(/mean Earth/);
  expect(
    () =>
      new GPUAnalysisSpacing({
        ...props,
        spatialContext: {coordinateSpace: 'topology-only', metric: 'none'}
      })
  ).toThrow();
  for (const capacity of [0, 0.5, 33554433]) {
    expect(() => new GPUAnalysisSpacing({...props, neighborCapacity: capacity})).toThrow(
      /neighborCapacity.*33554432/
    );
  }
  expect(() => new GPUAnalysisSpacing({...props, gridSize: [0, 1]})).toThrow(/gridSize/);
  expect(() => new GPUAnalysisSpacing({...props, gridSize: [2048, 2048]})).toThrow(
    /gridSize cells/
  );
  expect(
    () =>
      new GPUAnalysisSpacing({...props, mask: createTransientView(graph, 'bad-mask', 'uint32', 1)})
  ).toThrow(/mask.*16/);
  expect(
    () =>
      new GPUAnalysisSpacing({
        ...props,
        identities: createTransientView(graph, 'bad-identities', 'uint32', 1)
      })
  ).toThrow(/identities.*16/);
  expect(
    () =>
      new GPUAnalysisSpacing({
        ...props,
        output: {summary: createTransientView(graph, 'short-summary', 'uint32', 1)}
      })
  ).toThrow(/summary.*544/);
  const aliased = createProps(graph, GPU_ANALYSIS_SPACING_SUMMARY_LENGTH);
  expect(() => new GPUAnalysisSpacing({...aliased, mask: aliased.output.summary})).toThrow(
    /must not share/
  );
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPUAnalysisSpacing(props).getCommandNodes(otherGraph)).toThrow(/target graph/);
  const smallDevice = createNullWebGPUDevice();
  Object.defineProperty(smallDevice, 'limits', {
    value: {...smallDevice.limits, maxStorageBufferBindingSize: 128}
  });
  const smallGraph = new GPUCommandGraph(smallDevice);
  expect(() => new GPUAnalysisSpacing(createProps(smallGraph)).getCommandNodes(smallGraph)).toThrow(
    /neighborCapacity scratch/
  );
  const maximumProps = createProps(graph, GPU_ANALYSIS_SPACING_MAXIMUM_POINT_COUNT);
  expect(() =>
    new GPUAnalysisSpacing({
      ...maximumProps,
      neighborRank: 64,
      neighborCapacity: 1
    }).getCommandNodes(graph)
  ).toThrow(/positions.length \* neighborRank scratch/);
});

it('audits generated kernels, graph primitives and node budgets at small and million-row capacities', () => {
  for (const geographic of [false, true]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const props = createProps(graph);
    const contributor = new GPUAnalysisSpacing({
      ...props,
      mask: createTransientView(graph, `mask-${geographic}`, 'uint32', 16),
      identities: createTransientView(graph, `identities-${geographic}`, 'uint32', 16),
      spatialContext: geographic
        ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        : props.spatialContext
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(new Set(nodes.map(node => node.id)).size).toBe(nodes.length);
    expect(nodes.some(node => node.id.includes('valid-scan'))).toBe(true);
    expect(nodes.some(node => node.id.includes('grid'))).toBe(true);
    const sources = getParsedKernelSources(nodes);
    expect(sources.size).toBeGreaterThan(25);
    for (const {id: name, source: source, constants} of sources.configurations) {
      expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
      expect(getWorkgroupMemoryBytes(source, constants), name).toBeLessThanOrEqual(16384);
      expect(getPrivateMemoryBytes(source, constants), name).toBeLessThanOrEqual(8192);
    }
    const rowReducer = [...sources.values()].find(source =>
      source.includes('CSR rows are ID ordered')
    );
    expect(rowReducer).toContain('nearest = min(nearest, distance)');
    expect(rowReducer).toContain('farthest = max(farthest, distance)');
  }
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUAnalysisSpacing(
    createProps(graph, GPU_ANALYSIS_SPACING_MAXIMUM_POINT_COUNT)
  );
  expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
    contributor.estimatedCommandNodeCount
  );
});

function createSummary(options: {duplicates?: boolean; geographic?: boolean} = {}): Uint32Array {
  const words = new Uint32Array(GPU_ANALYSIS_SPACING_SUMMARY_LENGTH);
  const values = new Float32Array(words.buffer);
  words[0] = 4;
  words[3] = 12;
  words[5] = 3;
  words[6] = 4;
  words[7] = 1;
  words[25] = 4;
  values.set([0, 0, 3, 4, 1.5, 2, 2.5, 2.5, 3, 3, 3, 3, 3, 3, 3], 8);
  values[24] = 3;
  values.set([5, 5, 5, 5], 32);
  if (options.duplicates) {
    words[4] = 1;
    words[25] = 3;
    values.set([0, 0, 10, 0, 3.5, 0, Math.sqrt(16.75), 3.5, 0, 0, 2, 4.5, 2.5, 6, 6], 8);
    values[24] = 4;
    values.set([6, 10, 10, 10], 32);
  }
  if (options.geographic) words[26] = 1;
  values[28] = values[12] - values[8];
  values[29] = values[13] - values[9];
  return words;
}

it('decodes independently hand-computed rectangle and duplicate-location summaries against the CPU reference', () => {
  for (const duplicates of [false, true]) {
    const expected = computeGPUAnalysisSpacingStatistics(
      duplicates ? [0, 0, 0, 0, 4, 0, 10, 0] : [0, 0, 3, 0, 0, 4, 3, 4],
      {coordinateModel: 'planar'}
    )!;
    const decoded = decodeGPUAnalysisSpacingStatistics(createSummary({duplicates}))!;
    expect(decoded.nearestNeighbor).toEqual(expected.nearestNeighbor);
    expect(decoded.kDistance).toEqual(expected.kDistance);
    expect(decoded.center).toEqual(expected.center);
    expect(decoded.extent).toEqual(expected.extent);
    expect(decoded.standardDistance).toBeCloseTo(expected.standardDistance, 6);
    expect(decoded.medianDistanceFromCenter).toEqual(expected.medianDistanceFromCenter);
    expect(decoded.duplicateLocationCount).toBe(expected.duplicateLocationCount);
    expect(decoded.sampled).toBe(false);
    expect(decoded.sampleSize).toBe(4);
  }
  const bytes = new Uint32Array(GPU_ANALYSIS_SPACING_SUMMARY_LENGTH + 4);
  bytes.set(createSummary(), 4);
  expect(decodeGPUAnalysisSpacingStatistics(bytes.subarray(4))?.pointCount).toBe(4);
  const translated = createSummary();
  const translatedValues = new Float32Array(translated.buffer);
  translatedValues.set([10000000, 10000000, 10000003, 10000004], 8);
  translatedValues.set([1.5, 2], 28);
  expect(decodeGPUAnalysisSpacingStatistics(translated)?.center).toEqual([10000001.5, 10000002]);
});

it('decodes geographic extent sizes and empty/singleton states, and refuses overflow or broken summaries', () => {
  const words = createSummary({geographic: true});
  const values = new Float32Array(words.buffer);
  values.set([179, -1, -179, 1], 8);
  const geographic = decodeGPUAnalysisSpacingStatistics(words)!;
  expect(geographic.coordinateModel).toBe('geographic');
  expect(geographic.unit).toBe('metre');
  expect(geographic.extentWidth).toBeCloseTo(222390.160467, 4);
  words[0] = 1;
  words[6] = 0;
  expect(decodeGPUAnalysisSpacingStatistics(words)?.nearestNeighbor).toBeUndefined();
  words[0] = 0;
  expect(decodeGPUAnalysisSpacingStatistics(words)).toBeUndefined();
  words[2] = 1;
  expect(() => decodeGPUAnalysisSpacingStatistics(words)).toThrow(/overflow.*requiredCount 12/);
  const invalid = createSummary();
  new Float32Array(invalid.buffer)[14] = Infinity;
  expect(() => decodeGPUAnalysisSpacingStatistics(invalid)).toThrow(/finite/);
  invalid[7] = 9;
  expect(() => decodeGPUAnalysisSpacingStatistics(invalid)).toThrow(/version/);
  expect(() => decodeGPUAnalysisSpacingStatistics(new Uint32Array(1))).toThrow(
    /summary.length.*544/
  );
  const invalidCurve = createSummary();
  invalidCurve[6] = 1;
  expect(() => decodeGPUAnalysisSpacingStatistics(invalidCurve)).toThrow(/curve length/);
});

function createInterpretedSpacing(
  coordinates: Float32Array,
  geographic: boolean,
  neighborCapacity?: number
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const props = createProps(graph, coordinates.length / 2);
  const contributor = new GPUAnalysisSpacing({
    ...props,
    neighborCapacity,
    spatialContext: geographic
      ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      : props.spatialContext
  });
  const nodes = contributor.getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, coordinates);
  const view = (suffix: string, name: string): GraphDataView => {
    const binding = getWGSLKernelNodeRecord(graph, `${props.id}-${suffix}`)?.bindings.find(
      binding => binding.name === name
    );
    if (!binding) throw new Error(`Missing ${suffix} binding ${name}`);
    return binding.view;
  };
  const execute = (...suffixes: string[]) => {
    for (const suffix of suffixes) {
      const node = nodes.find(node => node.id === `${props.id}-${suffix}`);
      if (!node) throw new Error(`Missing node ${suffix}`);
      executeCommandNodesOnCPU([node], memory);
    }
  };
  return {graph, contributor, props, memory, view, execute};
}

function interpretGeographicBounds(coordinates: Float32Array) {
  const harness = createInterpretedSpacing(coordinates, true);
  const {memory, view, execute} = harness;
  execute('clear-state', 'prepare', 'keys');
  // Emulate only published sort/reduction primitives; transformations execute their generated WGSL.
  const state = new Uint32Array(64);
  const values = new Float32Array(state.buffer);
  state[0] = coordinates.length / 2;
  const sourceX = memory.readFloat32(view('prepare', 'sourceX'));
  const sourceY = memory.readFloat32(view('prepare', 'sourceY'));
  const canonicalX = memory.readFloat32(view('prepare', 'canonicalX'));
  values.set(
    [Math.min(...sourceX), Math.min(...sourceY), Math.max(...sourceX), Math.max(...sourceY)],
    8
  );
  values[38] = Math.min(...canonicalX);
  values[39] = Math.max(...canonicalX);
  const stateView = view('search-parameters', 'state');
  memory.write(stateView, state);
  memory.write(view('gaps', 'longitudes'), memory.readUint32(view('keys', 'longitudeKeys')).sort());
  execute('gaps');
  const gaps = memory.readFloat32(view('gaps', 'gaps'));
  values[28] = Math.max(...gaps);
  state[29] = gaps.indexOf(values[28]);
  memory.write(stateView, state);
  execute('covering-arc', 'search-parameters', 'full-search-grid-positions');
  memory.write(view('distinct-search-grid-positions', 'mask'), new Uint32Array(state[0]).fill(1));
  execute('distinct-search-grid-positions');
  return {...harness, extent: Array.from(memory.readFloat32(stateView).slice(8, 12))};
}

it('retains tiny negative longitude endpoints and includes both points in both geographic indices', () => {
  const coordinates = new Float32Array([-0.00001, 0, 0, 0]);
  const {extent, memory, view} = interpretGeographicBounds(coordinates);
  const oracle = computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: 'geographic'})!;
  expect(extent).toEqual(oracle.extent);
  expect(extent[0]).toBe(coordinates[0]);
  for (const search of ['full-search', 'distinct-search']) {
    const indexed = memory.readFloat32(view(`${search}-grid-positions`, 'gridPositions'));
    expect(
      Array.from(indexed).every(value => Number.isFinite(value) && value >= 0 && value <= 1)
    ).toBe(true);
  }
  expect(oracle.nearestNeighbor?.minimum).toBeCloseTo(1.111950774, 6);
});

it('keeps close negative longitudes distinct in the actual neighbour distance inputs', () => {
  const coordinates = new Float32Array([170, 0, -100, 0, -10, 0, -10.00001, 0]);
  const {memory, view} = interpretGeographicBounds(coordinates);
  for (const search of ['full-search', 'distinct-search']) {
    const distancePositions = memory.readFloat32(view(`${search}-grid-positions`, 'positions'));
    expect(distancePositions).toEqual(coordinates);
    const oracle = computeGPUAnalysisSpacingStatistics(distancePositions, {
      coordinateModel: 'geographic'
    })!;
    expect(oracle.nearestNeighbor?.minimum).toBeCloseTo(1.060438921, 6);
  }
});

it.each([
  1e20, -1e20, 3.4028232635611926e38, -3.4028232635611926e38
])('preserves the exact stored-f32 longitude remainder and duplicate keys for %s', longitude => {
  const storedLongitude = Math.fround(longitude);
  // Every f32 at this magnitude is an integer: BigInt modulo avoids the GPU's floating
  // quotient/product reduction. In particular, stored +1e20 is equivalent to -88 degrees.
  const remainder = Number(BigInt(storedLongitude) % 360n);
  const canonical =
    remainder > 180 ? remainder - 360 : remainder <= -180 ? remainder + 360 : remainder;
  if (longitude === 1e20) expect(canonical).toBe(-88);
  const coordinates = new Float32Array([storedLongitude, 0, canonical, 0]);
  const {graph, props, extent, memory, view, execute} = interpretGeographicBounds(coordinates);
  const oracle = computeGPUAnalysisSpacingStatistics(coordinates, {coordinateModel: 'geographic'})!;
  expect(oracle.duplicateLocationCount).toBe(1);
  expect(oracle.nearestNeighbor?.maximum).toBe(0);
  expect(extent).toEqual(oracle.extent);
  for (const suffix of ['prepare', 'keys']) {
    const record = getWGSLKernelNodeRecord(graph, `${props.id}-${suffix}`)!;
    // The interpreter evaluates floating remainder differently from GPU compilers. Pin the
    // shared integer reduction as well as executing it, so f32 '%' cannot regress unnoticed.
    expect(record.declarations).toContain(NEIGHBOR_SEARCH_CANONICAL_LONGITUDE_WGSL);
    expect(record.body).toContain('getNeighborSearchCanonicalLongitude(');
    expect(record.body).not.toContain('% 360.0');
  }
  const keys = memory.readUint32(view('keys', 'longitudeKeys'));
  expect(keys[0]).toBe(keys[1]);
  for (const search of ['full-search', 'distinct-search']) {
    expect(memory.readFloat32(view(`${search}-grid-positions`, 'positions'))).toEqual(
      new Float32Array([canonical, 0, canonical, 0])
    );
  }
  memory.write(view('distinct', 'order'), new Uint32Array([0, 1]));
  execute('distinct');
  expect(memory.readUint32(view('distinct', 'distinctFlags'))).toEqual(new Uint32Array([1, 0]));
});

it('publishes intermediate CSR exhaustion only through canonical candidateOverflow ports and refuses decoding', () => {
  const {contributor, props, memory, view, execute} = createInterpretedSpacing(
    new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]),
    false,
    2
  );
  execute('clear-state');
  memory.write(view('full-search-status', 'overflow'), new Uint32Array([1]));
  memory.write(view('full-search-status', 'required'), new Uint32Array([12]));
  execute('full-search-status', 'publish-summary');
  expect(memory.readUint32(contributor.output.status.overflow)[0]).toBe(0);
  expect(memory.readUint32(contributor.output.status.candidateOverflow)[0]).toBe(1);
  for (const port of Object.values(contributor.output.status)) {
    expect(port.buffer).toBe(props.output.summary.buffer);
    expect(port.format).toBe('uint32');
    expect(port.length).toBe(1);
  }
  const words = memory.readUint32(props.output.summary);
  words[7] = 1;
  expect(() => decodeGPUAnalysisSpacingStatistics(words)).toThrow(
    /candidateOverflow.*requiredCount 12/
  );
});

it('keeps the neighbor-search parameter view and clear endpoint at twelve words', () => {
  for (const geographic of [false, true]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    recordWGSLKernelNodes(graph);
    const props = createProps(graph, 4);
    new GPUAnalysisSpacing({
      ...props,
      spatialContext: geographic
        ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        : props.spatialContext
    }).getCommandNodes(graph);
    const record = getWGSLKernelNodeRecord(graph, `${props.id}-search-parameters`)!;
    expect(record.bindings.find(binding => binding.name === 'parameters')!.view.length).toBe(12);
    expect(record.body).toContain('word < 12u; word++');
    graph.device.destroy();
  }
});
