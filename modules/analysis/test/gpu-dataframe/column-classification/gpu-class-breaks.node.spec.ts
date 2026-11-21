// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  GPU_CLASS_BREAKS_METHOD_CODES,
  type GPUClassBreaksMethod
} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {
  GPUClassBreaks,
  type GPUClassBreaksProps
} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {validateClassificationOutput} from '../../../src/dataframe-analysis/column-classification/column-classification-rows';
import {
  makeGPURasterBand,
  makeGPURasterMetadata
} from '../../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {WgslReflect} from 'wgsl_reflect';
import {
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';
import {computeClassBreaksOracle, countClassesOracle} from './class-breaks-oracle';
let graphCount = 0;

/** Returns a fresh graph, so transient IDs never collide between cases. */
function createGraph(): GPUCommandGraph {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id: `graph-${graphCount++}`});
}

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUClassBreaksProps> = {},
  rows = 100,
  maximumClassCount = 8
): GPUClassBreaksProps {
  return {
    values: createTransientView(graph, 'values', 'float32', rows),
    mask: createTransientView(graph, 'mask', 'uint32', rows),
    parameters: createTransientView(
      graph,
      'parameters',
      'float32',
      getGPUClassBreaksParameterLength(maximumClassCount)
    ),
    maximumClassCount,
    output: {
      breaks: createTransientView(graph, 'breaks', 'float32', maximumClassCount + 1),
      classCount: createTransientView(graph, 'class-count', 'uint32', 1),
      classCounts: createTransientView(graph, 'class-counts', 'uint32', maximumClassCount)
    },
    ...overrides
  };
}

it('getGPUClassBreaksParameterValues packs the header and method data', () => {
  const maximumClassCount = 8;
  expect(getGPUClassBreaksParameterLength(maximumClassCount)).toBe(17);
  const quantile = getGPUClassBreaksParameterValues({method: 'quantile', classCount: 4}, 8);
  expect(Array.from(quantile.slice(0, 5))).toEqual([1, 4, 1, Math.fround(0.4), 1.5]);
  expect(Array.from(quantile.slice(8, 11))).toEqual([0.25, 0.5, 0.75]);
  expect(Number.isNaN(quantile[11])).toBe(true);
  const boxPlot = getGPUClassBreaksParameterValues({method: 'box-plot', boxPlotHinge: 3}, 8);
  expect(boxPlot[1]).toBe(6);
  expect(boxPlot[4]).toBe(3);
  expect(Array.from(boxPlot.slice(8, 11))).toEqual([0.25, 0.5, 0.75]);
  const custom = getGPUClassBreaksParameterValues({method: 'custom', customEdges: [0, 1, 5]}, 8);
  expect(custom[0]).toBe(GPU_CLASS_BREAKS_METHOD_CODES.custom);
  expect(custom[1]).toBe(2);
  expect(Array.from(custom.slice(8, 11))).toEqual([0, 1, 5]);
  const third = getGPUClassBreaksParameterValues({method: 'quantile', classCount: 3}, 8);
  expect(third[8]).toBe(Math.fround(1 / 3));
  expect(() => getGPUClassBreaksParameterValues({method: 'quantile', classCount: 9}, 8)).toThrow(
    /classCount must be an integer/
  );
  expect(() => getGPUClassBreaksParameterValues({method: 'custom', customEdges: [1]}, 8)).toThrow(
    /two edges/
  );
  // NaN settings would leave a NaN edge (the publish kernel then reports no classification).
  expect(() =>
    getGPUClassBreaksParameterValues({method: 'custom', customEdges: [0, NaN, 1]}, 8)
  ).toThrow(/NaN/);
  expect(() =>
    getGPUClassBreaksParameterValues({method: 'box-plot', boxPlotHinge: NaN}, 8)
  ).toThrow(/finite/);
  expect(() =>
    getGPUClassBreaksParameterValues({method: 'nope' as GPUClassBreaksMethod}, 8)
  ).toThrow(/unknown method/);
  expect(() =>
    getGPUClassBreaksParameterValues({method: 'quantile'}, 8, new Float32Array(4))
  ).toThrow(/must hold/);
});

it('GPUClassBreaks validates its views and options', () => {
  /** Builds the contributor on a fresh graph with overrides derived from that graph. */
  const build = (
    getOverrides: (graph: GPUCommandGraph) => Partial<GPUClassBreaksProps> = () => ({}),
    rows = 100,
    maximumClassCount = 8
  ) => {
    const graph = createGraph();
    return () =>
      new GPUClassBreaks(createProps(graph, getOverrides(graph), rows, maximumClassCount));
  };
  expect(build(() => ({maximumClassCount: 0}))).toThrow(/maximumClassCount/);
  expect(build(() => ({maximumClassCount: 65}))).toThrow(/maximumClassCount/);
  expect(build(() => ({methods: []}))).toThrow(/at least one/);
  expect(build(() => ({methods: ['bogus' as GPUClassBreaksMethod]}))).toThrow(/unknown method/);
  expect(build(() => ({methods: ['box-plot']}), 100, 5)).toThrow(/box-plot/);
  expect(build(() => ({methods: ['quantile']}), 100, 5)).not.toThrow();
  expect(build(() => ({naturalBreaksBinCount: 1}))).toThrow(/naturalBreaksBinCount/);
  expect(build(() => ({naturalBreaksBinCount: 4096}))).toThrow(/naturalBreaksBinCount/);
  expect(build(graph => ({mask: createTransientView(graph, 'short-mask', 'uint32', 99)}))).toThrow(
    /mask length/
  );
  expect(
    build(
      graph =>
        ({
          methods: ['maximum-breaks'],
          values: createVectorView('vector', 'float32', [
            createTransientView(graph, 'chunk', 'float32', 100)
          ])
        }) as unknown as Partial<GPUClassBreaksProps>
    )
  ).toThrow(/one packed view/);
  expect(build(graph => ({parameters: createTransientView(graph, 'short', 'float32', 4)}))).toThrow(
    /parameters must hold/
  );
  const getOutput = (graph: GPUCommandGraph) => ({
    breaks: createTransientView(graph, 'other-breaks', 'float32', 9),
    classCount: createTransientView(graph, 'other-class-count', 'uint32', 1),
    classCounts: createTransientView(graph, 'other-class-counts', 'uint32', 8)
  });
  expect(
    build(graph => ({
      output: {...getOutput(graph), breaks: createTransientView(graph, 'few', 'float32', 4)}
    }))
  ).toThrow(/output.breaks/);
  expect(
    build(graph => ({
      output: {
        ...getOutput(graph),
        classCounts: createTransientView(graph, 'few-counts', 'uint32', 2)
      }
    }))
  ).toThrow(/output.classCounts/);
  const graph = createGraph();
  const props = createProps(graph);
  expect(
    () =>
      new GPUClassBreaks({
        ...props,
        output: {...props.output, classCount: props.mask as GraphDataView<'uint32'>}
      })
  ).toThrow(/share buffers/);
});

it('GPUClassBreaks returns deterministic node IDs and rejects foreign views', () => {
  const getNodeIds = () => {
    const graph = createGraph();
    return new GPUClassBreaks({...createProps(graph), id: 'cb'})
      .getCommandNodes(graph)
      .map(node => node.id);
  };
  const first = getNodeIds();
  const second = getNodeIds();
  expect(second).toEqual(first);
  expect(new Set(first).size).toBe(first.length);
  expect(first[0]).toBe('cb-prepare');
  expect(first[1]).toBe('cb-extremes');
  expect(first).toContain('cb-head-tail-0-split');
  expect(first).toContain('cb-natural-breaks-backtrack');
  expect(first).toContain('cb-maximum-breaks-midpoints');
  expect(first.slice(-4)).toEqual([
    'cb-finish',
    'cb-publish',
    'cb-class-counts-clear',
    'cb-class-counts'
  ]);
  expect(first.every(id => id.startsWith('cb-'))).toBe(true);
  // Leaving out the sort-based and iterative methods drops their nodes.
  const leanGraph = createGraph();
  const lean = new GPUClassBreaks({
    ...createProps(leanGraph),
    id: 'lean',
    methods: ['equal-interval']
  });
  expect(lean.getCommandNodes(leanGraph).map(node => node.id)).toEqual([
    'lean-prepare',
    'lean-extremes',
    'lean-finish',
    'lean-publish',
    'lean-class-counts-clear',
    'lean-class-counts'
  ]);
  const contributor = new GPUClassBreaks(createProps(createGraph()));
  expect(() => contributor.getCommandNodes(createGraph())).toThrow(/target graph/);
});

it('class breaks oracle reproduces textbook breaks', () => {
  const values = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, NaN, Infinity]);
  const base = {values, maximumClassCount: 8, naturalBreaksBinCount: 10};
  const equal = computeClassBreaksOracle({
    ...base,
    parameters: {method: 'equal-interval', classCount: 3}
  });
  expect(equal.classCount).toBe(3);
  expect(Array.from(equal.breaks.slice(0, 4))).toEqual([1, 4, 7, 10]);
  const quantile = computeClassBreaksOracle({
    ...base,
    parameters: {method: 'quantile', classCount: 2}
  });
  // B-14: Infinity is missing for quantiles too (numpy.quantile of the finite values 1..10 is 5.5).
  expect(Array.from(quantile.breaks.slice(0, 3))).toEqual([1, 5.5, 10]);
  const gaps = new Float32Array([1, 2, 3, 10, 11, 30]);
  const maximum = computeClassBreaksOracle({
    ...base,
    values: gaps,
    parameters: {method: 'maximum-breaks', classCount: 3}
  });
  expect(Array.from(maximum.breaks.slice(0, 4))).toEqual([1, 6.5, 20.5, 30]);
  const natural = computeClassBreaksOracle({
    ...base,
    values: new Float32Array([0, 0.5, 1, 9, 9.5, 10]),
    parameters: {method: 'natural-breaks', classCount: 2}
  });
  // Every start inside the empty gap ties; the lowest start wins.
  expect(natural.naturalStartBins).toEqual([2]);
  const headTail = computeClassBreaksOracle({
    ...base,
    values: new Float32Array([1, 1, 1, 1, 1, 1, 1, 2, 4, 100]),
    parameters: {method: 'head-tail', classCount: 8, headTailRatio: 0.4}
  });
  // Mean 11.3 splits off [100]; a single-row head stops.
  expect(headTail.classCount).toBe(2);
  expect(headTail.breaks[1]).toBeCloseTo(11.3, 5);
  expect(Array.from(countClassesOracle(values, undefined, equal.breaks, 3, 3))).toEqual([3, 3, 5]);
});

it('GPUClassBreaks fits every kernel in 8 storage buffers with all methods compiled', () => {
  // The default WebGPU limit; the null device reports it, so an extra binding throws here.
  const graph = createGraph();
  const contributor = new GPUClassBreaks(createProps(graph));
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
});

it('GPUClassBreaks pipelines do not grow with maximumClassCount (rule 12)', () => {
  // Before: natural-breaks layers and head/tail rounds baked LAYER / round into WGSL, 62
  // pipelines at 8 classes and 65 at 9. Now the layer and round are GPU state words.
  const build = (maximumClassCount: number) => {
    const graph = createGraph();
    return new GPUClassBreaks(createProps(graph, {}, 100, maximumClassCount)).getCommandNodes(
      graph
    );
  };
  const counts = [8, 9, 64].map(
    classCount => getGPUCommandNodePipelineKeys(build(classCount)).size
  );
  expect(counts[1]).toBe(counts[0]);
  expect(counts[2]).toBe(counts[0]);
  for (const {id, source} of getGPUCommandNodeKernelSources(build(9))) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(source, id).not.toMatch(/const LAYER\b|override LAYER\b/);
  }
});

it('GPUClassBreaks gates natural-breaks layers and head/tail rounds indirectly (rule 13)', () => {
  const graph = createGraph();
  const nodes = new GPUClassBreaks({...createProps(graph), id: 'gate'}).getCommandNodes(graph);
  const byId = new Map(nodes.map(node => [node.id, node]));
  const gated = (id: string) => {
    const condition = (byId.get(id) as {condition?: {source?: string; mode?: string}}).condition;
    return condition?.source === 'gpu' && condition.mode === 'indirect';
  };
  for (let layer = 1; layer < 8; layer++) {
    expect(gated(`gate-natural-breaks-layer-${layer}`), `layer ${layer}`).toBe(true);
    // The advance kernel writes the layer dispatch, so it must not be gated by it.
    expect(gated(`gate-natural-breaks-layer-${layer}-advance`)).toBe(false);
  }
  for (let round = 0; round < 7; round++) {
    expect(gated(`gate-head-tail-${round}-tile`), `round ${round}`).toBe(true);
    expect(gated(`gate-head-tail-${round}-merge`), `round ${round}`).toBe(true);
    expect(gated(`gate-head-tail-${round}-split`)).toBe(false);
  }
  // Standard deviation is not a loop and stays method-gated inside the kernel.
  expect(gated('gate-moments-tile')).toBe(false);
});

it('GPUClassBreaks natural-breaks rounds and the second coarse stage share their pipelines (rule 12)', () => {
  const graph = createGraph();
  const nodes = new GPUClassBreaks({
    ...createProps(graph, {methods: ['natural-breaks']}, 100, 12),
    id: 'nb'
  }).getCommandNodes(graph);
  // Eight rounds of two half-rounds share advance, row-arm, plan, clear, accumulate, and solve.
  const halfRoundPipelines = getGPUCommandNodePipelineKeys(nodes, {
    idPattern: /^nb-natural-breaks-refine-\d+-/
  });
  expect(halfRoundPipelines.size).toBe(6);
  const halfRounds = new Set(
    getGPUCommandNodeKernelSources(nodes)
      .map(({id}) => /refine-(\d+)-/.exec(id)?.[1])
      .filter(Boolean)
  );
  expect(halfRounds.size).toBe(16);
  // The coarse stage over the most populated class runs the same shaders as the first.
  const first = getGPUCommandNodePipelineKeys(nodes, {
    idPattern: /^nb-natural-breaks-(clear|histogram|prefix|layer-\d+(-advance)?)$/
  });
  const both = getGPUCommandNodePipelineKeys(nodes, {
    idPattern: /^nb-natural-breaks(-bulk)?-(clear|histogram|prefix|layer-\d+(-advance)?)$/
  });
  expect(first.size).toBeGreaterThan(0);
  expect(both.size).toBe(first.size);
  // The extents pass over the merged partition and the snap kernel repeat their shaders too.
  expect(
    getGPUCommandNodePipelineKeys(nodes, {idPattern: /^nb-natural-breaks-(extents|snap)(-bulk)?$/})
      .size
  ).toBe(2);
});

it('GPUClassBreaks gates natural-breaks refinement indirectly and keeps the writers ungated (rule 13)', () => {
  const graph = createGraph();
  const nodes = new GPUClassBreaks({
    ...createProps(graph, {methods: ['natural-breaks']}),
    id: 'g'
  }).getCommandNodes(graph);
  const byId = new Map(nodes.map(node => [node.id, node]));
  const gated = (id: string) => {
    const condition = (byId.get(id) as {condition?: {source?: string; mode?: string}}).condition;
    return condition?.source === 'gpu' && condition.mode === 'indirect';
  };
  for (let half = 0; half < 16; half++) {
    const prefix = `g-natural-breaks-refine-${half}`;
    for (const name of ['plan', 'clear', 'accumulate', 'solve']) {
      expect(gated(`${prefix}-${name}`), `${prefix}-${name}`).toBe(true);
    }
    // The advance kernel writes every triple, so the buffer cannot also gate it.
    expect(gated(`${prefix}-advance`), `${prefix}-advance`).toBe(false);
  }
  expect(gated('g-natural-breaks-extents')).toBe(true);
  expect(gated('g-natural-breaks-extents-bulk')).toBe(true);
  for (const writer of ['backtrack', 'bulk-switch', 'bulk-assemble', 'snap', 'finalize']) {
    expect(gated(`g-natural-breaks-${writer}`), writer).toBe(false);
  }
});

it('GPUClassBreaks natural-breaks kernels fit core limits at 64 classes and 2048 bins', () => {
  const graph = createGraph();
  const nodes = new GPUClassBreaks({
    ...createProps(graph, {methods: ['natural-breaks'], naturalBreaksBinCount: 2048}, 1 << 24, 64),
    id: 'l'
  }).getCommandNodes(graph);
  for (const {id, source, constants} of getGPUCommandNodeKernelSources(nodes)) {
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source, constants), id).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  // The divide and conquer layer holds the optima of all bins plus the lane reduction arrays.
  const layer = getGPUCommandNodeKernelSources(nodes).find(
    ({id}) => id === 'l-natural-breaks-layer-1'
  )!;
  expect(getWorkgroupMemoryBytes(layer.source, layer.constants)).toBe(
    2048 * 4 + 256 * 4 + 256 * 4 + 16
  );
});

it('GPUClassBreaks natural breaks build for one class, two bins and a handful of rows', () => {
  for (const [maximumClassCount, naturalBreaksBinCount, rows] of [
    [1, 2, 3],
    [2, 2, 1],
    [2, 2048, 1],
    // Approximate mode needs at least maximumClassCount bins (DPC-120).
    [64, 64, 5]
  ] as const) {
    const graph = createGraph();
    const nodes = new GPUClassBreaks({
      ...createProps(
        graph,
        {methods: ['natural-breaks'], naturalBreaksBinCount},
        rows,
        maximumClassCount
      )
    }).getCommandNodes(graph);
    for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
      expect(() => new WgslReflect(source), id).not.toThrow();
    }
  }
});

it('GPUClassBreaks exact mode enforces its source capacity without an approximate fallback', () => {
  const graph = createGraph();
  expect(
    () =>
      new GPUClassBreaks(
        createProps(graph, {naturalBreaksMode: 'exact', methods: ['natural-breaks']}, 8193)
      )
  ).toThrow();
  const contributor = new GPUClassBreaks(
    createProps(createGraph(), {naturalBreaksMode: 'exact', methods: ['natural-breaks']}, 1024)
  );
  expect(contributor.props.naturalBreaksMode).toBe('exact');
});

it('GPUClassBreaks exact kernels parse within default storage limits', () => {
  const graph = createGraph();
  const nodes = new GPUClassBreaks(
    createProps(graph, {naturalBreaksMode: 'exact', methods: ['natural-breaks']}, 1024, 64)
  ).getCommandNodes(graph);
  for (const {id, source, constants} of getGPUCommandNodeKernelSources(nodes)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    expect(getWorkgroupMemoryBytes(source, constants), id).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
});

it('GPUClassBreaks exact layers reuse pipelines and dispatch inactive layers indirectly', () => {
  const graph = createGraph();
  const nodes = new GPUClassBreaks({
    ...createProps(graph, {naturalBreaksMode: 'exact', methods: ['natural-breaks']}, 1024, 64),
    id: 'exact'
  }).getCommandNodes(graph);
  const pipelines = getGPUCommandNodePipelineKeys(nodes, {
    idPattern: /exact-natural-breaks-layer-/
  });
  expect(pipelines.size).toBe(1);
  for (const node of nodes.filter(node => /exact-natural-breaks-layer-/.test(node.id))) {
    const condition = (node as {condition?: {source?: string; mode?: string}}).condition;
    expect(condition?.source).toBe('gpu');
    expect(condition?.mode).toBe('indirect');
  }
});

it('GPUClassBreaks explicit natural-breaks storage and search modes validate their capacities', () => {
  const build = (overrides: Partial<GPUClassBreaksProps>, rows: number) =>
    new GPUClassBreaks(
      createProps(createGraph(), {methods: ['natural-breaks'], ...overrides}, rows)
    );
  expect(() =>
    build({naturalBreaksMode: 'exact', naturalBreaksExactStorage: 'materialized'}, 1025)
  ).toThrow();
  expect(() =>
    build({naturalBreaksMode: 'exact', naturalBreaksExactStorage: 'recompute'}, 8192)
  ).not.toThrow();
  expect(() => build({naturalBreaksMode: 'monotone'}, 262144)).not.toThrow();
  expect(() => build({naturalBreaksMode: 'monotone'}, 262145)).toThrow();
  expect(() =>
    build({naturalBreaksMode: 'monotone', naturalBreaksExactStorage: 'recompute'}, 16)
  ).toThrow();
  expect(() => build({naturalBreaksMode: 'unknown' as 'exact'}, 16)).toThrow();
});

it('GPUClassBreaks recomputation removes the quadratic binding and preserves pipeline reuse', () => {
  const graph = createGraph();
  const nodes = new GPUClassBreaks(
    createProps(
      graph,
      {
        naturalBreaksMode: 'exact',
        naturalBreaksExactStorage: 'recompute',
        methods: ['natural-breaks']
      },
      8192,
      64
    )
  ).getCommandNodes(graph);
  expect(nodes.some(node => node.id.endsWith('-costs'))).toBe(false);
  expect(
    getGPUCommandNodePipelineKeys(nodes, {idPattern: /exact-natural-breaks-layer-/}).size
  ).toBe(1);
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
});

it('GPUClassBreaks monotone search spans workgroups and reuses all level pipelines', () => {
  for (const rows of [1, 257, 262144]) {
    const graph = createGraph();
    const nodes = new GPUClassBreaks(
      createProps(
        graph,
        {
          naturalBreaksMode: 'monotone',
          methods: ['natural-breaks']
        },
        rows,
        4
      )
    ).getCommandNodes(graph);
    expect(
      getGPUCommandNodePipelineKeys(nodes, {idPattern: /monotone-natural-breaks-search-/}).size
    ).toBe(1);
    for (const {id, source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(() => new WgslReflect(source), id).not.toThrow();
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
      expect(getWorkgroupMemoryBytes(source, constants), id).toBeLessThanOrEqual(
        CORE_WORKGROUP_STORAGE_LIMIT_BYTES
      );
    }
    for (const node of nodes.filter(node => /monotone-natural-breaks-search-/.test(node.id))) {
      const condition = (node as {condition?: {mode?: string}}).condition;
      expect(condition?.mode).toBe('indirect');
    }
  }
});

it('GPUClassBreaks validates row-proportional scratch against the actual device binding limit', () => {
  const graph = createGraph();
  Object.defineProperty(graph.device.limits, 'maxStorageBufferBindingSize', {value: 1048576});
  const contributor = new GPUClassBreaks(
    createProps(
      graph,
      {
        naturalBreaksMode: 'monotone',
        methods: ['natural-breaks']
      },
      65536,
      8
    )
  );
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /sorted moments\/prefix needs.*bytes in one binding/
  );
});

// DPC-120. Approximate natural breaks clamp the per-frame class count to the bin count, so
// maximumClassCount 8 over 4 bins could never return more than 4 classes. The contributor now
// refuses that combination at construction; exact and monotone modes do not bin and keep it.
it('GPUClassBreaks rejects more approximate natural-breaks classes than bins', () => {
  const build = (overrides: Partial<GPUClassBreaksProps>) => {
    const graph = createGraph();
    return () => new GPUClassBreaks(createProps(graph, {id: 'breaks-check', ...overrides}));
  };
  expect(build({methods: ['natural-breaks'], naturalBreaksBinCount: 4})).toThrow(
    /^breaks-check maximumClassCount 8 must not exceed naturalBreaksBinCount 4/
  );
  expect(
    build({methods: ['natural-breaks'], naturalBreaksBinCount: 4, naturalBreaksMode: 'approximate'})
  ).toThrow(/naturalBreaksBinCount 4/);
  expect(build({naturalBreaksBinCount: 8})).not.toThrow();
  expect(build({naturalBreaksBinCount: 4, methods: ['quantile']})).not.toThrow();
  expect(build({naturalBreaksBinCount: 4, naturalBreaksMode: 'exact'})).not.toThrow();
  expect(build({naturalBreaksBinCount: 4, naturalBreaksMode: 'monotone'})).not.toThrow();
});

it('WGS-05: ranks large gaps with an extended exponent and repairs midpoints in parallel', () => {
  const graph = createGraph();
  const kernels = getGPUCommandNodeKernelSources(
    new GPUClassBreaks(createProps(graph, {methods: ['maximum-breaks']}, 4, 2)).getCommandNodes(
      graph
    )
  );
  const gaps = kernels.find(({id}) => id.endsWith('maximum-breaks-gaps'))!;
  expect(gaps.source).toContain('0x00800000u');
  const midpoints = kernels.find(({id}) => id.endsWith('maximum-breaks-midpoints'));
  expect(midpoints).toBeDefined();
  expect(midpoints!.source).toContain('(lower + upper) * 0.5');
  for (const {source} of kernels) expect(() => new WgslReflect(source)).not.toThrow();
});

it('DEF-07: omitted methods include compatible routes at two classes and on chunked input', () => {
  const graph = createGraph();
  const props = createProps(graph, {}, 3, 2);
  const contributor = new GPUClassBreaks(props);
  expect(contributor.methods).toContain('equal-interval');
  expect(contributor.methods).not.toContain('box-plot');
  const chunked = createVectorView('default-chunks', 'float32', [
    props.values as GraphDataView<'float32'>
  ]);
  const chunkContributor = new GPUClassBreaks({...props, values: chunked, mask: undefined});
  expect(chunkContributor.methods).not.toContain('maximum-breaks');
  expect(chunkContributor.methods).toContain('natural-breaks');
  expect(
    () =>
      new GPUClassBreaks({...props, values: chunked, mask: undefined, methods: ['maximum-breaks']})
  ).toThrow(/packed/);
  expect(() => new GPUClassBreaks({...props, methods: ['box-plot']})).toThrow(/box-plot/);
});

it.each([
  {
    rows: 1025,
    naturalBreaksMode: 'exact' as const,
    naturalBreaksExactStorage: 'materialized' as const
  },
  {rows: 8193, naturalBreaksMode: 'exact' as const},
  {rows: 262145, naturalBreaksMode: 'monotone' as const}
])('SCL-05: inactive natural settings admit $rows equal-interval rows', settings => {
  const graph = createGraph();
  const {rows, ...options} = settings;
  const contributor = new GPUClassBreaks(
    createProps(graph, {methods: ['equal-interval'], ...options}, rows, 2)
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(6);
  expect(contributor.estimatedCommandNodeCount).toBe(48);
});

it('DEF-06: exposes GPU natural-breaks mode and algorithm provenance', () => {
  for (const naturalBreaksMode of [undefined, 'exact'] as const) {
    const graph = createGraph();
    const contributor = new GPUClassBreaks(
      createProps(graph, {methods: ['natural-breaks'], naturalBreaksMode}, 12, 4)
    );
    expect(contributor.provenance.naturalBreaksMode).toBe(naturalBreaksMode ?? 'approximate');
    expect(contributor.provenance.naturalBreaksAlgorithm).toBe(
      naturalBreaksMode === 'exact'
        ? 'exact-fisher-jenks-normalized-float32'
        : 'histogram-refinement'
    );
  }
});

it('SCL-16: describes conditional routes and plans admission without graph mutation', () => {
  expect(GPUClassBreaks).toHaveProperty('limits');
  expect(GPUClassBreaks).toHaveProperty('getExecutionPlan');
  const device = createNullWebGPUDevice();
  try {
    const equal = GPUClassBreaks.getExecutionPlan(
      {
        rowCount: 8193,
        maximumClassCount: 2,
        methods: ['equal-interval'],
        naturalBreaksMode: 'exact'
      },
      device.limits
    );
    expect(equal.feasible).toBe(true);
    const exact = GPUClassBreaks.getExecutionPlan(
      {
        rowCount: 8193,
        maximumClassCount: 2,
        methods: ['natural-breaks'],
        naturalBreaksMode: 'exact'
      },
      device.limits
    );
    expect(exact.feasible).toBe(false);
    expect(exact.constraints.find(constraint => constraint.name === 'exact-rows')).toMatchObject({
      maximum: 8192,
      received: 8193,
      satisfied: false
    });
    const binding = GPUClassBreaks.getExecutionPlan(
      {rowCount: 100, maximumClassCount: 2, methods: ['equal-interval']},
      {...device.limits, maxStorageBufferBindingSize: 128}
    );
    expect(binding.feasible).toBe(false);
    for (const chunks of [659, 660]) {
      const plan = GPUClassBreaks.getExecutionPlan(
        {
          rowCount: chunks,
          chunkRowCounts: Array(chunks).fill(1),
          maximumClassCount: 18,
          methods: ['natural-breaks']
        },
        device.limits
      );
      expect(plan.estimatedCommandNodeCount).toBe(chunks === 659 ? 32725 : 32771);
      expect(plan.feasible).toBe(chunks === 659);
    }
  } finally {
    device.destroy();
  }
});

it('ENT-02: classification accepts foreign vector shapes and preserves integer chunks', () => {
  const graph = createGraph();
  const chunks = [
    createTransientView(graph, 'foreign-first', 'uint32', 1),
    createTransientView(graph, 'foreign-second', 'uint32', 2)
  ];
  const vector = createVectorView('foreign', 'uint32', chunks);
  // Same published shape, without this module condition's GraphVectorView prototype.
  const foreign = {...vector} as typeof vector;
  const props = createProps(
    graph,
    {values: foreign, mask: undefined, methods: ['equal-interval']},
    3,
    2
  );
  const kernels = getGPUCommandNodeKernelSources(new GPUClassBreaks(props).getCommandNodes(graph));
  expect(kernels.filter(({id}) => /float-values-[01]$/.test(id))).toHaveLength(2);
  const output = createTransientView(graph, 'foreign-output', 'uint32', 3);
  expect(() =>
    validateClassificationOutput('foreign', 'output', output, 'uint32', foreign)
  ).toThrow(/topology|chunk/);
});

it('GPUClassBreaks captures construction options and nested output ports', () => {
  const graph = createGraph();
  const methods: GPUClassBreaksMethod[] = ['equal-interval'];
  const props = createProps(graph, {methods}, 3, 2);
  const contributor = new GPUClassBreaks(props);
  methods.push('quantile');
  props.maximumClassCount = 6;
  props.output.breaks = createTransientView(graph, 'replacement', 'float32', 7);
  expect(contributor.props.maximumClassCount).toBe(2);
  expect(contributor.props.methods).toEqual(['equal-interval']);
  expect(contributor.props.output.breaks).not.toBe(props.output.breaks);
  expect(contributor.props.values).toBe(props.values);
});

it.each([
  {
    rowCount: 1024,
    expectedFeasible: true,
    naturalBreaksMode: 'exact' as const,
    naturalBreaksExactStorage: 'materialized' as const
  },
  {
    rowCount: 1025,
    expectedFeasible: false,
    naturalBreaksMode: 'exact' as const,
    naturalBreaksExactStorage: 'materialized' as const
  },
  {
    rowCount: 8192,
    expectedFeasible: true,
    naturalBreaksMode: 'exact' as const,
    naturalBreaksExactStorage: 'recompute' as const
  },
  {
    rowCount: 8193,
    expectedFeasible: false,
    naturalBreaksMode: 'exact' as const,
    naturalBreaksExactStorage: 'recompute' as const
  },
  {rowCount: 262144, expectedFeasible: true, naturalBreaksMode: 'monotone' as const},
  {rowCount: 262145, expectedFeasible: false, naturalBreaksMode: 'monotone' as const}
])('SCL-16: planner and constructor agree at the $naturalBreaksMode boundary $rowCount', shape => {
  const graph = createGraph();
  const options = {...shape, maximumClassCount: 2, methods: ['natural-breaks'] as const};
  const plan = GPUClassBreaks.getExecutionPlan(options, graph.device.limits);
  const construct = () => new GPUClassBreaks(createProps(graph, options, shape.rowCount, 2));
  expect(plan.feasible).toBe(shape.expectedFeasible);
  if (shape.expectedFeasible) expect(construct).not.toThrow();
  else expect(construct).toThrow(/rows/);
});

it('CMP-05: a raster band supplies its grid-sized prefix without copying the allocation', () => {
  const graph = createGraph();
  const allocation = createTransientView(graph, 'density-allocation', 'float32', 5);
  const band = makeGPURasterBand(allocation, makeGPURasterMetadata({width: 2, height: 2}));
  expect(band.storage.values.length).toBe(4);
  expect(band.storage.values.buffer).toBe(allocation.buffer);
  expect(band.storage.values.byteOffset).toBe(allocation.byteOffset);
  const contributor = new GPUClassBreaks(
    createProps(
      graph,
      {values: band.storage.values, mask: undefined, methods: ['equal-interval']},
      4,
      2
    )
  );
  expect(contributor.props.values.length).toBe(4);
  expect(allocation.length).toBe(5);
});

it('ENT-02: foreign vectors must retain a single scalar format and their declared row total', () => {
  const graph = createGraph();
  const chunks = [
    createTransientView(graph, 'format-first', 'uint32', 1),
    createTransientView(graph, 'format-second', 'uint32', 2)
  ];
  const vector = createVectorView('format-vector', 'uint32', chunks);
  const mixed = {
    ...vector,
    data: [{...chunks[0], format: 'float32'}, chunks[1]]
  } as unknown as typeof vector;
  const props = createProps(graph, {mask: undefined, methods: ['equal-interval']}, 3, 2);
  const create = (values: typeof vector) => new GPUClassBreaks({...props, values});
  expect(() => create(mixed)).toThrow(/chunk format float32 must equal declared format uint32/);
  expect(() => create({...vector, length: 4})).toThrow(/chunk row total.*declared length 4/);
});

it('R1 B3: one-class defaults copy only probabilities within the public packed parameters', () => {
  const graph = createGraph();
  const parameters = getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 1}, 1);
  expect(parameters.length).toBe(10);
  const contributor = new GPUClassBreaks(createProps(graph, {}, 3, 1));
  expect(contributor.quantileCount).toBe(1);
  expect(8 + contributor.quantileCount).toBeLessThanOrEqual(parameters.length);
  const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  expect(kernels.find(({id}) => id.endsWith('-prepare'))!.source).toContain(
    'QUANTILE_COUNT: u32 = 1u'
  );
});

it('R1 B4: partial binding inventories cannot erase required input bindings', () => {
  const graph = createGraph();
  const shape = {rowCount: 35, maximumClassCount: 2, methods: ['equal-interval'] as const};
  const limits = {...graph.device.limits, maxStorageBufferBindingSize: 128};
  expect(GPUClassBreaks.getExecutionPlan(shape, limits).feasible).toBe(false);
  const partial = GPUClassBreaks.getExecutionPlan(
    {...shape, bindings: [{name: 'parameters', byteLength: 44, bufferByteLength: 44}]},
    limits
  );
  expect(partial.feasible).toBe(false);
  expect(partial.constraints.find(constraint => constraint.name === 'values.0')).toMatchObject({
    received: 140,
    satisfied: false
  });
});

it('R1 B5: a twelve-byte binding may use a thirteen-byte backing allocation', () => {
  const graph = createGraph();
  const buffer = graph.createTransientBuffer({
    id: 'odd-capacity',
    byteLength: 13,
    usage: Buffer.STORAGE
  });
  const values = graph.createDataView(buffer, {format: 'float32', length: 3});
  const shape = {
    rowCount: 3,
    maximumClassCount: 2,
    methods: ['equal-interval'] as const,
    bindings: [{name: 'values.0', byteLength: 12, bufferByteLength: 13}]
  };
  expect(GPUClassBreaks.getExecutionPlan(shape, graph.device.limits).feasible).toBe(true);
  const contributor = new GPUClassBreaks(
    createProps(graph, {values, mask: undefined, methods: ['equal-interval']}, 3, 2)
  );
  expect(() => contributor.getCommandNodes(graph)).not.toThrow();
});

it('R1 B7: aligned mask binding diagnostics precede derived quantile scratch failures', () => {
  const graph = createGraph();
  Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 1024});
  const maskBuffer = graph.createTransientBuffer({
    id: 'offset-mask',
    byteLength: 1028,
    usage: Buffer.STORAGE
  });
  const mask = graph.createDataView(maskBuffer, {format: 'uint32', length: 256, byteOffset: 4});
  const contributor = new GPUClassBreaks(createProps(graph, {mask, methods: ['quantile']}, 256, 6));
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /mask chunk 0 needs 1028 bytes in one binding \(limit 1024\)/
  );
});

it('R1 S3: opposite-sign midpoint sums before halving and no legacy overflow kernel executes', () => {
  const graph = createGraph();
  const kernels = getGPUCommandNodeKernelSources(
    new GPUClassBreaks(createProps(graph, {methods: ['maximum-breaks']}, 2, 2)).getCommandNodes(
      graph
    )
  );
  const midpoint = kernels.find(({id}) => id.endsWith('-midpoints'))!;
  // In units u=2^-149, (-3u+u)/2 = -u exactly; separate halving rounds -1.5u to -2u and u/2 to 0.
  expect(midpoint.source).toContain('midpoint = (lower + upper) * 0.5');
  const unit = 2 ** -149;
  expect(Math.fround(Math.fround(-3 * unit + unit) * 0.5)).toBe(-unit);
  expect(Math.fround(Math.fround(-3 * unit * 0.5) + Math.fround(unit * 0.5))).toBe(-2 * unit);
  expect(kernels.some(({id}) => id.endsWith('maximum-breaks-finish'))).toBe(false);
});

it.each([
  {rows: 256, feasible: true},
  {rows: 257, feasible: false}
])('R1 S4: dispatch dimension admission at $rows rows', ({rows, feasible}) => {
  const graph = createGraph();
  Object.assign(graph.device.limits, {maxComputeWorkgroupsPerDimension: 1});
  const shape = {rowCount: rows, maximumClassCount: 2, methods: ['equal-interval'] as const};
  expect(GPUClassBreaks.getExecutionPlan(shape, graph.device.limits).feasible).toBe(feasible);
  const contributor = new GPUClassBreaks(
    createProps(graph, {methods: ['equal-interval']}, rows, 2)
  );
  if (feasible) expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  else expect(() => contributor.getCommandNodes(graph)).toThrow(/dispatch|workgroup/);
});

it('R1 B7: retains the chunk-proportional refinement dispatch binding diagnostic', () => {
  const graph = createGraph();
  Object.assign(graph.device.limits, {maxStorageBufferBindingSize: 1024});
  const chunk = createTransientView(graph, 'repeated-row', 'float32', 1);
  const values = createVectorView(
    'forty-chunks',
    'float32',
    Array.from({length: 40}, () => chunk)
  );
  const contributor = new GPUClassBreaks(
    createProps(
      graph,
      {values, mask: undefined, methods: ['natural-breaks'], naturalBreaksBinCount: 16},
      40,
      6
    )
  );
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /natural-breaks refinement dispatch scratch needs 1488 bytes in one binding/
  );
});

it.each([
  'approximate',
  'exact',
  'monotone'
] as const)('R1 B7: retains bytes-in-one-binding for %s scratch', naturalBreaksMode => {
  const graph = createGraph();
  Object.assign(graph.device.limits, {
    maxStorageBufferBindingSize: naturalBreaksMode === 'approximate' ? 65536 : 1024
  });
  const contributor = new GPUClassBreaks(
    createProps(
      graph,
      {methods: ['natural-breaks'], naturalBreaksMode, naturalBreaksBinCount: 16},
      100,
      6
    )
  );
  expect(() => contributor.getCommandNodes(graph)).toThrow(/bytes in one binding/);
});
