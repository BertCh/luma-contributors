// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphBufferHandle,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createBoundaryReference} from './w2-gwr-review-reference';
import {createGridIndexNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-grid-index';
import {
  findReservedIdentifiers,
  findConstantNonFiniteBitcasts,
  stripWGSLComments,
  WGSL_RESERVED_WORDS
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

let serial = 0;

/** Numeric-token audit independent of wgsl_reflect's permissive constant evaluator. */
function getInvalidNumericLiterals(source: string): string[] {
  const code = stripWGSLComments(source);
  const invalid: string[] = [];
  const literals =
    /(?<![\w.])(?:0[xX][0-9a-fA-F]+[iu]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fhiu]?)(?![\w.])/g;
  for (const match of code.matchAll(literals)) {
    const literal = match[0];
    const hexadecimal = /^0x/i.test(literal);
    const suffix = /[iu]$/.test(literal) ? literal.slice(-1) : '';
    const integer = hexadecimal || !/[.eEfh]/.test(literal);
    if (integer) {
      const value = BigInt(suffix ? literal.slice(0, -1) : literal);
      // A leading minus is an operator, not part of a numeric token. Typed literals must
      // themselves fit their type before any unary operation is applied.
      const maximum =
        suffix === 'u' ? 0xffffffffn : suffix === 'i' ? 0x7fffffffn : 0x7fffffffffffffffn;
      if (value > maximum) invalid.push(literal);
    } else if (!Number.isFinite(Math.fround(Number(literal.replace(/[fh]$/, ''))))) {
      invalid.push(literal);
    }
  }
  return invalid;
}

function createFixture(rowCount: number, maximumPermutations = 64) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `input-${serial++}`, format, length);
  const props = {
    positions: view('float32x2', rowCount),
    predictors: view('float32', rowCount),
    predictorCount: 1,
    response: view('float32', rowCount),
    parameters: view('float32', 6),
    maximumBandwidthCount: 2,
    refinement: false,
    indexGridSize: [64, 64] as const,
    output: {
      coefficients: view('float32', rowCount * 2),
      selectedBandwidth: view('float32', 2),
      localStatus: view('uint32', rowCount),
      fitted: view('float32', rowCount),
      standardErrors: view('float32', rowCount * 2),
      tValues: view('float32', rowCount * 2),
      significant: view('uint32', rowCount * 2),
      status: {approximation: view('uint32', 1)}
    }
  };
  const testProps = {
    positions: props.positions,
    predictors: props.predictors,
    predictorCount: 1,
    response: props.response,
    bandwidthParameters: props.parameters,
    selectedBandwidth: props.output.selectedBandwidth,
    coefficients: props.output.coefficients,
    parameters: view('uint32', 4),
    maximumBandwidthCount: 2,
    maximumPermutations,
    indexGridSize: props.indexGridSize,
    tileRowCount: 256,
    output: {
      table: view('float32', 12),
      summary: view('float32', 4),
      status: {approximation: view('uint32', 1)}
    }
  };
  const transients: GraphBufferHandle[] = [];
  const createTransientBuffer = graph.createTransientBuffer.bind(graph);
  graph.createTransientBuffer = descriptor => {
    const handle = createTransientBuffer(descriptor);
    transients.push(handle);
    return handle;
  };
  return {device, graph, props, testProps, transients, view};
}

function inspect(rowCount: number, maximumPermutations: number, goldenSection = false) {
  const fixture = createFixture(rowCount, maximumPermutations);
  const regression = new GPUGeographicallyWeightedRegression({
    ...fixture.props,
    ...(goldenSection
      ? {bandwidthSearch: 'golden-section' as const, maximumSearchIterations: 4}
      : {})
  });
  const regressionNodes = regression.getCommandNodes(fixture.graph);
  const test = new GPUGeographicallyWeightedRegressionNonstationarityTest(fixture.testProps);
  const testNodes = test.getCommandNodes(fixture.graph);
  expect(regressionNodes.length).toBeLessThanOrEqual(regression.estimatedCommandNodeCount);
  expect(testNodes.length).toBeLessThanOrEqual(test.estimatedCommandNodeCount);
  const sources = getGPUCommandNodeKernelSources([...regressionNodes, ...testNodes]);
  fixture.device.destroy();
  return {...fixture, regression, test, sources, regressionNodes, testNodes};
}

it('accepts 2^24 indexed rows subject to actual binding ranges, using bounded focus scratch', () => {
  const fixture = createFixture(16777216, 1);
  const {transients} = fixture;
  const regression = new GPUGeographicallyWeightedRegression({
    ...fixture.props,
    maximumBandwidthCount: 32,
    parameters: fixture.view('float32', 36)
  });
  const nodes = regression.getCommandNodes(fixture.graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(nodes.length).toBeLessThanOrEqual(regression.estimatedCommandNodeCount);
  fixture.device.destroy();
  expect(regression.focusRowCount).toBe(262144);
  expect(regression.gaussianCutoff).toBe(8);
  for (const transient of transients)
    expect(transient.byteLength, transient.id).toBeLessThanOrEqual(134217728);
  expect(transients.find(view => view.id.endsWith('-candidate-scratch'))?.byteLength).toBe(
    262144 * 32 * 8
  );
  expect(sources.filter(source => /batch-\d+-fit$/.test(source.id))).toHaveLength(64);
});

it('keeps shader and pipeline count constant as focus batches and permutation batches grow', () => {
  for (const goldenSection of [false, true]) {
    const smaller = inspect(1048584, 64, goldenSection);
    const larger = inspect(2097160, 128, goldenSection);
    const pipelineCount = (sources: typeof smaller.sources, pattern: RegExp) =>
      new Set(
        sources
          .filter(({id}) => pattern.test(id))
          .map(({source, constants}) => source + JSON.stringify(constants))
      ).size;
    for (const pattern of [
      /batch-\d+-candidates$/,
      /batch-\d+-tile-candidates$/,
      /batch-\d+-fit$/,
      /batch-\d+-publish$/,
      /batch-\d+-publish-inference$/,
      /batch-\d+-permuted-tiles$/,
      /batch-\d+-merge$/
    ]) {
      expect(pipelineCount(smaller.sources, pattern), String(pattern)).toBe(1);
      expect(pipelineCount(larger.sources, pattern), String(pattern)).toBe(1);
    }
    const allPipelineKeys = (sources: typeof smaller.sources) =>
      new Set(sources.map(({source, constants}) => source + JSON.stringify(constants))).size;
    expect(allPipelineKeys(larger.sources)).toBe(allPipelineKeys(smaller.sources));
    const auditedSources = new Set<string>();
    for (const {id, source} of smaller.sources) {
      if (auditedSources.has(source)) continue;
      auditedSources.add(source);
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
      expect(findConstantNonFiniteBitcasts(stripWGSLComments(source)), id).toEqual([]);
      expect(getInvalidNumericLiterals(source), id).toEqual([]);
      expect(() => new WgslReflect(source), id).not.toThrow();
      expect(new WgslReflect(source).storage.length, id).toBeLessThanOrEqual(8);
    }
    const partialBytes = (result: typeof smaller) =>
      result.transients.find(view => view.id.endsWith('-partial'))!.byteLength;
    // Twice as many focus tiles, but permutation capacity no longer multiplies this again.
    expect(partialBytes(larger)).toBeLessThan(partialBytes(smaller) * 2.01);
  }
});

it('preflights every derived and caller binding before declaring resources or nodes', () => {
  const fixture = createFixture(1048584);
  Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {
    value: 12 * 1024 * 1024,
    configurable: true
  });
  const regression = new GPUGeographicallyWeightedRegression(fixture.props);
  // Coefficients fit, but the shared validity/local/inference state exceeds this test limit.
  expect(() => regression.getCommandNodes(fixture.graph)).toThrow(/row-state/);
  expect(fixture.transients).toHaveLength(0);
  Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {
    value: 9 * 1024 * 1024
  });
  const test = new GPUGeographicallyWeightedRegressionNonstationarityTest(fixture.testProps);
  expect(() => test.getCommandNodes(fixture.graph)).toThrow(/compacted training rows/);
  expect(fixture.transients).toHaveLength(0);
  fixture.device.destroy();
});

it('uses the aligned caller range, including a nonzero binding prefix', () => {
  const fixture = createFixture(16);
  const backing = fixture.view('float32', 80);
  const shifted = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 32,
    byteOffset: 4
  });
  const props = {...fixture.props, output: {...fixture.props.output, coefficients: shifted}};
  const regression = new GPUGeographicallyWeightedRegression(props);
  Object.defineProperty(fixture.device.limits, 'maxStorageBufferBindingSize', {value: 128});
  expect(() => regression.getCommandNodes(fixture.graph)).toThrow(/aligned binding.*132/);
  expect(fixture.transients).toHaveLength(1);
  fixture.device.destroy();
});

it('rejects overlapping writable fields and preserves an explicit exact Gaussian route with its old cap', () => {
  const fixture = createFixture(1048584);
  const shared = fixture.props.output.coefficients;
  expect(
    () =>
      new GPUGeographicallyWeightedRegression({
        ...fixture.props,
        output: {...fixture.props.output, fitted: shared as GraphDataView<'float32'>}
      })
  ).toThrow(/logical writable byte ranges overlap/);
  expect(
    () =>
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        ...fixture.testProps,
        output: {table: fixture.testProps.output.table, summary: fixture.testProps.output.table}
      })
  ).toThrow(/aligned binding byte ranges.*overlap/);
  expect(
    () => new GPUGeographicallyWeightedRegression({...fixture.props, gaussianMode: 'exact'})
  ).toThrow(/gaussianMode exact.*1048576/);
  expect(
    () =>
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        ...fixture.testProps,
        gaussianMode: 'exact'
      })
  ).toThrow(/gaussianMode exact.*1048576/);
  fixture.device.destroy();
});

it('retains the original permutation capacity within the named node budget', () => {
  const fixture = createFixture(1024, 1048576);
  const test = new GPUGeographicallyWeightedRegressionNonstationarityTest(fixture.testProps);
  expect(test.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
  expect(test.batchRunCount).toBeGreaterThan(32);
  fixture.device.destroy();
});

it('preserves the historical default tile size within the old row limit', () => {
  const fixture = createFixture(1048576, 1000);
  const test = new GPUGeographicallyWeightedRegressionNonstationarityTest({
    ...fixture.testProps,
    tileRowCount: undefined
  });
  const historicalTileLimit = Math.min(4096, Math.floor(16777216 / (1001 * 5)));
  expect(test.tileRowCount).toBe(Math.max(8, Math.ceil(1048576 / historicalTileLimit)));
  fixture.device.destroy();
});

it('fits the maximum golden-section depth at 2^24 rows within the command-node budget', () => {
  const fixture = createFixture(16777216, 1);
  const regression = new GPUGeographicallyWeightedRegression({
    ...fixture.props,
    bandwidthSearch: 'golden-section',
    maximumSearchIterations: 64
  });
  expect(regression.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
  fixture.device.destroy();
});

it('rejects logically adjacent outputs whose aligned writable bindings overlap', () => {
  const fixture = createFixture(16);
  const backing = fixture.view('float32', 128);
  const first = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 16,
    byteOffset: 0
  });
  const overlapping = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 16,
    byteOffset: 64
  });
  const aligned = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 16,
    byteOffset: 256
  });
  expect(
    () =>
      new GPUGeographicallyWeightedRegression({
        ...fixture.props,
        output: {...fixture.props.output, fitted: first, residuals: overlapping}
      })
  ).toThrow(/aligned binding byte ranges.*overlap/);
  expect(
    () =>
      new GPUGeographicallyWeightedRegression({
        ...fixture.props,
        output: {...fixture.props.output, fitted: first, residuals: aligned}
      })
  ).not.toThrow();
  expect(
    () =>
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        ...fixture.testProps,
        output: {table: first, summary: overlapping}
      })
  ).toThrow(/aligned binding byte ranges.*overlap/);
  fixture.device.destroy();
});

it('dispatches final focus and permutation batches at their own workgroup counts without specializing pipelines', () => {
  const inspected = inspect(1048584, 65, true);
  const fits = inspected.regressionNodes.filter(node => /batch-\d+-fit$/.test(node.id));
  expect(fits[0].workload?.maximumInvocationCount).toBe(262144);
  expect(fits.at(-1)?.workload?.maximumInvocationCount).toBe(8);
  expect(fits.at(-1)?.workload?.maximumWorkgroupCount).toBe(1);
  const merges = inspected.testNodes.filter(node => /batch-\d+-merge$/.test(node.id));
  expect(merges[0].workload?.maximumInvocationCount).toBe(32);
  expect(merges.at(-1)?.workload?.maximumInvocationCount).toBe(2);
  for (const pattern of [/batch-\d+-fit$/, /batch-\d+-merge$/, /batch-\d+-candidates$/]) {
    const requests = inspected.sources.filter(source => pattern.test(source.id));
    expect(
      new Set(requests.map(request => request.source + JSON.stringify(request.constants))).size
    ).toBe(1);
    expect(requests.every(request => Object.keys(request.constants).length === 0)).toBe(true);
  }
});

it('uses published hierarchical reductions instead of data-sized candidate, summary and grid extent loops', () => {
  const inspected = inspect(1048584, 1, true);
  expect(inspected.sources.some(source => /-sum-0-level-/.test(source.id))).toBe(true);
  expect(inspected.sources.some(source => /-variance-sum-2-level-/.test(source.id))).toBe(true);
  expect(inspected.sources.some(source => /extent-component-0-level-/.test(source.id))).toBe(true);
  expect(inspected.sources.find(source => /-search-init$/.test(source.id))?.source).toContain(
    'const TILE_COUNT: u32 = 1u;'
  );
  for (const {source} of inspected.sources.filter(source =>
    /-select$|-summarize$|-grid-bounds$/.test(source.id)
  ))
    expect(source).not.toMatch(/for \(var tile = 0u; tile < TILE_COUNT/);
});

it('preflights the shared grid helper before allocating any grid resources', () => {
  const fixture = createFixture(16);
  const before = fixture.transients.length;
  expect(() =>
    createGridIndexNodes(fixture.graph, {
      id: 'oversized-grid',
      operation: 'test',
      positions: fixture.props.positions,
      rowCount: 16,
      gridSize: [16777216, 4]
    })
  ).toThrow(/grid cell counts/);
  expect(fixture.transients).toHaveLength(before);
  fixture.device.destroy();
});

it('the independent heterogeneous oracle has a global optimum different from the first focus cohort', () => {
  const reference = createBoundaryReference(1048584);
  expect(reference.score(1.2, 0).aicc).toBeLessThan(reference.score(0.25, 0).aicc);
  expect(reference.score(0.25).aicc).toBeLessThan(reference.score(1.2).aicc);
  expect(reference.score(0.25).count).toBe(1048584);
  expect(reference.local(262143, 0.25).coefficients[0]).toBeCloseTo(
    reference.local(262144, 0.25).coefficients[0],
    10
  );
});

it('audits every generated identifier at the maximum design size against the full W3C list', () => {
  const fixture = createFixture(32, 1);
  const regression = new GPUGeographicallyWeightedRegression({
    ...fixture.props,
    predictors: fixture.view('float32', 32 * 11),
    predictorCount: 11,
    output: {
      ...fixture.props.output,
      coefficients: fixture.view('float32', 32 * 12),
      standardErrors: fixture.view('float32', 32 * 12),
      tValues: fixture.view('float32', 32 * 12),
      significant: fixture.view('uint32', 32 * 12),
      localConditionNumber: fixture.view('float32', 32)
    }
  });
  for (const {id, source} of getGPUCommandNodeKernelSources(
    regression.getCommandNodes(fixture.graph)
  ))
    expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
  for (const word of ['target', 'active', 'filter', 'module', 'meta'])
    expect(WGSL_RESERVED_WORDS.has(word)).toBe(true);
  expect(findReservedIdentifiers('let sample = 0u;')).toEqual([]);
  fixture.device.destroy();
});

it('retains construction-time refusal above the named indexed maximum', () => {
  const fixture = createFixture(16777217, 1);
  expect(() => new GPUGeographicallyWeightedRegression(fixture.props)).toThrow(
    /positions.length.*16777216/
  );
  expect(
    () => new GPUGeographicallyWeightedRegressionNonstationarityTest(fixture.testProps)
  ).toThrow(/positions.length.*16777216/);
  expect(fixture.transients).toHaveLength(0);
  fixture.device.destroy();
});

it('preflights derived scratch against maxBufferSize before graph mutation', () => {
  const fixture = createFixture(1048584, 1);
  Object.defineProperty(fixture.device.limits, 'maxBufferSize', {value: 12 * 1024 * 1024});
  expect(() =>
    new GPUGeographicallyWeightedRegression(fixture.props).getCommandNodes(fixture.graph)
  ).toThrow(/row-state scratch.*maxBufferSize/);
  expect(() =>
    new GPUGeographicallyWeightedRegressionNonstationarityTest(fixture.testProps).getCommandNodes(
      fixture.graph
    )
  ).toThrow(/compacted training rows scratch.*maxBufferSize/);
  expect(fixture.transients).toHaveLength(0);
  fixture.device.destroy();
});

it('preflights the complete backing allocation even when a small output binding fits', () => {
  const fixture = createFixture(16, 1);
  const backing = fixture.view('float32', 1024);
  const coefficients = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 32
  });
  Object.defineProperty(fixture.device.limits, 'maxBufferSize', {value: 1024});
  const regression = new GPUGeographicallyWeightedRegression({
    ...fixture.props,
    output: {...fixture.props.output, coefficients}
  });
  expect(() => regression.getCommandNodes(fixture.graph)).toThrow(
    /backing buffer.*4096.*maxBufferSize 1024/
  );
  expect(fixture.transients).toHaveLength(1);
  fixture.device.destroy();
});

it('permits adjacent outputs across separate dispatches while rejecting logical overlap', () => {
  const fixture = createFixture(16, 1);
  const backing = fixture.view('float32', 64);
  const coefficients = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 32
  });
  const standardErrors = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 32,
    byteOffset: 128
  });
  // Their aligned bindings overlap at byte zero, but fit and publish-inference bind them separately.
  const regression = new GPUGeographicallyWeightedRegression({
    ...fixture.props,
    output: {...fixture.props.output, coefficients, standardErrors}
  });
  expect(() => fixture.graph.add(regression)).not.toThrow();
  const overlapping = fixture.graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 32,
    byteOffset: 124
  });
  expect(
    () =>
      new GPUGeographicallyWeightedRegression({
        ...fixture.props,
        output: {...fixture.props.output, coefficients, standardErrors: overlapping}
      })
  ).toThrow(/logical writable byte ranges overlap/);
  fixture.device.destroy();
});

it('publishes the named approximation status in both GWR contributors', () => {
  const fixture = createFixture(32, 1);
  const regression = new GPUGeographicallyWeightedRegression(fixture.props);
  const test = new GPUGeographicallyWeightedRegressionNonstationarityTest(fixture.testProps);
  for (const contributor of [regression, test]) {
    const nodes = contributor.getCommandNodes(fixture.graph);
    const status = nodes.find(node => node.id.endsWith('-approximation'));
    expect(status).toBeDefined();
    const view = contributor.props.output.status!.approximation;
    expect(
      (status as {resources: {buffer: unknown}[]}).resources.some(
        resource => resource.buffer === view
      )
    ).toBe(true);
  }
  fixture.device.destroy();
});

it('names every focus, reduction and permutation resource with the contributor id', () => {
  const fixture = createFixture(1048584, 1);
  for (const id of ['first', 'second']) {
    fixture.graph.add(new GPUGeographicallyWeightedRegression({...fixture.props, id: `${id}-fit`}));
    fixture.graph.add(
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        ...fixture.testProps,
        id: `${id}-test`
      })
    );
  }
  expect(fixture.transients.length).toBeGreaterThan(0);
  for (const buffer of fixture.transients) expect(buffer.id).toMatch(/^(first|second)-(fit|test)-/);
  fixture.device.destroy();
});

it('rejects unrepresentable shader props at construction and formats large finite cutoff literals', () => {
  const fixture = createFixture(32, 1);
  for (const gaussianCutoff of [
    Number.NaN,
    Infinity,
    -Infinity,
    Number.MAX_VALUE,
    1e40,
    Number.MIN_VALUE,
    0,
    -1
  ]) {
    expect(
      () => new GPUGeographicallyWeightedRegression({...fixture.props, gaussianCutoff})
    ).toThrow(/geographically-weighted-regression gaussianCutoff.*finite/);
    expect(
      () =>
        new GPUGeographicallyWeightedRegressionNonstationarityTest({
          ...fixture.testProps,
          gaussianCutoff
        })
    ).toThrow(/gwr-nonstationarity-test gaussianCutoff.*finite/);
  }
  expect(
    () =>
      new GPUGeographicallyWeightedRegression({
        ...fixture.props,
        indexGridSize: undefined,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          sphereRadius: Number.MAX_VALUE
        }
      })
  ).toThrow(/spatialContext.sphereRadius.*finite/);
  expect(fixture.transients).toHaveLength(0);
  for (const contributor of [
    new GPUGeographicallyWeightedRegression({...fixture.props, gaussianCutoff: 1e20}),
    new GPUGeographicallyWeightedRegressionNonstationarityTest({
      ...fixture.testProps,
      gaussianCutoff: 1e20
    })
  ]) {
    for (const {id, source} of getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(fixture.graph)
    )) {
      expect(getInvalidNumericLiterals(source), id).toEqual([]);
      expect(findReservedIdentifiers(stripWGSLComments(source)), id).toEqual([]);
      expect(() => new WgslReflect(source), id).not.toThrow();
    }
  }
  for (const [gridSize, rowCount] of [
    [[Number.MAX_VALUE, 1], 32],
    [[NaN, 1], 32],
    [[1, Infinity], 32],
    [[1, 1], Number.MAX_VALUE]
  ] as const) {
    const before = fixture.transients.length;
    expect(() =>
      createGridIndexNodes(fixture.graph, {
        id: 'invalid-grid',
        operation: 'test',
        positions: fixture.props.positions,
        rowCount,
        gridSize
      })
    ).toThrow(/invalid-grid/);
    expect(fixture.transients).toHaveLength(before);
  }
  expect(
    getInvalidNumericLiterals('let bad = 18446744073709551616; let overflow = 1e999;')
  ).toEqual(['18446744073709551616', '1e999']);
  fixture.device.destroy();
});
