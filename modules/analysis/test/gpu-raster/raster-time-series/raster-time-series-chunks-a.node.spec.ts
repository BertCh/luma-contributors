// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {Buffer} from '@luma.gl/core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {GPURasterBestAvailablePixel} from '../../../src/raster-analysis/raster-time-series/gpu-raster-best-available-pixel';
import {GPURasterBFASTMonitor} from '../../../src/raster-analysis/raster-time-series/gpu-raster-bfast-monitor';
import {GPURasterClimateAnomaly} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-anomaly';
import {GPURasterClimateRunAccumulator} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-run-accumulator';
import {GPURasterCOLDMonitor} from '../../../src/raster-analysis/raster-time-series/gpu-raster-cold-monitor';
import {GPURasterEOFProjection} from '../../../src/raster-analysis/raster-time-series/gpu-raster-eof-projection';
import {GPURasterGeometricMedianComposite} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';
import {GPURasterHANTS} from '../../../src/raster-analysis/raster-time-series/gpu-raster-hants';
import {GPURasterHarmonicRegression} from '../../../src/raster-analysis/raster-time-series/gpu-raster-harmonic-regression';
import {
  planRasterTimeSeriesChunks,
  validateRasterTimeSeriesViews
} from '../../../src/raster-analysis/raster-time-series/raster-time-series-chunks';
import {
  auditWGSLModule,
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());
const families = [
  'best',
  'bfast',
  'anomaly',
  'runs',
  'cold',
  'eof',
  'geometric',
  'hants',
  'harmonic'
] as const;
type Family = (typeof families)[number];

function createCase(
  graph: GPUCommandGraph,
  family: Family,
  cells: number[],
  prefix = true,
  adjustView?: (name: string, view: GraphDataView, chunkIndex?: number) => GraphDataView
) {
  const visit = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    name: string,
    view: GraphDataView<Format>,
    chunkIndex?: number
  ): GraphDataView<Format> =>
    adjustView ? (adjustView(name, view, chunkIndex) as GraphDataView<Format>) : view;
  const packed = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    name: string,
    format: Format,
    length: number
  ) => visit(name, createTransientView(graph, name, format, length));
  let fieldIndex = 0;
  const rows = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    width: number
  ) => {
    const field = fieldIndex++;
    return createVectorView(
      name,
      format,
      cells.map((cellCount, chunkIndex) => {
        const padding = prefix ? (field + chunkIndex * 3) % 63 : 0;
        const backing = createTransientView(
          graph,
          `${name}-${chunkIndex}`,
          format,
          Math.max(1, cellCount * width) + padding
        );
        return visit(
          name,
          graph.createDataView(backing.buffer, {
            format,
            length: cellCount * width,
            byteOffset: padding * 4
          }),
          chunkIndex
        );
      })
    );
  };
  const parameters = packed('parameters', 'float32', 16);
  const dateAxis = {
    acquisitionTimes: packed('dates', 'uint32x2', 16),
    temporalContext: {
      encoding: 'int64-words',
      unit: 'second',
      epoch: 'unix',
      axis: {kind: 'irregular'}
    } as const
  };
  const base = {
    id: family,
    cellCount: cells.reduce((sum, count) => sum + count, 0),
    sliceCount: 16
  };
  if (family === 'anomaly')
    return new GPURasterClimateAnomaly({
      ...base,
      slotCount: 2,
      chunkSliceCount: 16,
      accumulatorParameters: parameters,
      state: {
        counts: rows('counts', 'uint32', 2),
        sums: rows('sums', 'uint32', 6),
        moments: rows('moments', 'uint32', 6)
      },
      values: rows('values', 'float32', 16),
      slotIndices: packed('slots', 'uint32', 16),
      standardized: true,
      output: {
        climatologyMean: rows('mean', 'float32', 2),
        standardDeviation: rows('deviation', 'float32', 2),
        anomaly: rows('anomaly', 'float32', 16)
      }
    });
  const values = rows('values', 'float32', 16);
  const mask = rows('mask', 'uint32', 16);
  const common = {...base, values, mask, parameters};
  switch (family) {
    case 'best':
      return new GPURasterBestAvailablePixel({
        ...common,
        bandCount: 1,
        cloudDistance: rows('cloud', 'float32', 16),
        sliceAttributes: packed('attributes', 'float32', 32),
        output: {
          bestSlice: rows('bestSlice', 'uint32', 1),
          bestScore: rows('score', 'float32', 1),
          composite: rows('composite', 'float32', 1)
        }
      });
    case 'bfast':
      return new GPURasterBFASTMonitor({
        ...common,
        dateAxis,
        historySliceCount: 12,
        harmonicCount: 1,
        output: {
          coefficients: rows('coefficients', 'float32', 4),
          scores: rows('scores', 'float32', 2),
          cellSummary: rows('summary', 'uint32', 3),
          process: rows('process', 'float32', 4)
        }
      });
    case 'runs':
      return new GPURasterClimateRunAccumulator({
        ...common,
        chunkSliceCount: 16,
        state: {runs: rows('runs', 'uint32', 6), degreeDays: rows('degreeDays', 'uint32', 3)}
      });
    case 'cold':
      return new GPURasterCOLDMonitor({
        ...common,
        dateAxis,
        bandCount: 1,
        minimumObservations: 5,
        maximumCoefficientCount: 4,
        maximumFitCount: 1,
        maximumSegmentCount: 2,
        output: {
          segments: rows('segments', 'uint32', 8),
          coefficients: rows('coefficients', 'float32', 16),
          segmentStatistics: rows('statistics', 'float32', 4),
          cellSummary: rows('summary', 'uint32', 3)
        }
      });
    case 'eof':
      return new GPURasterEOFProjection({
        ...common,
        modeCount: 2,
        weights: rows('weights', 'float32', 1),
        temporalModes: packed('modes', 'float32', 32),
        singularValues: packed('singular', 'float32', 2),
        output: {eofs: rows('eofs', 'float32', 2), status: rows('status', 'uint32', 1)}
      });
    case 'geometric':
      return new GPURasterGeometricMedianComposite({
        ...common,
        bandCount: 1,
        iterationCount: 2,
        output: {
          composite: rows('composite', 'float32', 1),
          status: rows('status', 'uint32', 1),
          lastStep: rows('lastStep', 'float32', 1)
        }
      });
    case 'hants':
      return new GPURasterHANTS({
        ...common,
        dateAxis,
        harmonicCount: 1,
        output: {
          smoothed: rows('smoothed', 'float32', 16),
          coefficients: rows('coefficients', 'float32', 3),
          status: rows('status', 'uint32', 1),
          diagnostics: rows('diagnostics', 'uint32', 2)
        }
      });
    case 'harmonic':
      return new GPURasterHarmonicRegression({
        ...common,
        dateAxis,
        harmonicCount: 1,
        output: {
          coefficients: rows('coefficients', 'float32', 4),
          summary: rows('summary', 'uint32', 2),
          fitted: rows('fitted', 'float32', 16),
          rmse: rows('rmse', 'float32', 1)
        }
      });
  }
}

for (const family of families) {
  it(`${family} shares pipelines across chunk sizes, counts and unaligned offsets`, () => {
    const pipelines = new Set<string>();
    for (const cells of [[1], [257, 65, 1], [0, 257, 65, 1, 2, 0]]) {
      const graph = new GPUCommandGraph(device);
      const contributor = createCase(graph, family, cells);
      const nodes = contributor.getCommandNodes(graph);
      const stageCount = family === 'anomaly' || family === 'harmonic' ? 2 : 1;
      expect(nodes).toHaveLength(cells.filter(Boolean).length * stageCount);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      expect(getParsedKernelSources(nodes).size).toBe(nodes.length);
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        pipelines.add(key);
      for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
        expect(constants).toEqual({});
        expect(auditWGSLModule(source, constants)).toEqual([]);
        const identifiers = stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      }
      if (family !== 'anomaly')
        expect(nodes.map(node => node.workload?.maximumInvocationCount)).toEqual(
          cells.filter(Boolean).flatMap(count => (family === 'harmonic' ? [count, count] : [count]))
        );
    }
    expect(pipelines.size).toBe(family === 'anomaly' || family === 'harmonic' ? 2 : 1);
  });
}

it('rejects split records, mismatched state/output topology and chunked shared tables', () => {
  const graph = new GPUCommandGraph(device);
  const values = createVectorView('values', 'float32', [
    createTransientView(graph, 'first', 'float32', 5),
    createTransientView(graph, 'last', 'float32', 7)
  ]);
  expect(() => planRasterTimeSeriesChunks('records', {cellCount: 3, values}, {values: 4})).toThrow(
    /^records values chunk 0 split cell record.*4/
  );
  const contributor = createCase(graph, 'runs', [2, 2, 1]);
  if (!(contributor instanceof GPURasterClimateRunAccumulator))
    throw new Error('expected run accumulator');
  expect(
    () =>
      new GPURasterClimateRunAccumulator({
        ...contributor.props,
        state: {
          ...contributor.props.state,
          runs: createVectorView('wrong-runs', 'uint32', [
            createTransientView(graph, 'wrong', 'uint32', 30)
          ])
        }
      })
  ).toThrow(/runs must match the cell chunk topology/);
  expect(() =>
    planRasterTimeSeriesChunks('tables', {cellCount: 3, parameters: values}, {})
  ).toThrow(/^tables parameters must be one packed view/);
});

it('rejects writable aliases across fields and chunks but accepts consecutive chunks', () => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'backing', 'uint32', 256);
  const view = (offset: number, length: number): GraphDataView<'uint32'> =>
    graph.createDataView(backing.buffer, {format: 'uint32', length, byteOffset: offset});
  expect(() =>
    validateRasterTimeSeriesViews(
      'aliases',
      [],
      [
        createVectorView('first', 'uint32', [view(4, 1)]),
        createVectorView('second', 'uint32', [view(8, 1)])
      ]
    )
  ).toThrow(/aligned binding byte ranges must not overlap/);
  expect(() =>
    validateRasterTimeSeriesViews(
      'aliases',
      [],
      [createVectorView('repeated', 'uint32', [view(0, 2), view(4, 2)])]
    )
  ).toThrow(/logical byte ranges must not overlap/);
  expect(() =>
    validateRasterTimeSeriesViews(
      'adjacent',
      [],
      [createVectorView('consecutive', 'uint32', [view(0, 2), view(8, 2), view(16, 1)])]
    )
  ).not.toThrow();
});

it('preflights later aligned bindings before creating any nodes or resources', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = createCase(graph, 'runs', [1, 2_097_152], true);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const declare = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /runs view 0 chunk 1 needs 134217740 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(declare).not.toHaveBeenCalled();
});

it('plans 30 million cells without allocating or packing their data', () => {
  const graph = new GPUCommandGraph(device);
  // Six run words/cell fit 128 MiB for each five-million-cell chunk.
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const values = createVectorView(
    'large-values',
    'float32',
    Array.from({length: 6}, (_, index) =>
      createTransientView(graph, `values-${index}`, 'float32', 5_000_000)
    )
  );
  const runs = createVectorView(
    'large-runs',
    'uint32',
    Array.from({length: 6}, (_, index) =>
      createTransientView(graph, `runs-${index}`, 'uint32', 30_000_000)
    )
  );
  const contributor = new GPURasterClimateRunAccumulator({
    cellCount: 30_000_000,
    chunkSliceCount: 1,
    values,
    parameters,
    state: {runs}
  });
  expect(contributor.getCommandNodes(graph)).toHaveLength(6);
});

it('keeps the original harmonic QR reference limits and removes slice arrays from storage QR', () => {
  const graph = new GPUCommandGraph(device);
  const small = createCase(graph, 'harmonic', [2, 2, 1]);
  if (!(small instanceof GPURasterHarmonicRegression))
    throw new Error('expected harmonic regression');
  const sliceCount = 512;
  const field = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    width: number
  ) =>
    createVectorView(
      name,
      format,
      [2, 2, 1].map((count, index) =>
        createTransientView(graph, `${name}-${index}`, format, count * width)
      )
    );
  const props = {
    ...small.props,
    sliceCount,
    values: field('long-values', 'float32', sliceCount),
    mask: field('long-mask', 'uint32', sliceCount),
    dateAxis: {
      ...small.props.dateAxis,
      acquisitionTimes: createTransientView(graph, 'long-dates', 'uint32x2', sliceCount)
    },
    output: {...small.props.output, fitted: field('long-fitted', 'float32', sliceCount)}
  };
  const storage = new GPURasterHarmonicRegression(props);
  expect(storage.mode).toBe('storage');
  const nodes = storage.getCommandNodes(graph);
  expect(nodes).toHaveLength(6);
  for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(source).not.toContain('sliceYears');
    expect(source).not.toContain('var basis');
    expect(source).not.toContain('array<f32, 512>');
    expect(source).not.toContain('array<f32, 2048>');
  }
  expect(getParsedKernelSources(nodes).size).toBe(6);
  expect(() => new GPURasterHarmonicRegression({...props, mode: 'reference'})).toThrow(
    /sliceCount.*256/
  );
  const reference = new GPURasterHarmonicRegression({...small.props, mode: 'reference'});
  expect(reference.getCommandNodes(graph)).toHaveLength(3);
});

it('batches harmonic scratch without changing the pipeline and preflights every batch', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = createCase(graph, 'harmonic', [1, 524_289]);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes).toHaveLength(6);
  expect(nodes.map(node => node.workload?.maximumInvocationCount)).toEqual([
    1, 1, 524288, 524288, 1, 1
  ]);
  expect(getGPUCommandNodePipelineKeys(nodes).size).toBe(2);
  const constrained = createNullWebGPUDevice();
  Object.defineProperty(constrained, 'limits', {
    value: {...constrained.limits, maxStorageBufferBindingSize: 64 * 1024 * 1024}
  });
  try {
    const limitedGraph = new GPUCommandGraph(constrained);
    const limited = createCase(limitedGraph, 'harmonic', [1, 524_289]);
    expectUnchangedOnFailure(
      limitedGraph,
      limited,
      /^harmonic QR basis scratch needs 134217728 bytes/
    );
  } finally {
    constrained.destroy();
  }
});

it('bounds chunk command counts at construction', () => {
  const graph = new GPUCommandGraph(device);
  const chunk = createTransientView(graph, 'input', 'float32', 1);
  const repeated = createVectorView('many', 'float32', Array(32769).fill(chunk));
  const parameters = createTransientView(graph, 'parameters', 'float32', 4);
  const runChunk = createTransientView(graph, 'run-chunk', 'uint32', 6);
  const runs = createVectorView('runs', 'uint32', Array(32769).fill(runChunk));
  expect(
    () =>
      new GPURasterClimateRunAccumulator({
        cellCount: 32769,
        chunkSliceCount: 1,
        values: repeated,
        parameters,
        state: {runs}
      })
  ).toThrow(/estimated command node count.*32768/);
});

function createHarmonicBoundaryProps(
  graph: GPUCommandGraph,
  cellCount: number,
  sliceCount: number,
  harmonicCount: number
) {
  let serial = 0;
  const field = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `boundary-${serial++}`, format, length);
  return {
    cellCount,
    sliceCount,
    harmonicCount,
    values: field('float32', cellCount * sliceCount),
    mask: field('uint32', cellCount * sliceCount),
    parameters: field('float32', 1),
    dateAxis: {
      acquisitionTimes: field('uint32x2', sliceCount),
      temporalContext: {
        encoding: 'int64-words',
        unit: 'second',
        epoch: 0n,
        axis: {kind: 'irregular'}
      } as const
    },
    output: {
      coefficients: field('float32', cellCount * (2 + 2 * harmonicCount)),
      summary: field('uint32', cellCount * 2),
      fitted: field('float32', cellCount * sliceCount),
      rmse: field('float32', cellCount)
    }
  };
}

it('retains the old memory rejection in reference mode while the corresponding storage audit case succeeds', () => {
  const graph = new GPUCommandGraph(device);
  const props = createHarmonicBoundaryProps(graph, 4096, 256, 2);
  expect(() => new GPURasterHarmonicRegression({...props, mode: 'reference'})).toThrow(
    /^raster-harmonic-regression per-thread QR working memory .*must fit 2048 floats$/
  );
  const contributor = new GPURasterHarmonicRegression(props);
  expect(contributor.mode).toBe('storage');
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes).toHaveLength(2);
  for (const {source, constants} of getGPUCommandNodeKernelSources(nodes))
    expect(auditWGSLModule(source, constants)).toEqual([]);
});

it('audits storage QR at the slice and private-memory boundaries and refuses invalid literal-generating props', () => {
  for (const [sliceCount, harmonicCount] of [
    [4096, 1],
    [4096, 21]
  ]) {
    const graph = new GPUCommandGraph(device);
    const props = createHarmonicBoundaryProps(graph, 1, sliceCount, harmonicCount);
    const nodes = new GPURasterHarmonicRegression(props).getCommandNodes(graph);
    expect(nodes).toHaveLength(2);
    expect(getParsedKernelSources(nodes).size).toBe(2);
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes))
      expect(auditWGSLModule(source, constants)).toEqual([]);
    expect(() => new GPURasterHarmonicRegression({...props, sliceCount: 4097})).toThrow(
      /sliceCount.*4096/
    );
    expect(() => new GPURasterHarmonicRegression({...props, harmonicCount: 22})).toThrow(
      /working memory.*2048/
    );
    for (const invalid of [
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      0x100000000
    ]) {
      expect(() => new GPURasterHarmonicRegression({...props, sliceCount: invalid})).toThrow(
        /sliceCount/
      );
      expect(() => new GPURasterHarmonicRegression({...props, harmonicCount: invalid})).toThrow(
        /harmonicCount|working memory/
      );
    }
  }
});

for (const family of families) {
  it(`${family} encodes each chunk with its own workgroup count`, () => {
    const graph = new GPUCommandGraph(device);
    const nodes = createCase(graph, family, [3, 65, 1]).getCommandNodes(graph);
    const dispatch = vi.spyOn(Kernel.prototype, 'dispatch').mockImplementation(() => {});
    const recordingDevice = createNullWebGPUDevice();
    Object.defineProperty(recordingDevice, 'props', {
      value: {...recordingDevice.props, _cacheShaders: false, _cachePipelines: false}
    });
    vi.spyOn(recordingDevice, 'createComputePipeline').mockImplementation(
      props => ({id: props.id ?? '', hash: '', destroy() {}}) as never
    );
    const backing = recordingDevice.createBuffer({byteLength: 65536});
    try {
      for (const node of nodes) {
        const compiled = node.compile?.({device: recordingDevice} as never);
        if (!compiled) throw new Error('expected compiled raster node');
        try {
          compiled.encode({getBuffer: () => backing, computePass: {}} as never);
        } finally {
          compiled.destroy?.();
        }
      }
      const workgroupSize =
        family === 'best' || family === 'runs' || family === 'geometric' || family === 'anomaly'
          ? 64
          : family === 'eof'
            ? 16
            : 8;
      const expected = [3, 65, 1].flatMap(cells =>
        family === 'anomaly'
          ? [Math.ceil((cells * 2) / 64), Math.ceil((cells * 16) / 64)]
          : Array(family === 'harmonic' ? 2 : 1).fill(Math.ceil(cells / workgroupSize))
      );
      expect(dispatch.mock.calls.map(([, options]) => options.x)).toEqual(expected);
      expect(dispatch.mock.calls.every(([, options]) => options.y === 1 && options.z === 1)).toBe(
        true
      );
    } finally {
      dispatch.mockRestore();
      backing.destroy();
      recordingDevice.destroy();
    }
  });
}

/** Rejecting a contributor must not declare buffers, views or scheduled nodes on the graph. */
function expectUnchangedOnFailure(
  graph: GPUCommandGraph,
  contributor: {getCommandNodes(graph: GPUCommandGraph): unknown},
  message: RegExp
) {
  const mutations = [
    vi.spyOn(graph, 'createTransientBuffer'),
    vi.spyOn(graph, 'createDataView'),
    vi.spyOn(graph, 'importBuffer'),
    vi.spyOn(graph, 'add'),
    vi.spyOn(graph, 'addComputePass'),
    vi.spyOn(graph, 'addCopyPass')
  ];
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(message);
    mutations.forEach(mutation => expect(mutation).not.toHaveBeenCalled());
  } finally {
    mutations.forEach(mutation => mutation.mockRestore());
  }
}

for (const family of families) {
  for (const failure of ['aligned-range', 'storage-usage'] as const) {
    it(`${family} preflights every caller field before graph mutation (${failure})`, () => {
      const probeGraph = new GPUCommandGraph(device);
      const fields = new Set<string>();
      createCase(probeGraph, family, [1, 1, 1], false, (name, view) => {
        // The common fixture creates dates even for contributors that never bind them.
        if (name !== 'dates' || ['bfast', 'cold', 'hants', 'harmonic'].includes(family))
          fields.add(name);
        return view;
      });
      for (const field of fields) {
        const constrained = createNullWebGPUDevice();
        Object.defineProperty(constrained, 'limits', {
          value: {...constrained.limits, maxStorageBufferBindingSize: 128}
        });
        try {
          const graph = new GPUCommandGraph(constrained);
          const contributor = createCase(
            graph,
            family,
            [1, 1, 1],
            false,
            (name, view, chunkIndex) => {
              if (name !== field || (chunkIndex !== undefined && chunkIndex !== 2)) return view;
              const byteOffset = failure === 'aligned-range' ? 128 : 0;
              const buffer = graph.createTransientBuffer({
                id: `invalid-${field}`,
                byteLength: byteOffset + view.length * view.rowByteLength,
                usage: failure === 'storage-usage' ? Buffer.COPY_DST : Buffer.STORAGE
              });
              return graph.createDataView(buffer, {
                format: view.format,
                length: view.length,
                byteOffset
              });
            }
          );
          expectUnchangedOnFailure(
            graph,
            contributor,
            failure === 'aligned-range'
              ? /view \d+ chunk \d+ needs \d+ bytes in one binding \(limit 128\)/
              : /usage must include Buffer.STORAGE/
          );
        } finally {
          constrained.destroy();
        }
      }
    });
  }
}

it('permits pooled harmonic reference outputs but rejects storage publication aliases', () => {
  for (const field of ['coefficients', 'summary'] as const) {
    const graph = new GPUCommandGraph(device);
    const props = createHarmonicBoundaryProps(graph, 1, 16, 1);
    // Import once, then use disjoint aligned ranges so the old writable-range check accepts them.
    const buffer = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
    try {
      const handle = graph.importBuffer(
        {id: 'publication', byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      const fitted = graph.createDataView(handle, {format: 'float32', length: 16, byteOffset: 256});
      const shared =
        field === 'coefficients'
          ? graph.createDataView(handle, {format: 'float32', length: 4})
          : graph.createDataView(handle, {format: 'uint32', length: 2});
      const output = {...props.output, fitted, [field]: shared};
      expect(() => new GPURasterHarmonicRegression({...props, output})).toThrow(
        /^raster-harmonic-regression .*must not share buffers/
      );
      expect(
        () => new GPURasterHarmonicRegression({...props, mode: 'reference', output})
      ).not.toThrow();
    } finally {
      buffer.destroy();
    }
  }
});

it('requires one imported handle for each physical raster buffer', () => {
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  try {
    // Deliberately invalid imports: writable hazards would otherwise be tracked as different buffers.
    const first = graph.importBuffer(
      {id: 'first-import', byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    const second = graph.importBuffer(
      {id: 'second-import', byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    const input = graph.createDataView(first, {format: 'float32', length: 16});
    const output = graph.createDataView(second, {format: 'float32', length: 1, byteOffset: 256});
    expect(() => validateRasterTimeSeriesViews('physical-raster', [input], [output])).toThrow(
      /one imported handle/
    );
    const oneHandleOutput = graph.createDataView(first, {
      format: 'float32',
      length: 1,
      byteOffset: 256
    });
    expect(() =>
      validateRasterTimeSeriesViews('physical-raster', [input], [oneHandleOutput])
    ).toThrow(/share buffers/);
  } finally {
    buffer.destroy();
  }
});

it('declares every QR scratch batch with STORAGE usage and its preflighted aligned range', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = createCase(graph, 'harmonic', [1, 524_289]);
  const nodes = contributor.getCommandNodes(graph);
  const basisViews = nodes
    .flatMap(node => node.resources ?? [])
    .flatMap(resource => {
      if (
        !('buffer' in resource) ||
        !('length' in resource.buffer) ||
        !resource.buffer.buffer.id.endsWith('-qr-basis')
      )
        return [];
      return [resource.buffer];
    });
  expect(basisViews.map(view => getViewBindingRange(view).size)).toEqual([256, 134217728, 256]);
  for (const view of basisViews) {
    expect(view.buffer.usage & Buffer.STORAGE).toBe(Buffer.STORAGE);
    expect(getViewBindingRange(view).offset).toBe(0);
    expect(getViewBindingRange(view).size).toBe(view.buffer.byteLength);
  }
});

it('checks named required and optional chunk ports before topology arithmetic', () => {
  const ports = {
    values: {required: true, viewType: 'either', formats: ['float32']},
    mask: {required: false, viewType: 'either', formats: ['uint32']}
  } as const;
  expect(() =>
    planRasterTimeSeriesChunks('required-first', {cellCount: 1}, {values: NaN}, ports)
  ).toThrow(/required-first.*values.*received undefined/);
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values-required', 'float32', 1);
  expect(
    planRasterTimeSeriesChunks('optional-mask', {cellCount: 1, values}, {values: 1, mask: 1}, ports)
  ).toHaveLength(1);
});
